import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum
import torch.nn.functional as F
# from .noneed.hadamard_utils import get_had_fn, get_qhad_fn
from analysis.plt import plot_tensor_histogram, plot_tensor_3d, plot_tensor_HW

# from quantize.quantizer import UniformAffineQuantizer
import time



class DULinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        rotate=True,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,rotate=rotate)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)


    def forward(self, input: torch.Tensor):
        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if not self.init_duquant_params:
                self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                self.init_duquant_params = torch.tensor(1) 
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_duquant_params(self, proj):
        assert proj.init_duquant_params
        self.init_duquant_params = torch.tensor(1)
        self.weight_quantizer.copy_duquant_params(proj.weight_quantizer)
        self.act_quantizer.copy_duquant_params(proj.act_quantizer)

class NoActQ(nn.Module):
    def __init__(self, nbits_a=4, **kwargs):
        super(NoActQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        return x


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    # forward:alpha
    # backward:alpha*g
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    # forward: round(x)
    # backward: x=(torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
    input = x
    x_round = input.round()
    x = input - input.floor().detach()
    x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
    out3 = x + input.floor().detach()
    return x_round.detach() - out3.detach() + out3


def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, nbits=4):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        pass
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _Conv2dQv2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQv2, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        
        self.alpha = Parameter(torch.ones(out_channels, 1, 1, 1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(_Conv2dQv2, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _LinearQv2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQv2, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.ones(out_features, 1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQv2, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

class ActQv2_linear(nn.Module):
    def __init__(self, **kwargs_q):
        super(ActQv2_linear, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        self.channel_size = kwargs_q['channel_size']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.ones((1, self.channel_size)))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)

class ActQv2_conv(nn.Module):
    def __init__(self, **kwargs_q):
        super(ActQv2_conv, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        self.channel_size = kwargs_q['channel_size']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.ones((1, self.channel_size, 1, 1)))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)


def quant_x_proj_mat_mul(x, weight, act):

    x_dbl = torch.matmul(
        act(x.permute(0, 1, 3, 2)),     # [B, K, D, L] -> [B, K, L, D]
        act(weight.permute(0, 2, 1))    # [K, C, D] -> [K, D, C]
    )                                   # [B, K, L, C]

    x_dbl = x_dbl.permute(0, 1, 3, 2)   # [B, K, L, C] -> [B, K, C, L]

    return x_dbl

def quant_dts_proj_mat_mul(x, weight, act):

    dts = torch.matmul(
    act(x.permute(0, 1, 3, 2)),         # [B, K, R, L] -> [B, K, L, R]
    act(weight.permute(0, 2, 1))             # [K, D, R] -> [K, R, D]
    )                                   # [B, K, L, D]

    dts = dts.permute(0, 1, 3, 2)       # [B, K, L, D] -> [B, K, D, L]

    return dts

class QuantSR_act(_ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(QuantSR_act, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return x
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
           
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        # global is_plot
        # if is_plot:
        #     plot_tensor_histogram(y, name="y_before_out_proj_" + str(plot_index))
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        return x

class QuantSR_actv2_linear(ActQv2_linear):
    def __init__(self, nbits_a=4, channel_size=60, **kwargs):
        super(QuantSR_actv2_linear, self).__init__(nbits=nbits_a, channel_size=channel_size)

    def forward(self, x):
        if self.alpha is None:
            return x
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            # import pdb;pdb.set_trace()
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_( x.reshape(-1, self.channel_size).max(dim=0)[0] / math.sqrt(Qp) )
           
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        # global is_plot
        # if is_plot:
        #     plot_tensor_histogram(y, name="y_before_out_proj_" + str(plot_index))
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        return x

class QuantSR_actv2_conv(ActQv2_conv):
    def __init__(self, nbits_a=4, channel_size=60, **kwargs):
        super(QuantSR_actv2_conv, self).__init__(nbits=nbits_a, channel_size=channel_size)

    def forward(self, x):
        if self.alpha is None:
            return x
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            # import pdb;pdb.set_trace()
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_( x.reshape(-1, self.channel_size).max(dim=0)[0] / math.sqrt(Qp) )
           
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        # global is_plot
        # if is_plot:
        #     plot_tensor_histogram(y, name="y_before_out_proj_" + str(plot_index))
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        return x

class QuantConv2d(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_act(nbits_a=nbits_a)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

    def forward(self, x):
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # if torch.isnan(x).any():
        #     import pdb; pdb.set_trace()
        #     print(f"alpha: {alpha}, weight: {self.weight}")
        # return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QuantLinear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_act(nbits_a=nbits_a)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)


    def forward(self, x):
        
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return F.linear(x, w_q, self.bias)

class QuantLinearv2(_LinearQv2):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4,**kwargs):
        super(QuantLinearv2, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_actv2_linear(nbits_a=nbits_a, channel_size=in_features)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)

    def forward(self, x):
        
        # import pdb;pdb.set_trace()
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:


            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return F.linear(x, w_q, self.bias)

class QuantConv2dv2(_Conv2dQv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QuantConv2dv2, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_actv2_conv(nbits_a=nbits_a, channel_size=in_channels)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

    def forward(self, x):
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QSSLinear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QSSLinear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_act(nbits_a=nbits_a)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
        if self.bias:
            self.bias_vanilla = self.bias.detach().cuda()
        else:
            self.bias_vanilla = None
        self.weight_vanilla = self.weight.detach().cuda()
        

    def forward(self, x):

        with torch.no_grad():
            target = F.linear(x, self.weight_vanilla, self.bias_vanilla)
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        pred = F.linear(x, w_q, self.bias)
        loss = F.l1_loss(pred, target, reduction='mean')
        return pred,  loss

class QSSConv2d(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QSSConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_act(nbits_a=nbits_a)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

        self.weight_vanilla = self.weight.detach().cuda()
        self.bias_vanilla = self.bias.detach().cuda()

    def forward(self, x):

        with torch.no_grad():
            target = F.conv2d(x, self.weight_vanilla, self.bias_vanilla, self.stride,
                            self.padding, self.dilation, self.groups)
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        pred = F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        loss = F.l1_loss(pred, target, reduction='mean')
        return pred, loss


class Quant_VAR_Conv2d(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_VAR_Conv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_var_act(nbits_a=nbits_a, var=2)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

    def forward(self, x):
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # if torch.isnan(x).any():
        #     import pdb; pdb.set_trace()
        #     print(f"alpha: {alpha}, weight: {self.weight}")
        # return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Quant_VAR_Linear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_VAR_Linear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_var_act(nbits_a=nbits_a, var=3)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)


    def forward(self, x):
        x = x + self.channel_threshold
        x = self.QuantSR_act(x)
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return F.linear(x, w_q, self.bias)
     
class Hadamard(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform_fn, self.N, self.had_scale = get_had_fn(dim)

    def to(self, *args, **kwargs):
        super(Hadamard, self).to(*args, **kwargs)
        return self

    def forward(self, x):
        return self.transform_fn(x.contiguous(), self.had_scale) 

    def __repr__(self):
        return f"Hadamard(dim={self.dim}, N={self.N}"
    
class HadLinear(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(HadLinear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        self.QuantSR_act = QuantSR_var_act(nbits_a=nbits_a, var=3)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
        self.had = Hadamard(in_features)
        self.transform_fn, self.N, self.had_scale = get_had_fn(in_features)

    def forward(self, x):
        
        # x = self.had(x)
        x = x + self.channel_threshold
        x = self.had(x)
        x = self.QuantSR_act(x)
        
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        #TODO loss to update alpha
        alpha = grad_scale(self.alpha, g)
        self.had_weight = self.transform_fn(self.weight, self.had_scale)
        w_q = round_pass((self.had_weight / alpha).clamp(Qn, Qp)) * alpha
        
        # import pdb; pdb.set_trace()
        return F.linear(x, w_q, self.bias)