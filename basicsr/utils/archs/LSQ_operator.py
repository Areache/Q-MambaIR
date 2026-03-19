import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum
import torch.nn.functional as F
# from .hadamard_utils import get_had_fn, get_qhad_fn
from analysis.plt import plot_tensor_histogram, plot_tensor_3d, plot_tensor_HW

from quantize.quantizer import UniformAffineQuantizer
import time


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
    # out3 = 
    input = x
    x_round = input.round()
    x = input - input.floor().detach()
    x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
    out3 = x + input.floor().detach()
    return x_round.detach() - out3.detach() + out3

def round_pass_vanilla(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

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
    elif isinstance(layer_type, _WeightQ):
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
        # self.alpha = Parameter(torch.Tensor(1))
        self.alpha = Parameter(torch.ones(out_features, 1))
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
        self.beta = Parameter(torch.Tensor(1))
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

class _WeightQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_WeightQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        out_features = kwargs_q['out_features']
        is_conv = kwargs_q['is_conv']
        is_matmut = kwargs_q['is_matmut']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        # self.alpha = Parameter(torch.Tensor(1))
        self.alpha = Parameter(torch.ones(out_features))
        # if is_conv:
        #     self.alpha = Parameter(torch.ones(out_features, 1, 1, 1))
        # elif is_matmut:
        #     self.alpha = Parameter(torch.ones(1, 1, out_features))
        # else:
        #     self.alpha = Parameter(torch.ones(out_features, 1))
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

from torch.autograd import Function
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = round_pass_vanilla(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        grad_alpha = ((smaller * Qn + bigger * Qp + 
            between * round_pass_vanilla(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        #返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha,  None, None, None, grad_beta

class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            # if alpha.shape[0] == 72:
            #     import pdb; pdb.set_trace()
            sizes = weight.size()
            shape = alpha.shape[0]
            if shape in sizes:
                dimension_index = sizes.index(shape)
            # non_one_dims = [i for i, size in enumerate(alpha.shape) if size != 1]
            weight = weight.contiguous().view(weight.size()[dimension_index], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = round_pass_vanilla(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = round_pass_vanilla(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha 
        return w_q
    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            out_features = alpha.shape[0]
            if out_features in sizes:
                dimension_index = sizes.index(out_features)
            weight = weight.contiguous().view(weight.size()[dimension_index], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + between * round_pass(q_w) - between * q_w)*grad_weight * g)
            if out_features in grad_alpha.size():
                features_index = grad_alpha.size().index(out_features)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[features_index], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * round(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        # import pdb; pdb.set_trace()
        return grad_weight, grad_alpha, None, None, None, None

class LSQ_act_quant(_ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(LSQ_act_quant, self).__init__(nbits=nbits_a)

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
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            mina = torch.min(x.detach())
            #! minmax
            self.alpha.data.copy_( (torch.max(x.detach()) - mina) / (Qp - Qn) )
            self.beta.data.copy_(mina - self.alpha.data * Qn)
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        
        # mina = torch.min(x.detach())
        # self.alpha.data.copy_( self.alpha.data*0.9 + 0.1*((torch.max(x.detach())-mina)/(Qp - Qn)) )
        # self.beta.data.copy_( self.beta.data*0.9 + 0.1*(mina - self.alpha.data * Qn))

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # alpha = grad_scale(self.alpha, g)
        # beta = grad_scale(self.beta, g)
        # global is_plot
        # if is_plot:
        #     plot_tensor_histogram(y, name="y_before_out_proj_" + str(plot_index))
        # x = round_pass(((x - beta) / alpha).clamp(Qn, Qp)) * alpha + beta
        # x = round_pass_vanilla(((x - beta) / alpha).clamp(Qn, Qp)) * alpha + beta
        x = ALSQPlus.apply(x, self.alpha, g, Qn, Qp, self.beta)
        return x

class LSQ_weight_quant(_WeightQ):
    def __init__(self, nbits_a=4, out_features=None, is_conv=False, is_matmut=False, **kwargs):
        super(LSQ_weight_quant, self).__init__(nbits=nbits_a, out_features=out_features, is_conv=is_conv, is_matmut=is_matmut)

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
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # # minmax
            # mina = torch.min(x.detach())
            # self.alpha.data.copy_( (torch.max(x.detach()) - mina) / (Qp - Qn) )
            #! std
            mean = torch.mean(x.detach())
            std = torch.std(x.detach())
            self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / (2**self.nbits-1))
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        # mina = torch.min(x.detach())
        # self.alpha.data.copy_( self.alpha.data*0.9 + 0.1*((torch.max(x.detach())-mina)/(Qp - Qn)) )
        # mean = torch.mean(x.detach())
        # std = torch.std(x.detach())
        # self.alpha.data.copy_( self.alpha.data*0.9 + 0.1*(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / (2**self.nbits-1)) )

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # alpha = grad_scale(self.alpha, g)
        # x = round_pass(( x / alpha).clamp(Qn, Qp)) * alpha
        x = WLSQPlus.apply(x, self.alpha, g, Qn, Qp, True) 
        return x

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        self.LSQ_act_quant = LSQ_act_quant(nbits_a=nbits_a)
        self.LSQ_weight_quant = LSQ_weight_quant(nbits_a=nbits_w, out_features=out_channels, is_conv=True)

    def forward(self, x):
        
        act_q = self.LSQ_act_quant(x)
        w_q = self.LSQ_weight_quant(self.weight)
        
        return F.conv2d(act_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(QuantLinear, self).__init__(
            in_features=in_features,out_features=out_features, bias=bias)
        self.LSQ_act_quant = LSQ_act_quant(nbits_a=nbits_a)
        self.LSQ_weight_quant = LSQ_weight_quant(nbits_a=nbits_w, out_features=out_features, is_conv=False)

    def forward(self, x):
        
        # import pdb;pdb.set_trace()
        act_q = self.LSQ_act_quant(x)
        w_q = self.LSQ_weight_quant(self.weight)

        return F.linear(act_q, w_q, self.bias)