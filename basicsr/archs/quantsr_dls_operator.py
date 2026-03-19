import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum
import torch.nn.functional as F


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
    return y.detach() - y_grad.detach() + y_grad


# PAMS TorchRound function (from PAMS_operator.py)
def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply


def round_pass(x):
    """Round pass without tanh, using PAMS TorchRound"""
    input = x
    x_round = input.round()
    # Use simple STE instead of tanh-based soft rounding
    return x_round.detach() - input.detach() + input


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


# PAMS quant_max function
def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8


# PAMS quant_weight class
class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):
        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight


# PAMS pams_quant_act class
class pams_quant_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits=4, ema_epoch=1, decay=0.9997):
        super(pams_quant_act, self).__init__()
        self.decay = decay
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits -1) -1.
        self.round = TorchRound()
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.ema_epoch = ema_epoch
        self.epoch = 1
        self.register_buffer('max_val', torch.ones(1))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.alpha, 10)

    def _ema(self, x):
        x = abs(x)
        for i in range(len(x.shape) - 1):
            x = torch.max(x, dim=1)[0]
        max_val = torch.mean(x)

        if self.epoch == 1:
            self.max_val = max_val
        else:
            self.max_val = (1.0-self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):
        if self.epoch > self.ema_epoch or not self.training:
            act = torch.max(torch.min(x, self.alpha), -self.alpha)
        elif self.epoch <= self.ema_epoch and self.training:
            act = x
            self._ema(x)
            self.alpha.data = self.max_val.unsqueeze(0)
        
        act = act * self.qmax / self.alpha
        q_act = self.round(act)
        q_act = q_act * self.alpha / self.qmax
    
        return q_act


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


class ActLSQ_soft(_ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(ActLSQ_soft, self).__init__(nbits=nbits_a)

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
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        
        return x


class Conv2dLSQ_soft(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Conv2dLSQ_soft, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, nbits_a=nbits_a)
        # Use PAMS quant_weight for weight quantization
        self.quant_weight = quant_weight(k_bits=nbits_w)
        # Use PAMS pams_quant_act for activation quantization
        self.pams_act_quant = pams_quant_act(k_bits=nbits_a)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

    def forward(self, x):
        x = x + self.channel_threshold
        x = self.pams_act_quant(x)
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # Use PAMS quant_weight instead of LSQ weight quantization
        w_q = self.quant_weight(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearLSQ_soft(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(LinearLSQ_soft, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, nbits_a=nbits_a)
        # Use PAMS quant_weight for weight quantization
        self.quant_weight = quant_weight(k_bits=nbits_w)
        # Use PAMS pams_quant_act for activation quantization
        self.pams_act_quant = pams_quant_act(k_bits=nbits_a)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)

    def forward(self, x):
        x = x + self.channel_threshold
        x = self.pams_act_quant(x)
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        # Use PAMS quant_weight instead of LSQ weight quantization
        w_q = self.quant_weight(self.weight)
        return F.linear(x, w_q, self.bias)

