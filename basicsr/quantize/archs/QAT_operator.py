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

class linear_act(nn.Module):
    def __init__(self, nbits_a=4,):
        super(linear_act, self).__init__()
        self.nbits = nbits_a

    def forward(self, x, symmetric, layerwise):
        
        if symmetric:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
            if layerwise:
                max_input = torch.max(torch.abs(x)) # .expand_as(x)
            else:
                # import pdb; pdb.set_trace()
                if x.ndimension() <= 3:
                    # weight & hidden layer
                    max_input = (
                        torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
                        .expand_as(x)
                        .detach()
                    )
                elif x.ndimension() == 4:
                    max_input = (
                        torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
                        .expand_as(x)
                        .detach()
                    )
                else:
                    raise ValueError
                
            alpha = Qp / (max_input + 1e-6)
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
            if layerwise:
                self.alpha.data.copy_((x.max() - x.min()))
                self.beta.data.copy_(x.min())
            else:
                if x.ndimension() <= 3:
                    # weight & hidden layer
                    alpha = (
                        (
                            x.max(dim=-1, keepdim=True)[0]
                            - x.min(dim=-1, keepdim=True)[0]
                        )
                        .expand_as(x)
                        .detach()
                    )
                    beta = (x.min(dim=-1, keepdim=True)[0].expand_as(x).detach())

                elif x.ndimension() == 4:
                    
                    # import pdb;pdb.set_trace()
                    alpha = (
                        (
                            x.max(dim=-1, keepdim=True)[0]
                            - x.min(dim=-1, keepdim=True)[0]
                        )
                        .expand_as(x)
                        .detach()
                    )
                    
                    beta = (x.min(dim=-1, keepdim=True)[0].expand_as(x).detach())
                else:
                    raise ValueError


        if symmetric:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        if symmetric:
            x = torch.round(x * alpha).div(alpha + 1e-6)
        else:
            input_normalized = (x - beta) / (alpha + 1e-8)
            x = torch.round(input_normalized * Qp).div(Qp)
            x = x * (alpha + 1e-8) + beta
        return x

class conv2d_act(nn.Module):
    def __init__(self, nbits_a=4,):
        super(conv2d_act, self).__init__()
        self.nbits = nbits_a

    def forward(self, x, symmetric, layerwise):
        
        if symmetric:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
            if layerwise:
                max_input = torch.max(torch.abs(x)) # .expand_as(x)
            else:
                # import pdb; pdb.set_trace()
                if x.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    # tmp = x.view(x.shape[0], x.shape[1], -1)
                    max_input = (
                        torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
                        .expand_as(x)
                        .detach()
                    )
                else:
                    raise ValueError
            alpha = Qp / (max_input + 1e-6)
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
            if layerwise:
                self.alpha.data.copy_((x.max() - x.min()))
                self.beta.data.copy_(x.min())
            else:
                if x.ndimension() == 4:
                    # TODO: per channel quantization
                    alpha = (
                        (
                            x.max(dim=1, keepdim=True)[0]
                            - x.min(dim=1, keepdim=True)[0]
                        )
                        .expand_as(x)
                        .detach()
                    )
                    beta = (
                        x.min(dim=1, keepdim=True)[0]
                        # .unsqueeze(-1)
                        .expand_as(x)
                        .detach()
                    )
                else:
                    raise ValueError

        if symmetric:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        if symmetric:
            x = torch.round(x * alpha).div(alpha + 1e-6)
        else:
            input_normalized = (x - beta) / (alpha + 1e-8)
            x = torch.round(input_normalized * Qp).div(Qp)
            x = x * (alpha + 1e-8) + beta
        return x

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 nbits_w=4, nbits_a=4, symmetric=False, **kwargs):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        self.conv2d_act = conv2d_act(nbits_a=nbits_a)
        self.symmetric = symmetric
        self.nbits_w = nbits_w

    def forward(self, x):

        x = self.conv2d_act(x, layerwise = False, symmetric = self.symmetric)
        
        real_weights = self.weight
        num_bits = 2 ** (self.nbits_w - 1)
        clip_val = 1 - 1e-2
        scaling_factor = (
            2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
        )
        quan_weights_no_grad = (
            scaling_factor
            * (
                torch.round(
                    torch.clamp(
                        real_weights / scaling_factor, -clip_val, clip_val
                    )
                    * num_bits
                    - 0.5
                )
                + 0.5
            )
            / num_bits
        )

        weight = (
            quan_weights_no_grad.detach() - real_weights.detach() + real_weights
        )

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, symmetric=False):
        super(QuantLinear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias)
        self.linear_act = linear_act(nbits_a=nbits_a)
        self.symmetric = symmetric
        self.nbits_w = nbits_w

    def forward(self, x):
        
        x = self.linear_act(x, layerwise = False, symmetric = self.symmetric)

        real_weights = self.weight
        num_bits = 2 ** (self.nbits_w - 1)
        clip_val = 1 - 1e-2
        scaling_factor = (
            2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
        )
        quan_weights_no_grad = (
            scaling_factor
            * (
                torch.round(
                    torch.clamp(
                        real_weights / scaling_factor, -clip_val, clip_val
                    )
                    * num_bits
                    - 0.5
                )
                + 0.5
            )
            / num_bits
        )

        weight = (
            quan_weights_no_grad.detach() - real_weights.detach() + real_weights
        )

        out = F.linear(x, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
    
        return  out


