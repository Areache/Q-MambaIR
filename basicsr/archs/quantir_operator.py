import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum
import torch.nn.functional as F
# from .noneed.hadamard_utils import get_had_fn, get_qhad_fn
from .QuantSR_operator import Hadamard
from analysis.plt import plot_tensor_histogram, plot_tensor_3d, plot_tensor_HW

# from quantize.quantizer import UniformAffineQuantizer
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

def round_pass_vanilla(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x, x_o, err_factor):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad  - err_factor*(x - x_o) + (err_factor*(x - x_o)).detach()

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
    if isinstance(layer_type, Conv_Weight_Quant):
        default.update({
            'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, Linear_Weight_Quant):
        pass
    elif isinstance(layer_type, Act_Quant):
        pass
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class Conv_Weight_Quant(nn.Module):
    def __init__(self, in_channels, groups=1, nbits=4):
        super(Conv_Weight_Quant, self).__init__()

        self.nbits = nbits     
        self.alpha = Parameter(torch.Tensor(1, in_channels // groups, 1, 1))
        self.register_buffer('init_state', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1, in_channels // groups, 1, 1))

    def extra_repr(self):
        return '{}'.format(self.nbits)
    
    def forward(self, weight):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        # w_q = round_pass((weight / alpha).clamp(Qn, Qp), weight, self.err_factor) * alpha
        w_q = round_pass_vanilla((weight / alpha).clamp(Qn, Qp)) * alpha

        return w_q

class Weight_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(Weight_Quant, self).__init__()

        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1, in_features))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)

    def forward(self, weight):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass_vanilla((weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = round_pass((weight / alpha).clamp(Qn, Qp), weight, self.err_factor) * alpha

        return w_q

class Linear_Weight_Quant(nn.Module):
    def __init__(self, in_features, nbits=4):
        super(Linear_Weight_Quant, self).__init__()

        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1, in_features))
        self.register_buffer('init_state', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1, in_features))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)

    def forward(self, weight):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass_vanilla((weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = round_pass((weight / alpha).clamp(Qn, Qp), weight, self.err_factor) * alpha

        return w_q

class MUL_Weight_Quant(nn.Module):
    def __init__(self, in_features, nbits=4):
        super(MUL_Weight_Quant, self).__init__()

        self.nbits = nbits
        self.alpha = Parameter(torch.zeros(1, in_features, 1))
        self.register_buffer('init_state', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1, in_features, 1))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)

    def forward(self, weight):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(weight.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        w_q = round_pass_vanilla((weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = round_pass((weight / alpha).clamp(Qn, Qp), weight, self.err_factor) * alpha

        return w_q

class Act_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(Act_Quant, self).__init__()

        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def forward(self, x):
        
        x = x + self.beta

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
            # self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / Qp)
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha

        return x

class Quant_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_conv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        
        # if in_channels == out_channels:
        #     self.learnable_shortcut = nn.Parameter(torch.zeros(1))
        self.conv_act_quant = Act_Quant(nbits=nbits_a)
        self.conv_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x):

        x_q = self.conv_act_quant(x)
        w_q = self.conv_weight_quant(self.weight)
        
        # if self.in_channels == self.out_channels:
        #     return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, 
        #                     self.dilation, self.groups) + x*self.learnable_shortcut
        # else:
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)

class Quant_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # if in_features == out_features:
        #     self.learnable_shortcut = nn.Parameter(torch.zeros(1))
        self.linear_act_quant = Act_Quant(nbits=nbits_a)
        self.linear_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x):

        x_q = self.linear_act_quant(x)
        w_q = self.linear_weight_quant(self.weight)
        
        # if self.in_features == self.out_features:
        #     return F.linear(x_q, w_q, self.bias) + x*self.learnable_shortcut
        # else:
        return F.linear(x_q, w_q, self.bias)

class Quant_out_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_out_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        self.linear_act_quant1 = Act_Quant(nbits=nbits_a)
        self.linear_act_quant2 = Act_Quant(nbits=nbits_a)
        self.linear_act_quant3 = Act_Quant(nbits=nbits_a)
        self.linear_act_quant4 = Act_Quant(nbits=nbits_a)
        self.linear_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x1, x2, x3, x4):

        x_q1 = self.linear_act_quant1(x1)
        x_q2 = self.linear_act_quant2(x2)
        x_q3 = self.linear_act_quant3(x3)
        x_q4 = self.linear_act_quant4(x4)
        x_q = x_q1 + x_q2 + x_q3 + x_q4
        w_q = self.linear_weight_quant(self.weight)
        
        return F.linear(x_q, w_q, self.bias)

class Had_linear(nn.Module):
    def __init__(self, quantlinear: Quant_linear):
        super(Had_linear, self).__init__()
        
        self.had = Hadamard(quantlinear.in_features)
        quantlinear.weight.data = self.had(quantlinear.weight)
        self.quantlinear = quantlinear

    def forward(self, x):
        # import pdb;pdb.set_trace() 
        
        # self.quantlinear.weight.data = self.quantlinear.linear_weight_quant(self.quantlinear.weight)
        # x = self.quantlinear.linear_act_quant(self.had(x) + self.quantlinear.channel_threshold)
        return self.quantlinear(self.had(x))
        # return F.linear(x, self.quantlinear.weight, self.quantlinear.bias)
    
