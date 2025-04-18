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
                NotImplementedError
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / Qp)
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            NotImplementedError
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        out = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha - self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # import pdb; pdb.set_trace()
        return out

class Quant_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_conv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        
        # self.conv_act_quant = BL_L_Linear_act(nbits=nbits_a)
        self.conv_act_quant = Act_Quant(nbits=nbits_a)
        
        # self.conv_act_quant = LTQ_Conv_act(nbits=nbits_a)
        # self.conv_weight_quant = Weight_Quant(nbits=nbits_w)
        self.conv_weight_quant = oneRK_U(nbits=nbits_w)
        
    def forward(self, x):
        
        # import pdb;pdb.set_trace()
        # plot_tensor_histogram(x, name='Quant_conv_x')
        x_q = self.conv_act_quant(x)
        # plot_tensor_histogram(x_q, name='in_proj_in_after_quant')
        w_q = self.conv_weight_quant(self.weight)
        # plot_tensor_histogram(w_q, name='in_proj_weight_after_quant')
        
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Quant_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # self.linear_act_quant = BL_L_Linear_act(nbits=nbits_a)
        self.linear_act_quant = Act_Quant(nbits=nbits_a)
        # self.linear_act_quant = LTQ_yj(nbits=nbits_a)
        
        # self.linear_weight_quant = Weight_Quant(nbits=nbits_w)
        self.linear_weight_quant = oneRK_U(nbits=nbits_w)

    def forward(self, x):

        x_q = self.linear_act_quant(x)
        w_q = self.linear_weight_quant(self.weight)
        
        return F.linear(x_q, w_q, self.bias)

class Quant_out_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_out_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # self.linear_act_quant = BL_L_Linear_act(nbits=nbits_a)
        self.linear_act_quant = DDA_Quant(nbits=nbits_a)
        # self.linear_act_quant = LTQ_yj(nbits=nbits_a)
        
        # self.linear_weight_quant = Weight_Quant(nbits=nbits_w)
        self.linear_weight_quant = oneRK_U(nbits=nbits_w)

    def forward(self, x):

        x_q = self.linear_act_quant(x)
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
    
class out_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(out_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        self.linear_act_quant = Act_Quant(nbits=nbits_a)
        self.linear_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x, index):

        # plot_tensor_histogram(x, name="outproj_before_quant" + str(index))
        x_q = self.linear_act_quant(x)
        # plot_tensor_histogram(x_q, name="outproj_after_quant" + str(index))
        w_q = self.linear_weight_quant(self.weight)
        
        return F.linear(x_q, w_q, self.bias)
    
from scipy.linalg import hadamard
class in_out_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(in_out_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer('init_state', torch.zeros(1))
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
        # self.linear_act_quant = LTQ_Linear_act(in_features, nbits=nbits_a)
        # self.linear_act_quant = PoT_Linear_act(in_features, nbits=nbits_a)
        self.linear_act_quant = PoT_BK_Linear_act(in_features, nbits=nbits_a)
        # self.linear_act_quant = BL_L_Linear_act(in_features, nbits=nbits_a)
        self.linear_weight_quant = Linear_Weight_Quant(in_features, nbits=nbits_w)

    def forward(self, x):
        
        x = x + self.channel_threshold
        x_q = self.linear_act_quant(x)
        w_q = self.linear_weight_quant(self.weight)

        return F.linear(x_q, w_q, self.bias)

class ss2d_dwconv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(ss2d_dwconv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))
        self.conv_act_quant = Act_Quant(nbits=nbits_a)
        self.conv_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x):

        x = x + self.channel_threshold
        # plot_tensor_histogram(x, name="activation_before_quant" + str(plot_index))
        # print('###### activation_before_out_proj_%d is finished ######' % plot_index)
        x_q = self.conv_act_quant(x)
        # plot_tensor_histogram(x_q, name="activation_after_quant" + str(plot_index))
        # print('###### activation_after_out_proj_%d is finished ######' % plot_index)
        # plot_tensor_histogram(self.weight, name="weight_before_quant" + str(plot_index))
        # print('###### weight_before_out_proj_%d is finished ######' % plot_index)
        w_q = self.conv_weight_quant(self.weight)
        # plot_tensor_histogram(self.weight, name="weight_after_quant" + str(plot_index))
        # print('###### weight_after_quant_out_proj_%d is finished ######' % plot_index)
        
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class y_Act_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(y_Act_Quant, self).__init__()

        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def forward(self, x):
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                NotImplementedError
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # mean = torch.mean(x.detach())
            # std = torch.std(x.detach())
            # self.alpha.data.copy_( 2*max(torch.abs(mean-4*std), torch.abs(mean + 4*std)) / Qp)
            # self.alpha.data.copy_( 8*std / Qp)
            self.init_state.fill_(1)
        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            NotImplementedError
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha

        return x
    
class BL_L_Linear_act(nn.Module):
    
    def __init__(self,in_features, nbits=4):
        super(BL_L_Linear_act, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** 4 - 1)), requires_grad=True)
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        # self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        # self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        # self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        # self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)

    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        additive_pot = [-7, -6, -5, -4, -3, -2, -1,  0.0,1, 2, 3, 4, 5, 6, 7]
        # print(a_pos)
        # print(additive_pot)
        # import pdb; pdb.set_trace()
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        p_1_p = additive_pot[0]
        p_2_p = additive_pot[1] 
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            k_p = 100 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = start
                thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p            
                delta_p = additive_pot[1]-additive_pot[0]
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                # y_step_p = x
                # y_step_p = self.phi_function_p(x+ delta_p/2-(a_pos[0] + a_pos[1])/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_backward = torch.where(x > thre_backward_p, y_step_p,  0.1*x+additive_pot[i]-0.1*a_pos[i])
                # x_backward = torch.where(x > thre_backward_p, y_step_p,  x)
            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                thre_backward_p = a_pos[i]
                p_1_p = a_pos[i-1]
                p_2_p = a_pos[i]
                # p_1_n += b_pos[i-1]
                # p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                if i!=len(a_pos)-1:
                    delta_p = (additive_pot[i+1] - additive_pot[i])
                    # y_step_p = x
                    y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = self.phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                else:
                    delta_p = 0
                    y_step_p = x
                    # y_step_p = self.phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_back_round_2")
        # plot_tensor_histogram(x_forward, name="PoT_for_round_2")
        # plot_tensor_histogram(x, name="ltq_x_for_round_4")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_4")
        # print(self.a)
        # import pdb; pdb.set_trace()
        #! APoT
        # tyler = self.build_power_value(B=3, additive=True)
        #! PoT
        tyler = [-7, -6, -5, -4, -3, -2, -1,  0.0,1, 2, 3, 4, 5, 6, 7]
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            # self.alpha.data.copy_((x.max() - x.min()) / 2)
            # self.alpha.data.copy_(x.abs().max())            
            # self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            # tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
            
            self.a.data = torch.Tensor(tyler).to(x.device)
            # self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # plot_tensor_histogram(x, name="PoT_in_2")
        out = self.round_ltq((x / alpha).clamp(-7, 7)) * alpha + self.beta
        # out = self.round_ltq(x / self.alpha) * self.alpha + self.beta
        # plot_tensor_histogram(out, name="PoT_out_2")
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")

        return out

class PoT_Linear_act(nn.Module):
    
    def __init__(self,in_features, nbits=4):
        super(PoT_Linear_act, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** 4 - 1)), requires_grad=True)
        self.a = torch.Tensor([-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, 0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]) 
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        # self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        # self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        # self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        # self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3

    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        additive_pot = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        # print(a_pos)
        # print(additive_pot)
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        p_1_p = additive_pot[0]
        p_2_p = additive_pot[1] 
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            k_p = 100 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = start
                # thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p            
                delta_p = additive_pot[1]-additive_pot[0]
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                # y_step_p = x
                # y_step_p = self.phi_function_p(x+ delta_p/2-(a_pos[0] + a_pos[1])/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                # y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                # x_backward = torch.where(x > thre_backward_p, y_step_p,  0.1*x+additive_pot[i]-0.1*a_pos[i])
                # x_backward = torch.where(x > thre_backward_p, y_step_p,  x)
            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                # thre_backward_p = a_pos[i]
                p_1_p = a_pos[i-1]
                p_2_p = a_pos[i]
                # p_1_n += b_pos[i-1]
                # p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                # if i!=len(a_pos)-1:
                    # delta_p = (additive_pot[i+1] - additive_pot[i])
                    # y_step_p = x
                    # y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = self.phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                # else:
                #     delta_p = 0
                    # y_step_p = x
                    # y_step_p = self.phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                # x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        # x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_back_round_2")
        # plot_tensor_histogram(x_forward, name="PoT_for_round_2")
        # plot_tensor_histogram(x, name="ltq_x_for_round_4")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_4")
        # print(self.a)
        # import pdb; pdb.set_trace()
        #! APoT
        # tyler = self.build_power_value(B=3, additive=True)
        #! PoT
        # tyler = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp)*15/2)
            # self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            self.alpha.data.copy_((x.max() - x.min()) / 2)
            # self.alpha.data.copy_(x.abs().max())            
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            # tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
            
            # self.a.data = torch.Tensor(tyler).to(x.device)
            # self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # plot_tensor_histogram(x, name="PoT_in_4")
        out = self.round_ltq((x / alpha).clamp(Qn, Qp)) * alpha + self.beta
        # out = self.round_ltq(x / self.alpha) * self.alpha + self.beta
        # plot_tensor_histogram(out, name="PoT_out_4")
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")
        

        return out

#!二段反向传播 非均匀
class PoT_BK_Linear_act(nn.Module):
    
    def __init__(self,in_features, nbits=4):
        super(PoT_BK_Linear_act, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** 4 - 1)), requires_grad=True)
        # self.a = torch.Tensor([-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]) 
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        # self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        # self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        # self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        # self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3

    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        additive_pot = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        ratio = 0.25
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            # k_p = 100 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = start
                # thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                delta_p = additive_pot[1]-additive_pot[0]
                k_b = 6
                thre_x = (a_pos[i] + a_pos[i+1])/2
                thre_backward_p = thre_x - 0.5*delta_p/k_b
                y_step_0 = k_b*(x - thre_backward_p) + additive_pot[0]                
                x_backward = torch.where(x > thre_backward_p, y_step_0, 0.1*(x-thre_backward_p)+additive_pot[0])

            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                thre_backward_p = thre_forward_p - 0.5*delta_p/k_b          
                if i!=len(a_pos)-1:
                    delta_p = (additive_pot[i] - additive_pot[i-1])
                    thre_backward_f = thre_forward_p + ratio*delta_p/k_b
                    y_step_f = ((1-ratio)*additive_pot[i] + ratio*additive_pot[i+1]-((1-ratio)*additive_pot[i] + ratio*additive_pot[i-1]))\
                    /((a_pos[i] + a_pos[i+1])/2 - (0.25-0.5*ratio)*(additive_pot[i+1] - additive_pot[i])/k_b-((a_pos[i-1] + a_pos[i])/2 + (0.25-0.5*ratio)*(additive_pot[i] - additive_pot[i-1])/k_b)) \
                    * (x - ((a_pos[i-1] + a_pos[i])/2 + (0.25-0.5*ratio)*(additive_pot[i] - additive_pot[i-1])/k_b)) + (1-ratio)*additive_pot[i] + ratio*additive_pot[i-1]

                    k_b = 6
                    thre_x = (a_pos[i-1] + a_pos[i])/2
                    y_step_p = k_b*(x - thre_x) + 0.5*(additive_pot[i-1]+additive_pot[i])
                    # y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                    thre_backward_p = thre_forward_p - (0.25-0.5*ratio)*delta_p/k_b 
                    thre_backward_f = thre_forward_p + (0.25-0.5*ratio)*delta_p/k_b            
                else:
                    delta_p = additive_pot[i] - additive_pot[i-1]
                    # y_step_p = 0.1*x+additive_pot[i]-0.1*a_pos[i]
                    k_b = 6
                    thre_x = (a_pos[i-1] + a_pos[i])/2
                    y_step_p = k_b*(x - thre_x) + 0.5*(additive_pot[i-1]+additive_pot[i])
                    thre_backward_p = thre_forward_p - (0.25-0.5*ratio)*delta_p/k_b
                    thre_backward_f = thre_forward_p + 0.5*delta_p/k_b
                    # #! ((a_pos[i-1] + a_pos[i])/2 + 0.25*(additive_pot[i] - additive_pot[i-1])/k_b, 0.75*additive_pot[i] + 0.25*additive_pot[i-1])
                    # #! ((a_pos[i] + a_pos[i+1])/2 - 0.25*(additive_pot[i+1] - additive_pot[i])/k_b, 0.75*additive_pot[i+1] - 0.25*additive_pot[i])
                    # y_step_f = (0.75*additive_pot[i+1] + 0.25*additive_pot[i]-(0.75*additive_pot[i] + 0.25*additive_pot[i-1]))/((a_pos[i] + a_pos[i+1])/2 + 0.25*(additive_pot[i+1] - additive_pot[i])/k_b-((a_pos[i-1] + a_pos[i])/2 + 0.25*(additive_pot[i] - additive_pot[i-1])/k_b)) * (x - thre_backward_f) + 0.75*additive_pot[i] + 0.25*additive_pot[i-1]
                    y_step_f = 0.1*(x - (thre_forward_p + 0.5*delta_p/k_b))+additive_pot[i]

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                k_s = 0.1
                if i!=1:
                    x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                x_backward = torch.where(x > thre_backward_f, y_step_f, x_backward)

        # p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        # x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_x_backward_1")
        # plot_tensor_histogram(x_forward, name="PoT_x_forward_1")
        # plot_tensor_histogram(x, name="PoT_x_1")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_4")
        # print(self.a)
        # import pdb; pdb.set_trace()
        #! APoT
        # tyler = self.build_power_value(B=3, additive=True)
        #! PoT
        tyler = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp)*15/2)
            # self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            self.alpha.data.copy_((x.max() - x.min()) / 2)
            # self.alpha.data.copy_(x.abs().max())            
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            # tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
            
            self.a.data = torch.Tensor(tyler).to(x.device)
            # self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # plot_tensor_histogram(x, name="PoT_in_2")
        out = self.round_ltq((x / alpha).clamp(-1, 1)) * alpha + self.beta
        # out = self.round_ltq(x / self.alpha) * self.alpha + self.beta
        # plot_tensor_histogram(out, name="PoT_out_2")
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")
        

        return out

#!二段反向传播 非均匀
class twoBK_N_r1(nn.Module):
    
    def __init__(self, nbits=4):
        super(twoBK_N_r1, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** 4 - 1)), requires_grad=True)
        # self.a = torch.Tensor([-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]) 
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        # self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        # self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        # self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        # self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3

    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        additive_pot = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        ratio = 0.25
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            # k_p = 100 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = start
                # thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                delta_p = additive_pot[1]-additive_pot[0]
                k_b = 6
                thre_x = (a_pos[i] + a_pos[i+1])/2
                thre_backward_p = thre_x - 0.5*delta_p/k_b
                y_step_0 = k_b*(x - thre_backward_p) + additive_pot[0]                
                x_backward = torch.where(x > thre_backward_p, y_step_0, 0.1*(x-thre_backward_p)+additive_pot[0])

            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                thre_backward_p = thre_forward_p - 0.5*delta_p/k_b          
                if i!=len(a_pos)-1:
                    delta_p = (additive_pot[i] - additive_pot[i-1])
                    thre_backward_f = thre_forward_p + ratio*delta_p/k_b
                    y_step_f = ((1-ratio)*additive_pot[i] + ratio*additive_pot[i+1]-((1-ratio)*additive_pot[i] + ratio*additive_pot[i-1]))\
                    /((a_pos[i] + a_pos[i+1])/2 - (0.25-0.5*ratio)*(additive_pot[i+1] - additive_pot[i])/k_b-((a_pos[i-1] + a_pos[i])/2 + (0.25-0.5*ratio)*(additive_pot[i] - additive_pot[i-1])/k_b)) \
                    * (x - ((a_pos[i-1] + a_pos[i])/2 + (0.25-0.5*ratio)*(additive_pot[i] - additive_pot[i-1])/k_b)) + (1-ratio)*additive_pot[i] + ratio*additive_pot[i-1]

                    k_b = 6
                    thre_x = (a_pos[i-1] + a_pos[i])/2
                    y_step_p = k_b*(x - thre_x) + 0.5*(additive_pot[i-1]+additive_pot[i])
                    # y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                    thre_backward_p = thre_forward_p - (0.25-0.5*ratio)*delta_p/k_b 
                    thre_backward_f = thre_forward_p + (0.25-0.5*ratio)*delta_p/k_b            
                else:
                    delta_p = additive_pot[i] - additive_pot[i-1]
                    # y_step_p = 0.1*x+additive_pot[i]-0.1*a_pos[i]
                    k_b = 6
                    thre_x = (a_pos[i-1] + a_pos[i])/2
                    y_step_p = k_b*(x - thre_x) + 0.5*(additive_pot[i-1]+additive_pot[i])
                    thre_backward_p = thre_forward_p - (0.25-0.5*ratio)*delta_p/k_b
                    thre_backward_f = thre_forward_p + 0.5*delta_p/k_b
                    # #! ((a_pos[i-1] + a_pos[i])/2 + 0.25*(additive_pot[i] - additive_pot[i-1])/k_b, 0.75*additive_pot[i] + 0.25*additive_pot[i-1])
                    # #! ((a_pos[i] + a_pos[i+1])/2 - 0.25*(additive_pot[i+1] - additive_pot[i])/k_b, 0.75*additive_pot[i+1] - 0.25*additive_pot[i])
                    # y_step_f = (0.75*additive_pot[i+1] + 0.25*additive_pot[i]-(0.75*additive_pot[i] + 0.25*additive_pot[i-1]))/((a_pos[i] + a_pos[i+1])/2 + 0.25*(additive_pot[i+1] - additive_pot[i])/k_b-((a_pos[i-1] + a_pos[i])/2 + 0.25*(additive_pot[i] - additive_pot[i-1])/k_b)) * (x - thre_backward_f) + 0.75*additive_pot[i] + 0.25*additive_pot[i-1]
                    y_step_f = 0.1*(x - (thre_forward_p + 0.5*delta_p/k_b))+additive_pot[i]

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                k_s = 0.1
                if i!=1:
                    x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                x_backward = torch.where(x > thre_backward_f, y_step_f, x_backward)

        # p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        # x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_x_backward_1")
        # plot_tensor_histogram(x_forward, name="PoT_x_forward_1")
        # plot_tensor_histogram(x, name="PoT_x_1")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_4")
        # print(self.a)
        # import pdb; pdb.set_trace()
        #! APoT
        # tyler = self.build_power_value(B=3, additive=True)
        #! PoT
        tyler = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp)*15/2)
            # self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            self.alpha.data.copy_((x.max() - x.min()) / 2)
            # self.alpha.data.copy_(x.abs().max())            
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            # tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
            
            self.a.data = torch.Tensor(tyler).to(x.device)
            # self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # plot_tensor_histogram(x, name="PoT_in_2")
        out = self.round_ltq((x / alpha).clamp(-1, 1)) * alpha + self.beta
        # out = self.round_ltq(x / self.alpha) * self.alpha + self.beta
        # plot_tensor_histogram(out, name="PoT_out_2")
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")
        

        return out

#!一段反向 非均匀
class oneBK_N_r1(nn.Module):
    
    def __init__(self, nbits=4):
        super(oneBK_N_r1, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        # self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** 4 - 1)), requires_grad=True)
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        # self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        # self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        # self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        # self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        # self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3
    
    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        additive_pot = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        # print(a_pos)
        # print(additive_pot)
        # import pdb; pdb.set_trace()
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        p_1_p = additive_pot[0]
        p_2_p = additive_pot[1] 
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            k_p = 100 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = start
                thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p            
                delta_p = additive_pot[1]-additive_pot[0]
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                # y_step_p = x
                # y_step_p = self.phi_function_p(x+ delta_p/2-(a_pos[0] + a_pos[1])/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_backward = torch.where(x > thre_backward_p, y_step_p,  0.1*x+additive_pot[i]-0.1*a_pos[i])
                # x_backward = torch.where(x > thre_backward_p, y_step_p,  x)
            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                thre_backward_p = a_pos[i]
                p_1_p = a_pos[i-1]
                p_2_p = a_pos[i]
                # p_1_n += b_pos[i-1]
                # p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                if i!=len(a_pos)-1:
                    delta_p = (additive_pot[i+1] - additive_pot[i])
                    # y_step_p = x
                    y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = self.phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                else:
                    delta_p = 0
                    y_step_p = x
                    # y_step_p = self.phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_back_round_2")
        # plot_tensor_histogram(x_forward, name="PoT_for_round_2")
        # plot_tensor_histogram(x, name="ltq_x_for_round_4")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_4")
        # print(self.a)
        # import pdb; pdb.set_trace()
        #! APoT
        # tyler = self.build_power_value(B=3, additive=True)
        #! PoT
        tyler = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp)*15/2)
            # self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            self.alpha.data.copy_((x.max() - x.min()) / 2)
            # self.alpha.data.copy_(x.abs().max())            
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            # tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
            
            self.a.data = torch.Tensor(tyler).to(x.device)
            # self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # plot_tensor_histogram(x, name="PoT_in_2")
        out = self.round_ltq((x / alpha).clamp(Qn, Qp)) * alpha + self.beta
        # out = self.round_ltq(x / self.alpha) * self.alpha + self.beta
        # plot_tensor_histogram(out, name="PoT_out_2")
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")
        

        return out

#!一段反向 均匀
class oneRK_U(nn.Module):
    def __init__(self, nbits=4):
        super(oneRK_U, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* (2 ** self.nbits)), requires_grad=True)
        # self.err_factor = Parameter(torch.zeros(1))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def round_ltq(self, x):
        a_pos = self.a
        #! APoT
        # additive_pot = self.build_power_value(B=3, additive=True)
        #! PoT
        # additive_pot = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,  0.0,0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
        #! QuantSR_l
        # additive_pot = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
        additive_pot = list(range(-2 ** (self.nbits - 1), 2 ** (self.nbits - 1)))
        # print(a_pos)
        # print(additive_pot)
        # import pdb; pdb.set_trace()
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        p_1_p = additive_pot[0]
        p_2_p = additive_pot[1] 
        for i in range(len(a_pos)):
            # import pdb; pdb.set_trace() 
            # k_p = 100 - (a_pos[i]-a_pos[i-1])
            if i == 0:
                thre_forward_p = start
                thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p 
                delta_p = additive_pot[1]-additive_pot[0]
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                # y_step_p = x
                # y_step_p = self.phi_function_p(x+ delta_p/2-(a_pos[0] + a_pos[1])/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_backward = torch.where(x > thre_backward_p, y_step_p, 0.1*x+additive_pot[i]-0.1*a_pos[i])
                # x_backward = torch.where(x > thre_backward_p, y_step_p,  x)
            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                thre_backward_p = a_pos[i]
                p_1_p = a_pos[i-1]
                p_2_p = a_pos[i]
                # p_1_n += b_pos[i-1]
                # p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                if i!=len(a_pos)-1:
                    delta_p = (additive_pot[i+1] - additive_pot[i])
                    # y_step_p = x
                    y_step_p = delta_p/(a_pos[i+1]-a_pos[i])*x +additive_pot[i]-delta_p/(a_pos[i+1]-a_pos[i])*a_pos[i]
                    # y_step_p = self.phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
                else:
                    delta_p = 0
                    y_step_p = x
                    # y_step_p = self.phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)

                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > thre_backward_p, y_step_p, x_backward)
                # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        x_backward = torch.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # print(x_forward.shape)
        # print(x_backward.shape)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # out = x
        # plot_tensor_histogram(x_backward, name="PoT_back_round_2")
        # plot_tensor_histogram(x_forward, name="PoT_for_round_2")
        # plot_tensor_histogram(x, name="ltq_x_for_round_4")
        # import pdb; pdb.set_trace()

        return out

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
            tyler = list(range(Qn, Qp + 1))
            self.a.data = torch.Tensor(tyler).to(x.device)
            # self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / Qp)
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            NotImplementedError
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        out = self.round_ltq((x / alpha).clamp(Qn, Qp)) * alpha - self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # import pdb; pdb.set_trace()
        return out


class  DDA_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(DDA_Quant, self).__init__()

        self.nbits = nbits
        # self.alpha = Parameter(torch.Tensor(1))
        # self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))
        # self.alpha_predictor = nn.Linear(2,1)
        # self.alpha = Parameter(torch.Tensor(1))
        self.weight1 = Parameter(torch.Tensor([1/(2**(self.nbits-1)-1)]))
        self.weight2 = Parameter(torch.Tensor([3/(2**(self.nbits-1)-1)]))
        self.weight3 = Parameter(torch.Tensor([0]))
        self.weight4 = Parameter(torch.Tensor([0]))
        self.weight21 = Parameter(torch.Tensor([-1]))
        self.weight22 = Parameter(torch.Tensor([0]))
        self.weight23 = Parameter(torch.Tensor([0]))
        self.weight24 = Parameter(torch.Tensor([0]))
    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        alpha_max = torch.tensor((x.max()-x.min())/ 2**self.nbits)
        x_max = x.max()       
        x_min = x.min()       
        # plot_tensor_histogram(x, "input21")
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
            # self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / Qp)
            # self.beta.data.copy_(-(x.max() + x.min()) / 2)
            # self.beta.data.copy_(-x.mean())
            self.init_state.fill_(1)
        self.beta = self.weight21*x.mean()+self.weight22*x.std()+self.weight23*x.min()+self.weight24*x.max()
        x = x + self.beta   
        # plot_tensor_histogram(x, "input22")
        self.alpha = (self.weight1*(x.mean()).abs()+self.weight2*x.std()+self.weight3*x.min()+self.weight4*x.max()).abs()
        # alpha_x = torch.cat((x.mean().unsqueeze(0), x.var().unsqueeze(0)), dim=0)
        # self.alpha = self.alpha_predictor(alpha_x)
        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        # plot_tensor_histogram(x/self.alpha, "x_div_self.alpha")
        # plot_tensor_histogram(round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha, "after_round")
        alpha = torch.maximum(alpha, torch.tensor(1e-6, device=alpha.device))
        alpha = torch.minimum(alpha, torch.tensor(alpha_max, device=alpha.device))
        x = (round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha - self.beta).clamp(x_min,x_max)
        # plot_tensor_histogram(x, "input24")
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # import pdb; pdb.set_trace()
        return x

