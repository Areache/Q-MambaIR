import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum
import torch.nn.functional as F
# from .noneed.hadamard_utils import get_had_fn, get_qhad_fn
from .QuantSR_operator import Hadamard
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
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha - self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha

        return x

class Quant_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_conv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        
        self.conv_act_quant = Act_Quant(nbits=nbits_a)
        # self.conv_act_quant = LTQ_yj(nbits=nbits_a)
        self.conv_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x):
        
        # import pdb;pdb.set_trace()
        # plot_tensor_histogram(x, name='in_proj_in_before_quant')
        x_q = self.conv_act_quant(x)
        # plot_tensor_histogram(x_q, name='in_proj_in_after_quant')
        w_q = self.conv_weight_quant(self.weight)
        # plot_tensor_histogram(w_q, name='in_proj_weight_after_quant')
        
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Quant_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4, **kwargs):
        super(Quant_linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        self.linear_act_quant = Act_Quant(nbits=nbits_a)
        # self.linear_act_quant = LTQ_yj(nbits=nbits_a)
        
        self.linear_weight_quant = Weight_Quant(nbits=nbits_w)

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
        self.linear_act_quant = LTQ_yj(in_features, nbits=nbits_a)
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
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha

        return x
    
def quaternion_rotation(tensor, quaternion):
    """
    对任意形状的张量 (B, C, H, W) 进行四元数旋转
    :param tensor: 输入张量，形状为 (B, C, H, W)，C 必须是 3（表示三维坐标）
    :param quaternion: 旋转四元数，形状为 (4,) 或 (B, 4) 
    :return: 旋转后的张量，形状不变
    """
    B, C, H, W = tensor.shape
    assert C == 3, "输入张量的通道维度 (C) 必须为 3，表示三维坐标"
    
    # 将张量转为向量形式 (B, 3, H*W)
    tensor = tensor.view(B, C, -1)
    
    # 将四元数拆分为 w, x, y, z
    if quaternion.dim() == 1:  # 如果四元数为单一旋转
        quaternion = quaternion.unsqueeze(0).repeat(B, 1)  # 扩展到 batch 维度
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # 计算四元数共轭
    q_conj = torch.stack([w, -x, -y, -z], dim=1)  # (B, 4)
    
    # 扩展坐标向量为四维： (0, x, y, z)
    v = torch.cat([torch.zeros(B, 1, tensor.shape[2], device=tensor.device), tensor], dim=1)  # (B, 4, H*W)
    
    # 四元数旋转： q * v * q_conj
    q = quaternion.unsqueeze(2)  # (B, 4, 1)
    q_conj = q_conj.unsqueeze(2)  # (B, 4, 1)
    
    # Hamilton product 实现 (q * v)
    t = quaternion_multiply(q, v)  # (B, 4, H*W)
    v_rotated = quaternion_multiply(t, q_conj)  # (B, 4, H*W)
    
    # 取前三维，去掉扩展的 0 维度
    tensor_rotated = v_rotated[:, 1:, :]  # (B, 3, H*W)
    
    # 恢复原形状
    tensor_rotated = tensor_rotated.view(B, C, H, W)
    return tensor_rotated

def quaternion_multiply(q1, q2):

    """
    实现两个四元数的哈密顿积
    :param q1: 第一个四元数，形状为 (B, 4, N)
    :param q2: 第二个四元数，形状为 (B, 4, N)
    :return: 哈密顿积结果，形状为 (B, 4, N)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=1)

class LTQ(nn.Module):
    def __init__(self, num_bits):
        super(LTQ, self).__init__()
        init_range = 2
        self.num_bits = num_bits
        self.n_val = 2 ** num_bits - 1
        self.interval = init_range / self.n_val
        # self.interval = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # self.scale2 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)


    def taylor_expansion_2n_minus_1(self, n):
        """
        计算 2^n - 1 的泰勒展开前若干项。
        
        参数:
        - n: 指数 n 的值。
        - terms: 泰勒展开的项数（默认为 8 项）。
        
        返回:
        - 每一项的值组成的列表。
        """

        terms = 2**(n-1)
        ln2 = math.log(2)  # 计算 ln(2)
        expansion_terms = []  # 用于存储展开的每一项
        
        for k in range(1,1+terms):  # 从第 1 项开始展开
            term = (ln2 ** k) * (n ** k) / math.factorial(k)
            expansion_terms.append(term)
        
        return expansion_terms

    def forward(self, x):
        
        # import pdb; pdb.set_trace()
        if self.init_state == 0:
            self.interval = ((x.max() - x.min()) / self.n_val).detach()
            list_tylor = self.taylor_expansion_2n_minus_1(self.num_bits)
            tyler = []
            
            for i in range(int((self.n_val-1)/2)):
                tyler.append(self.interval * list_tylor[int((self.n_val-1)/2)-i])

            self.a.data = torch.Tensor(tyler).to(x.device)
            self.b.data = torch.Tensor(tyler).to(x.device)

            self.init_state.fill_(1)
        # else:
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.start + 0.0
        step_left = self.start + 0.0
        # import pdb; pdb.set_trace()
        a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        b_pos = torch.where(self.b > self.eps, self.b, self.eps)

        for i in range(int((self.n_val-1)/2-1)):
            step_right += a_pos[i]
            step_left += b_pos[i]
            if i == 0:
                thre_forward_p = self.start + a_pos[0] / 2
                thre_forward_n = self.start - b_pos[0] / 2
                thre_backward_p = self.start + 0.0
                thre_backward_n = self.start + 0.0
                x_forward = torch.where(x > thre_forward_p, step_right, x)
                x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > thre_backward_p, x - thre_backward_p + step_right - a_pos[i], x)
                x_backward = torch.where(x < thre_backward_n, x - thre_backward_n - step_left + b_pos[i], x_backward)
            else:
                thre_forward_p += a_pos[i-1] / 2 +  a_pos[i] / 2
                thre_forward_n -= (b_pos[i-1] / 2 +  b_pos[i] / 2)
                thre_backward_p += a_pos[i-1]
                thre_backward_n -= b_pos[i-1]
                x_forward = torch.where(x > thre_forward_p, step_right, x_forward)
                x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > thre_backward_p, x - thre_backward_p + step_right - a_pos[i], x_backward)
                x_backward = torch.where(x < thre_backward_n,  x - thre_backward_n - step_left + b_pos[i], x_backward)
 
        thre_backward_p += a_pos[i]
        thre_backward_n -= b_pos[i]
        x_backward = torch.where(x > thre_backward_p, x.max(), x_backward)
        x_backward = torch.where(x < thre_backward_n, x.min(), x_backward)
        
        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out / self.scale1
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(x, name="ltq_x_5k")
        # plot_tensor_histogram(out, name="ltq_out_5k")
        # plot_tensor_histogram(x_backward, name="ltq_out_back")
        # import pdb; pdb.set_trace()

        return out


class LTQ_yj(nn.Module):
    
    def __init__(self,in_features, nbits=4):
        super(LTQ_yj, self).__init__()
        self.nbits = nbits
        # self.alpha = Parameter(torch.Tensor(1))
        self.alpha = Parameter(torch.Tensor(1, in_features))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1))), requires_grad=True)
        # self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)


    def taylor_expansion_2n_minus_1(self, n):
        """
        计算 2^n - 1 的泰勒展开前若干项。
        
        参数:
        - n: 指数 n 的值。
        - terms: 泰勒展开的项数（默认为 8 项）。
        
        返回:
        - 每一项的值组成的列表。
        """

        terms = 2**(n-1)
        ln2 = math.log(2)  # 计算 ln(2)
        expansion_terms = []  # 用于存储展开的每一项
        
        for k in range(1,1+terms):  # 从第 1 项开始展开
            term = (ln2 ** k) * (n ** k) / math.factorial(k)
            expansion_terms.append(term/2)
        negative_list = [-x for x in reversed(expansion_terms[0:])]
        expansion_terms = expansion_terms + negative_list
        return expansion_terms

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3
    
    def phi_function_p(self, x, l, delta, k, i):
    
        m_i = l + (i + 0.5) * delta
        s = delta/2

        # m_i = torch.tensor(m_i)
        # s = torch.tensor(s)
        # k = torch.tensor(k)

        phi = s * torch.tanh(k * (x - m_i)) + m_i

        return phi 

    def phi_function_n(self, x, l, delta, k, i):

        m_i = l + (i + 0.5) * delta
        s = delta/2

        # m_i = torch.tensor(m_i)
        # s = torch.tensor(s)
        # k = torch.tensor(k)

        phi = s * torch.tanh(k * (x + m_i)) - m_i

        return phi 
    
    def round_ltq(self, x):
        a_pos = self.a
        start = a_pos[0]
        x_forward = x
        x_backward = x
        # step_right = a_pos[0] 
        # step_left = self.start + 0.0
        # self.a.data = self.a.abs()
        # self.b.data = self.b.abs()
        # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        p_1_p = self.start
        p_2_p = a_pos[1] 
        for i in range(int(self.n_val-1)):
            # import pdb; pdb.set_trace() 
            k_p = 8 - (a_pos[i]-a_pos[i-1])

            if i == 0:
                thre_forward_p = self.start
                # thre_forward_n = self.start - b_pos[0]
                x_forward = torch.where(x > thre_forward_p, a_pos[i], a_pos[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p            
                delta_p = p_2_p - p_1_p 
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                
                y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_backward = torch.where(x > self.start, 0.4*x+0.6*y_step_p,  x)
                # x_backward = torch.where(x < self.start, 0.4*x+0.6*y_step_n, x_backward)
            else:
                thre_forward_p = (a_pos[i-1] + a_pos[i])/2
                p_1_p = a_pos[i-1]
                p_2_p = a_pos[i]
                # p_1_n += b_pos[i-1]
                # p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                delta_p = p_2_p - p_1_p 
                # l_n = p_1_n            
                # delta_n = p_2_n - p_1_n 
                y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
                # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, a_pos[i], x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > p_1_p, 0.4*x+0.6*y_step_p, x_backward)
                # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = a_pos[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x.max(), x_backward)
        # x_backward = torch.where(x < -p_1_n, x.min(), x_backward)
        x_backward = torch.where(x > p_1_p, x, x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_round_4")
        # plot_tensor_histogram(x_forward, name="ltq_out_for_round_4")
        # plot_tensor_histogram(x, name="ltq_x_for_round_4")
        # import pdb; pdb.set_trace()

        return out

    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_in_3")
        # print(self.a)
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                raise RuntimeError
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.alpha.data.copy_((x.max() - x.min()) / self.n_val)
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            # list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            # tyler = []
            # for i in range(int((self.n_val-1)/2)):
            #     tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            tyler = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
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
        # plot_tensor_histogram(x, name="ltq_in_nobeta_2")
        # plot_tensor_histogram(x / alpha, name="ltq_in_before_round_2")
        out = self.round_ltq(x / alpha).clamp(self.a[0],self.a[-1]) * alpha + self.beta
        # plot_tensor_histogram(self.round_ltq(x / alpha), name="ltq_in_after_round_2")
        # plot_tensor_histogram(self.round_ltq(x / alpha).clamp(self.a[0],self.a[-1]), name="ltq_in_after_round_clamp_2")
        # plot_tensor_histogram(out-self.beta, name="ltq_out_nobeta_2")
        # import pdb; pdb.set_trace()
        # # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_2")
        

        return out

from torch.autograd import Function
class LTQ_auto(Function):
    def __init__(self, nbits=4):
        super(LTQ_auto, self).__init__()
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.zeros(1))
        # self.alpha = Parameter(torch.Tensor([0]), requires_grad=True)
        # self.beta = Parameter(torch.Tensor([0]), requires_grad=True)
        self.register_buffer('signed', torch.zeros(1))
        init_range = 2
        self.n_val = 2 ** nbits - 1 # Qp-Qn
        # self.interval = Parameter(torch.Tensor([0]), requires_grad=True)
        self.start = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor([init_range / self.n_val]* int((self.n_val-1)/2)), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        # self.register_buffer('self.interval', torch.zeros(1))
        
        # self.shifting = Parameter(torch.zeros(1), requires_grad=True)
        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-4]), requires_grad=False)


    def taylor_expansion_2n_minus_1(self, n):
        """
        计算 2^n - 1 的泰勒展开前若干项。
        
        参数:
        - n: 指数 n 的值。
        - terms: 泰勒展开的项数（默认为 8 项）。
        
        返回:
        - 每一项的值组成的列表。
        """

        terms = 2**(n-1)
        ln2 = math.log(2)  # 计算 ln(2)
        expansion_terms = []  # 用于存储展开的每一项
        
        for k in range(1,1+terms):  # 从第 1 项开始展开
            term = (ln2 ** k) * (n ** k) / math.factorial(k)
            expansion_terms.append(term/2)
        
        return expansion_terms

    def round_pass(x):
        input = x
        x_round = input.round()
        x = input - input.floor().detach()
        x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
        out3 = x + input.floor().detach()
        return x_round.detach() - out3.detach() + out3
    
    def phi_function_p(self, x, l, delta, k, i):
    
        m_i = l + (i + 0.5) * delta
        s = delta/2

        # m_i = torch.tensor(m_i)
        # s = torch.tensor(s)
        # k = torch.tensor(k)

        phi = s * torch.tanh(k * (x - m_i)) + m_i

        return phi 

    def phi_function_n(self, x, l, delta, k, i):

        m_i = l + (i + 0.5) * delta
        s = delta/2

        # m_i = torch.tensor(m_i)
        # s = torch.tensor(s)
        # k = torch.tensor(k)

        phi = s * torch.tanh(k * (x + m_i)) - m_i

        return phi 
    
    def round_ltq(self, x):

        x_forward = x
        x_backward = x
        step_right = self.start + 0.0
        step_left = self.start + 0.0
        a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        b_pos = torch.where(self.b > self.eps, self.b, self.eps)
        p_1_p = self.start + 0.0
        p_2_p = a_pos[0]
        p_1_n = self.start + 0.0
        p_2_n = b_pos[0]

        for i in range(int((self.n_val-1)/2)):
            # import pdb; pdb.set_trace()  
            step_right += a_pos[i]
            step_left += b_pos[i]
            k_p = sum(a_pos) - a_pos[i]
            k_n = sum(b_pos) - b_pos[i]

            if i == 0:
                thre_forward_p = self.start + a_pos[0]
                thre_forward_n = self.start - b_pos[0]
                x_forward = torch.where(x > thre_forward_p, step_right, x)
                x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                l_p = p_1_p            
                delta_p = p_2_p - p_1_p 
                l_n = p_1_n            
                delta_n = p_2_n - p_1_n 
                y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
                y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_backward = torch.where(x > self.start, 0.4*x+0.6*y_step_p, x)
                x_backward = torch.where(x < self.start, 0.4*x+0.6*y_step_n, x_backward)
            else:
                thre_forward_p += a_pos[i]
                thre_forward_n -= b_pos[i]
                p_1_p += a_pos[i-1]
                p_2_p = p_1_p + a_pos[i]
                p_1_n += b_pos[i-1]
                p_2_n = p_1_n + b_pos[i]
                l_p = p_1_p            
                delta_p = p_2_p - p_1_p 
                l_n = p_1_n            
                delta_n = p_2_n - p_1_n 
                y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
                y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
                x_forward = torch.where(x > thre_forward_p, step_right, x_forward)
                x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
                x_backward = torch.where(x > p_1_p, 0.4*x+0.6*y_step_p, x_backward)
                x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p += a_pos[i]
        p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x.max(), x_backward)
        # x_backward = torch.where(x < -p_1_n, x.min(), x_backward)
        x_backward = torch.where(x > p_1_p, x, x_backward)
        x_backward = torch.where(x < -p_1_n, x, x_backward)
        out = x_forward.detach() + x_backward - x_backward.detach()
        # plot_tensor_histogram(x_backward, name="ltq_out_back_round_2")
        # plot_tensor_histogram(x_forward, name="ltq_out_for_round_2")
        # plot_tensor_histogram(x, name="ltq_x_for_round_2")
        # import pdb; pdb.set_trace()

        return out

    @staticmethod
    def forward(self, x):
        # plot_tensor_histogram(x, name="ltq_x_after_shift")
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
            self.beta.data.copy_(x.mean())
            # self.interval.data = ((x.max() - x.min()) / self.n_val).detach().view(1)
            # self.shifting.data = x.mean()
            list_tylor = self.taylor_expansion_2n_minus_1(self.nbits)
            tyler = []
            for i in range(int((self.n_val-1)/2)):
                tyler.append(list_tylor[int((self.n_val-1)/2)-i])
            self.a.data = torch.Tensor(tyler).to(x.device)
            self.b.data = torch.Tensor(tyler).to(x.device)
            self.init_state.fill_(1)
        # else:
        #     import pdb; pdb.set_trace()
        
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        x = x - self.beta
        g = 1.0 / math.sqrt(x.numel() * Qp)

        alpha = grad_scale(self.alpha, g)
        
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # x = x / self.interval
        # import pdb; pdb.set_trace()
        out = self.round_ltq((x / alpha)) * alpha + self.beta
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(out, name="ltq_out_all_1")
        # # plot_tensor_histogram(x_backward, name="ltq_out_back_yj_1")
        

        return out
        