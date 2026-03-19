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


class Ours_SSM_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(Ours_SSM_Quant, self).__init__()

        self.nbits = nbits
        # self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))
        # self.alpha_predictor = nn.Linear(2,1)
        # self.alpha = Parameter(torch.Tensor(1))
        # self.weight1 = Parameter(torch.Tensor([1/(2**(self.nbits-1)-1)]))
        # self.weight2 = Parameter(torch.Tensor([4/(2**(self.nbits-1)-1)]))
        # self.weight3 = Parameter(torch.Tensor([0]))
        # self.weight4 = Parameter(torch.Tensor([0]))
        self.weight1 = Parameter(torch.Tensor([0]))
        self.weight2 = Parameter(torch.Tensor([0]))
        self.weight3 = Parameter(torch.Tensor([-1/(2**self.nbits)]))
        self.weight4 = Parameter(torch.Tensor([1/2**self.nbits]))
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
            # self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / Qp)
            # self.beta.data.copy_((x.max() + x.min()) / 2)
            self.init_state.fill_(1)
        x = x - self.beta    
        self.alpha = (self.weight1*x.mean()+self.weight2*x.std()+self.weight3*x.min()+self.weight4*x.max()).abs()
        # alpha_x = torch.cat((x.mean().unsqueeze(0), x.var().unsqueeze(0)), dim=0)
        # self.alpha = self.alpha_predictor(alpha_x)
        if self.alpha < 1e-6:
            self.alpha+=1e-6
        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha + self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # import pdb; pdb.set_trace()
        return x

class Ours_SSM_beta_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(Ours_SSM_beta_Quant, self).__init__()

        self.nbits = nbits
        # self.alpha = Parameter(torch.Tensor(1))
        # self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))
        # self.alpha_predictor = nn.Linear(2,1)
        # self.alpha = Parameter(torch.Tensor(1))
        self.weight11 = Parameter(torch.Tensor([1/(2**(self.nbits-1)-1)]))
        self.weight12 = Parameter(torch.Tensor([4/(2**(self.nbits-1)-1)]))
        self.weight13 = Parameter(torch.Tensor([0]))
        self.weight14 = Parameter(torch.Tensor([0]))
        # self.weight11 = Parameter(torch.Tensor([0]))
        # self.weight12 = Parameter(torch.Tensor([0]))
        # self.weight13 = Parameter(torch.Tensor([-1/(2**self.nbits)]))
        # self.weight14 = Parameter(torch.Tensor([1/2**self.nbits]))
        self.weight21 = Parameter(torch.Tensor([0]))
        self.weight22 = Parameter(torch.Tensor([0]))
        self.weight23 = Parameter(torch.Tensor([-1/2]))
        self.weight24 = Parameter(torch.Tensor([-1/2]))

    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def forward(self, x):
        # import pdb; pdb.set_trace() 
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
            # self.beta.data.copy_((x.max() + x.min()) / 2)
            self.init_state.fill_(1)
                
        x_mean = x.mean()
        x_std = x.std()
        x_min = x.min()
        x_max = x.max()  
        self.alpha = self.weight11*x_mean+self.weight12*x_std+self.weight13*x_min+self.weight14*x_max
        self.beta = self.weight21*x_mean+self.weight22*x_std+self.weight23*x_min+self.weight24*x_max
        # alpha_x = torch.cat((x.mean().unsqueeze(0), x.var().unsqueeze(0)), dim=0)
        # self.alpha = self.alpha_predictor(alpha_x)
        if self.alpha < 1e-6:
            self.alpha+=1e-6
        x = x - self.beta         
        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha + self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        return x

class DDA_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(DDA_Quant, self).__init__()

        self.nbits = nbits
        # self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.err_factor = Parameter(torch.zeros(1))
        # self.alpha_predictor = nn.Linear(2,1)
        # self.alpha = Parameter(torch.Tensor(1))
        self.weight1 = Parameter(torch.Tensor([1/(2**(self.nbits-1)-1)]))
        self.weight2 = Parameter(torch.Tensor([4/(2**(self.nbits-1)-1)]))
        self.weight3 = Parameter(torch.Tensor([0]))
        self.weight4 = Parameter(torch.Tensor([0]))
        # self.weight1 = Parameter(torch.Tensor([0]))
        # self.weight2 = Parameter(torch.Tensor([0]))
        # self.weight3 = Parameter(torch.Tensor([-1/4]))
        # self.weight4 = Parameter(torch.Tensor([1/4]))
    def extra_repr(self):
  
        return '{}'.format(self.nbits)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
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
            self.beta.data.copy_(-(x.max() + x.min()) / 2)
            self.init_state.fill_(1)
        x = x + self.beta    
        self.alpha = self.weight1*x.mean()+self.weight2*x.std()+self.weight3*x.min()+self.weight4*x.max()
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
        x = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha - self.beta
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        # import pdb; pdb.set_trace()
        return x


class Ours_Weight_Quant(nn.Module):
    def __init__(self, nbits=4):
        super(Ours_Weight_Quant, self).__init__()
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
        start = a_pos[0]
        x_forward = x
        x_backward = x
        for i in range(len(a_pos)):
            if i == 0:
                thre_forward_p = start
                thre_backward_p = start
                x_forward = torch.where(x > thre_forward_p, additive_pot[i], additive_pot[i])
                # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
                # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
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

        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x, x_backward)
        x_backward = torch.where(x > a_pos[i], 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
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
        # self.conv_weight_quant = Ours_Weight_Quant(nbits=nbits_w)

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
        # self.linear_act_quant = Ours_SSM_Quant(nbits=nbits_a)
        # self.linear_act_quant = DDA_Quant(nbits=nbits_a)
        # self.linear_act_quant = LTQ_yj(nbits=nbits_a)
        
        self.linear_weight_quant = Weight_Quant(nbits=nbits_w)

    def forward(self, x):

        x_q = self.linear_act_quant(x)
        w_q = self.linear_weight_quant(self.weight)
        
        return F.linear(x_q, w_q, self.bias)

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
        out = round_pass_vanilla((x / alpha).clamp(Qn, Qp)) * alpha
        # x = round_pass((x / alpha).clamp(Qn, Qp), x, self.err_factor) * alpha
        return out

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


        
        