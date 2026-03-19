import torch
import torch.nn as nn

# import quant_linear_cuda

from .hadamard_utils import get_had_fn, get_qhad_fn
from .quantUtils import quantize_tensor_per_tensor_absmax
import collections
from itertools import repeat
import torch.nn.functional as Fu
import math
from analysis.plt import plot_tensor_histogram

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
        
        # import pdb; pdb.set_trace()
        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

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

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, k_bits=32):
        super(QuantLinear, self).__init__(in_features, out_features, bias=True)
        # import pdb; pdb.set_trace()
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).cuda()
        # self.weight = torch.nn.Parameter(self.weight.cuda())
        self.k_bits = k_bits
        self.transform_fn, self.N, self.had_scale = get_had_fn(in_features)
        self.quant_weight = quant_weight(self.k_bits)
        # import pdb; pdb.set_trace()
        # self.had_weight = self.transform_fn(
        #     self.weight, self.had_scale)
        # import pdb; pdb.set_trace()
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)
        # if bias:    
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()
        # import pdb; pdb.set_trace()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        
        # return Fu.linear(x, self.quant_weight(self.had_weight), self.bias)
        return Fu.linear(x, self.quant_weight(self.weight), self.bias)

class QHadLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, k_bits=32):
        super(QHadLinear, self).__init__(in_features, out_features, bias=True)
        # import pdb; pdb.set_trace()
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).cuda()
        self.weight = torch.nn.Parameter(self.weight.cuda())
        self.k_bits = k_bits
        self.transform_fn, self.N, self.had_scale = get_had_fn(in_features)
        self.quant_weight = quant_weight(self.k_bits)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)
        # if bias:    
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        self.had_weight = self.transform_fn(self.weight, self.had_scale)
        return Fu.linear(x, self.quant_weight(self.had_weight), self.bias)
        # return Fu.linear(x, self.quant_weight(self.weight), self.bias)
    
class QuantConv2d(nn.Conv2d):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=True, k_bits=32):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.weight = torch.nn.Parameter(self.weight.cuda())
        self.k_bits = k_bits
        self.quant_weight = quant_weight(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        # import pdb; pdb.set_trace()
        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

class QAct(nn.Module):
    def __init__(
        self,
        scale
    ):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return (x / self.scale).clamp(min=-128, max=127).to(torch.int8) # quant
    
    def __repr__(self):
        return f"QAct()"

class token_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits, ema_epoch=1, decay=0.9997):
        super(token_act, self).__init__()
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

        if len(x.shape) == 4:
            max_val = torch.mean(torch.max(torch.max(torch.max(abs(x),dim=1)[0],dim=1)[0],dim=1)[0])
        elif len(x.shape) == 3:
            max_val = torch.mean(torch.max(torch.max(abs(x),dim=1)[0],dim=1)[0])
        else:
            raise ValueError('############## wrong dimension of input for activation quantization  ##############')

        if self.epoch == 1:
            self.max_val = max_val
        else:
            self.max_val = (1.0-self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):
        if self.epoch > self.ema_epoch or not self.training:
            act = torch.max(torch.min(x, self.alpha), -self.alpha)

        elif self.epoch <= self.ema_epoch and self.training:
            # import pdb; pdb.set_trace()
            act = x
            self._ema(x)
            self.alpha.data = self.max_val.unsqueeze(0)
        
        act = act * self.qmax / self.alpha
        q_act = self.round(act)
        q_act = q_act * self.alpha / self.qmax
    
        return q_act

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