import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from quantize.quantizer import UniformAffineQuantizer
import time
import math
from enum import Enum
from basicsr.archs.QuantSR_operator import QuantSR_act

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
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,rotate=rotate)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        self.quant_weight = QuantSR_act(weight_quant_params['n_bits'])
        self.quant_act = QuantSR_act(act_quant_params['n_bits'])

    def forward(self, input: torch.Tensor):

        
        input_vanilla =input.clone()
        weight_vanilla = self.weight.clone()
        # mul_before = input_vanilla @ weight_vanilla.transpose(-1,-2)
        if len(input.shape) == 3:
            b, n, c_in = input.shape
            input = input.reshape(-1, c_in).contiguous()
            lenghth_input = 3
        elif len(input.shape) == 4:
            b, h, w, c_in = input.shape
            input = input.reshape(-1, c_in).contiguous()
            lenghth_input = 4

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
        
        if lenghth_input == 3:
            input = input.reshape(b, n, c_in).contiguous()
        elif lenghth_input == 4:
            input = input.reshape(b, h, w, c_in).contiguous()

        input_vanilla.data = input
        weight_vanilla.data = weight
        # mul_after = input_vanilla @ weight_vanilla.transpose(-1,-2)
        
        # mse = torch.mean((mul_before - mul_after) ** 2)
        out = self.fwd_func(self.quant_act(input_vanilla), self.quant_weight(weight_vanilla), bias, **self.fwd_kwargs)
        # out = self.fwd_func(input_vanilla, weight_vanilla, bias, **self.fwd_kwargs)

        # if self.init_duquant_params == torch.tensor(1):
        #     import pdb;pdb.set_trace()

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_duquant_params(self, proj):
        assert proj.init_duquant_params
        self.init_duquant_params = torch.tensor(1)
        self.weight_quantizer.copy_duquant_params(proj.weight_quantizer)
        self.act_quantizer.copy_duquant_params(proj.act_quantizer)

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, _Conv2dQ):
        default.update({
            'mode': Qmodes.layer_wise})
    # elif isinstance(layer_type, _LinearQ):
    #     pass
    # elif isinstance(layer_type, _ActQ):
    #     pass
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q

class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2

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

class DUConv2d(nn.Module):
    def __init__(
        self,
        org_module: nn.Conv2d,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        rotate=True,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv2d
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        # self.in_features = org_module.in_features
        # self.out_features = org_module.out_features
        self.padding = org_module.padding
        self.groups = org_module.groups
        self.stride = org_module.stride
        self.dilation = org_module.dilation
        # de-activate the quantized forward default
        self.use_weight_quant = True
        self.use_act_quant = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,rotate=rotate)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        self.quant_weight = QuantSR_act(weight_quant_params['n_bits'])
        self.quant_act = QuantSR_act(act_quant_params['n_bits'])

    def forward(self, input):

        input_vanilla =input.clone()
        weight_vanilla = self.weight.clone()
        # mul_before = input_vanilla @ weight_vanilla.transpose(-1,-2)
       
        b, c_in, h, w= input.shape
        input = input.reshape(-1, c_in).contiguous()

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
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups, **self.fwd_kwargs)
        
        return out
