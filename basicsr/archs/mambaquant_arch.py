# Code Implementation of the MambaIR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
import fast_hadamard_transform
# from basicsr.utils.plt import plot_tensor_histogram
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from analysis.plt import plot_tensor_3d
# from .quamba_operator import Hadamard
NEG_INF = -1000000

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert (is_pow2(n // 172))
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert (is_pow2(n // 156))
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate 
        assert (is_pow2(n // 140))
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate 
        assert (is_pow2(n // 108))
        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        # assert (is_pow2(n // 60))
        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert (is_pow2(n // 52))
        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert (is_pow2(n // 36))
        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        assert (is_pow2(n // 28))
        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        assert (is_pow2(n // 40))
        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert (is_pow2(n // 20))
        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert (is_pow2(n // 12))
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert (is_pow2(n))
        K = 1

    return hadK, K

def matmul_hadU_cuda(X, hadK, K, transpose=False):
    n = X.shape[-1]
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0/torch.tensor(n).sqrt()) 
    if transpose:
        hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)
    input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0/torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)

class HadamardTransform(nn.Module):
    def __init__(self, do_rotate=False):
        super().__init__()
        self.do_rotate = do_rotate
    def forward(self, x, transpose=False):
       
        if self.do_rotate:
            dtype = x.dtype
            n = x.shape[-1]
            had_K, K = get_hadK(n)
            x = matmul_hadU_cuda(x.contiguous(), had_K, K, transpose=transpose).to(dtype)
        return x
    
    def configure(self, do_rotate):
        self.do_rotate = do_rotate

class SmoothModule(nn.Module):
    def __init__(self, weight_to_smooth, tensor_name=None):
        super(SmoothModule, self).__init__()
        self.tensor_name = tensor_name
        self.weight_to_smooth=weight_to_smooth
        self.register_buffer("scales", None)
        self.activated = False
    @torch.no_grad()
    def forward(self, x, reverse=False):
        assert not torch.isnan(x).any(), "Input tensor x contains NaNs."
        if not self.activated:
            return x
        else:
            if reverse:
                return x.mul(self.scales)
            else:
                # import pdb;pdb.set_trace()
                return x.div(self.scales)
        
    def configure(self, scales):
        self.scales = scales
        assert not torch.isnan(self.scales).any(), "Scales contains NaNs."
        self.activated = True
  
class QAct(nn.Module):
    def __init__(
        self,
        tensor_name
    ):
        super().__init__()
        self.act_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
        self.is_sym = None
        self.is_static_quant = None
        self.tensor_name = tensor_name
        self.register_buffer("a_scales", None)
        
    @torch.no_grad()
    def forward(self, x):
        if not self.is_quant_mode:  
            # For calibration purpose
            return x
        else:
            # For quantized mode
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            # then, re-quant them to the target scale    
            # import pdb; pdb.set_trace()         
            return self.act_quantizer(x)
    
    def configure(self, 
            n_bits,
            sym,
            o_scales = None,
            o_base = None,
            clip_ratio = 1.0,
            static_quant=True,
            quantization_type = "per_tensor"
        ):
        # import pdb; pdb.set_trace()
        self.is_configured = True
        #NOTE(brian1009): Do no quantization when bit width is larger than 16
        if n_bits >= 16:
            self.act_quantizer = lambda x: x
            return
            
        self.is_static_quant = static_quant
        self.is_sys = sym
        if self.is_static_quant:
            assert (o_scales is not None or o_base is None), "Static quantization requires scales/base to be provided"
            self.act_quantizer = partial(
                uniform_affine_fake_quantization,
                n_bits=n_bits,
                sym=sym,
                scales=o_scales,
                base=o_base
            )
        else:
            if quantization_type == "per_tensor":
                self.act_quantizer = partial(
                    bached_dynamic_per_tensor_absmax_quantization,
                    n_bits=n_bits,
                    sym=sym,
                    clip_ratio=clip_ratio
                )
            elif quantization_type == "per_token":
                self.act_quantizer = partial(
                    dynamic_per_token_absmax_quantization,
                    n_bits=n_bits,
                    sym=sym,
                    clip_ratio=clip_ratio,
                )
            else:
                raise ValueError(f"Invalid activation quantization type: {quantization_type}")
        

    def __repr__(self):
        return f"QAct()"

class QConv2D(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Conv2d,
    ):
        super().__init__()
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None


        # Copy convolution-specific parameters
        self.in_channels = originalLayer.in_channels
        self.out_channels = originalLayer.out_channels
        self.kernel_size = originalLayer.kernel_size
        self.stride = originalLayer.stride
        self.padding = originalLayer.padding
        self.dilation = originalLayer.dilation
        self.groups = originalLayer.groups
        
        # Quantization Related 
        self.weight_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
    
    @property
    def fake_quant_weight(self):
        if not self.is_quant_mode:
            return self.weight
        w_fake_quant = self.weight_quantizer(w=self.weight.clone())
        return w_fake_quant

    @torch.no_grad()
    def forward(self, x, i_scales = torch.tensor(1.0)):
        if not self.is_quant_mode:
            #calibration mode
            return F.conv2d(
                x, 
                self.weight, 
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        else: 
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            #quantized mode
            w_fq = self.fake_quant_weight
            out = F.conv2d(
                            x,
                            w_fq,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups
                        )
            if self.bias is not None:
                out = out + self.bias.reshape(1, -1, 1, 1)
            return out

    def to(self, *args, **kwargs):
        super(QConv2D, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    def configure(self,
            n_bits,
            clip_ratio=1.0,
        ):
        self.weight_quantizer = partial(
            dynamic_per_tensor_absmax_quantization,
            n_bits = n_bits,
            clip_ratio = clip_ratio,
            sym=True
        )
        self.is_configured = True
            
    def __repr__(self):
        return f"Conv2d({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"

class QLinear(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
    ):
        super().__init__()

        if  originalLayer.weight.dtype == torch.int8:
            raise NotImplementedError("Are you quantizing as Bits&Bytes in addition to another quantization setting? Dtype = int8 is not implemented for QLinearLayer. Try adding model layer to --skip_modules (ex: self_attn, mamba, moe) to ensure Bits&Bytes is not quantizing first.")

            
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
    
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
        
        self.weight_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
        
    @torch.no_grad()
    def forward(self, x):
        if not self.is_quant_mode:
            # Calibration mode
            return torch.functional.F.linear(x, self.weight, self.bias)
        else: 
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            # Using fake quantized weights for inference
            # import pdb; pdb.set_trace()
            w_fq = self.fake_quant_weight
            out = torch.functional.F.linear(x, w_fq)
            if self.bias is not None:
                out = out + self.bias
            return out
            
    def to(self, *args, **kwargs):
        super(QLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    
    def configure(self, 
            n_bits,
            clip_ratio=1.0,
        ):
        self.weight_quantizer = partial(
            dynamic_per_tensor_absmax_quantization,
            n_bits = n_bits,
            clip_ratio = clip_ratio,
            sym=True
        )
        self.is_configured = True

    @property
    def fake_quant_weight(self):
        w_fake_quant = self.weight_quantizer(w=self.weight.clone())
        return w_fake_quant
        
    def __repr__(self):
        return f"QLinear(in_features={self.in_features}, out_features={self.out_features})"

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

@torch.no_grad()
def dynamic_per_tensor_absmax_quantization(
        w: torch.tensor, n_bits, sym, clip_ratio=1.0,
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    #Calculating the scale dynamically
    if sym:
        w_max = w.abs().amax().clamp(min=1e-5)
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(w)
    else:
        w_max = w.amax()
        w_min = w.amin()
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)  
        
    # fake quantization
    return uniform_affine_fake_quantization(w, n_bits, sym, scales, base)

@torch.no_grad()
def uniform_affine_fake_quantization(
        w: torch.tensor, n_bits, sym,
        scales = None, base = None
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=sym)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales

    return w

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.is_light_sr = is_light_sr
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2 

            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):

        return self.cab(x)

class QCAB(nn.Module):
    def __init__(self, originalLayer: CAB):
        super().__init__()

        self.is_light_sr = originalLayer.is_light_sr
        if originalLayer.is_light_sr:
            self.cab0_quant = QAct(tensor_name="cab0_quant")
            self.cab0 = QConv2D(originalLayer = originalLayer.cab[0])
            self.cab1_quant = QAct(tensor_name="cab1_quant")
            self.cab1 = QConv2D(originalLayer = originalLayer.cab[1])
            self.cab2 = originalLayer.cab[2]
            self.cab3_quant = QAct(tensor_name="cab3_quant")
            self.cab3 = QConv2D(originalLayer = originalLayer.cab[3])
            self.cab4_quant = QAct(tensor_name="cab4_quant")
            self.cab4 = QConv2D(originalLayer = originalLayer.cab[4])
            self.cab5 = originalLayer.cab[5]
        else:
            self.cab0_quant = QAct(tensor_name="cab0_quant")
            self.cab0 = QConv2D(originalLayer = originalLayer.cab[0])
            self.cab1 = originalLayer.cab[1]
            self.cab2_quant = QAct(tensor_name="cab2_quant")
            self.cab2 = QConv2D(originalLayer = originalLayer.cab[2])
            self.cab3 = originalLayer.cab[3]
    
    def forward(self, x):

        if self.is_light_sr:
            x = self.cab0(self.cab0_quant(x))
            x = self.cab1(self.cab1_quant(x))
            x = self.cab2(x)
            x = self.cab3(self.cab3_quant(x))
            x = self.cab4(self.cab4_quant(x))
            x = self.cab5(x)
        else:
            x = self.cab0(self.cab0_quant(x))
            x = self.cab1(x)
            x = self.cab2(self.cab2_quant(x))
            x = self.cab3(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_scale = dt_scale
        self.dt_init_zz = dt_init
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor
        self.dropout_rate = dropout

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        # self.had = Hadamard(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.x_proj_a_quant = QAct(tensor_name="x_proj_a_quant")
        self.x_proj_w_quant = QAct(tensor_name="x_proj_w_quant")
        self.dt_proj_a_quant = QAct(tensor_name="dt_proj_a_quant")
        self.dt_proj_w_quant = QAct(tensor_name="dt_proj_w_quant")

        self.xs_quant = QAct(tensor_name="xs_quant")
        self.dts_quant = QAct(tensor_name="dts_quant")
        self.As_quant = QAct(tensor_name="As_quant")
        self.Bs_quant = QAct(tensor_name="Bs_quant")
        self.Cs_quant = QAct(tensor_name="Cs_quant")
        self.Ds_quant = QAct(tensor_name="Ds_quant")
        self.dt_bias_quant = QAct(tensor_name="dt_bias_quant")

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", 
            self.x_proj_a_quant(xs.view(B, K, -1, L)), 
            self.x_proj_w_quant(self.x_proj_weight))
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", 
            self.dt_proj_a_quant(dts.view(B, K, -1, L)), 
            self.dt_proj_w_quant(self.dt_projs_weight))
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            self.xs_quant(xs), self.dts_quant(dts),
            self.As_quant(As), self.Bs_quant(Bs), self.Cs_quant(Cs), 
            self.Ds_quant(Ds), z=None,
            delta_bias=self.dt_bias_quant(dt_projs_bias),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # import pdb;pdb.set_trace()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class QSS2D(nn.Module):
    def __init__(self, originalLayer: SS2D):
        super().__init__()

        self.d_model = originalLayer.d_model
        self.d_state = originalLayer.d_state
        # self.expand = originalLayer.expand
        # self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = originalLayer.dt_rank

        # self.in_proj_smooth = SmoothModule(weight_to_smooth="in_proj", tensor_name="in_proj_smooth")
        self.in_proj_quant = QAct(tensor_name="in_proj_quant")
        self.in_proj = QLinear(originalLayer = originalLayer.in_proj)
        # self.in_proj = originalLayer.in_proj

        self.conv2d_quant = QAct(tensor_name="conv2d_quant")
        self.conv2d = QConv2D(originalLayer = originalLayer.conv2d)
        # self.conv2d = originalLayer.conv2d

        # self.x_proj_smooth = SmoothModule(weight_to_smooth="x_proj_weight", tensor_name="x_proj_smooth")
        self.x_proj_a_quant = QAct(tensor_name="x_proj_a_quant")
        self.x_proj_w_quant = QAct(tensor_name="x_proj_w_quant")

        # self.dt_proj_a_smooth = SmoothModule(weight_to_smooth="dt_proj_a",tensor_name="dt_proj_a_smooth")
        self.dt_proj_a_quant = QAct(tensor_name="dt_proj_a_quant")
        self.dt_proj_w_quant = QAct(tensor_name="dt_proj_w_quant")

        self.xs_quant = QAct(tensor_name="xs_quant")
        self.dts_quant = QAct(tensor_name="dts_quant")
        self.As_quant = QAct(tensor_name="As_quant")
        self.Bs_quant = QAct(tensor_name="Bs_quant")
        self.Cs_quant = QAct(tensor_name="Cs_quant")
        self.Ds_quant = QAct(tensor_name="Ds_quant")
        self.dt_bias_quant = QAct(tensor_name="dt_bias_quant")

        self.act = originalLayer.act
        self.x_proj_weight = originalLayer.x_proj_weight

        self.dt_projs_weight =originalLayer.dt_projs_weight
        self.dt_projs_bias = originalLayer.dt_projs_bias

        self.A_logs = originalLayer.A_logs
        self.Ds = originalLayer.Ds
        self.selective_scan = originalLayer.selective_scan

        self.out_proj_quant = QAct(tensor_name="out_proj_quant")
        self.out_proj_had = HadamardTransform()
        # self.out_proj_smooth = SmoothModule(weight_to_smooth="out_proj", tensor_name="out_proj_smooth")
        self.out_proj = QLinear(originalLayer = originalLayer.out_proj)
        # self.out_proj = originalLayer.out_proj

        self.out_norm = originalLayer.out_norm
        self.dropout = originalLayer.dropout

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", 
            self.x_proj_a_quant(xs.view(B, K, -1, L)), 
            self.x_proj_w_quant(self.x_proj_weight))
        # x_dbl = torch.einsum("b k d l, k c d -> b k c l", 
        #     xs.view(B, K, -1, L), 
        #     self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", 
            self.dt_proj_a_quant(dts.view(B, K, -1, L)), 
            self.dt_proj_w_quant(self.dt_projs_weight))
        # dts = torch.einsum("b k r l, k d r -> b k d l", 
        #     dts.view(B, K, -1, L), 
        #     self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            self.xs_quant(xs), self.dts_quant(dts),
            self.As_quant(As), self.Bs_quant(Bs), self.Cs_quant(Cs), 
            self.Ds_quant(Ds), z=None,
            delta_bias=self.dt_bias_quant(dt_projs_bias),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(self.in_proj_quant(x))
        # xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(self.conv2d_quant(x)))
        # x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        # out = self.out_proj(self.out_proj_quant(self.out_proj_had(y)))
        out = self.out_proj(self.out_proj_quant(y))
        # out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = QSS2D(SS2D(d_model=hidden_dim, 
        #     d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs))
        self.self_attention = SS2D(d_model=hidden_dim, 
            d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        # self.conv_blk = QCAB(CAB(hidden_dim, is_light_sr))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))



    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x

class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

@ARCH_REGISTRY.register()
class Mambaquant(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(Mambaquant, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def get_had36():
    return torch.FloatTensor([
        [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1,
         +1, +1, +1, +1, +1, +1, +1, +1],
        [+1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1,
         +1, -1, -1, -1, +1, -1, +1, +1],
        [+1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1,
         +1, +1, -1, -1, -1, +1, -1, +1],
        [+1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1,
         -1, +1, +1, -1, -1, -1, +1, -1],
        [+1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1,
         -1, -1, +1, +1, -1, -1, -1, +1],
        [+1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, +1, -1, +1, +1, -1, +1,
         -1, -1, -1, +1, +1, -1, -1, -1],
        [+1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1,
         +1, -1, -1, -1, +1, +1, -1, -1],
        [+1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, +1,
         -1, +1, -1, -1, -1, +1, +1, -1],
        [+1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1,
         +1, -1, +1, -1, -1, -1, +1, +1],
        [+1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1,
         +1, +1, -1, +1, -1, -1, -1, +1],
        [+1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1,
         -1, +1, +1, -1, +1, -1, -1, -1],
        [+1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, -1, +1,
         +1, -1, +1, +1, -1, +1, -1, -1],
        [+1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, +1, -1, -1, +1, +1, -1, -1, -1, +1, -1,
         +1, +1, -1, +1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1,
         -1, +1, +1, -1, +1, +1, -1, +1],
        [+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1,
         +1, -1, +1, +1, -1, +1, +1, -1],
        [+1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1,
         -1, +1, -1, +1, +1, -1, +1, +1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1,
         -1, -1, +1, -1, +1, +1, -1, +1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1,
         -1, -1, -1, +1, -1, +1, +1, -1],
        [-1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1,
         -1, +1, +1, +1, -1, +1, -1, -1],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, +1, +1,
         -1, -1, +1, +1, +1, -1, +1, -1],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, +1, +1,
         +1, -1, -1, +1, +1, +1, -1, +1],
        [+1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1,
         +1, +1, -1, -1, +1, +1, +1, -1],
        [+1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1,
         +1, +1, +1, -1, -1, +1, +1, +1],
        [+1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, +1,
         -1, +1, +1, +1, -1, -1, +1, +1],
        [+1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, -1, -1, -1,
         +1, -1, +1, +1, +1, -1, -1, +1],
        [+1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, -1, -1, -1, -1,
         -1, +1, -1, +1, +1, +1, -1, -1],
        [+1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1,
         -1, -1, +1, -1, +1, +1, +1, -1],
        [+1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, -1,
         -1, -1, -1, +1, -1, +1, +1, +1],
        [+1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, -1, +1, -1,
         -1, -1, -1, -1, +1, -1, +1, +1],
        [+1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, -1, +1,
         -1, -1, -1, -1, -1, +1, -1, +1],
        [+1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1,
         +1, -1, -1, -1, -1, -1, +1, -1],
        [+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1,
         -1, +1, -1, -1, -1, -1, -1, +1],
        [+1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1,
         +1, -1, +1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1,
         +1, +1, -1, +1, -1, -1, -1, -1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1,
         +1, +1, +1, -1, +1, -1, -1, -1],
    ])

def get_had60():
    return torch.FloatTensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, ],
        [+1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1,
         -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1,
         +1, +1, -1, +1, ],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1,
         -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1,
         +1, +1, +1, -1, ],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1,
         -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1,
         -1, +1, +1, +1, ],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1,
         +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1,
         +1, -1, +1, +1, ],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1,
         +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1,
         -1, +1, -1, +1, ],
        [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1,
         -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1,
         +1, -1, +1, -1, ],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1,
         -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1,
         -1, +1, -1, +1, ],
        [+1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1,
         -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1,
         -1, -1, +1, -1, ],
        [+1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1,
         -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1,
         +1, -1, -1, +1, ],
        [+1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1,
         +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1,
         -1, +1, -1, -1, ],
        [+1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1,
         -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1,
         -1, -1, +1, -1, ],
        [+1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1,
         -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1,
         +1, -1, -1, +1, ],
        [+1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1,
         -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1,
         +1, +1, -1, -1, ],
        [+1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1,
         +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1,
         +1, +1, +1, -1, ],
        [+1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1,
         +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1,
         -1, +1, +1, +1, ],
        [+1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1,
         -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1,
         +1, -1, +1, +1, ],
        [+1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1,
         +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1,
         +1, +1, -1, +1, ],
        [+1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1,
         +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1,
         +1, +1, +1, -1, ],
        [+1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1,
         -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1,
         +1, +1, +1, +1, ],
        [+1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1,
         +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1,
         -1, +1, +1, +1, ],
        [+1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1,
         -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1,
         -1, -1, +1, +1, ],
        [+1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1,
         +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1,
         +1, -1, -1, +1, ],
        [+1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1,
         -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1,
         +1, +1, -1, -1, ],
        [+1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1,
         -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1,
         +1, +1, +1, -1, ],
        [+1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1,
         -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1,
         +1, +1, +1, +1, ],
        [+1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1,
         +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1,
         +1, +1, +1, +1, ],
        [+1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1,
         -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1,
         -1, +1, +1, +1, ],
        [+1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1,
         +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1,
         -1, -1, +1, +1, ],
        [+1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1,
         +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1,
         -1, -1, -1, +1, ],
        [+1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1,
         -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1,
         -1, -1, -1, -1, ],
        [+1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1,
         +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1,
         -1, -1, -1, -1, ],
        [+1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1,
         +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1,
         +1, -1, -1, -1, ],
        [+1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1,
         +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1,
         +1, +1, -1, -1, ],
        [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1,
         -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1,
         -1, +1, +1, -1, ],
        [+1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1,
         +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1,
         -1, -1, +1, +1, ],
        [+1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1,
         -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1,
         -1, -1, -1, +1, ],
        [+1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1,
         +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1,
         -1, -1, -1, -1, ],
        [+1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1,
         -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1,
         +1, -1, -1, -1, ],
        [+1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1,
         -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1,
         -1, +1, -1, -1, ],
        [+1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1,
         +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1,
         -1, -1, +1, -1, ],
        [+1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1,
         -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1,
         -1, -1, -1, +1, ],
        [+1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1,
         -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1,
         +1, -1, -1, -1, ],
        [+1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1,
         +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1,
         +1, +1, -1, -1, ],
        [+1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1,
         +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1,
         -1, +1, +1, -1, ],
        [+1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1,
         +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1,
         +1, -1, +1, +1, ],
        [+1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1,
         -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1,
         +1, +1, -1, +1, ],
        [+1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1,
         +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1,
         -1, +1, +1, -1, ],
        [+1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1,
         +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1,
         +1, -1, +1, +1, ],
        [+1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1,
         +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1,
         -1, +1, -1, +1, ],
        [+1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1,
         +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1,
         +1, -1, +1, -1, ],
        [+1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1,
         -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1,
         -1, +1, -1, +1, ],
        [+1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1,
         -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1,
         -1, -1, +1, -1, ],
        [+1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1,
         +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1,
         -1, -1, -1, +1, ],
        [+1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1,
         +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1, -1,
         +1, -1, -1, -1, ],
        [+1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1,
         +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1, +1,
         -1, +1, -1, -1, ],
        [+1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1,
         +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1,
         +1, -1, +1, -1, ],
        [+1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1,
         +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1, -1,
         +1, +1, -1, +1, ],
        [+1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1,
         -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, +1,
         -1, +1, +1, -1, ],
        [+1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1,
         -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1,
         +1, -1, +1, +1, ],
    ])


