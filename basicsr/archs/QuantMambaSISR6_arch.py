## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
## Quantized version using arch_util_swinir quantization methods


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import copy

from einops import rearrange, repeat

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util_swinir import QConv2ds, QLinears, QActQs

from fvcore.nn import flop_count, parameter_count

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # input_t: float, fp16, bf16; weight_t: float;
        # u, B, C, delta: input_t
        # D, delta_bias: float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and (D.dtype != torch.float):
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and (delta_bias.dtype != torch.float):
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()
        
        assert u.shape[1] % (B.shape[1] * nrows) == 0 
        assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        
        _dD = None
        if D is not None:
            if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                _dD = dD.to(ctx._d_dtype)
            else:
                _dD = dD

        _ddelta_bias = None
        if delta_bias is not None:
            if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
            else:
                _ddelta_bias = ddelta_bias

        return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)


def selective_scan_fn_v1(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, qconv='FP32', nbits_w=4, nbits_a=4):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = QConv2ds[qconv](dim, hidden_features*2, kernel_size=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)

        self.dwconv = QConv2ds[qconv](hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)

        self.project_out = QConv2ds[qconv](hidden_features, dim, kernel_size=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class SS2D_1(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        softmax_version=False,
        forward_type="v2",
        # quantization ==========
        qconv='FP32',
        nbits_w=4,
        nbits_a=4,
        # ======================
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv

        dc_inner = 4 
        self.dtc_rank = 6
        self.dc_state = 16
        self.conv_cin = QConv2ds[qconv](in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0, nbits_w=nbits_w, nbits_a=nbits_a)
        self.conv_cout = QConv2ds[qconv](in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0, nbits_w=nbits_w, nbits_a=nbits_a)

        self.forward_core=self.forward_corev1

        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1        
        self.KC = 2
        self.K2C = self.KC if forward_type not in ["share_a"] else 1

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # in proj =======================================
        self.in_conv = QConv2ds[qconv](in_channels=d_model, out_channels=d_expand * 2, kernel_size=1, stride=1, padding=0, nbits_w=nbits_w, nbits_a=nbits_a)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            # DoReFa doesn't support device and dtype parameters
            conv_kwargs = {}
            if qconv not in ['DoReFa']:
                conv_kwargs = factory_kwargs
            self.conv2d = QConv2ds[qconv](
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                nbits_w=nbits_w,
                nbits_a=nbits_a,
                **conv_kwargs,
            )


        self.out_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # x proj ============================
        # DoReFa doesn't support device and dtype parameters
        linear_kwargs = {}
        if qconv not in ['DoReFa']:
            linear_kwargs = factory_kwargs
        self.x_proj = [
            QLinears[qconv](d_inner, (self.dt_rank + self.d_state * 2), bias=False, nbits_w=nbits_w, nbits_a=nbits_a, **linear_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        # xc proj ============================
        self.xc_proj = [
            QLinears[qconv](dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, nbits_w=nbits_w, nbits_a=nbits_a, **linear_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0))
        del self.xc_proj


        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, qconv, nbits_w, nbits_a, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_conv = QConv2ds[qconv](in_channels=d_expand, out_channels=d_model, kernel_size=1, stride=1, padding=0, nbits_w=nbits_w, nbits_a=nbits_a)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.K2C * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, qconv='FP32', nbits_w=4, nbits_a=4, **factory_kwargs):
        # DoReFa doesn't support device and dtype parameters
        linear_kwargs = {}
        if qconv not in ['DoReFa']:
            linear_kwargs = factory_kwargs
        dt_proj = QLinears[qconv](dt_rank, d_inner, bias=True, nbits_w=nbits_w, nbits_a=nbits_a, **linear_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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
        
        return dt_proj


    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log


    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    

    
    def forward_corev1(self, x: torch.Tensor):
        # use official mamba_ssm selective_scan implementation (same style as BL_L_U_mambair_arch)
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
            return xs 
        
        if self.K == 4:
            xs = cross_scan_2d(x)

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(self.A_logs.float())
            Ds = self.Ds
            dt_projs_bias = self.dt_projs_bias.view(-1)

            out_y = self.selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)


        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        

        y = y.view(B, C, H, W)
        y = self.out_norm(y).to(x.dtype)
        
        return y

    def cforward_corev1(self, xc: torch.Tensor):
        # use official mamba_ssm selective_scan implementation (same style as BL_L_U_mambair_arch)
        self.selective_scanC = selective_scan_fn

        b,d,h,w = xc.shape
        
        xc = self.pooling(xc)
        xc = xc.permute(0,2,1,3).contiguous()
        xc = self.conv_cin(xc)
        xc = xc.squeeze(-1)


        B, D, L = xc.shape 
        D, N = self.Ac_logs.shape 
        K, D, R = self.dtc_projs_weight.shape 

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1) 

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight) #8,2,1,96; 2,38,1 ->8,2,38,96
        
        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2) # 8,2,38,96-> 6,16,16
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2 
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2

        out_y = self.selective_scanC(
            xsc, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()


        y = y.unsqueeze(-1) 
        y = self.conv_cout(y) 
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.channel_norm(y)
        y = y.to(xc.dtype)

        
        return y


    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_conv(x)
        x, z = xz.chunk(2, dim=1) # (b, d, h, w)
        if not self.softmax_version:
            z = self.act(z)
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1 = self.forward_core(x)
        y2 = y1 * z
        c = self.cforward_core(y2)
        y3 = y2 * c
        y2 = y3 + y2
        out = self.out_conv(y2)
        return out


##########################################################################
class MamberBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, qconv='FP32', nbits_w=4, nbits_a=4):
        super(MamberBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SS2D_1(d_model=dim, ssm_ratio=1, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False, qconv='FP32', nbits_w=4, nbits_a=4):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = QConv2ds[qconv](in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat, qconv='FP32', nbits_w=4, nbits_a=4):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            QConv2ds[qconv](n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False, nbits_w=nbits_w, nbits_a=nbits_a),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, qconv='FP32', nbits_w=4, nbits_a=4):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            QConv2ds[qconv](n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False, nbits_w=nbits_w, nbits_a=nbits_a),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
## Common utilities for upsampling
def default_conv(in_channels, out_channels, kernel_size, bias=True, qconv='FP32', nbits_w=4, nbits_a=4):
    return QConv2ds[qconv](in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True, qconv='FP32', nbits_w=4, nbits_a=4):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(QConv2ds[qconv](n_feat, 4 * n_feat, 3, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(QConv2ds[qconv](n_feat, 9 * n_feat, 3, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

##########################################################################
##---------- Mamber -----------------------
@ARCH_REGISTRY.register()
class QuantMambaSISR6(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        scale= 4,
        dim = 48,
        num_blocks = [6,2,2,1], 
        num_refinement_blocks = 6,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        qconv='FP32',
        nbits_w=4,
        nbits_a=4,
    ):

        super(QuantMambaSISR6, self).__init__()

        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a)

        self.encoder_level1 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[MamberBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1), qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[MamberBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2), qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[MamberBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3), qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) ## From Level 4 to Level 3
        self.reduce_chan_level3 = QConv2ds[qconv](int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)
        self.decoder_level3 = nn.Sequential(*[MamberBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2), qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) ## From Level 3 to Level 2
        self.reduce_chan_level2 = QConv2ds[qconv](int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a)
        self.decoder_level2 = nn.Sequential(*[MamberBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1), qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[MamberBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[MamberBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, qconv=qconv, nbits_w=nbits_w, nbits_a=nbits_a) for i in range(num_refinement_blocks)])
        

        modules_tail = []
        # Upsampler
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules_tail.append(QConv2ds[qconv](int(dim*2**1), 4 * int(dim*2**1), 3, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a))
                modules_tail.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules_tail.append(QConv2ds[qconv](int(dim*2**1), 9 * int(dim*2**1), 3, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a))
            modules_tail.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        # Final conv
        modules_tail.append(QConv2ds[qconv](int(dim*2**1), out_channels, 3, padding=1, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a))
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.tail(out_dec_level1) + F.interpolate(inp_img, scale_factor=self.scale, mode='nearest') 


        return out_dec_level1


    def flops(self, shape=(3, 64, 64)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return f"params(M) {params/1e6} GFLOPs {sum(Gflops.values())}"


if __name__ == "__main__":
    print(QuantMambaSISR6().flops())

