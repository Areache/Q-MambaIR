# Code Implementation of the MambaIR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from analysis.plt import plot_tensor_histogram, plot_tensor_3d, plot_tensor_HW
# from .lsq_zz_operator import Weight_Quant, Act_Quant,  Quant_linear
from .BL_L_U_operator import Quant_conv, Quant_linear, Quant_out_linear, Act_Quant, oneRK_U as Weight_Quant, DDA_Quant
from analysis.feature_map import draw_feature_map
NEG_INF = -1000000

def quant_proj_mat_mul(x, weight, act_q, weight_q):
    x = torch.matmul(
        act_q(x.permute(0, 1, 3, 2)),     # [B, K, D, L] -> [B, K, L, D]
        weight_q(weight.permute(0, 2, 1))    # [K, C, D] -> [K, D, C]
    )                                    # [B, K, L, C]

    x = x.permute(0, 1, 3, 2)   # [B, K, L, C] -> [B, K, C, L]

    return x

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
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3, squeeze_factor=30, k_bits=32):
        super(CAB, self).__init__()
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2 
            self.cab = nn.Sequential(
                Quant_conv(num_feat, num_feat // compress_ratio, 1, 1, 0, nbits_w=k_bits, nbits_a=k_bits),
                Quant_conv(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1, groups=num_feat//compress_ratio,
                            nbits_w=k_bits, nbits_a=k_bits),
                # nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                # nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1, groups=num_feat//compress_ratio),
                nn.GELU(),
                Quant_conv(num_feat // compress_ratio, num_feat, 1, 1, 0, nbits_w=k_bits, nbits_a=k_bits),
                Quant_conv(num_feat, num_feat, 3, 1,padding=2, groups=num_feat,
                            dilation=2, nbits_w=k_bits, nbits_a=k_bits),
                # nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                # nn.Conv2d(num_feat, num_feat, 3, 1,padding=2, groups=num_feat, dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
            # self.skip_scale1= nn.Parameter(torch.zeros(1, num_feat//compress_ratio, 1, 1))
            # self.skip_scale2= nn.Parameter(torch.zeros(1, num_feat, 1, 1))
            # self.skip_scale3= nn.Parameter(torch.zeros(1, num_feat, 1, 1))
        else:
            self.cab = nn.Sequential(
                Quant_conv(num_feat, num_feat // compress_ratio, 3, 1, 1, nbits_w=k_bits, nbits_a=k_bits),
                nn.GELU(),
                Quant_conv(num_feat // compress_ratio, num_feat, 3, 1, 1, nbits_w=k_bits, nbits_a=k_bits),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):

        return self.cab(x)
    
    def flops(self, H, W):
        flops = 0
        if hasattr(self, 'cab'):
            for module in self.cab:
                if isinstance(module, Quant_conv):
                    # Conv2d FLOPs: (kernel_size^2 * in_channels * out_channels * H * W) / groups
                    if hasattr(module, 'weight'):
                        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                        in_c = module.in_channels
                        out_c = module.out_channels
                        groups = module.groups
                        flops += k * k * in_c * out_c * H * W / groups
                elif isinstance(module, nn.Conv2d):
                    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    in_c = module.in_channels
                    out_c = module.out_channels
                    groups = module.groups
                    flops += k * k * in_c * out_c * H * W / groups
                elif isinstance(module, ChannelAttention):
                    # ChannelAttention: AdaptiveAvgPool2d + 2 Conv2d(1x1) + element-wise multiply
                    # AdaptiveAvgPool2d: negligible
                    # First conv: num_feat -> num_feat // squeeze_factor
                    num_feat = module.attention[1].out_channels * module.attention[1].groups
                    squeeze_factor = num_feat // module.attention[1].out_channels
                    flops += num_feat * (num_feat // squeeze_factor) * H * W  # First conv
                    flops += (num_feat // squeeze_factor) * num_feat * H * W  # Second conv
                    flops += num_feat * H * W  # Element-wise multiply
        return flops

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
            k_bits=32,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.k_bits = k_bits
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
                                  
        self.in_proj = Quant_linear(self.d_model, self.d_inner * 2, bias=bias, nbits_w=k_bits, nbits_a=k_bits # **factory_kwargs
                                   ) 
        # self.in_proj = in_out_linear(self.d_model, self.d_inner * 2, bias=bias, nbits_w=k_bits, nbits_a=k_bits # **factory_kwargs
        #                            )   
        self.conv2d = Quant_conv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            nbits_w=k_bits, nbits_a=k_bits
            # **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (

            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False,  **factory_kwargs
                        ),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False,  **factory_kwargs
                        ),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
                        ),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
                        ),
        )
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        self.x_proj_act_q = Act_Quant(k_bits)
        # self.x_proj_act_q = LTQ_Linear_act(self.d_inner, k_bits)
        self.x_proj_weight_q = Weight_Quant(nbits=k_bits)
        
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         k_bits=k_bits, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         k_bits=k_bits, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         k_bits=k_bits, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         k_bits=k_bits, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.dt_proj_act_q = Act_Quant(k_bits)
        # self.dt_proj_act_q = LTQ_Linear_act(self.dt_rank, k_bits)
        self.dt_proj_weight_q = Weight_Quant(nbits=k_bits)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = Quant_out_linear(self.d_inner, self.d_model, bias=bias, nbits_w=k_bits, nbits_a=k_bits, )
        # self.out_proj = Quant_linear(self.d_inner, self.d_model, bias=bias, nbits_w=k_bits, nbits_a=k_bits, )
        # self.out_proj = in_out_linear(self.d_inner, self.d_model, bias=bias, nbits_w=k_bits, nbits_a=k_bits, )
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        # self.shortcut = nn.Parameter(torch.zeros(1,self.d_inner,1,1))
        self.xs_q = DDA_Quant(k_bits)
        self.dts_q = DDA_Quant(k_bits)
        self.As_q = DDA_Quant(k_bits)
        self.Bs_q = DDA_Quant(k_bits)
        self.Cs_q = DDA_Quant(k_bits) 
        self.Ds_q = DDA_Quant(k_bits)
        self.dt_projs_bias_q = DDA_Quant(k_bits)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                k_bits=32, **factory_kwargs):
      
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

        # import pdb;pdb.set_trace()
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
      
        x_dbl = quant_proj_mat_mul(xs.view(B, K, -1, L), self.x_proj_weight,
                                    act_q=self.x_proj_act_q, weight_q=self.x_proj_weight_q) 
        # x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        dts = quant_proj_mat_mul(dts.view(B, K, -1, L), self.dt_projs_weight,
                                  act_q=self.dt_proj_act_q, weight_q=self.dt_proj_weight_q) 
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        # out_y_all = []
        # out_y_no_q = self.selective_scan(
        #     xs, dts,
        #     As, Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # # print(out_y_no_q.max())
        # # print(out_y_no_q.min())
        # import pdb;pdb.set_trace()
        out_y = self.selective_scan(
            self.xs_q(xs), self.dts_q(dts),
            self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
            delta_bias=self.dt_projs_bias_q(dt_projs_bias),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        # out_y_q = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # # print(out_y_q.max())
        # # print(out_y_q.min())
        # # print(out_y_q.mean())
        # out_y_xs = self.selective_scan(
        #     xs, self.dts_q(dts),
        #     self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_xs=", F.mse_loss(out_y_xs, out_y_no_q))
        # print(F.mse_loss(out_y_xs, out_y_q))
        # # print(out_y_xs.max())
        # # print(out_y_xs.min())
        # # print(out_y_no_q.mean())
        # out_y_dts = self.selective_scan(
        #     self.xs_q(xs), dts,
        #     self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_dts=", F.mse_loss(out_y_dts, out_y_no_q))
        # print(F.mse_loss(out_y_dts, out_y_q))
        # out_y_As = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     As, self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_As=", F.mse_loss(out_y_As, out_y_no_q))
        # print(F.mse_loss(out_y_As, out_y_q))
        # out_y_Bs = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     self.As_q(As), Bs, self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_Bs=", F.mse_loss(out_y_Bs, out_y_no_q))
        # print(F.mse_loss(out_y_Bs, out_y_q))
        # out_y_Cs = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     self.As_q(As), self.Bs_q(Bs), Cs, self.Ds_q(Ds), z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_Cs=", F.mse_loss(out_y_Cs, out_y_no_q))
        # print(F.mse_loss(out_y_Cs, out_y_q))
        # out_y_Ds = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), Ds, z=None,
        #     delta_bias=self.dt_projs_bias_q(dt_projs_bias),
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print("out_y_Ds=", F.mse_loss(out_y_Ds, out_y_no_q))
        # print(F.mse_loss(out_y_Ds, out_y_q))        
        # out_y_dt_bias = self.selective_scan(
        #     self.xs_q(xs), self.dts_q(dts),
        #     self.As_q(As), self.Bs_q(Bs), self.Cs_q(Cs), self.Ds_q(Ds), z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)        
        # print("out_y_dt_bias=", F.mse_loss(out_y_dt_bias, out_y_no_q))
        # print(F.mse_loss(out_y_dt_bias, out_y_q))
        # import pdb; pdb.set_trace()
        # plot_tensor_histogram(out_y, name='out_y')
        # plot_tensor_histogram(As, name='As')
        # plot_tensor_histogram(self.As_q(As), name='As_q')        
        # plot_tensor_histogram(Bs, name='Bs')
        # plot_tensor_histogram(self.Bs_q(Bs), name='Bs_q')
        # plot_tensor_histogram(Cs, name='Cs')
        # plot_tensor_histogram(self.Cs_q(Cs), name='Cs_q')
        # plot_tensor_histogram(Ds, name='Ds')
        # plot_tensor_histogram(self.Ds_q(Ds), name='Ds_q')
        # plot_tensor_histogram(dts, name='dts')
        # plot_tensor_histogram(self.dts_q(dts), name='dts_q')
        # plot_tensor_histogram(xs, name='xs')
        # plot_tensor_histogram(self.xs_q(xs), name='xs_q')
        # plot_tensor_histogram(dt_projs_bias, name='dt_projs_bias')
        # plot_tensor_histogram(self.dt_projs_bias_q(dt_projs_bias), name='dt_projs_bias')
        # print(Cs)
        # old_y = self.selective_scan(
        #     xs, dts,
        #     As, Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # print(old_y.max())
        # print(Cs)
        # old_y_2B = self.selective_scan(
        #     xs, dts,
        #     As, 2*Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # old_y_2c = self.selective_scan(
        #     xs, dts,
        #     As, Bs, 2*Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # old_y_2a = self.selective_scan(
        #     xs, dts,
        #     2*As, Bs, Cs, Ds, z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # plot_tensor_histogram(old_y, name='old_y')
        # import pdb; pdb.set_trace()
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        # plot_tensor_histogram(y1, name='y1')
        # plot_tensor_histogram(y2, name='y2')
        # plot_tensor_histogram(y3, name='y3')
        # plot_tensor_histogram(y4, name='y4')
        # import pdb;pdb.set_trace()

        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        
        y = self.out_norm(y)
        y = y * F.silu(z)
        # import pdb; pdb.set_trace()
        # global index
        # out = self.out_proj(y, index)
        # index += 1
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    
    def flops(self, H, W):
        flops = 0
        # in_proj: Linear(d_model, d_inner * 2)
        flops += H * W * self.d_model * self.d_inner * 2
        
        # conv2d: Depthwise conv
        k = self.d_conv
        flops += k * k * self.d_inner * H * W
        
        # x_proj: 4 projections, each (d_inner, dt_rank + d_state * 2)
        # quant_proj_mat_mul: (B, K, d_inner, L) x (K, dt_rank + d_state * 2, d_inner)
        L = H * W
        K = 4
        flops += K * L * self.d_inner * (self.dt_rank + self.d_state * 2)
        
        # dt_projs: 4 projections, each (dt_rank, d_inner)
        flops += K * L * self.dt_rank * self.d_inner
        
        # selective_scan: approximate as O(N * d_inner * d_state) for each scan
        # 4 scans (K=4), each with length L
        flops += K * L * self.d_inner * self.d_state * 2  # approximate
        
        # out_norm: LayerNorm
        flops += H * W * self.d_inner * 2  # mean and variance
        
        # out_proj: Linear(d_inner, d_model)
        flops += H * W * self.d_inner * self.d_model
        
        return flops


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
            k_bits=32,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,
                                   dropout=attn_drop_rate, k_bits=k_bits, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr, k_bits=k_bits)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input*self.skip_scale + self.drop_path(x)
        # x = input*self.skip_scale + self.drop_path(self.self_attention(x) + x*self.skip_scaleSS)
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x
    
    def flops(self, H, W):
        flops = 0
        # ln_1: LayerNorm
        flops += H * W * self.ln_1.normalized_shape[0] * 2
        
        # self_attention (SS2D)
        flops += self.self_attention.flops(H, W)
        
        # ln_2: LayerNorm
        flops += H * W * self.ln_2.normalized_shape[0] * 2
        
        # conv_blk (CAB)
        flops += self.conv_blk.flops(H, W)
        
        return flops


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
                 use_checkpoint=False,
                 is_light_sr=False,
                 k_bits=32,):

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
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                k_bits=k_bits,))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for i in range(self.depth):

            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.blocks[i], x)
            else:
                x = self.blocks[i](x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        for blk in self.blocks:
            flops += blk.flops(h, w)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@ARCH_REGISTRY.register()
class QuantBL_L_MambaIR(nn.Module):
    r""" Quantized MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
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
                 k_bits=32,
                 **kwargs):
        super(QuantBL_L_MambaIR, self).__init__()

        global index
        index = 1
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
                is_light_sr = self.is_light_sr,
                k_bits = k_bits,
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

        # import pdb;pdb.set_trace()

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        
        # Upsample FLOPs - need to pass H and W
        if hasattr(self, 'upsample') and hasattr(self.upsample, 'flops'):
            if self.upsampler == 'pixelshuffle':
                # conv_before_upsample: Conv2d(embed_dim, num_feat, 3, 1, 1)
                if hasattr(self, 'conv_before_upsample'):
                    num_feat = self.upsample.num_feat
                    flops += 3 * 3 * self.embed_dim * num_feat * h * w
                flops += self.upsample.flops(h, w)
                # conv_last: Conv2d(num_feat, num_out_ch, 3, 1, 1)
                if hasattr(self, 'conv_last'):
                    num_feat = self.upsample.num_feat
                    num_out_ch = 3  # RGB
                    # After upsampling, H and W are multiplied by scale
                    scale = self.upsample.scale
                    flops += 3 * 3 * num_feat * num_out_ch * (h * scale) * (w * scale)
            elif self.upsampler == 'pixelshuffledirect':
                flops += self.upsample.flops(h, w)
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
                 is_light_sr = False,
                 k_bits = 32,):
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
            is_light_sr = is_light_sr,
            k_bits = k_bits,)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = Quant_conv(dim, dim, 3, 1, 1, nbits_a=k_bits, nbits_w=k_bits)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                Quant_conv(dim, dim // 4, 3, 1, 1, nbits_a=k_bits, nbits_w=k_bits), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Quant_conv(dim // 4, dim // 4, 1, 1, 0, nbits_a=k_bits, nbits_w=k_bits), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Quant_conv(dim // 4, dim, 3, 1, 1, nbits_a=k_bits, nbits_w=k_bits))

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
        self.scale = scale
        self.num_out_ch = num_out_ch
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
    
    def flops(self, H, W):
        """Calculate FLOPs for UpsampleOneStep"""
        flops = 0
        # Conv2d FLOPs: kernel_size^2 * in_channels * out_channels * H * W
        flops += 3 * 3 * self.num_feat * (self.scale**2) * self.num_out_ch * H * W
        # PixelShuffle is just a reshape, no FLOPs
        return flops



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        self.scale = scale
        self.num_feat = num_feat
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
    
    def flops(self, H, W):
        """Calculate FLOPs for Upsample"""
        flops = 0
        if (self.scale & (self.scale - 1)) == 0:  # scale = 2^n
            num_steps = int(math.log(self.scale, 2))
            h, w = H, W
            for _ in range(num_steps):
                # Conv2d FLOPs: kernel_size^2 * in_channels * out_channels * H * W
                flops += 3 * 3 * self.num_feat * (4 * self.num_feat) * h * w
                # PixelShuffle is just a reshape, no FLOPs
                h, w = h * 2, w * 2
        elif self.scale == 3:
            # Conv2d FLOPs
            flops += 3 * 3 * self.num_feat * (9 * self.num_feat) * H * W
        return flops


def buildQmambair_light_x2(upscale=2):
    return QuantBL_L_MambaIR(img_size=64,
                   patch_size=1,
                   in_chans=3,
                   embed_dim=60,
                   depths=(6, 6, 6, 6),
                   mlp_ratio=1.5,
                   drop_rate=0.,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   d_state=10,
                   img_range=1.,
                   upsampler='pixelshuffledirect',
                   resi_connection='1conv',
                   k_bits = 32)

def buildQmambair(upscale=2):
    return QuantBL_L_MambaIR(img_size=64,
                   patch_size=1,
                   in_chans=3,
                   embed_dim=180,
                   depths=(6, 6, 6, 6, 6, 6),
                   mlp_ratio=2.,
                   drop_rate=0.,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffle',
                   resi_connection='1conv',
                   k_bits=2)
