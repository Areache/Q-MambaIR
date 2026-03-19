
import logging
import torch
from os import path as osp
import sys
import json
import math
from sklearn.cluster import KMeans
# for some possible IMPORT ERROR
# sys.path.append('/data1/guohang/MambaIR-main')
# sys.path.append('/cluster/home/yujichen/MambaIR')
sys.path.append('/leonardo_work/IscrB_FM-EEG24/ychen004/QuantIR_IMPROTANT')
# Add fast_hadamard_transform directory to path for proper import
sys.path.insert(0, '/leonardo_work/IscrB_FM-EEG24/ychen004/QuantIR_IMPROTANT/fast_hadamard_transform')
from basicsr.archs.quamba_arch import QLinear, QConv2D, QAct, VSSBlock
from basicsr.archs.quamba_arch import QSS2D, QCAB, HadamardTransform, SmoothModule
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import fast_hadamard_transform
# Fix namespace package issue: ensure hadamard_transform is available
if not hasattr(fast_hadamard_transform, 'hadamard_transform'):
    from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform
    fast_hadamard_transform.hadamard_transform = hadamard_transform
from functools import partial
from tqdm import tqdm
import numpy as np

class PerTensorPercentileObserver:
    def __init__(self, n_bits, clip_ratio, sym,
                 percentile_sigma=0.01, percentile_alpha=0.99999):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.sym = sym
        self.w_max = None
        self.w_min = None
        self.has_statistic = False
        self.percentile_sigma = percentile_sigma
        self.percentile_alpha = percentile_alpha

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        w = w.clone().to(torch.float32) # quantile() input must be float
        
        if self.sym:
            # import pdb;pdb.set_trace()
            # cur_max = torch.quantile(w.abs().reshape(-1).cpu(), self.percentile_alpha)
            cur_max = np.quantile(w.abs().cpu().numpy().reshape(-1), self.percentile_alpha)
        else:
            cur_max = torch.quantile(w.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(w.reshape(-1),
                                        1.0 - self.percentile_alpha)

        if self.w_max is None:
            self.w_max = cur_max
        else:
            self.w_max = self.w_max + self.percentile_sigma * (cur_max - self.w_max)

        if not self.sym:
            if self.w_min is None:
                self.w_min = cur_min
            else:
                self.w_min = self.w_min + self.percentile_sigma * (cur_min - self.w_min)

    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            sym=self.sym,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio
        )

@torch.no_grad()
def smooth_fc(weight, act_scale, alpha=0.5):
    device = weight.device
    dtype = weight.dtype
    act_scale = act_scale.to(device).to(dtype)
    # linear fc weight shape [out_dim, in_dim]
    weight_scale = weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5) # [out_dim, in_dim] -> [1, in_dim]

    if act_scale.dim() == 0:
        sm_scale = (act_scale[None].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)
    else:
        sm_scale = (act_scale[None, :].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)

    return weight.mul_(sm_scale), sm_scale

def smooth_mamba(model, act_scales, logger):
    
    #TODO: Calculate the real act scales with linear layers.
    smooth_scales = {}
    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            name_prefix = ".".join(name.split(".")[:-1])
            weight_name = name_prefix + "." + m.weight_to_smooth

            weight_module = model.get_submodule(weight_name)
            original_weight = weight_module.weight.clone()
            scale = act_scales[name]
            smooth_weight, sm_scale = smooth_fc(weight_module.weight, scale, alpha=0.5)
            smooth_scales[name] = sm_scale
            m.weight = smooth_weight

            # logging.info(f"Configure smooth module {name}")
            # m.configure(smooth_scales[name])

    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            # logging.info(f"Configure smooth module {name}")
            m.configure(smooth_scales[name])
    
    logger.info(f"########### Smooth Done ###########")

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

def _get_minmax_quantization_params(w_max, w_min, n_bits, clip_ratio, sym):
    q_min, q_max = _get_quant_range(n_bits=n_bits, sym=sym)
    if sym:
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        # Ensure scales is a dense tensor (not sparse)
        if isinstance(scales, torch.Tensor):
            if scales.is_sparse:
                scales = scales.to_dense()
            scales_tensor = scales.clone().detach()
        else:
            scales_tensor = torch.tensor(scales, dtype=torch.float32)
        base = torch.zeros_like(scales_tensor)
    else:
        assert w_min is not None, "w_min should not be None for asymmetric quantization."
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        # Ensure scales is a dense tensor
        if isinstance(scales, torch.Tensor) and scales.is_sparse:
            scales = scales.to_dense()
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        # Ensure base is a dense tensor
        if isinstance(base, torch.Tensor) and base.is_sparse:
            base = base.to_dense()
        scales_tensor = scales if isinstance(scales, torch.Tensor) else torch.tensor(scales, dtype=torch.float32)
    return scales_tensor, base

class PerTensorMinmaxObserver:
    def __init__(self, n_bits, clip_ratio, sym):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.w_max = None
        self.w_min = None
        self.sym = sym
        self.has_statistic = False

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        if self.sym:
            coming_max = w.abs().amax().clamp(min=1e-5)
        else:
            coming_max = w.amax()
            coming_min = w.amin()

        if self.w_max is None:
            self.w_max = coming_max
        else:
            self.w_max = torch.max(coming_max, self.w_max)
        
        if not self.sym:
            if self.w_min is None:
                self.w_min = coming_min
            else:
                self.w_min = torch.min(coming_min, self.w_min)
        
    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio,
            sym=self.sym
        )

# def matmul_hadU_cuda(X, hadK, K, transpose=False):
#     n = X.shape[-1]
#     if K == 1:
#         return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0/torch.tensor(n).sqrt()) 
#     if transpose:
#         hadK = hadK.T.contiguous()
#     input = X.view(-1, K, n // K)
#     input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0/torch.tensor(n).sqrt())
#     input = hadK.to(input.device).to(input.dtype) @ input
#     return input.reshape(X.shape)

class PerStateGroupObserver:
    """
    Quamba2: Per-state-group observer for Bs and Cs matrices.
    Groups states and quantizes each group separately.
    """
    def __init__(self, n_bits, clip_ratio, sym, n_groups=None):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.sym = sym
        self.n_groups = n_groups  # Number of groups for state grouping
        self.state_groups = None  # Will store group assignments
        self.group_observers = {}  # One observer per group
        self.has_statistic = False
        self.all_values = []  # Store all values for clustering
        self.stored_data = []  # Store all calibration data for re-updating after clustering
        
    def update(self, w):
        """
        Update observer with state matrix values.
        w shape: (B, K, d_state, L) for Bs/Cs
        """
        self.has_statistic = True
        
        # Flatten to (B*K*L, d_state) for grouping
        B, K, d_state, L = w.shape
        w_flat = w.permute(0, 1, 3, 2).contiguous().view(-1, d_state)  # (B*K*L, d_state)
        
        # Store values for clustering (use numpy array) - only store max values, not raw data
        current_max = w_flat.abs().max(dim=0)[0].cpu().numpy()
        if len(self.all_values) == 0:
            self.all_values = current_max
        else:
            self.all_values = np.maximum(self.all_values, current_max)
        
        # If groups already determined, update group observers directly
        if self.state_groups is not None and len(self.group_observers) > 0:
            # Update each group's observer
            for g in range(self.n_groups):
                group_mask = (self.state_groups == g)
                if group_mask.sum() > 0:
                    group_values = w_flat[:, group_mask]  # (B*K*L, group_size)
                    self.group_observers[g].update(group_values)
        else:
            # Before clustering: only store a limited number of samples to save memory
            # Store only the first few batches (max 3) for re-updating after clustering
            if len(self.stored_data) < 3:
                self.stored_data.append(w_flat.clone().detach().cpu())
    
    def cluster_states(self, n_groups=None):
        """
        Cluster states based on collected statistics.
        Quamba2: Uses channel magnitude persistence for clustering.
        After clustering, re-update group observers with stored data.
        """
        if len(self.all_values) == 0:
            return
        
        d_state = len(self.all_values)
        
        if n_groups is None:
            # Default: use sqrt of d_state as number of groups
            n_groups = max(1, int(np.sqrt(d_state)))
        
        self.n_groups = min(n_groups, d_state)
        
        if self.n_groups == 1:
            self.state_groups = np.zeros(d_state, dtype=np.int32)
        else:
            # Use KMeans clustering based on magnitude
            values_2d = self.all_values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=self.n_groups, random_state=0, n_init=10)
            self.state_groups = kmeans.fit_predict(values_2d)
        
        # Find which groups actually have states assigned
        unique_groups = np.unique(self.state_groups)
        actual_n_groups = len(unique_groups)
        
        # Remap group IDs to be contiguous (0, 1, 2, ...)
        if actual_n_groups < self.n_groups:
            group_remap = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_groups))}
            self.state_groups = np.array([group_remap[g] for g in self.state_groups], dtype=np.int32)
            self.n_groups = actual_n_groups
        
        # Initialize group observers only for groups that have states
        self.group_observers = {}
        for g in range(self.n_groups):
            self.group_observers[g] = PerTensorMinmaxObserver(
                self.n_bits, self.clip_ratio, self.sym
            )
        
        # Re-update group observers with stored data (limited samples to save memory)
        if len(self.stored_data) > 0:
            for w_flat in self.stored_data:
                # w_flat is already on CPU, convert to tensor if needed
                if isinstance(w_flat, np.ndarray):
                    w_flat_tensor = torch.from_numpy(w_flat)
                else:
                    w_flat_tensor = w_flat
                
                for g in range(self.n_groups):
                    group_mask = (self.state_groups == g)
                    if group_mask.sum() > 0:
                        group_values = w_flat_tensor[:, group_mask]
                        self.group_observers[g].update(group_values)
        
        # Clear stored_data after clustering to free memory
        self.stored_data.clear()
    
    def get_quantization_parameters(self):
        """
        Get quantization parameters for each state group.
        Returns: dict mapping group_id -> (scales, base)
        """
        assert self.has_statistic, "Please run update() before getting quantization parameters."
        
        if self.n_groups is None or len(self.group_observers) == 0:
            # Fallback to per-tensor
            observer = PerTensorMinmaxObserver(self.n_bits, self.clip_ratio, self.sym)
            scales, base = observer.get_quantization_parameters()
            return {0: (scales, base)}, np.zeros(len(self.all_values), dtype=np.int32)
        
        group_params = {}
        for g in range(self.n_groups):
            # Only get parameters from groups that have statistics
            if g in self.group_observers and self.group_observers[g].has_statistic:
                scales, base = self.group_observers[g].get_quantization_parameters()
                group_params[g] = (scales, base)
            else:
                # If a group has no statistics, use default parameters from the first group that has statistics
                # This should not happen after the remapping fix, but keep as safety
                if len(group_params) > 0:
                    group_params[g] = list(group_params.values())[0]
                else:
                    # Fallback: create a dummy observer
                    observer = PerTensorMinmaxObserver(self.n_bits, self.clip_ratio, self.sym)
                    # Initialize with zeros to get valid parameters
                    dummy_data = torch.zeros(1, 1)
                    observer.update(dummy_data)
                    scales, base = observer.get_quantization_parameters()
                    group_params[g] = (scales, base)
        
        # Ensure state_groups is numpy array for consistency
        if isinstance(self.state_groups, torch.Tensor):
            state_groups_np = self.state_groups.cpu().numpy()
        else:
            state_groups_np = self.state_groups
        
        return group_params, state_groups_np

def build_observer(observer_type, n_bits, clip_ratio, sym,
        percentile_sigma=0.01, percentile_alpha=0.99999, n_groups=None
    ):
    if observer_type == "PerTensorMinmaxObserver":
        return PerTensorMinmaxObserver(n_bits, clip_ratio, sym)
    elif observer_type == "PerTensorPercentileObserver":
        logging.debug("Set up PerTensorPercentileObserver with sigma: %.4f, alpha: %.5f" % (percentile_sigma, percentile_alpha))
        return PerTensorPercentileObserver(
            n_bits, clip_ratio, sym, 
            percentile_sigma=percentile_sigma, percentile_alpha=percentile_alpha
        )
    elif observer_type == "PerStateGroupObserver":
        return PerStateGroupObserver(n_bits, clip_ratio, sym, n_groups=n_groups)
    else:
        raise ValueError("Invalid observer type")

# matmul_hadU_cuda and is_pow2 functions are defined here (same as quamba_test.py)
# get_hadK and get_hadXXX functions are defined in this file

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

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

# Placeholder implementations for Hadamard matrices (if not used in your architecture)
def get_had172():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(172, dtype=torch.float32)

def get_had156():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(156, dtype=torch.float32)

def get_had140():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(140, dtype=torch.float32)

def get_had108():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(108, dtype=torch.float32)

def get_had52():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(52, dtype=torch.float32)

def get_had28():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(28, dtype=torch.float32)

def get_had40():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(40, dtype=torch.float32)

def get_had20():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(20, dtype=torch.float32)

def get_had12():
    """Placeholder - returns identity matrix if not implemented"""
    return torch.eye(12, dtype=torch.float32)

def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    assert isinstance(module, (torch.nn.Linear, QLinear))
    in_features, out_features = module.in_features, module.out_features
    
    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()
    
    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        # Apply Hadamard to the last had_dim chunks of the weights
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            W_ = fast_hadamard_transform.hadamard_transform(
                W_.reshape(-1, transposed_shape[-1]//had_dim, had_dim), 
                scale=1/math.sqrt(had_dim)
                ).reshape(transposed_shape).t()
        else:
            raise NotImplementedError("Not implemented (or tested) yet!")
            n = W_.shape[1]
            W_ = fast_hadamard_transform.hadamard_transform(W_.reshape(-1, n//had_dim, had_dim), scale=1/math.sqrt(had_dim)).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)

def rotate_out_proj(model, logger):
    """
    Quamba: Apply Hadamard transform to out_proj weights at runtime.
    """
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            if "out_proj" in name:
                apply_exact_had_to_linear(m, had_dim=-1, output=False)
    logger.info(f"########### Apply Hadamard Weight ###########")

def apply_offline_hadamard_fusion(model, logger):
    """
    Quamba2: Offline Hadamard fusion - fuse Hadamard matrices into weights before quantization.
    This is done offline to avoid runtime overhead.
    """
    fused_count = 0
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            # Fuse Hadamard on input side for in_proj
            if "in_proj" in name:
                apply_exact_had_to_linear(m, had_dim=-1, output=False)
                fused_count += 1
            # Fuse Hadamard on both sides for out_proj (Quamba2 approach)
            elif "out_proj" in name:
                # Apply Hadamard on input side
                apply_exact_had_to_linear(m, had_dim=-1, output=False)
                # Apply Hadamard on output side (transpose)
                apply_exact_had_to_linear(m, had_dim=-1, output=True)
                fused_count += 1
    
    if fused_count > 0:
        logger.info(f"########### Offline Hadamard Fusion: {fused_count} layers fused ###########")
    else:
        logger.info(f"########### No layers found for Hadamard fusion ###########")

def activate_rotate_module(model, logger):
    for name, m in model.named_modules():
        if isinstance(m, (HadamardTransform)):
            m.configure(do_rotate=True)
    logger.info(f"########### Apply Hadamard Act ###########")

def configure_act_quant(model, act_scales, logger, a_bits, hybrid_config=None, state_group_configs=None):
    """
    Quamba2: Configure activation quantization with support for hybrid precision and per-state-group quantization.
    
    Args:
        model: The model to configure
        act_scales: Dictionary of activation scales (may contain per-state-group configs)
        logger: Logger instance
        a_bits: Default activation bits
        hybrid_config: Dictionary mapping layer names to their activation bits (for hybrid precision)
        state_group_configs: Dictionary with per-state-group configurations for Bs/Cs
    """
    per_state_group_count = 0
    
    for name, m in model.named_modules():
        if isinstance(m, QAct):
            # Determine activation bits for this layer
            layer_a_bits = a_bits
            if hybrid_config is not None:
                # Check if this layer has a specific configuration
                for key, value in hybrid_config.items():
                    if key in name:
                        if isinstance(value, dict) and 'a_bits' in value:
                            layer_a_bits = value['a_bits']
                        elif isinstance(value, int):
                            layer_a_bits = value
                        break
            
            scale_data = act_scales.get(name)
            
            # Quamba2: Check if this is per-state-group quantization
            # Per-state-group quantization has (group_params_dict, state_groups_array)
            # where group_params is a dict and state_groups is an array
            is_per_state_group = False
            if scale_data is not None and isinstance(scale_data, tuple) and len(scale_data) == 2:
                group_params, state_groups = scale_data
                # Check if first element is a dict (group_params) and second is array/tensor (state_groups)
                if isinstance(group_params, dict) and (isinstance(state_groups, (np.ndarray, torch.Tensor, list)) or hasattr(state_groups, '__len__')):
                    is_per_state_group = True
            
            if is_per_state_group:
                # Per-state-group quantization: (group_params, state_groups)
                group_params, state_groups = scale_data
                
                # Convert state_groups to tensor if needed
                if isinstance(state_groups, np.ndarray):
                    state_groups_tensor = torch.from_numpy(state_groups).long()
                elif isinstance(state_groups, torch.Tensor):
                    state_groups_tensor = state_groups.long()
                else:
                    state_groups_tensor = torch.tensor(state_groups, dtype=torch.long)
                
                # Store state groups in the module for use during forward
                if not hasattr(m, 'state_groups'):
                    m.register_buffer('state_groups', state_groups_tensor)
                else:
                    m.state_groups = state_groups_tensor
                
                # Store group parameters
                if not hasattr(m, 'group_params'):
                    m.group_params = {}
                m.group_params = group_params
                
                # Use first group's parameters as default (for compatibility)
                # Convert to list first to avoid sparse tensor issues
                group_params_list = [(k, v) for k, v in group_params.items()]
                first_group_params = group_params_list[0][1]
                scale, base = first_group_params
                # Ensure scale and base are dense tensors
                if isinstance(scale, torch.Tensor) and scale.is_sparse:
                    scale = scale.to_dense()
                if isinstance(base, torch.Tensor) and base.is_sparse:
                    base = base.to_dense()
                
                m.configure(
                    n_bits=layer_a_bits,
                    sym=True,
                    o_scales=scale,
                    o_base=base,
                )
                per_state_group_count += 1
            else:
                # Standard per-tensor quantization
                if scale_data is not None:
                    (scale, base) = scale_data
                    m.configure(
                        n_bits=layer_a_bits,
                        sym=True,
                        o_scales=scale,
                        o_base=base,
                    )
    
    if per_state_group_count > 0:
        logger.info(f"########### Set Act to %d bit quant (hybrid: %s, per-state-group: %d tensors) ###########" % 
                   (a_bits, "enabled" if hybrid_config else "disabled", per_state_group_count))
    else:
        logger.info(f"########### Set Act to %d bit quant (hybrid: %s) ###########" % (a_bits, "enabled" if hybrid_config else "disabled"))

def activate_quant_module(model):
    for name, m in model.named_modules():
        if isinstance(m, (QAct, QLinear, QConv2D)):
            m.is_quant_mode = True

def configure_weight_quant(model, logger, w_bits, hybrid_config=None):
    """
    Configure weight quantization with support for hybrid precision.
    
    Args:
        model: The model to configure
        logger: Logger instance
        w_bits: Default weight bits
        hybrid_config: Dictionary mapping layer names to their weight bits (for hybrid precision)
    """
    for name, m in model.named_modules():
        if isinstance(m, (QLinear, QConv2D)):
            # Determine weight bits for this layer
            layer_w_bits = w_bits
            if hybrid_config is not None:
                # Check if this layer has a specific configuration
                for key, value in hybrid_config.items():
                    if key in name:
                        if isinstance(value, dict) and 'w_bits' in value:
                            layer_w_bits = value['w_bits']
                        elif isinstance(value, int):
                            layer_w_bits = value
                        break
            
            m.configure(
                n_bits=layer_w_bits,
            )
    logger.info(f"########### Set Weight to %d bit quant (hybrid: %s) ###########" % (w_bits, "enabled" if hybrid_config else "disabled"))

def prepare_quantize_model_mamba(model, logger):
    logger.info(f"########### Insert Quantized module ###########")

    for i in range(len(model.net_g.layers)):
        for j in range(len(model.net_g.layers[i].residual_group.blocks)):
            m = None
            n = None
            if isinstance(model.net_g.layers[i].residual_group.blocks[j], VSSBlock):
                m = QSS2D(originalLayer=model.net_g.layers[i].residual_group.blocks[j].self_attention)
                n = QCAB(originalLayer=model.net_g.layers[i].residual_group.blocks[j].conv_blk)
            if m is None:
                continue

            m = m.to(model.device)
            n = n.to(model.device)
            model.net_g.layers[i].residual_group.blocks[j].self_attention = m
            model.net_g.layers[i].residual_group.blocks[j].conv_blk = n
    
    return model

def prepare_act_scales(model, logger, a_bits, test_loader, num_samples, hybrid_config=None, 
                      use_per_state_group=True, n_state_groups=None):
    """
    Quamba2: Prepare activation scales with support for per-state-group quantization.
    
    Args:
        model: The model to calibrate
        logger: Logger instance
        a_bits: Default activation bits
        test_loader: Data loader for calibration
        num_samples: Number of calibration samples
        hybrid_config: Hybrid precision configuration
        use_per_state_group: Whether to use per-state-group quantization for Bs/Cs
        n_state_groups: Number of groups for state grouping (None for auto)
    """
    smooth_scales = {}
    logger.info(f"########### Start calibration (Quamba2) ###########")
    observers = {}
    state_group_configs = {}  # Store per-state-group configurations
    
    def stat_act_hook(m, inputs, outputs, name):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        # register the new information to observer
        observers[name].update(inputs.clone().detach())
    
    def stat_smooth_hook(m, inputs, outputs, name):
        x = inputs[0].clone().detach() if isinstance(inputs, tuple) else inputs.clone().detach()
        # assert x.dim() == 3, "Assuming x is of input shape (B, L, D)"
        if x.dim() == 3 :
            comming_max = x.abs().amax(dim=(0, 1))
        elif x.dim() == 4 :  
            b,k,l,c = x.shape
            comming_max = x.reshape(b, -1, c).abs().amax(dim=(0, 1))
        
        if name not in smooth_scales:
            smooth_scales[name] = comming_max
        else:
            # import pdb;pdb.set_trace()
            smooth_scales[name] = torch.max(smooth_scales[name], comming_max)

    hooks = []
    for name, m in model.named_modules():
        # import pdb;pdb.set_trace()
        if isinstance(m, QAct):
            tensor_name = m.tensor_name
            
            # Determine activation bits for this layer (for observer)
            layer_a_bits = a_bits
            if hybrid_config is not None:
                for key, value in hybrid_config.items():
                    if key in name:
                        if isinstance(value, dict) and 'a_bits' in value:
                            layer_a_bits = value['a_bits']
                        elif isinstance(value, int):
                            layer_a_bits = value
                        break
            
            hooks.append(
                m.register_forward_hook(partial(stat_act_hook, name=name))
            )
            
            # Quamba2: Use per-state-group observer for Bs and Cs
            if use_per_state_group and (tensor_name == "Bs_quant" or tensor_name == "Cs_quant"):
                a_observer_type = "PerStateGroupObserver"
                observers[name] = build_observer(
                    observer_type=a_observer_type,
                    n_bits=layer_a_bits,
                    clip_ratio=1.0,
                    sym=True,
                    n_groups=n_state_groups
                )
                logger.debug(f"Using PerStateGroupObserver for {name}")
            elif tensor_name == "x_proj_a_quant":
                a_observer_type = "PerTensorPercentileObserver"
                observers[name] = build_observer(
                    observer_type=a_observer_type, 
                    n_bits=layer_a_bits,
                    clip_ratio=1.0,
                    sym=True,
                    percentile_alpha=0.99999
                )
            else:
                a_observer_type = "PerTensorMinmaxObserver"
                observers[name] = build_observer(
                    observer_type=a_observer_type, 
                    n_bits=layer_a_bits,
                    clip_ratio=1.0,
                    sym=True
                )
        if isinstance(m, SmoothModule):
            hooks.append(
                m.register_forward_hook(partial(stat_smooth_hook, name=name))
            )

    # Calibration phase - first pass: collect statistics for clustering
    data_iter = iter(test_loader)
    for i in tqdm(range(num_samples), desc="### Start Observering ###"):
        batch = next(data_iter)
        model.lq = batch['lq'].to(model.device)
        model.test()
    
    # Quamba2: Cluster states for per-state-group observers
    has_per_state_group = False
    for name, observer in observers.items():
        if isinstance(observer, PerStateGroupObserver):
            observer.cluster_states(n_groups=n_state_groups)
            logger.debug(f"Clustered states for {name} into {observer.n_groups} groups")
            has_per_state_group = True
    
    # Second pass: re-run calibration data to update group observers after clustering
    # This ensures group observers have sufficient statistics without storing all data
    if has_per_state_group:
        data_iter = iter(test_loader)
        for i in tqdm(range(min(num_samples, 3)), desc="### Update Group Observers ###"):
            batch = next(data_iter)
            model.lq = batch['lq'].to(model.device)
            model.test()
    
    for h in hooks:
        h.remove()
        
    act_scales = {}
    for name, observer in observers.items():
        if isinstance(observer, PerStateGroupObserver):
            # Per-state-group quantization returns (group_params, state_groups)
            group_params, state_groups = observer.get_quantization_parameters()
            act_scales[name] = (group_params, state_groups)
            state_group_configs[name] = {
                'n_groups': observer.n_groups,
                'state_groups': state_groups
            }
        else:
            act_scales[name] = observer.get_quantization_parameters()

    if use_per_state_group and len(state_group_configs) > 0:
        logger.info(f"########### Per-state-group quantization configured for {len(state_group_configs)} tensors ###########")

    return act_scales, smooth_scales, state_group_configs

def quantize_model_mamba(model, logger, w_bits, a_bits, act_scales, smooth_scales, hybrid_config=None, 
                         state_group_configs=None, use_offline_hadamard=False):
    """
    Quamba2: Quantize model with support for hybrid precision, per-state-group quantization, and offline Hadamard fusion.
    
    Args:
        model: The model to quantize
        logger: Logger instance
        w_bits: Default weight bits
        a_bits: Default activation bits
        act_scales: Dictionary of activation scales
        smooth_scales: Dictionary of smooth scales
        hybrid_config: Dictionary for hybrid precision configuration
        state_group_configs: Dictionary with per-state-group configurations
        use_offline_hadamard: Whether to apply offline Hadamard fusion (Quamba2)
    """
    # Quamba2: Apply offline Hadamard fusion before quantization
    if use_offline_hadamard:
        apply_offline_hadamard_fusion(model, logger)
    else:
        # Quamba: Use runtime Hadamard (legacy)
        # rotate_out_proj(model, logger)
        # activate_rotate_module(model, logger)
        pass
    
    # smooth_mamba(model, smooth_scales, logger)
    configure_weight_quant(model, logger, w_bits, hybrid_config)
    configure_act_quant(model, act_scales, logger, a_bits, hybrid_config, state_group_configs)
    activate_quant_module(model)

    return model

def load_hybrid_config(config_path):
    """
    Load hybrid blocks configuration from JSON file.
    
    Args:
        config_path: Path to the hybrid blocks configuration JSON file
        
    Returns:
        Dictionary mapping layer patterns to quantization bits, or None if file doesn't exist
    """
    if config_path is None or not osp.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.warning(f"Failed to load hybrid config from {config_path}: {e}")
        return None

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    print('############## We are testing %s ! Good Luck ! ##############' % opt['name'] )
    print('############## We are testing %s ! Good Luck ! ##############' % opt['name'] )
    print('############## We are testing %s ! Good Luck ! ##############' % opt['name'] )

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(dict2str(opt))

    # Load hybrid blocks configuration if specified (Quamba2 feature)
    hybrid_config = None
    hybrid_config_path = opt.get('network_g', {}).get('hybrid_blocks_config', None)
    if hybrid_config_path:
        hybrid_config = load_hybrid_config(hybrid_config_path)
        if hybrid_config:
            logger.info(f"########### Loaded Hybrid Blocks Config from {hybrid_config_path} ###########")
        else:
            logger.warning(f"########### Failed to load Hybrid Blocks Config from {hybrid_config_path} ###########")

    # create test dataset and dataloader
    test_loaders = []
    for i, dataset_opt in sorted(opt['datasets'].items()):
        if i == 'test_0':
            observation_datasets_opt = dataset_opt
        else:
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)
    # import pdb;pdb.set_trace()
    act_set = build_dataset(observation_datasets_opt)
    act_loader = build_dataloader(
                act_set, observation_datasets_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], 
                sampler=None, seed=opt['manual_seed'])
    logger.info(f"Number of observation images in {observation_datasets_opt['name']}: {len(act_set)}")

    # create model
    # Quamba2: Support separate w_bits and a_bits
    w_bits = opt['network_g'].get('w_bits', opt['network_g'].get('w_bits', 4))
    a_bits = opt['network_g'].get('a_bits', opt['network_g'].get('a_bits', 8))
    
    # Quamba2: Per-state-group quantization settings
    use_per_state_group = opt['network_g'].get('use_per_state_group', True)
    n_state_groups = opt['network_g'].get('n_state_groups', None)  # None for auto
    use_offline_hadamard = opt['network_g'].get('use_offline_hadamard', False)
    
    logger.info(f"########### Quantization Config: W{w_bits}A{a_bits} ###########")
    if hybrid_config:
        logger.info(f"########### Hybrid Precision: Enabled ###########")
    if use_per_state_group:
        logger.info(f"########### Per-state-group Quantization: Enabled (groups: {n_state_groups or 'auto'}) ###########")
    if use_offline_hadamard:
        logger.info(f"########### Offline Hadamard Fusion: Enabled ###########")
    
    model = build_model(opt)
    model.eval()
    model = prepare_quantize_model_mamba(model, logger)
    q_model = model
    logger.info(f"########### Start Quantizing Model (Quamba2) ###########")
    act_scales, smooth_scales, state_group_configs = prepare_act_scales(
        model, logger, a_bits, act_loader, num_samples=1, 
        hybrid_config=hybrid_config,
        use_per_state_group=use_per_state_group,
        n_state_groups=n_state_groups
    )
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        q_model = quantize_model_mamba(
            q_model, logger, w_bits, a_bits, act_scales, smooth_scales, 
            hybrid_config=hybrid_config,
            state_group_configs=state_group_configs,
            use_offline_hadamard=use_offline_hadamard
        )
        q_model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

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


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

    