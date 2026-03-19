import logging
import torch
from os import path as osp
import sys
# for some possible IMPORT ERROR
# sys.path.append('/data1/guohang/MambaIR-main')
# sys.path.append('/cluster/home/yujichen/MambaIR')
sys.path.append('/leonardo_work/IscrB_FM-EEG24/ychen004/QuantIR_IMPROTANT')
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.archs.mambaquant_arch import QAct
from basicsr.quamba_test import PerTensorMinmaxObserver, PerTensorPercentileObserver
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy
from typing import Union
import abc, sys, torch, torch.distributed as dist, math
from tqdm import tqdm
import functools
from typing import Iterable
from functools import partial

CLIPMIN = 1e-5

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    # quant,_ = parse_options_quant(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(dict2str(opt))

    # import pdb;pdb.set_trace()
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
    
    act_set = build_dataset(observation_datasets_opt)
    act_loader = build_dataloader(
                act_set, observation_datasets_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], 
                sampler=None, seed=opt['manual_seed'])
    logger.info(f"Number of observation images in {observation_datasets_opt['name']}: {len(act_set)}")
    
    # create model
    model = build_model(opt)
    w_cfg = {"dynamic_method":"per_channel", "per_channel_axes":[0], "n_bits": opt['network_g']['k_bits']}
    a_cfg = {"dynamic_method": "per_tensor", "n_bits": opt['network_g']['k_bits']}

    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if 'conv_first' in name:continue
            if 'conv_after_body' in name:continue
            if 'conv_before_upsample' in name:continue
            if 'conv_last' in name:continue
            if 'upsample' in name:continue
            if 'cab.attention' in name:continue
            if  "patch" in name:continue
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                setattr(model, name, 
                        replacement_class(
                            child, weight_quant_params=w_cfg, act_quant_params=a_cfg, observe='percentile'))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)
    logger.info(f"########### Start Quantizing Model ###########")
    replace_layers(model, nn.Linear, QuantLinear)
    replace_layers(model, nn.Conv2d, QuantConv2d)
    set_quant_state(model, weight_quant=True, act_quant=True)
    logger.info(f"########### Set Weight to %d bit quant ###########" % w_cfg['n_bits'])
    logger.info(f"########### Set Act to %d bit quant ###########" % a_cfg['n_bits'])

    act_model = build_model(opt)
    '''use_klt + use_hadmard'''
    test_set_name = act_loader.dataset.opt['name']
    # act_klt, act_scales, act_shifts = get_act(model, act_loader, num_samples=1, test_set_name=test_set_name)
    # act_scales = get_act_scales(act_model, act_loader, test_set_name, num_samples=1)
    act_scales = prepare_act_scales(act_model, logger, a_cfg['n_bits'], act_loader, num_samples=1)
    # klt_scales = get_klt_scales(act_model, act_loader, test_set_name, num_samples=1)
    device = model.net_g.layers[0].residual_group.blocks[0].self_attention.out_proj.weight.device
    dtype = model.net_g.layers[0].residual_group.blocks[0].self_attention.out_proj.weight.dtype
    logger.info(f'############ {test_set_name} Act Scales Done ############')

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        q_model = model
        q_model = mamba_quant(q_model, None, act_scales, None, logger, test_set_name, device, dtype, a_cfg['n_bits'])
        q_model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

def mamba_quant(model, klt_scales, act_scales, act_shifts, logger, test_set_name, device, dtype, n_bits):

    configure_act_quant(model, act_scales, logger, n_bits)
    # '''use_S2'''
    # for i in range(len(model.net_g.layers)):
    #     for j in range(len(model.net_g.layers[i].residual_group.blocks)):   
    #         act = act_scales[f"net_g.layers.{i}.residual_group.blocks.{j}.self_attention.out_proj"].to(device=device, dtype=dtype)
    #         weight_scales = model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
    #         alpha = 0.5
    #         scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
    #         model.net_g.layers[i].residual_group.blocks[j].register_parameter("s_out", nn.Parameter(scales))

    #         act = act_scales[f"net_g.layers.{i}.residual_group.blocks.{j}.self_attention.in_proj"].to(device=device, dtype=dtype)
    #         weight_scales = model.net_g.layers[i].residual_group.blocks[j].self_attention.in_proj.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
    #         alpha = 0.5
    #         scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
    #         model.net_g.layers[i].residual_group.blocks[j].register_parameter("s_in", nn.Parameter(scales))
    
    # for i in range(len(model.net_g.layers)):
    #     for j in range(len(model.net_g.layers[i].residual_group.blocks)):
    #         k=1
    #         model.net_g.layers[i].residual_group.blocks[j].self_attention.in_proj.weight.data = \
    #             (1/model.net_g.layers[i].residual_group.blocks[j].s_in) * \
    #                 model.net_g.layers[i].residual_group.blocks[j].self_attention.in_proj.weight.data
    #         model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data = \
    #             model.net_g.layers[i].residual_group.blocks[j].s_out * \
    #                 model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data
    #         # model.net_g.layers[i].residual_group.blocks[j].self_attention.act = Swiglu(\
    #         #     model.net_g.layers[i].residual_group.blocks[j].s_out)
    # logger.info(f'############ {test_set_name} S2 done ############')

    # R1 = random_hadamard_matrix(model.net_g.layers[0].residual_group.blocks[0].self_attention.in_proj.in_features,  device).to(device=device, dtype=dtype)
    # R2 = random_hadamard_matrix(model.net_g.layers[0].residual_group.blocks[0].self_attention.out_proj.in_features,device).to(device=device, dtype=dtype)
    
    # K = klt_scales[f"net_g.layers.{i}.residual_group.blocks.{j}.self_attention.in_proj"].to(device=device, dtype=dtype)
    # R1 = K@R1
    # for i in range(len(model.net_g.layers)):
    #     for j in range(len(model.net_g.layers[i].residual_group.blocks)):   
    #         model.net_g.layers[i].residual_group.blocks[j].self_attention.in_proj.weight.data = \
    #             model.net_g.layers[i].residual_group.blocks[j].self_attention.in_proj.weight.data@R1 
    #         model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data = \
    #             R1.T@model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data
    # logger.info(f'############ {test_set_name} R1 done ############')
    # # model.R1 =R1 
    # R2 = R2.T
    # for i in range(len(model.net_g.layers)):
    #     for j in range(len(model.net_g.layers[i].residual_group.blocks)):  
    #         model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data = \
    #             model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data@\
    #                 R2.to(model.net_g.layers[i].residual_group.blocks[j].self_attention.out_proj.weight.data)
    # logger.info(f'############ {test_set_name} R2 done ############')

    return model

def configure_act_quant(model, act_scales, logger, n_bits):
    # import pdb;pdb.set_trace()
    for name, m in model.named_modules():
        if isinstance(m, QAct):
            # import pdb;pdb.set_trace()
            (scale, base) = act_scales.get(name)
            m.configure(
                n_bits = n_bits,
                sym = True,
                o_scales=scale, 
                o_base=base,
            )
            m.is_quant_mode = True
    logger.info(f"########### Set Act to %d bit quant ###########" % n_bits)

def prepare_act_scales(model, logger, n_bits, test_loader, num_samples):

    logger.info(f"########### Start calibration ###########")
    observers = {}
    
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
            hooks.append(
                m.register_forward_hook(partial(stat_act_hook, name=name))
            )
            if tensor_name == "x_proj_a_quant":
                a_observer_type = "PerTensorPercentileObserver"
            else:
                a_observer_type = "PerTensorMinmaxObserver"
            # logging.debug(f"Create observer for tensor {name} with {a_observer_type} observer")
            observers[name] = build_observer(
                observer_type=a_observer_type, 
                n_bits=n_bits,
                clip_ratio=1.0,
                sym=True,
                percentile_alpha=0.99999
            )

    data_iter = iter(test_loader)
    for i in tqdm(range(num_samples), desc="### Start Observering ###"):
        batch = next(data_iter)
        model.lq = batch['lq'].to(model.device)
        model.test()
    
    for h in hooks:
        h.remove()
        
    act_scales = {}
    for name, observer in observers.items():
        act_scales[name] = observer.get_quantization_parameters()

    return act_scales

def build_observer(observer_type, n_bits, clip_ratio,sym,
        percentile_sigma=0.01, percentile_alpha=0.99999
    ):
    if observer_type == "PerTensorMinmaxObserver":
        return PerTensorMinmaxObserver(n_bits, clip_ratio, sym)
    elif observer_type == "PerTensorPercentileObserver":
        logging.debug("Set up PerTensorPercentileObserver with sigma: %.4f, alpha: %.5f" % (percentile_sigma, percentile_alpha))
        return PerTensorPercentileObserver(
            n_bits, clip_ratio, sym, 
            percentile_sigma=percentile_sigma, percentile_alpha=percentile_alpha
        )
    else:
        raise ValueError("Invalid observer type")
    
def get_act_scales(model, dataloader, test_set_name, num_samples=1):
    act_loader = dataloader
    act_model = model
    act_model.eval()
    device = next(act_model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    def stat_input_hook_2(m, x, y, name):
        if isinstance(x, tuple):
            # import pdb;pdb.set_trace()
            # x = x[0].squeeze(2).permute(0,2,1)
            x = x[0].permute(0, 2, 3, 1)
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        # if isinstance(m, nn.Linear):
        #     hooks.append(
        #         m.register_forward_hook(
        #             functools.partial(stat_input_hook, name=name)))
        # if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        #     if 'patch_embed' in name:continue
        #     if 'conv_first' in name:continue
        #     if 'conv_after_body' in name:continue
        #     if 'conv_before_upsample' in name:continue
        #     if 'conv_last' in name:continue
        #     if 'upsample' in name:continue
        #     if 'cab.attention' in name:continue
        #     hooks.append(
        #         m.register_forward_hook(
        #             functools.partial(stat_input_hook_2, name=name)))
        if isinstance(m, QAct):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    data_iter = iter(act_loader)
    for i in tqdm(range(num_samples),  desc=f"{test_set_name}_Get_Act_Scale", dynamic_ncols=True, leave=True):
        batch = next(data_iter)
        act_model.lq = batch['lq'].to(act_model.device)
        act_model.test()

    for h in hooks:
        h.remove()

    return act_scales

def get_klt_scales(model, dataloader, test_set_name, num_samples=1):
    act_loader = dataloader
    act_model = model
    act_model.eval()
    device = next(act_model.parameters()).device
    klt_scales = {}

    def klt_scales_stat_tensor(name, tensor):
        shape = tensor.shape
        cov_matrix = torch.cov(tensor.reshape(-1,shape[-1]).double().T)
    
        # 计算协方差矩阵的特征值和特征向量
        # eig_values, klt_matrix = torch.linalg.eig(cov_matrix)
        eig_values, K = torch.linalg.eig(cov_matrix.double())
        if (K @ K.T).real[0,0] >0.99 and (K @ K.T).real[0,1]<0.0001 :
            # print(f"{name} input is orthogonal")
            K = K
        else:
            # K, S, Vt = torch.linalg.svd(K)
            K = gram_schmidt(K)
            # print(f"{name} input is not orthogonal")
        # K = optimize_klt_matrix_1(cov_matrix.float(),K.T.float()).T
        klt_scales[name] = K.real.float()

    def klt_stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        klt_scales_stat_tensor(name, x)

    hooks = []
    for name, m in act_model.named_modules():
        if "head" in name:continue
        if isinstance(m, (nn.Linear, QuantLinear)):
            hooks.append(m.register_forward_hook(functools.partial(klt_stat_input_hook, name=name)))

    data_iter = iter(act_loader)
    for i in tqdm(range(num_samples),  desc=f"{test_set_name}_Get_Klt_Scale", dynamic_ncols=True, leave=True):
        batch = next(data_iter)
        act_model.lq = batch['lq'].to(act_model.device)
        act_model.test()

    for h in hooks:
        h.remove()
    del act_model, act_loader

    return klt_scales

def gram_schmidt(K):
    # 假设 K 是 n x n 矩阵
    n = K.size(1)
    Q = torch.zeros_like(K)
    for i in range(n):
        # 取出 K 的第 i 列
        q = K[:, i]
        # 对当前列向量 q 进行正交化
        for j in range(i):
            q -= torch.dot(Q[:, j], K[:, i]) * Q[:, j]
        # 归一化处理
        Q[:, i] = q / q.norm()
    return Q

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for name,m in self.named_modules():
        if isinstance(m, (QuantLinear, QuantConv2d)):
            m.set_quant_state(weight_quant, act_quant)

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
    elif n % 45 == 0:  # 新增分支
        assert (is_pow2(n // 45))
        K = 45
        hadK = get_had45().T if transpose else get_had45()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert (is_pow2(n // 60))
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
    elif n % 28 == 0: #llama-3 up
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
    if K == None:
        print(n.shape)
        raise RuntimeError
    return hadK, K

def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    if input.shape[1] == None:
        raise RuntimeError
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()

def random_hadamard_matrix(size, device):
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1 # 生成随机的-1或1
    Q = torch.diag(Q)
    Q = matmul_hadU(Q).to(device)
    return Q # 根据符号构造H矩阵

class Swiglu(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        return x*self.sigmod(x*self.s)
                    
class QuantLinear(nn.Linear):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
    ):
        super().__init__(org_module.in_features,org_module.out_features)
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight=True,observe=observe)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True,observe=observe)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        
        self.weight_quantized = False
    
    def forward(self, input: torch.Tensor):

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if self.weight_quantizer.is_observing:
                weight = self.weight
            elif not self.weight_quantized:
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                weight = self.weight
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        if bias is not None:bias = bias.to(weight)
        out = self.fwd_func(
                input.to(weight), weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class QuantConv2d(nn.Conv2d):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
    ):
        super().__init__(org_module.in_channels, org_module.out_channels, org_module.kernel_size,)
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv2d
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight=True,observe=observe)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True,observe=observe)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

     
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        rescale=False,
        rescale_limit=False,
        has_batch_dim = False,
        is_weight=False,
        observe="minmax",
        percent = 0.999999,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rescale = rescale # for channel-rescale
        self.rescale_limit = rescale_limit

        init_value = 4.0  # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        
        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1,1) )
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1,1) )

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        
        self.has_batch_dim = has_batch_dim
        self.is_observing = False
        self.is_dynamic_quant = True
        granularity = 'dim{}'.format(per_channel_axes[0]) if len(per_channel_axes) > 0 else 'tensor'
        
        if observe == "percentile":
            self.observer = PercentileObserver(percent=0.999999,granularity=granularity)
        else:
            self.observer = MinMaxObserver(granularity=granularity)
 
        self.observered = False
        
        self.is_weight = is_weight

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape)==3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant*rescale_param.to(x_dequant.device)
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        
        if self.is_weight:#权重量化，没有observe过程
            if True:#not self.is_dynamic_quant:
                if  self.is_observing:
                    return x
                if self.observer is not None:
                    self.observer.update(x)
                    xmin,xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin,xmax)
                    self.scale = self.expand_scale_shape_2_x(x, self.scale)
                    self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                    self.observer = None
                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)
            # else:
            #     if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            #         self.per_token_dynamic_calibration(x)
            #     else:
            #         self.dynamic_per_tensor_calibration(x)
            #     x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            #     return x_dequant
        else:#激活量化
            if not self.is_dynamic_quant:
                if self.is_observing:
                    self.observer.update(x)
                    return x.type_as(x)
                else:
                    if not self.observered:
                        xmin,xmax = self.observer.cal_min_max()
                        self.assymmetric_cal_scale(xmin,xmax)
                        self.scale = self.expand_scale_shape_2_x(x, self.scale)
                        self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                        self.observered = True
                        self.observer = None
                    x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                    return x_dequant.type_as(x)
                    
            else:
                if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
                    self.per_token_dynamic_calibration(x)
                else:
                    self.dynamic_per_tensor_calibration(x)

                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)

    def expand_scale_shape_2_x(self, x, scale):
        if self.per_channel_axes:
            dim=self.per_channel_axes[0]
            for i in range(len(x.shape)):
                if i != dim:
                    scale = scale.unsqueeze(i)
        return scale

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1,"must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim()-1))
        else:
            reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        self.xmin_tmp = xmin.detach()
        self.xmax_tmp = xmax.detach()
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def MaxMin_except_first_dim(self,tensor,func):
        # 获取张量的维度数
        dims = list(range(1, tensor.dim()))
        # 逐步在每个维度上取最大值
        for dim in dims:
            tensor, _ = func(tensor, dim=dim, keepdim=True)
        return tensor
    
    def dynamic_per_tensor_calibration(self,x):
        if not self.has_batch_dim:
            xmin = x.min()
            xmax = x.max()
        else:
            shape = [1] * len(x.shape)
            shape[0] = -1
            xmin = self.MaxMin_except_first_dim(x,torch.min).view(shape)
            xmax = self.MaxMin_except_first_dim(x,torch.max).view(shape)
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin,xmax)
        else:
            self.assymmetric_cal_scale(xmin,xmax)

    def symmetric_cal_scale(self,xmin,xmax):
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (2 ** (self.n_bits - 1) - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.round_zero_point = None
        
    def assymmetric_cal_scale(self,xmin,xmax):
        dynamic_range = xmax - xmin
        scale = dynamic_range / (2**self.n_bits - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None
        
    def scale_frexp(self):
        k = 16
        m = (self.scale*(2**k)).round()
        self.scale = m*(2**(-k))
        
        return self.scale

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point
        
    def quant2int(self, x):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / self.scale)
        if self.round_zero_point is not None:
            x_int = x_int.add(self.round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        
        if self.group_size:
            x_int = x_int.reshape(dim1, dim2)
        return x_int
    
    def dequant(self, x_int):
        if self.group_size:
            assert len(x_int.shape) == 2, "only support linear layer now"
            dim1, dim2 = x_int.shape
            x_int = x_int.reshape(-1, self.group_size)
            
        x_dequant = x_int
        if self.round_zero_point is not None:
            x_dequant = x_dequant.sub(self.round_zero_point)
        x_dequant = x_dequant.mul(self.scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = F.sigmoid(rescale_param) + 0.5
            x_dequant = x_dequant*self.rescale_param
        return x_dequant

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

POW_QUANTIZATION = False

def set_pow_quantization(value: bool):
    global POW_QUANTIZATION
    assert value in (True, False)
    POW_QUANTIZATION = value

def get_pow_quantization():
    global POW_QUANTIZATION
    return POW_QUANTIZATION

def analysis_dim(granularities: list or str or int): # type: ignore
    """解析granularity, 并翻译为具体的channel数值, -1表示per-tensor, list将提取具体的通道数组成list, dim开头提取其后的通道数.

    Args:
        granularities (str or list): tensor or dimx, or [dim0, dim1, ...]

    Returns:
        int or list: 通道id
    """
    ch_axis = None
    if isinstance(granularities, list):
        ch_axis = []
        for granularity in granularities:
            if isinstance(granularity, str):
                assert len(granularity) == 4 and granularity[:3] == "dim"
                ch_axis.append(int(granularity[3:]))
            elif isinstance(granularity, int):
                ch_axis.append(granularity)
            else:
                raise NotImplemented
        for ch in ch_axis:
            assert ch >= 0, "for stability"
    elif isinstance(granularities, int):
        if granularities == -1:
            return -1
        return [granularities]
    elif granularities == "tensor":
        ch_axis = -1
    elif granularities[:3] == "dim":
        ch_axis = [
            int(granularities[3:]),
        ]
    return ch_axis

class ObserverABC(abc.ABC, torch.nn.Module):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, granularity="tensor", min_limit=None, max_limit=None):
        super().__init__()
        self._granularity = None
        self._ch_axis = analysis_dim(granularity)
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)
        self.align_with_set: Set[ObserverABC] = set()
        self.granularity = granularity
        self.manager: ObserverABC = None
        self.dtype = None
        self.symmetric = True
        self.min_limit = min_limit
        self.max_limit = max_limit

    def align_with(self, *args: Iterable["ObserverABC"]):
        for arg in args:
            if arg is not self:
                self.align_with_set.add(arg)

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, value):
        self._granularity = value
        self._ch_axis = analysis_dim(value)
        self.clear()

    def clear(self):
        self.min_val.resize_(0).fill_(0)
        self.max_val.resize_(0).fill_(0)

    @property
    def observer_name(self):
        return type(self).__name__

    def _pre_load_state_dict_hook(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        min_val = state_dict.get(prefix + "min_val", None)
        max_val = state_dict.get(prefix + "max_val", None)
        if min_val is not None:
            self.min_val.resize_(min_val.shape).copy_(min_val)
        if max_val is not None:
            self.max_val.resize_(max_val.shape).copy_(max_val)

    @property
    def ch_axis(self):
        return self._ch_axis
    
    @ch_axis.setter
    def ch_axis(self, new_ch_axis):
        self._ch_axis = new_ch_axis

    @torch.no_grad()
    def calculate_scale_zero_point(self, dtype, symmetric=True):
        self.symmetric = symmetric
        min_val, max_val = self.cal_min_max()
        if len(self.align_with_set):
            for to_align in self.align_with_set:
                if to_align.observer_name == 'FixedObserver':
                    _min_val, _max_val = to_align.cal_min_max()
                    if (_min_val.numel() + _max_val.numel() ) < 2 :
                        continue
                
                to_align.symmetric = symmetric
                to_align.dtype = dtype
                _min_val, _max_val = to_align.cal_min_max()
                min_val = torch.min(min_val, _min_val)
                max_val = torch.max(max_val, _max_val)
                # except:
                    # logger.error("observer align error")

        assert min_val is not None and max_val is not None
        quant_min, quant_max = dtype.qmin, dtype.qmax
        device = min_val.device
        scale = torch.ones(min_val.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int32, device=device)

        if symmetric:
            scale = torch.max(
                torch.abs(min_val / quant_min),
                torch.abs(max_val / quant_max),
            )
            scale = torch.max(scale, eps.to(scale.device))
            if POW_QUANTIZATION:
                scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
        else:
            scale = ((max_val - min_val) / float(quant_max - quant_min)).abs()
            scale = torch.max(scale, eps.to(scale.device))
            if POW_QUANTIZATION:
                scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
            zero_point = quant_min - torch.round(min_val / scale).to(torch.int32)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    @torch.no_grad()
    def calculate_qparams(self, dtype, symmetric=True):
        from .quant_param import QuantParam
        if self.manager is not None:
            return self.manager.calculate_qparams(dtype, symmetric)
        scale, zero_point = self.calculate_scale_zero_point(dtype, symmetric)
        grans = ""
        if isinstance(self.granularity, list):
            for gran in self.granularity:
                grans += gran
                grans += ","
            grans = grans[:-1]
        else:
            grans = self.granularity
        quant_param = QuantParam(
            dtype=f"int{dtype.bitwidth}",
            scale=scale,
            zero_point=zero_point,
            granularity=grans,
        )
        frame = sys._getframe()
        pre_frame = frame.f_back
        file_name = pre_frame.f_code.co_filename
        file_no = pre_frame.f_lineno
        self.quant_param = quant_param
        return self.quant_param

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.max_val.device != x.device:
            self.to(x.device)
        if self.manager is not None:
            self.manager.update(x)
        else:
            if x.dim() == 1 and self.ch_axis != -1:
                x = x.reshape(-1, 1)
            self._update_(x)

    def cal_min_max(self):
        min_val, max_val = self._cal_min_max_()
        if self.min_limit is not None:
            min_val.clip_(min=self.min_limit)
            max_val.clip_(min=self.min_limit)
        if self.max_limit is not None:
            max_val.clip_(max=self.max_limit)
            max_val.clip_(max=self.max_limit)
        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.zeros_like(max_val))
        if dist.is_initialized():
            dist.all_reduce(min_val, op=ReduceOp.MIN)
            dist.all_reduce(max_val, op=ReduceOp.MAX)
        return min_val, max_val

    def forward(self, x: torch.Tensor):
        self.update(x)
        return x

    # ********************必须要实现的抽象方法************************#
    @abc.abstractmethod
    def _cal_min_max_(self):
        return self.min_val, self.max_val

    @abc.abstractmethod
    def _update_(self, tensor: torch.Tensor):
        pass

    # ************************END**********************************#

class PercentileObserver(ObserverABC):
    ch_shapes: list

    def __init__(
        self,
        granularity="tensor",
        hist_bin_num: int = 2048,
        percent: Union[float, list] = 1.0,
        percentile_mode: str = "line",
        min_limit = None,
        max_limit = None
    ):
        super().__init__(granularity,min_limit,max_limit)
        if isinstance(percent, float):
            self.left_percent = self.right_percent = percent
        else:
            self.left_percent, self.right_percent = percent
        self.percentile_mode = percentile_mode
        self.hist_manager = HistManager(num_bins=hist_bin_num)
        self.ch_shapes = 1

    def _update_(self, tensor: torch.Tensor):
        hist_manager = self.hist_manager
        if self.ch_axis == -1:
            x = tensor.contiguous().view(1, -1)
        elif isinstance(self.ch_axis, list):
            self.ch_shapes = [tensor.shape[i] for i in self.ch_axis]
            dims = list(range(tensor.dim()))  # self.ch_shapes =
            permute_dims = deepcopy(self.ch_axis)
            for dim in dims:
                if dim not in permute_dims:
                    permute_dims.append(dim)
            x = tensor.permute(permute_dims)
            x = x.reshape(int(np.prod(self.ch_shapes)), -1)  # (#channels, -1)
        else:
            raise NotImplementedError("ch axis must be int or list.")
        hist_manager.collect(data=x)

    def percentile(self):
        if self.left_percent >= 1.0 or self.right_percent >= 1.0:
            assert (
                self.percentile_mode == "line"
            ), "If percent is 1.0, must use line for no loss."
        min_clip_tensor, max_clip_tensor = self.hist_manager.percentile(
            left_percent=self.left_percent,
            right_percent=self.right_percent,
            mode=self.percentile_mode,
        )
        return min_clip_tensor, max_clip_tensor

    def _cal_min_max_(self):
        min_val, max_val = self.percentile()
        min_val = min_val.reshape(self.ch_shapes)
        max_val = max_val.reshape(self.ch_shapes)
        self.min_val.resize_(min_val.shape).copy_(min_val)
        self.max_val.resize_(max_val.shape).copy_(max_val)
        return self.min_val, self.max_val

class MinMaxObserver(ObserverABC):
    def _cal_min_max_(self):
        return super()._cal_min_max_()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))
        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
        max_val = torch.amax(x, dim=dims, keepdim=False)
        min_val = torch.amin(x, dim=dims, keepdim=False)
        if max_val.dim() == 0 or min_val.dim() == 0:
            assert max_val.dim() == min_val.dim()
            max_val = max_val.reshape(-1)
            min_val = min_val.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val.shape).fill_(0)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val.shape).fill_(0)
        self.max_val.data.copy_(torch.max(self.max_val, max_val))
        self.min_val.data.copy_(torch.min(self.min_val, min_val))

class HistManager:
    def __init__(self, num_bins: int) -> None:
        """初始化直方图管理器. (用于Observer的update方法.)

        Args:
            num_bins (int): 直方图的bin个数.
        """
        self.hists_mat = None
        self.bin_edges_mat = None
        self.num_bins = num_bins
        self.left_bins_num = 0
        self.right_bins_num = 0

    def clear(self):
        self.hists_mat = None
        self.bin_edges_mat = None
        self.num_bins = num_bins
        self.left_bins_num = 0
        self.right_bins_num = 0

    def collect(self, data: torch.Tensor) -> list:
        """根据data更新得到一个最新的直方图.

        Args:
            data (torch.Tensor): 新进入的tensor数据. 一定要保证传入的data应该是根据设定的dims已经做过flatten了. shape=(#channel, #elements).

        Returns:
            list: self.hists_mat, self.bin_edges_mat
        """
        assert data.dim() == 2, "batch hist_manager only support dim=2"
        data_range_perchannel = data.abs().amax(dim=-1, keepdim=True)  # (B,1)
        if self.hists_mat is None:
            self.bin_width = data_range_perchannel / (
                self.num_bins // 2
            )  # init bin_widht : (B,1)
            self.bin_width.clip_(min=eps)
            # 初始化中心0bin.
            self.hists_mat = torch.zeros(
                size=(data.shape[0], 1), device=data.device
            )  # init hists_mat : (B,1)
            self.bin_edges_mat = (
                torch.tensor([-0.5, 0.5], device=data.device).repeat(data.shape[0], 1)
                * self.bin_width
            )
        B, N = self.hists_mat.shape
        normalized_data = data.float() / self.bin_width  # align measurement-unit
        max_bound = max(
            normalized_data.abs().amax().item(), self.hists_mat.shape[-1] / 2
        )  # (1,)
        edge_bound = math.ceil(max_bound + 0.5) - 0.5  # (1,)
        num_bins = round(edge_bound * 2)
        shift_vec = torch.arange(
            start=0, end=data.shape[0], step=1, device=data.device
        ).unsqueeze(1) * (edge_bound * 2)
        normalized_data = normalized_data + shift_vec
        hists_vec = torch.histc(
            input=normalized_data,
            bins=num_bins * B,
            min=-edge_bound,
            max=(2 * B - 1) * edge_bound,
        )
        hists_mat = hists_vec.reshape(B, -1)  # (B,Nnew)
        start_idx = int((num_bins - N) / 2)
        hists_mat[:, start_idx : start_idx + N] += self.hists_mat
        """update self.hists_mat & self.bin_edges_mat"""
        self.hists_mat = hists_mat
        self.bin_edges_mat = (
            torch.arange(
                -edge_bound, edge_bound + 0.5, step=1, device=data.device
            ).repeat(B, 1)
            * self.bin_width
        )
        self.left_bins_num = self.right_bins_num = (num_bins + 1) // 2
        """"""
        return [self.hists_mat, self.bin_edges_mat]

    def percentile(
        self, left_percent: float, right_percent: float, mode: str = "center"
    ):
        """根据参数对直方图的频度进行percentile操作.
        注意, 是对两边分别percentile同样的percent.
        这个method具体参考了PaddleSlim/paddleslim/quant/observers/hist.py中的PercentHistObserverLayer.

        Args:
            left_percent (float): 对头部做截断.保留前percent的bin, percent属于[0., 1.]
            right_percent (float): 对尾部做截断.保留前percent的bin, percent属于[0., 1.]
            mode (str, optional): 以percent落在的bin的中线为界(center)还是线性插值(line). Defaults to "center".

        Returns:
            torch.Tensor, torch.Tensor: 两个tensor, 表示clip的min和max界限.
        """
        assert (0.0 <= left_percent <= 1.0) and (
            0.0 <= right_percent <= 1.0
        ), "Error Percent setting."
        assert mode in ["center", "line"], "Only support center or line."
        if mode == "center":
            centers = (
                self.bin_edges_mat[..., 1:] + self.bin_edges_mat[..., :-1]
            ) / 2  # 算出每个bin的中线.
            min_clip_bound = centers[..., 0]
            max_clip_bound = centers[..., -1]
            # 从左向右, 给出截掉max端的percent的clip_value.
            max_clip_bound = percentile_center_onedirection(
                hists_mat=self.hists_mat[..., (self.left_bins_num - 1) :],
                centers=centers[..., (self.left_bins_num - 1) :],
                percent=right_percent,
            )
            # 从右向左, 给出截掉min端的percent的clip_value.
            min_clip_bound = percentile_center_onedirection(
                hists_mat=torch.flip(
                    self.hists_mat[..., : self.left_bins_num], dims=(-1,)
                ),
                centers=torch.flip(centers[..., : self.left_bins_num], dims=(-1,)),
                percent=left_percent,
            )
            return min_clip_bound, max_clip_bound
        else:
            min_clip_bound = self.bin_edges_mat[..., 0]
            max_clip_bound = self.bin_edges_mat[..., -1]
            # 从左向右, 给出截掉max端的percent的clip_value.
            max_clip_bound = percentile_linear_onedirection(
                hists_mat=self.hists_mat[..., (self.left_bins_num - 1) :],
                bin_edges_mat=self.bin_edges_mat[..., (self.left_bins_num - 1) :],
                percent=right_percent,
            )
            min_clip_bound = percentile_linear_onedirection(
                hists_mat=torch.flip(
                    self.hists_mat[..., : self.left_bins_num], dims=(-1,)
                ),
                bin_edges_mat=torch.flip(
                    self.bin_edges_mat[..., : (self.left_bins_num + 1)], dims=(-1,)
                ),
                percent=left_percent,
            )
            return min_clip_bound, max_clip_bound

    def find_lowest_kl_bound(self, bit_num: int, iter_times: int = 512):
        """搜索出KL最小的min_bound, max_bound.
        1. 以0.所在bin为起点, 分别向左向右搜索能够使得KL散度最小的thresholds.
        2. 两个thresholds拼起来作为min_bound, max_bound.

        Args:
            bit_num (int): 量化位数.
            stride (int): 搜索步长.
        Returns:
            torch.Tensor, torch.Tensor: 搜索得到的min_bound和max_bound.
        """
        # 找到搜索的起点bin.
        # 由于搜集方式的原因, 我们一定有0.在某个bin的中线, 这个bin的左侧是负edge, 右侧是正edge.
        B, N = self.bin_edges_mat.shape
        _, N_bins = self.hists_mat.shape
        device = self.hists_mat.device
        assert N - 1 == N_bins, "must"
        post_bin_edges = self.bin_edges_mat[:, N // 2 :]
        post_hists = self.hists_mat[:, N_bins // 2 :]
        neg_bin_edges = self.bin_edges_mat[:, : N // 2]
        neg_hists = self.hists_mat[:, : N_bins // 2]

        left_threshold = torch.zeros((B,), device=device)
        right_threshold = torch.zeros((B,), device=device)
        stride = max(1, N // iter_times)
        for i in range(B):
            left_threshold[i] = -get_kl_threshold_onedirection(
                bit_num,
                stride,
                torch.flip(neg_hists[i], [0]),
                -torch.flip(neg_bin_edges[i], [0]),
            )
            right_threshold[i] = get_kl_threshold_onedirection(
                bit_num, stride, post_hists[i], post_bin_edges[i]
            )
        return left_threshold, right_threshold

eps = torch.finfo(torch.float32).eps * 10
def percentile_linear_onedirection(
    hists_mat: torch.Tensor, bin_edges_mat: torch.Tensor, percent: float):
    """根据参数对直方图的频度进行linear模式(线性插值)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和bin_edges进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图.
        bin_edges_mat (torch.Tensor): 直方图各个bin的edges.
        percent (float): 保留前percent的bin, percent属于[0., 1.]

    Returns:
        torch.Tensor: 只有一个值的tensor, 表示尾端的clip界限.
    """
    assert hists_mat.dim() == 2 and bin_edges_mat.dim() == 2, "must"
    hists_sum = hists_mat.sum(dim=-1, keepdim=True)  # (B, 1)
    hists_cum = hists_mat.cumsum(dim=-1)  # (B,N)
    target = percent * hists_sum  # (B,1)
    idx = ((hists_cum - target) >= 0).int().argmax(dim=-1)  # first idx that >= target
    r_csum = hists_cum.gather(-1, idx.reshape(-1, 1))
    l_csum = hists_cum.gather(-1, (idx - 1).clip_(0).reshape(-1, 1))
    p = (r_csum - target) / (r_csum - l_csum + eps)
    p = p.clip(0, 1)
    return (
        bin_edges_mat.gather(-1, idx.reshape(-1, 1)) * p
        + bin_edges_mat.gather(-1, (idx + 1).reshape(-1, 1)) * (1 - p)
    ).reshape(-1)

def percentile_center_onedirection(
    hists_mat: torch.Tensor, centers: torch.Tensor, percent: float):
    """根据参数对直方图的频度进行center模式(中线)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和centers进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图构成的矩阵.
        centers (torch.Tensor): 直方图各个bin的中线.
        percent (float): 保留前percent的bin, percent属于[0., 1.]
    Returns:
        torch.Tensor: 各个通道最小值组成的tensor, 表示尾端的clip界限.
    """
    assert hists_mat.dim() == 2 and centers.dim() == 2, "must"
    hists_sum = hists_mat.sum(dim=-1, keepdim=True)  # (B, 1)
    hists_cum = hists_mat.cumsum(dim=-1)  # (B,N)
    target = percent * hists_sum  # (B,1)
    idx = (hists_cum - target).abs().argmin(dim=-1)  # distance:(B,), idx(B,)
    target_centers = centers.gather(dim=-1, index=idx.reshape(-1, 1))
    return target_centers.reshape(-1)

@torch.no_grad()
def get_kl_threshold_onedirection(
    bit_num: int, stride: int, hist: torch.Tensor, edges: torch.Tensor):
    """只有正半轴有bin的hist进行KL最低的threshold搜索.

    Args:
        bit_num (int): 要量化的bit数. 这个是按照对称量化考虑的, 所以如果正半轴量化到127个bin, 这里应该给8.
        stride (int): 搜索步长.
        hist (torch.Tensor): 只有正半轴有值的hist.
        edges (torch.Tensor): hist对应的edges, 如果一个edge为负, 其余edges为正.

    Returns:
        torch.Tensor: threshold
    """
    quant_range = 2 ** (bit_num - 1) - 1
    start = quant_range
    if hist.numel() > 0 and hist.shape[0] > start:
        ret_device = hist.device
        edges = edges.to(ret_device)
        bin_width = edges[-1] - edges[-2]
        n_hist = hist.shape[0]
        hist = hist.clone()
        hist[: int(n_hist * 0.001)] = 0  # optional ,让前面几个值为0
        losses = list()
        min_loss = torch.inf
        ret = edges[-1] - bin_width / 2
        for i in range(start, n_hist + 1, stride):
            ref = torch.clone(hist[:i])
            ref[-1] += torch.sum(hist[i:])
            space = torch.linspace(
                edges[0], bin_width * i, quant_range + 1, device=ret_device
            )
            hb2space = (
                torch.bucketize(bin_width / 2 + edges[:i], space)[None, :] - 1
            )  # (1,i)
            to_judge = torch.arange(quant_range, device=ret_device)[:, None]  # (127,1)
            mask = (hb2space == to_judge) & ((hist[:i] != 0))[None, :]
            values = hist[:i][None, :].repeat(quant_range, 1)  # (127,i)
            values[~mask] = 0
            sum_histv_perbin = torch.sum(values, dim=-1, keepdim=True)
            sum_hist_perbin = torch.sum(mask, -1, keepdim=True)
            sum_hist_perbin[sum_histv_perbin == 0] = 1
            mean = (sum_histv_perbin / sum_hist_perbin).repeat(1, i)  # (127,i)
            mean[~mask] = 0
            cand = torch.sum(mean, 0)  # (,i)
            loss = torch_kl_stable(cand, ref)
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                ret = edges[0] + bin_width * (i - 0.5)
        return ret
    elif hist.numel() > 0:
        logger.warning(f"The amount of collected data is too small.")
        return edges[-1]
    else:
        raise ValueError("Histogram is empty!")

def torch_kl_stable(pred: torch.Tensor, ref: torch.Tensor):
    """计算伪量化后的hist与量化前浮点直方图之间的KL散度.

    Args:
        pred (torch.Tensor): 伪量化后的hist.
        ref (torch.Tensor): 量化前的hist.

    Returns:
        torch.Tensor: KL散度.
    """
    mask = ref != 0
    if pred[-1] == 0:  # for numerical stability
        pred[-1] = 1
    pred = pred.to(torch.float)[mask]
    ref = ref.to(torch.float)[mask]
    psum = pred.sum()
    rsum = ref.sum()
    p_sum = (ref * torch.log(psum * pred)).sum()
    r_sum = (ref * torch.log(rsum * ref)).sum()
    return (r_sum - p_sum) / rsum

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

def get_had45():
    # 定义一个 5x5 的 Hadamard 矩阵
    had5 = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1],
        [1, -1, -1, 1, 1],
        [1, 1, 1, 1, -1]
    ], dtype=torch.float32)

    # 定义一个 9x9 的 Hadamard 矩阵
    had9 = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1]
    ], dtype=torch.float32)

    # 使用 Kronecker 积生成 45x45 的矩阵
    had45 = torch.kron(had5, had9)

    return had45
    
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    model, test_loaders = test_pipeline(root_path)

    