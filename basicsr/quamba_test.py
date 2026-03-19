
import logging
import torch
from os import path as osp
import sys
# for some possible IMPORT ERROR
# sys.path.append('/data1/guohang/MambaIR-main')
# sys.path.append('/cluster/home/yujichen/MambaIR')
sys.path.append('/leonardo_work/IscrB_FM-EEG24/ychen004/QuantIR_IMPROTANT')
from basicsr.archs.quamba_arch import QLinear, QConv2D, QAct, VSSBlock
from basicsr.archs.quamba_arch import QSS2D, QCAB, HadamardTransform, SmoothModule
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import fast_hadamard_transform
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
        scales_tensor = torch.tensor(scales)
        scales_tensor = scales_tensor.clone().detach()
        base = torch.zeros_like(scales_tensor)
    else:
        assert w_min is not None, "w_min should not be None for asymmetric quantization."
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    return scales, base

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

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

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
            W_ = hadamard_transform(W_.reshape(-1, n//had_dim, had_dim), scale=1/math.sqrt(had_dim)).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)

def rotate_out_proj(model, logger):
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            if "out_proj" in name:
                apply_exact_had_to_linear(m, had_dim=-1, output=False)
    logger.info(f"########### Apply Hadamard Weight ###########")

def activate_rotate_module(model, logger):
    for name, m in model.named_modules():
        if isinstance(m, (HadamardTransform)):
            m.configure(do_rotate=True)
    logger.info(f"########### Apply Hadamard Act ###########")

def configure_act_quant(model, act_scales, logger, n_bits):
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
    logger.info(f"########### Set Act to %d bit quant ###########" % n_bits)

def activate_quant_module(model):
    for name, m in model.named_modules():
        if isinstance(m, (QAct, QLinear, QConv2D)):
            m.is_quant_mode = True

def configure_weight_quant(model, logger, n_bits):
    for name, m in model.named_modules():
        if isinstance(m, (QLinear, QConv2D)):
            m.configure(
                n_bits=n_bits,
            )
    logger.info(f"########### Set Weight to %d bit quant ###########" % n_bits)

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

def prepare_act_scales(model, logger, n_bits, test_loader, num_samples):

    smooth_scales = {}
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
        if isinstance(m, SmoothModule):
            hooks.append(
                m.register_forward_hook(partial(stat_smooth_hook, name=name))
            )

    # len(test_loader)//10
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

    return act_scales, smooth_scales

def quantize_model_mamba(model, logger, n_bits, act_scales, smooth_scales):

    # smooth_mamba(model, smooth_scales, logger)
    # rotate_out_proj(model, logger)
    # activate_rotate_module(model, logger)
    configure_weight_quant(model, logger, n_bits)
    configure_act_quant(model, act_scales, logger, n_bits)
    activate_quant_module(model)

    return model

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
    n_bits = opt['network_g']['k_bits']
    model = build_model(opt)
    model.eval()
    model = prepare_quantize_model_mamba(model, logger)
    q_model = model
    logger.info(f"########### Start Quantizing Model ###########")
    act_scales, smooth_scales = prepare_act_scales(model, logger, n_bits, act_loader, num_samples=1)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        q_model = quantize_model_mamba(q_model, logger, n_bits, act_scales, smooth_scales)
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

    