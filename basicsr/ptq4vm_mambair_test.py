import logging
import torch
from os import path as osp
import sys

# Ensure QuantIR root is on path
sys.path.append('/leonardo_work/IscrB_FM-EEG24/ychen004/QuantIR_IMPROTANT')

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import parse_options

# Import QuantLinear/QuantConv2d from mambaquant_test (they are generic wrappers)
# These use UniformAffineQuantizer which is compatible with ptq4vm-style quantization
from basicsr.mambaquant_test import QuantLinear, QuantConv2d, set_pow_quantization


def replace_with_ptq4vm_quant(model, w_bits: int = 4, a_bits: int = 4, logger=None):
    """
    Replace nn.Linear / nn.Conv2d layers in MambaIR backbone with ptq4vm's
    QuantLinear / QuantConv2d.

    This is a generic wrapper: it keeps weights/bias and attaches UniformAffineQuantizer
    from ptq4vm for both weights and activations.
    """
    import torch.nn as nn

    weight_quant_params = {
        "n_bits": w_bits,
        "symmetric": True,
        "per_channel_axes": [0],
        "dynamic": False,
        "dynamic_method": "per_channel",
        # Note: is_weight, shape, and observe are passed explicitly by QuantLinear/QuantConv2d
    }
    act_quant_params = {
        "n_bits": a_bits,
        "symmetric": False,
        "dynamic": True,
        "dynamic_method": "per_token",
        # Note: is_weight, has_batch_dim, and observe are passed explicitly by QuantLinear/QuantConv2d
    }

    # Enable power-of-two scaling if desired
    set_pow_quantization(False)

    def _should_skip(name: str) -> bool:
        # Keep first/last convolutions and upsampler in full precision
        skip_keywords = [
            "conv_first",
            "conv_after_body",
            "conv_before_upsample",
            "conv_last",
            "upsample",
        ]
        for k in skip_keywords:
            if k in name:
                return True
        return False

    replaced_linear = 0
    replaced_conv = 0

    for name, module in list(model.named_modules()):
        # Only touch modules that are direct attributes of their parent
        parent = model
        sub_name = name
        if "." in name:
            *parents, sub_name = name.split(".")
            parent = model
            for p in parents:
                parent = getattr(parent, p)

        if isinstance(module, nn.Linear) and not _should_skip(name):
            q_mod = QuantLinear(
                org_module=module,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                disable_input_quant=False,
                observe="percentile",
            )
            setattr(parent, sub_name, q_mod)
            replaced_linear += 1
        elif isinstance(module, nn.Conv2d) and not _should_skip(name):
            q_mod = QuantConv2d(
                org_module=module,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                disable_input_quant=False,
                observe="percentile",
            )
            setattr(parent, sub_name, q_mod)
            replaced_conv += 1

    if logger is not None:
        logger.info(
            f"########### ptq4vm wrappers inserted (Linear: {replaced_linear}, Conv2d: {replaced_conv}) "
            f"with W{w_bits}A{a_bits} ###########"
        )


def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    """
    Mirror helper from ptq4vm.quantizer: recursively enable / disable
    quantization flags on QuantLinear / QuantConv2d modules.
    """
    model.use_weight_quant = weight_quant
    model.use_act_quant = act_quant
    for _, m in model.named_modules():
        if isinstance(m, (QuantLinear, QuantConv2d)):
            m.set_quant_state(weight_quant, act_quant)


def test_pipeline(root_path):
    # parse options
    opt, _ = parse_options(root_path, is_train=False)
    print('############## We are testing %s with ptq4vm ! ##############' % opt['name'])

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    # create test datasets and loaders
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # bits configuration from yaml (fallback to 4/4)
    net_opt = opt.get('network_g', {})
    w_bits = int(net_opt.get('w_bits', net_opt.get('k_bits', 4)))
    a_bits = int(net_opt.get('a_bits', w_bits))

    logger.info(f"########### ptq4vm Quantization Config: W{w_bits}A{a_bits} ###########")

    # build model
    model = build_model(opt)
    model.eval()

    # insert ptq4vm quant wrappers into backbone
    replace_with_ptq4vm_quant(model.net_g, w_bits=w_bits, a_bits=a_bits, logger=logger)
    set_quant_state(model.net_g, weight_quant=True, act_quant=True)

    # run validation
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name} with ptq4vm quantization...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)


