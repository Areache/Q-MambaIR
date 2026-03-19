import logging
import torch
from os import path as osp
import sys
# for some possible IMPORT ERROR
# sys.path.append('/data1/guohang/MambaIR-main')
# sys.path.append('/cluster/home/yujichen/MambaIR')
sys.path.append('/leonardo/home/userexternal/ychen004/QuantIR')
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

# /leonardo/home/userexternal/ychen004/anaconda3/envs/mambair/bin/python /leonardo/home/userexternal/ychen004/QuantIR/basicsr/onnx.py -opt /leonardo/home/userexternal/ychen004/QuantIR/options/test/DN/test_MambaIR_ColorDN_level25_ours_2b.yml
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

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)
    
    # import pdb;pdb.set_trace()
    # create model
    model = build_model(opt)
    model = model.net_g
    # for test_loader in test_loaders:
    #     test_set_name = test_loader.dataset.opt['name']
    #     logger.info(f'Testing {test_set_name}...')
    #     model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
    onnx_file_name = "Qmambair.onnx"
    # 假设 test_loader 是你需要的 DataLoader 对象
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # import pdb;pdb.set_trace()
    # for batch in test_loader:
    #     # 取出模型的输入部分，可能是一个 dict，也可能是 tuple，具体取决于你的数据集定义
    #     if isinstance(batch, dict):
    #         dummy_input = batch['lq'].to(device)  # 例如图像在 'lq' 键中，放到对应设备上
    #         import pdb;pdb.set_trace()
    #         print(dummy_input.shape)
    #     elif isinstance(batch, (tuple, list)):
    #         dummy_input = batch[0].to(device)
    #     else:
    #         dummy_input = batch.to(device)
    #     break  # 只需要一个 batch 就够了

    logger.info(f'....start export onnx....')
    dummy_input = torch.randn(1, 3, 64, 64, requires_grad=True).to(device) 
    # dummy_input = torch.random(1, 3, 64, 64).to(device)
    torch.onnx.export(model,        # 模型的名称
                    dummy_input,   # 一组实例化输入
                    onnx_file_name,   # 文件保存路径/名称
                    export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                    opset_version=11,          # ONNX 算子集的版本，当前已更新到15
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names = ['input'],   # 输入模型的张量的名称
                    output_names = ['output'], # 输出模型的张量的名称
                    # dynamic_axes将batch_size的维度指定为动态，
                    # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}})

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    model, test_loaders = test_pipeline(root_path)

    