import datetime
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# from tensorboardX import SummaryWriter
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options, parse_options_teacher


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    
    def partial_unfreeze(model):
        if opt["dist"] == True:
            module = model.net_g.module
        else:
            module = model.net_g
        unfreeze_model(module.upsample)
        for layer in module.layers:
            block_num = len(layer.residual_group.blocks)
            block_index = 1
            for block in layer.residual_group.blocks:
                if block_index ==  block_num-1:
                    unfreeze_model(block.self_attention)
                block_index += 1
        if opt["dist"] == True:
            model.net_g.module = module
        else:
            model.net_g = module

    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    teacher_opt, teacher_args = parse_options_teacher(root_path, is_train=True)
    master_opt, master_args = parse_options_teacher(root_path, is_train=True)
    opt['root_path'] = root_path
    opt['find_unused_parameters'] = True
    master_opt['find_unused_parameters'] = True
    # writer = SummaryWriter(log_dir="/cluster/home/yujichen/QuantIR/logs/runs") 

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None and opt['rank'] == 0:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # import pdb;pdb.set_trace()
    # create model
    model = build_model(opt)
    teacher_model = build_model(teacher_opt)
    master_model = build_model(master_opt)
    # freeze model
    freeze_model(teacher_model)
    freeze_model(master_model)
    
    teacher_model.schedulers = model.schedulers
    teacher_model.optimizer_g = model.optimizer_g
    teacher_model.optimizers = model.optimizers
    master_model.schedulers = model.schedulers
    master_model.optimizer_g = model.optimizer_g
    master_model.optimizers = model.optimizers

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")


    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            
            lq = train_data['lq'].to(model.device)
            gt = train_data['gt'].to(model.device)

            # if current_iter <= total_iters//2:
            #     # switch model state
            #     unfreeze_model(teacher_model)
            #     teacher_model.train()
            #     # update learning rate
            #     model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            #     teacher_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            #     # training
            #     model.optimizer_g.zero_grad()
            #     teacher_model.optimizer_g.zero_grad()
            #     model.output = model.net_g(lq)
            #     teacher_model.output = teacher_model.net_g(lq)
            #     # loss
            #     loss_total = 0
            #     loss_student_g = model.cri_pix(model.output, gt)
            #     loss_teacher_g = model.cri_pix(teacher_model.output, gt)
            #     loss_k = model.cri_pix(teacher_model.output, model.output)
            #     loss_total = 0.25*loss_student_g + 0.25*loss_teacher_g + 0.5*loss_k
            #     loss_total.backward()
            #     model.optimizer_g.step()
            #     teacher_model.optimizer_g.step()

            #     # log
            #     if current_iter % opt['logger']['print_freq'] == 0:
            #         log_vars = {'epoch': epoch, 'iter': current_iter}
            #         log_vars.update({'lrs': model.get_current_learning_rate()})
            #         log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            #         log_vars.update({'loss_student_g':loss_student_g})
            #         log_vars.update({'loss_teacher_g':loss_teacher_g})
            #         log_vars.update({'loss_k':loss_k})
            #         log_vars.update({'lrs_student': model.get_current_learning_rate()[0]})
            #         log_vars.update({'lrs_teacher': teacher_model.get_current_learning_rate()[0]})
            #         msg_logger(log_vars)
            # elif current_iter >= total_iters//3 and current_iter <= 2*(total_iters//3):
            #     # switch model state
            #     freeze_model(teacher_model)
            #     teacher_model.eval()
            #     # update learning rate
            #     model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            #     # training
            #     model.optimizer_g.zero_grad()
            #     model.output = model.net_g(lq)
            #     with torch.no_grad():
            #         teacher_model.output = teacher_model.net_g(lq)
            #     # loss
            #     # loss_student_g = model.cri_pix(model.output, gt)
            #     loss_k = model.cri_pix(model.output, teacher_model.output)
            #     loss_total = loss_k
            #     loss_total.backward()
            #     model.optimizer_g.step()
            #     # log
            #     if current_iter % opt['logger']['print_freq'] == 0:
            #         log_vars = {'epoch': epoch, 'iter': current_iter}
            #         log_vars.update({'lrs': model.get_current_learning_rate()})
            #         log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            #         log_vars.update({'loss_total':loss_total})
            #         log_vars.update({'lrs_k': model.get_current_learning_rate()[0]})
            #         msg_logger(log_vars)                
            # else:
            #     # switch model state
            #     # freeze_model(teacher_model)
            #     # teacher_model.eval()
            #     # update learning rate
            #     model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            #     # training
            #     model.optimizer_g.zero_grad()
            #     model.output = model.net_g(lq)
            #     # with torch.no_grad():
            #     #     teacher_model.output = teacher_model.net_g(lq)
            #     # loss
            #     loss_student_g = model.cri_pix(model.output, gt)
            #     # loss_k = model.cri_pix(model.output, teacher_model.output)
            #     loss_total = loss_student_g
            #     loss_total.backward()
            #     model.optimizer_g.step()
            #     # log
            #     if current_iter % opt['logger']['print_freq'] == 0:
            #         log_vars = {'epoch': epoch, 'iter': current_iter}
            #         log_vars.update({'lrs': model.get_current_learning_rate()})
            #         log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            #         log_vars.update({'loss_total':loss_total})
            #         log_vars.update({'lrs_k': model.get_current_learning_rate()[0]})
            #         msg_logger(log_vars)  

            if current_iter <= total_iters//3:
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                model.optimizer_g.zero_grad()
                model.output = model.net_g(lq)
                # loss
                loss_quant = model.cri_pix(model.output, gt)
                loss_quant.backward()
                model.optimizer_g.step()
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update({'loss_quant':loss_quant})
                    log_vars.update({'lrs_quant': model.get_current_learning_rate()[0]})
                    msg_logger(log_vars)
            # elif current_iter >= total_iters//3 and current_iter <= 2*(total_iters//3):
            elif current_iter >= total_iters//4 and current_iter <= 2*(total_iters//4):
                # switch model state
                unfreeze_model(teacher_model)
                teacher_model.train()
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                teacher_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                model.optimizer_g.zero_grad()
                teacher_model.optimizer_g.zero_grad()
                model.output = model.net_g(lq)
                teacher_model.output = teacher_model.net_g(lq)
                # loss
                loss_total = 0
                loss_student_g = model.cri_pix(model.output, gt)
                loss_teacher_g = model.cri_pix(teacher_model.output, gt)
                loss_k = model.cri_pix(teacher_model.output, model.output)
                loss_total = 0.25*loss_student_g + 0.25*loss_teacher_g + 0.5*loss_k
                loss_total.backward()
                model.optimizer_g.step()
                teacher_model.optimizer_g.step()

                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update({'loss_student_g':loss_student_g})
                    log_vars.update({'loss_teacher_g':loss_teacher_g})
                    log_vars.update({'loss_k':loss_k})
                    log_vars.update({'lrs_student': model.get_current_learning_rate()[0]})
                    log_vars.update({'lrs_teacher': teacher_model.get_current_learning_rate()[0]})
                    msg_logger(log_vars)
            # else:
            elif current_iter >= 2*(total_iters//4) and current_iter <= 3*(total_iters//4):
                # switch model state
                freeze_model(teacher_model)
                teacher_model.eval()
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                model.optimizer_g.zero_grad()
                model.output = model.net_g(lq)
                with torch.no_grad():
                    teacher_model.output = teacher_model.net_g(lq)
                # loss
                loss_k = model.cri_pix(model.output, teacher_model.output)
                loss_k.backward()
                model.optimizer_g.step()
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update({'loss_k':loss_k})
                    log_vars.update({'lrs_k': model.get_current_learning_rate()[0]})
                    msg_logger(log_vars)
            # finetuning
            else:
                freeze_model(model)
                partial_unfreeze(master_model)
                partial_unfreeze(model)
                # update learning rate
                master_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                master_model.optimizer_g.zero_grad()
                model.optimizer_g.zero_grad()
                master_model.output = master_model.net_g(lq)
                model.output = model.net_g(lq)
                loss_total = 0
                loss_master = master_model.cri_pix(master_model.output, gt)
                loss_student = model.cri_pix(model.output, master_model.output)
                loss_total = 0.5*loss_master + 0.5*loss_student
                loss_total.backward()
                master_model.optimizer_g.step()
                model.optimizer_g.step()

                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update({'loss_student':loss_student})
                    log_vars.update({'loss_master':loss_master})
                    log_vars.update({'lrs_student': model.get_current_learning_rate()[0]})
                    log_vars.update({'lrs_master': master_model.get_current_learning_rate()[0]})
                    msg_logger(log_vars)

            #     # # switch model state
            #     # freeze_model(master_model)
            #     # master_model.eval()
            #     # # training
            #     # with torch.no_grad():
            #     #     master_model.output = master_model.net_g(lq)

            #     # # partial_unfreeze(model)
            #     # model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            #     # model.optimizer_g.zero_grad()
            #     # model.output = model.net_g(lq)
            #     # # loss
            #     # loss_t = 0
            #     # loss_student_gt = model.cri_pix(model.output, gt)
            #     # loss_k = model.cri_pix(model.output, master_model.output)
            #     # loss_t = 0.5*loss_student_gt + 0.5*loss_k
            #     # loss_t.backward()
            #     # model.optimizer_g.step()
                
            #     # # log
            #     # if current_iter % opt['logger']['print_freq'] == 0:
            #     #     log_vars = {'epoch': epoch, 'iter': current_iter}
            #     #     log_vars.update({'lrs': model.get_current_learning_rate()})
            #     #     log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
            #     #     log_vars.update({'loss_student_g':loss_student_g})
            #     #     log_vars.update({'loss_k':loss_k})
            #     #     log_vars.update({'lrs_k': model.get_current_learning_rate()[0]})
            #     #     msg_logger(log_vars)
           
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            
            # for name, value in model.named_parameters():
            #     if 'alpha' in name:
            #         if value.grad is not None:
            #             writer().add_scalar(f'{name}_grad', value.grad.cpu().data.numpy(), total_iters)
            #             writer().add_scalar(f'{name}_data', value.cpu().data.numpy(), total_iters)
            # writer.close()
            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
    

def freeze_model(model):

    for (name, param) in model.named_parameters():
        param.requires_grad = False

    return 

def unfreeze_model(model):

    for (name, param) in model.named_parameters():
        param.requires_grad = True

    return 


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
