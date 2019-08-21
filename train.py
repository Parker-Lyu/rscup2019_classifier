# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/8/10 17:26
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import time
import datetime
import argparse
import os
import os.path as osp
import random

from rs_dataset import RSDataset
from get_logger import get_logger
from networks import Dense201
from utils import mixup_criterion, mixup_data, LinearScheduler


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--pre_epoch', type=int, default=2)      # 预训练轮次
    parse.add_argument('--warmup_epoch', type=int, default=1)   # warmup轮次
    parse.add_argument('--normal_epoch', type=int, default=15)  # 正常训练轮次

    parse.add_argument('--base_lr', type=float, default=2.5e-3) # 基础学习率
    parse.add_argument('--min_lr',type=float, default=5e-6)     # 最低学习率
    parse.add_argument('--batch_size', type=int, default=40)
    parse.add_argument('--test_batch_size', type=int, default=64)

    parse.add_argument('--sgdn', type=bool, default=True)       # SGD参数
    parse.add_argument('--weight_decay', type=float, default=0.001) # 权重衰减参数
    parse.add_argument('--mixup_alpha',type=float, default=0.2) # mixup 参数

    parse.add_argument('--num_workers', type=int, default=8)
    parse.add_argument('--eval_fre', type=int, default=1)  # 验证频率
    parse.add_argument('--msg_fre', type=int, default=10)  # message 频率
    parse.add_argument('--save_fre', type=int, default=1)  # 保存频率
    parse.add_argument('--save_after',type=int, default=10) # 10epo之后再保存

    parse.add_argument('--name', type=str, default='dense201',
                       help='unique out file name of this task include log/model_out/tensorboard log')
    parse.add_argument('--data_dir', type=str, default='C:\dataset\\rscup')
    parse.add_argument('--log_dir', type=str, default='./logs')
    parse.add_argument('--tensorboard_dir', type=str, default='./tensorboard')
    parse.add_argument('--model_out_dir', type=str, default='./model_out')
    parse.add_argument('--model_out_name', type=str, default='final_model.pth')
    parse.add_argument('--seed', type=int, default=5, help='random seed') # 随机种子
    return parse.parse_args()

# 评估函数
def evalute(net, val_loader, writer, epoch, logger):
    logger.info('------------after epo {}, eval...-----------'.format(epoch))
    total = 0
    correct = 0
    loss = 0.
    net.eval()
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img)
            outputs = F.log_softmax(outputs, dim=1)
            predicted = torch.max(outputs, dim=1)[1]
            total += lb.size()[0]
            correct += (predicted == lb).sum().cpu().item()

            loss += F.nll_loss(outputs, lb, reduction='sum').cpu().item()

    logger.info('accuracy:{}/{}={:.4f}, val_loss={:.4f}'.format(correct, total, correct * 1. / total, loss / total))
    writer.add_scalar('val_acc', correct * 1. / total, epoch)
    writer.add_scalar('val_loss', loss / total, epoch)
    net.train()

def main_worker(args, logger):
    try:
        writer = SummaryWriter(logdir=args.sub_tensorboard_dir)

        train_set = RSDataset(rootpth=args.data_dir, mode='train')
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers
                                  )

        # 权重list，每个样本被选择的概率，重采样效果不好，不使用，但是留作实例，以后参考
        # sampler_weight = train_set.get_sampler_weight()
        #
        # train_sampler = WeightedRandomSampler(sampler_weight,
        #                                 num_samples=100000,     # 每次循环，使用的样本数量
        #                                 replacement=True)
        #
        # train_loader = DataLoader(train_set,
        #                           batch_size=args.batch_size,
        #                           pin_memory=True,
        #                           num_workers=args.num_workers,
        #                           sampler=train_sampler)

        val_set = RSDataset(rootpth=args.data_dir, mode='val')
        val_loader = DataLoader(val_set,
                                batch_size=args.test_batch_size,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.num_workers)

        net = Dense201()
        logger.info('net name: {}'.format(net.__class__.__name__))
        net.train()
        input_ = torch.randn((1, 3, 224, 224))
        writer.add_graph(net, input_)
        net = net.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        if args.pre_epoch:
            # 预训练：冻结前面的层，只训练新增加的全连接层
            for name, param in net.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.base_lr,
                                  momentum=0.9,nesterov=args.sgdn,weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.pre_epoch*len(train_loader), eta_min=args.min_lr)

        loss_record = []
        iter_step = 0
        running_loss = []
        st = glob_st = time.time()
        total_epoch = args.pre_epoch + args.warmup_epoch +  args.normal_epoch
        total_iter_step = len(train_loader) * total_epoch

        logger.info('len(train_set): {}'.format(len(train_set)))
        logger.info('len(train_loader): {}'.format(len(train_loader)))
        logger.info('len(val_set): {}'.format(len(val_set)))
        logger.info('len(val_loader): {}'.format(len(val_loader)))
        logger.info('total_epoch: {}'.format(total_epoch))
        logger.info('total_iter_step: {}'.format(total_iter_step))

        if args.pre_epoch:
            logger.info('----- start pre train ------')
        for epoch in range(total_epoch):

            # 评估
            # if epoch % args.eval_fre == 0 and epoch!=0 :
            if epoch % args.eval_fre == 0 :
                evalute(net, val_loader, writer, epoch, logger)

            # 保存
            if epoch % args.save_fre == 0 and epoch > args.save_after:
                model_out_name = osp.join(args.sub_model_out_dir, 'out_{}.pth'.format(epoch))
                # 防止分布式训练保存失败
                state_dict = net.modules.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state_dict, model_out_name)

            # 预训练结束，训练所有参数，重构optimizer--但是只对全连接和卷积层的乘权重进行衰减
            if epoch == args.pre_epoch:
                for param in net.parameters():
                    param.requires_grad = True

                wd_params, nowd_params = [], []
                for name, module in net.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        wd_params.append(module.weight)
                        if not module.bias is None:
                            nowd_params.append(module.bias)
                    # todo 这种paramlist会不会漏掉了一些参数
                    elif isinstance(module, nn.BatchNorm2d):
                        nowd_params += list(module.parameters())
                    # else:
                    #     nowd_params += list(module.parameters())
                param_list = [
                    {'params': wd_params},
                    {'params': nowd_params, 'weight_decay': 0}]

                optimizer = optim.SGD(param_list, lr=args.base_lr, momentum=0.9, nesterov=args.sgdn, weight_decay=args.weight_decay)
                # 重构学习率调度器
                if args.warmup_epoch:
                    scheduler = LinearScheduler(optimizer, start_lr=args.min_lr, end_lr=args.base_lr, all_steps=args.warmup_epoch*len(train_loader))
                    logger.info('-------- start warmup for {} epochs -------'.format(args.warmup_epoch))


            # 如果到了正式训练，构建新的scheduller
            if epoch == args.pre_epoch + args.warmup_epoch:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.normal_epoch*len(train_loader), eta_min=args.min_lr)
                logger.info('---- start normal train for {} epoch ----'.format(args.normal_epoch))

            for img, lb in train_loader:
                iter_step += 1
                img = img.cuda()
                lb = lb.cuda()

                optimizer.zero_grad()

                inputs, targets_a, targets_b, lam = mixup_data(img, lb,
                                                               args.mixup_alpha)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                # outputs = net(img)
                # loss = criterion(outputs, lb)

                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss.append(loss.item())

                if iter_step % args.msg_fre == 0:
                    ed = time.time()
                    spend = ed - st
                    global_spend = ed - glob_st
                    st = ed

                    eta = int((total_iter_step - iter_step) * (global_spend / iter_step))
                    eta = str(datetime.timedelta(seconds=eta))
                    global_spend = str(datetime.timedelta(seconds=(int(global_spend))))

                    avg_loss = np.mean(running_loss)
                    loss_record.append(avg_loss)
                    running_loss = []

                    lr = optimizer.param_groups[0]['lr']

                    msg = '. '.join([
                        'epoch:{epoch}',
                        'iter/total_iter:{iter}/{total_iter}',
                        'lr:{lr:.7f}',
                        'loss:{loss:.4f}',
                        'spend/global_spend:{spend:.4f}/{global_spend}',
                        'eta:{eta}'
                    ]).format(
                        epoch=epoch,
                        iter=iter_step,
                        total_iter=total_iter_step,
                        lr=lr,
                        loss=avg_loss,
                        spend=spend,
                        global_spend=global_spend,
                        eta=eta
                    )
                    logger.info(msg)
                    writer.add_scalar('loss', avg_loss, iter_step)
                    writer.add_scalar('lr', lr, iter_step)

        # 训练完最后评估一次
        evalute(net, val_loader, writer, args.pre_epoch + args.normal_epoch, logger)

        out_name = osp.join(args.sub_model_out_dir, args.model_out_name)
        torch.save(net.cpu().state_dict(), out_name)

        logger.info('-----------Done!!!----------')

    except:
        logger.exception('Exception logged')
    finally:
        writer.close()


if __name__ == '__main__':

    args = parse_args()

    # 固定随机数种子，以使训练结果可以复现
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # 唯一标识
    unique_name = time.strftime('%y%m%d-%H%M%S_') + args.name
    args.unique_name = unique_name

    # 每次创建作业使用不同的tensorboard目录
    args.sub_tensorboard_dir = osp.join(args.tensorboard_dir, args.unique_name)
    # 保存模型的目录
    args.sub_model_out_dir = osp.join(args.model_out_dir, args.unique_name)

    # 创建所有用到的目录
    for sub_dir in [args.sub_tensorboard_dir, args.sub_model_out_dir, args.log_dir]:
        if not osp.exists(sub_dir):
            os.makedirs(sub_dir)

    log_file_name = osp.join(args.log_dir, args.unique_name + '.log')
    logger = get_logger(log_file_name)

    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main_worker(args, logger=logger)

