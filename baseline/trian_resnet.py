# -*- coding: utf-8 -*-
"""
@File    : trian_res34.py
@Time    : 2019/6/23 15:40
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

from rs_dataset import RSDataset
from get_logger import get_logger
from res_network import Resnet18,Resnet34


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch',type=int,default=10)
    parse.add_argument('--schedule_step',type=int,default=2)

    parse.add_argument('--batch_size',type=int,default=128)
    parse.add_argument('--test_batch_size',type=int,default=128)
    parse.add_argument('--num_workers', type=int, default=8)

    parse.add_argument('--eval_fre',type=int,default=2)
    parse.add_argument('--msg_fre',type=int,default=10)
    parse.add_argument('--save_fre',type=int,default=2)

    parse.add_argument('--name',type=str,default='res34_baseline', help='unique out file name of this task include log/model_out/tensorboard log')
    parse.add_argument('--data_dir',type=str,default='C:\dataset\\rscup')
    parse.add_argument('--log_dir',type=str, default='./logs')
    parse.add_argument('--tensorboard_dir',type=str,default='./tensorboard')
    parse.add_argument('--model_out_dir',type=str,default='./model_out')
    parse.add_argument('--model_out_name',type=str,default='final_model.pth')
    parse.add_argument('--seed',type=int,default=5,help='random seed')
    return parse.parse_args()

def evalute(net,val_loader,writer,epoch,logger):
    logger.info('------------after epo {}, eval...-----------'.format(epoch))
    total=0
    correct=0
    net.eval()
    with torch.no_grad():
        for img,lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img)
            outputs = F.softmax(outputs,dim=1)
            predicted = torch.max(outputs,dim=1)[1]
            total += lb.size()[0]

            correct += (predicted == lb).sum().cpu().item()
    logger.info('correct:{}/{}={:.4f}'.format(correct,total,correct*1./total,epoch))
    writer.add_scalar('acc',correct*1./total,epoch)
    net.train()

def main_worker(args,logger):
    try:
        writer = SummaryWriter(logdir=args.sub_tensorboard_dir)

        train_set = RSDataset(rootpth=args.data_dir,mode='train')
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers)

        val_set = RSDataset(rootpth=args.data_dir,mode='val')
        val_loader = DataLoader(val_set,
                                batch_size=args.test_batch_size,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers)

        net = Resnet34()
        net = net.train()
        input_ = torch.randn((1,3,224,224))
        writer.add_graph(net,input_)

        net = net.cuda()

        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.schedule_step,gamma=0.3)

        loss_record = []

        iter = 0
        running_loss = []
        st = glob_st = time.time()
        total_iter = len(train_loader)*args.epoch
        for epoch in range(args.epoch):

            # 评估
            if epoch!=0 and epoch%args.eval_fre == 0:
            # if epoch%args.eval_fre == 0:
                evalute(net, val_loader, writer, epoch, logger)

            if epoch!=0 and epoch%args.save_fre == 0:
                model_out_name = osp.join(args.sub_model_out_dir,'out_{}.pth'.format(epoch))
                # 防止分布式训练保存失败
                state_dict = net.modules.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state_dict,model_out_name)

            for img, lb in train_loader:
                iter += 1
                img = img.cuda()
                lb = lb.cuda()

                optimizer.zero_grad()

                outputs = net(img)
                loss = criterion(outputs,lb)

                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

                if iter%args.msg_fre ==0:
                    ed = time.time()
                    spend = ed-st
                    global_spend = ed-glob_st
                    st=ed

                    eta = int((total_iter-iter)*(global_spend/iter))
                    eta = str(datetime.timedelta(seconds=eta))
                    global_spend = str(datetime.timedelta(seconds=(int(global_spend))))

                    avg_loss = np.mean(running_loss)
                    loss_record.append(avg_loss)
                    running_loss = []

                    lr = optimizer.param_groups[0]['lr']

                    msg = '. '.join([
                        'epoch:{epoch}',
                        'iter/total_iter:{iter}/{total_iter}',
                        'lr:{lr:.5f}',
                        'loss:{loss:.4f}',
                        'spend/global_spend:{spend:.4f}/{global_spend}',
                        'eta:{eta}'
                    ]).format(
                        epoch=epoch,
                        iter=iter,
                        total_iter=total_iter,
                        lr=lr,
                        loss=avg_loss,
                        spend=spend,
                        global_spend=global_spend,
                        eta=eta
                    )
                    logger.info(msg)
                    writer.add_scalar('loss',avg_loss,iter)
                    writer.add_scalar('lr',lr,iter)

            scheduler.step()
        # 训练完最后评估一次
        evalute(net, val_loader, writer, args.epoch, logger)

        out_name = osp.join(args.sub_model_out_dir,args.model_out_name)
        torch.save(net.cpu().state_dict(),out_name)

        logger.info('-----------Done!!!----------')

    except:
        logger.exception('Exception logged')
    finally:
        writer.close()

if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 唯一标识
    unique_name = time.strftime('%y%m%d-%H%M%S_') + args.name
    args.unique_name = unique_name

    # 每次创建作业使用不同的tensorboard目录
    args.sub_tensorboard_dir = osp.join(args.tensorboard_dir, args.unique_name)
    # 保存模型的目录
    args.sub_model_out_dir = osp.join(args.model_out_dir, args.unique_name)

    # 创建所有用到的目录
    for sub_dir in [args.sub_tensorboard_dir,args.sub_model_out_dir,  args.log_dir]:
        if not osp.exists(sub_dir):
            os.makedirs(sub_dir)

    log_file_name = osp.join(args.log_dir,args.unique_name + '.log')
    logger = get_logger(log_file_name)

    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main_worker(args,logger=logger)
