# -*- coding: utf-8 -*-
"""
@File    : infer.py
@Time    : 2019/7/9 22:04
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import time
import datetime
import argparse
import os
import os.path as osp

from rs_dataset import RSDataset,InferDataset
from get_logger import get_logger
from res_network import Resnet34

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size',type=int,default=128)
    parse.add_argument('--num_workers', type=int, default=8)

    parse.add_argument('--data_dir',type=str,default='C:\dataset\\rscup')
    parse.add_argument('--model_out_name',type=str,default='./model_out/190706-192015_dense201_pre_aug/final_model.pth')

    return parse.parse_args()


def main_worker(args):
    data_set = InferDataset(rootpth=args.data_dir)
    data_loader = DataLoader(data_set,
                             batch_size=args.test_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)


    net = Resnet34()

    net.load_state_dict(torch.load(args.model_out_name))

    net.cuda()

    net.eval()

    with open('classification.txt','w') as f:
        with torch.no_grad():
            for img,names in data_loader:
                img = img.cuda()
                size = img.size(0)
                outputs = net(img)
                outputs = F.softmax(outputs, dim=1)
                predicted = torch.max(outputs, dim=1)[1].cpu().numpy()

                for i in range(size):
                    msg = '{} {}'.format(names[i], predicted[i]+1)
                    f.write(msg)
                    f.write('\n')

    print('----------Done!----------')


# 用于测试验证集
def evaluate_val(args):
    val_set = RSDataset(rootpth=args.data_dir, mode='val')
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            drop_last=True,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    net = Resnet34()

    net.load_state_dict(torch.load(args.model_out_name))

    net.cuda()

    net.eval()

    total = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img)
            outputs = F.softmax(outputs, dim=1)
            predicted = torch.max(outputs, dim=1)[1]
            total += lb.size()[0]

            correct += (predicted == lb).sum().cpu().item()
    print('correct:{}/{}={:.4f}'.format(correct, total, correct * 1. / total))

    print('----------Done!----------')


if __name__ == '__main__':
    args = parse_args()
    # 推理测试集
    main_worker(args)
    # evaluate_val(args)
