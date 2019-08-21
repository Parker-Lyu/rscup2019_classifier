# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2019/8/10 19:12
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import numpy as np

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LinearScheduler:
    def __init__(self,optimizer, start_lr, end_lr, all_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.all_steps = all_steps
        self.cur_step = 0
    def step(self):
        self.cur_step += 1
        if self.cur_step>=self.all_steps:
            self.cur_step=self.all_steps
        cur_lr = (self.end_lr-self.start_lr) * (self.cur_step*1./self.all_steps) + self.start_lr
        for param in self.optimizer.param_groups:
            param['lr'] = cur_lr