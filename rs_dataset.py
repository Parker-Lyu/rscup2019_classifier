# -*- coding: utf-8 -*-
"""
@File    : rs_dataset.py
@Time    : 2019/6/22 10:57
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : data set
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import random

class RSDataset(Dataset):
    def __init__(self, rootpth='E:\Zip&Data\dataset\\rscup',des_size=(224,224),mode='train', ):

        self.des_size = des_size
        self.mode = mode

        # 处理对应标签
        assert (mode=='train' or mode=='val' or mode=='test')
        lines = open(osp.join(rootpth,'ClsName2id.txt'),'r',encoding='utf-8').read().rstrip().split('\n')
        self.catetory2idx = {}
        for line in lines:
            line_list = line.strip().split(':')
            self.catetory2idx[line_list[0]] = int(line_list[2])-1

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth,mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

        # 随机选择一小部分数据做测试
        # self.file_names = random.choices(self.file_names[:200],k=5000)

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # totensor 转换
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        name = self.file_names[idx]
        category = name.split(self.split_char)[-2]
        cate_int = self.catetory2idx[category]
        img = Image.open(name)
        img = img.resize(self.des_size,Image.BILINEAR)
        return self.to_tensor(img),cate_int

    def __len__(self):
        return len(self.file_names)


class InferDataset(Dataset):
    def __init__(self, rootpth='C:\\dataset\\rscup', dsize = (224,224)):
        self.dsize=dsize
        # 读取文件名称
        self.file_names = []
        for root, dirs, names in os.walk(osp.join(rootpth, 'test')):
            for name in names:
                self.file_names.append(osp.join(root, name))

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        self.base_names = [osp.split(name)[1] for name in self.file_names]

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        base_name = self.base_names[idx]

        img = Image.open(file_name)
        return self.to_tensor(img),base_name

if __name__ == '__main__':
    aaa = RSDataset(rootpth='C:\dataset\\rscup',mode='val')
    # img,cat = aaa.__getitem__(13000)
    # print(cat)
    # print(img.size())
    # print(img)
    print(len(aaa))
    print(aaa.__getitem__(2))