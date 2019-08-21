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
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

class RSDataset(Dataset):
    def __init__(self, rootpth='C:\dataset\\rscup',re_size=280,crop_size=224,erase_size=48,crop_p=0.5,erase_p=0.7,mode='train', ):

        '''

        :param rootpth: 根目录
        :param re_size: 数据同一resize到这个尺寸再后处理
        :param crop_size: 剪切
        :param erase: 遮罩
        :param crop_p: 应用剪切的概率
        :param erase_p：应用遮罩的概率
        :param mode: train/val/test
        '''


        # 处理对应标签
        assert (mode=='train' or mode=='val' or mode=='test')

        # 中文对应数字标签
        lines = open(osp.join(rootpth,'ClsName2id.txt'),'r',encoding='utf-8').read().rstrip().split('\n')
        self.category2idx = {}
        for line in lines:
            line_list = line.strip().split(':')
            self.category2idx[line_list[0]] = int(line_list[2])-1 # 减去1是为了从0开始编号

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth,mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # 拿到labels
        self.labels = [self.category2idx[name.split(self.split_char)[-2]] for name in self.file_names]

        if mode=='train':
            self.to_tensor=get_aug_to_tensor(re_size=re_size,
                                             crop_size=crop_size,
                                             erase_size=erase_size,
                                             crop_p=crop_p,
                                             erase_p=erase_p)
        else:
            self.to_tensor=get_to_tensor(dsize=crop_size)


    def __getitem__(self, idx):
        name = self.file_names[idx]
        # print(name)
        cate_int = self.labels[idx]
        src_img = Image.open(name)
        return self.to_tensor(src_img),cate_int

    def __len__(self):
        return len(self.file_names)

    # 应对数据不均衡的重采样，但是得到负面效果
    def get_sampler_weight(self):
        # 每种标签出现的数量
        statistics = dict(Counter(self.labels))
        # 每种标签对应的概率采样，使得每种标签出现的概率相同
        for k in statistics.keys():
            statistics[k] = 1000/statistics[k]

        # 对应样本及标签顺序的权重list
        weight = list(map(statistics.get,self.labels))
        return weight


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
        self.to_tensor = get_to_tensor(dsize=dsize)

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        base_name = self.base_names[idx]

        img = Image.open(file_name)
        return self.to_tensor(img),base_name


# 随机crop
class random_crop(object):
    def __init__(self, re_size=280, crop_size=224, p=0.5):
        '''
        以一定概率随机剪切
        :param re_size: 先调整到这个尺寸
        :param crop_size: 剪切尺寸
        :param p: 执行随机剪切的概率
        '''
        self.resize_big = transforms.Resize(re_size)
        self.resize_small = transforms.Resize(crop_size)
        self.crop = transforms.RandomCrop(crop_size)
        self.p = p

    def __call__(self, img):
        # 执行剪切
        if random.random()<self.p:
            # print('crop')
            img = self.resize_big(img)
            return self.crop(img)
        # resize 返回
        return self.resize_small(img)


# 随机遮罩
class random_erase(object):
    def __init__(self, erase_size=48, p=0.7):
        self.erase_size = erase_size
        self.p = p

    def __call__(self, img):
        if random.random()<self.p:
            # print('erase')
            w,h = img.size
            start_x = random.randint(0, w-self.erase_size-1)
            start_y = random.randint(0, h-self.erase_size-1)

            erase_np = np.random.random((self.erase_size,self.erase_size,3))
            erase_img = Image.fromarray(erase_np, mode='RGB')
            img.paste(erase_img,(start_x,start_y,start_x+self.erase_size,start_y+self.erase_size))
        return img


def get_to_tensor(dsize):
    return transforms.Compose([
                transforms.Resize(dsize),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Normalize((0.3876, 0.3968, 0.3508), (0.1429, 0.1358, 0.1306)),
            ])

# 颜色抖动发现效果一般，不用了
def get_aug_to_tensor(re_size,crop_size,erase_size,crop_p,erase_p):
    return transforms.Compose([
                random_crop(re_size=re_size,crop_size=crop_size,p=crop_p),             # resize & crop
                random_erase(erase_size=erase_size,p=erase_p),
                transforms.RandomHorizontalFlip(),      # 水平翻转
                transforms.RandomVerticalFlip(),        # 垂直翻转
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Normalize((0.3876, 0.3968, 0.3508), (0.1429, 0.1358, 0.1306)),
        ])


#
# todo 计算数据的统计指标 计算方式：每张图片计算6个指标，最后求平均；不知道计算方式对不对，希望大佬赐教
def calculate_statistics(data_dir='C:\\dataset\\rscup'):
    file_names = []
    for sub_dir in ['train','val','test']:
        for root,dir,names in os.walk(osp.join(data_dir,sub_dir)):
            for name in names:
                file_names.append(osp.join(root,name))
    print(len(file_names))

    statis_means = np.array([0.,0.,0.])
    statis_stds = np.array([0.,0.,0.])
    for file in tqdm(file_names):
        img = Image.open(file)
        img_np = np.array(img).astype(np.float32)/255
        statis_means += img_np.mean(axis=(0,1))
        statis_stds += img_np.std(axis=(0,1))

    print('means',statis_means/len(file_names))
    print('stds',statis_stds/len(file_names))


if __name__ == '__main__':
    aaa = RSDataset(rootpth='C:\dataset\\rscup',mode='train')
    print(len(aaa))
    # # aaa = InferDataset(rootpth='C:\dataset\\rscup')
    #
    # deal_img,cat = aaa.__getitem__(1000)
    # deal_img.show()

    # print(cat)
    # # print(src_img.size)
    # print(deal_img.size())

    # src_img.show()
    # deal_img.show()
    # print(img)
    # print(len(aaa))

    # print(aaa.__getitem__(2))
    # calculate_statistics()