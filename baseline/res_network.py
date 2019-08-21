# -*- coding: utf-8 -*-
"""
@File    : resNetwork.py
@Time    : 2019/6/23 15:29
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34
import torch.nn.functional as F

class Resnet18(nn.Module):
    def __init__(self, n_classes = 45):
        super(Resnet18,self).__init__()

        src_net = resnet18(pretrained=True)
        modules = list(src_net.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512,n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class Resnet34(nn.Module):
    def __init__(self, n_classes = 45):
        super(Resnet34,self).__init__()

        src_net = resnet34(pretrained=True)
        modules = list(src_net.children())[:-2]

        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512,n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # net = Resnet18()
    net = Resnet34()
    aa = torch.randn((5,3,100,100))
    print(net(aa).size())

