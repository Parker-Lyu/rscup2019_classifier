# -*- coding: utf-8 -*-
"""
@File    : new_work.py
@Time    : 2019/6/22 11:43
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : net_work
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121,resnet18,resnet34,densenet201
import torch.nn.functional as F

class Dense201(nn.Module):
    def __init__(self, n_classes = 45):
        super(Dense201, self).__init__()

        self.net = densenet201(pretrained=True)
        self.net.classifier = nn.Linear(1920,n_classes)
        nn.init.constant_(self.net.classifier.bias, 0)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = Dense201()

    # for modules in net.modules():
    #     print(type(modules))
    print(len(list(net.named_modules())))

    wd_params, nowd_params = [], []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            wd_params.append(module.weight)
            if not module.bias is None:
                nowd_params.append(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nowd_params += list(module.parameters())
        else:
            print(name)
    print(len(wd_params),len(nowd_params))

    # aa = torch.randn((5,3,100,100))
    # print(net(aa).size())
    # for name, param in net.named_parameters():
    #     print(name, param.requires_grad)
    #     if ('norm5' not in name) and ('classifier' not in name):
    #         param.requires_grad = False

    # for param in net.parameters():
    #     print(param.requires_grad)