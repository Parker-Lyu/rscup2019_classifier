# rscup2019_classifier
rscup2019，分类赛道 baseline

[English](https://github.com/Parker-Lyu/rscup2019_classifier_baseline/blob/master/README.md)

## 简介
[RSCUP: 遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)

## 本baseline得分
0.893

## 依赖
pytorch==1.1 (1.0应该也可以)
tensorboard==1.8
tensorboardX 
pillow

8G内存，1080Ti，更弱一点的显卡也可以，注意调低batch_size参数

## 使用方法
最简单的使用方法，只需要指明data_dir参数即可，90分钟之后，就可以得到最终模型以及log、tensorboard_log了
'''
python train_resnet.py --data_dir your_data_dir
'''

data_dir 之下的目录结构应如下：
```
data_dir/
    |->train
    |->val
    |->test
    |->ClsName2id.txt
```