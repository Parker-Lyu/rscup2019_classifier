# rscup2019_classifier
rscup2019，classifier challenge, baseline

[中文版本](https://github.com/Parker-Lyu/rscup2019_classifier/blob/master/README_CN.md)

## Introduction
[RSCUP: 遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)

## Accuracy
This baseline got 0.893

## Requirements

pytorch==1.1 (1.0 should OK)

tensorboard==1.8

tensorboardX==1.7

pillow

Memory>=8G

GPU=1080Ti 

## Usage
The simplest way to use it is simply to specify the data_dir parameter. You will get 0.893 after 85m, also you will get log and tensorboard_log.
```
python train_resnet.py --data_dir your_data_dir
```

your data dir should be organized as follows:
```
data_dir/
    |->train
    |->val
    |->test
    |->ClsName2id.txt
```

## tensorboard
![](https://github.com/Parker-Lyu/rscup2019_classifier/blob/master/train.png)
