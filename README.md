# rscup2019_classifier
rscup2019，分类赛道 baseline

## 简介
[RSCUP: 遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)

## 目前val得分（dense201单模无TTA）
0.9387

## tricks
[x] dataAug
[x] pretrain
[x] warmup
[x] 余弦退火
[x] 仅对conv层及FC层的乘权重进行权重衰减
[x] mixup
[ ] mean-teacher(测试中）
[ ] fast-autoAugment


## 一些经验
- 如果要增大batchsize（土豪多卡），根据一些论文，lr和batchsize保持同比例缩放即可    
- crop的概率不要设置为1   
- 根据efficientnet论文, 网络增大分辨率、wider、deeper，3个调整方向是相关的    


## 不确定的地方（希望大神解惑）
- [ ] train.py  line174-184, 本意是写仅对conv及FC的W进行权重衰减，但是不确定是否漏掉了params没有进行优化
- [ ] rs_dataset.py line184-201 计算数据集统计指标的方式不知道是否正确
- [ ] 分类问题中，拿到混淆矩阵，对容易分错的两个类如何处理？
- [ ] 数据均衡该如何做？train.py 中注释掉的代码是为了数据均衡写的，但是有负面效果


## 依赖
pytorch==1.1 (1.0应该也可以)

tensorboard==1.8

tensorboardX 

pillow

8G内存，1080Ti，更弱一点的显卡也可以，注意调低batch_size参数


## 使用方法
最简单的使用方法，只需要指明data_dir参数即可，默认参数在8G内存1080Ti的训练时间是12.5h（穷啊）   
```
python train_resnet.py --data_dir your_data_dir
```
验证及推理代码见infer.py     

本仓附带了最近一次实验的 tensorboard 及 log    

以及，附带了一个粗略的数据探索代码   

data_dir 之下的目录结构应如下：
```
data_dir/
    |->train
    |->val
    |->test
    |->ClsName2id.txt
```

## tensorboard
![](https://github.com/Parker-Lyu/rscup2019_classifier/baseline/blob/master/train.png)