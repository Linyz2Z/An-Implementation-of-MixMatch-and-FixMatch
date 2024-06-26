# MixMatch
PyTorch 版本的 MixMatch 实现

## 使用方法

```
默认参数：
epochs = 32
train_iteration = 1000
batch_size = 64
lr = 0.002
manualSeed = 0
gpu = '0'
n_labeled = 250
ema_decay = 0.999

alpha = 0.75
lambda_u = 75
T = 0.5
K = 2
```

### 进行训练:

```
python train.py
```


### 使用tensorboard查看日志
```
tensorboard --logdir=<your out_dir>
```

## 引用
- [Official TensorFlow implementation of MixMatch](https://github.com/google-research/mixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
