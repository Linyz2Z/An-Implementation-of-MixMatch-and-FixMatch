# FixMatch
PyTorch 版本的 FixMatch 实现

## 使用方法

### 默认参数
```
--dataset = 'cifar10'
--arch = 'wideresnet'
--batch_size = 64
--lr = 0.03
--mu = 7
--lambda_u = 1
--threshold = 0.95
--total_steps = 32000
--eval_step = 1000
```
### Train
使用250个 CIFAR-10 标注数据数据进行训练:

```
python train.py --num-labeled 250 --out results/cifar10@250
```

使用4000个 CIFAR-10 标注数据数据进行训练:
```
python train.py --num-labeled 4000 --out results/cifar10@4000
```

### 使用tensorboard查看日志
```
tensorboard --logdir=<your out_dir>
```

## 引用
- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Unofficial PyTorch implementation of FixMatch](https://github.com/ildoonet/pytorch-randaugment)