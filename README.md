# SYSU-2023-Spring-Pattern-Recognition-MJH-HW3
# SYSU 2023春 模式识别 mjh hw3 

## 实验要求
1. 阅读原始论文和相关参考资料，基于 Pytorch 分别实现 MixMatch 和 FixMatch 半监督图像分类算法，按照原始论文的设置，MixMatch 和 FixMatch 均使用 WideResNet-28-2 作为 Backbone 网络，即深度为 28，扩展因子为 2，在 CIFAR-10 数据集上进行半监督图像分类实验，报告算法在分别使用 40, 250,
4000 张标注数据的情况下的图像分类效果（标注数据随机选取指定数量）
1. 使用 TorchSSL中提供的 MixMatch 和 FixMatch 的实现进行半监督训练和测试，对比自己实现的算法和 TorchSSL 中的实现的效果
2. 提交源代码，不需要包含数据集，并提交实验报告，实验报告中应该包含代码的使用方法，对数据集数据的处理步骤以及算法的主要实现步骤，并分析对比 MixMatch 和 FixMatch 的相同点和不同点。
   
## 硬性要求
1. 鉴于部分同学没有GPU可以使用，统一要求训练的迭代数为20000和batch大小为64，所有报告结果都是基于此设置。
2. 报告中必须包含对MixMatch和FixMatch方法的解读（结合代码）
3. 作业三要求内容全部完成

## 加分项
1. 可以完成GPU环境的配置，并成功用于加速训练（需要CPU,GPU训练时常比较与截图）
2. 对方法中的组件进行细致的分析，采取不同超参数对准确率的影响
   
## 实验结果

| Algorithm | CIFAR-10 (40) | CIFAR-10 (250) | CIFAR-10 (4000) |
|-----------|---------------|----------------|-----------------|
| MixMatch  | 30.26         | 72.81          | 88.98           |
| FixMatch  | 47.63         | 86.73          | 91.05           |
