import numpy as np
from PIL import Image

import torchvision
import torch

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar10(root, n_labeled, transform_train=None, transform_val=None, download=True):
    # root: 数据集的根目录
    # n_labeled: 带标签的数据总量
    # transform_train: 训练数据的转换
    # transform_val: 验证数据的转换
    # download: 是否下载数据集

    # 下载并加载基础的 CIFAR-10 训练数据集
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    
    # 根据标签数量进行训练、未标记和验证数据的划分
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))
    # 创建带标签的训练数据集
    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    # 创建未标记的训练数据集，应用两次数据增强
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    # 创建验证数据集
    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    # 创建测试数据集
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    # 打印各数据集的大小
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    
    # 返回训练、未标记、验证和测试数据集
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

'''
    # 将部分索引分配给带标签训练数据
    train_labeled_idxs.extend(idxs[:n_labeled_per_class])
    # 将部分索引分配给未标记训练数据
    train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
    # 将最后500个索引分配给验证数据
    val_idxs.extend(idxs[-500:])
'''
def train_val_split(labels, n_labeled_per_class):
    # labels: 数据集的标签
    # n_labeled_per_class: 每个类别中有标签的数据数量

    labels = np.array(labels)       # 将标签转换为 numpy 数组
    train_labeled_idxs = []         # 存储带标签的训练数据索引
    train_unlabeled_idxs = []       # 存储未标记的训练数据索引
    val_idxs = []                   # 存储验证数据索引

    # 对每个类别进行数据划分
    for i in range(10):
        idxs = np.where(labels == i)[0]  # 找出所有属于第 i 类的索引
        np.random.shuffle(idxs)          # 随机打乱这些索引

        # 将部分索引分配给带标签训练数据
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        # 将部分索引分配给未标记训练数据
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        # 将最后500个索引分配给验证数据
        val_idxs.extend(idxs[-500:])
    
    # 随机打乱每个数据集的索引
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    # 返回划分后的数据索引
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR10_labeled(torchvision.datasets.CIFAR10):
    # CIFAR10_labeled 类用于处理带标签的 CIFAR-10 数据

    def __init__(self, root, indexs=None, train=True, transform=None, target_transform=None, download=False):
        # 初始化函数
        # root: 数据集的根目录
        # indexs: 用于选择特定索引的数据
        # train: 指定是训练集还是测试集
        # transform: 对图像的转换
        # target_transform: 对标签的转换
        # download: 是否下载数据集
        super(CIFAR10_labeled, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # 调用父类的初始化方法，下载并加载数据集

        if indexs is not None:
            # 如果提供了索引列表，只保留这些索引的数据
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        
        # 对数据进行归一化和转置
        self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        # 获取指定索引的数据和标签
        """
        Args:
            index (int): 索引

        Returns:
            tuple: (image, target) 图像和标签的元组
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            # 如果提供了图像转换，应用转换
            img = self.transform(img)

        if self.target_transform is not None:
            # 如果提供了标签转换，应用转换
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):
    # CIFAR10_unlabeled 类用于处理无标签的 CIFAR-10 数据

    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False):
        # 初始化函数
        # root: 数据集的根目录
        # indexs: 用于选择特定索引的数据
        # train: 指定是训练集还是测试集
        # transform: 对图像的转换
        # target_transform: 对标签的转换
        # download: 是否下载数据集
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train, transform=transform, target_transform=target_transform, download=download)
        # 调用父类的初始化方法，下载并加载数据集

        self.targets = np.array([-1 for i in range(len(self.targets))])
        # 将所有标签设置为 -1，因为这些数据是无标签的
        