import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wideresnet as models
import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from tensorboardX import SummaryWriter

epochs = 32
batch_size = 64
lr = 0.002
manualSeed = 0
gpu = '1'
n_labeled = 250             # 带有标签的数据总量
train_iteration = 1000
output_dir = f'result/n_labeled={n_labeled}'
ema_decay = 0.999

# important hyper-parameter
alpha = 0.75
lambda_u = 75
T = 0.5
K = 2

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
cuda_available = torch.cuda.is_available()

np.random.seed(manualSeed)

best_acc = 0  # best test accuracy

def get_model(ema=False):
    model = models.WideResNet(num_classes=10)
    model = model.cuda()
    
    # 指数移动平均模型 (EMA) 
    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def main():
    global best_acc

    if not os.path.isdir(output_dir):
        mkdir_p(output_dir)

    # 数据处理
    print(f'Preparing dataset...')
    # 应用于训练集的数据变换
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    # 应用于验证集的数据变换
    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('./data', n_labeled, transform_train=transform_train, transform_val=transform_val)
    # 生成对应的DataLoader
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 获取Model
    print("Creating WideResnet-28-2...")

    model = get_model()
    ema_model = get_model(ema=True)     # 指数移动平均 (EMA) 模型

    cudnn.benchmark = True

    train_criterion = SemiLoss()        # 自定义的半监督损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=ema_decay)
    start_epoch = 0

    # 记录训练数据
    logger = Logger(os.path.join(output_dir, 'log.txt'), title='MixMatch-cifar-10')
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])
    writer = SummaryWriter(output_dir)  # tensorboard --logdir=.

    step = 0
    test_accs = []

    # Train and val
    for epoch in range(start_epoch, epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, lr))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, cuda_available)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, cuda_available, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, cuda_available, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, cuda_available, mode='Test Stats ')

        step = train_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, cuda_available):

    # AverageMeter用于跟踪和计算一些变量的平均值和总和
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=train_iteration)
    # shuffle=True, 故在每个epoch重新初始化iter迭代器后的数据顺序不同
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()      # K=2, transfrom twice 

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # 将标签转换为 one-hot 编码  targets_x.shape:(batch_size, 10)
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

        if cuda_available:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            # 计算未标记样本的预测标签
            outputs_u = model(inputs_u)         # (batch_size, 10)
            outputs_u2 = model(inputs_u2)       # (batch_size, 10)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            # 进行sharpen操作
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)    # (batch_size, 1)
            targets_u = targets_u.detach()

        # 进行mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        lambda_ = np.random.beta(alpha, alpha)
        lambda_ = max(lambda_, 1-lambda_)

        # 生成随机排列
        idx = torch.randperm(all_inputs.size(0))

        # all_inputs中既包括X^, 也包括U^
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = lambda_ * input_a + (1 - lambda_) * input_b
        mixed_target = lambda_ * target_a + (1 - lambda_) * target_b

        # 将标记样本和未标记样本在批次之间交错 
        # 确保批量归一化层在计算统计量时能够同时考虑标记数据和未标记数据的统计特性，减少偏差
        mixed_input = list(torch.split(mixed_input, batch_size))    # mixed_input.shape:  (total_input_size, feature) -> (n(3), batch_size, feature)
        mixed_input = interleave(mixed_input, batch_size)           # (n(3), batch_size, feature) -> (total_input_size, feature)(interleave)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)     # (total_input_size, feature)(interleave) -> (n(3), batch_size, feature)
        logits_x = logits[0]                        # P X'
        logits_u = torch.cat(logits[1:], dim=0)     # P U'

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, cuda_available, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=output_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):  # __call__ 方法用于计算半监督学习的损失
        probs_u = torch.softmax(outputs_u, dim=1)   # 计算未标注数据的预测概率

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))  # 计算有标签数据的分类损失 Lx
        Lu = torch.mean((probs_u - targets_u)**2)   # 计算无标签数据的一致性损失 Lu

        return Lx, Lu, lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()
