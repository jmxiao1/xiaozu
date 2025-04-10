import collections
import os
import math
import torch
import torchvision
import shutil
import pandas as pd
from IPython.core.pylabtools import figsize
from nltk import accuracy
from param import output
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as plt
from matplotlib import pyplot as plt
from matplotlib import font_manager
train_loss=list()
test_loss=list()

def plot(train_loss, test_loss):
    # 创建一个5x5大小的图像
    plt.figure(figsize=(5,5))
    # 绘制训练损失曲线，标签为'train loss'，透明度为0.5
    plt.plot(train_loss, label='train loss',alpha =0.5)
    # 绘制测试损失曲线，标签为'test loss'，透明度为0.5
    plt.plot(test_loss, label='test loss',alpha =0.5)
    # 设置图像标题为'cifar-10'
    plt.title('cifar-10')
    # 设置x轴标签为'train_epoch'
    plt.xlabel('train_epoch')
    # 设置y轴标签为'loss'
    plt.ylabel('loss')
    # 显示图例
    plt.legend()
    # 显示图像
    plt.show()



# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练数据集，并进行数据增强和归一化处理
train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms. RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

# 加载测试数据集，并进行数据增强和归一化处理
test_data_set= torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((32,32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

# 创建训练数据加载器，批量大小为64，打乱数据顺序，丢弃最后一个不完整的批次
train_data_load =DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True)
# 创建测试数据加载器，批量大小为64，打乱数据顺序，丢弃最后一个不完整的批次
test_data_load =DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True)

# 获取训练数据集和测试数据集的长度
train_data_set=len(train_data_set)
test_data_set=len(test_data_set)

# 定义网络模型
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),

        )

        self.fc=nn.Sequential(
            nn.Flatten(),

            nn.Linear(128*4*4, 1024), nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(self.main(x))

# 实例化网络模型，并将其移动到指定的设备上
mynet=MyNet()
mynet = mynet.to(device)
print(mynet)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 将损失函数移动到指定的设备上
loss_fn = loss_fn.to(device)

# 定义学习率
learning_rate=1e-3
# 定义优化器
optim =torch.optim.Adam(mynet.parameters(), lr=learning_rate)

# 定义训练步数和测试步数
train_step =0
test_step =0
# 定义训练轮数
epochs =50

# 如果是主程序，则执行以下代码
if __name__ == '__main__':
    # 遍历训练轮数
    for i in range(epochs):
        # 将模型设置为训练模式
        mynet.train()

        # 遍历训练数据集
        for j, (imgs,targets) in enumerate(train_data_load):

            # 将数据移动到指定的设备上
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 前向传播
            output = mynet(imgs)
            # 计算损失
            loss =loss_fn(output, targets)
            # 梯度清零
            optim.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()

            # 更新训练步数
            train_step +=1
            # 每隔100步打印一次训练损失
            if train_step % 100 == 0:
                print('train_step:',train_step,'loss:',loss.item())
                # 将训练损失添加到列表中
                train_loss.append(loss.data.cpu())


        # 将模型设置为评估模式
        mynet.eval()
        # 定义准确率
        accuracy=0
        accuracy_total=0
        # 不计算梯度
        with torch.no_grad():
            # 遍历测试数据集
            for j, (imgs,targets) in enumerate(test_data_load):

                # 将数据移动到指定的设备上
                imgs = imgs.to(device)
                targets = targets.to(device)

                # 前向传播
                output = mynet(imgs)
                # 计算损失
                loss =loss_fn(output, targets)

                # 计算准确率
                accuracy = (output.argmax(1) == targets).sum()
                # 累加准确率
                accuracy_total +=accuracy
                # 更新测试步数
                test_step+=1

                # 每隔100步打印一次测试损失
                if test_step % 100 == 0:
                   # 将测试损失添加到列表中
                   test_loss.append(loss.data.cpu())

            # 打印当前轮次的准确率
            print(f'第{i+1}轮训练结束，准确率为{accuracy_total / test_data_set}')
            # 保存模型
            torch.save(mynet, f'CIFAR_10_{i+1}_acc_{accuracy_total/test_data_set}.pth')

# 绘制训练损失和测试损失曲线
plot(train_loss, test_loss)




