import os
import torch
from PIL import Image
import torchvision
from openpyxl.styles.builtins import output
from torch import nn
import torch.serialization

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络结构
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),  # 第一个卷积层，输入通道数为3，输出通道数为32，卷积核大小为3，步长为1，填充为1
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化核大小为2，步长为2
            nn.BatchNorm2d(num_features=32),  # 批标准化层，输入通道数为32

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),  # 第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1，填充为1
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化核大小为2，步长为2
            nn.BatchNorm2d(num_features=64),  # 批标准化层，输入通道数为64

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),  # 第三个卷积层，输入通道数为64，输出通道数为128，卷积核大小为3，步长为1，填充为1
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化核大小为2，步长为2
            nn.BatchNorm2d(num_features=128),  # 批标准化层，输入通道数为128

        )

        self.fc=nn.Sequential(
            nn.Flatten(),  # 展平层，将输入展平为一维向量

            nn.Linear(128*4*4, 1024), nn.ReLU(inplace=True),  # 全连接层，输入维度为128*4*4，输出维度为1024，激活函数为ReLU
            nn.Dropout(),  # Dropout层，用于防止过拟合

            nn.Linear(1024, 256), nn.ReLU(inplace=True),  # 全连接层，输入维度为1024，输出维度为256，激活函数为ReLU
            nn.Dropout(),  # Dropout层，用于防止过拟合

            nn.Linear(256, 10),  # 全连接层，输入维度为256，输出维度为10
        )

    def forward(self, x):
        # 前向传播函数
        return self.fc(self.main(x))

# 定义目标索引
target_index={0:'飞机',1:'汽车',2:'鸟',3:'猫',4:'鹿',5:'狗',6:'青蛙',7:'马',8:'船',9:'卡车'}



# 定义根目录和图片目录
root_dir='test_CIFAR_10'
obj_dir ='test.png'
img_dir=os.path.join(root_dir,obj_dir)
# 打开图片
img =Image.open(img_dir)


# 定义数据预处理
tran_poss =torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),  # 将图片大小调整为32*32
    torchvision.transforms.ToTensor()  # 将图片转换为张量
])

# 添加安全全局变量
torch.serialization.add_safe_globals([MyNet])
torch.serialization.safe_globals([MyNet])
# 加载模型
mynet =torch.load('CIFAR_10_50_acc_0.847599983215332.pth',weights_only=False,map_location='cpu')
# 设置模型为评估模式
mynet.eval()


# 对图片进行预处理
img =tran_poss(img)
# 将图片调整为模型输入的形状
img =torch.reshape(img,(1,3,32,32))

# 使用模型进行预测
output=mynet(img)
# 打印预测结果
print(target_index[output.argmax(1).item()])
