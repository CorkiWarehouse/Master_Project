import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):

    def __init__(self):
        '''
         5 层的卷积神经网络，包含两层卷积层和三层全连接层
        '''
        super(Net, self).__init__()
        # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
        self.conv1 = nn.Conv2d(1, 6, 5)
        # conv2 kernel size=5*5, 输出通道 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max-pooling 采用一个 (2,2) 的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 除了 batch 维度外的所有维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print('参数数量: ', len(params))
# conv1.weight
print('第一个参数大小: ', params[0].size())
# print(params[0])

# 随机定义一个变量输入网络
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 清空所有参数的梯度缓存，然后计算随机梯度进行反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

'''
The above part will be the loss function 
'''

output = net(input)
# 定义伪标签
target = torch.randn(10)
# 调整大小，使得和 output 一样的 size
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# 清空所有参数的梯度缓存
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 个输入特征（28x28 图像），128 个输出
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)   # 128 个输入特征，10 个输出（数字类别 0-9）

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层后应用激活函数
        x = self.fc2(x)  # 输出层
        return x

model = Net2()

# 虚拟输入和目标
input = torch.randn(64, 784)  # 64 张图像的批次，每个图像展平为 784 维
target = torch.randint(0, 10, (64,))  # 64 个目标标签的批次

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播
output = model(input)
loss = criterion(output, target)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(output)

# evaluation
with torch.no_grad():
    output = model(input)
    # print(output)


s_shape = numpy.zeros(10)
s_shape[random.randint(0,9)] = 1
s_shape = torch.tensor(s_shape)

a_shape = numpy.zeros(10)
a_shape[random.randint(0,9)] = 1
a_shape = torch.tensor(a_shape)

print(torch.cat([s_shape, a_shape], dim=0))