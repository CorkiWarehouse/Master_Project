import torch
import numpy as np
import core


# x1 = torch.rand(5,3)
#
# '''
# Test for the cuda:
# print(torch.cuda.is_available())
# '''
#
# # print(x)
#
# # create a matrix of 5*3
# # 5 lines * 3 column
# x = torch.empty(5, 3)
# print(x)
#
# tensor1 = torch.tensor([5.5, 3])
# print(tensor1)
#
# tensor2 = tensor1.new_ones(5,3,dtype=torch.long)
# print(tensor2)
#
# tensor3 = torch.randn_like(tensor2, dtype=torch.float)
# print('tensor3: ', tensor3)
#
# print(tensor3.size())
#
#
# '''
# Here is all the operations of the Tensor
# '''
#
# # TODO ADD
# tensor4 = torch.randn(5, 3)
# print('tensor3 + tensor4= : ', tensor3 + tensor4)
# print('tensor3 + tensor4= :', torch.add(tensor3, tensor4))
#
# # we need to announce a new variable
# # and this variable must be the same size with the result
# result = torch.empty(5,3)
# torch.add(tensor3, tensor4, out = result)
# print('add result= ',result)
#
# # directly change the variable
# tensor3.add_(tensor4)
# print('tensor3 = ',tensor3)
#
# # split the tensor
# print(tensor3[:,0])
#
# # change the tensor's size
# x = torch.randn(4, 4)
# y = x.view(16)
# print(y)
# # -1 表示除给定维度外的其余维度的乘积
# # in another word should be (a)*8 = 16
# # so that it can be reshaped
# # if it can not be the product , we will get the error
# z = x.view(-1, 8)
# print(x.size(), y.size(), z.size())
#
# # if tensor only have one variable we could get it out
# # not as a tensor
# x = torch.randn(1)
# print(x)
# print(x.item())
#
# '''
# We could change the tensor to numpy and could be done in reserve
# '''
#
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
#
# # TODO they share the same storage space
# # if we change the tensor, it will affect the numpy
# # this add is the matrix operation
# a.add_(1)
# print(a)
# print(b)
#
# # and we could also change back
# a = np.ones(5)
# # this will change b to the tensor
# b = torch.from_numpy(a)
# np.add(a, 1, out = a)
# print(a)
# print(b)
#
# '''
# Put these item to the CUDA
# '''
# '''
# 输出结果，第一个结果就是在 GPU 上的结果，
# 打印变量的时候会带有 device='cuda:0'，而第二个是在 CPU 上的变量。
# '''
#
# # 当 CUDA 可用的时候，可用运行下方这段代码，采用 torch.device() 方法来改变 tensors 是否在 GPU 上进行计算操作
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # 定义一个 CUDA 设备对象
#     y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
#     x = x.to(device)                       # 也可以采用 .to("cuda")
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # .to() 方法也可以改变数值类型
#
# '''
# Here is the "AUTOGRAD"
# 主要是提供了对 Tensors 上所有运算操作的自动微分功能，
# 也就是计算梯度的功能
# '''
#
# # here requires_grad let us can track the variable
# # And then we could do some calculation
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# # y 是一个操作的结果，所以它带有属性 grad_fn
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# # then we continue on the y
#
# z = y * y * 3
# out = z.mean()
#
# print('z=', z)
# print('out=', out)
#
# # here we change tensor's attribute
# # we let it could be gradient
#
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)
#
# '''
# Here above will be the gradient value
# x = all one tensor size = (2,2)
# y = x + 2
# z = y * y * 3
# out = z.mean()
#
# out will be the tensor(27) for z will all the the 27
# '''
#
# out.backward()
# # 输出梯度 d(out)/dx
# # 相当于对于x进行求导
# # 最后在带入x的值，每个位置对应进去
# print(x.grad)
#
# '''
# result :
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])
#
# So that we have all the gradient for the x_i on the O
# here we could see it as the gradient chaine
#
# we start wil O
# '''
#
# x = torch.randn(3, requires_grad=True)
#
# y = x * 2
# # here we scale the gradient
# while y.data.norm() < 1000:
#     y = y * 2
#
# print(y)
#
# # this v will be the weight of all the gradient
# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)
#
#
# '''
# Now let's consider the Neural Network
# '''
#
#
#
# print(x.grad)

# a = core.PolicyFlow(s=3,a=4,t=5)
# a.val[1,1,2] = 1
#
# # this where will give the index for each dimension
# # so that if we want the inner one for 1
# # we need to give the depth of the dimensions like how deep we want
# # then we can deside which one we want like [0] is the first one
# index = np.where(a.val[1,1] == 1)
# print(index[0][0])
# print(a.val)
#
# print(np.random.choice([1,2,3,4], 1, p=a.val[1, 1, :])[0])


# b = torch.randn(2,3)
# print(b[1,2])
#
# print(b[1,2].item())
#
#
# # 创建形状为 (3, 4, 5) 的随机数组
# arr = np.random.rand(3, 4, 5)
#
# # 将最后一个维度的值归一化为 1
# arr /= arr.sum(axis=-1, keepdims=True)
#
# print("随机数组（最后一个维度总和为 1）:")
# print(arr)


'''
cuda checking list 
'''

# print(torch.cuda.is_available())
#
# print(torch.cuda.device_count())
#
# print(torch.cuda.current_stream())
#
# print(torch.version.cuda)
#
# a = np.array([[1,2], [2,3], [4,5]])
#
# # this np.argmax will give the max index for this array
# print(np.argmax(a))


# init_policy_flow = np.random.rand(1, 2, 3)
#
#         # 将最后一个维度的值归一化为 1
# init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)
#
# print(init_policy_flow)
#
# a = core.Action(action= 3)
# s = core.State(state= 3)
# print(a.val)
# print(s.val[0])


# import Environments.CARS as CARS
#
# car1 = CARS.CARS(True,True)
#
# s = core.State(state=2)


import torch

# 单个损失的示例张量
# loss1 = torch.tensor([1.0, 2.0, 3.0])
# loss2 = torch.tensor([4.0, 5.0, 6.0])
#
# value_per_sampler = [loss1, loss2]
#
# # 连接和重塑
# concatenated = torch.cat(value_per_sampler, dim=0)
# reshaped = concatenated.reshape((1, -1))
#
# # 求和元素
# total_loss = torch.sum(reshaped)
#
# print("Concatenated:", concatenated)
# print("Reshaped:", reshaped)
# print("Total Loss:", total_loss)

# a = core.Action(action=10)
#
# m = core.MeanField(mean_field=[1,2,3])
#
# print(a.val[0])
#
# print(m.val)
#
#
# c = [0.2,0.3,0.4]
# c = np.array(c)
# print(c/sum(c))
#
# import os
# full_path = os.path.abspath('./model_saved/original_dynamics/CARS/10/2024_05_27_14:15/')
# print(full_path)
#
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")  # 这会产生一个类似 '2024_05_27_141505' 的字符串
# model_save_path = os.path.join('model_saved', 'original_dynamics', 'CARS', '10', '2024_05_27_14:15')
#
# if not os.path.exists(model_save_path):
#     try:
#         os.makedirs(model_save_path)
#     except OSError as e:
#         print(f"Error: {e.strerror}")


from Environments import CARS
from core import State, Action

car = CARS.Env(0,0)

print(car.action_option)
print(car.state_option)

state_0 = State(state=0)
action_0 = Action(action=1)

# print(car.get_all_valid_actions(0,0))
# print(car.trans_prob(state_0,action_0,mean_field=None))