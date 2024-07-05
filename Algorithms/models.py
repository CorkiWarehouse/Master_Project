import torch
import torch.nn as nn
import torch.functional as F

'''
3 NN model 
'''
class RewardModel(nn.Module):
    '''
    This is used to give the immediate rewards
    according to the current state,policy and mean field
    '''
    def __init__(self, state_shape, action_shape, mf_shape, num_of_units):
        super(RewardModel, self).__init__()
        '''
        This is the input layer
            and this value is the slope of this leakyReLU
        '''
        self.LReLU = nn.LeakyReLU(0.01)
        '''
        Here is 2 linear layer with ReLU activation function
        The use of LeakyReLU helps to keep gradients flowing during training
        '''
        self.linear_c1 = nn.Linear(state_shape + action_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)

        '''This is the output layer'''
        self.linear_c = nn.Linear(num_of_units, 1)

        self.reset_parameters()
        self.train()

    '''This is used to init the parameters'''
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))


    '''Inputdata's transition direction 
    
    State_input: one-hot encoding of the current state of agent
    
    Action_input: one-hot encoding of the current state of agent
    
    MF_Input: one-hot encoding of the current mean_field
    '''
    def forward(self, state_input, action_input, mf_input):
        """
        input_g: input_global, input features of all agents
        """
        '''
        x_cat : 将状态、动作和平均场输入连接到单个向量中，表明模型处理组合特征以确定奖励。
            我们在这里是三维的一个tensors
        
        First hidden layer: linaer_c1 
            连接的输入通过第一个线性层（`linear_c1`）传递，将输入转换到更高或不同的维度空间。 
            然后该层的输出通过 LeakyReLU 激活函数（`LReLU`）。 
            LeakyReLU 有助于在训练期间保持梯度流，特别是对于使用标准 ReLU 会产生零梯度的输入。
        
        second: linaer_c2
            第一次转换的输出再次由“linear_c2”进行线性转换，并由另一个 LeakyReLU 进行处理。
            此步骤进一步处理数据，提取更复杂的特征和关系。
            
        Output layer: linea_c 
            abd output is a value 
        '''
        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, action_input, mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        #print('value', value)
        return value

'''
No action comparing with Reward_Model
'''
class ShapingModel(nn.Module):
    def __init__(self, state_shape, mf_shape, num_of_units):
        super(ShapingModel, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(state_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)
        self.linear_c = nn.Linear(num_of_units, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state_input, mf_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        #print('value', value)
        return value


class RewardMDPModel(nn.Module):
    def __init__(self, state_shape, action_shape, mf_shape, num_of_units):
        super(RewardMDPModel, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(state_shape * action_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)
        self.linear_c = nn.Linear(num_of_units, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, policy_input, mf_input):
        '''
        What we have done in x_cat
        1. Here reshape(1, -1):
            将 `policy_input` 张量转换为二维张量（或矩阵），
            其中第一维（行）为 1，第二维（列）为 从输入的长度推断。
        2. torch.cat([policy_input.reshape(1, -1), mf_input], dim=0)
            这将重构后的“policy_input”与“mf_input”连接起来
            `torch.cat` 用于沿指定维度连接张量。 这里，“dim=0”表示串联是沿着垂直轴的（将一个堆叠在另一个之上）。
        3. self.linear_c1(...):
            这是应用于串联输入的线性变换。
            “linear_c1”是神经网络中定义的线性（或完全连接）层。
            它将输入向量转换为另一个预定义大小的向量（由“linear_c1”层中的“num_of_units”定义）
        4. self.LReLU(...):
            将非线性激活函数应用于线性变换的输出
        '''
        x_cat = self.LReLU(self.linear_c1(torch.cat([policy_input.reshape(1, -1), mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value



'''
Here is new 
'''
class DiscriminatorModel(nn.Module):
    def __init__(self, state_size, action_size, num_of_units):
        super(DiscriminatorModel, self).__init__()

        '''
        Uses a Leaky ReLU activation function with a negative slope of 0.01
            this is not the learning rate 
        '''
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(state_size + action_size, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)
        self.linear_c3 = nn.Linear(num_of_units, num_of_units)
        self.linear_c = nn.Linear(num_of_units, 1)

        '''
        Sigmoid Activation (self.sigmoid)
            Used at the output to squeeze the output between 0 and 1, 
            making it useful for binary classification tasks.
        '''
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    '''
    Initializes the weights of the linear layers using Xavier uniform initialization
         ensure that the weights are not too small or too large, 
         aiding in maintaining a healthy gradient flow through the network
    '''
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, action_input], dim=0)))
        x2 = self.LReLU(self.linear_c2(x_cat))
        x = self.LReLU(self.linear_c3(x2))
        value = self.sigmoid(self.linear_c(x))
        return value



