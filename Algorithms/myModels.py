'''
Here is new Models
'''
import torch
import torch.nn as nn
import torch.functional as F


# class DiscriminatorModel(nn.Module):
#     def __init__(self, state_shape, action_shape, num_of_units):
#         super(DiscriminatorModel, self).__init__()

    #     '''
    #     Uses a Leaky ReLU activation function with a negative slope of 0.01
    #         this is not the learning rate
    #     '''
    #     self.LReLU = nn.LeakyReLU(0.01)
    #     self.linear_c1 = nn.Linear(state_shape + action_shape, num_of_units)
    #     self.linear_c2 = nn.Linear(num_of_units, num_of_units)
    #     self.linear_c3 = nn.Linear(num_of_units, num_of_units)
    #     self.linear_c = nn.Linear(num_of_units, 1)
    #
    #     '''
    #     Sigmoid Activation (self.sigmoid)
    #         Used at the output to squeeze the output between 0 and 1,
    #         making it useful for binary classification tasks.
    #     '''
    #     self.sigmoid = nn.Sigmoid()
    #
    #     self.reset_parameters()
    #
    # '''
    # Initializes the weights of the linear layers using Xavier uniform initialization
    #      ensure that the weights are not too small or too large,
    #      aiding in maintaining a healthy gradient flow through the network
    # '''
    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
    #     nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
    #     nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
    #     nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))
    #
    # def forward(self, state_input, action_input):
    #     x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, action_input], dim=0)))
    #     x2 = self.LReLU(self.linear_c2(x_cat))
    #     x = self.LReLU(self.linear_c3(x2))
    #     value = self.sigmoid(self.linear_c(x))
    #     return value

'''
This is the NN for the mean field 

We have the parameters from the article
    so there is no need to refactor the layer 
    
And this is not the true mean field for this t,
    we only get the mean field for this (s,t)
'''

# TODO 放入cpu进行修改
# TODO detach


# this is the NN for the reward model



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

        state_input, action_input, mf_input, time_input = prepare_tensors(state=state_input, action= action_input, mf=mf_input)

        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, action_input, mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        # print('value', value)
        return value


'''
This is the neural network which is used to train the policy 
'''


# class MeanFieldModel(nn.Module):
#     def __init__(self, state_shape, time_horizon, num_of_units):
#         super(MeanFieldModel, self).__init__()
#         self.ReLU = nn.ReLU()
#         self.layer1 = nn.Linear(time_horizon + state_shape, num_of_units)
#         self.layer2 = nn.Linear(num_of_units, num_of_units // 2)
#         self.layer3 = nn.Linear(num_of_units // 2, num_of_units // 2 // 2)
#         self.layer4 = nn.Linear(num_of_units // 2 // 2, num_of_units // 2 // 2 // 2)
#         self.output = nn.Linear(num_of_units // 2 // 2 // 2, 1)
#         self.reset_parameters()
#         self.train()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.layer2.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.layer3.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.layer4.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('relu'))
#
#     def forward(self, state_input, time_input):
#
#         state_input, action_input, mf_input, time_input = prepare_tensors(state=state_input, time=time_input)
#
#
#
#         x_cat = self.ReLU(self.layer1(torch.cat([state_input, time_input], dim=0)))
#         x2 = self.ReLU(self.layer2(x_cat))
#         x3 = self.ReLU(self.layer3(x2))
#         x4 = self.ReLU(self.layer4(x3))
#         value = torch.sigmoid(self.output(x4))  # Use sigmoid to ensure output is between 0 and 1
#         return value

class MeanFieldModel(nn.Module):
    def __init__(self, state_shape, time_horizon, num_of_units):
        super(MeanFieldModel, self).__init__()
        input_size = time_horizon + state_shape

        self.layers = nn.Sequential(
            nn.Linear(input_size, num_of_units),
            nn.ReLU(),
            nn.Linear(num_of_units, num_of_units // 2),
            nn.ReLU(),
            nn.Linear(num_of_units // 2, num_of_units // 4),
            nn.ReLU(),
            nn.Linear(num_of_units // 4, num_of_units // 8),
            nn.ReLU(),
            nn.Linear(num_of_units // 8, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state_input, time_input):
        state_input, action_input, mf_input, time_input = prepare_tensors(state=state_input, time=time_input)
        x = torch.cat([state_input, time_input], dim=0)
        x = self.layers[:-1](x)  # Pass through all layers except the last one
        x = self.layers[-1](x)  # Apply the final Sigmoid activation
        return x

class PolicyModel(nn.Module):
    def __init__(self, state_shape, action_shape, mf_shape, num_of_units):
        super(PolicyModel, self).__init__()
        '''
                This is the input layer
                    and this value is the slope of this leakyReLU
                '''
        self.LReLU = nn.LeakyReLU(0.01)
        '''
        Here is 2 linear layer with ReLU activation function
        The use of LeakyReLU helps to keep gradients flowing during training
        '''
        self.linear_c1 = nn.Linear(state_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)

        '''This is the output layer
        For we train the policy so that our output should be the 
            probability for each actions
        '''
        self.linear_c = nn.Linear(num_of_units, action_shape)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    '''Inputdata's transition direction 

    State_input: one-hot encoding of the current state of agent

    Action_input: one-hot encoding of the current state of agent

    MF_Input: one-hot encoding of the current mean_field
    '''

    def forward(self, state_input, mf_input):
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

        state_input, action_input, mf_input, time_input = prepare_tensors(state=state_input, mf=mf_input)

        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        action_probs = torch.softmax(self.linear_c(x), dim=-1)
        return action_probs

def prepare_tensors(state=None, action=None, mf=None, time=None):
    """
    Convert inputs to PyTorch tensors with appropriate shapes, and return them in a specific order.
    Inputs can be None, in which case None is returned in their place.

    Args:
        state: The state input which can be a list, scalar, or tensor. Default is None.
        action: The action input which can be a list, scalar, or tensor. Default is None.
        mf: The mean-field input which can be a list, scalar, or tensor. Default is None.
        time: The time input which can be a list, scalar, or tensor. Default is None.

    Returns:
        tuple: A tuple of tensors or None, corresponding to state, action, mf, and time in that order.
    """
    inputs = {'state': state, 'action': action, 'mf': mf, 'time': time}
    results = []
    for key in ['state', 'action', 'mf', 'time']:
        input_data = inputs[key]
        if input_data is not None:
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            if input_data.dim() == 0:
                input_data = input_data.unsqueeze(0)  # Convert scalar to 1D tensor
            tensor = input_data
            results.append(tensor)
        else:
            results.append(None)

    return tuple(results)


