import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as U

from Algorithms.myModels import RewardModel
from Algorithms.models import ShapingModel
from core import IRL

MAX = 1000000  # maximum number of iterations
MIN = 1e-32

# torch.gradient()
class MFAIRL(IRL):
    '''
    Max_epoch : 训练迭代的最大次数
    learning_rate：优化器的学习率。
    ax_grad_norm：梯度裁剪的阈值，有助于通过避免梯度爆炸来稳定训练。
    num_of_units：指定奖励和塑造模型中神经网络层的大小。
    '''

    '''
    
    '''
    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        reward_model = RewardModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_shape,
                                   mf_shape=self.env.state_count,
                                   num_of_units=num_of_units).to(self.device)
        shaping_model = ShapingModel(state_shape=self.env.state_shape,
                                     mf_shape=self.env.state_count,
                                     num_of_units=num_of_units).to(self.device)
        optimizer1 = optim.Adam(reward_model.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(shaping_model.parameters(), lr=learning_rate)


        '''
        What is this ?
        '''
        # estimate mean field flow
        est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
        '''
        Sample is the a trajectory from t = 0 to the self.horizon
            And sample must include the action and the states 
            
            And these states and actions are not onehot. 
                it is directly be the number like 1 means state 1 or action 1
                
        '''
        for sample in self.data:
            for t in range(self.horizon):
                est_expert_mf_flow[t, int(sample.states[t])] += 1
        est_expert_mf_flow /= len(self.data)

        # training starts
        '''One epoch means a training loop'''
        for epoch in range(max_epoch):
            # calculate rewards of each sample (demonstrated trajectories)
            rewards = []
            for sample in self.data:
                '''
                The `reward_model` computes the immediate reward for each state-action pair at each time step, 
                augmented with the mean field estimate at that time step. 
                '''


                reward_per_step = [reward_model(torch.tensor(self.env.state_option[int(sample.states[t])]).to(self.device, torch.float),
                                                torch.tensor(self.env.action_option[int(sample.actions[t])]).to(self.device, torch.float),
                                                torch.from_numpy(est_expert_mf_flow[t, :]).to(self.device, torch.float)) for t in range(self.horizon)
                                   ]



                # reward sum
                reward_per_sample = torch.sum(torch.cat(reward_per_step, dim=0).reshape((1, -1)))



                '''
                What is this formular
                
                Why we need lastest - init
                '''
                reward_shaping_per_sample = reward_per_sample \
                                            + shaping_model(torch.tensor(self.env.state_option[int(sample.states[self.horizon-1])]).to(self.device, torch.float),
                                                          torch.from_numpy(est_expert_mf_flow[self.horizon-1, :]).to(self.device, torch.float)
                                                          )\
                                            - shaping_model(torch.tensor(self.env.state_option[int(sample.states[0])]).to(self.device, torch.float),
                                                          torch.from_numpy(est_expert_mf_flow[0, :]).to(self.device, torch.float)
                                                          )
                rewards.append(reward_shaping_per_sample)

            '''
            On above, we have already run through every sample in the data 
            '''

            # calculate expected shaping rewards of demonstrations
            expected_shaping_reward = torch.sum(torch.cat(rewards, dim=0).reshape((1, -1))) / len(self.data)

            # estimate partition function
            logZ = torch.log(torch.sum(torch.exp(torch.cat(rewards, dim=0).reshape((1, -1))))).to(self.device,
                                                                                                  torch.float)

            # gradient descent
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss = - expected_shaping_reward + logZ
            loss.backward()
            U.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            U.clip_grad_norm_(shaping_model.parameters(), max_grad_norm)
            optimizer1.step()
            optimizer2.step()

            print('=MFIRL: epoch:{}'.format(epoch) + ', loss:{}'.format(str(loss.detach().cpu().numpy())), end='\r')


        self.reward_model = reward_model
        # torch.save(self.reward_model, './model_saved/mfirl_' + self.env.name + '_' + str(len(self.data)) + '.pt')
