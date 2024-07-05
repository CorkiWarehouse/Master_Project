import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as U

from Algorithms.models import RewardModel
from core import IRL


class PLIRL(IRL):

    def onehot_encoding(self, shape, entry):
        code = np.zeros(shape)
        code[entry] = 1.0
        return code

    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):

        reward_model = RewardModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_shape,
                                   mf_shape=self.env.state_shape,
                                   num_of_units=num_of_units).to(self.device)
        optimizer = optim.Adam(reward_model.parameters(), learning_rate)

        for epoch in range(int(max_epoch)):
            # calculate rewards of each sample (demonstrated trajectories)
            rewards = []
            for sample in self.data:
                reward_per_sample = torch.zeros(1,device=self.device)

                for t in range(self.horizon):
                    for s in range(self.env.state_shape):
                        for a in range(self.env.action_shape):

                            # first part
                            f = reward_model(torch.from_numpy(self.onehot_encoding(self.env.state_shape, s)).to(self.device, torch.float),
                                                        torch.from_numpy(self.onehot_encoding(self.env.action_shape, a)).to(self.device, torch.float),
                                                        torch.from_numpy(sample.mf_flow[t, :]).to(self.device, torch.float))
                            f_2 = f*sample.mf_flow[t, s] * sample.p_flow[t, s, a]


                            # reward_per_sample += sample.mf_flow[t, s] * sample.p_flow[t, s, a] * \
                            #                reward_model(torch.from_numpy(self.onehot_encoding(self.env.state_shape, s)).to(self.device, torch.float),
                            #                             torch.from_numpy(self.onehot_encoding(self.env.action_shape, a)).to(self.device, torch.float),
                            #                             torch.from_numpy(sample.mf_flow[t, :]).to(self.device, torch.float))

                            reward_per_sample = reward_per_sample + f_2

                rewards.append(reward_per_sample)


            # calculate expected shaping rewards of demonstrations
            expected_reward = torch.sum(torch.cat(rewards, dim=0).reshape((1, -1))) / len(self.data)

            # estimate partition function
            logZ = torch.log(torch.sum(torch.exp(torch.cat(rewards, dim=0).reshape((1, -1)))) + 1e-6).to(self.device, torch.float)

            # gradient descent
            optimizer.zero_grad()
            loss = - expected_reward + logZ
            loss.backward()
            U.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer.step()

            print('=MDPMFG: epoch:{}'.format(epoch) + ', loss:{}'.format(str(loss.detach().cpu().numpy())), end='\r')
            #print('MDPMFG: ' + 'Epoch ' + str(epoch) + ', ' + 'Loss ' + str(loss.detach().cpu().numpy()))

        self.reward_model = reward_model
        #torch.save(self.reward_model, './model_saved/mdp_' + self.env.name + '_' + str(len(self.data)) + '.pt')