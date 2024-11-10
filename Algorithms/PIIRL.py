"""
This is the physical informed AIRL


Remainder value type:
    1. estimated_mean_field_flow : numpy.array()

"""
import copy
import logging
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as U
import torch.nn.functional as F
from scipy.special import entr
import torch.nn as nn

from Algorithms.myModels import MeanFieldModel, RewardModel, PolicyModel
from core import Environment, State, Action, MeanField, MeanFieldFlow, PolicyFlow, Policy, IRL, Trajectory
from Environments import CARS
from Algorithms.expert_training import Expert

# from sklearn.model_selection import train_test_split

import time

'''
Constrain all the variables 
We need to make sure it is tractable
'''

MAX = 100000  # maximum number of iterations
MIN = 1e-10


class PIIRL(IRL):
    '''
        Max_epoch : 训练迭代的最大次数
        learning_rate：优化器的学习率。
        ax_grad_norm：梯度裁剪的阈值，有助于通过避免梯度爆炸来稳定训练。
        num_of_units：指定奖励和塑造模型中神经网络层的大小。
    '''

    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        warm_epoch = 0.5 * max_epoch

        # Initialize models with adjusted architectures
        reward_model = RewardModel(
            state_shape=self.env.state_shape,
            action_shape=self.env.action_shape,
            mf_shape=self.env.state_count,
            num_of_units=num_of_units
        ).to(self.device)

        policy_model = PolicyModel(
            state_shape=self.env.state_shape,
            action_shape=self.env.action_count,
            mf_shape=self.env.state_count,
            num_of_units=num_of_units
        ).to(self.device)

        # Modify MeanFieldModel to accept continuous inputs
        mean_field_model = MeanFieldModel(
            state_shape=self.env.state_shape,
            time_horizon=1,  # Adjusted to accept continuous time input
            num_of_units=num_of_units
        ).to(self.device)

        optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
        optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
        optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Initialize policy flow with random probabilities
        init_policy_flow = np.random.rand(self.horizon, self.env.state_count, self.env.action_count)
        init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)
        self.p_flow.val = init_policy_flow

        # Estimate expert mean field flow from expert data
        init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
        for sample in self.data_expert:
            for t in range(self.horizon):
                init_est_expert_mf_flow[t, int(sample.states[t])] += 1
        init_est_expert_mf_flow /= len(self.data_expert)
        self.mf_flow.val = init_est_expert_mf_flow.copy()

        epoch = 0
        last_loss = float('inf')
        count = 0

        while epoch < max_epoch:
            start_time = time.time()

            # Generate trajectories from current policy and mean field

            # Estimate mean field flow from data
            if epoch>0:
                estimated_mean_field_flow = np.zeros((self.horizon, self.env.state_count))
                for sample in self.data_expert:
                    for t in range(self.horizon):
                        # Here we only add the shown states
                        # and all the values from the NN will be a tensor value
                        estimated_mean_field_flow[t, int(sample.states[t])] += np.exp( self.mean_field_model(
                            torch.tensor(int(sample.states[t])).to(self.device, torch.float),
                                                    torch.tensor(t).to(self.device, torch.float)
                        ).item())
                for t in range(self.horizon):
                    sum_values = estimated_mean_field_flow[t].sum()
                    if sum_values > 0:
                        estimated_mean_field_flow[t] /= sum_values
            else:
                estimated_mean_field_flow = init_est_expert_mf_flow.copy()

            # Update mean field flow
            self.mf_flow.val = estimated_mean_field_flow.copy()

            self.mean_field_model = mean_field_model
            self.data_policy_theta = self.generate_trajectories_from_policy_flow(
                self.num_of_game_plays, self.num_traj, self.p_flow, self.mf_flow
            )

            # Compute losses
            value_per_sample_expert_data = []
            value_per_sample_policy_data = []

            # Compute loss from expert data
            for sample_expert in self.data_expert:
                value_per_step = []
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_expert.states[t])]).to(self.device, torch.float),
                        torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).to(self.device,
                                                                                               torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )
                    up = torch.exp(reward_component)
                    down = torch.exp(reward_component) + self.p_flow.val[
                        t, int(sample_expert.states[t]), int(sample_expert.actions[t])
                    ]
                    value_per_step.append(up / down)
                value_per_sample_expert_data.append(
                    torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1))
                )

            estimated_expert_data = torch.mean(torch.cat(value_per_sample_expert_data, dim=0).reshape((1, -1)))

            # Compute loss from policy data
            for sample_policy_theta in self.data_policy_theta:
                value_per_step = []
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_policy_theta.states[t])]).to(self.device,
                                                                                                   torch.float),
                        torch.tensor(self.env.action_option[int(sample_policy_theta.actions[t])]).to(self.device,
                                                                                                     torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )
                    up = torch.exp(reward_component)
                    down = torch.exp(reward_component) + self.p_flow.val[
                        t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])
                    ]
                    value_per_step.append(1 - (up / down))
                value_per_sample_policy_data.append(
                    torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1))
                )

            estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data, dim=0).reshape((1, -1)))

            # Update reward model
            optimizer_reward.zero_grad()
            entropy_weight = 0.01  # Adjust if necessary
            policy_entropy = -torch.sum(torch.tensor(self.p_flow.val) *
                                        torch.log(torch.tensor(self.p_flow.val) + 1e-9))
            loss = -(estimated_expert_data + estimated_policy_data) # - entropy_weight * policy_entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer_reward.step()

            # Adjust learning rate or restart if loss increases significantly
            if epoch > warm_epoch:
                distance = abs(last_loss - loss.detach().cpu().numpy()) / abs(last_loss)
                if distance > 0.1:
                    # Reset models and optimizers
                    reward_model = RewardModel(
                        state_shape=self.env.state_shape,
                        action_shape=self.env.action_shape,
                        mf_shape=self.env.state_count,
                        num_of_units=num_of_units
                    ).to(self.device)

                    policy_model = PolicyModel(
                        state_shape=self.env.state_shape,
                        action_shape=self.env.action_count,
                        mf_shape=self.env.state_count,
                        num_of_units=num_of_units
                    ).to(self.device)

                    mean_field_model = MeanFieldModel(
                        state_shape=self.env.state_shape,
                        time_horizon=1,  # Adjusted to accept continuous time input
                        num_of_units=num_of_units
                    ).to(self.device)

                    optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
                    optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)

                    optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)

                    count += 1
                    if count < int(0.8 * max_epoch):
                        epoch = 0
                        continue

                last_loss = loss.detach().cpu().numpy()
            else:
                last_loss = loss.detach().cpu().numpy()

            # Update policy model
            for t in range(self.horizon - 1, -1, -1):
                # here we calculate the sum for the current
                sum_current = []

                # here we use the formula
                for sample in self.data_policy_theta:
                    value_per_sampler = []
                    for current in range(t, self.horizon):
                        # at here we only update the previous
                        # So we can just use the "estimated_mean_field_flow" we got before
                        value_per_sampler.append(
                            reward_model(
                                torch.tensor(self.env.state_option[int(sample.states[current])]).to(self.device,
                                                                                                    torch.float),
                                torch.tensor(self.env.action_option[int(sample.actions[current])]).to(self.device,
                                                                                                      torch.float),
                                torch.from_numpy(estimated_mean_field_flow[current, :]).to(self.device, torch.float)
                            ) - torch.log(torch.tensor(
                                self.p_flow.val[current, int(sample.states[current]), int(sample.actions[current])]))
                        )
                    sum_current.append(torch.sum(torch.cat(value_per_sampler, dim=0)).reshape((1, -1)))

                # this is the loss function
                estimated_update = torch.mean(torch.sum(torch.cat(sum_current, dim=0).reshape((1, -1))))

                # FIXME 假设唯一一个分布

                # train the policy model
                optimizer_policy.zero_grad()
                loss2 = -(estimated_update)
                loss2.backward()
                U.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer_policy.step()

                # then we need to update the policy
                # for s in range(self.env.state_count):
                #     new_policy = policy_model(
                #         torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
                #         torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                #     )
                #
                #     # action_exps = torch.exp(new_policy / self.env.beta)
                #     # # Normalize and move to CPU before assigning to NumPy array
                #     # self.p_flow.val[t, s] = (action_exps / torch.sum(action_exps)).detach().cpu().numpy()
                #     tensor_list = new_policy.tolist()
                #     # Here we update the policy flow
                #     self.p_flow.val[t, s] = tensor_list

                for s in range(self.env.state_count):
                    new_policy = policy_model(
                        torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )
                    # Apply softmax to get probabilities
                    # new_policy_probs = torch.softmax(new_policy, dim=0).detach().cpu().numpy()
                    # self.p_flow.val[t, s] = new_policy_probs

                    action_exps = torch.exp(new_policy / self.env.beta)
                    # Normalize and move to CPU before assigning to NumPy array
                    self.p_flow.val[t, s] = (action_exps / torch.sum(action_exps)).detach().cpu().numpy()

                    # tensor_list = new_policy_logits.tolist()
                    # # Here we update the policy flow
                    # self.p_flow.val[t, s] = tensor_list

                # FIXME We need to fill in the parameters

                '''
                    Here we need to update trajectory
                    And we need to update the trajectory first, so that 
                        our new mean_field model is the correct 
                '''
                # self.data_policy_theta = self.generate_trajectories(self.num_of_game_plays,self.num_traj)
                self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
                                                                                     self.num_traj,
                                                                                     self.p_flow, self.mf_flow)

            # Update mean field model over the entire time horizon
            if self.env.dim == 1:
                self.train_mean_field_dim_1(
                    max_epoch, learning_rate, max_grad_norm, num_of_units,
                    mean_field_model, optimizer_meanfield
                )
            else:
                self.train_mean_field_dim_2_new(
                    max_epoch, learning_rate, max_grad_norm, num_of_units,
                    mean_field_model, optimizer_meanfield
                )

            # Update estimated mean field flow using the mean field model
            for t in range(self.horizon):
                for s in range(self.env.state_count):
                    x_input = torch.tensor(s).to(self.device, torch.float)
                    t_input = torch.tensor(t).to(self.device, torch.float)
                    estimated_mean_field_flow[t, s] = mean_field_model(x_input, t_input).detach().cpu().numpy()

                # Normalize the mean field flow to sum to one
                total = np.sum(estimated_mean_field_flow[t, :])
                if total != 0:
                    estimated_mean_field_flow[t, :] /= total
                else:
                    # Use baseline or previous mean field if total is zero
                    estimated_mean_field_flow[t, :] = init_est_expert_mf_flow[t, :]

            # Update the mean field flow
            self.mf_flow.val = estimated_mean_field_flow.copy()

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Time: {epoch_duration:.2f}s")
            epoch += 1

        # Assign the trained models
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.mean_field_model = mean_field_model

    def train_mean_field_dim_1(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                                   num_of_units: int, mean_field_model, optimizer1):
        mean_field_model.train()

        # Collect all training samples over the entire time horizon
        loss_total = []
        for sample in self.data_policy_theta:
            for t in range(self.horizon):
                x_onehot = torch.tensor([int(sample.states[t])]).to(self.device, torch.float)
                t_onehot = torch.tensor([t]).to(self.device, torch.float)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute mean fields
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_onehot + 1)  # Assume time unit is 1

                # Get neighboring states
                neighbours = self.env.get_neighbors(int(sample.states[t]))
                mean_field_last_x = []
                for neighbour in neighbours:
                    x_last = torch.tensor([int(neighbour)]).to(self.device, torch.float)
                    mean_field_last_x.append(mean_field_model(x_last, t_onehot))
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)

                # Compute expected velocities
                current_policy = self.p_flow.val[t, int(sample.states[t])]
                velocity_now = np.dot(current_policy, self.env.action_option)
                velocity_last_x = []
                for neighbour in neighbours:
                    neighbour_policy = self.p_flow.val[t, int(neighbour) % self.env.state_count]
                    expected_velocity = np.dot(neighbour_policy, self.env.action_option)
                    velocity_last_x.append(expected_velocity)
                velocity_last_x = np.mean(velocity_last_x)

                # Compute the loss using finite differences
                delta_t = self.env.time_unit  # Adjust if necessary
                delta_x = self.env.position_unit  # Adjust if necessary

                left = (mean_field_next_time - mean_field_now) / delta_t
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x

                loss = ((left + right) ** 2).mean()
                loss_total.append(loss)

        # Optimize the mean field model
        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        self.mean_field_model = mean_field_model

    # def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
    #     warm_epoch = 0.5*max_epoch
    #
    #     # def set_seed(seed):
    #     #     torch.manual_seed(seed)
    #     #     torch.cuda.manual_seed_all(seed)
    #     #     np.random.seed(seed)
    #     #     random.seed(seed)
    #     #     torch.backends.cudnn.deterministic = True
    #     #
    #     # set_seed(1)  # 选择一个固定种子
    #
    #     reward_model = RewardModel(state_shape=self.env.state_shape,
    #                                action_shape=self.env.action_shape,
    #                                mf_shape=self.env.state_count,
    #                                num_of_units=num_of_units).to(self.device)
    #
    #     policy_model = PolicyModel(state_shape=self.env.state_shape,
    #                                action_shape=self.env.action_count,
    #                                mf_shape=self.env.state_count,
    #                                num_of_units=num_of_units).to(self.device)
    #
    #     # this is the nn for the mean field value
    #     mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
    #                                       # this is the special attribute for our model
    #                                       # here is the time shape should be the 1
    #                                       time_horizon=1,
    #                                       num_of_units=num_of_units).to(self.device)
    #
    #     optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
    #     optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #     optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #
    #     # Initialize learning rate schedulers
    #     # scheduler_reward = optim.lr_scheduler.StepLR(optimizer_reward, step_size=10, gamma=0.1)
    #     # scheduler_policy = optim.lr_scheduler.StepLR(optimizer_policy, step_size=10, gamma=0.1)
    #     # scheduler_meanfield = optim.lr_scheduler.StepLR(optimizer_meanfield, step_size=10, gamma=0.1)
    #     '''
    #     Here is the init part
    #         we need to initialize the policy  at first
    #         so that we can get the trajectory from it
    #             we just randomly initialize the policy
    #     '''
    #     init_policy_flow = np.random.rand(self.horizon, self.env.state_count, self.env.action_count)
    #
    #     # 将最后一个维度的值归一化为 1
    #     init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)
    #
    #     self.p_flow.val = init_policy_flow
    #
    #     # here we use the expert meanfield to generate the trajectory
    #     init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
    #     for sample in self.data_expert:
    #         for t in range(self.horizon):
    #             init_est_expert_mf_flow[t, int(sample.states[t])] += 1
    #     init_est_expert_mf_flow /= len(self.data_expert)
    #
    #     self.mf_flow.val = init_est_expert_mf_flow.copy()
    #
    #
    #     count = 0
    #     epoch = 0
    #
    #     while epoch < max_epoch:
    #
    #         start_time = time.time()
    #
    #         '''
    #         1. we get the meanfield_alpha from current policy
    #             we store our policy in the self as an attribute
    #
    #         2. we also need the new trajectory which is induced by current policy
    #         '''
    #
    #         # TODO This is the replace part for the
    #         # TODO and we store all the tensor value in it
    #         # after this we get the mean filed flow from current policy and trajectory
    #         estimated_mean_field_flow = np.zeros((self.horizon, self.env.state_count))
    #         self.mean_field_model = mean_field_model
    #         self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
    #                                                                                  self.num_traj, self.p_flow,
    #                                                                                  self.mf_flow)
    #
    #         # self.data_policy_theta = self.generate_trajectories(self.num_of_game_plays, self.num_traj)
    #
    #         # For we do have the trained value before we train our meanfield
    #         # So we use the estimated value from the expert for the first round
    #
    #         if epoch > 0:  # this means that we are not in the first round
    #             for sample in self.data_policy_theta:
    #                 for t in range(self.horizon):
    #                     # here we only add the show states
    #                     # and all the value form the NN will be a tensor value
    #                     estimated_mean_field_flow[t, int(sample.states[t])] += self.mean_field_model(
    #                         torch.tensor(self.env.state_option[int(sample.states[t])]).to(self.device, torch.float),
    #                         torch.tensor(t).to(self.device, torch.float)
    #                     )
    #             for t in range(self.horizon):
    #                 sum_values = estimated_mean_field_flow[t].sum()
    #                 if sum_values > 0:
    #                     estimated_mean_field_flow[t] /= sum_values
    #         else:
    #             estimated_mean_field_flow = init_est_expert_mf_flow.copy()
    #
    #         # change the new meanfield to our current
    #         self.mf_flow.val = estimated_mean_field_flow
    #
    #         # after get the init mf then we sample the trajectory
    #
    #
    #         #self.data_policy_theta = self.generate_trajectories(self.num_of_game_plays, self.num_traj)
    #
    #         # these 2 list is to store the D value from expert and policy
    #         value_per_sample_expert_data = []
    #         value_per_sample_policy_data = []
    #
    #         '''
    #         First part of our D
    #             which is induce from the expert trajectory
    #         '''
    #         # here are the expert value
    #         for sample_expert in self.data_expert:
    #             value_per_step = []
    #             value_reward_exp = []
    #
    #             # At here we create the mean field flow for every time
    #             for t in range(self.horizon):
    #                 reward_component = reward_model(
    #                     torch.tensor(self.env.state_option[int(sample_expert.states[t])]).to(self.device, torch.float),
    #                     torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).to(self.device,
    #                                                                                            torch.float),
    #                     torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
    #                 )
    #
    #                 up = torch.exp(reward_component)
    #
    #                 # here we add the policy for this state and action
    #                 down = torch.exp(reward_component) + self.p_flow.val[
    #                     t, int(sample_expert.states[t]), int(sample_expert.actions[t])]
    #
    #                 # here is the D for this (s_t,a_t)
    #                 value_per_step.append(up / down)
    #
    #             # here is the sum of the log D
    #             # print(torch.cat(value_per_step, dim=0).reshape((1, -1)))
    #             # print(torch.log(torch.cat(value_per_step, dim=0).reshape((1, -1))))
    #             # print(torch.sum(torch.log(torch.cat(value_per_step, dim=0).reshape((1, -1)))))
    #             value_per_sample_expert_data.append(
    #                 torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
    #
    #         # print(value_per_sample_expert_data)
    #         # print(torch.cat(value_per_sample_expert_data))
    #         # we use mean to get the estimated value
    #         estimated_expert_data = torch.mean(torch.cat(value_per_sample_expert_data, dim=0).reshape((1, -1)))
    #
    #         '''
    #         Here is the second part
    #             which is induced from the policy trajectory
    #         '''
    #         for sample_policy_theta in self.data_policy_theta:
    #             value_per_step = []
    #             value_reward_exp = []
    #             for t in range(self.horizon):
    #                 # here we give addition [] for torch.cat in the nn
    #                 reward_component = reward_model(
    #                     torch.tensor(self.env.state_option[int(sample_policy_theta.states[t])]).to(self.device,
    #                                                                                                torch.float),
    #                     torch.tensor(self.env.action_option[int(sample_policy_theta.actions[t])]).to(self.device,
    #                                                                                                  torch.float),
    #                     torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
    #                 )
    #
    #                 up = torch.exp(reward_component)
    #
    #                 # here we add the policy for this state and action
    #                 down = torch.exp(reward_component) + self.p_flow.val[
    #                     t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])]
    #
    #                 # here is the 1-D for this (s_t,a_t)
    #                 value_per_step.append(1 - (up / down))
    #                 value_reward_exp.append(reward_component)
    #
    #             # log(1-D) sum for policy_theta on this trajectory
    #             value_per_sample_policy_data.append(
    #                 torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
    #
    #         # same as above
    #         estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data, dim=0).reshape((1, -1)))
    #
    #         # then we need to use the gradient to maximum the sum of these 2
    #         # So we use it to minmize the - of the sum
    #
    #         optimizer_reward.zero_grad()
    #         # 含熵正则化的策略损失
    #         entropy_weight = 0.01  # 可根据需要调整
    #         policy_entropy = -torch.sum(torch.tensor(self.p_flow.val) *
    #                                     torch.log(torch.tensor(self.p_flow.val) + 1e-9))
    #         loss = -(estimated_expert_data + estimated_policy_data) - entropy_weight * policy_entropy
    #         # loss = -(estimated_expert_data + estimated_policy_data)
    #
    #         # loss = - (torch.mean(torch.cat([estimated_policy_data_reward,estimated_expert_reward],dim=0)))
    #         loss.backward()
    #         U.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
    #         optimizer_reward.step()
    #
    #         if epoch> 0.5*max_epoch :
    #             distance = abs(last_loss - loss.detach().cpu().numpy()) / last_loss
    #             if distance > 0.1:
    #                 reward_model = RewardModel(state_shape=self.env.state_shape,
    #                                            action_shape=self.env.action_shape,
    #                                            mf_shape=self.env.state_count,
    #                                            num_of_units=num_of_units).to(self.device)
    #                 policy_model = PolicyModel(state_shape=self.env.state_shape,
    #                                            action_shape=self.env.action_count,
    #                                            mf_shape=self.env.state_count,
    #                                            num_of_units=num_of_units).to(self.device)
    #                 optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
    #                 optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #
    #
    #                 # for param_group in optimizer_reward.param_groups:
    #                 #     param_group['lr'] *= 0.5  # Reduce learning rate by half
    #                 # for param_group in optimizer_policy.param_groups:
    #                 #     param_group['lr'] *= 0.5
    #                 # print(f"High loss change detected. Reducing learning rate to {param_group['lr']}")
    #
    #                 # last_loss = loss.detach().cpu().numpy()
    #                 count += 1
    #                 print(count)
    #                 print(int(0.8*max_epoch))
    #                 if count < int(0.8*max_epoch):
    #                     epoch = 0
    #                     continue
    #
    #
    #             last_loss = loss.detach().cpu().numpy()
    #
    #
    #         else:
    #             last_loss = loss.detach().cpu().numpy()
    #
    #
    #         # TODO where should i do the policy update ?
    #         #  But how the trajectory change while we update the policy
    #         # here we update the policy
    #
    #         for t in range(self.horizon - 1, -1, -1):
    #             # here we calculate the sum for the current
    #             sum_current = []
    #
    #             # here we use the formula
    #             for sample in self.data_policy_theta:
    #                 value_per_sampler = []
    #                 for current in range(t, self.horizon):
    #                     # at here we only update the previous
    #                     # So we can just use the "estimated_mean_field_flow" we got before
    #                     value_per_sampler.append(
    #                         reward_model(
    #                             torch.tensor(self.env.state_option[int(sample.states[current])]).to(self.device,
    #                                                                                                 torch.float),
    #                             torch.tensor(self.env.action_option[int(sample.actions[current])]).to(self.device,
    #                                                                                                   torch.float),
    #                             torch.from_numpy(estimated_mean_field_flow[current, :]).to(self.device, torch.float)
    #                         ) - torch.log(torch.tensor(
    #                             self.p_flow.val[current, int(sample.states[current]), int(sample.actions[current])]))
    #                     )
    #                 sum_current.append(torch.sum(torch.cat(value_per_sampler, dim=0)).reshape((1, -1)))
    #
    #             # this is the loss function
    #             estimated_update = torch.mean(torch.sum(torch.cat(sum_current, dim=0).reshape((1, -1))))
    #
    #             # FIXME 假设唯一一个分布
    #
    #             # train the policy model
    #             optimizer_policy.zero_grad()
    #             loss2 = -(estimated_update)
    #             loss2.backward()
    #             U.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
    #             optimizer_policy.step()
    #
    #             # then we need to update the policy
    #             for s in range(self.env.state_count):
    #                 new_policy = policy_model(
    #                     torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
    #                     torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
    #                 )
    #
    #                 # action_exps = torch.exp(new_policy / self.env.beta)
    #                 # # Normalize and move to CPU before assigning to NumPy array
    #                 # self.p_flow.val[t, s] = (action_exps / torch.sum(action_exps)).detach().cpu().numpy()
    #                 tensor_list = new_policy.tolist()
    #                 # Here we update the policy flow
    #                 self.p_flow.val[t, s] = tensor_list
    #
    #             self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
    #                                                                                  self.num_traj, self.p_flow,
    #                                                                                  self.mf_flow)
    #
    #
    #         history_velocities = [[[] for _ in range(self.env.state_count)] for _ in
    #                               range(self.horizon)]
    #
    #         for t in range(0, self.horizon):
    #             if self.env.dim == 1:
    #                 self.train_mean_field_dim_1_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
    #                                                 history_velocities, mean_field_model, optimizer_meanfield, t)
    #             elif self.env.dim == 2:
    #                 self.train_mean_field_dim_2_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
    #                                                 history_velocities, mean_field_model, optimizer_meanfield, t)
    #
    #             # # Evaluate on the test dataset
    #             # test_loss = self.evaluate_mean_field_model(mean_field_model, history_velocities, t)
    #             # print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    #
    #             # here is the code which we used to update the mean field value
    #
    #             for s in range(self.env.state_count):
    #                 new_mean_field = self.mean_field_model(
    #                     torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
    #                     torch.tensor(t).to(self.device, torch.float)
    #                 )
    #
    #                 estimated_mean_field_flow[t, s] = new_mean_field
    #
    #             total = sum(estimated_mean_field_flow[t, :])
    #             if total != 0:
    #                 estimated_mean_field_flow[t, :] /= total
    #             else:
    #                 raise ValueError("Sum of values is zero, cannot normalize")
    #
    #             self.mf_flow.val = estimated_mean_field_flow
    #             self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
    #                                                                                  self.num_traj, self.p_flow,
    #                                                                                  self.mf_flow)
    #
    #
    #
    #             # self.data_policy_theta = self.generate_trajectories(self.num_of_game_plays, self.num_traj)
    #
    #             # self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
    #             #                                                                      self.num_traj, self.p_flow,
    #             #                                                                      self.mf_flow)
    #
    #             # FIXME We need to fill in the parameters
    #             # we also need to update the meanfield
    #             # and we need to judge if it is more than 2 dim
    #             # if self.env.dim == 1:
    #             #     self.train_mean_field_dim_1_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
    #             #                                     history_velocities, mean_field_model, optimizer_meanfield, t)
    #             # elif self.env.dim == 2:
    #             #     self.train_mean_field_dim_2_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
    #             #                                     history_velocities, mean_field_model, optimizer_meanfield, t)
    #             #
    #             # # # Evaluate on the test dataset
    #             # # test_loss = self.evaluate_mean_field_model(mean_field_model, history_velocities, t)
    #             # # print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    #             #
    #             # # here is the code which we used to update the mean field value
    #             #
    #             # for s in range(self.env.state_count):
    #             #     new_mean_field = self.mean_field_model(
    #             #         torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
    #             #         torch.tensor(t).to(self.device, torch.float)
    #             #     )
    #             #
    #             #     estimated_mean_field_flow[t, s] = new_mean_field
    #             #
    #             # total = sum(estimated_mean_field_flow[t, :])
    #             # if total != 0:
    #             #     estimated_mean_field_flow[t, :] /= total
    #             # else:
    #             #     raise ValueError("Sum of values is zero, cannot normalize")
    #             #
    #             # self.mf_flow.val = estimated_mean_field_flow
    #
    #
    #
    #                 # self.data_policy_theta = self.generate_trajectories(self.num_of_game_plays,self.num_traj)
    #
    #
    #         end_time = time.time()
    #         epoch_duration = end_time - start_time
    #
    #         print(f"Time taken for one epoch: {epoch_duration:.4f} seconds")
    #         print('=PIIRL: epoch:{}'.format(epoch) + ', loss:{}'.format(str(loss.detach().cpu().numpy())))
    #         if np.isinf(loss.detach().cpu().numpy()):
    #             reward_model = RewardModel(state_shape=self.env.state_shape,
    #                                        action_shape=self.env.action_shape,
    #                                        mf_shape=self.env.state_count,
    #                                        num_of_units=num_of_units).to(self.device)
    #             policy_model = PolicyModel(state_shape=self.env.state_shape,
    #                                        action_shape=self.env.action_count,
    #                                        mf_shape=self.env.state_count,
    #                                        num_of_units=num_of_units).to(self.device)
    #             mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
    #                                               # this is the special attribute for our model
    #                                               # here is the time shape should be the 1
    #                                               time_horizon=1,
    #                                               num_of_units=num_of_units).to(self.device)
    #             optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate)
    #             optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
    #             optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #
    #             epoch = 0
    #             continue
    #
    #
    #         epoch += 1
    #         # print(epoch)
    #         # self.logger.info('=PIIRL: epoch:{}, loss:{}'.format(epoch, str(loss.detach().cpu().numpy())))
    #
    #     # send the most optimal back
    #     # this is the last
    #     self.reward_model = reward_model
    #     self.policy_model = policy_model
    #     self.mean_field_model = mean_field_model
    #
    # '''
    # This is the physics-informed which is fitted for the 1-d
    #     For we just consider the before position and time
    #
    # We should consider more if we consider 2-d
    #     Like last position could be around (like 9 points around)
    #     but the time should still be last
    # '''
    #
    # def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
    #                                num_of_units: int, mean_field_model, optimizer1, start_time):
    #     max_phi = 5
    #
    #     mean_field_model.train()
    #
    #     for sample in self.data_policy_theta:
    #         loss_total = []
    #
    #         for t in range(start_time, self.horizon):
    #             x_onehot = torch.tensor([self.env.state_option[int(sample.states[t])]]).to(self.device, torch.float)
    #             t_onehot = torch.tensor([t]).to(self.device, torch.float)
    #
    #             # Get the policy at time t for the current state
    #             current_policy = self.p_flow.val[t, :]
    #
    #             neighbours = self.env.get_neighbors(int(sample.states[t]))
    #             x_last_around = []
    #             for neighbour in neighbours:
    #                 x_last_around.append(
    #                     torch.tensor(
    #                         self.env.state_option[int(neighbour)]).to(self.device, torch.float))
    #
    #             t_next_onehot = torch.tensor([(t + 1) % self.horizon]).to(self.device, torch.float)
    #
    #             # Set requires_grad to True
    #             x_onehot.requires_grad = True
    #             t_onehot.requires_grad = True
    #
    #             # Compute mean fields
    #             mean_field_now = mean_field_model(x_onehot, t_onehot)
    #             mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)
    #             mean_field_last_x = []
    #             for x_last in x_last_around:
    #                 mean_field_last_x.append(
    #                     mean_field_model(x_last, t_onehot)
    #                 )
    #             mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)
    #
    #             # Compute expected velocities
    #             velocity_now = np.dot(current_policy[int(sample.states[t])], self.env.action_option)
    #             velocity_last_x = []
    #             for neighbour in neighbours:
    #                 neighbour_policy = current_policy[int(neighbour) % self.env.state_count]
    #                 expected_velocity = np.dot(neighbour_policy, self.env.action_option)
    #                 velocity_last_x.append(expected_velocity)
    #             velocity_last_x = np.mean(velocity_last_x)
    #
    #             # Compute the custom loss using finite differences
    #             phi = np.random.uniform(1, max_phi)
    #             delta_t = phi * self.env.time_unit
    #             delta_x = phi * self.env.position_unit
    #
    #             left = (mean_field_model(x_onehot, t_onehot + delta_t) - mean_field_now) / delta_t
    #             right = ((mean_field_now * velocity_now) - (
    #                     mean_field_model(x_onehot - delta_x, t_onehot) * velocity_last_x)) / delta_x
    #
    #             loss = ((left + right) ** 2).mean()
    #
    #             loss_total.append(loss)
    #
    #         # Aggregate losses and optimize
    #         residual = torch.stack(loss_total).mean()
    #         optimizer1.zero_grad()
    #         residual.backward()
    #         torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
    #         optimizer1.step()
    #
    #     self.mean_field_model = mean_field_model

    # def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
    #                                num_of_units: int, history_velocities, mean_field_model, optimizer1, start_time):
    #     n_cell = self.env.state_count
    #     T_terminal = self.horizon
    #     u = self.p_flow.val  # Policy probabilities: shape (horizon, state_count, action_count)
    #     d = np.zeros(n_cell)
    #     option = "ring"  # Assuming option is "ring"
    #     n_iterations = max_epoch  # Number of training iterations for the mean field model
    #
    #     # Extract position and time units
    #     delta_x = self.env.position_unit
    #     delta_t = self.env.time_unit
    #
    #     # Compute lambda = delta_t / delta_x
    #     lambda_ = delta_t / delta_x
    #
    #     # Initialize the initial distribution d (e.g., uniform distribution)
    #     d[:] = 1.0 / n_cell
    #
    #     mean_field_model.train()
    #
    #     # Prepare the states and rho_values lists
    #     states = []
    #     rho_values = []
    #
    #     T = T_terminal
    #     for t in range(T):
    #         for i in range(n_cell):
    #             # Normalize positions and times for NN input
    #             x_i = i / n_cell  # Normalized position between 0 and 1
    #             t_i = t / T  # Normalized time between 0 and 1
    #
    #             # Append the normalized state and time as a tensor
    #             state_input = torch.tensor([x_i, t_i]).to(self.device, torch.float)
    #             states.append(state_input)
    #
    #             # Initialize rho_values based on the discretized PDE
    #             if t == 0:
    #                 # At initial time, rho_values are given by d[i]
    #                 rho_value = torch.tensor(d[i]).to(self.device, torch.float)
    #                 rho_values.append(rho_value)
    #             else:
    #                 # For t > 0, compute rho_values using the discretized PDE
    #                 # Handle boundary conditions
    #                 if i == 0 and option == "ring":
    #                     i_minus_1 = n_cell - 1
    #                 else:
    #                     i_minus_1 = i - 1
    #
    #                 # Compute rho_network outputs at required positions and times
    #                 x_i_t_minus_1 = torch.tensor([i / n_cell, (t - 1) / T]).to(self.device, torch.float)
    #                 x_i_minus_1_t_minus_1 = torch.tensor([i_minus_1 / n_cell, (t - 1) / T]).to(self.device, torch.float)
    #
    #                 rho_i_t_minus_1 = mean_field_model(x_i_t_minus_1)
    #                 rho_i_minus_1_t_minus_1 = mean_field_model(x_i_minus_1_t_minus_1)
    #
    #                 # Get policy distributions at previous time step
    #                 policy_i_t_minus_1 = u[t - 1, i]  # Shape: (action_count,)
    #                 policy_i_minus_1_t_minus_1 = u[t - 1, i_minus_1]
    #
    #                 # Expected action values (u) at previous time step
    #                 action_values = self.env.action_option  # Action values
    #                 action_values_tensor = torch.tensor(action_values).to(self.device, torch.float)
    #
    #                 u_i_t_minus_1 = torch.dot(torch.tensor(policy_i_t_minus_1).to(self.device, torch.float),
    #                                           action_values_tensor)
    #                 u_i_minus_1_t_minus_1 = torch.dot(
    #                     torch.tensor(policy_i_minus_1_t_minus_1).to(self.device, torch.float),
    #                     action_values_tensor)
    #
    #                 # Compute the "true" rho value at position i and time t using the adjusted formula
    #                 rho_value = rho_i_t_minus_1 - lambda_ * (
    #                             rho_i_t_minus_1 * u_i_t_minus_1 - rho_i_minus_1_t_minus_1 * u_i_minus_1_t_minus_1)
    #
    #                 rho_values.append(rho_value)
    #
    #     # Now train the mean field model using the computed rho_values
    #     for iteration in range(n_iterations):
    #         total_loss = 0.0
    #         optimizer1.zero_grad()
    #
    #         # Loop over all states and rho_values
    #         for state_input, rho_value in zip(states, rho_values):
    #             pred_rho = mean_field_model(state_input)
    #             loss = (rho_value - pred_rho).abs()
    #             loss.backward(retain_graph=True)
    #             total_loss += loss.item()
    #
    #         # Perform optimization step
    #         U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
    #         optimizer1.step()
    #
    #         # Optionally, print the loss for monitoring
    #         print(f'Mean Field Model Training Iteration {iteration + 1}/{n_iterations}, Loss: {total_loss}')
    #
    #     # Update the mean field model in the class
    #     self.mean_field_model = mean_field_model


    def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                                   num_of_units: int, history_velocities, mean_field_model, optimizer1, start_time):
        #
        # # this is the nn for the mean field value
        # mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
        #                                   # this is the special attribute for our model
        #                                   # here is the time shape should be the 1
        #                                   time_horizon=1,
        #                                   num_of_units=num_of_units).to(self.device)
        # optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate)
        max_phi = 5

        mean_field_model.train()

        for sample in self.data_policy_theta:
            # Initialize a list to store historical velocities

            loss_total = []

            for t in range(start_time, self.horizon):
                x_onehot = torch.tensor([self.env.state_option[int(sample.states[t])]]).to(self.device, torch.float)
                t_onehot = torch.tensor([t]).to(self.device, torch.float)

                # Get the policy at time t for the current state
                current_policy = self.p_flow.val[t, :]

                neighbours = self.env.get_neighbors(int(sample.states[t]))
                x_last_around = []
                action_index_last = []
                for neighbour in neighbours:
                    x_last_around.append(
                        torch.tensor(
                            self.env.state_option[int(neighbour)]).to(self.device, torch.float))
                    action_index_last.append(
                        np.argmax(current_policy[int((neighbour) % self.env.state_count)])
                    )

                # x_last_onehot = torch.tensor(
                #     [self.env.state_option[int(sample.states[t] - 1) % self.env.state_count]]).to(self.device,
                #                                                                                   torch.float)
                t_next_onehot = torch.tensor([(t + 1) % self.horizon]).to(self.device, torch.float)

                # Set requires_grad to True for tensors where gradients are needed
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute mean fields for current, next time step, and last x position
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)
                # mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)
                mean_field_last_x = []
                for x_last in x_last_around:
                    mean_field_last_x.append(
                        mean_field_model(x_last, t_onehot)
                    )
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)



                # Compute expected velocity at time t
                velocity_now = self.env.action_option[(np.argmax(current_policy[int(sample.states[t])]))]
                # action_index_last = np.argmax(current_policy[int((sample.states[t] - 1) % self.env.state_count)])

                velocity_last_x = []
                for action_index in action_index_last:
                    velocity_last_x.append(self.env.action_option[action_index])
                velocity_last_x = sum(velocity_last_x) / len(velocity_last_x)

                history_velocities[t][int((sample.states[t]) % self.env.state_count)].append(
                    velocity_last_x)

                # Compute mean of historical velocities
                if history_velocities[t][int((sample.states[t]) % self.env.state_count)]:
                    mean_velocity = np.mean(history_velocities[t][int((sample.states[t]) % self.env.state_count)])
                    # Find the closest value in self.env.action_option to mean_velocity
                    velocity_last_x = mean_velocity

                # Compute the custom loss using the given formula

                # phi = np.random.uniform(1, max_phi)
                # delta_t = phi * self.env.time_unit
                # delta_x = phi * self.env.position_unit
                #
                # left = (mean_field_model(x_onehot, t_onehot + delta_t) - mean_field_now) / delta_t
                # right = ((mean_field_now * velocity_now) - (
                #             mean_field_model(x_onehot - delta_x, t_onehot) * velocity_last_x)) / delta_x

                left = (mean_field_next_time - mean_field_now) / self.env.time_unit
                right = ((mean_field_now * velocity_now) - (
                        mean_field_last_x * velocity_last_x)) / self.env.position_unit
                loss = ((left + right)**2).mean()

                # Add regularization term
                # reg_loss = 1e-4 * torch.norm(mean_field_now, p=2)
                # total_loss = torch.mean(loss) + reg_loss
                loss_total.append(loss)  # Ensure total_loss is at least 1


            # Aggregate losses from the trajectory and perform a single optimization step per sample
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step


        # Log the loss
        # print('=Mean_Field: , loss:{}'.format(str(residual.detach().cpu().numpy())), end='\r')

        self.mean_field_model = mean_field_model






    def train_mean_field_dim_2_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,
                                   history_velocities, mean_field_model, optimizer1):

        # # this is the nn for the mean field value
        # mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
        #                                   # this is the special attribute for our model
        #                                   # here is the time shape should be the 1
        #                                   time_horizon=1,
        #                                   num_of_units=num_of_units).to(self.device)
        # optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate)
        mean_field_model.train()

        for sample in self.data_policy_theta:
            # get the mean field for this time and states
            # This will give us all the mean field for this sample through our model
            '''
            For we only need the x-position and t-time for our model 
            '''

            # here is used to store all the loss on this trajectory
            # for we only update the policy on the whole trajectory

            loss_total = []

            # Initialize a list to store losses for each trajectory in the sample

            for t in range(self.horizon):

                # then also need to record the policy
                # this will give us the pi^theta_t
                # TODO we get the probability of actions in this time
                # TODO But for we are deterministic, so it will be all 0 but 1 for one action
                # TODO we assume that it has been one_hot well
                current_policy = self.p_flow.val[t, :]

                x_onehot = torch.tensor(self.env.state_option[int(sample.states[t])]).to(self.device, torch.float)
                t_onehot = torch.tensor(t).to(self.device, torch.float)

                # x_last could be the around
                # here we need to get all the neighbours around it

                # x_last_onehot = torch.tensor(
                #     self.env.state_option[int(sample.states[t] - 1) % self.env.state_count]).to(self.device,
                #                                                                                 torch.float)

                neighbours = self.env.get_neighbors(int(sample.states[t]))
                x_last_around = []
                action_index_last = []
                for neighbour in neighbours:
                    x_last_around.append(
                        torch.tensor(
                            self.env.state_option[int(neighbour)]).to(self.device, torch.float))
                    action_index_last.append(
                        np.argmax(current_policy[int((neighbour) % self.env.state_count)])
                    )

                t_next_onehot = torch.tensor((t + 1) % self.horizon).to(self.device, torch.float)

                # Set requires_grad to True for tensors where gradients are needed
                # x_onehot.requires_grad = True
                # t_onehot.requires_grad = True

                # Compute mean fields for current, next time step, and last x position
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)

                # mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)

                # here we get all the last mean field
                mean_field_last_x = []
                for x_last in x_last_around:
                    mean_field_last_x.append(
                        mean_field_model(x_last, t_onehot)
                    )
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)

                # then we need to get the policy which is the velocity
                # And we use the mean velocity to represent
                # policy is s[a[]] like this
                # and we have every probability for every action
                '''
                    we need the max one 
                '''
                action_index_now = np.argmax(current_policy[int(sample.states[t])])

                # here like the above x_last we still need to get the action last

                # action_index_last = np.argmax(current_policy[int((sample.states[t] - 1) % self.env.state_count)])

                # Calculate velocity for current and last x position
                velocity_now = self.encode_length_and_angle(self.env.action_option[action_index_now])

                velocity_last_x = []
                for action_index in action_index_last:
                    velocity_last_x.append(self.encode_length_and_angle(self.env.action_option[action_index]))
                velocity_last_x = sum(velocity_last_x) / len(velocity_last_x)

                # history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)].append(velocity_last_x)
                #
                # # Compute mean of historical velocities
                # if history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)]:
                #     mean_velocity = np.mean(history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)])
                #     # Find the closest value in self.env.action_option to mean_velocity
                #     velocity_last_x = self.encode_length_and_angle(min(
                #         self.env.action_option,
                #         key=lambda x: abs(self.encode_length_and_angle(x) - mean_velocity)
                #     ))
                '''
                Here we change our residual calculation process:
                    we choose use the delt_t and delt_x to get the residual
                    For we can not add 2 value with different dimension 

                And we let the random step size fai = 1 
                '''

                # Compute the custom loss using the given formula
                left = (mean_field_next_time - mean_field_now) / self.env.time_unit
                right = ((mean_field_now * velocity_now) - (
                        mean_field_last_x * velocity_last_x)) / self.env.position_unit
                loss = (left + right).abs()
                loss_total.append(loss)

                # Store velocity_now for the next iteration
                velocity_last_prev = velocity_last_x

            # Aggregate losses from the trajectory and perform a single optimization step per sample
            residual = torch.mean(torch.cat((loss_total), dim=0).reshape((1, -1)))
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step

            # Log the loss
            # print('=Mean_Field: epoch:, loss:{}'.format(str(residual.detach().cpu().numpy())),end='\r')
            # self.logger.info(f'=Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')

            # print()  # for better formatting of print output

        self.mean_field_model = mean_field_model

    def encode_length_and_angle(self, velocity):
        # 计算模
        r = np.linalg.norm(velocity)
        # 计算角度
        theta = np.arctan2(velocity[1], velocity[0])
        # 将角度映射到 [-1, 1] 并计算编码值
        encoded_value = r * np.cos(theta)
        return encoded_value

    # 示例

    def train_mean_field_dim_2(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,
                               mean_field_model):

        # this is the optimizer which is the actual runner
        optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate)

        for sample in self.data_policy_theta:
            # get the mean field for this time and states
            # This will give us all the mean field for this sample through our model
            '''
            For we only need the x-position and t-time for our model 
            '''

            # here is used to store all the loss on this trajectory
            # for we only update the policy on the whole trajectory

            loss_total = []

            # Initialize a list to store losses for each trajectory in the sample

            for t in range(self.horizon):

                # then also need to record the policy
                # this will give us the pi^theta_t
                # TODO we get the probability of actions in this time
                # TODO But for we are deterministic, so it will be all 0 but 1 for one action
                # TODO we assume that it has been one_hot well
                current_policy = self.p_flow.val[t, :]

                x_onehot = torch.tensor(self.env.state_option[int(sample.states[t])]).to(self.device, torch.float)
                t_onehot = torch.tensor(t).to(self.device, torch.float)

                # x_last could be the around
                # here we need to get all the neighbours around it

                # x_last_onehot = torch.tensor(
                #     self.env.state_option[int(sample.states[t] - 1) % self.env.state_count]).to(self.device,
                #                                                                                 torch.float)

                neighbours = self.env.get_neighbors(int(sample.states[t]))
                x_last_around = []
                action_index_last = []
                for neighbour in neighbours:
                    x_last_around.append(
                        torch.tensor(
                            self.env.state_option[int(neighbour)]).to(self.device, torch.float))
                    action_index_last.append(
                        np.argmax(current_policy[int((neighbour) % self.env.state_count)])
                    )

                t_next_onehot = torch.tensor((t + 1) % self.horizon).to(self.device, torch.float)

                # Set requires_grad to True for tensors where gradients are needed
                # x_onehot.requires_grad = True
                # t_onehot.requires_grad = True

                # Compute mean fields for current, next time step, and last x position
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)

                # mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)

                # here we get all the last mean field
                mean_field_last_x = []
                for x_last in x_last_around:
                    mean_field_last_x.append(
                        mean_field_model(x_last, t_onehot)
                    )
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)

                # then we need to get the policy which is the velocity
                # And we use the mean velocity to represent
                # policy is s[a[]] like this
                # and we have every probability for every action
                '''
                    we need the max one 
                '''
                action_index_now = np.argmax(current_policy[int(sample.states[t])])

                # here like the above x_last we still need to get the action last

                # action_index_last = np.argmax(current_policy[int((sample.states[t] - 1) % self.env.state_count)])

                # Calculate velocity for current and last x position
                velocity_now = np.linalg.norm(self.env.action_option[action_index_now])

                velocity_last_x = []
                for action_index in action_index_last:
                    velocity_last_x.append(np.linalg.norm(self.env.action_option[action_index]))
                velocity_last_x = sum(velocity_last_x) / len(velocity_last_x)

                # velocity_last_x = np.linalg.norm(self.env.action_option[action_index_last])

                # time do not need one-hot

                # then we need to train this one
                # Compute gradients with autograd
                # mu_t = torch.autograd.grad(mean_field_now, torch.tensor([t], dtype=torch.float32, requires_grad=True),
                #                            grad_outputs=torch.ones_like(mean_field_now),
                #                            create_graph=True)[0]
                #
                # # mu * policy and its gradient
                # product = mean_field_now * velocity_now
                # product_x = torch.autograd.grad(product, torch.tensor([sample.states[t]], dtype=torch.float32, requires_grad=True),
                #                     grad_outputs=torch.ones_like(product),
                #                     create_graph=True)[0]

                '''
                Here we change our residual calculation process:
                    we choose use the delt_t and delt_x to get the residual
                    For we can not add 2 value with different dimension 

                And we let the random step size fai = 1 
                '''

                # Compute the custom loss using the given formula
                left = (mean_field_next_time - mean_field_now) / self.env.time_unit
                right = ((mean_field_now * velocity_now) - (
                        mean_field_last_x * velocity_last_x)) / self.env.position_unit
                loss = (left + right).abs()
                loss_total.append(loss)

            # Aggregate losses from the trajectory and perform a single optimization step per sample
            residual = torch.mean(torch.cat((loss_total), dim=0).reshape((1, -1)))
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step

            # Log the loss
            # print('=Mean_Field: epoch:{}' + ', loss:{}'.format(str(residual.detach().cpu().numpy())),
            #       end='\r')
            # self.logger.info(f'=Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')

            # print()  # for better formatting of print output

        self.mean_field_model = mean_field_model

    '''
    This is the generate method which can give the 
        trajectory under current mean field and policy 
        we need to use this method whenever we update the policy
    '''

    def generate_trajectories_from_policy_flow(self, num_game_play: int, num_traj: int, current_policy_flow,
                                               current_mean_field_flow, deterministic=False):
        states = [i for i in range(self.env.state_count)]  # State space
        actions = [i for i in range(self.env.action_count)]  # Action space
        assert (current_mean_field_flow is not None) and (self.p_flow is not None)

        data = [Trajectory(states=None, actions=None, horizon=self.horizon) for _ in range(num_game_play * num_traj)]

        for i in range(num_game_play * num_traj):
            # Sample the initial state (possibly from a given initial distribution)
            # and we need to change this numpy.int to int
            s = int(np.random.choice(states, 1, p=current_mean_field_flow.val[0, :])[0])
            data[i].states[0] = s

            for t in range(self.horizon):
                # Sample action based on the current policy flow
                # For we have all the 0 but 1, so that the argmax can get
                # Still we need the int type

                current_policy_flow.val[t, s, :] = (current_policy_flow.val[t, s, :]/
                                                    sum(current_policy_flow.val[t, s, :]))

                a = np.argmax(current_policy_flow.val[t, s, :]) if deterministic else \
                    np.random.choice(actions, 1, p=current_policy_flow.val[t, s, :])[0]

                a = int(a)

                # print(type(a))
                # print(type(int(a)))
                # print(type(s))
                data[i].actions[t] = a

                # Compute the next state based on the current state and action
                # Assuming the environment has a method `next_state` to compute this
                if t < self.horizon - 1:  # Check to prevent indexing error on the last step
                    s_next = self.env.dynamics(State(state=s), Action(action=a), MeanField(
                        mean_field=current_mean_field_flow.val[
                            t]))  # Update this method as per your environment dynamics
                    data[i].states[t + 1] = s_next.val[0]
                    s = int(s_next.val[0])  # Update current state to the next state

        return data

    def generate_trajectories(self, num_game_play: int, num_traj: int):
        states = [i for i in range(self.env.state_count)]
        actions = [i for i in range(self.env.action_count)]
        assert (self.mf_flow is not None) and (self.p_flow is not None)
        data = [Trajectory(states=None, actions=None, horizon=self.horizon) for _ in range(num_game_play * num_traj)]
        for i in range(num_game_play * num_traj):
            for t in range(self.horizon):
                # print(self.mf_flow.val[t, :])
                '''
                According to the mean_field, we select the state randomly from states
                '''
                # FIXME : Here we let the output from the numpy to the int value
                # FIXME: original : s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])
                s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])[0]
                # print(self.p_flow.val[t, s, :][0])

                # FIXME We change the code here from p=self.p_flow.val[t, s, :][0]
                # FIXME to p=self.p_flow.val[t, s, :])[0] PUT THE [0] out side
                self.p_flow.val[t, s, :] = self.p_flow.val[t, s, :] / (sum(self.p_flow.val[t, s, :]))
                a = np.random.choice(actions, 1, p=self.p_flow.val[t, s, :])[0]
                data[i].states[t] = s
                data[i].actions[t] = a
        return data



    def train_mean_field_dim_1_new_2(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                                   num_of_units: int, history_velocities, mean_field_model, optimizer1, start_time,init_mf_flow):

        # this is the nn for the mean field value
        mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
                                          # this is the special attribute for our model
                                          # here is the time shape should be the 1
                                          time_horizon=1,
                                          num_of_units=num_of_units).to(self.device)
        optimizer1 = optim.Adam(mean_field_model.parameters(), lr=1e-4)

        mean_field_model.train()

        mean_field_values,states = list(),list()

        times = list()

        for t in range(0, self.horizon):

            for s in range(self.env.state_count):
                states.append([s])
                times.append([t])
                if t == 0:
                    mean_field_values.append(init_mf_flow[t,s])
                    policy_current = self.p_flow.val[t, :]
                    history_velocities[t][int((s) % self.env.state_count)].append(
                        self.env.action_option[
                            np.argmax(policy_current[int((s) % self.env.state_count)])
                        ])


                else :
                    x_onehot = torch.tensor([self.env.state_option[int(s)]]).to(self.device, torch.float)
                    x_last_onehot = torch.tensor(
                        [self.env.state_option[int(s - 1) % self.env.state_count]]).to(self.device,
                                                                                                      torch.float)
                    t_last_onehot = torch.tensor([(t - 1) % self.horizon]).to(self.device, torch.float)

                    # Set requires_grad to True for tensors where gradients are needed
                    x_onehot.requires_grad = True
                    t_last_onehot.requires_grad = True

                    # Compute mean fields for current, next time step, and last x position
                    mean_field_last_t = mean_field_model(x_onehot, t_last_onehot)
                    mean_field_last_x_t = mean_field_model(x_last_onehot, t_last_onehot)

                    policy_last = self.p_flow.val[t-1, :]
                    velocity_last_t_x = self.env.action_option[(np.argmax(policy_last[int(s - 1) % self.env.state_count]))]

                    action_index_last_t = np.argmax(policy_last[int((s) % self.env.state_count)])

                    history_velocities[t - 1][int((s) % self.env.state_count)].append(
                        self.env.action_option[action_index_last_t])

                    # Compute mean of historical velocities
                    if history_velocities[t-1][int((s) % self.env.state_count)]:
                        mean_velocity = np.mean(
                            history_velocities[t][int((s) % self.env.state_count)])
                        # Find the closest value in self.env.action_option to mean_velocity
                        velocity_last_x = min(self.env.action_option, key=lambda x: abs(x - mean_velocity))

                    mean_field_values.append(
                        (mean_field_last_t +
                        mean_field_last_x_t * velocity_last_t_x -
                        mean_field_last_t * velocity_last_x).detach()
                    )

        mean_field_values_filtered = []
        for v in mean_field_values:
            if isinstance(v, torch.Tensor):
                mean_field_values_filtered.append(v)  # 如果已经是 tensor，直接添加
            elif v is not None:  # 过滤掉 None 和空列表
                mean_field_values_filtered.append(torch.tensor([[v]]).to(self.device,torch.float))  # 将标量转换为形状为 [1] 的 tensor

        # 确保所有张量的大小一致后再进行堆叠
        mean_field_values = torch.stack(mean_field_values_filtered).detach()

        # 将 states 和 times 转换为张量，然后再拼接
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        times_tensor = torch.tensor(times, dtype=torch.float32).to(self.device)



        for _ in range(10*max_epoch):
            preds = torch.reshape(mean_field_model(states_tensor,times_tensor), (1, len(mean_field_values)))
            # Aggregate losses from the trajectory and perform a single optimization step per sample
            residual = (mean_field_values - preds).abs().mean()
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step


                # Log the loss
                # print('=Mean_Field: , loss:{}'.format(str(residual.detach().cpu().numpy())), end='\r')

        self.mean_field_model = mean_field_model







