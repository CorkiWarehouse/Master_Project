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
        optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate)

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
            self.train_mean_field_dim_1(
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

    def train_mean_field_dim_2_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,
                                   history_velocities, mean_field_model, optimizer1, start_time):

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

            for t in range(start_time,self.horizon):

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
