"""
This is the physical informed AIRL


Remainder value type:
    1. estimated_mean_field_flow : numpy.array()

"""
import copy
import logging
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as U
import torch.nn.functional as F
from scipy.special import entr
import torch.nn as nn

from Algorithms.myModels import MeanFieldModel, RewardModel, PolicyModel
from core import Environment, State, Action, MeanField, MeanFieldFlow, PolicyFlow, Policy,IRL,Trajectory
from Environments import CARS
from Algorithms.expert_training import Expert

from sklearn.model_selection import train_test_split

import time

'''
Constrain all the variables 
We need to make sure it is tractable
'''

MAX = 100000  # maximum number of iterations
MIN = 1e-10


class PIIRL():
    '''
        Max_epoch : 训练迭代的最大次数
        learning_rate：优化器的学习率。
        ax_grad_norm：梯度裁剪的阈值，有助于通过避免梯度爆炸来稳定训练。
        num_of_units：指定奖励和塑造模型中神经网络层的大小。
    '''
    def __init__(self, data_expert, env: CARS, horizon: int, num_of_game_plays, num_traj,device):

        # for we need 2 types of the trajectory
        self.data_expert = data_expert
        self.data_policy_theta = None

        # here is the environment
        self.env = env
        self.device = device

        self.horizon = horizon
        self.mf_flow = MeanFieldFlow(s=self.env.state_count,t=self.horizon)

        # TODO we need the policy flow to get the loss function
        self.p_flow = PolicyFlow(s=self.env.state_count,t=self.horizon,a=self.env.action_count)
        self.expected_return = 0.0

        #TODO we need the parameter to train
        self.num_of_game_plays = num_of_game_plays
        self.num_traj = num_traj

        # TODO this is not the true meanfield estimated
        # TODO we only have the value for this position
        # TODO mean field need the value for all position under this time t
        # And here we init all the model
        self.reward_model = None

        self.mean_field_model = None

        self.policy_model = None

        self.train_data = None
        self.test_data = None

        self.expert = Expert(env=env, horizon=self.horizon)

        # Setup logging
        self.logger = logging.getLogger('MFIRL_Training')
        self.logger.setLevel(logging.INFO)  # Set log level to INFO

        # Create file handler which logs even debug messages
        fh = logging.FileHandler('PIAIRL_training_log.txt')
        fh.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handlers to logger
        self.logger.addHandler(fh)

    # Saves the trained reward model to a specified path
    def save_model(self, path: str):
        assert self.reward_model is not None
        torch.save(self.reward_model, path)

    # Loads a reward model from a specified path
    def load_model(self, path: str):
        self.reward_model = torch.load(path)
        self.reward_model.to(self.device)

    # Converts a categorical variable into a one-hot encoded vector
    # all to 0 with length = shape
    # and let entry index be 1
    def onehot_encoding(self, shape, entry):
        code = np.zeros(shape)
        code[entry] = 1.0
        return code

    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        reward_model = RewardModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_shape,
                                   mf_shape=self.env.state_count,
                                   num_of_units=num_of_units).to(self.device)

        policy_model = PolicyModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_count,
                                   mf_shape=self.env.state_count,
                                   num_of_units=num_of_units).to(self.device)

        # this is the nn for the mean field value
        mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
                                          # this is the special attribute for our model
                                          # here is the time shape should be the 1
                                          time_horizon=1,
                                          num_of_units=num_of_units).to(self.device)


        optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
        optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
        optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate,weight_decay=1e-4)
        '''
        Here is the init part 
            we need to initialize the policy  at first
            so that we can get the trajectory from it 
                we just randomly initialize the policy
        '''
        init_policy_flow = np.random.rand(self.horizon, self.env.state_count, self.env.action_count)

        # 将最后一个维度的值归一化为 1
        init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)

        self.p_flow.val = init_policy_flow


        # here we use the expert meanfield to generate the trajectory
        init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
        for sample in self.data_expert:
            for t in range(self.horizon):
                init_est_expert_mf_flow[t, int(sample.states[t])] += 1
        init_est_expert_mf_flow /= len(self.data_expert)

        self.mf_flow.val = init_est_expert_mf_flow



        for epoch in range(max_epoch):

            start_time = time.time()

            '''
            1. we get the meanfield_alpha from current policy
                we store our policy in the self as an attribute 
            
            2. we also need the new trajectory which is induced by current policy  
            '''


            # TODO This is the replace part for the
            # TODO and we store all the tensor value in it
            # after this we get the mean filed flow from current policy and trajectory
            estimated_mean_field_flow = np.zeros((self.horizon, self.env.state_count))

            # For we do have the trained value before we train our meanfield
            # So we use the estimated value from the expert for the first round

            if epoch != 0 : # this means that we are not in the first round
                for sample in self.data_policy_theta:
                    for t in range(self.horizon):
                        # here we only add the show states
                        # and all the value form the NN will be a tensor value
                        estimated_mean_field_flow[t, int(sample.states[t])] += self.mean_field_model(
                            torch.tensor(self.env.state_option[int(sample.states[t])]).to(self.device, torch.float),
                            torch.tensor(t).to(self.device, torch.float)
                        )
                for t in range(self.horizon):
                    sum_values = estimated_mean_field_flow[t].sum()
                    if sum_values > 0:
                        estimated_mean_field_flow[t] /= sum_values
            else:
                estimated_mean_field_flow = self.mf_flow.val


            # change the new meanfield to our current
            self.mf_flow.val = estimated_mean_field_flow

            # after get the init mf then we sample the trajectory
            self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays, self.num_traj,
                                                                                 self.p_flow, self.mf_flow)


            # these 2 list is to store the D value from expert and policy
            value_per_sample_expert_data = []
            value_per_sample_policy_data = []


            '''
            First part of our D
                which is induce from the expert trajectory
            '''
            # here are the expert value
            for sample_expert in self.data_expert:
                value_per_step = []
                value_reward_exp = []

                # At here we create the mean field flow for every time
                for t in range(self.horizon):
                    reward_component = reward_model(
                            torch.tensor(self.env.state_option[int(sample_expert.states[t])]).to(self.device, torch.float),
                            torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).to(self.device, torch.float),
                            torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )

                    up = torch.exp(reward_component)


                    # here we add the policy for this state and action
                    down = torch.exp(reward_component) + self.p_flow.val[t,int(sample_expert.states[t]),int(sample_expert.actions[t])]

                    # here is the D for this (s_t,a_t)
                    value_per_step.append(up/down)

                # here is the sum of the log D
                # print(torch.cat(value_per_step, dim=0).reshape((1, -1)))
                # print(torch.log(torch.cat(value_per_step, dim=0).reshape((1, -1))))
                # print(torch.sum(torch.log(torch.cat(value_per_step, dim=0).reshape((1, -1)))))
                value_per_sample_expert_data.append(torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))

            # print(value_per_sample_expert_data)
            # print(torch.cat(value_per_sample_expert_data))
            # we use mean to get the estimated value
            estimated_expert_data = torch.mean(torch.cat(value_per_sample_expert_data,dim=0).reshape((1, -1)))



            '''
            Here is the second part 
                which is induced from the policy trajectory
            '''
            for sample_policy_theta in self.data_policy_theta:
                value_per_step = []
                value_reward_exp = []
                for t in range(self.horizon):
                    # here we give addition [] for torch.cat in the nn
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_policy_theta.states[t])]).to(self.device, torch.float),
                        torch.tensor(self.env.action_option[int(sample_policy_theta.actions[t])]).to(self.device, torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )

                    up = torch.exp(reward_component)

                    # here we add the policy for this state and action
                    down = torch.exp(reward_component) + self.p_flow.val[t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])]

                    # here is the 1-D for this (s_t,a_t)
                    value_per_step.append(1 - (up / down))
                    value_reward_exp.append(reward_component)

                # log(1-D) sum for policy_theta on this trajectory
                value_per_sample_policy_data.append(torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))

            # same as above
            estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data,dim=0).reshape((1, -1)))

            # then we need to use the gradient to maximum the sum of these 2
            # So we use it to minmize the - of the sum


            optimizer_reward.zero_grad()
            loss = -(estimated_expert_data + estimated_policy_data)
            # loss = - (torch.mean(torch.cat([estimated_policy_data_reward,estimated_expert_reward],dim=0)))
            loss.backward()
            U.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer_reward.step()



            # TODO where should i do the policy update ?
            #  But how the trajectory change while we update the policy
            # here we update the policy
            history_velocities = [[[] for _ in range(self.env.state_count)] for _ in
                                  range(self.horizon)]



            for t in range(self.horizon-1, -1, -1):
                # here we calculate the sum for the current
                sum_current = []
                best_loss = float('inf')
                epoch_no_improve = 0

                self.train_data, self.test_data = train_test_split(
                    self.data_policy_theta, test_size=0.2, random_state=42
                )

                # here we use the formula
                for sample in self.train_data:
                    value_per_sampler = []
                    for current in range(t, self.horizon):
                        # at here we only update the previous
                        # So we can just use the "estimated_mean_field_flow" we got before
                        value_per_sampler.append(
                            reward_model(
                                torch.tensor(self.env.state_option[int(sample.states[current])]).to(self.device, torch.float),
                                torch.tensor(self.env.action_option[int(sample.actions[current])]).to(self.device, torch.float),
                                torch.from_numpy(estimated_mean_field_flow[current, :]).to(self.device, torch.float)
                            ) - torch.log(torch.tensor(self.p_flow.val[current, int(sample.states[current]), int(sample.actions[current])]))
                        )

                    sum_current.append(torch.sum(torch.cat(value_per_sampler,dim=0)).reshape((1, -1)))

                # this is the loss function
                estimated_update = torch.mean(torch.sum(torch.cat(sum_current,dim=0).reshape((1, -1))))

# FIXME 假设唯一一个分布

                # train the policy model
                optimizer_policy.zero_grad()
                loss2 =-(estimated_update)
                loss2.backward()
                U.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer_policy.step()

                # then we need to update the policy
                for s in range(self.env.state_count):
                    new_policy = policy_model(
                        torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )

                    action_exps = torch.exp(new_policy / self.env.beta)
                    # Normalize and move to CPU before assigning to NumPy array
                    self.p_flow.val[t, s] = (action_exps / torch.sum(action_exps)).detach().cpu().numpy()

                # self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,
                #                                                                      self.num_traj, self.p_flow,
                #                                                                      self.mf_flow)

                # FIXME We need to fill in the parameters
                # we also need to update the meanfield
                # and we need to judge if it is more than 2 dim
                if self.env.dim == 1:
                    self.train_mean_field_dim_1_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
                                                    history_velocities,mean_field_model,optimizer_meanfield,t,epoch_no_improve,best_loss)
                elif self.env.dim == 2:
                    self.train_mean_field_dim_2_new(max_epoch, learning_rate, max_grad_norm, num_of_units,
                                                    history_velocities,mean_field_model,optimizer_meanfield,t)

                # # Evaluate on the test dataset
                # test_loss = self.evaluate_mean_field_model(mean_field_model, history_velocities, t)
                # print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')

                # here is the code which we used to update the mean field value

                for s in range(self.env.state_count):
                    new_mean_field = self.mean_field_model(
                        torch.tensor(self.env.state_option[s]).to(self.device, torch.float),
                        torch.tensor(t).to(self.device, torch.float)
                    )

                    estimated_mean_field_flow[t, s] = new_mean_field

                total = sum(estimated_mean_field_flow[t, :])
                if total != 0:
                    estimated_mean_field_flow[t, :] /= total
                else:
                    raise ValueError("Sum of values is zero, cannot normalize")

                self.mf_flow.val = estimated_mean_field_flow

                # we will use the new tra
                self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays,self.num_traj,self.p_flow,self.mf_flow)




            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Time taken for one epoch: {epoch_duration:.4f} seconds")
            print('=PIIRL: epoch:{}'.format(epoch) + ', loss:{}'.format(str(loss.detach().cpu().numpy())))
            # print(epoch)
            # self.logger.info('=PIIRL: epoch:{}, loss:{}'.format(epoch, str(loss.detach().cpu().numpy())))

        # send the most optimal back
        # this is the last
        self.reward_model = reward_model
        self.policy_model = policy_model


    '''
    This is the physics-informed which is fitted for the 1-d
        For we just consider the before position and time 
    
    We should consider more if we consider 2-d
        Like last position could be around (like 9 points around)
        but the time should still be last 
    '''

    def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                                   num_of_units: int, history_velocities,mean_field_model, optimizer1, start_time,epoch_no_improve,best_loss):
        mean_field_model.train()
        patience = (len(self.data_policy_theta) / 2) if (len(self.data_policy_theta) / 2) < 100 else 100
        best_model = copy.deepcopy(mean_field_model.state_dict())
        for sample in self.train_data:
            # Initialize a list to store historical velocities

            loss_total = []

            for t in range(start_time, self.horizon):
                x_onehot = torch.tensor([self.env.state_option[int(sample.states[t])]]).to(self.device, torch.float)
                t_onehot = torch.tensor([t]).to(self.device, torch.float)
                x_last_onehot = torch.tensor(
                    [self.env.state_option[int(sample.states[t] - 1) % self.env.state_count]]).to(self.device,
                                                                                                  torch.float)
                t_next_onehot = torch.tensor([(t + 1) % self.horizon]).to(self.device, torch.float)

                # Set requires_grad to True for tensors where gradients are needed
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute mean fields for current, next time step, and last x position
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)
                mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)

                # Get the policy at time t for the current state
                current_policy = self.p_flow.val[t, :]


                # Compute expected velocity at time t
                velocity_now = self.env.action_option[(np.argmax(current_policy[int(sample.states[t])]))]
                action_index_last = np.argmax(current_policy[int((sample.states[t] - 1) % self.env.state_count)])

                history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)].append(self.env.action_option[action_index_last])

                # Compute mean of historical velocities
                if history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)]:
                    mean_velocity = np.mean(history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)])
                    # Find the closest value in self.env.action_option to mean_velocity
                    velocity_last_x = min(self.env.action_option, key=lambda x: abs(x - mean_velocity))

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
            # print('=Mean_Field: , loss:{}'.format(str(residual.detach().cpu().numpy())), end='\r')

            val_loss = self.evaluate_mean_field_model(mean_field_model, history_velocities, start_time)
            print(f'Epoch , Training Loss: {residual.item():.6f}, Validation Loss: {val_loss:.6f}', end='\r')

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                best_model = copy.deepcopy(mean_field_model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print('Early stopping triggered!',end='\r')
                break

            # self.logger.info(f'=Mean_Field: , Sample Loss: {residual.item():.4f}')

            #print()  # for better formatting of print output

        mean_field_model.load_state_dict(best_model)
        self.mean_field_model = mean_field_model

    def evaluate_mean_field_model(self, mean_field_model, history_velocities, start_time):
        mean_field_model.eval()  # Set model to evaluation mode
        loss_total_test = []

        with torch.no_grad():
            for sample in self.test_data:
                for t in range(start_time, self.horizon):
                    x_onehot = torch.tensor([self.env.state_option[int(sample.states[t])]]).to(self.device, torch.float)
                    t_onehot = torch.tensor([t]).to(self.device, torch.float)
                    x_last_onehot = torch.tensor(
                        [self.env.state_option[int(sample.states[t] - 1) % self.env.state_count]]).to(self.device,
                                                                                                      torch.float)
                    t_next_onehot = torch.tensor([(t + 1) % self.horizon]).to(self.device, torch.float)

                    # Set requires_grad to True for tensors where gradients are needed
                    x_onehot.requires_grad = True
                    t_onehot.requires_grad = True

                    # Compute mean fields for current, next time step, and last x position
                    mean_field_now = mean_field_model(x_onehot, t_onehot)
                    mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)
                    mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)

                    # Get the policy at time t for the current state
                    current_policy = self.p_flow.val[t, :]

                    # Compute expected velocity at time t
                    velocity_now = self.env.action_option[(np.argmax(current_policy[int(sample.states[t])]))]
                    action_index_last = np.argmax(current_policy[int((sample.states[t] - 1) % self.env.state_count)])

                    history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)].append(
                        self.env.action_option[action_index_last])

                    # Compute mean of historical velocities
                    if history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)]:
                        mean_velocity = np.mean(
                            history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)])
                        # Find the closest value in self.env.action_option to mean_velocity
                        velocity_last_x = min(self.env.action_option, key=lambda x: abs(x - mean_velocity))

                    # Compute the custom loss using the given formula
                    left = (mean_field_next_time - mean_field_now) / self.env.time_unit
                    right = ((mean_field_now * velocity_now) - (
                            mean_field_last_x * velocity_last_x)) / self.env.position_unit
                    loss = (left + right).abs()
                    loss_total_test.append(loss)

            # Aggregate losses from the test dataset
            residual_test = torch.mean(torch.cat((loss_total_test), dim=0).reshape((1, -1)))

        mean_field_model.train()
        avg_val_loss = residual_test

        # Return the test loss value
        return avg_val_loss



    def train_mean_field_dim_2_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,history_velocities,mean_field_model,optimizer1, start_time):


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
                                self.env.state_option[int(neighbour)]).to(self.device,torch.float))
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
                velocity_now =  self.encode_length_and_angle(self.env.action_option[action_index_now])

                velocity_last_x = []
                for action_index in action_index_last:
                    velocity_last_x.append( self.encode_length_and_angle(self.env.action_option[action_index]))
                velocity_last_x =  sum(velocity_last_x) / len(velocity_last_x)

                history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)].append(velocity_last_x)

                # Compute mean of historical velocities
                if history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)]:
                    mean_velocity = np.mean(history_velocities[t][int((sample.states[t] - 1) % self.env.state_count)])
                    # Find the closest value in self.env.action_option to mean_velocity
                    velocity_last_x =  self.encode_length_and_angle(min(
                        self.env.action_option,
                        key=lambda x: abs( self.encode_length_and_angle(x) - mean_velocity)
                    ))
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
            residual = torch.mean(torch.cat((loss_total),dim=0).reshape((1, -1)))
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step

            # Log the loss
            print('=Mean_Field: epoch:{}' + ', loss:{}'.format(str(residual.detach().cpu().numpy())),
                  end='\r')
            # self.logger.info(f'=Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')

            print()  # for better formatting of print output


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

    def train_mean_field_dim_2(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,mean_field_model):

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
                                self.env.state_option[int(neighbour)]).to(self.device,torch.float))
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
                velocity_last_x =  sum(velocity_last_x) / len(velocity_last_x)

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
            residual = torch.mean(torch.cat((loss_total),dim=0).reshape((1, -1)))
            optimizer1.zero_grad()  # Zero gradients before backward pass
            residual.backward()  # Backpropagation
            U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer1.step()  # Optimizer step

            # Log the loss
            print('=Mean_Field: epoch:{}' + ', loss:{}'.format(str(residual.detach().cpu().numpy())),
                  end='\r')
            # self.logger.info(f'=Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')

            print()  # for better formatting of print output


        self.mean_field_model = mean_field_model

    '''
    This is the generate method which can give the 
        trajectory under current mean field and policy 
        we need to use this method whenever we update the policy
    '''


    def generate_trajectories_from_policy_flow(self, num_game_play: int, num_traj: int, current_policy_flow,
                                               current_mean_field_flow, deterministic=True):
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
                #print(self.mf_flow.val[t, :])
                '''
                According to the mean_field, we select the state randomly from states
                '''
                # FIXME : Here we let the output from the numpy to the int value
                # FIXME: original : s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])
                s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])[0]
                #print(self.p_flow.val[t, s, :][0])

                # FIXME We change the code here from p=self.p_flow.val[t, s, :][0]
                # FIXME to p=self.p_flow.val[t, s, :])[0] PUT THE [0] out side
                self.p_flow.val[t, s, :] = self.p_flow.val[t, s, :] / (sum(self.p_flow.val[t, s, :]))
                a = np.random.choice(actions, 1, p=self.p_flow.val[t, s, :])[0]
                data[i].states[t] = s
                data[i].actions[t] = a
        return data


    def recover_ermfne(self) -> [MeanFieldFlow, PolicyFlow]:
        assert self.reward_model is not None
        # Init the mf_flow and p_flow
        '''状态特征向量长度'''
        mf_flow = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
        p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)

        # Here is the training process
        for _ in range(MAX):
            p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)

            # Here q-value is the p_flow value
            '''
            From range value, we could know that
            self.env.action_shape is int
            self.env.state_shape is int
            '''
            q_values = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)

            '''
            Initialization of Policy at Final Time Step (self.horizon-1)
                Policy is initialized to a uniform distribution across all actions for each state.
                This reflects an assumption of equal likelihood of actions in the absence of future information.
            '''
            for s in range(self.env.state_count):
                # let last time's policy_flow's action's probability be init = 1/(number_of_action)
                '''Here is the current time for we start at 0'''
                p_flow.val[self.horizon - 1, s, :] = (
                    np.array([1 / self.env.action_count for _ in range(self.env.action_count)]))

            # compute Q values and policy flow
            '''
            Calculation of Q-values and Policy Flow

            '''
            # for this time t
            # We only calculate until the last 2
            # self.horizon - 2
            for t in reversed(range(0, self.horizon - 1)):
                # every state s
                for s_current in range(0, self.env.state_count):
                    # for every possible action
                    for a_current in range(0, self.env.action_count):

                        q_values.val[t, s_current, a_current] += self.reward_model(
                            torch.tensor(self.env.state_option[s_current]).to(self.device,torch.float),
                            torch.tensor(self.env.action_option[a_current]).to(self.device, torch.float),
                            torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                            ).detach().cpu().numpy()

                        for s_next in range(0, self.env.state_count):
                            # Consider the current state and action
                            # we let our q-value at this time to the sum of all next
                            # Same as we first consider the s_next's future reward
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(
                                                                                             mean_field=mf_flow.val[
                                                                                                 t]))[s_next] \
                                                                     * self.env.beta * np.sum(
                                entr(p_flow.val[t + 1, s_next, :]))
                            # Above will be the sum of t+1's all policy's value
                            # and this "entr" is the entropy term

                            # then we consider all the action of the s_next
                            for a_next in range(0, self.env.action_count):
                                q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                             Action(action=a_current),
                                                                                             MeanField(
                                                                                                 mean_field=mf_flow.val[
                                                                                                     t]))[s_next] \
                                                                         * p_flow.val[t + 1, s_next, a_next] \
                                                                         * q_values.val[t + 1, s_next, a_next]
                                # this p_flow.val is the possible that we taking action

                # compute policy induced by the mean filed
                '''
                We have already consider the current S and T 's all actions
                then we need to update the policy under current T

                policy is updated using a softmax function = q_values
                '''
                for s in range(0, self.env.state_count):
                    partition = 0.0
                    # let our value from 0-1
                    for a in range(0, self.env.action_count):
                        # have all the exp value beh
                        partition += np.exp(q_values.val[t, s, a] / self.env.beta)
                    for a in range(0, self.env.action_count):
                        p_flow.val[t, s, a] = np.exp(q_values.val[t, s, a] / self.env.beta) / partition

            # compute mean field flow induced by the policy flow
            mf_flow_next = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
            mf_flow_next.val[0] = mf_flow.val[0, :]
            for t in range(1, self.horizon):
                # this will give the next Mean Field
                mf = self.env.advance(Policy(policy=p_flow.val[t - 1]), MeanField(mean_field=mf_flow.val[t - 1]))
                mf_flow_next.val[t] = mf.val

            # check the distance between new and old mean field flows
            '''
            This is Convergence Check
            '''
            distance = torch.nn.MSELoss(reduction='sum', size_average=True)
            if distance(torch.from_numpy(mf_flow_next.val), torch.from_numpy(mf_flow.val)) < MIN:
                break
            else:
                mf_flow = mf_flow_next

        # this is return value
        self.mf_flow = mf_flow
        self.p_flow = p_flow
        return [mf_flow, p_flow]




    def recover_expected_return(self):
        assert self.mf_flow is not None and self.p_flow is not None
        q_values = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
        # we update it from reverse
        for t in reversed(range(0, self.horizon - 1)):
            for s_current in range(0, self.env.state_count):
                for a_current in range(0, self.env.action_count):
                    # unlike above, we directly get the reward
                    q_values.val[t, s_current, a_current] += self.env.get_reward(State(state=s_current),
                                                                                 Action(action=a_current),
                                                                                 MeanField(mean_field=self.mf_flow.val[t])).val[0]
                    # next step
                    for s_next in range(self.env.state_count):
                        q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                     Action(action=a_current),
                                                                                     MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                 * self.env.beta * np.sum(entr(self.p_flow.val[t+1, s_next, :]))

                        for a_next in range(0, self.env.action_count):
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                     * self.p_flow.val[t+1, s_next, a_next] \
                                                                     * q_values.val[t+1, s_next, a_next]

        # compute expected return under equilibrium and terminate iteration

        # For we have update the q_values from back
        # So we can get the expected return from the initial state
        for s in range(0, self.env.state_count):
            partition = 0.0
            for a in range(0, self.env.action_count):
                partition += np.exp(q_values.val[0, s, a] / self.env.beta)
            for a in range(0, self.env.action_count):
                self.expected_return += self.mf_flow.val[0, s] * np.exp(q_values.val[0, s, a] / self.env.beta) * q_values.val[0, s, a] / partition


    '''
    This we could see it as D
    '''
    def divergence(self, expert_mf_flow, expert_p_flow):
        self.recover_ermfne()
        self.recover_expected_return()

        # Here we calculate the Divergence
        kl_div = nn.KLDivLoss(reduction='sum')
        dev_mean_field = kl_div((torch.from_numpy(self.mf_flow.val) + 1e-9).log(),
                                torch.from_numpy(expert_mf_flow.val)
                                ).detach().cpu().numpy()
        kl_div = nn.KLDivLoss(reduction='none')
        kl_policy = kl_div( torch.from_numpy(self.p_flow.val + 1e-9).log(), torch.from_numpy(expert_p_flow.val) ).reshape(-1, self.env.action_count).sum(1)
        dev_policy = torch.mul(kl_policy, torch.from_numpy(expert_mf_flow.val.reshape(1, -1))).sum().detach().cpu().numpy()
        return [self.expected_return, dev_mean_field, dev_policy]









