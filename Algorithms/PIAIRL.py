"""
This is the physical informed AIRL


Remainder value type:
    1. estimated_mean_field_flow : numpy.array()

"""


import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as U
from scipy.special import entr
import torch.nn as nn

from Algorithms.myModels import MeanFieldModel, RewardModel, PolicyModel
from core import Environment, State, Action, MeanField, MeanFieldFlow, PolicyFlow, Policy,IRL,Trajectory
from Environments import CARS
from Algorithms.expert_training import Expert

'''
Constrain all the variables 
We need to make sure it is tractable
'''

MAX = 100000  # maximum number of iterations
MIN = 1e-10


class PIAIRL():
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
        self.mf_flow = MeanFieldFlow(s=self.env.state_shape,t=self.horizon)

        # TODO we need the policy flow to get the loss function
        self.p_flow = PolicyFlow(s=self.env.state_shape,t=self.horizon,a=self.env.action_shape)
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

        self.expert = Expert(env=env, horizon=self.horizon)

    # Saves the trained reward model to a specified path
    def save_model(self, path: str):
        assert self.reward_model is not None
        torch.save(self.reward_model, path)

    # Loads a reward model from a specified path
    def load_model(self, path: str):
        self.reward_model = torch.load(path)

    # Converts a categorical variable into a one-hot encoded vector
    # all to 0 with length = shape
    # and let entry index be 1
    def onehot_encoding(self, shape, entry):
        code = np.zeros(shape)
        code[entry] = 1.0
        return code

    def train_reward_model(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        reward_model = RewardModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_shape,
                                   mf_shape=self.env.state_shape,
                                   num_of_units=num_of_units).to(self.device)

        policy_model = PolicyModel(state_shape=self.env.state_shape,
                                   action_shape=self.env.action_shape,
                                   mf_shape=self.env.state_shape,
                                   num_of_units=num_of_units).to(self.device)

        optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
        optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)

        '''
        Here is the init part 
            we need to initialize the policy  at first
            so that we can get the trajectory from it 
                we just randomly initialize the policy
        '''
        init_policy_flow = np.random.rand(self.horizon, self.env.state_shape, self.env.action_shape)

        # 将最后一个维度的值归一化为 1
        init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)

        self.p_flow.val = init_policy_flow


        # here we use the expert meanfield to generate the trajectory
        init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_shape))
        for sample in self.data_expert:
            for t in range(self.horizon):
                init_est_expert_mf_flow[t, int(sample.states[t])] += 1
        init_est_expert_mf_flow /= len(self.data_expert)

        self.mf_flow.val = init_est_expert_mf_flow

        for epoch in range(max_epoch):

            '''
            1. we get the meanfield_alpha from current policy
                we store our policy in the self as an attribute 
            
            2. we also need the new trajectory which is induced by current policy  
            '''

            # 1. we need sample the trajectory from our original policy
            self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays, self.num_traj, self.p_flow, self.mf_flow,True)

            # TODO This is the replace part for the
            # TODO and we store all the tensor value in it
            # after this we get the mean filed flow from current policy and trajectory
            estimated_mean_field_flow = np.zeros((self.horizon, self.env.state_shape))

            # For we do have the trained value before we train our meanfield
            # So we use the estimated value from the expert for the first round

            if epoch != 0 : # this means that we are not in the first round
                for sample in self.data_policy_theta:
                    for t in range(self.horizon):
                        # here we only add the show states
                        # and all the value form the NN will be a tensor value
                        estimated_mean_field_flow[t, int(sample.states[t])] += self.mean_field_model(
                            torch.from_numpy(self.onehot_encoding(self.env.state_shape, int(sample.states[t]))).to(self.device, torch.float),
                            torch.from_numpy(self.onehot_encoding(self.horizon, int(t))).to(self.device, torch.float)
                        )
                estimated_mean_field_flow /= len(self.data_policy_theta)
            else:
                estimated_mean_field_flow = self.mf_flow.val


            # change the new meanfield to our current
            self.mf_flow.val = estimated_mean_field_flow


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

                # At here we create the mean field flow for every time
                for t in range(self.horizon):
                    reward_component = reward_model(
                            torch.from_numpy(self.onehot_encoding(self.env.state_shape, int(sample_expert.states[t]))).to(
                                self.device, torch.float),
                            torch.from_numpy(self.onehot_encoding(self.env.action_shape, int(sample_expert.actions[t]))).to(
                                self.device, torch.float),
                            torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float))

                    up = torch.exp(reward_component)

                    # print(type(sample_expert.states[t]))
                    # print(type(sample_expert.actions[t]))
                    # # print(self.p_flow.val)
                    # print(self.p_flow.val[t,sample_expert.states[t],sample_expert.actions[t]])

                    # here we add the policy for this state and action
                    down = torch.exp(reward_component +
                                     self.p_flow.val[t,int(sample_expert.states[t]),int(sample_expert.actions[t])])

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
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.from_numpy(self.onehot_encoding(self.env.state_shape, int(sample_policy_theta.states[t]))).to(
                            self.device, torch.float),
                        torch.from_numpy(self.onehot_encoding(self.env.action_shape, int(sample_policy_theta.actions[t]))).to(
                            self.device, torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )

                    up = torch.exp(reward_component)

                    # here we add the policy for this state and action
                    down = torch.exp(reward_component +
                                     self.p_flow.val[t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])])

                    # here is the 1-D for this (s_t,a_t)
                    value_per_step.append(1 - (up / down))

                # log(1-D) sum for policy_theta on this trajectory
                value_per_sample_policy_data.append(torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))

            # same as above
            estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data,dim=0).reshape((1, -1)))

            # then we need to use the gradient to maximum the sum of these 2
            # So we use it to minmize the - of the sum

            optimizer_reward.zero_grad()
            loss = - (estimated_expert_data + estimated_policy_data)
            loss.backward()
            U.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer_reward.step()

            # TODO where should i do the policy update ?
            #  But how the trajectory change while we update the policy
            # here we update the policy
            for t in range(self.horizon-1, -1, -1):
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
                                torch.from_numpy(
                                    self.onehot_encoding(self.env.state_shape, int(sample.states[current]))).to(
                                    self.device, torch.float),
                                torch.from_numpy(self.onehot_encoding(self.env.action_shape,
                                                                      int(sample.actions[current]))).to(
                                    self.device, torch.float),
                                torch.from_numpy(estimated_mean_field_flow[current, :]).to(self.device, torch.float)
                            ) - torch.log(torch.tensor(self.p_flow.val[current, int(sample.states[current]), int(sample.actions[current])]))
                        )
                    sum_current.append(torch.sum(torch.cat(value_per_sampler,dim=0)).reshape((1, -1)))

                # this is the loss function
                estimated_update = torch.mean(torch.sum(torch.cat(sum_current,dim=0).reshape((1, -1))))

# FIXME 假设唯一一个分布

                # train the policy model
                optimizer_policy.zero_grad()
                loss2 = - (estimated_update)
                loss2.backward()
                U.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer_policy.step()

                # then we need to update the policy
                for s in range(self.env.state_shape):
                    new_policy = policy_model(
                        torch.from_numpy(
                            self.onehot_encoding(self.env.state_shape, int(s))).to(
                            self.device, torch.float),
                        torch.from_numpy(estimated_mean_field_flow[t, :]).to(self.device, torch.float)
                    )

                    tensor_list = new_policy.tolist()
                    print(tensor_list)

                    # here we update the policy flow
                    self.p_flow.val[t,s] = tensor_list

                # FIXME We need to fill in the parameters

                '''
                    Here we need to update trajectory
                    And we need to update the trajectory first, so that 
                        our new mean_field model is the correct 
                '''
                self.data_policy_theta = self.generate_trajectories_from_policy_flow(self.num_of_game_plays, self.num_traj,self.p_flow,self.mf_flow)



                # we also need to update the meanfield
                self.train_mean_field(max_epoch, learning_rate, max_grad_norm, num_of_units)




            print('=MFIRL: epoch:{}'.format(epoch) + ', loss:{}'.format(str(loss.detach().cpu().numpy())), end='\r')

        # send the most optimal back
        # this is the last
        self.reward_model = reward_model
        self.policy_model = policy_model


    def train_mean_field(self,max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        # this is the nn for the mean field value
        mean_field_model = MeanFieldModel(state_shape=self.env.state_shape,
                                          # this is the special attribute for our model
                                          time_horizon=self.horizon,
                                          num_of_units=num_of_units).to(self.device)

        # this is the optimizer which is the actual runner
        optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate)

        # then we start our nn for the mean field
        for epoch in range(max_epoch):
            # this value is used to replace "estimate mean field flow" in our code
            est_mf_flow = None
            for sample in self.data_policy_theta:

                # get the mean field for this time and states
                # This will give us all the mean field for this sample through our model
                '''
                For we only need the x-position and t-time for our model 
                '''

                # here is used to store all the loss on this trajectory
                # for we only update the policy on the whole trajectory
                loss_total = []

                for t in range(self.horizon):
                    # for we need the mean_field for every t and every sample
                    # we use this iteration

                    # position x_onehot and time t_onehot
                    # we use this 2 variable to track the gradient so that we can do the loss
                    x_onehot = torch.from_numpy(self.onehot_encoding(self.env.state_shape, int(sample.states[t]))).to(self.device, torch.float)
                    t_onehot = torch.from_numpy(self.onehot_encoding(self.horizon, int(t))).to(self.device, torch.float)

                    # last position
                    # and here we need to make sure that this is legal
                    x_last_onehot = torch.from_numpy(
                        self.onehot_encoding(self.env.state_shape, int(sample.states[t] - self.env.position_unit) % self.env.state_shape)
                    ).to(self.device, torch.float)

                    t_next_onehot = torch.from_numpy(
                        self.onehot_encoding(self.horizon, int(t + self.env.time_unit) % self.horizon )
                    ).to(self.device, torch.float)

                    # so that we can track its gradient
                    x_onehot.requires_grad = True
                    t_onehot.requires_grad = True

                    mean_field_now = mean_field_model(x_onehot, t_onehot)
                    mean_field_next_time = mean_field_model(x_onehot, t_next_onehot)
                    mean_field_last_x = mean_field_model(x_last_onehot, t_onehot)

                    # then also need to record the policy
                    # this will give us the pi^theta_t
                    # TODO we get the probability of actions in this time
                    # TODO But for we are deterministic, so it will be all 0 but 1 for one action
                    # TODO we assume that it has been one_hot well
                    current_policy = self.p_flow.val[t,:]

                    # then we need to get the policy which is the velocity
                    # And we use the mean velocity to represent
                    # policy is s[a[]] like this
                    # and we have every probability for every action
                    '''
                        we need the max one 
                    '''

                    action_index_now = np.argmax(current_policy[int(sample.states[t])])
                    action_index_last = np.argmax(current_policy[int((sample.states[t] - self.env.position_unit)%self.env.state_shape)])
                    velocity_now = self.env.velocity_option[action_index_now]
                    velocity_last_x =self.env.velocity_option[action_index_last]


                    # time do not need one-hot

                    # # then we need to train this one
                    # # Compute gradients with autograd
                    # mu_t = torch.autograd.grad(mean_field_now, t_onehot,
                    #                            grad_outputs=torch.ones_like(mean_field_now),
                    #                            create_graph=True)[0]
                    #
                    # # mu * policy and its gradient
                    # product = mean_field_now * velocity
                    # product_x = torch.autograd.grad(product, x_onehot,
                    #                     grad_outputs=torch.ones_like(product),
                    #                     create_graph=True)[0]

                    '''
                    Here we change our residual calculation process:
                        we choose use the delt_t and delt_x to get the residual
                        For we can not add 2 value with different dimension 
                        
                    And we let the random step size fe = 1 
                    '''

                    # First part role_t
                    #
                    left = (mean_field_next_time - mean_field_now) / self.env.time_unit
                    right = ((mean_field_now * velocity_now)-(mean_field_last_x * velocity_last_x)) / self.env.position_unit


                    # Compute the custom loss as described
                    loss = left + right
                    loss_total.append(loss)

                # here is the optimal path
                residual = torch.mean(torch.cat(loss_total,dim=0).reshape((1, -1)))
                optimizer1.zero_grad()
                residual.backward()
                U.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
                optimizer1.step()

                print('=Mean_Field: epoch:{}'.format(epoch) + ', loss:{}'.format(str(residual.detach().cpu().numpy())),
                      end='\r')


        self.mean_field_model = mean_field_model


    '''
    This is the generate method which can give the 
        trajectory under current mean field and policy 
        we need to use this method whenever we update the policy
    '''
    def generate_trajectories_from_policy_flow(self, num_game_play: int, num_traj: int, current_policy_flow,
                                               current_mean_field_flow, deterministic=True):
        states = [i for i in range(self.env.state_shape)]  # State space
        actions = [i for i in range(self.env.action_shape)]  # Action space
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
                    s_next = self.env.dynamics(State(state=s), Action(action=a))  # Update this method as per your environment dynamics
                    data[i].states[t + 1] = s_next.val[0]
                    s = int(s_next.val[0])  # Update current state to the next state

        return data

    def recover_ermfne(self) -> [MeanFieldFlow, PolicyFlow]:
        assert self.reward_model is not None
        # Init the mf_flow and p_flow
        '''状态特征向量长度'''
        mf_flow = MeanFieldFlow(mean_field_flow=None, s=self.env.state_shape, t=self.horizon)
        p_flow = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

        # Here is the training process
        for _ in range(MAX):
            p_flow = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

            # Here q-value is the p_flow value
            '''
            From range value, we could know that 
            self.env.action_shape is int 
            self.env.state_shape is int
            '''
            q_values = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

            '''
            Initialization of Policy at Final Time Step (self.horizon-1)
                Policy is initialized to a uniform distribution across all actions for each state. 
                This reflects an assumption of equal likelihood of actions in the absence of future information.
            '''
            for s in range(self.env.state_shape):
                # let last time's policy_flow's action's probability be init = 1/(number_of_action)
                '''Here is the current time for we start at 0'''
                p_flow.val[self.horizon - 1, s, :] = (
                    np.array([1 / self.env.action_shape for _ in range(self.env.action_shape)]))

            # compute Q values and policy flow
            '''
            Calculation of Q-values and Policy Flow

            '''
            # for this time t
            # We only calculate until the last 2
            # self.horizon - 2
            for t in reversed(range(0, self.horizon - 1)):
                # every state s
                for s_current in range(0, self.env.state_shape):
                    # for every possible action
                    for a_current in range(0, self.env.action_shape):
                        '''
                        Onehot_encoding is used to fit input requirements of a typical neural network model

                        We could see it separately :
                            1. torch.from_numpy(self.onehot_encoding(self.env.state_shape, s_current)).to(self.device, torch.float)
                                1) One_hot will be list with length = self.env.state_shape and all be 0 but index of s_current is 1
                            2. torch.from_numpy(self.onehot_encoding(self.env.action_shape, a_current)).to(self.device, torch.float)
                                1) One_hot will be list with length = self.env.action_shape and all be 0 but index of a_current is 1
                            3. torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                                1) Input the current mf_flow to the NN

                        So that we can get the current reward which is the Q-value

                        In another word this is immediate Reward 
                        '''
                        q_values.val[t, s_current, a_current] += self.reward_model(
                            torch.from_numpy(self.onehot_encoding(self.env.state_shape, s_current)).to(self.device,
                                                                                                       torch.float),
                            torch.from_numpy(self.onehot_encoding(self.env.action_shape, a_current)).to(self.device,
                                                                                                        torch.float),
                            torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                            ).detach().cpu().numpy()
                        # next step
                        '''
                        First we need to consider the next reward

                        In a word, Future Reward
                        '''
                        for s_next in range(0, self.env.state_shape):
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
                            for a_next in range(0, self.env.action_shape):
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
                for s in range(0, self.env.state_shape):
                    partition = 0.0
                    # let our value from 0-1
                    for a in range(0, self.env.action_shape):
                        # have all the exp value beh
                        partition += np.exp(q_values.val[t, s, a] / self.env.beta)
                    for a in range(0, self.env.action_shape):
                        p_flow.val[t, s, a] = np.exp(q_values.val[t, s, a] / self.env.beta) / partition

            # compute mean field flow induced by the policy flow
            mf_flow_next = MeanFieldFlow(mean_field_flow=None, s=self.env.state_shape, t=self.horizon)
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
        q_values = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)
        # we update it from reverse
        for t in reversed(range(0, self.horizon - 1)):
            for s_current in range(0, self.env.state_shape):
                for a_current in range(0, self.env.action_shape):
                    # unlike above, we directly get the reward
                    q_values.val[t, s_current, a_current] += self.env.get_reward(State(state=s_current),
                                                                                 Action(action=a_current),
                                                                                 MeanField(mean_field=self.mf_flow.val[t])).val[0]
                    # next step
                    for s_next in range(self.env.state_shape):
                        q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                     Action(action=a_current),
                                                                                     MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                 * self.env.beta * np.sum(entr(self.p_flow.val[t+1, s_next, :]))

                        for a_next in range(0, self.env.action_shape):
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                     * self.p_flow.val[t+1, s_next, a_next] \
                                                                     * q_values.val[t+1, s_next, a_next]

        # compute expected return under equilibrium and terminate iteration

        # For we have update the q_values from back
        # So we can get the expected return from the initial state
        for s in range(0, self.env.state_shape):
            partition = 0.0
            for a in range(0, self.env.action_shape):
                partition += np.exp(q_values.val[0, s, a] / self.env.beta)
            for a in range(0, self.env.action_shape):
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
        kl_policy = kl_div( torch.from_numpy(self.p_flow.val + 1e-9).log(), torch.from_numpy(expert_p_flow.val) ).reshape(-1, self.env.action_shape).sum(1)
        dev_policy = torch.mul(kl_policy, torch.from_numpy(expert_mf_flow.val.reshape(1, -1))).sum().detach().cpu().numpy()
        return [self.expected_return, dev_mean_field, dev_policy]









