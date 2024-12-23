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
from torch.optim.lr_scheduler import StepLR, ExponentialLR
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


class TEST(IRL):
    '''
        Max_epoch : 训练迭代的最大次数
        learning_rate：优化器的学习率。
        max_grad_norm：梯度裁剪的阈值，有助于通过避免梯度爆炸来稳定训练。
        num_of_units：指定奖励和塑造模型中神经网络层的大小。
    '''

    def update_target_mean_field_flow(self):
        # Compute target_mf_flow from the target_mean_field_model
        with torch.no_grad():
            for t in range(self.horizon):
                vals = []
                for s in range(self.env.state_count):
                    x_input = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
                    t_input = torch.tensor(t).float().to(self.device)
                    val = torch.exp(self.target_mean_field_model(x_input, t_input)).cpu().numpy()
                    vals.append(val)
                vals = np.array(vals).flatten()
                total = np.sum(vals)
                if total > 0:
                    vals /= total
                else:
                    vals = np.ones(self.env.state_count) / self.env.state_count
                self.target_mf_flow[t] = vals

    def train_mean_field_dim_1(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                               num_of_units: int, mean_field_model, optimizer1, alpha=0.1,
                               history_length=5, regularization_factor=0.01, target_mean_field_model=None):
        mean_field_model.train()
        combined_data = self.data_policy_theta # + self.data_expert
        loss_total = []

        for sample in combined_data:
            for t in range(self.horizon):
                # Convert current state to one-hot
                x_onehot = F.one_hot(
                    torch.tensor(int(sample.states[t])), num_classes=self.env.state_count
                ).float().to(self.device)
                t_onehot = torch.tensor([t]).float().to(self.device)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute current and next mean fields
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_onehot + 1)

                # Current timestep velocity
                current_policy = self.p_flow.val[t, int(sample.states[t])]
                velocity_now = np.dot(current_policy, self.env.action_option)

                # Historical averaging for velocity_last_x
                start_h = max(0, t - history_length)
                velocity_last_x_hist = []

                for h in range(start_h, t + 1):
                    neighbours_h = self.env.get_neighbors(int(sample.states[h]))
                    velocity_last_x_h = []
                    for neighbour in neighbours_h:
                        neighbour_policy_h = self.p_flow.val[h, int(neighbour) % self.env.state_count]
                        expected_velocity_h = np.dot(neighbour_policy_h, self.env.action_option)
                        velocity_last_x_h.append(expected_velocity_h)
                    velocity_last_x_hist.append(np.mean(velocity_last_x_h))

                velocity_last_x = np.mean(velocity_last_x_hist)

                # Compute finite differences
                delta_t = self.env.time_unit
                delta_x = self.env.position_unit

                neighbours = self.env.get_neighbors(int(sample.states[t]))
                mean_field_last_x = []
                for neighbour in neighbours:
                    x_last = F.one_hot(
                        torch.tensor(int(neighbour)), num_classes=self.env.state_count
                    ).float().to(self.device)
                    mean_field_last_x.append(mean_field_model(x_last, t_onehot))
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)

                left = (mean_field_next_time - mean_field_now) / delta_t
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x
                main_loss = ((left + right)**2)

                loss_total.append(main_loss)

        # Optimize the mean field model
        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        # Update the mean field flow with exponential smoothing
        # with torch.no_grad():
        #     for t in range(self.horizon):
        #         for s in range(self.env.state_count):
        #             x_input = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
        #             t_input = torch.tensor(t).float().to(self.device)
        #             new_value = np.exp(mean_field_model(x_input, t_input).detach().cpu().numpy())
        #             self.mf_flow.val[t, s] = alpha * new_value + (1 - alpha) * self.mf_flow.val[t, s]

        self.mean_field_model = mean_field_model

    def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                               num_of_units: int, mean_field_model, optimizer1, alpha=0.1,
                               history_length=5, regularization_factor=0.01, target_mean_field_model=None):
        mean_field_model.train()
        combined_data = self.data_policy_theta # + self.data_expert
        loss_total = []

        for sample in combined_data:
            for t in range(self.horizon):
                # Convert current state to one-hot
                x_onehot = F.one_hot(
                    torch.tensor(int(sample.states[t])), num_classes=self.env.state_count
                ).float().to(self.device)
                t_onehot = torch.tensor([t]).float().to(self.device)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute current and next mean fields
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_onehot + 1)

                # Current timestep velocity
                current_policy = self.p_flow.val[t, int(sample.states[t])]
                velocity_now = np.dot(current_policy, self.env.action_option)

                # Historical averaging for velocity_last_x
                start_h = max(0, t - history_length)
                velocity_last_x_hist = []

                for h in range(start_h, t + 1):
                    neighbours_h = self.env.get_neighbors(int(sample.states[h]))
                    velocity_last_x_h = []
                    for neighbour in neighbours_h:
                        neighbour_policy_h = self.p_flow.val[h, int(neighbour) % self.env.state_count]
                        expected_velocity_h = np.dot(neighbour_policy_h, self.env.action_option)
                        velocity_last_x_h.append(expected_velocity_h)
                    velocity_last_x_hist.append(np.mean(velocity_last_x_h))

                velocity_last_x = np.mean(velocity_last_x_hist)

                # Compute finite differences
                delta_t = self.env.time_unit
                delta_x = self.env.position_unit

                neighbours = self.env.get_neighbors(int(sample.states[t]))
                mean_field_last_x_list = []
                for neighbour in neighbours:
                    x_last = F.one_hot(
                        torch.tensor(int(neighbour)), num_classes=self.env.state_count
                    ).float().to(self.device)
                    mean_field_last_x_list.append(mean_field_model(x_last, t_onehot))
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x_list), dim=0)

                left = (mean_field_next_time - mean_field_now) / delta_t
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x

                # 使用绝对误差代替平方误差
                main_loss = (left + right).abs()

                # 如果有 target_mean_field_model，则加上正则项
                if target_mean_field_model is not None:
                    with torch.no_grad():
                        target_mean_field_now = target_mean_field_model(x_onehot, t_onehot)
                        target_mean_field_next_time = target_mean_field_model(x_onehot, t_onehot + 1)
                        target_mean_field_last_x_list = []
                        for neighbour in neighbours:
                            x_last = F.one_hot(
                                torch.tensor(int(neighbour)), num_classes=self.env.state_count
                            ).float().to(self.device)
                            target_mean_field_last_x_list.append(target_mean_field_model(x_last, t_onehot))
                        target_mean_field_last_x = torch.mean(torch.stack(target_mean_field_last_x_list), dim=0)

                    # 计算与目标模型的偏差正则项
                    reg_term = ((mean_field_now - target_mean_field_now) ** 2
                                + (mean_field_next_time - target_mean_field_next_time) ** 2
                                + (mean_field_last_x - target_mean_field_last_x) ** 2).mean()
                    main_loss = main_loss + regularization_factor * reg_term

                loss_total.append(main_loss)

        # Optimize the mean field model
        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            # torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        # Update the mean field flow with exponential smoothing
        # with torch.no_grad():
        #     for t in range(self.horizon):
        #         for s in range(self.env.state_count):
        #             x_input = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
        #             t_input = torch.tensor(t).float().to(self.device)
        #             new_value = np.exp(mean_field_model(x_input, t_input).detach().cpu().numpy())
        #             self.mf_flow.val[t, s] = alpha * new_value + (1 - alpha) * self.mf_flow.val[t, s]

        self.mean_field_model = mean_field_model

    def train_mean_field_dim_1_new2(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                                   num_of_units: int, mean_field_model, optimizer1, alpha=0.1,
                                   history_length=5, regularization_factor=0.01, target_mean_field_model=None,
                                   option="ring"):
        mean_field_model.train()
        combined_data = self.data_policy_theta  # 只使用 data_policy_theta
        loss_total = []

        n_cell = self.env.state_count
        # 假设 self.horizon = n_cell * T_terminal
        T_terminal = self.horizon // n_cell

        for sample in combined_data:
            for t in range(self.horizon - 1):
                s = int(sample.states[t])

                # Convert current state to one-hot
                x_onehot = F.one_hot(torch.tensor(s), num_classes=n_cell).float().to(self.device)
                t_onehot = torch.tensor([t]).float().to(self.device)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # 当前时刻mean field预测
                mean_field_now = mean_field_model(x_onehot, t_onehot)

                # 根据与 get_rho_network_from_u 类似的逻辑计算 rho_value
                if t == 0:
                    # 没有d，使用当前self.mf_flow.val[0, s]作为初始参考
                    rho_value = self.mf_flow.val[0, s]
                else:
                    # 利用上一时刻的mean_field_model预测值计算
                    with torch.no_grad():
                        def mf(i, tt):
                            x_ = F.one_hot(torch.tensor(i), num_classes=n_cell).float().to(self.device)
                            t_ = torch.tensor([tt]).float().to(self.device)
                            return mean_field_model(x_, t_)

                        if s == 0 and option == "ring":
                            # 参考 get_rho_network_from_u 中的公式
                            # rho(i,t) = rho(i,t-1) + rho(n_cell-1,t-1)*u[-1,t-1] - rho(i,t-1)*u[i,t-1]
                            rho_0_tm1 = mf(s, t - 1)
                            rho_nm1_tm1 = mf(n_cell - 1, t - 1)

                            current_policy = self.p_flow.val[t - 1, s]
                            velocity_now_tm1 = np.dot(current_policy, self.env.action_option)

                            im1_policy = self.p_flow.val[t - 1, n_cell - 1]
                            velocity_nm1_tm1 = np.dot(im1_policy, self.env.action_option)

                            rho_value = (rho_0_tm1
                                         + rho_nm1_tm1 * velocity_nm1_tm1
                                         - rho_0_tm1 * velocity_now_tm1)
                            rho_value = float(rho_value.cpu().numpy())
                        else:
                            # 通用情况：
                            # rho(i,t) = rho(i,t-1) + rho(i-1,t-1)*u[i-1,t-1] - rho(i,t-1)*u[i,t-1]
                            rho_i_tm1 = mf(s, t - 1)
                            rho_im1_tm1 = mf((s - 1) % n_cell, t - 1)

                            current_policy = self.p_flow.val[t - 1, s]
                            velocity_now_tm1 = np.dot(current_policy, self.env.action_option)

                            im1_policy = self.p_flow.val[t - 1, (s - 1) % n_cell]
                            velocity_im1_tm1 = np.dot(im1_policy, self.env.action_option)

                            rho_value = (rho_i_tm1
                                         + rho_im1_tm1 * velocity_im1_tm1
                                         - rho_i_tm1 * velocity_now_tm1)
                            rho_value = float(rho_value.cpu().numpy())

                # 使用 (mean_field_now - rho_value) 来计算误差
                main_loss = (mean_field_now - rho_value) ** 2

                # 如果有 target_mean_field_model，则加入正则项
                if target_mean_field_model is not None:
                    with torch.no_grad():
                        target_mean_field_now = target_mean_field_model(x_onehot, t_onehot)
                    reg_term = (mean_field_now - target_mean_field_now) ** 2
                    main_loss = main_loss + regularization_factor * reg_term

                loss_total.append(main_loss)

        # 优化 mean field model
        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            # 如果需要梯度裁剪请取消注释
            # torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        # 使用指数平滑更新 mean field flow
        with torch.no_grad():
            for t in range(self.horizon):
                for s in range(self.env.state_count):
                    x_input = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
                    t_input = torch.tensor(t).float().to(self.device)
                    new_value = np.exp(mean_field_model(x_input, t_input).detach().cpu().numpy())
                    self.mf_flow.val[t, s] = alpha * new_value + (1 - alpha) * self.mf_flow.val[t, s]

        self.mean_field_model = mean_field_model

    def train_mean_field_dim_1_new3(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                               num_of_units: int, mean_field_model, optimizer1, alpha=0.1,
                               # 移除 history_length 参数，因为不再使用历史平均
                               regularization_factor=0.01, target_mean_field_model=None):

        mean_field_model.train()
        # 不使用历史平均，只使用 data_policy_theta，假设仍沿用原有数据来源
        combined_data = self.data_policy_theta + self.data_expert
        loss_total = []

        # 假设一维空间中 state 对应离散格点0,1,...,s-1
        # 为了实现 (\rho u)_x，我们采用后向差分或前向差分。例如后向差分：
        # (rho u)_x ≈ (rho(x,t)*u(x,t) - rho(x-1,t)*u(x-1,t)) / Δx
        # 对x=0的格点，可采用环状条件：x-1 = s-1 (或根据需求选择边界条件)

        n_state = self.env.state_count
        delta_t = self.env.time_unit
        delta_x = self.env.position_unit
        history_length = 5

        for sample in combined_data:
            for t in range(self.horizon):
                # 当前状态s
                s = int(sample.states[t])
                x_onehot = F.one_hot(torch.tensor(s), num_classes=n_state).float().to(self.device)
                t_onehot = torch.tensor([t]).float().to(self.device)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute mean fields
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_onehot + 1)

                # 当前时刻的 velocity
                current_policy = self.p_flow.val[t, s]
                velocity_now = np.dot(current_policy, self.env.action_option)

                # 获取 x-1 对应的前一格点的 mean field 与 velocity
                # 假设使用环状边界条件（可根据实际需求修改）
                s_prev = (s - 1) % n_state
                x_prev_onehot = F.one_hot(torch.tensor(s_prev), num_classes=n_state).float().to(self.device)
                mean_field_last_x = mean_field_model(x_prev_onehot, t_onehot)

                # 前一格点的velocity
                start_h = max(0, t - history_length)
                velocity_last_x_hist = []

                for h in range(start_h, t + 1):
                    neighbours_h = self.env.get_neighbors(int(sample.states[h]))
                    velocity_last_x_h = []
                    for neighbour in neighbours_h:
                        neighbour_policy_h = self.p_flow.val[h, int(neighbour) % self.env.state_count]
                        expected_velocity_h = np.dot(neighbour_policy_h, self.env.action_option)
                        velocity_last_x_h.append(expected_velocity_h)
                    velocity_last_x_hist.append(np.mean(velocity_last_x_h))

                velocity_last_x = np.mean(velocity_last_x_hist)
                # prev_policy = self.p_flow.val[t, s_prev]
                # velocity_last_x = np.dot(prev_policy, self.env.action_option)

                # PDE残差计算
                # left = (rho(x,t+Δt)-rho(x,t))/Δt
                left = (mean_field_next_time - mean_field_now) / delta_t

                # right = ((rho(x,t)*u(x,t)-rho(x-1,t)*u(x-1,t))/Δx)
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x

                main_loss = ((left + right) ** 2)
                loss_total.append(main_loss)

        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        self.mean_field_model = mean_field_model

    # def encode_length_and_angle(self, velocity):
    #     # 计算模
    #     r = np.linalg.norm(velocity)
    #     # 计算角度
    #     theta = np.arctan2(velocity[1], velocity[0])
    #     # 将角度映射到 [-1, 1] 并计算编码值
    #     encoded_value = r * np.cos(theta)
    #     return encoded_value

    def encode_length_and_angle(self, velocity):
        # 计算模
        r = np.linalg.norm(velocity)
        # 计算角度
        theta = np.arctan2(velocity[1], velocity[0]) if (velocity[0] != 0 or velocity[1] != 0) else 0.0
        # 将角度映射到 [-1, 1] 并计算编码值
        encoded_value = r * np.cos(theta)
        return encoded_value

    def train_mean_field_dim_2(self, max_epoch: int, learning_rate: float, max_grad_norm: float,
                               num_of_units: int, mean_field_model, optimizer1, alpha=0.1,
                               history_length=5, regularization_factor=0.01, target_mean_field_model=None):
        mean_field_model.train()
        combined_data = self.data_policy_theta  # + self.data_expert
        loss_total = []

        for sample in combined_data:
            for t in range(self.horizon):
                # 将状态索引转换为 one-hot 编码
                x_onehot = F.one_hot(
                    torch.tensor(int(sample.states[t])), num_classes=self.env.state_count
                ).float().to(self.device)
                t_onehot = torch.tensor([t]).float().to(self.device)
                x_onehot.requires_grad = True
                t_onehot.requires_grad = True

                # Compute current and next mean fields
                mean_field_now = mean_field_model(x_onehot, t_onehot)
                mean_field_next_time = mean_field_model(x_onehot, t_onehot + 1)

                # 获取当前时刻的策略与动作选择
                current_policy = self.p_flow.val[t, int(sample.states[t])]
                action_index_now = np.argmax(current_policy)
                # 将action转换为对应的二维向量后编码
                velocity_now = self.encode_length_and_angle(self.env.action_option[action_index_now])

                # 历史平均 velocity_last_x 的计算（维度2）
                start_h = max(0, t - history_length)
                velocity_last_x_hist = []
                for h in range(start_h, t + 1):
                    neighbours_h = self.env.get_neighbors(int(sample.states[h]))
                    velocity_last_x_h = []
                    for neighbour in neighbours_h:
                        neighbour_policy_h = self.p_flow.val[h, int(neighbour) % self.env.state_count]
                        # 对邻居对应时刻最优动作进行速度编码
                        neighbour_action_index = np.argmax(neighbour_policy_h)
                        expected_velocity_h = self.encode_length_and_angle(
                            self.env.action_option[neighbour_action_index])
                        velocity_last_x_h.append(expected_velocity_h)
                    # 对该时刻的邻居速度取平均
                    velocity_last_x_hist.append(np.mean(velocity_last_x_h))

                velocity_last_x = np.mean(velocity_last_x_hist)

                # 计算 finite differences
                delta_t = self.env.time_unit
                delta_x = self.env.position_unit

                # 计算 mean_field_last_x
                neighbours = self.env.get_neighbors(int(sample.states[t]))
                mean_field_last_x = []
                for neighbour in neighbours:
                    x_last = F.one_hot(
                        torch.tensor(int(neighbour)), num_classes=self.env.state_count
                    ).float().to(self.device)
                    mean_field_last_x.append(mean_field_model(x_last, t_onehot))
                mean_field_last_x = torch.mean(torch.stack(mean_field_last_x), dim=0)

                left = (mean_field_next_time - mean_field_now) / delta_t
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x
                main_loss = (left + right).abs()

                loss_total.append(main_loss)

        # 优化 mean field model
        if loss_total:
            residual = torch.stack(loss_total).mean()
            optimizer1.zero_grad()
            residual.backward()
            torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model.")

        # 指数平滑更新 mean field flow
        # with torch.no_grad():
        #     for t in range(self.horizon):
        #         for s in range(self.env.state_count):
        #             x_input = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
        #             t_input = torch.tensor(t).float().to(self.device)
        #             new_value = np.exp(mean_field_model(x_input, t_input).detach().cpu().numpy())
        #             self.mf_flow.val[t, s] = alpha * new_value + (1 - alpha) * self.mf_flow.val[t, s]

        self.mean_field_model = mean_field_model

    def _build_cache(self):
        """
        在初始化时, 对每个 state 预先计算:
          1) neighbors_list
          2) one_hot_encoding
        这样在训练循环中就无需重复调用.
        """
        self._neighbors_cache = [None] * self.env.state_count
        self._one_hot_cache = [None] * self.env.state_count

        for s in range(self.env.state_count):
            # 缓存邻居
            neighbors = self.env.get_neighbors(s)
            self._neighbors_cache[s] = neighbors

            # 缓存 one-hot
            x_onehot = F.one_hot(torch.tensor(s), num_classes=self.env.state_count).float().to(self.device)
            self._one_hot_cache[s] = x_onehot


    def train_mean_field_dim_2_new(self,
                               max_epoch: int,
                               learning_rate: float,
                               max_grad_norm: float,
                               num_of_units: int,
                               mean_field_model,
                               optimizer1,
                               alpha=0.1,
                               history_length=5,
                               regularization_factor=0.01,
                               target_mean_field_model=None):
        """
        优化后的 train 方法:
          1. 使用缓存后的 neighbor 和 onehot, 避免重复调用
          2. 减少循环深度, 避免过多张量转换
          3. 可视需求, 简化 history_velocity 计算
        """
        mean_field_model.train()
        combined_data = self.data_policy_theta  # + self.data_expert
        loss_list = []

        # 将 env.action_option 先转成 numpy 数组(如果不是), 加速索引
        action_options = np.array(self.env.action_option)

        for sample in combined_data:
            # 对该sample的 states 进行一次处理
            sample_states = sample.states  # shape: (horizon,)
            # (可选) 如果有 sample.actions 也可在此处一次读取

            for t in range(self.horizon):
                s_int = int(sample_states[t])
                x_onehot = self._one_hot_cache[s_int]  # 直接从缓存取 one-hot
                t_tensor = torch.tensor([t], dtype=torch.float, device=self.device, requires_grad=True)

                # Compute mean_field_now & mean_field_next_time
                mean_field_now = mean_field_model(x_onehot, t_tensor)
                mean_field_next_time = mean_field_model(x_onehot, t_tensor + 1)

                # 获取当前时刻的策略, 选最优动作
                current_policy = self.p_flow.val[t, s_int]  # shape: (action_count,)
                action_index_now = np.argmax(current_policy)
                velocity_now = self.encode_length_and_angle(action_options[action_index_now])

                # 历史平均 velocity
                # 如果 history_length 很大, 可改小; 或者只用上一时刻
                start_h = max(0, t - history_length)
                velocity_hist = []
                for hh in range(start_h, t + 1):
                    s_hh = int(sample_states[hh])
                    neighbours_h = self._neighbors_cache[s_hh]  # 缓存
                    local_vel_list = []
                    for neigh_s in neighbours_h:
                        neighbour_policy = self.p_flow.val[hh, neigh_s]
                        neigh_a_idx = np.argmax(neighbour_policy)
                        neigh_vel = self.encode_length_and_angle(action_options[neigh_a_idx])
                        local_vel_list.append(neigh_vel)
                    # 对邻居速度取平均
                    velocity_hist.append(np.mean(local_vel_list) if local_vel_list else 0.0)

                velocity_last_x = np.mean(velocity_hist) if velocity_hist else 0.0

                # finite differences
                delta_t = self.env.time_unit
                delta_x = self.env.position_unit

                # mean_field_last_x
                neighbours_s = self._neighbors_cache[s_int]  # 缓存
                mf_list = []
                for neigh_s in neighbours_s:
                    x_last = self._one_hot_cache[neigh_s]
                    mf_list.append(mean_field_model(x_last, t_tensor))
                if mf_list:
                    mean_field_last_x = torch.mean(torch.stack(mf_list), dim=0)
                else:
                    mean_field_last_x = mean_field_now.detach() * 0.0  # 无邻居则视为0

                left = (mean_field_next_time - mean_field_now) / delta_t
                right = ((mean_field_now * velocity_now) - (mean_field_last_x * velocity_last_x)) / delta_x
                main_loss = (left + right).abs()

                loss_list.append(main_loss)

        # 优化 mean_field_model
        if loss_list:
            residual = torch.stack(loss_list).mean()
            optimizer1.zero_grad()
            residual.backward()
            torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
            optimizer1.step()
        else:
            print("No data to train mean field model (loss_list empty).")

        # 可选: 指数平滑更新 self.mf_flow
        # with torch.no_grad():
        #     for t in range(self.horizon):
        #         for s in range(self.env.state_count):
        #             x_input = self._one_hot_cache[s]
        #             t_input = torch.tensor([t], dtype=torch.float, device=self.device)
        #             new_val = torch.exp(mean_field_model(x_input, t_input)).detach().cpu().numpy()
        #             self.mf_flow.val[t, s] = alpha*new_val + (1-alpha)*self.mf_flow.val[t, s]

        self.mean_field_model = mean_field_model

    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        warm_epoch = 0.5 * max_epoch
        reset_relative_threshold = 0.1
        reset_absolute_threshold = 3
        history_window = 10
        loss_history = []

        self._neighbors_cache = []
        self._one_hot_cache = []

        # 预先构建缓存
        self._build_cache()

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
            state_shape=self.env.state_count,
            time_horizon=1,
            num_of_units=num_of_units
        ).to(self.device)

        self.target_mean_field_model = MeanFieldModel(
            state_shape=self.env.state_count,
            time_horizon=1,
            num_of_units=num_of_units
        ).to(self.device)
        self.target_mean_field_model.load_state_dict(mean_field_model.state_dict())

        optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
        optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
        optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        init_policy_flow = np.random.rand(self.horizon, self.env.state_count, self.env.action_count)
        init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)
        self.p_flow.val = init_policy_flow

        init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
        for sample in self.data_expert:
            for t in range(self.horizon):
                init_est_expert_mf_flow[t, int(sample.states[t])] += 1
        init_est_expert_mf_flow /= len(self.data_expert)
        self.mf_flow.val = init_est_expert_mf_flow.copy()

        self.update_target_mean_field_flow()

        epoch = 0
        last_loss = float('inf')
        count = 0

        # Polyak averaging参数
        tau = 0.99

        while epoch < max_epoch:
            if epoch == 0:
                loss_history = []
            start_time = time.time()
            if epoch == 0:
                current_mf_flow = init_est_expert_mf_flow.copy()
                self.target_mf_flow = init_est_expert_mf_flow.copy()
            else:
                current_mf_flow = self.target_mf_flow.copy()

            self.mf_flow.val = current_mf_flow
            self.data_policy_theta = self.generate_trajectories_from_policy_flow(
                self.num_of_game_plays, self.num_traj, self.p_flow, self.mf_flow
            )

            # 与 baseline 一样的计算loss逻辑，不修改
            # expert数据计算
            value_per_sample_expert_data = []
            for sample_expert in self.data_expert:
                value_per_step = []
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_expert.states[t])]).float().to(self.device),
                        torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).float().to(self.device),
                        torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
                    )
                    up = torch.exp(reward_component)
                    down = torch.exp(reward_component) + self.p_flow.val[
                        t, int(sample_expert.states[t]), int(sample_expert.actions[t])]
                    value_per_step.append(up / down)
                value_per_sample_expert_data.append(
                    torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
            estimated_expert_data = torch.mean(torch.cat(value_per_sample_expert_data, dim=0).reshape((1, -1)))

            # policy数据计算
            value_per_sample_policy_data = []
            for sample_policy_theta in self.data_policy_theta:
                value_per_step = []
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_policy_theta.states[t])]).float().to(self.device),
                        torch.tensor(self.env.action_option[int(sample_policy_theta.actions[t])]).float().to(
                            self.device),
                        torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
                    )
                    up = torch.exp(reward_component)
                    down = torch.exp(reward_component) + self.p_flow.val[
                        t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])]
                    value_per_step.append(1 - (up / down))
                value_per_sample_policy_data.append(
                    torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
            estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data, dim=0).reshape((1, -1)))

            # 更新reward模型
            optimizer_reward.zero_grad()
            loss = -(estimated_expert_data + estimated_policy_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer_reward.step()

            current_loss = loss.detach().cpu().numpy()
            loss_history.append(current_loss)
            if len(loss_history) > history_window:
                avg_loss = np.mean(loss_history[-history_window:])
            else:
                avg_loss = np.mean(loss_history)

            # 重置判断逻辑与baseline一致，不改 reward/policy 模型结构
            if epoch > 0:
                if epoch < 0.5 * max_epoch:
                    distance_relative = abs(last_loss - current_loss) / (abs(last_loss) + 1e-10)
                    distance_absolute = abs(last_loss - current_loss)
                    if ((distance_relative > reset_relative_threshold or distance_absolute > reset_absolute_threshold)):
                        # 重置模型
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
                            state_shape=self.env.state_count,
                            time_horizon=1,
                            num_of_units=num_of_units
                        ).to(self.device)
                        self.target_mean_field_model = MeanFieldModel(
                            state_shape=self.env.state_count,
                            time_horizon=1,
                            num_of_units=num_of_units
                        ).to(self.device)
                        self.target_mean_field_model.load_state_dict(mean_field_model.state_dict())

                        optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
                        optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
                        optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)

                        count += 1
                        if count < int(0.5 * max_epoch):
                            epoch = 0
                            last_loss = float('inf')
                            continue
                last_loss = current_loss
            else:
                last_loss = current_loss

            # 更新policy模型逻辑不变
            for t in range(self.horizon - 1, -1, -1):
                sum_current = []
                for sample in self.data_policy_theta:
                    value_per_sampler = []
                    for current in range(t, self.horizon):
                        value_per_sampler.append(
                            reward_model(
                                torch.tensor(self.env.state_option[int(sample.states[current])]).float().to(
                                    self.device),
                                torch.tensor(self.env.action_option[int(sample.actions[current])]).float().to(
                                    self.device),
                                torch.from_numpy(self.target_mf_flow[current, :]).float().to(self.device)
                            ) - torch.log(
                                torch.tensor(self.p_flow.val[current, int(sample.states[current]), int(
                                    sample.actions[current])] + 1e-8))
                        )
                    sum_current.append(torch.sum(torch.cat(value_per_sampler, dim=0)).reshape((1, -1)))
                estimated_update = torch.mean(torch.sum(torch.cat(sum_current, dim=0).reshape((1, -1))))

                optimizer_policy.zero_grad()
                loss2 = -(estimated_update)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
                optimizer_policy.step()

                # 更新策略分布
                for s in range(self.env.state_count):
                    new_policy = policy_model(
                        torch.tensor(self.env.state_option[s]).float().to(self.device),
                        torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
                    )
                    tensor_list = new_policy.tolist()
                    self.p_flow.val[t, s] = np.array(tensor_list) / (np.sum(tensor_list) + 1e-8)

                self.data_policy_theta = self.generate_trajectories_from_policy_flow(
                    self.num_of_game_plays,
                    self.num_traj,
                    self.p_flow,
                    self.mf_flow
                )

            # 更新mean field模型间隔保持不变，训练稳定性通过polyak平均平滑
            if (((epoch + 1) % self.update_mean_field_interval == 0) and epoch > 0 and epoch != (
                    max_epoch - 1)) or epoch == 1:
                # 在后半段降低mean field优化器的学习率，增加稳定性
                if epoch > warm_epoch:
                    for g in optimizer_meanfield.param_groups:
                        g['lr'] = learning_rate * 0.5

                for _ in range(int(10)):
                    if self.env.dim == 1:
                        self.train_mean_field_dim_1_new2(
                            max_epoch, learning_rate, max_grad_norm, num_of_units,
                            mean_field_model, optimizer_meanfield,
                            alpha=0.2,
                            regularization_factor=self.mean_field_regularization_factor,
                            target_mean_field_model=self.target_mean_field_model
                        )
                    else:
                        self.train_mean_field_dim_2_new(
                            max_epoch, learning_rate, max_grad_norm, num_of_units,
                            mean_field_model, optimizer_meanfield,
                            alpha=0.2,
                            history_length=max_epoch,
                            regularization_factor=self.mean_field_regularization_factor,
                            target_mean_field_model=self.target_mean_field_model
                        )

                    # Polyak平均更新 target_mean_field_model
                    with torch.no_grad():
                        for param, target_param in zip(mean_field_model.parameters(),
                                                       self.target_mean_field_model.parameters()):
                            target_param.data.copy_(tau * target_param.data + (1 - tau) * param.data)

                    self.update_target_mean_field_flow()

            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, AvgLossWindow: {avg_loss:.4f}, Time: {epoch_duration:.2f}s")

            epoch += 1

        self.reward_model = reward_model
        self.policy_model = policy_model
        self.mean_field_model = mean_field_model

    # def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
    #     warm_epoch = 0.5 * max_epoch
    #
    #     reset_relative_threshold = 0.1  # 相对变化率阈值
    #     reset_absolute_threshold = 3  # 绝对差值阈值，可根据初始loss量级选择
    #     history_window = 10  # 用于计算滑动平均的窗口大小
    #     loss_history = []
    #
    #     reward_model = RewardModel(
    #         state_shape=self.env.state_shape,
    #         action_shape=self.env.action_shape,
    #         mf_shape=self.env.state_count,
    #         num_of_units=num_of_units
    #     ).to(self.device)
    #
    #     policy_model = PolicyModel(
    #         state_shape=self.env.state_shape,
    #         action_shape=self.env.action_count,
    #         mf_shape=self.env.state_count,
    #         num_of_units=num_of_units
    #     ).to(self.device)
    #
    #     mean_field_model = MeanFieldModel(
    #         state_shape=self.env.state_count,
    #         time_horizon=1,
    #         num_of_units=num_of_units
    #     ).to(self.device)
    #
    #     # Target mean field model for stability
    #     self.target_mean_field_model = MeanFieldModel(
    #         state_shape=self.env.state_count,
    #         time_horizon=1,
    #         num_of_units=num_of_units
    #     ).to(self.device)
    #     self.target_mean_field_model.load_state_dict(mean_field_model.state_dict())
    #
    #     optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
    #     optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #     optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #
    #     # Initialize policy flow
    #     init_policy_flow = np.random.rand(self.horizon, self.env.state_count, self.env.action_count)
    #     init_policy_flow /= init_policy_flow.sum(axis=-1, keepdims=True)
    #     self.p_flow.val = init_policy_flow
    #
    #     # Estimate expert mean field flow
    #     init_est_expert_mf_flow = np.zeros((self.horizon, self.env.state_count))
    #     for sample in self.data_expert:
    #         for t in range(self.horizon):
    #             init_est_expert_mf_flow[t, int(sample.states[t])] += 1
    #     init_est_expert_mf_flow /= len(self.data_expert)
    #     self.mf_flow.val = init_est_expert_mf_flow.copy()
    #
    #     # Use the initial mean_field_model to set target_mf_flow
    #     self.update_target_mean_field_flow()
    #
    #     epoch = 0
    #     last_loss = float('inf')
    #     count = 0
    #
    #     while epoch < max_epoch:
    #         start_time = time.time()
    #
    #         # Use stable target_mf_flow for the current epoch
    #         if epoch == 0:
    #             current_mf_flow = init_est_expert_mf_flow.copy()
    #             self.target_mf_flow = init_est_expert_mf_flow.copy()
    #         else:
    #             # After the first epoch, we rely on the stable target_mf_flow
    #             current_mf_flow = self.target_mf_flow.copy()
    #
    #         self.mf_flow.val = current_mf_flow
    #
    #         # Generate new policy trajectories
    #         self.data_policy_theta = self.generate_trajectories_from_policy_flow(
    #             self.num_of_game_plays, self.num_traj, self.p_flow, self.mf_flow
    #         )
    #
    #         # Compute losses from expert data
    #         value_per_sample_expert_data = []
    #         for sample_expert in self.data_expert:
    #             value_per_step = []
    #             for t in range(self.horizon):
    #                 reward_component = reward_model(
    #                     torch.tensor(self.env.state_option[int(sample_expert.states[t])]).float().to(self.device),
    #                     torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).float().to(self.device),
    #                     torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
    #                 )
    #                 up = torch.exp(reward_component)
    #                 down = torch.exp(reward_component) + self.p_flow.val[t, int(sample_expert.states[t]), int(sample_expert.actions[t])]
    #                 value_per_step.append(up / down)
    #             value_per_sample_expert_data.append(torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
    #         estimated_expert_data = torch.mean(torch.cat(value_per_sample_expert_data, dim=0).reshape((1, -1)))
    #
    #         # Compute losses from policy data
    #         value_per_sample_policy_data = []
    #         for sample_policy_theta in self.data_policy_theta:
    #             value_per_step = []
    #             for t in range(self.horizon):
    #                 reward_component = reward_model(
    #                     torch.tensor(self.env.state_option[int(sample_policy_theta.states[t])]).float().to(self.device),
    #                     torch.tensor(self.env.action_option[int(sample_policy_theta.actions[t])]).float().to(self.device),
    #                     torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
    #                 )
    #                 up = torch.exp(reward_component)
    #                 down = torch.exp(reward_component) + self.p_flow.val[t, int(sample_policy_theta.states[t]), int(sample_policy_theta.actions[t])]
    #                 value_per_step.append(1 - (up / down))
    #             value_per_sample_policy_data.append(torch.sum(torch.log(torch.cat(value_per_step, dim=0))).reshape((1, -1)))
    #         estimated_policy_data = torch.mean(torch.cat(value_per_sample_policy_data, dim=0).reshape((1, -1)))
    #
    #         # Update reward model
    #         optimizer_reward.zero_grad()
    #         loss = -(estimated_expert_data + estimated_policy_data)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
    #         optimizer_reward.step()
    #
    #         current_loss = loss.detach().cpu().numpy()
    #         loss_history.append(current_loss)
    #         if len(loss_history) > history_window:
    #             # 计算最近history_window个epoch的平均loss
    #             avg_loss = np.mean(loss_history[-history_window:])
    #         else:
    #             avg_loss = np.mean(loss_history)
    #
    #         # Check for loss divergence and reset if needed
    #         if epoch > 0:
    #             if epoch < 0.5 * max_epoch:
    #
    #                 distance_relative = abs(last_loss - loss.detach().cpu().numpy()) / (abs(last_loss) + 1e-10)
    #                 distance_absolute = abs(last_loss - loss.detach().cpu().numpy())
    #
    #                 # 我们同时检查绝对变化和相对变化
    #                 # 只有在两者都很大时才考虑重置（根据需求自行调整逻辑）
    #
    #                 # 例如，当loss已经非常低时，绝对变化很小但相对变化看似很大就没必要重置
    #                 # 当loss非常高时，即使有较大的绝对变化，也许是正常范围
    #                 # 也可以比较当前loss与平均loss的偏差，如果当前loss明显高于最近平均值，也可作为触发条件
    #
    #                   # 当当前loss > 2倍滑动平均时认为是不正常上升
    #                 if (distance_relative > reset_relative_threshold and distance_absolute > reset_absolute_threshold) \
    #                         or (current_loss > avg_loss * 2.0):
    #                     # Reset models and optimizers if needed
    #                     reward_model = RewardModel(
    #                         state_shape=self.env.state_shape,
    #                         action_shape=self.env.action_shape,
    #                         mf_shape=self.env.state_count,
    #                         num_of_units=num_of_units
    #                     ).to(self.device)
    #
    #                     policy_model = PolicyModel(
    #                         state_shape=self.env.state_shape,
    #                         action_shape=self.env.action_count,
    #                         mf_shape=self.env.state_count,
    #                         num_of_units=num_of_units
    #                     ).to(self.device)
    #
    #                     mean_field_model = MeanFieldModel(
    #                         state_shape=self.env.state_count,
    #                         time_horizon=1,
    #                         num_of_units=num_of_units
    #                     ).to(self.device)
    #
    #                     self.target_mean_field_model = MeanFieldModel(
    #                         state_shape=self.env.state_count,
    #                         time_horizon=1,
    #                         num_of_units=num_of_units
    #                     ).to(self.device)
    #                     self.target_mean_field_model.load_state_dict(mean_field_model.state_dict())
    #
    #                     optimizer_reward = optim.Adam(reward_model.parameters(), lr=learning_rate)
    #                     optimizer_policy = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #                     optimizer_meanfield = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #
    #                     count += 1
    #                     if count < int(0.5 * max_epoch):
    #                         epoch = 0
    #                         last_loss = float('inf')
    #                         continue
    #                 last_loss = loss.detach().cpu().numpy()
    #         else:
    #             last_loss = loss.detach().cpu().numpy()
    #
    #         # Update policy model
    #         for t in range(self.horizon - 1, -1, -1):
    #             sum_current = []
    #             for sample in self.data_policy_theta:
    #                 value_per_sampler = []
    #                 for current in range(t, self.horizon):
    #                     value_per_sampler.append(
    #                         reward_model(
    #                             torch.tensor(self.env.state_option[int(sample.states[current])]).float().to(self.device),
    #                             torch.tensor(self.env.action_option[int(sample.actions[current])]).float().to(self.device),
    #                             torch.from_numpy(self.target_mf_flow[current, :]).float().to(self.device)
    #                         ) - torch.log(torch.tensor(self.p_flow.val[current, int(sample.states[current]), int(sample.actions[current])]))
    #                     )
    #                 sum_current.append(torch.sum(torch.cat(value_per_sampler, dim=0)).reshape((1, -1)))
    #             estimated_update = torch.mean(torch.sum(torch.cat(sum_current, dim=0).reshape((1, -1))))
    #             optimizer_policy.zero_grad()
    #             loss2 = -(estimated_update)
    #             loss2.backward()
    #             U.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
    #             optimizer_policy.step()
    #
    #             # Update policy
    #             for s in range(self.env.state_count):
    #                 new_policy = policy_model(
    #                     torch.tensor(self.env.state_option[s]).float().to(self.device),
    #                     torch.from_numpy(self.target_mf_flow[t, :]).float().to(self.device)
    #                 )
    #                 tensor_list = new_policy.tolist()
    #                 # Here we update the policy flow
    #                 self.p_flow.val[t, s] = tensor_list
    #
    #             self.data_policy_theta = self.generate_trajectories_from_policy_flow(
    #                 self.num_of_game_plays,
    #                 self.num_traj,
    #                 self.p_flow,
    #                 self.mf_flow
    #             )
    #
    #         # Update mean field model only every update_mean_field_interval epochs
    #         #  (((epoch+1) % self.update_mean_field_interval == 0) and epoch > 0 and epoch != (max_epoch-1) )
    #         if (((epoch+1) % self.update_mean_field_interval == 0) and epoch > 0 and epoch != (max_epoch-1) ) or epoch == 1:
    #             for _ in range(int(0.5 * max_epoch)):
    #                 if self.env.dim == 1:
    #                     self.train_mean_field_dim_1_new2(
    #                         max_epoch, learning_rate, max_grad_norm, num_of_units,
    #                         mean_field_model, optimizer_meanfield,
    #                         alpha=0.2,
    #                         history_length=max_epoch,
    #                         regularization_factor=self.mean_field_regularization_factor,
    #                         target_mean_field_model=self.target_mean_field_model
    #                     )
    #                 else:
    #                     self.train_mean_field_dim_2(
    #                         max_epoch, learning_rate, max_grad_norm, num_of_units,
    #                         mean_field_model, optimizer_meanfield,
    #                         alpha=0.2,
    #                         history_length=max_epoch,
    #                         regularization_factor=self.mean_field_regularization_factor,
    #                         target_mean_field_model=self.target_mean_field_model
    #                     )
    #
    #
    #                 # Update the target mean field model
    #                 self.target_mean_field_model.load_state_dict(mean_field_model.state_dict())
    #                 # Recompute target_mf_flow
    #                 self.update_target_mean_field_flow()
    #
    #         end_time = time.time()
    #         epoch_duration = end_time - start_time
    #         print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Time: {epoch_duration:.2f}s")
    #
    #         epoch += 1
    #
    #     # Assign trained models
    #     self.reward_model = reward_model
    #     self.policy_model = policy_model
    #     self.mean_field_model = mean_field_model



    def generate_trajectories_from_policy_flow(self, num_game_play: int, num_traj: int, current_policy_flow,
                                               current_mean_field_flow, deterministic=False):
        states = [i for i in range(self.env.state_count)]
        actions = [i for i in range(self.env.action_count)]
        assert current_mean_field_flow is not None

        data = [Trajectory(states=None, actions=None, horizon=self.horizon) for _ in range(num_game_play * num_traj)]

        for i in range(num_game_play * num_traj):
            # 采样初始状态
            # probabilities = current_mean_field_flow.val[0, :]
            # probabilities = probabilities / np.sum(probabilities)  # Normalize to sum to 1

            probabilities = current_mean_field_flow.val[0, :]
            exp_values = np.exp(probabilities - np.max(probabilities))  # Subtract max for numerical stability
            probabilities = exp_values / np.sum(exp_values)

            s = int(np.random.choice(states, p=probabilities))
            data[i].states[0] = s

            for t in range(self.horizon):
                # 获取策略概率并归一化
                # policy_probs = current_policy_flow.val[t, s, :].copy()
                # sum_probs = sum(policy_probs)
                # if sum_probs > 0:
                #     policy_probs /= sum_probs
                # else:
                #     policy_probs = np.ones_like(policy_probs) / len(policy_probs)

                # Original policy probabilities
                policy_probs = current_policy_flow.val[t, s, :].copy()

                # Apply softmax for normalization
                exp_probs = np.exp(policy_probs - np.max(policy_probs))  # Subtract max for numerical stability
                policy_probs = exp_probs / np.sum(exp_probs)

                # 选择动作
                if deterministic:
                    max_actions = np.flatnonzero(policy_probs == policy_probs.max())
                    a = int(np.random.choice(max_actions))
                else:
                    a = int(np.random.choice(actions, p=policy_probs))

                data[i].actions[t] = a

                # 计算下一个状态
                if t < self.horizon - 1:
                    s_next = self.env.dynamics(State(state=s), Action(action=a),
                                               MeanField(mean_field=current_mean_field_flow.val[t]))
                    data[i].states[t + 1] = s_next.val[0]
                    s = int(s_next.val[0])

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
