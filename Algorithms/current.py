"""
This is the optimized Physical Informed AIRL (PIIRL)
"""

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

import time

MAX = 100000  # maximum number of iterations
MIN = 1e-10


class PIIRLN(IRL):
    '''
        PIIRL Algorithm with optimized mean field training and adjusted hyperparameters
    '''

    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        # Adjusted warm-up epochs
        warm_epoch = int(0.2 * max_epoch)

        # Initialize models
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

        # MeanFieldModel adjusted to accept time as input
        mean_field_model = MeanFieldModel(
            state_shape=self.env.state_shape,
            time_horizon=1,
            num_of_units=num_of_units
        ).to(self.device)

        # Optimizers with adjusted learning rates
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

        # Separate mean field model training
        # self.train_mean_field_model(mean_field_model, optimizer_meanfield, max_grad_norm,num_epochs=100)

        epoch = 0
        last_loss = float('inf')
        count = 0

        while epoch < max_epoch:
            start_time = time.time()

            # Generate trajectories from current policy and mean field
            self.data_policy_theta = self.generate_trajectories_from_policy_flow(
                self.num_of_game_plays, self.num_traj, self.p_flow, self.mf_flow
            )

            # Update mean field flow using the mean field model
            estimated_mean_field_flow = np.zeros((self.horizon, self.env.state_count))
            for t in range(self.horizon):
                for s in range(self.env.state_count):
                    x_input = torch.tensor(self.env.state_option[s]).to(self.device, torch.float)
                    t_input = torch.tensor([t / self.horizon]).to(self.device, torch.float)  # Normalize time
                    with torch.no_grad():
                        estimated_mean_field_flow[t, s] = mean_field_model(x_input, t_input).item()
                # Normalize the mean field flow to sum to one
                total = np.sum(estimated_mean_field_flow[t, :])
                if total > 0:
                    estimated_mean_field_flow[t, :] /= total
                else:
                    # Use initial mean field if total is zero
                    estimated_mean_field_flow[t, :] = init_est_expert_mf_flow[t, :]

            # Update the mean field flow
            self.mf_flow.val = estimated_mean_field_flow.copy()

            # Compute losses
            value_per_sample_expert_data = []
            value_per_sample_policy_data = []

            # Compute loss from expert data
            for sample_expert in self.data_expert:
                value_per_step = []
                for t in range(self.horizon):
                    reward_component = reward_model(
                        torch.tensor(self.env.state_option[int(sample_expert.states[t])]).to(self.device, torch.float),
                        torch.tensor(self.env.action_option[int(sample_expert.actions[t])]).to(self.device, torch.float),
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
            loss = -(estimated_expert_data + estimated_policy_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_grad_norm)
            optimizer_reward.step()

            # Update policy model
            self.update_policy_model(policy_model, optimizer_policy, estimated_mean_field_flow,max_grad_norm)

            # Periodically retrain the mean field model with updated data
            self.train_mean_field_model(mean_field_model, optimizer_meanfield, max_grad_norm,num_epochs=50)

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Time: {epoch_duration:.2f}s")
            epoch += 1

        # Assign the trained models
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.mean_field_model = mean_field_model

    def train_mean_field_model(self, mean_field_model, optimizer_meanfield,max_grad_norm, num_epochs=100):
        """
        Separately train the mean field model using data from expert and policy trajectories
        """
        mean_field_model.train()

        # Collect training data
        state_list = []
        time_list = []
        mf_list = []

        # From expert data
        for sample in self.data_expert:
            for t in range(self.horizon):
                state = self.env.state_option[int(sample.states[t])]
                time_norm = t / self.horizon  # Normalize time
                state_list.append(state)
                time_list.append([time_norm])
                mf_list.append(self.mf_flow.val[t, int(sample.states[t])])

        # From policy data
        for sample in self.data_policy_theta:
            for t in range(self.horizon):
                state = self.env.state_option[int(sample.states[t])]
                time_norm = t / self.horizon
                state_list.append(state)
                time_list.append([time_norm])
                mf_list.append(self.mf_flow.val[t, int(sample.states[t])])

        # Convert to tensors
        states_tensor = torch.tensor(state_list, dtype=torch.float32).to(self.device)
        times_tensor = torch.tensor(time_list, dtype=torch.float32).to(self.device)
        mf_tensor = torch.tensor(mf_list, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Training loop
        batch_size = 64
        dataset = torch.utils.data.TensorDataset(states_tensor, times_tensor, mf_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_states, batch_times, batch_mf in dataloader:
                optimizer_meanfield.zero_grad()
                preds = mean_field_model(batch_states, batch_times)
                loss = F.mse_loss(preds, batch_mf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
                optimizer_meanfield.step()
                epoch_loss += loss.item() * batch_states.size(0)
            epoch_loss /= len(dataset)
            # Optional: print loss for mean field model training
            # print(f"Mean Field Model Training - Epoch {epoch}, Loss: {epoch_loss:.4f}")

        self.mean_field_model = mean_field_model

    def update_policy_model(self, policy_model, optimizer_policy, estimated_mean_field_flow, max_grad_norm):
        """
        Update the policy model using the current estimated mean field flow
        """
        optimizer_policy.zero_grad()
        loss_policy = 0.0

        # Collect data for training
        state_list = []
        mf_list = []
        action_probs_list = []

        for t in range(self.horizon):
            for s in range(self.env.state_count):
                state = self.env.state_option[s]
                mf = estimated_mean_field_flow[t]
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                mf_tensor = torch.tensor(mf, dtype=torch.float32).to(self.device)
                policy_output = policy_model(state_tensor, mf_tensor)
                policy_probs = F.softmax(policy_output / self.env.beta, dim=0)
                # Update policy flow
                self.p_flow.val[t, s] = policy_probs.detach().cpu().numpy()
                # Collect data for loss computation
                action_probs_list.append(policy_probs)
                state_list.append(state_tensor)
                mf_list.append(mf_tensor)

        # Compute policy entropy loss (optional)
        entropy_loss = -torch.mean(torch.stack([
            torch.sum(p * torch.log(p + 1e-9)) for p in action_probs_list
        ]))

        # Total loss (adjust weight as needed)
        loss_policy = -entropy_loss  # Encourage higher entropy for exploration
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        optimizer_policy.step()

    def generate_trajectories_from_policy_flow(self, num_game_play: int, num_traj: int, current_policy_flow,
                                               current_mean_field_flow, deterministic=False):
        """
        Generate trajectories based on the current policy flow and mean field flow
        """
        states = [i for i in range(self.env.state_count)]
        actions = [i for i in range(self.env.action_count)]
        assert current_mean_field_flow is not None

        data = [Trajectory(states=None, actions=None, horizon=self.horizon) for _ in range(num_game_play * num_traj)]

        for i in range(num_game_play * num_traj):
            # Sample the initial state
            s = int(np.random.choice(states, p=current_mean_field_flow.val[0, :]))
            data[i].states[0] = s

            for t in range(self.horizon):
                # Get policy probabilities and normalize
                policy_probs = current_policy_flow.val[t, s, :].copy()
                sum_probs = policy_probs.sum()
                if sum_probs > 0:
                    policy_probs /= sum_probs
                else:
                    policy_probs = np.ones_like(policy_probs) / len(policy_probs)

                # Choose action
                if deterministic:
                    max_actions = np.flatnonzero(policy_probs == policy_probs.max())
                    a = int(np.random.choice(max_actions))
                else:
                    a = int(np.random.choice(actions, p=policy_probs))

                data[i].actions[t] = a

                # Compute next state
                if t < self.horizon - 1:
                    s_next = self.env.dynamics(State(state=s), Action(action=a),
                                               MeanField(mean_field=current_mean_field_flow.val[t]))
                    data[i].states[t + 1] = s_next.val[0]
                    s = int(s_next.val[0])

        return data
