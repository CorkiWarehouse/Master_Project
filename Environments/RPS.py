"""
Each of the agents can choose between Rock (0), Paper (1) and Scissors (2), and obtains a reward proportional to double
the number of beaten agents minus the number of agents beating the agent. Use 3 to denote Null state.

Original Dynamics: The next state is the current action with probability 1
New Dynamics: The next state is the current action with probability 0.8 and an arbitrary one with probability 0.2
"""


import numpy as np
from core import State, Reward, Environment, MeanField


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'RPS'
        self.state_shape = 3
        self.action_shape = 3
        # hyper parameters for new dynamics
        self.prob = 0.3

    def get_reward(self, state, action, mean_field):
        if int(action.val[0]) == 0:
            return Reward(reward=2 * mean_field.val[2] - mean_field.val[1])
        elif int(action.val[0]) == 1:
            return Reward(reward=4 * mean_field.val[0] - 2 * mean_field.val[2])
        else:
            return Reward(reward=6 * mean_field.val[1] - 2 * mean_field.val[0])

    def advance(self, policy, mean_field):
        next_mean_field = MeanField(mean_field=None, s=self.state_shape)
        if self.is_original_dynamics == 0:
            next_mean_field.val[0] = mean_field.val[0] * policy.val[0, 0] + mean_field.val[1] * policy.val[1, 0] + mean_field.val[2] * policy.val[2, 0]
            next_mean_field.val[1] = mean_field.val[0] * policy.val[0, 1] + mean_field.val[1] * policy.val[1, 1] + mean_field.val[2] * policy.val[2, 1]
            next_mean_field.val[2] = mean_field.val[0] * policy.val[0, 2] + mean_field.val[1] * policy.val[1, 2] + mean_field.val[2] * policy.val[2, 2]
        else:
            next_mean_field.val[0] = 0.7 * (mean_field.val[0] * policy.val[0, 0] + mean_field.val[1] * policy.val[1, 0] + mean_field.val[2] * policy.val[2, 0]) + 0.1
            next_mean_field.val[1] = 0.7 * (mean_field.val[0] * policy.val[0, 1] + mean_field.val[1] * policy.val[1, 1] + mean_field.val[2] * policy.val[2, 1]) + 0.1
            next_mean_field.val[2] = 0.7 * (mean_field.val[0] * policy.val[0, 2] + mean_field.val[1] * policy.val[1, 2] + mean_field.val[2] * policy.val[2, 2]) + 0.1
        return next_mean_field

    def dynamics(self, state, action, mean_field):
        if self.is_original_dynamics == 0:
            return State(state=action.val[0])
        else:
            if int(action.val[0] == 0):
                s = np.random.choice([0, 1, 2], 1, p=[0.8, 0.1, 0.1])
                return State(state=s)
            elif int(action.val[0] == 1):
                s = np.random.choice([0, 1, 2], 1, p=[0.1, 0.8, 0.1])
                return State(state=s)
            else:
                s = np.random.choice([0, 1, 2], 1, p=[0.1, 0.1, 0.8])
                return State(state=s)

    def trans_prob(self, state, action, mean_field):
        if self.is_original_dynamics == 0:
            if int(action.val[0]) == 0:
                return np.array([1.0, 0.0, 0.0])
            elif int(action.val[0]) == 1:
                return np.array([0.0, 1.0, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        else:
            if int(action.val[0] == 0):
                return np.array([0.8, 0.1, 0.1])
            elif int(action.val[0] == 1):
                return np.array([0.1, 0.8, 0.1])
            else:
                return np.array([0.1, 0.1, 0.8])

