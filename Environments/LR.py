"""
LR model. Agents can chose to move LEFT (0) or RIGHT (1).
The reward is determine by the proportion of agents that chose
to move left or right.
"""

import numpy as np
from core import State, Reward, Environment, MeanField


class Env(Environment):
    def __init__(self, is_original_dynamics, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'LR'
        self.state_shape = 2
        self.action_shape = 2
        # hyper parameters for new dynamics
        self.prob = 0.2

    def get_reward(self, state, action, mean_field):
        if int(action.val[0]) == 0:
            return Reward(reward=-mean_field.val[0])
        else:
            return Reward(reward=-mean_field.val[1] * 2)

    def advance(self, policy, mean_field):
        next_mean_field = MeanField(mean_field=None, s=self.state_shape)
        if self.is_original_dynamics == 0:
            next_mean_field.val[0] = mean_field.val[0] * policy.val[0, 0] + mean_field.val[1] * policy.val[1, 0]
            next_mean_field.val[1] = mean_field.val[0] * policy.val[0, 1] + mean_field.val[1] * policy.val[1, 1]
        else:
            next_mean_field.val[0] = 0.8 *( mean_field.val[0] * policy.val[0, 0] + mean_field.val[1] * policy.val[1, 0] ) + 0.1
            next_mean_field.val[1] = 0.8 *( mean_field.val[0] * policy.val[0, 1] + mean_field.val[1] * policy.val[1, 1] ) + 0.1
        return next_mean_field

    def dynamics(self, state, action, mean_field):
        if self.is_original_dynamics == 0:
            return State(state=int(action.val[0]))
        else:
            if int(action.val[0] == 0):
                s = np.random.choice([0, 1], 1, p=[0.9, 0.1])
                return State(state=s)
            else:
                s = np.random.choice([0, 1], 1, p=[0.1, 0.9])
                return State(state=s)

    def trans_prob(self, state, action, mean_field):
        if self.is_original_dynamics == 0:
            if int(action.val[0]) == 1:
                return np.array([1.0, 0.0])
            else:
                return np.array([0.0, 1.0])
        else:
            if int(action.val[0] == 0):
                return np.array([0.9, 0.1])
            else:
                return np.array([0.1, 0.9])
