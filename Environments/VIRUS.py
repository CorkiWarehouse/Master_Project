"""
a large number of agents can choose between social Distancing (D, action 0) or going Out (U, action 1). If a Susceptible (S, state 0)
agent chooses social distancing, they may not become Infected (I, state 1). Otherwise, an agent may become infected with a
probability proportional to the number of agents being infected. If infected, an agent will recover with a fixed chance
every time step. Both social distancing and being infected have an associated cost.
"""

from core import State, Action, Reward, Environment, MeanField
import numpy as np


class Env(Environment):
    def __init__(self, is_original_dynamics, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'virus'
        self.state_shape = 1
        self.action_shape = 1
        # hyper parameters for new dynamics
        self.prob = 0.3

        self.state_count = 2
        self.action_count = 2

        self.action_option = [0,1]
        self.state_option = [0,1]

        self.time_unit = 1
        self.position_unit = 1

        self.init_mf = None
        self.dim = 1

    def get_reward(self, state, action, mean_field):
        if int(action.val[0] == 0):
            return Reward(reward=-0.5)
        else:
            return Reward(reward=-1.0)

    def advance(self, policy, mean_field):
        next_mean_field = MeanField(mean_field=None, s=self.state_count)
        for s in range(self.state_count):
            for ss in range(self.state_count):
                for a in range(self.action_count):
                    next_mean_field.val[s] += mean_field.val[ss] * policy.val[ss, a] * self.trans_prob(State(state=ss), Action(action=a), mean_field=mean_field)[s]
        return next_mean_field

    def dynamics(self, state, action, mean_field):
        if int(state.val[0] == 1):
            s = np.random.choice([0, 1], 1, p=self.trans_prob(state, action, mean_field))
            return State(state=s)
        else:
            if int(action.val[0] == 1):
                s = np.random.choice([0, 1], 1, p=self.trans_prob(state, action, mean_field))
                return State(state=s)
            else:
                return State(state=1)

    def trans_prob(self, state, action, mean_field):
        if self.is_original_dynamics == 0:
            if int(state.val[0] == 1):
                return np.array([0.3, 0.7])
            else:
                if int(action.val[0] == 1):
                    return np.array([1 - 0.81 * mean_field.val[1], 0.81 * mean_field.val[1]])
                else:
                    return np.array([0.0, 1.0])
        else:
            if int(state.val[0] == 1):
                return np.array([0.2, 0.8])
            else:
                if int(action.val[0] == 1):
                    return np.array([1 - 0.64 * mean_field.val[1], 0.64 * mean_field.val[1]])
                else:
                    return np.array([0.0, 1.0])