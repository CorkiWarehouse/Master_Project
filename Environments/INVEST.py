import math
import numpy as np

from core import State, Reward, Environment, MeanField


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'invest'
        self.max_quality_level = 9
        self.state_shape = 10
        self.action_shape = 2
        if is_original_dynamics == 0:
            self.quality_threshold = 4
        else:
            self.quality_threshold = 5
        # hyper-parameters of the reward function
        self.d, self.c, self.l = 0.3, 0.2, 0.2

    def get_reward(self, state, action, mean_field):
        mean = self.mean_quality(mean_field)
        reward = self.d * state.val / 10 - self.c * mean - self.l * action.val
        return Reward(reward=reward)

    def mean_quality(self, mean_field):
        mean = 0.0
        for quality_level in range(self.state_shape):
            mean += quality_level * mean_field.val[quality_level]
        return mean

    def advance(self, policy, mean_field):
        next_mean_field = MeanField(mean_field=None, s=self.state_shape)
        mean_quality = self.mean_quality(mean_field)
        for quality_level in range(self.state_shape):
            none_prob = policy.val[quality_level, 0]
            imp_prob = policy.val[quality_level, 1] / (self.state_shape - quality_level)
            if mean_quality <= self.quality_threshold:
                next_mean_field.val[quality_level] += mean_field.val[quality_level] * (none_prob + imp_prob)
                for s in range(quality_level + 1, self.state_shape):
                    next_mean_field.val[s] += mean_field.val[quality_level] * imp_prob
            else:
                if quality_level == self.max_quality_level:
                    next_mean_field.val[quality_level] += mean_field.val[quality_level]
                else:
                    next_mean_field.val[quality_level] += mean_field.val[quality_level] * (none_prob + 2 * imp_prob)
                    # get the max quality level that could be reached
                    level_bound = quality_level + math.ceil((self.state_shape - quality_level) / 2) - 1
                    for s in range(quality_level + 1, level_bound):
                        next_mean_field.val[s] += mean_field.val[quality_level] * 2 * imp_prob
                    # get the probability of reaching the level bound
                    if level_bound > quality_level:
                        if (self.state_shape - quality_level) % 2 == 0:
                            next_mean_field.val[level_bound] += mean_field.val[quality_level] * 2 * imp_prob
                        else:
                            next_mean_field.val[level_bound] += mean_field.val[quality_level] * imp_prob
        return next_mean_field

    def dynamics(self, state, action, mean_field):
        if int(action.val[0]) == 1:
            mean_quality = self.mean_quality(mean_field)
            if mean_quality < self.quality_threshold:
                s = int(np.random.choice(np.array(range(state.val[0], self.state_shape))))
                return State(state=s)
            else:
                s = int(np.random.choice(np.array(range(state.val[0], state.val[0] + int((10 - state.val[0]) / 2) +1 ))))
                return State(state=s)
        else:
            return state

    def trans_prob(self, state, action, mean_field):
        prob = np.zeros(10)
        if int(action.val[0]) == 1:
            mean_quality = self.mean_quality(mean_field)
            if mean_quality < self.quality_threshold:
                for i in range(int(state.val[0]), self.state_shape):
                    prob[i] = 1.0 / (self.state_shape - int(state.val[0]))
            else:
                num = (self.state_shape - int(state.val[0])) / 2.0
                for i in range(int(state.val[0]), int(state.val[0]) + int(num)):
                    prob[i] = 2.0 / (self.state_shape - int(state.val[0]))
                if num > int(num):
                    prob[int(state.val[0]) + int(num)] = 1.0 / (self.state_shape - int(state.val[0]))
        else:
            prob[int(state.val[0])] = 1.0
        return prob
