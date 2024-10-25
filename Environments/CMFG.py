

import numpy as np
import random

from core import State, Reward, Environment, MeanField, Action


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'CMFG'

        self.state_shape = 2  # 2D grid
        self.action_shape = 2  # 2D actions

        # Define the grid size
        grid_size = 5

        self.state_option = np.array([[x, y] for x in range(1, grid_size + 1) for y in range(1, grid_size + 1)])
        self.state_count = len(self.state_option)

        self.action_option = np.array([[x, y] for x in [-1,1] for y in [-1, 1]])
        self.action_option = self.action_option[np.any(self.action_option != 0, axis=1)]  # Remove (0,0) if not allowed
        self.action_count = len(self.action_option)

        # Other parameters
        self.time_unit = 1
        self.position_unit = 1
        self.dim = 2
        self.c = 0.5  # Action cost coefficient

        # Initialize mean field
        self.init_mf = np.random.uniform(0, 1, self.state_count)
        self.init_mf /= np.sum(self.init_mf)
        self.init_mf = MeanField(mean_field=self.init_mf)

        # Noise parameters
        self.p = 0.7
        self.noise_option = self.action_option

    # this method is used to test the Distance-Based Reward
    # To have a smoother reward function that decreases with distance
    def R(self,s):
        favorable_states = np.array([[3, 3], [3, 4], [4, 3], [4, 4]])
        A = 1  # Peak reward
        sigma = 1  # Controls the spread
        distances = np.linalg.norm(favorable_states - s, axis=1)
        return np.sum(A * np.exp(- (distances ** 2) / (2 * sigma ** 2)))

    def get_reward(self, state, action, mean_field):


        rs = self.R(state.val[0])
        reward = (1 - self.c * mean_field.val[state.val[0]]) * rs

        return Reward(reward=reward)

    # here is the mean field changing
    def advance(self, policy, mean_field) -> MeanField:
        next_mean_field = MeanField(mean_field=None, s=self.state_count)

        # here is all the next_state
        for next_state in range(self.state_count):
            sum_next = 0
            # here is all the current state which can reach to next one
            # but for our time_unit is 5
            # not all the action that can reach this 'next_state'
            for current_state in range(self.state_count):

                sum_policy_transition = 0

                # we directly check all the valid actions
                for current_action in range(self.action_count):
                    current_state_policy = policy.val[current_state, current_action]

                    # for we have the deterministic policy
                    # our prob is 0 or 1
                    prob_transition = \
                    self.trans_prob(State(state=current_state), Action(action=current_action), mean_field)[next_state]

                    sum_policy_transition += prob_transition * current_state_policy

                # then we need to multiply it with mean field
                # print("sum_policy_transition", sum_policy_transition)
                sum_next += sum_policy_transition * mean_field.val[current_state]
                # print("sum_next", sum_next)

            next_mean_field.val[next_state] = sum_next
            # print(test_set)

        # at last, we need to normalize the output
        # Normalize the mean field values so they sum to 1
        total = np.sum(next_mean_field.val)
        if total > 0:  # Avoid division by zero
            next_mean_field.val /= total

        return next_mean_field

    # we do not consider the noise in our environment
    def dynamics(self, state, action, mean_field=None) -> State:

        # get the current position and action
        current_x = self.state_option[state.val[0]][0]
        current_y = self.state_option[state.val[0]][1]
        current_action_x = self.action_option[action.val[0]][0]
        current_action_y = self.action_option[action.val[0]][1]

        # do the transition
        next_x = current_x +  current_action_x * self.time_unit
        next_y = current_y + current_action_y * self.time_unit


        # find the corresponding state
        next_state = [next_x, next_y]

        next_state = self.modulate_value(next_state)

        # Convert next_state to a numpy array for broadcasting
        next_state_np = np.array(next_state)


        # Calculate the index of the next_state in the grid
        # must give the int
        index = int(np.where((self.state_option == next_state_np).all(axis=1))[0][0])

        return State(state=index)

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        # this is the length value
        next_prob = np.zeros(self.state_count)

        # get the current position and action
        current_x = self.state_option[state.val[0]][0]
        current_y = self.state_option[state.val[0]][1]
        current_action_x = self.action_option[action.val[0]][0]
        current_action_y = self.action_option[action.val[0]][1]

        # do the transition
        next_x = current_x + current_action_x * self.time_unit
        next_y = current_y + current_action_y * self.time_unit

        # find the corresponding state
        next_state = [next_x, next_y]

        next_state = self.modulate_value(next_state)
        # Convert next_state to a numpy array for broadcasting
        next_state_np = np.array(next_state)

        # Calculate the index of the next_state in the grid
        # must give the int
        index = int(np.where((self.state_option == next_state_np).all(axis=1))[0][0])
        next_prob[index] = 1

        return next_prob

    # this used to make the value do not go outof the bound
    def modulate_value(self, value):
        new_result = [0 for i in range(len(value))]
        for i in range(len(value)):
            if value[i] > 5:
                new_result[i] = 1
            elif value[i] < 1:
                new_result[i] = 5
            else:
                new_result[i] = value[i]

        return new_result

    def get_neighbors(self, state, mean_field=None):
        neighbors = []
        current_position = self.state_option[state]  # Current state's position

        # Iterate over all possible previous states
        for prev_state_index, prev_position in enumerate(self.state_option):
            # Iterate over all possible actions
            for action_index, action in enumerate(self.action_option):
                if self.is_original_dynamics == 0:
                    # Deterministic dynamics without noise
                    next_position = (prev_position + action - 1) % 5 + 1  # Adjust for 1-based indexing
                    if np.array_equal(next_position, current_position):
                        neighbors.append(prev_state_index)
                        # break  # No need to check other actions for this state
                else:
                    # Stochastic dynamics with noise
                    for noise_index, noise in enumerate(self.noise_option):
                        # Compute total velocity
                        total_velocity = action + noise
                        next_position = (prev_position + total_velocity - 1) % 5 + 1  # Adjust for 1-based indexing

                        # Check if next_position matches current_position
                        if np.array_equal(next_position, current_position):
                            neighbors.append(prev_state_index)
                            break  # Found a valid transition
                    else:
                        continue  # Continue if inner loop wasn't broken
                    break  # Break outer loop if inner loop was broken
        return neighbors




