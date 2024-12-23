# import random
# import numpy as np
# from core import State, Reward, Environment, MeanField, Action
#
# class Env(Environment):
#
#     def __init__(self, is_original_dynamics: int, beta: float):
#         super().__init__(is_original_dynamics, beta)
#         self.name = 'CARS'
#
#         self.state_shape = 1
#         self.action_shape = 1
#
#         self.position_unit = 1 /8
#         self.time_unit = self.position_unit
#         self.road_length = 1
#         self.velocity_max = 1
#         self.total_time = 20
#
#         self.position_precision = 5  # Precision for positions and velocities
#
#         self.action_option = [round(v, self.position_precision) for v in np.arange(self.time_unit, self.velocity_max + self.time_unit, self.time_unit).tolist()]
#         self.state_option = [round(pos, self.position_precision) for pos in np.arange(0, 1, self.position_unit).tolist()]
#
#         self.state_count = len(self.state_option)
#         self.action_count = len(self.action_option)
#
#         self.p = 0.1
#         self.noise_option = self.action_option.copy()
#
#         self.init_mf = None
#         self.dim = 1
#
#         # Create a mapping from positions to indices
#         self.position_to_index = {pos: idx for idx, pos in enumerate(self.state_option)}
#
#     def get_reward(self, state, action, mean_field):
#         velocity = self.action_option[action.val[0]]
#         x_position = state.val[0]
#
#         U = 1 - mean_field.val[x_position]
#
#         reward = 0.5 * ((U - velocity) ** 2)
#         return Reward(reward=reward)
#
#     def advance(self, policy, mean_field) -> MeanField:
#         next_mean_field = MeanField(mean_field=None, s=self.state_count)
#         for next_state in range(self.state_count):
#             sum_next = 0
#             for current_state in range(self.state_count):
#                 sum_policy_transition = 0
#                 for current_action in range(self.action_count):
#                     current_state_policy = policy.val[current_state, current_action]
#                     if current_state_policy == 0:
#                         continue
#                     prob_transition = self.trans_prob(State(state=current_state), Action(action=current_action), mean_field)[next_state]
#                     sum_policy_transition += prob_transition * current_state_policy
#                 sum_next += sum_policy_transition * mean_field.val[current_state]
#             next_mean_field.val[next_state] = sum_next
#
#         total = np.sum(next_mean_field.val)
#         if total > 0:
#             next_mean_field.val /= total
#
#         return next_mean_field
#
#     def dynamics(self, state, action, mean_field=None) -> State:
#         x_position = self.state_option[state.val[0]]
#         velocity = self.action_option[action.val[0]]
#
#         if self.is_original_dynamics == 0:
#             x_next_position = (x_position + self.time_unit * velocity) % self.road_length
#         else:
#             noise_velocity = (1 - self.p) * velocity + self.p * random.choice(self.noise_option)
#             x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length
#
#         x_next_position = round(x_next_position, self.position_precision)
#
#         x_next_index = self.position_to_index.get(x_next_position)
#         if x_next_index is None:
#             x_next_index = self.find_closest_position_index(x_next_position)
#
#         return State(state=x_next_index)
#
#     def trans_prob(self, state, action, mean_field) -> np.ndarray:
#         next_prob = np.zeros(self.state_count)
#         x_position = self.state_option[state.val[0]]
#         velocity = self.action_option[action.val[0]]
#
#         if self.is_original_dynamics == 0:
#             x_next_position = (x_position + self.time_unit * velocity) % self.road_length
#         else:
#             noise_velocity = (1 - self.p) * velocity + self.p * random.choice(self.noise_option)
#             x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length
#
#         x_next_position = round(x_next_position, self.position_precision)
#
#         x_next_index = self.position_to_index.get(x_next_position)
#         if x_next_index is None:
#             x_next_index = self.find_closest_position_index(x_next_position)
#
#         next_prob[x_next_index] = 1
#
#         return next_prob
#
#     def find_closest_position_index(self, target_position):
#         differences = [abs(x - target_position) for x in self.state_option]
#         min_index = differences.index(min(differences))
#         return min_index
#
#     def get_neighbors(self, state, mean_field=None):
#         x_current = self.state_option[state]
#
#         neighbors = set()
#
#         for velocity in self.action_option:
#             if self.is_original_dynamics == 0:
#                 nx = (x_current - self.time_unit * velocity) % self.road_length
#                 nx = round(nx, self.position_precision)
#                 nx_index = self.position_to_index.get(nx)
#                 if nx_index is not None:
#                     neighbors.add(nx_index)
#                 else:
#                     nx_index = self.find_closest_position_index(nx)
#                     neighbors.add(nx_index)
#             else:
#                 for noise_velocity in self.noise_option:
#                     effective_velocity = (1 - self.p) * velocity + self.p * noise_velocity
#                     nx = (x_current - self.time_unit * effective_velocity) % self.road_length
#                     nx = round(nx, self.position_precision)
#                     nx_index = self.position_to_index.get(nx)
#                     if nx_index is not None:
#                         neighbors.add(nx_index)
#                     else:
#                         nx_index = self.find_closest_position_index(nx)
#                         neighbors.add(nx_index)
#
#         return list(neighbors)


import random
import numpy as np
from core import State, Reward, Environment, MeanField, MeanFieldFlow, PolicyFlow, Policy, Action

class Env(Environment):

    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'CARS'

        self.state_shape = 1
        self.action_shape = 1

        # Consider increasing position_unit to reduce state_count if needed
        self.position_unit = 1 / 8
        self.time_unit = self.position_unit
        self.road_length = 1
        self.velocity_max = 1
        self.total_time = 20

        self.position_precision = 5  # Precision for positions and velocities

        self.action_option = [round(v, self.position_precision) for v in np.arange(self.time_unit, self.velocity_max + self.time_unit, self.time_unit).tolist()]
        self.state_option = [round(pos, self.position_precision) for pos in np.arange(0, 1, self.position_unit).tolist()]

        self.state_count = len(self.state_option)
        self.action_count = len(self.action_option)

        # Try reducing p if needed
        self.p = 0.1
        self.noise_option = self.action_option.copy()

        self.init_mf = None
        self.dim = 1

        # Create a mapping from positions to indices
        self.position_to_index = {pos: idx for idx, pos in enumerate(self.state_option)}

        # Additional parameters for smoothing/stabilization
        self.friction_factor = 0.97  # How much to reduce effective velocity each step
        self.stable_point = 0.5      # Stable point to slightly pull positions towards
        self.stable_point_weight = 0.1  # Strength of damping towards stable point
        self.noise_range_factor = 0.1   # Factor for local noise range around current velocity
        # For mean field smoothing in advance()
        self.mf_smoothing_factor = 0.05

    def get_reward(self, state, action, mean_field):
        velocity = self.action_option[action.val[0]]
        x_position = state.val[0]

        U = 1 - mean_field.val[x_position]

        # Basic reward as before
        # Optionally, add a penalty term for large deviations to stabilize
        velocity_deviation = abs(velocity - U)
        penalty = 0.1 * velocity_deviation

        reward_val = 0.5 * ((U - velocity) ** 2) - penalty
        return Reward(reward=reward_val)

    def advance(self, policy, mean_field) -> MeanField:
        next_mean_field = MeanField(mean_field=None, s=self.state_count)
        for next_state in range(self.state_count):
            sum_next = 0
            for current_state in range(self.state_count):
                sum_policy_transition = 0
                for current_action in range(self.action_count):
                    current_state_policy = policy.val[current_state, current_action]
                    if current_state_policy == 0:
                        continue
                    prob_transition = self.trans_prob(State(state=current_state), Action(action=current_action), mean_field)[next_state]
                    sum_policy_transition += prob_transition * current_state_policy
                sum_next += sum_policy_transition * mean_field.val[current_state]
            next_mean_field.val[next_state] = sum_next

        total = np.sum(next_mean_field.val)
        if total > 0:
            next_mean_field.val /= total

        # Smooth the mean field to reduce oscillations
        # Blend it slightly with a uniform distribution
        uniform_dist = np.ones(self.state_count) / self.state_count
        next_mean_field.val = (1 - self.mf_smoothing_factor) * next_mean_field.val + self.mf_smoothing_factor * uniform_dist

        return next_mean_field

    def dynamics(self, state, action, mean_field=None) -> State:
        x_position = self.state_option[state.val[0]]
        velocity = self.action_option[action.val[0]]

        if self.is_original_dynamics == 0:
            x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        else:
            # Instead of uniform [0, velocity_max], sample noise locally around velocity
            noise_range = self.noise_range_factor * self.velocity_max
            lower_bound = max(0, velocity - noise_range)
            upper_bound = min(self.velocity_max, velocity + noise_range)
            sampled_noise = random.uniform(lower_bound, upper_bound)

            noise_velocity = (1 - self.p) * velocity + self.p * sampled_noise

            # Apply friction
            noise_velocity *= self.friction_factor

            # Update position with noise and friction
            x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length

        # Slightly pull the next position towards a stable point to reduce oscillations
        x_next_position = x_next_position * (1 - self.stable_point_weight) + self.stable_point * self.stable_point_weight

        x_next_position = round(x_next_position, self.position_precision)

        x_next_index = self.position_to_index.get(x_next_position)
        if x_next_index is None:
            x_next_index = self.find_closest_position_index(x_next_position)

        return State(state=x_next_index)

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        next_prob = np.zeros(self.state_count)
        x_position = self.state_option[state.val[0]]
        velocity = self.action_option[action.val[0]]

        if self.is_original_dynamics == 0:
            x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        else:
            noise_range = self.noise_range_factor * self.velocity_max
            lower_bound = max(0, velocity - noise_range)
            upper_bound = min(self.velocity_max, velocity + noise_range)
            sampled_noise = random.uniform(lower_bound, upper_bound)

            noise_velocity = (1 - self.p) * velocity + self.p * sampled_noise
            noise_velocity *= self.friction_factor

            x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length

            # Apply the same stable point damping in trans_prob for consistency (optional)
            x_next_position = x_next_position * (1 - self.stable_point_weight) + self.stable_point * self.stable_point_weight

        x_next_position = round(x_next_position, self.position_precision)

        x_next_index = self.position_to_index.get(x_next_position)
        if x_next_index is None:
            x_next_index = self.find_closest_position_index(x_next_position)

        next_prob[x_next_index] = 1
        return next_prob

    def find_closest_position_index(self, target_position):
        differences = [abs(x - target_position) for x in self.state_option]
        min_index = differences.index(min(differences))
        return min_index

    def get_neighbors(self, state, mean_field=None):
        x_current = self.state_option[state]

        neighbors = set()

        if self.is_original_dynamics == 0:
            for velocity in self.action_option:
                nx = (x_current - self.time_unit * velocity) % self.road_length
                nx = nx * (1 - self.stable_point_weight) + self.stable_point * self.stable_point_weight
                nx = round(nx, self.position_precision)
                nx_index = self.position_to_index.get(nx)
                if nx_index is not None:
                    neighbors.add(nx_index)
                else:
                    nx_index = self.find_closest_position_index(nx)
                    neighbors.add(nx_index)
        else:
            for velocity in self.action_option:
                # For neighbors calculation, we can keep original logic or also narrow noise range:
                noise_range = self.noise_range_factor * self.velocity_max
                lower_bound = max(0, velocity - noise_range)
                upper_bound = min(self.velocity_max, velocity + noise_range)
                # Sample a few noise values from a smaller range to find potential neighbors
                sampled_noises = [random.uniform(lower_bound, upper_bound) for _ in range(3)]
                for sampled_noise in sampled_noises:
                    effective_velocity = (1 - self.p) * velocity + self.p * sampled_noise
                    effective_velocity *= self.friction_factor
                    nx = (x_current - self.time_unit * effective_velocity) % self.road_length
                    nx = nx * (1 - self.stable_point_weight) + self.stable_point * self.stable_point_weight
                    nx = round(nx, self.position_precision)
                    nx_index = self.position_to_index.get(nx)
                    if nx_index is not None:
                        neighbors.add(nx_index)
                    else:
                        nx_index = self.find_closest_position_index(nx)
                        neighbors.add(nx_index)

        return list(neighbors)

