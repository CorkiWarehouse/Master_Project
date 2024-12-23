import math
import numpy as np
import random

from core import State, Reward, Environment, MeanField, Action


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'FLOCK'

        # Define positions and velocities including -1, 0, 1
        positions = [-1, 0, 1]
        velocities = [-1, 0, 1]

        # State and action shapes
        self.state_shape = 4
        self.action_shape = 2

        # Create state options (all combinations of positions and velocities)
        self.state_option = np.array(
            np.meshgrid(positions, positions, velocities, velocities)
        ).T.reshape(-1, 4)
        self.state_count = len(self.state_option)

        # Create action options (accelerations)
        self.action_option = np.array(
            [[x, y] for x in [-1,0, 1] for y in [-1,0, 1]]
        )
        self.action_count = len(self.action_option)

        # Map from state tuple to index for quick lookup
        self.state_to_index = {}
        for idx, state in enumerate(self.state_option):
            # Convert state elements to integers to avoid floating-point issues
            state_tuple = tuple(int(val) for val in state)
            self.state_to_index[state_tuple] = idx

        # Other attributes
        self.velocity_option = self.action_option  # Update if necessary
        self.noise_option = [-1,1]

        self.time_unit = 1
        self.position_unit = math.sqrt(2)
        self.limit = 1  # Since positions and velocities range from -1 to 1
        self.dim = 2

        self.init_mf = None
        self.p = 0.1

    def get_reward(self, state, action, mean_field):
        # Reward is composed of three parts
        # f_value, action penalty, and velocity contribution

        v_x = 0
        v_y = 0

        # # Compute the mean velocity from the mean field
        # for s in range(self.state_count):
        #     v_x += self.state_option[s][2] * mean_field.val[s]
        #     v_y += self.state_option[s][3] * mean_field.val[s]

        # Compute the mean velocity from the mean field using matrix multiplication
        v_x = np.dot(self.state_option[:, 2], mean_field.val)
        v_y = np.dot(self.state_option[:, 3], mean_field.val)

        # Mean velocities (mean_field.val sums to 1)
        v_x_mean = v_x
        v_y_mean = v_y

        # Compute f_value component
        current_velocity = self.state_option[state.val[0]][2:4]
        inner = current_velocity - np.array([v_x_mean, v_y_mean])
        f_value = -np.linalg.norm(inner, ord=2) ** 2

        # Action penalty
        action_contribution = np.linalg.norm(self.action_option[action.val[0]], ord=2) ** 2

        # Velocity contribution
        velocity_contribution = np.linalg.norm(current_velocity, ord=2) ** 2

        reward = f_value - action_contribution + velocity_contribution

        return Reward(reward=reward)

    def advance(self, policy, mean_field) -> MeanField:
        next_mean_field = MeanField(mean_field=None, s=self.state_count)

        # Iterate over all possible next states
        for next_state_idx in range(self.state_count):
            sum_next = 0.0

            # For all current states
            for current_state_idx in range(self.state_count):
                sum_policy_transition = 0.0

                # For all actions
                for current_action_idx in range(self.action_count):
                    current_state_policy = policy.val[current_state_idx, current_action_idx]

                    # Skip if the policy probability is zero
                    if current_state_policy == 0:
                        continue

                    # Get transition probability
                    prob_transition = self.trans_prob_idx(
                        current_state_idx, current_action_idx, mean_field
                    )[next_state_idx]

                    sum_policy_transition += prob_transition * current_state_policy

                sum_next += sum_policy_transition * mean_field.val[current_state_idx]

            next_mean_field.val[next_state_idx] = sum_next

        # Normalize the mean field values so they sum to 1
        total = np.sum(next_mean_field.val)
        if total > 0:
            next_mean_field.val /= total
        else:
            # Handle zero total by setting a uniform distribution
            next_mean_field.val = np.ones(self.state_count) / self.state_count

        return next_mean_field

    def dynamics(self, state, action, mean_field=None) -> State:
        # Get the current state and action
        current_x_v = self.state_option[state.val[0]]
        current_action = self.action_option[action.val[0]]

        # Compute the next position and velocity
        next_x = current_x_v[0] + current_x_v[2] * self.time_unit
        next_y = current_x_v[1] + current_x_v[3] * self.time_unit

        if self.is_original_dynamics == 0:
            next_vx = current_x_v[2] + current_action[0] * self.time_unit
            next_vy = current_x_v[3] + current_action[1] * self.time_unit
        else:
            # Add random noise to velocity
            next_vx = current_x_v[2] + (
                (1 - self.p) * current_action[0] + self.p * random.choice(self.noise_option)
            ) * self.time_unit
            next_vy = current_x_v[3] + (
                (1 - self.p) * current_action[1] + self.p * random.choice(self.noise_option)
            ) * self.time_unit

            # Find the closest allowed velocity
            next_vx = self.find_closest_noise(next_vx)
            next_vy = self.find_closest_noise(next_vy)

        # Modulate values to stay within allowed range and handle wrapping
        next_state = [next_x, next_y, next_vx, next_vy]
        next_state = self.modulate_value(next_state)

        # Find the index of the next state
        next_state_idx = self.get_state_index(next_state)

        if next_state_idx is not None:
            return State(state=next_state_idx)
        else:
            # Handle invalid next state
            raise ValueError(f"Invalid next state: {next_state}")

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        return self.trans_prob_idx(state.val[0], action.val[0], mean_field)

    def trans_prob_idx(self, state_idx, action_idx, mean_field) -> np.ndarray:
        next_prob = np.zeros(self.state_count)

        # Get the current state and action
        current_x_v = self.state_option[state_idx]
        current_action = self.action_option[action_idx]

        # Compute the next position and velocity
        next_x = current_x_v[0] + current_x_v[2] * self.time_unit
        next_y = current_x_v[1] + current_x_v[3] * self.time_unit

        if self.is_original_dynamics == 0:
            next_vx = current_x_v[2] + current_action[0] * self.time_unit
            next_vy = current_x_v[3] + current_action[1] * self.time_unit
        else:
            # Add random noise to velocity
            next_vx = current_x_v[2] + (
                (1 - self.p) * current_action[0] + self.p * random.choice(self.noise_option)
            ) * self.time_unit
            next_vy = current_x_v[3] + (
                (1 - self.p) * current_action[1] + self.p * random.choice(self.noise_option)
            ) * self.time_unit

            # Find the closest allowed velocity
            next_vx = self.find_closest_noise(next_vx)
            next_vy = self.find_closest_noise(next_vy)

        # Modulate values to stay within allowed range and handle wrapping
        next_state = [next_x, next_y, next_vx, next_vy]
        next_state = self.modulate_value(next_state)

        # Find the index of the next state
        next_state_idx = self.get_state_index(next_state)

        if next_state_idx is not None:
            next_prob[next_state_idx] = 1.0
        else:
            # Handle invalid next state
            pass  # You can choose to raise an exception or assign a default value

        return next_prob

    def modulate_value(self, value):
        new_result = []
        for val in value:
            # Handle wrapping around
            if val > self.limit:
                new_val = -self.limit
            elif val < -self.limit:
                new_val = self.limit
            else:
                new_val = val

            # Map the value to the closest allowed value
            new_val = self.find_closest_allowed_value(new_val)
            new_result.append(new_val)
        return new_result

    def find_closest_allowed_value(self, val):
        allowed_values = [-1, 0, 1]
        differences = [abs(av - val) for av in allowed_values]
        min_index = differences.index(min(differences))
        return allowed_values[min_index]

    def find_closest_noise(self, target_value):
        differences = [abs(n - target_value) for n in self.noise_option]
        min_index = differences.index(min(differences))
        return self.noise_option[min_index]

    def get_state_index(self, state):
        # Convert state elements to integers to avoid floating-point issues
        state_tuple = tuple(int(val) for val in state)
        return self.state_to_index.get(state_tuple, None)

    def wrap_value(self, val):
        # Handles wrapping around the limits
        if val > self.limit:
            return -self.limit
        elif val < -self.limit:
            return self.limit
        else:
            return val

    def get_neighbors(self, state, mean_field=None):
        # Get the current state
        x, y, vx, vy = self.state_option[state]

        neighbors = []

        # Possible movements (velocity options)
        movements = self.velocity_option
        actions = self.action_option

        for dx, dy in movements:
            # Compute the neighbor's position
            nx = x - dx * self.time_unit
            ny = y - dy * self.time_unit

            # Wrap around positions
            nx = self.wrap_value(nx)
            ny = self.wrap_value(ny)

            for dvx, dvy in actions:
                # Compute the neighbor's velocity
                n_vx = vx - dvx * self.time_unit
                n_vy = vy - dvy * self.time_unit

                # Wrap around velocities
                n_vx = self.wrap_value(n_vx)
                n_vy = self.wrap_value(n_vy)

                # Modulate values to allowed grid points
                nx_mod, ny_mod = self.modulate_value([nx, ny])
                n_vx_mod, n_vy_mod = self.modulate_value([n_vx, n_vy])

                # Find the neighbor's index
                neighbor_idx = self.get_state_index([nx_mod, ny_mod, n_vx_mod, n_vy_mod])
                if neighbor_idx is not None:
                    neighbors.append(neighbor_idx)

        return np.array(neighbors)
