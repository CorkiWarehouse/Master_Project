import numpy as np
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):

    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'BEACH'

        # State and action representations
        self.state_shape = 1  # Each state is represented by a single position
        self.action_shape = 1  # Each action is a movement direction

        # Environment parameters
        self.state_count = 30  # Total number of states along the beach
        self.action_count = 3   # Three possible actions: move left, stay, move right

        # Define possible states and actions
        self.state_option = [i for i in range(self.state_count)]  # Positions 0 to 99
        self.action_option = [-1, 0, 1]  # Actions: left (-1), stay (0), right (+1)

        self.bar_position = 15  # Position of the bar along the beach
        self.p = 0.8  # Probability of no noise in movement
        self.horizon = 15  # Time horizon N = 15

        self.dim = 1  # Dimensionality of the state space

        self.init_mf = None  # Initial mean field (to be set externally if needed)

        self.time_unit = 1
        self.position_unit = 1

    def get_reward(self, state, action, mean_field):
        """
        Computes the reward for a given state, action, and mean field.

        Reward function:
        r(x_n, a_n, μ_n) = -|x_n - x_bar| / |X| - log(μ_n(x_n))
        """
        x_n = self.state_option[int(state.val[0])]  # Current position
        epsilon = 1e-8  # Small constant to prevent log(0)

        # Distance-based reward component
        tilde_r = abs(x_n - self.bar_position)

        action_p = -abs(action.val[0]) / self.state_count

        # Crowding penalty component
        mu_xn = mean_field.val[x_n] + epsilon
        crowding_penalty = -np.log(mu_xn)

        # Total reward
        reward_value = tilde_r + crowding_penalty + action_p

        return Reward(reward=reward_value)

    def dynamics(self, state, action, mean_field=None):
        """
        Determines the next state based on the current state, action, and stochastic noise.

        Dynamics:
        x_{n+1} = (x_n + b(x_n, a_n) + ε_n) mod |X|
        """
        x_n = self.state_option[int(state.val[0])]
        a_n = self.action_option[action.val[0]]  # Convert action index to movement (-1, 0, 1)
        b = a_n  # Drift term

        # Generate noise ε_n
        rand = np.random.rand()
        if rand < (1 - self.p) / 2:
            epsilon_n = -1
        elif rand < (1 - self.p) / 2 + self.p:
            epsilon_n = 0
        else:
            epsilon_n = 1

        # Compute next state with wrap-around
        x_next = (x_n + b + epsilon_n) % self.state_count
        next_state = State(state=int(x_next))

        return next_state

    def trans_prob(self, state, action, mean_field):
        """
        Computes the transition probability distribution over next states.

        Transition probabilities account for stochastic noise in movement.
        """
        x_n = self.state_option[int(state.val[0])]
        a_n = self.action_option[action.val[0]]
        b = a_n

        next_prob = np.zeros(self.state_count)

        # Probabilities for each possible noise value
        noise_probs = [((1 - self.p)/2, -1), (self.p, 0), ((1 - self.p)/2, 1)]

        for prob, epsilon_n in noise_probs:
            x_next = (x_n + b + epsilon_n) % self.state_count
            next_prob[int(x_next)] += prob

        return next_prob

    def advance(self, policy, mean_field):
        """
        Advances the mean field distribution to the next time step.

        Updates the distribution of players over states based on the policy and dynamics.
        """
        next_mean_field = MeanField(mean_field=None, s=self.state_count)
        next_mean_field.val = np.zeros(self.state_count)

        # Iterate over all states and actions
        for x_n in range(self.state_count):
            for a_idx in range(self.action_count):
                # Policy probability of taking action a at state x_n
                pi = policy.val[x_n, a_idx]
                if pi == 0:
                    continue

                a_n = self.action_option[a_idx]
                b = a_n

                # Probabilities for each possible noise value
                noise_probs = [((1 - self.p)/2, -1), (self.p, 0), ((1 - self.p)/2, 1)]

                for prob_epsilon, epsilon_n in noise_probs:
                    x_next = (x_n + b + epsilon_n) % self.state_count
                    prob = pi * prob_epsilon
                    next_mean_field.val[int(x_next)] += mean_field.val[x_n] * prob

        # Normalize the mean field to ensure it sums to 1
        total = np.sum(next_mean_field.val)
        if total > 0:
            next_mean_field.val /= total

        return next_mean_field

    def get_neighbors(self, state, mean_field=None):
        """
        Returns a list of states that can transition to the current state.

        Useful for algorithms that require knowledge of possible predecessors.
        """
        x_n = self.state_option[int(state)]
        neighbors = set()

        # Consider all possible actions and noise combinations
        for a_n in self.action_option:
            b = a_n
            for epsilon_n in [-1, 0, 1]:
                x_prev = (x_n - b - epsilon_n) % self.state_count
                neighbors.add(int(x_prev))

        return list(neighbors)


# import numpy as np
# from core import State, Reward, Environment, MeanField, Action
#
# class Env(Environment):
#
#     def __init__(self, is_original_dynamics: int, beta: float):
#         super().__init__(is_original_dynamics, beta)
#         self.name = 'BEACH'
#
#         # State and action representations
#         self.state_shape = 1  # Each state is represented by a single position
#         self.action_shape = 1  # Each action is a movement direction
#
#         # Environment parameters
#         self.state_count = 100  # Total number of states along the beach
#         self.action_count = 3   # Three possible actions: move left, stay, move right
#
#         # Define possible states and actions
#         self.state_option = [i for i in range(self.state_count)]  # Positions 0 to 99
#         self.action_option = [-1, 0, 1]  # Actions: left (-1), stay (0), right (+1)
#
#         self.bar_position = 50  # Position of the bar along the beach
#         self.p = 0.8  # Probability of no noise in movement
#         self.horizon = 15  # Time horizon N = 15
#
#         self.dim = 1  # Dimensionality of the state space
#
#         self.init_mf = None  # Initial mean field (to be set externally if needed)
#
#         self.time_unit = 1
#         self.position_unit = 1
#
#     def get_reward(self, state, action, mean_field):
#         """
#         Computes the reward for a given state, action, and mean field.
#
#         Reward function:
#         r(x_n, a_n, μ_n) = -|x_n - x_bar| / |X| - log(μ_n(x_n))
#         """
#         x_n = self.state_option[int(state.val[0])]  # Current position
#
#         # Ensure action index is valid
#         a_idx = int(action.val[0])
#         if a_idx < 0 or a_idx >= self.action_count:
#             raise ValueError(f"Invalid action index: {a_idx}")
#
#         a_n = self.action_option[a_idx]  # Convert action index to movement (-1, 0, 1)
#
#         # Distance-based reward component
#         distance = abs(x_n - self.bar_position)
#         max_distance = self.state_count - 1
#         tilde_r = -distance / max_distance  # Normalize to [-1, 0]
#
#         # Action penalty (optional)
#         action_p = -abs(a_n) / max(abs(self.action_option))  # Normalize to [-1, 0]
#
#         # Crowding penalty component
#         epsilon = 1e-3  # Larger epsilon to prevent extremely large penalties
#         mu_xn = mean_field.val[x_n] + epsilon
#         mu_xn = min(mu_xn, 1.0)  # Ensure mu_xn does not exceed 1
#         crowding_penalty = -np.log(mu_xn)
#
#         # Limit the crowding penalty to a maximum value to prevent numerical issues
#         max_crowding_penalty = 10.0
#         crowding_penalty = min(crowding_penalty, max_crowding_penalty)
#
#         # Total reward
#         reward_value = tilde_r + crowding_penalty + action_p
#
#         return Reward(reward=reward_value)
#
#     def dynamics(self, state, action, mean_field=None):
#         """
#         Determines the next state based on the current state, action, and stochastic noise.
#
#         Dynamics:
#         x_{n+1} = (x_n + b(x_n, a_n) + ε_n) mod |X|
#         """
#         x_n = self.state_option[int(state.val[0])]
#
#         # Ensure action index is valid
#         a_idx = int(action.val[0])
#         if a_idx < 0 or a_idx >= self.action_count:
#             raise ValueError(f"Invalid action index: {a_idx}")
#
#         a_n = self.action_option[a_idx]  # Convert action index to movement (-1, 0, 1)
#         b = a_n  # Drift term
#
#         # Generate noise ε_n
#         rand = np.random.rand()
#         if rand < (1 - self.p) / 2:
#             epsilon_n = -1
#         elif rand < (1 - self.p) / 2 + self.p:
#             epsilon_n = 0
#         else:
#             epsilon_n = 1
#
#         # Compute next state with wrap-around
#         x_next = (x_n + b + epsilon_n) % self.state_count
#         next_state = State(state=int(x_next))
#
#         return next_state
#
#     def trans_prob(self, state, action, mean_field):
#         """
#         Computes the transition probability distribution over next states.
#
#         Transition probabilities account for stochastic noise in movement.
#         """
#         x_n = self.state_option[int(state.val[0])]
#
#         # Ensure action index is valid
#         a_idx = int(action.val[0])
#         if a_idx < 0 or a_idx >= self.action_count:
#             raise ValueError(f"Invalid action index: {a_idx}")
#
#         a_n = self.action_option[a_idx]
#         b = a_n
#
#         next_prob = np.zeros(self.state_count)
#
#         # Probabilities for each possible noise value
#         noise_probs = [((1 - self.p)/2, -1), (self.p, 0), ((1 - self.p)/2, 1)]
#
#         for prob, epsilon_n in noise_probs:
#             x_next = (x_n + b + epsilon_n) % self.state_count
#             next_prob[int(x_next)] += prob
#
#         # Normalize to ensure it sums to 1
#         total_prob = np.sum(next_prob)
#         if total_prob > 0:
#             next_prob /= total_prob
#
#         return next_prob
#
#     def advance(self, policy, mean_field):
#         """
#         Advances the mean field distribution to the next time step.
#
#         Updates the distribution of players over states based on the policy and dynamics.
#         """
#         next_mean_field = MeanField(mean_field=None, s=self.state_count)
#         next_mean_field.val = np.zeros(self.state_count)
#
#         # Iterate over all states and actions
#         for x_n in range(self.state_count):
#             for a_idx in range(self.action_count):
#                 # Policy probability of taking action a at state x_n
#                 pi = policy.val[x_n, a_idx]
#                 if pi == 0:
#                     continue
#
#                 a_n = self.action_option[a_idx]
#                 b = a_n
#
#                 # Probabilities for each possible noise value
#                 noise_probs = [((1 - self.p)/2, -1), (self.p, 0), ((1 - self.p)/2, 1)]
#
#                 for prob_epsilon, epsilon_n in noise_probs:
#                     x_next = (x_n + b + epsilon_n) % self.state_count
#                     prob = pi * prob_epsilon * mean_field.val[x_n]
#                     next_mean_field.val[int(x_next)] += prob
#
#         # Normalize the mean field to ensure it sums to 1
#         total = np.sum(next_mean_field.val)
#         if total > 0:
#             next_mean_field.val /= total
#         else:
#             # If total is zero, reset to uniform distribution
#             next_mean_field.val = np.ones(self.state_count) / self.state_count
#
#         return next_mean_field
#
#     def get_neighbors(self, state, mean_field=None):
#         """
#         Returns a list of states that can transition to the current state.
#
#         Useful for algorithms that require knowledge of possible predecessors.
#         """
#         x_n = self.state_option[int(state)]
#         neighbors = set()
#
#         # Consider all possible actions and noise combinations
#         for a_n in self.action_option:
#             b = a_n
#             for epsilon_n in [-1, 0, 1]:
#                 x_prev = (x_n - b - epsilon_n) % self.state_count
#                 neighbors.add(int(x_prev))
#
#         return list(neighbors)
