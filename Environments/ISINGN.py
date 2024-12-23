# import numpy as np
# from core import State, Reward, Environment, MeanField, Action
#
# class Env(Environment):
#     def __init__(self, is_original_dynamics: int, beta: float):
#         super().__init__(is_original_dynamics, beta)
#         self.name = 'ISINGN'
#
#         # State and action options: indices 0 and 1 correspond to spins -1 and +1
#         self.state_option = [-1, 1]  # Spin values
#         self.action_option = [-1, 1]  # Possible actions (spin choices)
#
#         self.state_shape = 1
#         self.action_shape = 1
#
#         self.state_count = 2  # Number of possible states
#         self.action_count = 2  # Number of possible actions
#
#         # Mean field variable
#         self.init_mf = None
#         self.dim = 1  # Dimension of the state space
#
#         # Environment parameters
#         self.h = 0  # External magnetic field
#         self.lam = 1  # Interaction term (lambda)
#         self.temperature = 1 / beta  # System temperature
#
#         # PINN units (if needed)
#         self.time_unit = 1
#         self.position_unit = 2
#
#     def get_reward(self, state, action, mean_field):
#         """
#         Calculate the reward for taking a certain action in a given state,
#         considering the mean field (average magnetization).
#         """
#         # Get the spin value corresponding to the action
#         a_j = self.action_option[int(action.val[0])]  # Action spin (-1 or +1)
#
#         # Calculate the average magnetization (mean field)
#         m = np.dot(mean_field.val, self.state_option)  # Average magnetization
#
#         # Reward function: r = h * a_j + lambda * a_j * m
#         reward = self.h * a_j + self.lam * a_j * m
#
#         return Reward(reward=reward)
#
#     def dynamics(self, state, action, mean_field=None) -> State:
#         """
#         State transitions depend on the dynamics type.
#         When is_original_dynamics == 0:
#             Deterministic dynamics: the next state is directly determined by the action.
#         When is_original_dynamics != 0:
#             MCMC dynamics: the next state is determined based on the Metropolis criterion.
#         """
#         if self.is_original_dynamics == 0:
#             # Deterministic dynamics
#             next_state_index = int(action.val[0])  # Index 0 or 1
#             next_state = State(state=next_state_index)  # Next state
#             return next_state
#         else:
#             # MCMC dynamics
#             # Current spin
#             a_j = self.state_option[int(state.val[0])]  # Current spin (-1 or +1)
#             # Proposed spin from action
#             a_j_proposed = self.action_option[int(action.val[0])]  # Proposed spin (-1 or +1)
#
#             # Calculate average magnetization (mean field)
#             m = np.dot(mean_field.val, self.state_option)  # Average magnetization
#
#             # Compute energies
#             E_current = - (self.h * a_j + self.lam * a_j * m)
#             E_proposed = - (self.h * a_j_proposed + self.lam * a_j_proposed * m)
#
#             # Compute acceptance probability
#             delta_E = E_proposed - E_current
#             acceptance_prob = np.exp(-delta_E / self.temperature)
#
#             # Generate a random number for acceptance check
#             epsilon = np.random.uniform(0, 1)
#             if acceptance_prob > epsilon:
#                 # Accept the proposed spin
#                 next_spin = a_j_proposed
#             else:
#                 # Keep the current spin
#                 next_spin = a_j
#
#             # Map next_spin back to state index
#             next_state_index = self.state_option.index(next_spin)
#             next_state = State(state=next_state_index)
#             return next_state
#
#     def advance(self, policy, mean_field) -> MeanField:
#         """
#         Update the mean field based on the current policy and mean field.
#         This function computes the expected value of the agent's spin under the current policy and mean field.
#         """
#         # Initialize the next mean field
#         next_mean_field = MeanField(mean_field=None, s=self.state_count)
#         next_mean_field.val = np.zeros(self.state_count)
#
#         # For each possible state
#         for s in range(self.state_count):
#             # For each possible action
#             for a in range(self.action_count):
#                 # Use the policy (policy.val[s, a]) and mean field to calculate the probability of taking action 'a' in state 's'
#                 action_prob = policy.val[s, a]
#
#                 # Compute the transition probabilities
#                 trans_probs = self.trans_prob(State(state=s), Action(action=a), mean_field)
#
#                 # Update the mean field for each possible next state
#                 for next_state_index in range(self.state_count):
#                     next_mean_field.val[next_state_index] += mean_field.val[s] * action_prob * trans_probs[next_state_index]
#
#         # Normalize the mean field to ensure it sums to 1
#         total = np.sum(next_mean_field.val)
#         if total > 0:
#             next_mean_field.val /= total
#         else:
#             # Handle the case where total probability is zero
#             next_mean_field.val = np.ones(self.state_count) / self.state_count
#
#         return next_mean_field
#
#     def get_neighbors(self, state, mean_field=None):
#         """
#         In the mean field game version, individual neighbors are not considered.
#         The interaction is with the overall mean field.
#         This method can return all possible states.
#         """
#         if self.state_option[state] == 1:
#             return np.array([0, 1])
#         else:
#             return np.array([1,0])
#
#     def trans_prob(self, state, action, mean_field) -> np.ndarray:
#         """
#         Transition probabilities depend on the dynamics type.
#         When is_original_dynamics == 0:
#             Deterministic transition: probability 1 for the state determined by action.
#         When is_original_dynamics != 0:
#             Transition probabilities according to MCMC acceptance probabilities.
#         """
#         next_prob = np.zeros(self.state_count)
#
#         if self.is_original_dynamics == 0:
#             next_state_index = int(action.val[0])  # Action determines the state
#             next_prob[next_state_index] = 1
#             return next_prob
#         else:
#             # MCMC dynamics
#             # Current spin
#             a_j = self.state_option[int(state.val[0])]  # Current spin (-1 or +1)
#             # Proposed spin from action
#             a_j_proposed = self.action_option[int(action.val[0])]  # Proposed spin (-1 or +1)
#
#             # Calculate average magnetization (mean field)
#             m = np.dot(mean_field.val, self.state_option)  # Average magnetization
#
#             # Compute energies
#             E_current = - (self.h * a_j + self.lam * a_j * m)
#             E_proposed = - (self.h * a_j_proposed + self.lam * a_j_proposed * m)
#
#             delta_E = E_proposed - E_current
#             acceptance_prob = np.exp(-delta_E / self.temperature)
#
#             # Map spins back to state indices
#             current_state_index = self.state_option.index(a_j)
#             proposed_state_index = self.state_option.index(a_j_proposed)
#
#             # Transition probabilities
#             next_prob[proposed_state_index] = acceptance_prob
#             next_prob[current_state_index] = 1 - acceptance_prob
#
#             return next_prob


import numpy as np
import random
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float, seed=None):
        """
        :param is_original_dynamics: 0 => deterministic, !=0 => Glauber (logistic) dynamics
        :param beta: 1 / temperature
        :param seed: optional random seed for reproducibility
        """
        super().__init__(is_original_dynamics, beta)
        self.name = 'ISINGN'

        # State and action: 2 states => spin = -1 or +1
        self.state_option = [-1, 1]
        # Action: similarly choose spin flip proposals from [-1, +1]
        self.action_option = [-1, 1]

        self.state_shape = 1
        self.action_shape = 1

        self.state_count = 2
        self.action_count = 2

        # Additional environment parameters
        self.init_mf = None
        self.dim = 1
        self.h = 0         # external magnetic field
        self.lam = 1       # coupling constant
        self.temperature = 1 / beta  # used in logistic
        if seed is not None:
            seed = 1
            random.seed(seed)
            np.random.seed(seed)

        # Possibly used for PDE/PI methods
        self.time_unit = 1
        self.position_unit = 2

    def get_reward(self, state, action, mean_field):
        """
        Reward = h * a_j + lam * a_j * m,
        where a_j is the chosen spin (i.e. action),
        and m = sum_{s} mean_field[s]* self.state_option[s].
        """
        a_j = self.action_option[int(action.val[0])]
        m = np.dot(mean_field.val, self.state_option)

        reward = self.h * a_j + self.lam * a_j * m
        return Reward(reward=reward)

    def dynamics(self, state, action, mean_field=None) -> State:
        """
        If is_original_dynamics==0 => Deterministic (as in your original code).
        Otherwise => use Glauber:
          - We interpret 'action' as 'propose flipping to spin = action_option[int(action.val[0])]'.
          - Then we do a logistic-like acceptance for the next spin.
        """
        if self.is_original_dynamics == 0:
            # Deterministic from your original code
            next_state_index = int(action.val[0])  # 0 or 1
            return State(state=next_state_index)
        else:
            # GLAUBER DYNAMICS
            return self._sample_next_glauber(state, action, mean_field)

    def _sample_next_glauber(self, state, action, mean_field):
        """
        Single-spin-flip with Glauber (logistic) acceptance:
           p(flip to proposed) = 1 / (1 + exp(+ beta * Delta E))
        Here we define Delta E based on flipping from current_spin -> proposed_spin.
        """
        current_spin = self.state_option[int(state.val[0])]   # -1 or +1
        proposed_spin = self.action_option[int(action.val[0])] # -1 or +1

        # Let's define local field = h + lam*m
        # Then E(current_spin) = - current_spin * local_field
        local_field = self.h + self.lam * np.dot(mean_field.val, self.state_option)

        # For the 2-state case:
        # E_current = - (current_spin * local_field)
        # E_proposed = - (proposed_spin * local_field)
        E_current = -(current_spin * local_field)
        E_proposed = -(proposed_spin * local_field)
        deltaE = E_proposed - E_current  # how energy changes if we flip to proposed_spin

        # Glauber acceptance: p_flip = 1 / [1 + exp(+ beta * deltaE)]
        p_flip = 1.0 / (1.0 + np.exp(self.beta * deltaE))

        # Sample
        eps = np.random.rand()
        if eps < p_flip:
            next_spin = proposed_spin
        else:
            next_spin = current_spin

        next_state_index = self.state_option.index(next_spin)
        return State(state=next_state_index)

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        """
        Return the probability distribution [p(s_next = 0), p(s_next = 1)].
        If is_original_dynamics==0 => deterministic, next = action.
        Otherwise => same Glauber logic as dynamics:
          p_flip = 1 / [1 + exp(+ beta * deltaE)] for flipping to proposed_spin,
          else remain current.
        """
        next_prob = np.zeros(self.state_count)

        if self.is_original_dynamics == 0:
            # Deterministic
            next_state_index = int(action.val[0])
            next_prob[next_state_index] = 1.0
        else:
            # GLAUBER distribution
            current_spin = self.state_option[int(state.val[0])]
            proposed_spin = self.action_option[int(action.val[0])]

            # local field
            local_field = self.h + self.lam * np.dot(mean_field.val, self.state_option)

            E_current = -(current_spin * local_field)
            E_proposed = -(proposed_spin * local_field)
            deltaE = E_proposed - E_current

            p_flip = 1.0 / (1.0 + np.exp(self.beta * deltaE))
            # if flip => next_spin= proposed_spin
            # else => next_spin= current_spin
            current_idx = self.state_option.index(current_spin)
            proposed_idx = self.state_option.index(proposed_spin)

            next_prob[proposed_idx] = p_flip
            next_prob[current_idx] = 1.0 - p_flip

        return next_prob

    def advance(self, policy, mean_field) -> MeanField:
        """
        next_mean_field[s'] = sum_{s,a} mean_field[s]* policy[s,a] * trans_prob(s-> s')
        """
        next_mean_field = MeanField(mean_field=None, s=self.state_count)
        next_mean_field.val = np.zeros(self.state_count)

        for s in range(self.state_count):
            for a in range(self.action_count):
                prob_a = policy.val[s,a]
                tprob = self.trans_prob(State(state=s), Action(action=a), mean_field)
                for nxt in range(self.state_count):
                    next_mean_field.val[nxt] += mean_field.val[s]* prob_a* tprob[nxt]

        # normalize
        total = np.sum(next_mean_field.val)
        if total > 1e-12:
            next_mean_field.val /= total
        else:
            next_mean_field.val = np.ones(self.state_count)/ self.state_count

        return next_mean_field

    def get_neighbors(self, state, mean_field=None):
        """
        In the mean field game version, individual neighbors are not considered.
        The interaction is with the overall mean field.
        This method can return all possible states.
        """
        if self.state_option[state] == 1:
            return np.array([0, 1])
        else:
            return np.array([1,0])
