import numpy as np
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'ISINGNEW'

        # States and actions are spins: -1 or +1
        self.state_option = [-1, 1]
        self.action_option = [-1, 1]

        self.state_count = 2  # Number of possible states
        self.action_count = 2  # Number of possible actions

        # Variables used in the PINN
        self.time_unit = 1
        self.position_unit = 1  # Set to 1 for consistency

        self.init_mf = None
        self.dim = 1

    def get_reward(self, state, action, mean_field, h=0, lam=1):
        # Action spin value (-1 or +1)
        a_j = action.val  # Action is already a spin value

        # Mean magnetization (mean field)
        m = np.dot(mean_field.val, self.state_option)  # Average magnetization

        # Reward function: r = h * a_j + lambda * a_j * m
        reward_value = h * a_j + lam * a_j * m

        return Reward(reward=reward_value)

    def advance(self, policy, mean_field) -> MeanField:
        # Initialize the next mean field
        next_mean_field = MeanField(mean_field=None, s=self.state_count)

        # Compute the probability of transitioning to each state
        for next_state_index, next_state_spin in enumerate(self.state_option):
            total_probability = 0.0

            # Sum over all current states and actions
            for current_state_index, current_state_spin in enumerate(self.state_option):
                # Mean field probability of being in the current state
                prob_current_state = mean_field.val[current_state_index]

                # Policy probabilities for current state
                policy_probs = policy.val[current_state_index]

                # Transition probabilities
                for action_index, action_spin in enumerate(self.action_option):
                    # Policy probability of taking the action
                    prob_action = policy_probs[action_index]

                    # Transition probability from current state and action to next state
                    prob_transition = self.trans_prob(
                        State(state=current_state_spin),
                        Action(action=action_spin),
                        mean_field
                    )[next_state_index]

                    # Accumulate the total probability
                    total_probability += prob_current_state * prob_action * prob_transition

            # Set the next mean field value
            next_mean_field.val[next_state_index] = total_probability

        # Normalize the mean field values so they sum to 1
        total = np.sum(next_mean_field.val)
        if total > 0:
            next_mean_field.val /= total

        return next_mean_field

    def dynamics(self, state, action, mean_field=None) -> State:
        # In the Ising model, the next state is determined by the action (spin flip or not)
        # For simplicity, assume action represents the proposed spin value
        # However, acceptance depends on the Metropolis-Hastings criterion

        current_spin = self.state_option[state.val[0]]  # Current spin (-1 or +1)
        proposed_spin = self.action_option[action.val[0]]  # Proposed spin (-1 or +1)

        # Compute the energy difference ΔE = 2 * J * s_i * m
        # Assuming J = 1 (interaction strength)
        m = np.dot(mean_field.val, self.state_option)  # Average magnetization
        delta_E = 2 * proposed_spin * m  # ΔE = 2 * s_i * m

        # Metropolis-Hastings acceptance probability
        acceptance_prob = min(1, np.exp(-self.beta * delta_E))

        # Decide whether to accept the proposed spin
        if np.random.rand() < acceptance_prob:
            next_spin = proposed_spin  # Accept the proposed spin
        else:
            next_spin = current_spin  # Reject the proposal, stay in the current state

        return State(state=next_spin)

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        # Transition probabilities to each possible next state
        # Based on the Metropolis-Hastings acceptance probability

        next_probs = np.zeros(self.state_count)

        current_spin = self.state_option[state.val[0]]  # Current spin (-1 or +1)
        proposed_spin = self.action_option[action.val[0]]  # Proposed spin (-1 or +1)

        # Compute the energy difference ΔE = 2 * J * s_i * m
        m = np.dot(mean_field.val, self.state_option)
        delta_E = 2 * proposed_spin * m

        # Metropolis-Hastings acceptance probability
        acceptance_prob = min(1, np.exp(-self.beta * delta_E))

        # Determine the index of the proposed and current spins
        proposed_index = self.state_option.index(proposed_spin)
        current_index = self.state_option.index(current_spin)

        # Transition probabilities
        next_probs[proposed_index] = acceptance_prob
        next_probs[current_index] = 1 - acceptance_prob

        return next_probs

    def get_neighbors(self, state, mean_field=None):
        # In the Ising model, the possible next spins are -1 and +1
        return np.array(self.state_option)
