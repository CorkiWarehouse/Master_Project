"""
(Although we have the random part in FLOCK's v, but we do not
    consider it in our environment )
This is the Flock environment, we have :
    1. Agent : i
    2. Position: x, x^i means that Agent_i is at x position
    3. Velocity: v, v^i means that Agent_i with velocity v
        (Position with velocity will be the state)
    4. Acceleration (Action):
        u, u^i means that Agent_i with acceleration u
    5. Interactions (mean field):
        mu_t^N = 1/N(sum( empirical distribution of (x_t,v_t)))
    6. Flocking criterion (Reward): at t time, Agent_i's reward
        f_i^t = f(x_t^i,v_t^i,u_t^i,mu_t^i)
            where f() is a function


We assume that we only have the 4 v = {(-1,-1),(-1,1),(1,-1),(1,1)}
    如果初始速度是服从  N(0,1) 分布的，即均值为0，
    标准差为1，则群集算法（Flock'n RL）倾向于随机选择四个速度均衡之一

And then we assume that our x will be [-5,5] all the discrete value

According to the one-hot encoding
So that our state_shape will be 5*5*4 = 100
    For we have 25 x position and 4 velocity choice
    So we have 100 total
"""

'''
Here is a simple explanation of the mean field

Interactions (mean field):
    Empirical Distribution (\( \mu_t^N \)):
        To compute the empirical distribution \( \mu_t^5 \) for 
            our example, we use the positions and 
            velocities of the birds at time \( t \):

\[ \mu_t^5 = 
    \frac{1}{5} 
    (\delta(1, 2) + \delta(2, 2) + \delta(1, 3) + \delta(2, 1) + \delta(1, 2)) \]

This expression means that if we wanted to know the 
    probability density of finding a bird with position 1 and 
    velocity 2 at time \( t \), we would see that 
    there are 2 birds (Bird 1 and Bird 5) with 
    these exact characteristics out of the 5 birds total, 
    resulting in a density of \( \frac{2}{5} \) for the 
    state (1,2).
    
'''

'''

'''


import numpy as np

from core import State, Reward, Environment, MeanField

class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta:float):
        super().__init__(is_original_dynamics,beta)
        self.name = 'FLOCK'

        # Here is the ont-hot encoding
        # we consider 5*5's points
        self.state_shape = 25

        # for we only allow 4 types actions, We only have action_shape = 4
        self.action_shape = 4



    def get_reward(self, state, action,mean_field):

        # we use the decode_state method to get the position and velocity
        position, velocity = self.decode_state(state)

        # then we use the reward function in the article
        current_reward =None
        return None

    '''
    This method is to change One-Hot to the original grid
    Like we have state_24 = all zero but index 23 is 1
    
    So we can have this to get our original 
    '''
    def decode_state(self, state, grid_size=(5,5), velocity_options=[(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        """
        Decode the one-hot encoded state vector to the corresponding position and velocity.

        :param state_vector: List, one-hot encoded state vector.
        :param grid_size: Tuple, the size of the grid (m, n).
        :param velocity_options: List of tuples, possible velocity vectors.
        :return: Tuple, the decoded position (x, y) and velocity (vx, vy).
        """
        assert len(velocity_options) * grid_size[0] * grid_size[1] == len(
            state), "State vector length does not match the grid and velocity options."

        # Find the index of '1' in the state vector
        state_index = state.index(1)

        # The number of states per grid cell is the number of velocity options
        states_per_position = len(velocity_options)

        # Decode the position and velocity
        position_index = state_index // states_per_position
        velocity_index = state_index % states_per_position

        # Convert position index to 2D grid coordinates
        x = position_index // grid_size[1] + 1  # +1 because positions are 1-indexed
        y = position_index % grid_size[1] + 1  # +1 because positions are 1-indexed

        # Get the velocity
        vx, vy = velocity_options[velocity_index]

        return (x, y), (vx, vy)


    '''
    This is the function to get the f_pay_off
    '''
    def calculate_flocking_payoff(self, positions, velocities, i, beta=0):
        """
        Calculate the flocking reward for agent i.

        :param positions: A 2D array of shape (N, d) containing the positions of all agents.
        :param velocities: A 2D array of shape (N, d) containing the velocities of all agents.
        :param i: The index of the agent for which to calculate the reward.
        :param beta: The exponent beta in the flocking criterion.
        :return: The flocking reward for agent i.
        """
        N, d = positions.shape
        assert velocities.shape == (N, d), "Positions and velocities must have the same shape."

        velocity_diffs = velocities - velocities[i]
        position_diffs = positions - positions[i]

        # Calculate the norm squared of the velocity differences
        velocity_diffs_norm_sq = np.sum(velocity_diffs ** 2, axis=1)

        # Calculate the weighted norm squared of the position differences
        position_diffs_weighted_norm = (1 + np.sum(position_diffs ** 2, axis=1)) ** beta

        # Sum up the flocking criteria for all agents
        flocking_sum = np.sum(velocity_diffs_norm_sq / position_diffs_weighted_norm)

        # Calculate the flocking reward
        flocking_reward = -1 / N * flocking_sum

        return flocking_reward


# # Example usage:
# grid_size = (5, 5)  # Assuming a 5x5 grid
# velocity_options = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Possible velocity vectors
#
# # Let's say you have a one-hot encoded state with '1' at index 24 (25th state)
# state_n = [0] * 100
# state_n[24] = 1
#
# # Decode the state to get position and velocity
# position, velocity = decode_state(state_n, grid_size, velocity_options)
# print("Position:", position)
# print("Velocity:", velocity)

def calculate_flocking_reward(positions, velocities, i, beta):
    """
    Calculate the flocking reward for agent i.

    :param positions: A 2D array of shape (N, d) containing the positions of all agents.
    :param velocities: A 2D array of shape (N, d) containing the velocities of all agents.
    :param i: The index of the agent for which to calculate the reward.
    :param beta: The exponent beta in the flocking criterion.
    :return: The flocking reward for agent i.
    """
    N, d = positions.shape
    assert velocities.shape == (N, d), "Positions and velocities must have the same shape."

    velocity_diffs = velocities - velocities[i]
    position_diffs = positions - positions[i]

    # Calculate the norm squared of the velocity differences
    velocity_diffs_norm_sq = np.sum(velocity_diffs ** 2, axis=1)

    # Calculate the weighted norm squared of the position differences
    position_diffs_weighted_norm = (1 + np.sum(position_diffs ** 2, axis=1)) ** beta

    # Sum up the flocking criteria for all agents
    flocking_sum = np.sum(velocity_diffs_norm_sq / position_diffs_weighted_norm)

    # Calculate the flocking reward
    flocking_reward = -1 / N * flocking_sum

    return flocking_reward


# Example usage:
# Assuming you have the positions and velocities of N agents stored in numpy arrays
N = 10  # Number of agents
d = 2  # Dimension of space

# For simplicity, let's create random positions and velocities
np.random.seed(0)  # Seed for reproducibility
positions = np.random.rand(N, d)
velocities = np.random.rand(N, d)

# Choose the agent index and beta value
agent_index = 0  # Index for agent i
beta_value = 1.5  # Exponent beta

# Get the flocking reward for agent i
reward = calculate_flocking_reward(positions, velocities, agent_index, beta_value)
print(f"Flocking reward for agent {agent_index}: {reward}")