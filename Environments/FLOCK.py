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
import random

from core import State, Reward, Environment, MeanField, Action

class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta:float):
        super().__init__(is_original_dynamics,beta)
        self.name = 'FLOCK'

        # Here is the ont-hot encoding
        # we consider 5*5's points with (x,y)
        # so our state_shape should be 2
        self.state_shape = 4

        # This action is the accelerate value
        # so the action_shape is also 1
        self.action_shape = 2


        # here means that we consider the grid be 5*5
        # but we still have 4 choices for the velocity
        # So we will have 25 points with 3 types of actions
        # For our velocity's range will be about [-1,1], so we only give 3 choices for the acceleration
        self.state_count = 9 * 9
        self.action_count = 3*3

        # then we need to give the actual action and state options
        # for this state we need 25 grids, so that we give this
        self.velocity_option = np.array([[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]]) # here is all velocity choices (Here is the action )
        self.state_option = np.array(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))).T.reshape(-1, 4)
        self.action_option = np.array([[x, y] for x in [-1, 0, 1] for y in [-1, 0, 1]])

        # here is the delt_t and delt_x
        # for we have 2 dims, we give one but control 2 of them
        self.time_unit = 1
        self.position_unit = 1

        self.limit = 2

        self.dim = 2


        # here we have the demand that mean velocity
        # so that we will give the number of agent and its init state

        self.init_mf = None
        # self.init_mf[-1] = 1
        # self.init_mf = MeanField(mean_field=self.init_mf)

        self.p = 0.1
        self.noise_option = [ i for i in range(-1,self.limit)]




    def get_reward(self, state, action, mean_field):

        # Our reward is made from 3 parts
        # f value and L-2 norm for u (action) and v

        # here is the f-value's components
        # beta, N = state_count
        N = self.state_count
        beta = 0
        v_x = 0
        v_y = 0

        # Here we get the mean_v from the mean field value
        # we just need to calculate all the percentage in the mf, for this value is made from all the states
        # so that we can use the percentage from mf to get all the mean velocity
        for s in range(self.state_count):
            # get all the v and the percentage for it
            v_x += self.state_option[s][2] * mean_field.val[s]
            v_y += self.state_option[s][3] * mean_field.val[s]

        v_x_mean = v_x / self.state_count
        v_y_mean = v_y / self.state_count

        inner = np.array([-v_x_mean + self.state_option[state.val[0]][2], -v_y_mean + self.state_option[state.val[0]][3]]) * mean_field.val[state.val[0]]
        f_value = -np.linalg.norm(inner, ord=1)**2

        # here we get the action part
        action_contribution = np.linalg.norm(self.action_option[action.val[0]], ord=2) **2

        # here is the 2 direction's attribute
        velocity_contribution = np.linalg.norm([self.state_option[state.val[0]][2],self.state_option[state.val[0]][3]]
                                               , ord=2) **2

        reward = f_value - action_contribution + velocity_contribution

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
                    current_state_policy = policy.val[current_state,current_action]

                    # for we have the deterministic policy
                    # our prob is 0 or 1
                    prob_transition = self.trans_prob(State(state=current_state), Action(action=current_action),mean_field)[next_state]

                    sum_policy_transition += prob_transition*current_state_policy

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
    def dynamics(self, state, action, mean_field= None) -> State:

        # get the current position and action
        current_x_v = self.state_option[state.val[0]]
        current_action = self.action_option[action.val[0]]

        # do the transition
        next_x = current_x_v[0] + current_x_v[2] * self.time_unit
        next_y = current_x_v[1] + current_x_v[3] * self.time_unit
        if self.is_original_dynamics == 0:
            next_vx = current_x_v[2] + current_action[0] * self.time_unit
            next_vy = current_x_v[3] + current_action[1] * self.time_unit
        else:
            # here we give random choice
            # we let this random be one of the action and the probability to have any of these action is equal
            next_vx = current_x_v[2] + ( (1-self.p) * current_action[0] + self.p * random.choice(self.noise_option)) * self.time_unit
            next_vy = current_x_v[3] + ( (1-self.p) * current_action[1] + self.p * random.choice(self.noise_option)) * self.time_unit

            next_vx = self.find_closest_noise(next_vx)
            next_vy = self.find_closest_noise(next_vy)

        # find the corresponding state
        next_state = [next_x,next_y,next_vx,next_vy]

        # then we need to make sure that
        # our value do not be out of the bound
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
        current_x_v = self.state_option[state.val[0]]
        current_action = self.action_option[action.val[0]]

        # do the transition
        next_x = (current_x_v[0] + current_x_v[2] * self.time_unit)
        next_y = current_x_v[1] + current_x_v[3] * self.time_unit
        # next_vx = current_x_v[2] + current_action[0] * self.time_unit
        # next_vy = current_x_v[3] + current_action[1] * self.time_unit

        if self.is_original_dynamics == 0:
            next_vx = current_x_v[2] + current_action[0] * self.time_unit
            next_vy = current_x_v[3] + current_action[1] * self.time_unit
        else:
            # here we give random choice
            # we let this random be one of the action and the probability to have any of these action is equal

            next_vx = current_x_v[2] + (
                        (1 - self.p) * current_action[0] + self.p * random.choice(self.noise_option)) * self.time_unit
            next_vy = current_x_v[3] + (
                        (1 - self.p) * current_action[1] + self.p * random.choice(self.noise_option)) * self.time_unit

            next_vx = self.find_closest_noise(next_vx)
            next_vy = self.find_closest_noise(next_vy)

        # find the corresponding state
        next_state = [next_x, next_y, next_vx, next_vy]

        next_state = self.modulate_value(next_state)

        # Convert next_state to a numpy array for broadcasting
        next_state_np = np.array(next_state)

        # Calculate the index of the next_state in the grid
        index = np.where((self.state_option == next_state_np).all(axis=1))[0][0]

        next_prob[index] = 1

        return next_prob


    # this used to make the value do not go outof the bound
    def modulate_value(self, value):
        new_result = [0 for i in range(len(value))]
        for i in range(len(value)):
            if value[i] > 1:
                new_result[i] = -1
            elif value[i] < -1:
                new_result[i] = 1

        return new_result

    def get_neighbors(self, state, mean_field=None):
        # Convert state_index to x, y coordinates
        x, y, vx, vy = self.state_option[state]

        # Possible movements: left, right, up, down
        movements = self.velocity_option
        actions = self.action_option
        neighbors = []

        for dx, dy in movements:
            nx, ny = x - dx * self.time_unit, y - dy*self.time_unit
            for d_v_x, d_v_y in actions:
                n_v_x, n_v_y = vx - d_v_x*self.time_unit, vy - d_v_y*self.time_unit

                # Check if the new position is within grid limits
                if 0 <= nx < self.limit and 0 <= ny < self.limit:
                    # Find the index of this valid neighbor position
                    neighbor_index = np.flatnonzero((self.state_option == [nx, ny, n_v_x, n_v_y]).all(axis=1))
                    if neighbor_index.size > 0:
                        neighbors.append(neighbor_index[0])

        return np.array(neighbors)

    def find_closest_noise(self, target_value):
        # 计算与每个noise值的绝对差距
        differences = [abs(n - target_value) for n in self.noise_option]
        # 找到最小差距的索引
        min_index = differences.index(min(differences))
        # 返回最接近的值
        return self.noise_option[min_index]

