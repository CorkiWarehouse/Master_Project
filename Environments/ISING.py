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

from core import State, Reward, Environment, MeanField,Action


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'ISLING'

        # Here is the ont-hot encoding
        # we consider 5*5's points

        self.state_shape = 25

        # for we only allow [-1,1] to be our
        self.action_shape = 2

    def get_reward(self, state, action, mean_field):
        h = mean_field.val[state.val[0]]

        left = h * action.val[0]

        lamed = 1

        sum = 0
        for i in range(self.state_shape):


        return None

    def advance(self, policy, mean_field) -> MeanField:
        # init
        next_mean_field = MeanField(mean_field=None, s=self.state_shape)

        # here is all the next_state
        for next_state in range(self.state_shape):
            test_set = []
            sum_next = 0
            # here is all the current state which can reach to next one
            # but for our time_unit is 5
            # not all the action that can reach this 'next_state'
            for current_state in range(self.state_shape):

                sum_policy_transition = 0

                # we directly check all the valid actions
                for current_action in range(self.action_shape):
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

    '''
    In article it is deterministic, so we do not have the probability 

        if we have the state and action, we will get the next state
        and this state is not relevant with mean field 
    '''

    def dynamics(self, state, action, mean_field=None) -> State:

        # get the x
        x_position = self.state_option[state.val[0]]

        # get the v
        velocity = self.velocity_option[action.val[0]]

        # get the next time and position
        # if it out of range
        x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        x_next_index = self.state_option.index(round(x_next_position, 2))

        # then return the next state
        next_state = State(state=x_next_index)

        return next_state

    '''
    For we have the deterministic policy, our trans_prob is also a deterministic unit

        this should be the state 
    '''

    # FIXME this should be the state
    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        # this is the length value
        next_prob = np.zeros(self.state_shape)

        # change the only choice to 1
        # But we need to consider that our car's next state is deterministic
        current_location = self.state_option[state.val[0]]

        # then get the action
        current_action = self.velocity_option[action.val[0]]

        # then get the next state's position
        next_location = (current_location + self.time_unit * current_action) % self.road_length

        next_state_index = self.state_option.index(round(next_location, 2))

        next_prob[next_state_index] = 1

        return next_prob