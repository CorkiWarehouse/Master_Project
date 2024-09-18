"""
This is the Cars environment

We let our cars run on a ring road with length = 1000
    which means that our cars can be in 1000 different points

So we suppose that we have 6 points on it as the state
    we have 10 times(including 0) points with difference = 1
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

In the action, we also let it be in 1000(velocity)
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

In state, we have this options:
    [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

And for we have the time

"""
import random

import numpy as np
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):

    # init function
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'CARS'

        self.state_shape = 1

        # For we have 10 choices
        self.action_shape = 1

        # this is one hyper-parameter
        self.position_unit = 1/8 # here is the delt_x
        self.time_unit = self.position_unit  # delt_t time difference
        self.road_length = 1 # this is the length of the road

        self.velocity_max = 1 # max velocity

        self.total_time = 10  # time horizon

        # TODO Could we still use this value ?
        self.action_option = np.round(np.arange(0, 1, self.time_unit), 3).tolist()  # here is all velocity choices
        self.state_option = np.round(np.arange(0,1,self.position_unit),3).tolist()
        # self.current_velocity = [0 for i in range(self.state_shape)]

        # this value is replaced by the env.horizen
        self.state_count = len(self.state_option)
        self.action_count = len(self.action_option)

        self.p = 0.1
        self.noise_option = self.action_option

        # here is the init meanField
        values = np.random.rand(self.state_count)
        self.init_mf = MeanField( mean_field= values/values.sum(), s = self.state_count)


        self.dim = 1

    '''
    Here is the reward function 
    We use f((action(x,t), mean_field(x, t))) * t_delt
    
    For we need the Monotone MFG
    We use separable cost :
        1/2*(U(ro)-u)**2
    
    NOTICE: we need to minimize the cost function
            which is the reward function
    
    
    State and Action are the index 
    '''
    def get_reward(self, state, action, mean_field):

        # For we have one-hot encoding
        # get the velocity from the action list
        velocity = self.action_option[action.val[0]]

        # get x from the state
        x_position = state.val[0]

        U = 1 - mean_field.val[x_position]

        reward = 0.5*((U - velocity)**2)

        '''

        Here is the alternative way to get the reward

        '''
        # 0.5 * speed ** 2 + rho - speed

        # velocity = self.velocity_option[action.val[0]]
        # reward = 0.5 * velocity ** 2 + mean_field.val[state.val[0]] - velocity

        # reward = np.random.rand()


        return Reward(reward=reward)


    '''
    This is the method that we use to update our mean_field
    
    We need to update the mean_field through the policy
    
    Variable's type :
        1. next_state/current_state: int
        2. action: int
        3. 
    '''
    def advance(self, policy, mean_field) -> MeanField:
        # init
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



        '''
        Now we consider a new way to update the mean_field
            1. We just focus on this place's flow. 
            2. We let last time's mean field + last place & last time's flow - current place & last time's flow
            3. In another word: last time's value + change of the flow 
            
        Here is the code :
            rho[i, t] = (
                rho[i][t - 1]
                + rho[i - 1, t - 1] * u[i - 1, t - 1]
                - rho[i, t - 1] * u[i, t - 1]
                )
        '''

        # So we could have
        # for next_state in range(self.state_shape):
        #     now_mean_field = mean_field[next_state]
        #     last_velocity = policy.val[]

        return next_mean_field


    '''
    In article it is deterministic, so we do not have the probability 
        
        if we have the state and action, we will get the next state
        and this state is not relevant with mean field 
    '''

    def dynamics(self, state, action, mean_field = None) -> State:

        # get the x
        x_position = self.state_option[state.val[0]]

        # get the v
        velocity = self.action_option[action.val[0]]

        # get the next time and position
        # if it out of range
        # x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        if self.is_original_dynamics == 0:
            # Original dynamics without noise
            x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        else:
            # Modified dynamics with noise
            noise_velocity = (1 - self.p) * velocity + self.p * random.choice(self.noise_option)
            x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length
            x_next_position = self.find_closest_position(x_next_position)

        # here we give this value's closest value
        x_next_index =  self.state_option.index(min(self.state_option, key=lambda x: abs(x - x_next_position)))

        if x_next_index == state.val[0]:
            x_next_index = (x_next_index + 1) % self.state_count

        # then return the next state
        next_state = State(state = x_next_index)

        return next_state


    '''
    For we have the deterministic policy, our trans_prob is also a deterministic unit
        
        this should be the state 
    '''
    # FIXME this should be the state
    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        # this is the length value
        next_prob = np.zeros(self.state_count)

        # change the only choice to 1
        # But we need to consider that our car's next state is deterministic
        x_position = self.state_option[state.val[0]]

        # then get the action
        velocity = self.action_option[action.val[0]]


        # then get the next state's position
        # next_location = (current_location + self.time_unit * current_action) % self.road_length

        if self.is_original_dynamics == 0:
            # Original dynamics without noise
            x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        else:
            # Modified dynamics with noise
            noise_velocity = (1 - self.p) * velocity + self.p * random.choice(self.noise_option)
            x_next_position = (x_position + self.time_unit * noise_velocity) % self.road_length
            x_next_position = self.find_closest_position(x_next_position)


        next_state_index = self.state_option.index(min(self.state_option, key=lambda x: abs(x - x_next_position)))

        if next_state_index == state.val[0]:
            next_state_index = (next_state_index + 1) % self.state_count

        next_prob[next_state_index] = 1


        return next_prob

    def find_closest_position(self, target_position):
        # Calculate absolute differences
        differences = [abs(x - target_position) for x in self.state_option]
        # Find index of the minimum difference
        min_index = differences.index(min(differences))
        # Return the closest position
        return self.state_option[min_index]

    '''
    Here current_position is the 
        we return the action which is the index 
    '''
    # def get_all_valid_actions(self,current_position,next_position):
    #     valid_actions = []
    #
    #     # for we have the actions for all the valid choices
    #     for action in range(self.action_shape):
    #
    #         if (round((self.time_unit * self.velocity_option[action] + self.state_option[current_position])
    #                 % self.road_length,2) == self.state_option[next_position]):
    #             valid_actions.append(action)
    #
    #     return valid_actions

