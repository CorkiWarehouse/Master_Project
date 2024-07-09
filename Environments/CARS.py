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

import numpy as np
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):

    # init function
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'CARS'

        # one-hot encoding
        # 1000 points
        self.state_shape = 20

        # For we have 10 choices
        self.action_shape = 10

        # this is one hyper-parameter
        self.time_unit = 1 # delt_t time difference
        self.position_unit = 1 # here is the delt_x
        self.road_length = 1 # this is the length of the road

        self.velocity_max = 1 # max velocity

        # TODO Could we still use this value ?
        self.velocity_option = [round(0.1 * i, 2)  for i in range(1,self.action_shape+1)]  # here is all velocity choices
        self.state_option = [round(0.05 * i, 2)  for i in range(0,self.state_shape)]
        self.current_velocity = [0 for i in range(self.state_shape)]

        # this value is replaced by the env.horizen
        self.total_time = 10 # time horizon
        self.current_time = 0 # this is the time

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
        velocity = self.velocity_option[action.val[0]]

        # get x from the state
        x_position = state.val[0]

        U = 1 - mean_field.val[x_position]

        reward = 0.5*((U - velocity)**2)

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
        next_mean_field = MeanField(mean_field=None, s=self.state_shape)

        # here is all the next_state
        for next_state in range(self.state_shape):
            test_set = []
            sum_next = 0
            # here is all the current state which can reach to next one
            # but for our time_unit is 5
            # not all the action that can reach this 'next_state'
            for current_state in range(self.state_shape):

                # so that, at first, we need to select the valid action
                # and for we let it be the current_position and next_position
                # TODO do we need to check all the valid action, or we can just ignore it
                valid_actions = self.get_all_valid_actions(current_state,next_state)
                # print("current_state", current_state, "valid_actions", valid_actions)
                # print(valid_actions)
                test_set.extend(valid_actions)
                sum_policy_transition = 0

                # we directly check all the valid actions
                for current_action in range(self.action_shape):
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
        velocity = self.velocity_option[action.val[0]]

        # get the next time and position
        # if it out of range
        x_next_position = (x_position + self.time_unit * velocity) % self.road_length
        x_next_index = self.state_option.index(round(x_next_position, 2))

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


    '''
    Here current_position is the 
        we return the action which is the index 
    '''
    def get_all_valid_actions(self,current_position,next_position):
        valid_actions = []

        # for we have the actions for all the valid choices
        for action in range(self.action_shape):

            if (round((self.time_unit * self.velocity_option[action] + self.state_option[current_position])
                    % self.road_length,2) == self.state_option[next_position]):
                valid_actions.append(action)

        return valid_actions

