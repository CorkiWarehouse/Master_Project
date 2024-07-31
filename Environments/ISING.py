"""
This is the ISing environment, we have :
    1. Agent: We have the agents which aim to max the reward
    2. State: In this environment, our state only 2 choices: -1 or 1.
        means the Spin direction
    3. Action: Here action is also can be seen as the state.
        this times action will be the agent's next state (With the MCMC to check if we take this action )
    4. MeanFiled: this shows the  population density of this system
    5. Reward: In our model, this reward is related with MF and the action of this agent

In the original part, we have to consider the order parameter which is used to
    control when to stop. But we all let the epoch be the same
"""



import numpy as np

from core import State, Reward, Environment, MeanField,Action


class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'ISLING'

        # will be [-1,1]
        self.state_shape = 2

        # for we only allow [-1,1] to be our
        self.action_shape = 2

        # here is the true value
        self.state_value = [-1,1]
        self.action_value = [-1,1]

        # here is the variables that is used in the PINN


    def get_reward(self, state, action, mean_field,h = 0):

        # Here we give the external part will be the left
        # And right this action will give the influnce on this System
        # we use the MF to represent this world

        left = h * self.state_value[state.val[0]]

        lamed = 1

        right = mean_field.val[state.val[0]] * self.action_value[action.val[0]]

        reward = left + 0.5 * lamed * right


        return Reward(reward = reward)

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
    In article it is deterministic.
        But we need to calculate one judgement to decide if we will change 
    '''

    def dynamics(self, state, action, mean_field=None) -> State:

        next_state = int(action.val[0])

        # if we decide to do not change we will stay the state
        if self.trans_prob(state, action, mean_field)[action.val[0]] == 0 :
            next_state = int(state.val[0])

        # then return the next state
        next_state = State(state=next_state)

        return next_state

    '''
    For we have the deterministic policy, our trans_prob is also a deterministic unit

        this should be the state 
        
    But we keep this the directly for we need to make sure 
        the PINN is worked
    '''

    # FIXME this should be the state
    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        # this is the length value
        next_prob = np.zeros(self.state_shape)

        # here we will consider a process to judge if we will change

        next_prob[action.val[0]] = 1

        return next_prob