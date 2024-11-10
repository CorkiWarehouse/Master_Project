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
        self.state_shape = 1

        # for we only allow [-1,1] to be our
        self.action_shape = 1

        # here is the true value
        self.state_option = [-1,1]
        self.action_option = [-1,1]

        self.state_count = 2
        self.action_count = 2

        # here is the variables that is used in the PINN
        self.time_unit = 1
        self.position_unit = 2

        self.init_mf = None
        self.dim = 1


    def get_reward(self, state, action, mean_field,h = 0,lam=1):

        # Here we give the external part will be the left
        # And right this action will give the influnce on this System
        # we use the MF to represent this world

        # 获取与动作对应的自旋值
        a_j = self.action_option[int(action.val[0])]  # 动作自旋（-1 或 +1）

        # 计算平均磁化强度（均值场）
        m = np.dot(mean_field.val, self.state_option)  # 平均磁化强度

        # 奖励函数：r = h * a_j + lambda * a_j * m
        reward = h * a_j +  lam * a_j * m


        return Reward(reward = reward)

    def advance(self, policy, mean_field) -> MeanField:
        # init
        next_mean_field = MeanField(mean_field=None, s=self.state_count)

        # here is all the next_state
        for next_state in range(self.state_count):
            test_set = []
            sum_next = 0
            # here is all the current state which can reach to next one
            # but for our time_unit is 5
            # not all the action that can reach this 'next_state'
            for current_state in range(self.state_count):
                sum_policy_transition = 0

                # we directly check all the valid actions
                for current_action in range(self.action_count):
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

    # def advance(self, policy, mean_field) -> MeanField:
    #     next_mean_field = MeanField(mean_field=None, s=self.state_count)
    #     next_mean_field.val = np.zeros(self.state_count)
    #
    #     for s in range(self.state_count):
    #         for a in range(self.action_count):
    #             action_prob = policy.val[s, a]
    #             next_state_index = int(a)  # Next state is determined by action
    #             next_mean_field.val[next_state_index] += mean_field.val[s] * action_prob
    #
    #     # Normalize the mean field
    #     total = np.sum(next_mean_field.val)
    #     if total > 1e-12:
    #         next_mean_field.val /= total
    #     else:
    #         next_mean_field.val = np.ones(self.state_count) / self.state_count
    #
    #     return next_mean_field

    '''
    In article it is deterministic.
        But we need to calculate one judgement to decide if we will change 
    '''

    def dynamics(self, state, action, mean_field=None) -> State:

        next_state = np.array([0,0])

        if int(action.val[0]) == 1:
            next_state[1] = 1
        else:
            next_state[0] = 1

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
        next_prob = np.zeros(self.state_count)

        # here we will consider a process to judge if we will change

        if int(action.val[0]) == 1:
            next_prob[1] = 1
        else:
            next_prob[0] = 1

        return next_prob


    # this is because that our action will directly
    # give the results so we just need to all the states
    def get_neighbors(self, state, mean_field = None):

        neighbors = []

        for d in range(self.state_count):
            neighbors.append(d)
        return np.array(neighbors)
