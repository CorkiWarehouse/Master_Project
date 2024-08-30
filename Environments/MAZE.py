# this envrionment is about crowd motion with congestion in maze


import numpy as np
from core import State, Reward, Environment, MeanField, Action


class Env(Environment):

    # init function
    def __init__(self, is_original_dynamics: int, beta: float):
        super().__init__(is_original_dynamics, beta)
        self.name = 'MAZE'

        self.state_shape = 2

        # For we have 10 choices
        self.action_shape = 2

        # this is one hyper-parameter
        self.position_unit = 1  # here is the delt_x
        self.time_unit = 1  # delt_t time difference

        # here is the limit of the maze
        self.limit = 5

        # this value is replaced by the env.horizen
        self.state_count = self.limit ** 2
        self.action_count = 5

        # TODO Could we still use this value ?
        # here is the action option which is 5
        # means that we can go left,right,stay,up and down
        self.action_option =  np.array([[-1,0],[1,0],[0,0],[0,1],[0,-1]])
        self.state_option = np.array([[x, y] for x in range(self.limit) for y in range(self.limit)])
        # self.current_velocity = [0 for i in range(self.state_shape)]


        self.c_cos = 0.1

        self.ref_position = np.array([0,0])

        # for we have defined a maze , we need the wall
        # here we let 30% of the maze will be the wall
        self.wall_number = int(0.3 * self.state_count)

        self.wall_position = self.generate_wall(self.wall_number)
    '''
    Here is the reward function 
    
    r(s,a,mu) = c_cos * r_pos(s) + r_move(a,mu)
    
    here we define :
        c_cos = 0.1
        r_pos = -c_cos * dist(x,x_ref) where x_ref is the object position 
        r_move = -mu * ||a|| where ||a|| is 1 from our about definition 
     
    NOTICE: we need to minimize the cost function
            which is the reward function


    State and Action are the index 
    '''

    def get_reward(self, state, action, mean_field):

        current_position = np.array(self.state_option[state.val[0]])

        current_action = np.array(self.action_option[action.val[0]])

        left = self.c_cos * np.linalg.norm(current_position - self.ref_position)

        # here 1 is the ||a||
        right = -mean_field.val[state.val[0]] * 1

        reward = left + right

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

        # here we get the action
        current_action = self.action_option[action.val[0]]

        x_next_index = state.val[0]

        next_position = x_position + current_action

        # here the next position should be limited by the wall
        if (next_position in self.wall_position):
            x_next_index = x_next_index
        else:
            if not (0 <= next_position[0] < self.limit) or not (0 <= next_position[1] < self.limit):
                x_next_index = x_next_index  # If out of bounds, revert to the original position
            else:
                x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])

        # then return the next state
        next_state = State(state=int(x_next_index))

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
        current_action = self.action_option[action.val[0]]

        next_position = x_position + current_action

        x_next_index = state.val[0]

        # here the next position should be limited by the wall
        if (next_position in self.wall_position):
            x_next_index = x_next_index
        else:
            if not (0 <= next_position[0] < self.limit) or not (0 <= next_position[1] < self.limit):
                x_next_index = x_next_index  # If out of bounds, revert to the original position
            else:
                x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])

        next_prob[x_next_index] = 1

        return next_prob


    def generate_wall(self, wall_number):

        # Generate all available points excluding the destination [0, 0]
        all_points = [(x, y) for x in range(self.limit) for y in range(self.limit) if (x, y) != (0, 0)]

        # Randomly choose wall points
        walls = np.random.choice(len(all_points), size=wall_number, replace=False)
        wall_points = [all_points[i] for i in walls]

        return np.array(wall_points)


