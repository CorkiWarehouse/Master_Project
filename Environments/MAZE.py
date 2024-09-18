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

        self.dim = 2

        # TODO Could we still use this value ?
        # here is the action option which is 5
        # means that we can go left,right,stay,up and down
        self.action_option =  np.array([[-1,0],[1,0],[0,0],[0,1],[0,-1]])
        self.state_option = np.array([[x, y] for x in range(self.limit) for y in range(self.limit)])
        # self.current_velocity = [0 for i in range(self.state_shape)]


        self.c_cos = 0.1

        self.ref_position = np.array([0,0])

        self.wall_position = self.generate_wall()

        # here we need an init-MF as the start point
        # we let all the agent be the right corner
        self.init_mf = [0.0 for _ in range(self.state_count)]
        self.init_mf[-1] = 1
        self.init_mf = MeanField(mean_field=self.init_mf)


        # here is to give the random part in our method
        self.p = 0.8
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
                    prob_transition = self.trans_prob(State(state=current_state), Action(action=current_action), mean_field)[next_state]

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

    # def dynamics(self, state, action, mean_field=None) -> State:
    #
    #     # get the x
    #     x_position = self.state_option[state.val[0]]
    #
    #     # here we get the action
    #     current_action = self.action_option[action.val[0]]
    #
    #     x_next_index = state.val[0]
    #
    #
    #     next_position = x_position + current_action * self.time_unit
    #
    #     # here the next position should be limited by the wall
    #     if (next_position in self.wall_position):
    #         x_next_index = x_next_index
    #     else:
    #         if not (0 <= next_position[0] < self.limit) or not (0 <= next_position[1] < self.limit):
    #             x_next_index = x_next_index  # If out of bounds, revert to the original position
    #         else:
    #             x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
    #
    #     # then return the next state
    #     next_state = State(state=int(x_next_index))
    #
    #     return next_state
    #
    # def dynamics(self, state, action, mean_field=None) -> State:
    #     # Constants
    #     NOISE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1),
    #                      (0, -1)]  # Assuming (0, 0) means no movement and others are possible disturbances
    #
    #     # Current position and action
    #     x_position = self.state_option[state.val[0]]
    #     current_action = self.action_option[action.val[0]]
    #
    #     # Determine whether to apply the intended action or a noise action
    #     if np.random.rand() <= self.p:
    #         next_position = x_position + current_action * self.time_unit
    #     else:
    #         noise_action = np.random.choice(range(len(NOISE_ACTIONS)), p=[1 - self.p] + [self.p / 4] * 4)
    #         next_position = x_position + np.array(NOISE_ACTIONS[noise_action]) * self.time_unit
    #
    #     # Check if the next position is within bounds and not a wall
    #     if self.is_valid_position(next_position):
    #         x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
    #     else:
    #         x_next_index = state.val[0]  # Revert to original position if invalid
    #
    #     # Return the next state
    #     return State(state=int(x_next_index))

    def dynamics(self, state, action, mean_field=None) -> State:
        if self.is_original_dynamics == 0:
            # Original dynamics function
            x_position = self.state_option[state.val[0]]
            current_action = self.action_option[action.val[0]]
            x_next_index = state.val[0]

            next_position = x_position + current_action * self.time_unit

            if (next_position in self.wall_position):
                x_next_index = x_next_index
            else:
                if not (0 <= next_position[0] < self.limit) or not (0 <= next_position[1] < self.limit):
                    x_next_index = x_next_index  # If out of bounds, revert to the original position
                else:
                    x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])

        else:
            # Modified dynamics function with noise actions
            NOISE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

            x_position = self.state_option[state.val[0]]
            current_action = self.action_option[action.val[0]]

            if np.random.rand() <= self.p:
                next_position = x_position + current_action * self.time_unit
            else:
                noise_action = np.random.choice(range(len(NOISE_ACTIONS)), p=[1 - self.p] + [self.p / 4] * 4)
                next_position = x_position + np.array(NOISE_ACTIONS[noise_action]) * self.time_unit

            if self.is_valid_position(next_position):
                x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
            else:
                x_next_index = state.val[0]

        return State(state=int(x_next_index))

    '''
    For we have the deterministic policy, our trans_prob is also a deterministic unit

        this should be the state 
    '''

    # FIXME this should be the state
    # def trans_prob(self, state, action, mean_field) -> np.ndarray:
    #     # this is the length value
    #     next_prob = np.zeros(self.state_count)
    #
    #     # change the only choice to 1
    #     # But we need to consider that our car's next state is deterministic
    #     x_position = self.state_option[state.val[0]]
    #
    #     # then get the action
    #     current_action = self.action_option[action.val[0]]
    #
    #     next_position = x_position + current_action * self.time_unit
    #
    #     x_next_index = state.val[0]
    #
    #     # here the next position should be limited by the wall
    #     if (next_position in self.wall_position):
    #         x_next_index = x_next_index
    #     else:
    #         if not (0 <= next_position[0] < self.limit) or not (0 <= next_position[1] < self.limit):
    #             x_next_index = x_next_index  # If out of bounds, revert to the original position
    #         else:
    #             x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
    #
    #     next_prob[x_next_index] = 1
    #
    #     return next_prob
    #
    # def trans_prob(self, state, action, mean_field=None) -> np.ndarray:
    #     # Probabilities initialization
    #     next_prob = np.zeros(self.state_count)
    #
    #     # Define noise actions (assuming (0, 0) is no movement, others are directions)
    #     NOISE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    #
    #     # Get the current position and action
    #     x_position = self.state_option[state.val[0]]
    #     current_action = self.action_option[action.val[0]]
    #
    #     # Calculate probabilities for each possible state outcome
    #     for noise_action in NOISE_ACTIONS:
    #         if noise_action == (0, 0):
    #             action_prob = self.p
    #         else:
    #             action_prob = (1 - self.p) / 4
    #
    #         next_position = x_position + (np.array(current_action) + np.array(noise_action)) * self.time_unit
    #
    #         if self.is_valid_position(next_position):
    #             x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
    #             next_prob[x_next_index] += action_prob
    #
    #     return next_prob

    def trans_prob(self, state, action, mean_field=None) -> np.ndarray:
        # Probabilities initialization
        next_prob = np.zeros(self.state_count)

        # Get the current position and action
        x_position = self.state_option[state.val[0]]
        current_action = self.action_option[action.val[0]]

        if self.is_original_dynamics == 0:
            # Deterministic transition probability calculation
            next_position = x_position + current_action * self.time_unit

            if self.is_valid_position(next_position):
                x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
                next_prob[x_next_index] = 1
            else:
                next_prob[state.val[0]] = 1  # Stay in the current state if out of bounds

        else:
            # Probabilistic transition with noise model
            # Define noise actions (assuming (0, 0) is no movement, others are directions)
            NOISE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

            for noise_action in NOISE_ACTIONS:
                if noise_action == (0, 0):
                    action_prob = self.p
                else:
                    action_prob = (1 - self.p) / 4

                next_position = x_position + (np.array(current_action) + np.array(noise_action)) * self.time_unit

                if self.is_valid_position(next_position):
                    x_next_index = int(np.where((self.state_option == next_position).all(axis=1))[0][0])
                    next_prob[x_next_index] += action_prob
                else:
                    next_prob[state.val[0]] += action_prob  # Distribute probability to staying in current state

        return next_prob

    # def generate_wall(self, wall_number):
    #
    #     # Generate all available points excluding the destination [0, 0]
    #     all_points = [(x, y) for x in range(self.limit) for y in range(self.limit) if (x, y) != (0, 0)]
    #
    #     # Randomly choose wall points
    #     walls = np.random.choice(len(all_points), size=wall_number, replace=False)
    #     wall_points = [all_points[i] for i in walls]
    #
    #     return np.array(wall_points)

    def generate_wall(self):
        # Fixed set of wall points, ensure no duplicates and valid within the grid
        wall_points = [[2, 1], [0, 1]]
        return np.array(wall_points)

    def get_neighbors(self, state, mean_field = None):
        # Convert state_index to x, y coordinates
        x, y = self.state_option[state]

        # Possible movements: left, right, up, down
        movements = self.action_option
        neighbors = []

        for dx, dy in movements:
            nx, ny = x - dx * self.time_unit, y - dy*self.time_unit

            # Check if the new position is within grid limits
            if 0 <= nx < self.limit and 0 <= ny < self.limit:
                # Check if the new position is not a wall
                if not any((nx == wx and ny == wy) for wx, wy in self.wall_position):
                    # Find the index of this valid neighbor position
                    neighbor_index = np.flatnonzero((self.state_option == [nx, ny]).all(axis=1))
                    if neighbor_index.size > 0:
                        neighbors.append(neighbor_index[0])

        return np.array(neighbors)

    def is_valid_position(self, pos):
        if not (0 <= pos[0] < self.limit) or not (0 <= pos[1] < self.limit):
            return False
        if pos in self.wall_position:
            return False
        return True

