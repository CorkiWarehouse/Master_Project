"""
Classes for entities in mean field games.
All entities should inherit from their corresponding classes listed here.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import entr

'''
Constrain all the variables 
We need to make sure it is tractable
'''

MAX = 100000  # maximum number of iterations
MIN = 1e-10


class Agent(object):
    """Base class for all agents."""
    def __init__(self):
        """Initialize the agent."""
        self.state = None
        self.action = None
        self.policy = None

        # this is used to fill in
        # so that we do not have the error report
        pass

    '''
    
    assert isinstance(.. , ..) : Make sure it is the corresponding class
    
    '''

    def set_state(self, state):

        # This is to make sure that state is the State class
        assert isinstance(state, State)
        self.state = state

    def set_action(self, action):
        assert isinstance(action, Action)
        self.action = action

    def set_policy(self, policy):
        '''
        This seems to be wrong with the code.
        We need Class Policy instead of policy which is an instance
        '''
        # FIXME We change this to Policy
        assert isinstance(policy, Policy)
        self.policy = policy

    def update_observation(self, observation, action, reward):
        """Add an observation to the records."""
        pass

    def pick_action(self, observation):
        """Select an action based upon the policy + observation."""
        pass


'''
Action & State & Reward :
    All of them are one-dimensional array
    And the explanation of the process is similar to the action part 
    
k can been seen as agents
T is the time 
s is the agents' state
'''


class Action(object):
    # we let k=1 for init to show that we consider the action for the t=1
    # we do not do anything when t=0
    def __init__(self, action=None, k=1):

        '''

        We need to make sure that :
            1. action is None and k>0 ?
            2. Or action is N-dimensional array from Numpy
            3. Or action is Int type

        We have 3 situation to handle the result
        '''
        assert (action is None and k > 0) or isinstance(action, np.ndarray) or isinstance(action, int)

        # self.val to store the action (which is a N-dimensional in the end)
        if isinstance(action, np.ndarray):
            self.val = action.copy()
        elif isinstance(action, int):
            # change this value to array
            # if action = 10, then self.val = [10]
            self.val = np.array([action])
        else:
            # 0~k-1 array and it is one dimension array
            # if k = 1, np.array(range(k)) = [0]
            self.val = np.array(range(k))

        # k is the action's length
        # this len is the outermost layer of the array
        self.k = len(self.val)


class State(object):
    # k is the length of state
    def __init__(self, state=None, k=1):
        # print(type(state))
        assert (state is None and k > 0) or isinstance(state, np.ndarray) or isinstance(state, int)
        if isinstance(state, np.ndarray):
            self.val = state.copy()
        elif isinstance(state, int):
            self.val = np.array([state])
        else:
            self.val = np.array(range(k))
        self.k = len(self.val)

'''
Unlike above, the init reward is zero

Action and State will be range(k)
'''
class Reward(object):
    def __init__(self, reward=None, k=1):
        assert (reward is None and k > 0) or isinstance(reward, np.ndarray) \
               or isinstance(reward, float) or isinstance(reward, int)
        if isinstance(reward, np.ndarray):
            self.val = reward.copy()
        elif isinstance(reward, int) or isinstance(reward, float):
            self.val = np.array([reward])
        else:
            self.val = np.zeros(k)
        self.k = len(self.val)


'''
Mean Field :
    we let it be list for it represent the population state
    If we have the list input we will have an additional dimension
    
np.array and np.ndarray 's type are all numpy.ndarray
'''

class MeanField(object):
    def __init__(self, mean_field=None, s=1):
        # list or np.ndarray
        assert (mean_field is None and s > 0) or isinstance(mean_field, list) or isinstance(mean_field, np.ndarray)
        if isinstance(mean_field, np.ndarray):
            self.val = mean_field.copy()

        # if it is list, we will have a nparray
        # FIXME Here we change [mean_field] to mean_field
        elif isinstance(mean_field, list):
            self.val = np.array(mean_field)
        else:
            # this is only fitted for s = int
            # will give a 0-array with length = s
            self.val = np.zeros(s)
        self.k = len(self.val)


'''
MeanFieldFlow: with s-agent  and t-time
'''

class MeanFieldFlow(object):
    def __init__(self, mean_field_flow=None, s=1, t=1):
        assert (mean_field_flow is None and s > 0 and t > 0) or isinstance(mean_field_flow, list) \
               or isinstance(mean_field_flow, np.ndarray)
        if isinstance(mean_field_flow, np.ndarray):
            self.val = mean_field_flow.copy()
        elif isinstance(mean_field_flow, list):
            '''
            This seems to be something wrong, we need to give the shape and input value 
            And we can not just put list into it .
            
            So I change it to array
            '''
            # FIXME change np.ndarray to np.array
            self.val = np.array(mean_field_flow)
        else:
            # initial process
            # t-rows and s-columns
            self.val = np.zeros((t, s))
            for _ in range(t):
                self.val[_] = np.array([1/s for _ in range(s)]) # initial mean field is a uniform distribution

        # T is the time and it is the row number
        self.T = np.size(self.val, 0)

        # k is the number of the agents which is the columns
        self.k = np.size(self.val, 1)

'''
Here policy we have already considered the whole 
population 

'''
class Policy(object):
    def __init__(self, policy=None, s=1, a=1):
        assert (policy is None and s > 0) or isinstance(policy, list) or isinstance(policy, np.ndarray)
        if isinstance(policy, np.ndarray):
            self.val = policy.copy()
        elif isinstance(policy, list):
            # FIXME We change [policy] to policy
            self.val = np.array(policy)
        else:
            # s-rows and a-columns
            # One state can be a group of policy
            self.val = np.zeros((s, a))
        self.k = len(self.val)

'''
This should be the 3-d array for we need 
t time's state's all actions
'''
class PolicyFlow(object):
    def __init__(self, policy_flow=None, s=1, a=1, t=1):
        assert (policy_flow is None and s > 0 and a > 0 and t > 0) or isinstance(policy_flow, list) \
               or isinstance(policy_flow, np.ndarray)
        if isinstance(policy_flow, np.ndarray):
            self.val = policy_flow.copy()
        elif isinstance(policy_flow, list):

            '''
            I do not know if we have this input like we can 
            input list into ndarray()
            '''
            # FIXME Here we change ndarray to array
            self.val = np.array(policy_flow)
        else:
            # create 3-dimensions array
            # t - items
            # s lines each
            # a actions
            self.val = np.zeros((t, s, a))

        self.t = np.size(self.val, 0)
        self.s = np.size(self.val, 1)
        self.a = np.size(self.val, 2)

'''
This is the "tao" of an agent

We have 2 array : states and actions 
'''
class Trajectory(object):
    def __init__(self, states=None, actions=None, horizon=1):
        assert (states is None and actions is None and horizon > 0) or \
               (isinstance(states, np.ndarray) and isinstance(actions, np.ndarray) and horizon > 0)
        if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray) and horizon > 0:
            self.states = states.copy()
            self.actions = actions.copy()
        else:
            self.states = np.zeros(horizon)
            self.actions = np.zeros(horizon)

        # reset time to the current horizon
        self.horizon = horizon

'''
This is the whole population's "tao"
So that we use p_flow and mf_flow in it
'''
class TrajectoryMDP(object):
    def __init__(self, p_flow=None, mf_flow=None, horizon=1, state_shape=1, action_shape=1):
        assert (p_flow is None and mf_flow is None and horizon > 0 and state_shape > 0 and action_shape > 0) or \
               (isinstance(p_flow, np.ndarray) and isinstance(mf_flow, np.ndarray) and horizon > 0 and state_shape > 0 and action_shape > 0)
        if isinstance(p_flow, np.ndarray) and isinstance(mf_flow, np.ndarray) and horizon > 0:
            self.p_flow = p_flow
            self.mf_flow = mf_flow
        else:
            self.p_flow = np.zeros((horizon, state_shape, action_shape))
            self.mf_flow = np.zeros((horizon, state_shape))
        self.horizon = horizon


class Environment(object):
    """Base class for all environments."""

    def __init__(self, is_original_dynamics, beta):
        """Initialize the environment."""
        self.action_shape = None
        self.state_shape = None

        # 0 -- original
        # 1 -- not
        self.is_original_dynamics = is_original_dynamics
        self.name = ''
        self.beta = beta

        # FIXME We add num_of_states and num_of_actions
        self.num_of_states = None
        self.num_of_actions = None

        # TODO We add the time variable
        # TODO this is the variable which is used to fit to our CAR model
        self.total_time = None
        pass

    def get_observation(self):
        """Returns an observation from the environment."""
        pass

    def get_reward(self, state, action, mean_field):
        """Gets the reward for a state-action-mean field triplet."""
        pass

    def advance(self, policy, mean_field) -> MeanField:
        """Updating the mean field under infinitely many agents scenarios."""
        pass

    def dynamics(self, state, action, mean_field):
        """Sample the next state according to the transition function."""
        pass

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        """State transition function."""
        pass

    def get_expected_return(self, p_flow, mf_flow):
        """Get expected return of a policy-mean field flow pair"""
        pass


class FiniteWorld(object):
    def __init__(self, num_of_agents: int, environment: Environment, init_mean_field: MeanField, num_of_steps: int):
        # list of agents and entities (can change at execution-time!)
        assert num_of_agents > 0
        self.environment = environment
        self.policy_agents = []
        for i in range(num_of_agents):
            agent = Agent()
            '''
            Our environment do not have this attribute "num_of_states" or "num_of_actions"
            We should add these 2 attributes into our environment
            '''

            # FIXME we do not have this value num_of_states
            agent.policy_flow = PolicyFlow(policy_flow=None, s=self.environment.num_of_states, a=self.environment.num_of_actions, t=num_of_steps)
            # Here we let the agent choose a random state and with the init_mean_field[0]
            agent.state = State(state=int(np.random.choice(np.array(range(self.environment.num_of_states)), init_mean_field.val[0])))

'''
This is the world with infinite agents
So we can not init each agent in an environment
'''
class InfWorld(object):
    def __init__(self, environment: Environment, init_mean_field: MeanField, num_of_steps: int):
        self.environment = environment
        self.mean_field_flow = MeanFieldFlow(mean_field_flow=None, s=self.environment.num_of_states, t=num_of_steps)
        self.mean_field_flow.val[0:] = init_mean_field.val[0]
        self.policy_flow = PolicyFlow(policy_flow=None, s=self.environment.num_of_states, a=self.environment.num_of_actions, t=num_of_steps)


'''

IRL process
'''

class IRL(object):
    def __init__(self, data, env: Environment, horizon: int, device):
        self.data = data
        self.env = env
        self.device = device
        self.reward_model = None
        self.horizon = horizon
        self.mf_flow = None
        self.p_flow = None
        self.expected_return = 0.0


    # Saves the trained reward model to a specified path
    def save_model(self, path: str):
        assert self.reward_model is not None
        torch.save(self.reward_model, path)

    # Loads a reward model from a specified path
    def load_model(self, path: str):
        self.reward_model = torch.load(path)

    # Converts a categorical variable into a one-hot encoded vector
    # all to 0 with length = shape
    # and let entry index be 1
    def onehot_encoding(self, shape, entry):
        code = np.zeros(shape)
        code[entry] = 1.0
        return code

    # this is the training mode
    def train(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
        pass

    # -> [] give the class of the output
    '''
    Recovers the equilibrium mean field Nash equilibrium (ERM-FNE) 
    using an iterative approach.
    '''
    def recover_ermfne(self) -> [MeanFieldFlow, PolicyFlow]:
        assert self.reward_model is not None
        # Init the mf_flow and p_flow
        '''状态特征向量长度'''
        mf_flow = MeanFieldFlow(mean_field_flow=None, s=self.env.state_shape, t=self.horizon)
        p_flow = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

        # Here is the training process
        for _ in range(MAX):
            p_flow = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

            # Here q-value is the p_flow value
            '''
            From range value, we could know that 
            self.env.action_shape is int 
            self.env.state_shape is int
            '''
            q_values = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)

            '''
            Initialization of Policy at Final Time Step (self.horizon-1)
                Policy is initialized to a uniform distribution across all actions for each state. 
                This reflects an assumption of equal likelihood of actions in the absence of future information.
            '''
            for s in range(self.env.state_shape):
                # let last time's policy_flow's action's probability be init = 1/(number_of_action)
                '''Here is the current time for we start at 0'''
                p_flow.val[self.horizon - 1, s, :] = (
                    np.array([1 / self.env.action_shape for _ in range(self.env.action_shape)]))

            # compute Q values and policy flow
            '''
            Calculation of Q-values and Policy Flow
             
            '''
            # for this time t
            # We only calculate until the last 2
            # self.horizon - 2
            for t in reversed(range(0, self.horizon - 1)):
                # every state s
                for s_current in range(0, self.env.state_shape):
                    # for every possible action
                    for a_current in range(0, self.env.action_shape):
                        '''
                        Onehot_encoding is used to fit input requirements of a typical neural network model
                        
                        We could see it separately :
                            1. torch.from_numpy(self.onehot_encoding(self.env.state_shape, s_current)).to(self.device, torch.float)
                                1) One_hot will be list with length = self.env.state_shape and all be 0 but index of s_current is 1
                            2. torch.from_numpy(self.onehot_encoding(self.env.action_shape, a_current)).to(self.device, torch.float)
                                1) One_hot will be list with length = self.env.action_shape and all be 0 but index of a_current is 1
                            3. torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                                1) Input the current mf_flow to the NN
                                
                        So that we can get the current reward which is the Q-value
                        
                        In another word this is immediate Reward 
                        '''
                        q_values.val[t, s_current, a_current] += self.reward_model(torch.from_numpy(self.onehot_encoding(self.env.state_shape, s_current)).to(self.device, torch.float),
                                                                                   torch.from_numpy(self.onehot_encoding(self.env.action_shape, a_current)).to(self.device, torch.float),
                                                                                   torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                                                                                   ).detach().cpu().numpy()
                        # next step
                        '''
                        First we need to consider the next reward
                        
                        In a word, Future Reward
                        '''
                        for s_next in range(0, self.env.state_shape):
                            # Consider the current state and action
                            # we let our q-value at this time to the sum of all next
                            # Same as we first consider the s_next's future reward
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(mean_field=mf_flow.val[t]))[s_next] \
                                                                     * self.env.beta * np.sum(entr(p_flow.val[t+1, s_next, :]))
                                                                    # Above will be the sum of t+1's all policy's value
                                                                    # and this "entr" is the entropy term

                            # then we consider all the action of the s_next
                            for a_next in range(0, self.env.action_shape):
                                q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                             Action(action=a_current),
                                                                                             MeanField(mean_field=mf_flow.val[t]))[s_next] \
                                                                         * p_flow.val[t+1, s_next, a_next] \
                                                                         * q_values.val[t+1, s_next, a_next]
                                                                        # this p_flow.val is the possible that we taking action

                # compute policy induced by the mean filed
                '''
                We have already consider the current S and T 's all actions
                then we need to update the policy under current T
                
                policy is updated using a softmax function = q_values 
                '''
                for s in range(0, self.env.state_shape):
                    partition = 0.0
                    # let our value from 0-1
                    for a in range(0, self.env.action_shape):
                        # have all the exp value beh
                        partition += np.exp(q_values.val[t, s, a] / self.env.beta)
                    for a in range(0, self.env.action_shape):
                        p_flow.val[t, s, a] = np.exp(q_values.val[t, s, a] / self.env.beta) / partition

            # compute mean field flow induced by the policy flow
            mf_flow_next = MeanFieldFlow(mean_field_flow=None, s=self.env.state_shape, t=self.horizon)
            mf_flow_next.val[0] = mf_flow.val[0, :]
            for t in range(1, self.horizon):
                # this will give the next Mean Field
                mf = self.env.advance(Policy(policy=p_flow.val[t-1]), MeanField(mean_field=mf_flow.val[t-1]))
                mf_flow_next.val[t] = mf.val

            # check the distance between new and old mean field flows
            '''
            This is Convergence Check
            '''
            distance = torch.nn.MSELoss(reduction='sum', size_average=True)
            if distance(torch.from_numpy(mf_flow_next.val), torch.from_numpy(mf_flow.val)) < MIN:
                break
            else:
                mf_flow = mf_flow_next

        # this is return value
        self.mf_flow = mf_flow
        self.p_flow = p_flow
        return [mf_flow, p_flow]


    '''
    We use this method to get the return from q_values
    '''
    def recover_expected_return(self):
        assert self.mf_flow is not None and self.p_flow is not None
        q_values = PolicyFlow(policy_flow=None, s=self.env.state_shape, t=self.horizon, a=self.env.action_shape)
        # we update it from reverse
        for t in reversed(range(0, self.horizon - 1)):
            for s_current in range(0, self.env.state_shape):
                for a_current in range(0, self.env.action_shape):
                    # unlike above, we directly get the reward
                    q_values.val[t, s_current, a_current] += self.env.get_reward(State(state=s_current),
                                                                                 Action(action=a_current),
                                                                                 MeanField(mean_field=self.mf_flow.val[t])).val[0]
                    # next step
                    for s_next in range(self.env.state_shape):
                        q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                     Action(action=a_current),
                                                                                     MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                 * self.env.beta * np.sum(entr(self.p_flow.val[t+1, s_next, :]))

                        for a_next in range(0, self.env.action_shape):
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(mean_field=self.mf_flow.val[t]))[s_next] \
                                                                     * self.p_flow.val[t+1, s_next, a_next] \
                                                                     * q_values.val[t+1, s_next, a_next]

        # compute expected return under equilibrium and terminate iteration

        # For we have update the q_values from back
        # So we can get the expected return from the initial state
        for s in range(0, self.env.state_shape):
            partition = 0.0
            for a in range(0, self.env.action_shape):
                partition += np.exp(q_values.val[0, s, a] / self.env.beta)
            for a in range(0, self.env.action_shape):
                self.expected_return += self.mf_flow.val[0, s] * np.exp(q_values.val[0, s, a] / self.env.beta) * q_values.val[0, s, a] / partition


    '''
    This we could see it as D
    '''
    def divergence(self, expert_mf_flow, expert_p_flow):
        self.recover_ermfne()
        self.recover_expected_return()

        # Here we calculate the Divergence
        kl_div = nn.KLDivLoss(reduction='sum')
        dev_mean_field = kl_div((torch.from_numpy(self.mf_flow.val) + 1e-9).log(),
                                torch.from_numpy(expert_mf_flow.val)
                                ).detach().cpu().numpy()
        kl_div = nn.KLDivLoss(reduction='none')
        kl_policy = kl_div( torch.from_numpy(self.p_flow.val + 1e-9).log(), torch.from_numpy(expert_p_flow.val) ).reshape(-1, self.env.action_shape).sum(1)
        dev_policy = torch.mul(kl_policy, torch.from_numpy(expert_mf_flow.val.reshape(1, -1))).sum().detach().cpu().numpy()
        return [self.expected_return, dev_mean_field, dev_policy]
