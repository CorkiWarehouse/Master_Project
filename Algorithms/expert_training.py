"""
Compute MFLQRE and generate trajectories
"""

import numpy as np
import torch

'''
Here we have a path problem:
 If we do not give any absolute path or .., our code will search from the source root.
 So we can just put all the code we need in the root.
'''

from core import State, Action, Environment, MeanField, MeanFieldFlow, Policy, PolicyFlow, Trajectory, TrajectoryMDP
from scipy.special import entr

MAX = 1000000  # maximum number of iterations
MIN = 1e-10


class Expert(object):

    # init the Expert
    def __init__(self, env: Environment, horizon: int):

        '''
        env: Environment
        horizon: Time Horizon
        mf_flow: Mean Field Flow (mu)
        p_flow: Policy Flow (pi)
        expected_return: Expected return value
        '''

        self.env = env
        self.horizon = horizon
        self.mf_flow = None
        self.p_flow = None
        self.expected_return = 0.0

    # generate the mean field nash equilibrium
    '''
        
    '''

    def compute_ermfne(self):
        mf_flow = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
        p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
        for _ in range(MAX):
            p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
            q_values = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
            for s in range(self.env.state_count):
                p_flow.val[self.horizon - 1, s, :] = np.array(
                    [1 / self.env.action_count for _ in range(self.env.action_count)])

            # print(p_flow.val)

            # compute Q values and policy flow
            for t in reversed(range(0, self.horizon - 1)):
                for s_current in range(0, self.env.state_count):
                    for a_current in range(0, self.env.action_count):
                        q_values.val[t, s_current, a_current] += self.env.get_reward(State(state=s_current),
                                                                                     Action(action=a_current),
                                                                                     MeanField(mean_field=mf_flow.val[
                                                                                         t])).val[0]
                        # next step
                        for s_next in range(self.env.state_count):
                            q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                         Action(action=a_current),
                                                                                         MeanField(
                                                                                             mean_field=mf_flow.val[
                                                                                                 t]))[s_next] \
                                                                     * self.env.beta * np.sum(
                                entr(p_flow.val[t + 1, s_next, :]))

                            for a_next in range(0, self.env.action_count):
                                q_values.val[t, s_current, a_current] += self.env.trans_prob(State(state=s_current),
                                                                                             Action(action=a_current),
                                                                                             MeanField(
                                                                                                 mean_field=mf_flow.val[
                                                                                                     t]))[s_next] \
                                                                         * p_flow.val[t + 1, s_next, a_next] \
                                                                         * q_values.val[t + 1, s_next, a_next]

                # print(p_flow.val)
                # print(mf_flow.val)
                # print(q_values.val)

                # compute policy induced by the mean filed
                for s in range(0, self.env.state_count):
                    partition = 0.0
                    for a in range(0, self.env.action_count):
                        partition += np.exp(q_values.val[t, s, a] / self.env.beta)
                    for a in range(0, self.env.action_count):
                        p_flow.val[t, s, a] = np.exp(q_values.val[t, s, a] / self.env.beta) / partition

            # compute mean field flow induced by the policy flow
            mf_flow_next = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
            mf_flow_next.val[0] = mf_flow.val[0, :]
            for t in range(1, self.horizon):
                mf = self.env.advance(Policy(policy=p_flow.val[t - 1]), MeanField(mean_field=mf_flow.val[t - 1]))
                mf_flow_next.val[t] = mf.val

            # check the distance between new and old mean field flows
            #sinkhorn = geomloss.SamplesLoss('sinkhorn')
            #distance = np.array([sinkhorn(torch.from_numpy(mf_flow_next.val), torch.from_numpy(mf_flow.val)) for t in range(0, self.horizon)])
            distance = torch.nn.MSELoss(reduction='sum', size_average=True)
            if distance(torch.from_numpy(mf_flow_next.val), torch.from_numpy(mf_flow.val)) < MIN:
                # compute expected return under equilibrium and terminate iteration
                for s in range(0, self.env.state_count):
                    partition = 0.0
                    for a in range(0, self.env.action_count):
                        partition += np.exp(q_values.val[0, s, a] / self.env.beta)
                    for a in range(0, self.env.action_count):
                        self.expected_return += mf_flow.val[0, s] * np.exp(q_values.val[0, s, a] / self.env.beta) * \
                                                q_values.val[0, s, a] / partition
                break
            else:
                # continue iteration
                mf_flow = mf_flow_next

        self.mf_flow = mf_flow
        self.p_flow = p_flow

    def generate_trajectories(self, num_game_play: int, num_traj: int):
        states = [i for i in range(self.env.state_count)]
        actions = [i for i in range(self.env.action_count)]
        assert (self.mf_flow is not None) and (self.p_flow is not None)
        data = [Trajectory(states=None, actions=None, horizon=self.horizon) for _ in range(num_game_play * num_traj)]
        for i in range(num_game_play * num_traj):
            for t in range(self.horizon):
                #print(self.mf_flow.val[t, :])
                '''
                According to the mean_field, we select the state randomly from states
                '''
                # FIXME : Here we let the output from the numpy to the int value
                # FIXME: original : s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])
                s = np.random.choice(states, 1, p=self.mf_flow.val[t, :])[0]
                #print(self.p_flow.val[t, s, :][0])

                # FIXME We change the code here from p=self.p_flow.val[t, s, :][0]
                # FIXME to p=self.p_flow.val[t, s, :])[0] PUT THE [0] out side
                a = np.random.choice(actions, 1, p=self.p_flow.val[t, s, :])[0]
                data[i].states[t] = s
                data[i].actions[t] = a
        return data

    # def generate_trajectories_MDP(self, num_game_play: int, num_traj: int):
    #     # invoke generate_trajectories for N times
    #     assert (self.mf_flow is not None) and (self.p_flow is not None)
    #     dataMDP = [TrajectoryMDP(p_flow=None,
    #                              mf_flow=None,
    #                              horizon=self.horizon,
    #                              state_shape=self.env.state_shape,
    #                              action_shape=self.env.action_shape) for _ in range(num_game_play)]
    #     for n in range(num_game_play):
    #         data = self.generate_trajectories(1, num_traj)
    #         for t in range(self.horizon):
    #             for sample in data:
    #                 '''
    #                 Why we have +1 in this case?
    #                 '''
    #                 dataMDP[n].mf_flow[t, int(sample.states[t])] += 1
    #                 dataMDP[n].p_flow[t, int(sample.states[t]), int(sample.actions[t])] += 1
    #
    #     for sampleMDP in dataMDP:
    #         for t in range(self.horizon):
    #             sampleMDP.mf_flow[t] /= num_traj
    #             normalise_factor = np.zeros(self.env.state_shape)
    #             for s in range(self.env.state_shape):
    #                 normalise_factor[s] = np.sum(sampleMDP.p_flow[t, s, :])
    #             #print(normalise_factor)
    #             for s in range(self.env.state_shape):
    #                 if int(normalise_factor[s]) != 0:
    #                     sampleMDP.p_flow[t, s] /= normalise_factor[s]
    #
    #     return dataMDP

    def load_opt_mfe(self, path: str):
        """
        directly load a socially optimal MFNE through averaging over trajectories
        :param path: file path of demons
        :return: mean field and policy of the socially optimal MFNE
        """
        mean_field = np.loadtxt(path + 'mf.txt', delimiter=',')
        policy = np.fromfile(path + 'policy.bin', dtype=np.float32)
        self.mf_flow = MeanFieldFlow(mean_field_flow=mean_field[0:self.horizon, ])
        self.p_flow = PolicyFlow(
            policy_flow=np.reshape(policy, (100, self.env.state_count, self.env.action_count))[0:self.horizon, :])
