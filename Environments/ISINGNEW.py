# import numpy as np
# from core import State, Reward, Environment, MeanField, Action
#
#
# class Env(Environment):
#     def __init__(self, is_original_dynamics: int, beta: float):
#         super().__init__(is_original_dynamics, beta)
#         self.name = 'ISINGNEW'
#
#         # 扩大状态空间为 [-2, -1, 1, 2]
#         self.state_option = [-2, -1, 1, 2]
#         self.action_option = [-1, 1]  # 动作保持不变
#
#         self.state_shape = 1
#         self.action_shape = 1
#
#         self.state_count = len(self.state_option)  # 4
#         self.action_count = len(self.action_option)  # 2
#
#         # 为状态建立从spin值到索引的映射
#         self.spin_to_index = {spin: i for i, spin in enumerate(self.state_option)}
#
#         # 环境参数
#         self.h = 0  # 外场
#         self.lam = 1  # 相互作用项lambda
#         self.temperature = 1 / beta
#
#         # PINN units (如果需要)
#         self.time_unit = 1
#         self.position_unit = 2
#         self.init_mf = None
#         self.dim = 1
#
#     def get_reward(self, state, action, mean_field):
#         # 根据当前动作选择的spin获取其实际值
#         a_j = self.action_option[int(action.val[0])]  # 动作对应的spin值(-1或+1)
#
#         # 计算平均磁化强度
#         m = np.dot(mean_field.val, self.state_option)
#
#         # r = h * a_j + lam * a_j * m
#         reward = self.h * a_j + self.lam * a_j * m
#         return Reward(reward=reward)
#
#     def dynamics(self, state, action, mean_field=None) -> State:
#         """
#         状态转移：
#         如果 is_original_dynamics == 0（确定性动力学）：
#           根据动作选择对应的下一个状态。例如动作=-1时选用spin=-1的状态, 动作=1时选用spin=1的状态。
#
#         如果 is_original_dynamics != 0（MCMC动力学）：
#           使用Metropolis准则决定是否接受 propose 的spin变换。
#         """
#         current_state_idx = int(state.val[0])
#         current_spin = self.state_option[current_state_idx]
#
#         a_j_proposed = self.action_option[int(action.val[0])]  # 提议动作spin(-1或+1)
#
#         if self.is_original_dynamics == 0:
#             # 确定性动力学：动作为-1则下一个状态为spin=-1对应的状态，
#             # 动作为1则下一个状态为spin=1对应的状态
#             # 我们这里简单定义：动作=-1 -> 下一个状态=spin=-1（索引为1）
#             # 动作=1 -> 下一个状态=spin=1（索引为2）
#             # （可根据需要修改逻辑）
#             if a_j_proposed == -1:
#                 next_spin = -1
#             else:
#                 next_spin = 1
#             next_state_index = self.spin_to_index[next_spin]
#             return State(state=next_state_index)
#         else:
#             # MCMC动力学
#             m = np.dot(mean_field.val, self.state_option)
#
#             E_current = -(self.h * current_spin + self.lam * current_spin * m)
#             E_proposed = -(self.h * a_j_proposed + self.lam * a_j_proposed * m)
#
#             delta_E = E_proposed - E_current
#             acceptance_prob = np.exp(-delta_E / self.temperature)
#
#             epsilon = np.random.uniform(0, 1)
#             if acceptance_prob > epsilon:
#                 next_spin = a_j_proposed
#             else:
#                 next_spin = current_spin
#
#             next_state_index = self.spin_to_index[next_spin]
#             return State(state=next_state_index)
#
#     def advance(self, policy, mean_field) -> MeanField:
#         next_mean_field = MeanField(mean_field=None, s=self.state_count)
#         next_mean_field.val = np.zeros(self.state_count)
#
#         for s in range(self.state_count):
#             for a in range(self.action_count):
#                 action_prob = policy.val[s, a]
#                 trans_probs = self.trans_prob(State(state=s), Action(action=a), mean_field)
#                 for nxt in range(self.state_count):
#                     next_mean_field.val[nxt] += mean_field.val[s] * action_prob * trans_probs[nxt]
#
#         # 归一化
#         total = np.sum(next_mean_field.val)
#         if total > 1e-12:
#             next_mean_field.val /= total
#         else:
#             next_mean_field.val = np.ones(self.state_count) / self.state_count
#
#         return next_mean_field
#
#     def trans_prob(self, state, action, mean_field) -> np.ndarray:
#         next_prob = np.zeros(self.state_count)
#
#         current_spin = self.state_option[int(state.val[0])]
#         a_j_proposed = self.action_option[int(action.val[0])]
#
#         if self.is_original_dynamics == 0:
#             # 确定性转移
#             if a_j_proposed == -1:
#                 next_spin = -1
#             else:
#                 next_spin = 1
#             next_state_index = self.spin_to_index[next_spin]
#             next_prob[next_state_index] = 1.0
#             return next_prob
#         else:
#             # MCMC转移概率
#             m = np.dot(mean_field.val, self.state_option)
#             E_current = -(self.h * current_spin + self.lam * current_spin * m)
#             E_proposed = -(self.h * a_j_proposed + self.lam * a_j_proposed * m)
#
#             delta_E = E_proposed - E_current
#             acceptance_prob = np.exp(-delta_E / self.temperature)
#
#             current_state_index = self.spin_to_index[current_spin]
#             proposed_state_index = self.spin_to_index[a_j_proposed]
#
#             next_prob[proposed_state_index] = acceptance_prob
#             next_prob[current_state_index] = 1 - acceptance_prob
#             return next_prob
#
#     def get_neighbors(self, state, mean_field=None):
#         return np.arange(self.state_count)

import numpy as np
import random
from core import State, Reward, Environment, MeanField, Action

class Env(Environment):
    def __init__(self, is_original_dynamics: int, beta: float, seed=None):
        super().__init__(is_original_dynamics, beta)
        self.name = 'ISINGNEW'

        # We have four possible spin values: -2, -1, 1, 2
        self.state_option = [-2, -1, 1, 2]
        # Two possible actions: -1 or +1
        self.action_option = [-1, 1]

        self.state_shape = 1
        self.action_shape = 1

        self.state_count = len(self.state_option)  # 4
        self.action_count = len(self.action_option)  # 2

        # spin -> index
        self.spin_to_index = {spin: i for i, spin in enumerate(self.state_option)}

        # Environment params
        self.h = 0   # external field
        self.lam = 1 # interaction coefficient
        self.temperature = 1.0 / beta  # used for Boltzmann factor

        # If you want reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.time_unit = 1
        self.position_unit = 2
        self.init_mf = None
        self.dim = 1

    def get_reward(self, state, action, mean_field):
        """
        reward = h * (proposed spin) + lam * (proposed spin)* m
        But here we simply compute the immediate reward from the action spin
        relative to the global mean_field magnetization
        """
        a_j = self.action_option[int(action.val[0])]  # proposed spin increment: -1 or +1
        # The spin "value" from action alone is either -1 or +1
        # But you might want to interpret or clamp it if needed.

        # The average magnetization (m)
        m = np.dot(mean_field.val, self.state_option)
        reward_val = self.h * a_j + self.lam * a_j * m
        return Reward(reward=reward_val)

    def dynamics(self, state, action, mean_field=None) -> State:
        """
        If is_original_dynamics == 0 => deterministic update
        else => Heat-Bath style update
        """
        s_idx = int(state.val[0])
        current_spin = self.state_option[s_idx]

        a_j = self.action_option[int(action.val[0])]  # -1 or +1

        if self.is_original_dynamics == 0:
            # Deterministic approach:
            # We do "moveSpinDeterministic": if a_j==-1 => spin-> -1, if a_j==+1 => spin-> +1
            # or a step from the current spin. For example:
            # If current spin = -2 and action=+1 => next spin = -1
            # (In short, a one-step movement in { -2, -1, 1, 2 }, with boundary checks)
            return self._move_spin_deterministic(current_spin, a_j)
        else:
            # Heat-Bath approach
            return self._move_spin_heatbath(current_spin, a_j, mean_field)

    def _move_spin_deterministic(self, curr_spin, action_val):
        # We treat the 4 spins as stepping by 1, ignoring boundary for simplicity or clamp
        # e.g., if curr_spin=-2 and action=+1 => next_spin=-1
        # if curr_spin=2 and action=+1 => stays 2, etc.
        next_spin = curr_spin + action_val

        # clamp within [-2, -1, 1, 2]
        # if result=0 => use -1 or 1 ? or skip
        # for simplicity if next_spin=0 => we push it to +1
        if next_spin == 0:
            next_spin = 1
        # clamp the extremes
        if next_spin < -2:
            next_spin = -2
        if next_spin > 2:
            next_spin = 2

        next_idx = self.spin_to_index[next_spin]
        return State(state=next_idx)

    def _move_spin_heatbath(self, curr_spin, action_val, mean_field):
        """
        Heat-Bath local update:
          We have two possible spins: current_spin or proposed_spin = (some step from current).
          We pick each with probability ~ exp(-beta* E(spin)).
        """
        # Step from current spin
        proposed_spin = curr_spin + action_val
        # clamp if needed
        if proposed_spin == 0:
            proposed_spin = 1
        if proposed_spin < -2:
            proposed_spin = -2
        if proposed_spin > 2:
            proposed_spin = 2

        # define energies
        # E(spin) = - spin * (h + lam * m)
        local_field = (self.h + self.lam * np.dot(mean_field.val, self.state_option))
        E_curr = - (curr_spin * local_field)
        E_prop = - (proposed_spin * local_field)

        # Probability that we pick proposed_spin
        # p_prop = exp(-beta E_prop) / [ exp(-beta E_prop) + exp(-beta E_curr) ]
        beta = 1.0 / self.temperature
        w_curr = np.exp(-beta * E_curr)
        w_prop = np.exp(-beta * E_prop)
        p_prop = w_prop / (w_curr + w_prop)

        eps = np.random.rand()
        if eps < p_prop:
            final_spin = proposed_spin
        else:
            final_spin = curr_spin

        final_idx = self.spin_to_index[final_spin]
        return State(state=final_idx)

    def advance(self, policy, mean_field) -> MeanField:
        """
        next_mean_field[s'] = sum_{s,a} mean_field[s]* policy[s,a]* trans_prob(s->s')
        """
        next_val = np.zeros(self.state_count)
        for s in range(self.state_count):
            for a_idx in range(self.action_count):
                prob_a = policy.val[s, a_idx]
                p_trans = self.trans_prob(State(state=s), Action(action=a_idx), mean_field)
                for s_next in range(self.state_count):
                    next_val[s_next] += mean_field.val[s] * prob_a * p_trans[s_next]

        total = next_val.sum()
        if total>1e-12:
            next_val /= total
        else:
            next_val = np.ones(self.state_count)/self.state_count

        next_mean_field = MeanField(mean_field=next_val, s=self.state_count)
        return next_mean_field

    def trans_prob(self, state, action, mean_field) -> np.ndarray:
        """
        Return the distribution over next states given (s,a).
        If is_original_dynamics==0 => deterministic
        If not => same Heat-Bath rule
        """
        s_idx = int(state.val[0])
        curr_spin = self.state_option[s_idx]
        a_j = self.action_option[int(action.val[0])]
        next_prob = np.zeros(self.state_count)

        if self.is_original_dynamics==0:
            # same logic as _move_spin_deterministic
            next_s = self._move_spin_deterministic(curr_spin,a_j).val[0]
            next_prob[next_s] = 1.0
        else:
            # Heat-Bath distribution
            # We have two possible outcomes: remain curr_spin or adopt proposed_spin
            proposed_spin = curr_spin + a_j
            if proposed_spin==0:
                proposed_spin=1
            if proposed_spin<-2:
                proposed_spin=2
            if proposed_spin>2:
                proposed_spin=-2

            local_field = self.h + self.lam * np.dot(mean_field.val, self.state_option)
            E_curr = -(curr_spin*local_field)
            E_prop = -(proposed_spin*local_field)
            beta = 1.0/self.temperature
            w_curr = np.exp(-beta*E_curr)
            w_prop = np.exp(-beta*E_prop)
            p_prop = w_prop/(w_curr+w_prop)

            idx_curr = self.spin_to_index[curr_spin]
            idx_prop = self.spin_to_index[proposed_spin]

            next_prob[idx_prop] = p_prop
            next_prob[idx_curr] = 1.0 - p_prop

        return next_prob


    def get_neighbors(self, state, mean_field=None):
        """
        Return all states s' (as integer indices) from which we can arrive at `state`
        in exactly one time step, via some action.

        'state' is an index in [0,1,2,3]. We want all s' in [0,1,2,3] s.t.
          dynamics(s', a, mean_field) = state  (deterministic or MCMC)
        or equivalently trans_prob(s', a, mean_field)[ state ] > 0.
        We'll just loop over s' in range(state_count) & over each action,
        and check if there's a non-zero probability for next= state.
        """
        s_target = int(state)  # ensure it's int
        neighbors = []
        for s_prime in range(self.state_count):
            for a_idx in range(self.action_count):
                # Get transition distribution from s_prime under action a_idx
                distr = self.trans_prob(State(state=s_prime),
                                        Action(action=a_idx), mean_field)
                if distr[s_target] > 1e-12:
                    # means s_prime can lead to s_target with some prob
                    neighbors.append(s_prime)
                    break  # no need to check other actions once we found a match

        return np.array(neighbors, dtype=int)

