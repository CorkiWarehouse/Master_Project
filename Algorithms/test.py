def recover_ermfne(self, expert_mf_flow) -> [MeanFieldFlow, PolicyFlow]:
    assert self.reward_model is not None
    mf_flow = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
    p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
    lambda_ = 10.0  # 惩罚项强度（根据需要调整）

    for _ in range(MAX):
        p_flow = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)
        q_values = PolicyFlow(policy_flow=None, s=self.env.state_count, t=self.horizon, a=self.env.action_count)

        for s in range(self.env.state_count):
            p_flow.val[self.horizon - 1, s, :] = (
                np.array([1 / self.env.action_count for _ in range(self.env.action_count)]))

        for t in reversed(range(0, self.horizon - 1)):
            for s_current in range(0, self.env.state_count):
                for a_current in range(0, self.env.action_count):
                    original_reward = self.reward_model(
                        torch.tensor(self.env.state_option[s_current]).to(self.device, torch.float),
                        torch.tensor(self.env.action_option[a_current]).to(self.device, torch.float),
                        torch.from_numpy(mf_flow.val[t]).to(self.device, torch.float)
                    ).detach().cpu().numpy()

                    divergence_penalty = self.compute_divergence_penalty(
                        mf_flow.val[t],
                        expert_mf_flow.val[t]
                    )

                    q_values.val[t, s_current, a_current] += original_reward - lambda_ * divergence_penalty

                    for s_next in range(0, self.env.state_count):
                        trans_prob = self.env.trans_prob(
                            State(state=s_current),
                            Action(action=a_current),
                            MeanField(mean_field=mf_flow.val[t])
                        )[s_next]

                        entropy_term = self.env.beta * np.sum(entr(p_flow.val[t + 1, s_next, :]))
                        q_values.val[t, s_current, a_current] += trans_prob * entropy_term

                        for a_next in range(0, self.env.action_count):
                            q_values.val[t, s_current, a_current] += (
                                    trans_prob * p_flow.val[t + 1, s_next, a_next] * q_values.val[t + 1, s_next, a_next]
                            )

            for s in range(0, self.env.state_count):
                partition = 0.0
                for a in range(0, self.env.action_count):
                    policy_numerator = np.exp(q_values.val[t, s, a] / self.env.beta)
                    adjusted_numerator = policy_numerator * np.exp(-lambda_ * divergence_penalty)
                    partition += adjusted_numerator
                for a in range(0, self.env.action_count):
                    p_flow.val[t, s, a] = adjusted_numerator / partition

        mf_flow_next = MeanFieldFlow(mean_field_flow=None, s=self.env.state_count, t=self.horizon)
        mf_flow_next.val[0] = mf_flow.val[0, :]
        for t in range(1, self.horizon):
            mf = self.env.advance(Policy(policy=p_flow.val[t - 1]), MeanField(mean_field=mf_flow.val[t - 1]))
            mf_flow_next.val[t] = mf.val

        divergence = self.compute_divergence_penalty(mf_flow_next.val, expert_mf_flow.val)
        if divergence < MIN:
            break
        else:
            mf_flow = mf_flow_next

    self.mf_flow = mf_flow
    self.p_flow = p_flow
    return [mf_flow, p_flow]
