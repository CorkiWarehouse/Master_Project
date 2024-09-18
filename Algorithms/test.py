def train_mean_field_dim_1_new(self, max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int):
    mean_field_model = MeanFieldModel(state_shape=self.env.state_shape, time_horizon=1, num_of_units=num_of_units).to(self.device)
    optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate)

    mean_field_values = []
    states = []

    for t in range(self.horizon):
        for s in range(self.env.state_count):
            states.append([s * self.env.position_unit, t * self.env.time_unit])
            if t == 0:
                # Convert the initial value to a tensor of shape [1] to match the other tensors
                mean_field_values.append(torch.tensor([self.env.init_mf.val[s]], dtype=torch.float, device=self.device))
            else:
                x_onehot = torch.tensor(self.env.state_option[s], device=self.device, dtype=torch.float)
                t_onehot = torch.tensor(t, device=self.device, dtype=torch.float)
                x_last_onehot = torch.tensor(self.env.state_option[(s - 1) % self.env.state_count], device=self.device, dtype=torch.float)
                t_last_onehot = torch.tensor((t - 1) % self.horizon, device=self.device, dtype=torch.float)

                x_onehot.requires_grad_(True)
                t_onehot.requires_grad_(True)
                x_last_onehot.requires_grad_(True)
                t_last_onehot.requires_grad_(True)

                mean_field_last_x_t = mean_field_model(x_last_onehot, t_last_onehot)
                mean_field_last_t = mean_field_model(x_onehot, t_last_onehot)

                last_policy = self.p_flow.val[t-1, :]
                action_index_last_x_t = np.argmax(last_policy[int((s-1) % self.env.state_count)])
                action_index_last_t = np.argmax(last_policy[s])

                velocity_last_t = np.linalg.norm(self.env.action_option[action_index_last_t])
                velocity_last_x_t = np.linalg.norm(self.env.action_option[action_index_last_x_t])

                # Ensure the result is a tensor of shape [1]
                value = mean_field_last_t + mean_field_last_x_t * velocity_last_x_t - mean_field_last_t * velocity_last_t
                mean_field_values.append(value.view(1))

    # Stack all tensors into a single tensor
    states_tensor = torch.tensor(states, dtype=torch.float, device=self.device)
    mean_field_values_tensor = torch.stack(mean_field_values)

    for epoch in range(max_epoch * 10):
        preds = torch.reshape(mean_field_model(states_tensor), (1, -1))
        residual = (mean_field_values_tensor - preds).abs().mean()
        optimizer1.zero_grad()
        residual.backward()
        torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
        optimizer1.step()

        print(f'Mean_Field: epoch: {epoch + 1}, loss: {residual.item():.4f}', end='\r')
        self.logger.info(f'Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')

    print()  # for better formatting of print output
    self.mean_field_model = mean_field_model
