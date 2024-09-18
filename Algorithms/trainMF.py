import numpy as np
import torch
from torch import optim


def train_mean_field_dim_1_new(max_epoch: int, learning_rate: float, max_grad_norm: float, num_of_units: int,
                               mean_field_model, optimizer1):
    optimizer1 = optim.Adam(mean_field_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Prepare the states and times tensors
    states_all = []
    times_all = []

    for trajectory in self.data_policy_theta:
        states = trajectory.states
        num_steps = len(states)
        time_steps = np.arange(num_steps)

        # Extend states and times lists
        states_all.extend(states * self.env.time_unit)
        times_all.extend(time_steps * self.env.position_unit)

    # Convert lists to tensors
    states_tensor = torch.tensor(states_all, dtype=torch.float, device=self.device).view(-1, 1)
    times_tensor = torch.tensor(times_all, dtype=torch.float, device=self.device).view(-1, 1)

    # Compute indices
    s_indices = (states_tensor.squeeze() / self.env.position_unit).long()
    t_indices = (times_tensor.squeeze() / self.env.time_unit).long()

    # Create masks
    t_zero_mask = (t_indices == 0)
    t_nonzero_mask = ~t_zero_mask

    # Convert state_option to a tensor
    state_option_tensor = torch.tensor(self.env.state_option, dtype=torch.float, device=self.device)

    for epoch in range(max_epoch):
        optimizer1.zero_grad()

        # Initialize mean_field_values tensor
        mean_field_values = torch.zeros(states_tensor.size(0), device=self.device, dtype=torch.float)

        # Handle t = 0 cases
        mean_field_values[t_zero_mask] = torch.tensor(
            self.env.init_mf.val, device=self.device, dtype=torch.float
        )[s_indices[t_zero_mask]]

        # Handle t > 0 cases
        if t_nonzero_mask.any():
            # Prepare inputs for mean_field_model
            x_onehot = state_option_tensor[s_indices[t_nonzero_mask]]
            t_onehot = (t_indices[t_nonzero_mask] - 1).float().unsqueeze(1)

            x_last_onehot = state_option_tensor[(s_indices[t_nonzero_mask] - 1) % self.env.state_count]
            t_last_onehot = (t_indices[t_nonzero_mask] - 1).float().unsqueeze(1)

            # Compute mean fields
            mean_field_last_x_t = mean_field_model(x_last_onehot, t_last_onehot).squeeze()
            mean_field_last_t = mean_field_model(x_onehot, t_onehot).squeeze()

            # Get last policies
            last_policy_indices = (t_indices[t_nonzero_mask] - 1).cpu().numpy()
            last_policy = torch.tensor(
                self.p_flow.val[last_policy_indices],
                device=self.device, dtype=torch.float
            )

            # Get action indices
            action_indices_last_t = torch.argmax(last_policy, dim=1)[s_indices[t_nonzero_mask]]
            action_indices_last_x_t = torch.argmax(last_policy, dim=1)[
                (s_indices[t_nonzero_mask] - 1) % self.env.state_count]

            # Compute velocities
            action_option_tensor = torch.tensor(self.env.action_option, device=self.device)
            velocity_last_t = torch.norm(action_option_tensor[action_indices_last_t], dim=1)
            velocity_last_x_t = torch.norm(action_option_tensor[action_indices_last_x_t], dim=1)

            # Compute mean field values for t > 0
            mean_field_values[t_nonzero_mask] = (
                    mean_field_last_t +
                    mean_field_last_x_t * velocity_last_x_t -
                    mean_field_last_t * velocity_last_t
            )

        # Compute predictions
        preds = mean_field_model(states_tensor, times_tensor).squeeze()

        # Compute residual loss
        residual = (mean_field_values - preds).abs().mean()

        # Backpropagation and optimization
        residual.backward()
        torch.nn.utils.clip_grad_norm_(mean_field_model.parameters(), max_grad_norm)
        optimizer1.step()

        # Logging
        print(f'Mean_Field: epoch: {epoch}, loss: {residual.item():.4f}', end='\r')
        self.logger.info(f'Mean_Field: Epoch {epoch + 1}, Sample Loss: {residual.item():.4f}')
        print()

    return mean_field_model