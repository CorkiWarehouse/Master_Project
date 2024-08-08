import torch


def prepare_tensors(state=None, action=None, mf=None, time=None):
    """
    Convert inputs to PyTorch tensors with appropriate shapes, and return them in a specific order.
    Inputs can be None, in which case None is returned in their place.

    Args:
        state: The state input which can be a list, scalar, or tensor. Default is None.
        action: The action input which can be a list, scalar, or tensor. Default is None.
        mf: The mean-field input which can be a list, scalar, or tensor. Default is None.
        time: The time input which can be a list, scalar, or tensor. Default is None.

    Returns:
        tuple: A tuple of tensors or None, corresponding to state, action, mf, and time in that order.
    """
    inputs = {'state': state, 'action': action, 'mf': mf, 'time': time}
    results = []
    for key in ['state', 'action', 'mf', 'time']:
        input_data = inputs[key]
        if input_data is not None:
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            if input_data.dim() == 0:
                input_data = input_data.unsqueeze(0)  # Convert scalar to 1D tensor
            tensor = input_data
            results.append(tensor)
        else:
            results.append(None)

    return tuple(results)

# Example usage with state and mf provided, expecting to unpack state and mf directly
state_input = [1, 2, 3]
mf_input = [0.1, 0.2, 0.3]
state_tensor, action_tensor, mf_tensor, time_tensor = prepare_tensors(state=state_input, mf=mf_input)

# Print to see the result
print('State tensor:', state_tensor)
print('Action tensor:', action_tensor)  # This should print None
print('MF tensor:', mf_tensor)
print('Time tensor:', time_tensor)  # This should print None

