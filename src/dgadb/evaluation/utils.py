import numpy as np
import torch
from functools import wraps


def check_matching_device(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        y_true, y_pred, *_ = args
        device = kwargs.get('device')

        if device is not None:
            target_device = torch.device(device)
        elif isinstance(y_pred, torch.Tensor):
            target_device = y_pred.device
        elif isinstance(y_true, torch.Tensor):
            target_device = y_true.device
        else:
            target_device = torch.device("cpu")

        if (
            not isinstance(y_true, (torch.Tensor, np.ndarray))
            or not isinstance(y_pred, (torch.Tensor, np.ndarray))
        ):
            raise TypeError(
                "inputs must be either torch.Tensor or np.ndarray.")

        y_true_on_device = torch.as_tensor(y_true, device=target_device)
        y_pred_on_device = torch.as_tensor(y_pred, device=target_device)
        new_args = (y_true_on_device, y_pred_on_device) + args[2:]

        return func(*new_args, **kwargs)

    return wrapper


def check_matching_shapes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        y_true, y_pred, *_ = args
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shapes of true labels and predictions must match, "
                f"got: {y_true.shape} != {y_pred.shape}"
            )

        return func(*args, **kwargs)

    return wrapper
