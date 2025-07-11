import torch
import functools
from typing import Tuple


def metric_ensure_same_device(func):
    """
    A decorator that ensures `y_true` and `y_pred` are on the same device
    before calling the decorated metric function.
    
    It assumes the first two arguments of the decorated function are 
    `y_true` and `y_pred`.
    
    It assumes `y_pred` is on the target device. 
    It also looks for a `device` keyword argument to allow for user override.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract y_true and y_pred from the function's arguments
        y_true, y_pred = args[0], args[1]
        
        # Extract the optional 'device' keyword argument
        device = kwargs.get('device')
        
        # Determine target device
        if device is None:
            target_device = y_pred.device
        else:
            target_device = torch.device(device)
            
        # Move tensors if necessary
        y_true_on_device = y_true.to(target_device)
        y_pred_on_device = y_pred.to(target_device)
        
        # Reconstruct the arguments to pass to the original function
        new_args = (y_true_on_device, y_pred_on_device) + args[2:]
        
        # Call the original metric function with the corrected tensors
        return func(*new_args, **kwargs)
        
    return wrapper


@metric_ensure_same_device
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    
    y_true_bool = y_true.bool()
    y_pred_bool = y_pred.bool()

    tp = (y_true_bool & y_pred_bool).sum().item()
    fp = (~y_true_bool & y_pred_bool).sum().item()
    tn = (~y_true_bool & ~y_pred_bool).sum().item()
    fn = (y_true_bool & ~y_pred_bool).sum().item()
    
    return tp, fp, tn, fn


@metric_ensure_same_device
def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    return (y_true == y_pred).float().mean().item()


def precision(y_true: torch.Tensor, y_pred: torch.Tensor):
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


@metric_ensure_same_device
def precision_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int):
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."

    if k == 0:
        return 0.0
    
    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / k).item()


@metric_ensure_same_device
def average_precision(y_true: torch.Tensor, y_scores: torch.Tensor):
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."

    total_y_true = torch.sum(y_true)
    if total_y_true == 0:
        return 0.0

    _, sorted_indices = torch.sort(y_scores, descending=True)
    sorted_y_true = y_true[sorted_indices]
    tp_cumul = torch.cumsum(sorted_y_true, dim=-1)
    ranks = torch.arange(1, len(y_true) + 1)
    precision_at_steps = tp_cumul / ranks
    precisions_at_hits = precision_at_steps * sorted_y_true

    return torch.sum(precisions_at_hits) / total_y_true.float()


def recall(y_true: torch.Tensor, y_pred: torch.Tensor):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


@metric_ensure_same_device
def recall_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int):
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."

    if k == 0:
        return 0.0
    
    total_y_true = torch.sum(y_true)
    if total_y_true == 0:
        return 0.0
    
    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return num_hits_at_k / total_y_true.float()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)
    
    precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision_value + recall_value

    return 2 * precision_value * recall_value / denom if denom > 0 else 0.0


def best_f1_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> Tuple[float, float]:
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."

    best_f1 = 0.0
    thresholds = torch.unique(y_scores).cpu()
    best_thr = thresholds[0].item() if len(thresholds) > 0 else 0.0

    for thr in thresholds:
        y_pred = y_scores >= thr
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr.item()

    return best_f1, best_thr


@metric_ensure_same_device
def mean_reciprocal_rank(y_true: torch.Tensor, y_scores: torch.Tensor):
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."
    y_true_bool = y_true.to(torch.bool)

    sorted_indices = torch.argsort(y_scores, descending=True)
    true_anomaly_indices = torch.where(y_true_bool)[0]

    if len(true_anomaly_indices) == 0:
        return 0.0

    ranks = torch.zeros(len(y_true_bool), dtype=torch.int32)
    ranks[sorted_indices] = torch.arange(1, len(y_true_bool) + 1, dtype=torch.int32)
    anomaly_ranks = ranks[y_true_bool]

    return torch.mean(1.0 / anomaly_ranks.float())    
