import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    
    y_true_bool = y_true.bool()
    y_pred_bool = y_pred.bool()

    tp = (y_true_bool & y_pred_bool).sum().item()
    fp = (~y_true_bool & y_pred_bool).sum().item()
    tn = (~y_true_bool & ~y_pred_bool).sum().item()
    fn = (y_true_bool & ~y_pred_bool).sum().item()
    
    return tp, fp, tn, fn


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    return (y_true == y_pred).float().mean().item()


def precision(y_true: torch.Tensor, y_pred: torch.Tensor):
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def precision_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int):
    assert y_true.shape == y_scores.shape, "Shapes of true labels and scores must match."

    if k == 0:
        return 0.0
    
    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / k).item()


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
