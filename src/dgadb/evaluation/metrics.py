import torch
import numpy as np
from .utils import check_matching_shapes

from sklearn.metrics import precision_recall_curve


def max_f1_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple[float, float, float, float]:
    """Find the threshold that maximises F1 using sklearn's precision-recall curve.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.

    Returns:
        Tuple of (max_f1, best_threshold, precision_at_best, recall_at_best).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(
        numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1_score_index = np.argmax(f1_scores)
    return (f1_scores[max_f1_score_index], thresholds[max_f1_score_index],
            precision[max_f1_score_index], recall[max_f1_score_index])


@check_matching_shapes
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[int, int, int, int]:
    """Compute TP, FP, TN, FN counts.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Tuple of (tp, fp, tn, fn) counts.
    """
    y_true_bool = y_true.bool()
    y_pred_bool = y_pred.bool()

    tp = int((y_true_bool & y_pred_bool).sum().item())
    fp = int((~y_true_bool & y_pred_bool).sum().item())
    tn = int((~y_true_bool & ~y_pred_bool).sum().item())
    fn = int((y_true_bool & ~y_pred_bool).sum().item())

    return tp, fp, tn, fn


@check_matching_shapes
def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Fraction of edges classified correctly.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Accuracy score in [0, 1].
    """
    return (y_true == y_pred).float().mean().item()


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Precision (TP / (TP + FP)); returns 0.0 when the denominator is zero.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Precision score in [0, 1].
    """
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


@check_matching_shapes
def precision_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    """Fraction of the top-k scored edges that are truly anomalous.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.
        k: Number of top-scored edges to consider.

    Returns:
        Precision@k in [0, 1].
    """
    if k == 0:
        return 0.0

    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / k).item()


@check_matching_shapes
def average_precision(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Area under the precision-recall curve computed manually from score ranks.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.

    Returns:
        Average precision in [0, 1]; 0.0 if there are no positive labels.
    """
    total_y_true = torch.sum(y_true)
    if total_y_true == 0:
        return 0.0

    _, sorted_indices = torch.sort(y_scores, descending=True)
    sorted_y_true = y_true[sorted_indices]
    tp_cumul = torch.cumsum(sorted_y_true, dim=-1)
    ranks = torch.arange(1, len(y_true) + 1, device=y_true.device)
    precision_at_steps = tp_cumul / ranks
    precisions_at_hits = precision_at_steps * sorted_y_true

    return (torch.sum(precisions_at_hits) / total_y_true.float()).item()


def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Recall (TP / (TP + FN)); returns 0.0 when the denominator is zero.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Recall score in [0, 1].
    """
    tp, _, _, fn = confusion_matrix(y_true, y_pred)

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


@check_matching_shapes
def recall_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    """Fraction of all anomalous edges captured in the top-k scored edges.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.
        k: Number of top-scored edges to consider.

    Returns:
        Recall@k in [0, 1]; 0.0 when ``k == 0`` or there are no positive labels.
    """
    if k == 0:
        return 0.0

    total_y_true = torch.sum(y_true)
    if total_y_true == 0:
        return 0.0

    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / total_y_true.float()).item()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Harmonic mean of precision and recall; returns 0.0 when both are zero.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        F1 score in [0, 1].
    """
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)

    precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision_value + recall_value

    return 2 * precision_value * recall_value / denom if denom > 0 else 0.0


@check_matching_shapes
def best_f1_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple[float, float]:
    """Find the threshold that maximises F1 using a cumulative TP/FP scan.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.

    Returns:
        Tuple of (best_f1, best_threshold). ``best_threshold`` is ``nan``
        if there are no positive labels.
    """
    y_true = y_true.float()

    desc_score_indices = torch.argsort(y_scores, descending=True)
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    tp_cumsum = torch.cumsum(y_true, dim=0)
    fp_cumsum = torch.cumsum(1 - y_true, dim=0)

    total_positives = torch.sum(y_true)

    if total_positives == 0:
        return 0.0, float('nan')

    recall_vec = tp_cumsum / total_positives
    precision_vec = tp_cumsum / (tp_cumsum + fp_cumsum)

    f1_vec = (2 * precision_vec * recall_vec) / \
        (precision_vec + recall_vec + 1e-12)

    best_f1, best_idx = torch.max(f1_vec, dim=0)

    if best_idx < len(y_scores) - 1:
        best_thr = (y_scores[best_idx] + y_scores[best_idx + 1]) / 2
    else:
        best_thr = y_scores[best_idx]

    return best_f1.item(), best_thr.item()


@check_matching_shapes
def mean_reciprocal_rank(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Mean reciprocal rank of anomalous edges when scored in descending order.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted anomaly scores.

    Returns:
        MRR in [0, 1]; 0.0 if there are no positive labels.
    """
    y_true_bool = y_true.to(torch.bool)

    sorted_indices = torch.argsort(y_scores, descending=True)

    if not torch.any(y_true_bool):
        return 0.0

    ranks = torch.zeros(len(y_true_bool), dtype=torch.int32,
                        device=y_true.device)
    ranks[sorted_indices] = torch.arange(
        1, len(y_true_bool) + 1, dtype=torch.int32, device=y_true.device)
    anomaly_ranks = ranks[y_true_bool]

    return torch.mean(1.0 / anomaly_ranks.float()).item()
