import torch
from .utils import check_matching_shapes


@check_matching_shapes
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[int, int, int, int]:
    y_true_bool = y_true.bool()
    y_pred_bool = y_pred.bool()

    tp = int((y_true_bool & y_pred_bool).sum().item())
    fp = int((~y_true_bool & y_pred_bool).sum().item())
    tn = int((~y_true_bool & ~y_pred_bool).sum().item())
    fn = int((y_true_bool & ~y_pred_bool).sum().item())

    return tp, fp, tn, fn


@check_matching_shapes
def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


@check_matching_shapes
def precision_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    if k == 0:
        return 0.0

    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / k).item()


@check_matching_shapes
def average_precision(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
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
    tp, _, _, fn = confusion_matrix(y_true, y_pred)

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


@check_matching_shapes
def recall_at_k(y_true: torch.Tensor, y_scores: torch.Tensor, k: int) -> float:
    if k == 0:
        return 0.0

    total_y_true = torch.sum(y_true)
    if total_y_true == 0:
        return 0.0

    _, top_k_indices = torch.topk(y_scores, k)
    num_hits_at_k = torch.sum(y_true[top_k_indices])

    return (num_hits_at_k / total_y_true.float()).item()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)

    precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision_value + recall_value

    return 2 * precision_value * recall_value / denom if denom > 0 else 0.0


@check_matching_shapes
def best_f1_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple[float, float]:
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
