import torch
from torch.types import Device


def cartesian_sample(tensors: list[torch.Tensor], num_samples: int, device: Device = None):
    device = device or tensors[0].device
    sizes = torch.tensor([t.size(0) for t in tensors], device=device)
    product_size = int(torch.prod(sizes).item())

    idx = torch.randint(0, product_size, (num_samples,), device=device)

    # strides[i] = product(sizes[i+1:])
    strides = torch.cumprod(
        torch.cat((sizes[1:], torch.tensor([1], device=device))),
        dim=0
    )

    # coords[:, i] = (idx // strides[i]) % sizes[i]
    coords = (idx.unsqueeze(1) // strides) % sizes

    return torch.stack([t[coords[:, i]] for i, t in enumerate(tensors)], dim=-1)


def compute_unique_inverse_count_probabilities(items: torch.Tensor, power: float = -1.5, device: Device = None) -> torch.Tensor:
    _, inverse_indices, counts = torch.unique(
        items, dim=0, return_inverse=True, return_counts=True)
    unique_weights = torch.pow(counts, power)
    probs = unique_weights / (unique_weights.sum() + 1e-10)
    mapped_probs = probs[inverse_indices]
    if device is not None:
        mapped_probs = mapped_probs.to(device)

    return mapped_probs


def unique_with_indices(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = t.device
    unique_t, inverse_indices = torch.unique(t, return_inverse=True)
    inverse_indices_sorted, perm_sorted = torch.sort(inverse_indices)
    unique_mask = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        inverse_indices_sorted[1:] != inverse_indices_sorted[:-1]
    ])
    first_occurrence_indices = perm_sorted[unique_mask]

    return unique_t, first_occurrence_indices
