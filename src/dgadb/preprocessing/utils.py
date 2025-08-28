import torch
from torch.types import Device


def cartesian_sample(tensors: list[torch.Tensor], num_samples: int, device: Device = None):
    device = device or tensors[0].device
    sizes = torch.tensor([t.size(0) for t in tensors], device=device)
    product_size = int(torch.prod(sizes).item())

    idx = torch.randint(0, product_size, (num_samples,), device=device)

    # strides[i] = product(sizes[i+1:])
    strides = torch.cumprod(
        torch.cat((sizes[1:], torch.tensor([1], device=device))), dim=0)

    # coords[:, i] = (idx // strides[i]) % sizes[i]
    coords = (idx.unsqueeze(1) // strides) % sizes

    return torch.stack([t[coords[:, i]] for i, t in enumerate(tensors)], dim=-1)


def compute_unique_inverse_count_probabilities(
    items: torch.Tensor, power: float = -1.5, device: Device = None
) -> torch.Tensor:
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
    unique_mask = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=device),
         inverse_indices_sorted[1:] != inverse_indices_sorted[:-1]]
    )
    first_occurrence_indices = perm_sorted[unique_mask]

    return unique_t, first_occurrence_indices


def to_canonical(src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edges = torch.stack([src, tgt])
    canonical_edges, _ = torch.sort(edges, dim=0)
    return canonical_edges[0], canonical_edges[1]


def to_undirected(src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_indices = torch.arange(src.numel(), device=src.device)
    combined_src = torch.cat([src, tgt])
    combined_tgt = torch.cat([tgt, src])
    combined_indices = torch.cat([original_indices, original_indices])

    canonical_src, canonical_tgt = to_canonical(combined_src, combined_tgt)
    max_node_id = canonical_tgt.max()
    canonical_edge_ids = canonical_src * max_node_id + canonical_tgt

    _, first_occurrence_indices = unique_with_indices(canonical_edge_ids)
    unique_u = canonical_src[first_occurrence_indices]
    unique_v = canonical_tgt[first_occurrence_indices]

    indices_mapping = combined_indices[first_occurrence_indices]

    final_src = torch.cat([unique_u, unique_v])
    final_tgt = torch.cat([unique_v, unique_u])
    final_indices_mapping = torch.cat([indices_mapping, indices_mapping])

    return final_src, final_tgt, final_indices_mapping


def reindex_nodes(src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if src.numel() == 0:
        return src, tgt, torch.tensor([], dtype=src.dtype, device=src.device)

    unique_nodes = torch.unique(torch.cat([src, tgt]))
    num_nodes = unique_nodes.numel()

    max_id = unique_nodes.max()
    if max_id == num_nodes - 1:
        # Already contiguous
        return src, tgt, unique_nodes

    mapping = torch.full(size=(max_id + 1,), fill_value=-1,
                         dtype=torch.long, device=src.device)
    mapping[unique_nodes] = torch.arange(num_nodes, device=src.device)

    new_src = mapping[src]
    new_tgt = mapping[tgt]

    # Also return original unique node IDs for reverse mapping
    return new_src, new_tgt, unique_nodes
