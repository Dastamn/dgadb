import copy
import torch
from typing import Optional
from src.dgadb.storage import TemporalGraph
from src.dgadb.preprocessing.utils import to_canonical


@torch.no_grad()
def inject_anomalies_addgraph_style(
    tg: TemporalGraph,
    anom_train_ratio: float = 0.0,
    anom_val_ratio: float = 0.0,
    anom_test_ratio: float = 0.10,
    noise_ratio: float = 0.0,        # flips a fraction of TRAIN labels, like the original
    seed: int = 1,
) -> TemporalGraph:

    device = tg.src.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # ----- Build observed undirected edge set from ALL edges (prevents collisions)
    src_all, tgt_all = to_canonical(tg.src, tg.tgt)
    all_edges = torch.stack([src_all, tgt_all], dim=1)
    uniq_edges = torch.unique(all_edges, dim=0).cpu().tolist()
    observed = set((u, v) for u, v in uniq_edges)

    # Number of nodes (assumes 0..N-1 or at least dense-ish ids)
    n = int(torch.max(torch.maximum(tg.src, tg.tgt)).item()) + 1
    fdim: Optional[int] = tg.msg.size(1) if tg.msg is not None else None

    def _make_split(mask: torch.Tensor, ratio: float):
        if ratio <= 0.0 or mask is None or not mask.any():
            return None

        e_src = tg.src[mask]
        e_tgt = tg.tgt[mask]
        e_t = tg.t[mask]
        e_msg = tg.msg[mask] if tg.msg is not None else None

        m = e_src.numel()
        num_anom = int(m * ratio)
        if num_anom == 0:
            labels = torch.ones(m, dtype=torch.long, device=device)
            return e_src, e_tgt, e_t, e_msg, labels

        fake_pairs = []
        tries = 0
        while len(fake_pairs) < num_anom and tries < 20:
            k = max(4 * (num_anom - len(fake_pairs)), num_anom)
            u = torch.randint(0, n, (k,), generator=gen)
            v = torch.randint(0, n, (k,), generator=gen)
            keep = (u != v)
            u, v = u[keep], v[keep]
            u, v = torch.minimum(u, v), torch.maximum(u, v)
            cand = torch.unique(torch.stack([u, v], dim=1), dim=0).tolist()
            for (a, b) in cand:
                if (a, b) not in observed:
                    fake_pairs.append((a, b))
                    if len(fake_pairs) == num_anom:
                        break
            tries += 1

        if len(fake_pairs) < num_anom:
            num_anom = len(fake_pairs)

        fake = torch.tensor(fake_pairs[:num_anom],
                            dtype=tg.src.dtype, device=device)

        total = m + num_anom
        labels = torch.ones(total, dtype=torch.long,
                            device=device)  # 1 = normal
        anom_pos = torch.randperm(total, device=device)[:num_anom]
        labels[anom_pos] = 0  # 0 = anomaly

        # output containers
        out_src = torch.empty(total, dtype=tg.src.dtype, device=device)
        out_tgt = torch.empty(total, dtype=tg.tgt.dtype, device=device)
        out_t = torch.empty(total, dtype=tg.t.dtype,   device=device)
        out_msg = None
        if fdim is not None:
            out_msg = torch.empty(
                total, fdim, dtype=tg.msg.dtype, device=device)

        # fill normals sequentially in remaining slots
        idx_norm = (labels == 1).nonzero(as_tuple=False).squeeze(1)
        out_src[idx_norm] = e_src
        out_tgt[idx_norm] = e_tgt
        out_t[idx_norm] = e_t
        if out_msg is not None:
            out_msg[idx_norm] = e_msg

        # fill anomalies sequentially
        idx_anom = anom_pos
        out_src[idx_anom] = fake[:, 0]
        out_tgt[idx_anom] = fake[:, 1]
        # borrow timestamps & edge features from random in-split edges (time-agnostic in original)
        rand_idx = torch.randint(0, m, (num_anom,), device=device)
        out_t[idx_anom] = e_t[rand_idx]
        if out_msg is not None:
            out_msg[idx_anom] = e_msg[rand_idx]

        return out_src, out_tgt, out_t, out_msg, labels

    train_pack = _make_split(tg.train_mask, anom_train_ratio)
    val_pack = _make_split(
        tg.val_mask if tg.val_mask is not None else None, anom_val_ratio)
    test_pack = _make_split(tg.test_mask, anom_test_ratio)

    def _pass_through(mask: torch.Tensor):
        e_src = tg.src[mask]
        e_tgt = tg.tgt[mask]
        e_t = tg.t[mask]
        e_msg = tg.msg[mask] if tg.msg is not None else None
        labels = torch.ones(e_src.numel(), dtype=torch.long, device=device)
        return e_src, e_tgt, e_t, e_msg, labels

    if train_pack is None and tg.train_mask.any():
        train_pack = _pass_through(tg.train_mask)
    if tg.val_mask is not None and val_pack is None and tg.val_mask.any():
        val_pack = _pass_through(tg.val_mask)
    if test_pack is None and tg.test_mask.any():
        test_pack = _pass_through(tg.test_mask)

    parts = [p for p in [train_pack, val_pack, test_pack] if p is not None]
    new_src = torch.cat([p[0] for p in parts])
    new_tgt = torch.cat([p[1] for p in parts])
    new_t = torch.cat([p[2] for p in parts])
    if fdim is not None:
        new_msg = torch.cat([p[3] for p in parts])
    else:
        new_msg = tg.msg  # keep as-is (unused by RustGraph)
    # 1=normal, 0=anomaly (original)
    new_lab = torch.cat([p[4] for p in parts])

    sizes = [p[0].numel() for p in parts]
    split_names = []
    if train_pack is not None:
        split_names.append("train")
    if val_pack is not None:
        split_names.append("val")
    if test_pack is not None:
        split_names.append("test")

    split_masks = []
    for name, sz in zip(split_names, sizes):
        m = torch.zeros(sz, dtype=torch.bool, device=device)
        m[:] = True
        split_masks.append((name, m))

    train_mask = torch.zeros_like(new_lab, dtype=torch.bool)
    val_mask = torch.zeros_like(
        new_lab, dtype=torch.bool) if tg.val_mask is not None else None
    test_mask = torch.zeros_like(new_lab, dtype=torch.bool)

    ofs = 0
    for (name, m), sz in zip(split_masks, sizes):
        if name == "train":
            train_mask[ofs:ofs+sz] = m
        elif name == "val" and val_mask is not None:
            val_mask[ofs:ofs+sz] = m
        else:  # test
            test_mask[ofs:ofs+sz] = m
        ofs += sz

    if noise_ratio > 0 and train_mask.any():
        k = int(noise_ratio * train_mask.sum().item())
        if k > 0:
            idx = train_mask.nonzero(as_tuple=False).squeeze(1)
            pick = idx[torch.randperm(idx.numel(), device=device)[:k]]
            new_lab[pick] = 1 - new_lab[pick]

    meta = copy.deepcopy(tg.metadata)
    meta["anomaly_injection"] = {
        "is_injected": True,
        "type": "structural_random_uniform",
        "train_ratio": anom_train_ratio,
        "val_ratio": anom_val_ratio,
        "test_ratio": anom_test_ratio,
        "noise_ratio": noise_ratio,
        "total_anomalies": int((new_lab == 0).sum().item()),
    }

    return TemporalGraph(
        src=new_src,
        tgt=new_tgt,
        t=new_t,
        msg=new_msg,
        edge_labels=new_lab,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        w=(tg.w.clone() if tg.w is not None else None),
        node_attr=(tg.node_attr.clone() if tg.node_attr is not None else None),
        node_labels=(tg.node_labels.clone()
                     if tg.node_labels is not None else None),
        metadata=meta,
    )
