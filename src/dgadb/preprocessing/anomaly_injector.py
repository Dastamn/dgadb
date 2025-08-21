import copy
import logging
import torch
import torch.nn.functional as F
from math import gcd
from functools import reduce
from datetime import datetime, timezone
from typing import Literal, Optional
from src.dgadb.storage import TemporalGraphData
from .utils import (
    cartesian_sample,
    compute_unique_inverse_count_probabilities,
    unique_with_indices,
    to_undirected,
    to_canonical,
)

_ANOMALY_TYPE_MAP: dict[str, str] = {
    "s": "structural",
    "t": "temporal",
    "c": "contextual",
    "sc": "structural-contextual",
    "tc": "temporal-contextual",
    "tsc": "temporal-structural-contextual",
}

_CANONICAL_ANOMALY_TYPES: set[str] = set(_ANOMALY_TYPE_MAP.values())

VALID_ANOMALY_TYPES: list[str] = list(
    _ANOMALY_TYPE_MAP) + list(_ANOMALY_TYPE_MAP.values())


class AnomalyInjector:
    def __init__(self, temporal_graph: TemporalGraphData, rnd_seed: int = 123) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.temporal_graph = temporal_graph
        self.device = temporal_graph.src.device

        self.logger.info(
            f"Initializing AnomalyInjector for dataset '{temporal_graph.dataset_name}'...")

        self.max_node_id = (
            torch.max(temporal_graph.src.max(), temporal_graph.tgt.max()) + 1).float()

        self.encoded_observed_edges = self._encode_edges(
            temporal_graph.src, temporal_graph.tgt, keep_unique=True)
        self.logger.info(
            f"> Created lookup tensor with {self.encoded_observed_edges.numel()} unique edges.")

        self.encoded_observed_edges_t = self._encode_edges(
            temporal_graph.src, temporal_graph.tgt, t=temporal_graph.t, keep_unique=True
        )
        self.logger.info(
            f"> Created lookup tensor with {self.encoded_observed_edges_t.numel()} unique timestamped edges."
        )

        self.granularity, self.t_deltas, self.t_deltas_size, self.t_deltas_p = self._compute_time_properties(
            temporal_graph.t[temporal_graph.train_mask]
        )
        self.logger.info(
            f"> Time granularity: {self.granularity}, time deltas size: {self.t_deltas_size}.")

        self.first_t = temporal_graph.t.min()
        self.last_t = temporal_graph.t.max()

        self.logger.info("AnomalyInjector initialized successfully.")

    def _validate_and_get_anomaly_type(self, anom_type: str) -> str:
        anom_type = anom_type.lower()

        if anom_type in _ANOMALY_TYPE_MAP:
            return _ANOMALY_TYPE_MAP[anom_type]

        if anom_type in _CANONICAL_ANOMALY_TYPES:
            return anom_type

        raise ValueError(
            f"Unknown anomaly type '{anom_type}', expected: {list(VALID_ANOMALY_TYPES)}")

    def _encode_edges(
        self, src: torch.Tensor, tgt: torch.Tensor, t: Optional[torch.Tensor] = None, keep_unique: bool = False
    ) -> torch.Tensor:
        components = [src, tgt]
        if t is not None:
            components.append(t)

        combined = torch.stack(components, dim=1)

        if keep_unique:
            rows_to_encode = torch.unique(combined, dim=0)
        else:
            rows_to_encode = combined

        src_unpacked, tgt_unpacked, * \
            t_unpacked_list = rows_to_encode.unbind(dim=1)
        base = float(self.max_node_id)

        src_ = src_unpacked.long()
        tgt_ = tgt_unpacked.long()

        if t_unpacked_list:
            t_ = t_unpacked_list[0].long()
            encoded_values = t_ * (base**2) + src_ * base + tgt_
        else:
            encoded_values = src_ * base + tgt_

        return encoded_values.long()

    def _add_to_encoded_observed_edges(self, src: torch.Tensor, tgt: torch.Tensor, t: Optional[torch.Tensor]) -> None:
        encoded_edges = self._encode_edges(src, tgt, t)
        encoded_base_name = "encoded_observed_edges_t" if t is not None else "encoded_observed_edges"
        encoded_base = getattr(self, encoded_base_name)
        is_new_mask = ~torch.isin(encoded_edges, encoded_base)
        valid_indices = torch.where(is_new_mask)[0]
        new_edges = encoded_edges[valid_indices]
        if new_edges.shape[0] > 0:
            encoded_base = torch.cat([encoded_base, new_edges], dim=0)
            setattr(self, encoded_base_name, encoded_base)

    def _compute_time_properties(self, t_train: torch.Tensor) -> tuple[int, torch.Tensor, int, Optional[torch.Tensor]]:
        unique_t = torch.unique(t_train)

        if unique_t.numel() < 2:
            t_deltas = torch.empty(0, dtype=torch.long)
            t_deltas_p = None
            self.logger.warning(
                "Training time has <= 1 unique value. Time deltas will be empty.")
        else:
            unique_t, _ = torch.sort(unique_t)
            t_deltas = torch.diff(unique_t)
            t_deltas_p = compute_unique_inverse_count_probabilities(
                t_deltas, device=self.device)

        t_deltas_size = t_deltas.numel()

        if t_deltas_size == 0:
            granularity = 1
        else:
            granularity = reduce(gcd, t_deltas.tolist())

        return (
            granularity,
            t_deltas.to(self.device),
            t_deltas_size,
            t_deltas_p.to(self.device) if t_deltas_p is not None else None,
        )

    def _sample_t_deltas(self, size: int | torch.Tensor) -> torch.Tensor:
        if self.t_deltas_size == 0:
            torch.empty(0, dtype=torch.int64, device=self.device)

        if self.t_deltas_p is None:
            raise ValueError("'t_deltas_p' is None.")

        indices = torch.multinomial(
            self.t_deltas_p, num_samples=size, replacement=True)
        return self.t_deltas[indices]

    def _generate_plausible_timestamps(
        self,
        n: int,  # Number of timestamps to generate
        first_t: torch.Tensor,
        last_t: torch.Tensor,
        random_time_walk_max_steps: int = 15,
    ) -> torch.Tensor:
        start_base_point = (first_t + self.granularity - 1) // self.granularity
        end_base_point = last_t // self.granularity

        if start_base_point > end_base_point:
            return torch.full((n,), first_t, dtype=torch.int64, device=self.device)

        random_base_points = torch.randint(
            start_base_point, end_base_point + 1, size=(n,), device=self.device)
        start_ts = random_base_points * self.granularity

        # Fallback if no deltas were learned from training data
        if self.t_deltas_size == 0:
            return start_ts

        num_steps_per_walk = torch.randint(
            1, random_time_walk_max_steps, size=(n,), device=self.device)
        total_steps = torch.sum(num_steps_per_walk)

        t_deltas = self._sample_t_deltas(total_steps)

        t_signs = torch.randint(0, 2, size=(
            total_steps,), device=self.device) * 2 - 1  # {-1, 1}

        signed_deltas = t_deltas * t_signs

        walk_ids = torch.arange(
            n, device=self.device).repeat_interleave(num_steps_per_walk)
        total_displacements = torch.zeros(
            n, dtype=torch.float32, device=self.device)
        total_displacements.scatter_add_(0, walk_ids, signed_deltas.float())

        final_ts = start_ts.float() + total_displacements

        clipped_ts = torch.clamp(
            final_ts, min=float(first_t), max=float(last_t))

        return clipped_ts.to(torch.int64)

    def _sample_contextually_inconsistent_features(
        self,
        curr_msg_batch: torch.Tensor,
        msg_pool: torch.Tensor,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        sample_size: Optional[int] = 100,
    ):
        batch_size, _ = curr_msg_batch.shape
        num_candidates_in_pool = msg_pool.shape[0]

        if sample_size is None or sample_size >= num_candidates_in_pool:
            candidate_indices = torch.arange(
                num_candidates_in_pool, device=self.device).expand(batch_size, -1)
        else:
            candidate_indices = torch.randint(
                0, num_candidates_in_pool, size=(batch_size, sample_size), device=self.device
            )

        # Shape: (batch_size, sample_size, feature_dim)
        candidates = msg_pool[candidate_indices]
        # Shape: (batch_size, 1, feature_dim)
        curr_msgs_expanded = curr_msg_batch.unsqueeze(1)

        # Distances shape: (batch_size, sample_size)
        if distance_metric == "cosine":
            distances = 1.0 - \
                F.cosine_similarity(curr_msgs_expanded, candidates, dim=2)
        elif distance_metric == "l2":
            distances = torch.norm(candidates - curr_msgs_expanded, p=2, dim=2)
        else:
            raise ValueError(f"'{distance_metric}' is not supported.")

        # Shape: (batch_size)
        most_dissimilar_indices = torch.argmax(distances, dim=1)

        # Shape: (batch_size, feature_dim)
        return candidates[torch.arange(batch_size, device=self.device), most_dissimilar_indices]

    def _sample_from_temporal_window(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: torch.Tensor,
        temporal_window_size: int | float,
    ):
        num_edges = src.numel()
        if isinstance(temporal_window_size, float):
            # Interpret float as a proportion of data
            win_len = int(num_edges * temporal_window_size)
        else:  # int
            win_len = temporal_window_size

        win_len = max(1, min(win_len, num_edges))

        anchor_index = torch.multinomial(edge_p, num_samples=1)
        anchor_index = anchor_index.squeeze(0)

        start_index = (anchor_index - win_len).clamp(min=0)
        end_index = (anchor_index + win_len).clamp(max=num_edges)
        idx = torch.arange(start_index, end_index, device=src.device)

        return src[idx], tgt[idx], t[idx], msg[idx]

    def _select_new_edge_indices(
        self, size: int, src: torch.Tensor, tgt: torch.Tensor, t: Optional[torch.Tensor], encode_unique: bool = False
    ):
        if t is None:
            anomaly_type = "structural"
            encoded_observed_edges_pool = self.encoded_observed_edges
        else:
            anomaly_type = "temporal"
            encoded_observed_edges_pool = self.encoded_observed_edges_t

        encoded_edges = self._encode_edges(
            src, tgt, t, keep_unique=encode_unique)
        is_new_edge_mask = ~torch.isin(
            encoded_edges, encoded_observed_edges_pool)
        valid_indices = torch.where(is_new_edge_mask)[0]

        if valid_indices.numel() > 1:
            # Keep only unique indices
            valid_encoded_edges = encoded_edges[valid_indices]
            _, first_occurrence_in_valid = unique_with_indices(
                valid_encoded_edges)
            valid_indices = valid_indices[first_occurrence_in_valid]

        num_found = valid_indices.numel()
        total_candidates = src.numel()

        if num_found == 0:
            self.logger.warning(
                f"Found 0 unique '{anomaly_type}' anomalies. Using random fallbacks.")
            # Might have duplicates, but are observed anyways
            final_indices = torch.randperm(
                total_candidates, device=self.device)[:size]

        elif num_found >= size:
            perm = torch.randperm(num_found, device=self.device)[:size]
            final_indices = valid_indices[perm]

        else:  # num_found < size
            self.logger.warning(
                f"Found only {num_found}/{size} unique '{anomaly_type}' anomalies, using all found and adding fallbacks."
            )

            num_missing = size - num_found
            non_unique_indices = torch.where(~is_new_edge_mask)[0]
            num_available_fallbacks = len(non_unique_indices)

            if num_available_fallbacks == 0:
                self.logger.warning(
                    "No observed edges available for fallback. Reusing unique anomalies.")
                random_indices = torch.randint(
                    0, num_found, size=(num_missing,), device=self.device)
                fallback_indices = valid_indices[random_indices]

            else:
                # Sample with replacement from the available fallback pool
                # This guarantees to get 'num_missing' indices
                random_indices = torch.randint(
                    0, num_available_fallbacks, size=(num_missing,), device=self.device)
                fallback_indices = non_unique_indices[random_indices]

            final_indices = torch.cat([valid_indices, fallback_indices])

        return final_indices

    def _generate_structural_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: Optional[torch.Tensor] = None,
        temporal_window_size: Optional[int | float] = 1500,
        struct_max_num_candidate: int = 1_000_000,
        struct_oversampling_ratio: float = 1.2,
    ):
        if size == 0:
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device),
            )

        if temporal_window_size is not None:
            if edge_p is None:
                raise ValueError("'edge_p' is None.")

            window_src, window_tgt, window_t, window_msg = self._sample_from_temporal_window(
                src, tgt, t, msg, edge_p, temporal_window_size
            )
        else:
            self.logger.warning(
                "'temporal_window_size' is None. Sampling nodes from the entire split.")
            window_src, window_tgt, window_t, window_msg = src, tgt, t, msg

        src_pool = torch.unique(window_src)
        tgt_pool = torch.unique(window_tgt)

        num_candidates = src_pool.numel() * tgt_pool.numel()
        if num_candidates > struct_max_num_candidate:
            sample_size = int(struct_max_num_candidate *
                              max(1.0, min(struct_oversampling_ratio, 2.0)))
            self.logger.info(
                f"'num_candidates' is too large ({num_candidates}), sampling {sample_size} instead.")
            cand_edges = cartesian_sample(
                [src_pool, tgt_pool], sample_size, self.device)
        else:
            cand_edges = torch.cartesian_prod(src_pool, tgt_pool)

        cand_src, cand_tgt = cand_edges.unbind(dim=1)

        new_edge_indices = self._select_new_edge_indices(
            size=size, src=cand_src, tgt=cand_tgt, t=None, encode_unique=True
        )

        anom_src, anom_tgt = cand_src[new_edge_indices], cand_tgt[new_edge_indices]

        # Map 'anom_src' to original 't' and 'msg'
        unique_window_src, first_occurrence_indices = unique_with_indices(
            window_src)

        # Shape: (anom_src, unique_window_src) <<< (anom_src, window_src)
        matches = anom_src.unsqueeze(1) == unique_window_src.unsqueeze(0)

        indices_in_unique_lookup = torch.argmax(matches.byte(), dim=1)
        original_indices = first_occurrence_indices[indices_in_unique_lookup]

        anom_t = window_t[original_indices]
        anom_msg = window_msg[original_indices]

        return anom_src, anom_tgt, anom_t, anom_msg

    def _generate_temporal_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: torch.Tensor,
        first_t: torch.Tensor,
        last_t: torch.Tensor,
        e_num_candidates: int = 1000,
        t_num_candidates: int = 1000,
        random_time_walk_max_steps: int = 15,
    ):
        if size == 0:
            self.logger.warning(
                "Input size 0 in '_generate_temporal_anomalies', returning empty tensors.")
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device),
            )

        cand_num_edges = max(size, e_num_candidates)
        cand_indices = torch.multinomial(
            edge_p, num_samples=cand_num_edges, replacement=True)
        cand_src, cand_tgt, cand_msg = src[cand_indices], tgt[cand_indices], msg[cand_indices]

        cand_t = self._generate_plausible_timestamps(
            t_num_candidates, first_t, last_t, random_time_walk_max_steps)

        # Shape: (num_base_edges * timestamp_num_tries)
        expanded_src = cand_src.repeat_interleave(t_num_candidates, dim=0)
        expanded_tgt = cand_tgt.repeat_interleave(t_num_candidates, dim=0)
        expanded_msg = cand_msg.repeat_interleave(t_num_candidates, dim=0)
        expanded_t = cand_t.repeat(cand_num_edges)

        new_edge_indices = self._select_new_edge_indices(
            size=size,
            src=expanded_src,
            tgt=expanded_tgt,
            t=expanded_t,
            encode_unique=False,  # to link back to 'expanded_msg'
        )

        anom_src = expanded_src[new_edge_indices]
        anom_tgt = expanded_tgt[new_edge_indices]
        anom_t = expanded_t[new_edge_indices]
        anom_msg = expanded_msg[new_edge_indices]

        return anom_src, anom_tgt, anom_t, anom_msg

    def _generate_contextual_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: torch.Tensor,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        ctx_sample_size: int = 25,
    ):
        if size == 0:
            self.logger.warning(
                "Input size 0 in '_generate_contextual_anomalies', returning empty tensors.")
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device),
            )

        cand_indices = torch.multinomial(
            edge_p, num_samples=size, replacement=True)
        cand_src, cand_tgt, cand_t, cand_msg = (
            src[cand_indices],
            tgt[cand_indices],
            t[cand_indices],
            msg[cand_indices],
        )

        anom_msg = self._sample_contextually_inconsistent_features(
            cand_msg, msg, distance_metric, ctx_sample_size)

        random_indices = torch.randint(
            0, cand_src.numel(), size=(size,), device=self.device)

        return cand_src[random_indices], cand_tgt[random_indices], cand_t[random_indices], anom_msg

    def _generate_structural_contextual_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: torch.Tensor,
        temporal_window_size: Optional[int | float] = 1500,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        ctx_sample_size: int = 100,
    ):
        if size == 0:
            self.logger.warning(
                "Input size 0 in '_generate_structural_contextual_anomalies', returning empty tensors.")
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device),
            )

        anom_src, anom_tgt, t_, msg_ = self._generate_structural_anomalies(
            size, src, tgt, t, msg, edge_p, temporal_window_size
        )

        anom_msg = self._sample_contextually_inconsistent_features(
            msg_, msg, distance_metric, ctx_sample_size)

        return anom_src, anom_tgt, t_, anom_msg

    def _generate_temporal_contextual_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_p: torch.Tensor,
        first_t: torch.Tensor,
        last_t: torch.Tensor,
        e_num_candidates: int = 1000,
        t_num_candidates: int = 1000,
        random_time_walk_max_steps: int = 15,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        ctx_sample_size: int = 100,
    ):
        if size == 0:
            self.logger.warning(
                "Input size 0 in '_generate_temporal_contextual_anomalies', returning empty tensors.")
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device),
            )

        src_, tgt_, anom_t, msg_ = self._generate_temporal_anomalies(
            size,
            src,
            tgt,
            t,
            msg,
            edge_p,
            first_t,
            last_t,
            e_num_candidates,
            t_num_candidates,
            random_time_walk_max_steps,
        )

        anom_msg = self._sample_contextually_inconsistent_features(
            msg_, msg, distance_metric, ctx_sample_size)

        return src_, tgt_, anom_t, anom_msg

    def _generate_temporal_structural_contextual_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        first_t: torch.Tensor,
        last_t: torch.Tensor,
    ):
        anom_src, anom_tgt, _, _ = self._generate_structural_anomalies(
            size, src, tgt, t, msg, temporal_window_size=None
        )

        anom_t = self._generate_plausible_timestamps(size, first_t, last_t)

        random_msg_indices = torch.randperm(
            msg.size(0), device=self.device)[:size]
        anom_msg = msg[random_msg_indices]

        return anom_src, anom_tgt, anom_t, anom_msg

    def generate_anomalous_samples(
        self,
        anom_type: str,
        anom_train_ratio: float = 0.0,
        anom_val_ratio: float = 0.0,
        anom_test_ratio: float = 0.05,
        reset_labels: bool = True,
        max_attempts: int = 10,
        **kwargs,
    ) -> TemporalGraphData:
        anom_type = self._validate_and_get_anomaly_type(anom_type)
        anom_ratios = {"train": anom_train_ratio,
                       "test": anom_test_ratio, "val": anom_val_ratio}

        injection_metadata = {
            "is_injected": True,
            "last_injection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "type": anom_type,
            "generation_parameters": copy.deepcopy(kwargs),
            "splits": {
                "train": {"ratio": anom_train_ratio, "requested": 0, "generated": 0},
                "val": {"ratio": anom_val_ratio, "requested": 0, "generated": 0},
                "test": {"ratio": anom_test_ratio, "requested": 0,  "generated": 0}
            },
            "total": 0
        }

        src, tgt, t, msg, edge_labels, train_mask, val_mask, test_mask = (
            self.temporal_graph.src,
            self.temporal_graph.tgt,
            self.temporal_graph.t,
            self.temporal_graph.msg,
            self.temporal_graph.edge_labels,
            self.temporal_graph.train_mask,
            self.temporal_graph.val_mask,
            self.temporal_graph.test_mask,
        )

        if reset_labels and (self.temporal_graph.edge_labels == 1).any():
            self.logger.info(
                "'reset_labels' set to 'True', removing anomalies present in dataset...")
            is_normal_mask = edge_labels == 0
            src, tgt, t, msg, edge_labels, train_mask, val_mask, test_mask = (
                src[is_normal_mask],
                tgt[is_normal_mask],
                t[is_normal_mask],
                msg[is_normal_mask],
                edge_labels[is_normal_mask],
                train_mask[is_normal_mask],
                val_mask[is_normal_mask],
                test_mask[is_normal_mask],
            )

        gen_anom: dict[str, tuple[torch.Tensor, ...]] = {}

        for split, anom_ratio in anom_ratios.items():
            if anom_ratio == 0.0:
                continue

            if not 0.0 <= anom_ratio <= 1.0:
                raise ValueError(f"Invalid anomaly ratio for '{split}' split.")

            if split == "train":
                mask = train_mask
            elif split == "test":
                mask = test_mask
            else:
                mask = val_mask

            if not mask.any():
                self.logger.info(f"Split '{split}' is empty, skipping.")
                continue

            eval_src = src[mask]
            eval_tgt = tgt[mask]
            eval_t = t[mask]
            eval_msg = msg[mask]

            num_anom = int(eval_src.shape[0] * anom_ratio)

            injection_metadata["splits"][split]["requested"] = num_anom

            eval_first_t, eval_last_t = eval_t.min(), eval_t.max()
            eval_edges = torch.stack([eval_src, eval_tgt], dim=1)
            eval_edge_p = compute_unique_inverse_count_probabilities(
                eval_edges, device=self.device)

            self.logger.info(
                f"Attempting to generate {num_anom} '{anom_type}' anomalous edges in '{split}' split...")

            split_gen_anom_list = []
            num_generated = 0
            attempts = 0

            while num_generated < num_anom and attempts < max_attempts:
                attempts += 1
                still_needed = num_anom - num_generated
                request_size = max(still_needed, num_anom)

                self.logger.info(
                    f"[Attempt {attempts}/{max_attempts}] Generating a batch of {request_size}. Need {still_needed} more...")

                if anom_type == "structural":
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_structural_anomalies(
                        num_anom, eval_src, eval_tgt, eval_t, eval_msg, eval_edge_p, **kwargs
                    )

                elif anom_type == "temporal":
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_temporal_anomalies(
                        num_anom, eval_src, eval_tgt, eval_t, eval_msg, eval_edge_p, eval_first_t, eval_last_t, **kwargs
                    )

                elif anom_type == "contextual":
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_contextual_anomalies(
                        num_anom, eval_src, eval_tgt, eval_t, eval_msg, eval_edge_p, **kwargs
                    )

                elif anom_type == "structural-contextual":
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_structural_contextual_anomalies(
                        num_anom, eval_src, eval_tgt, eval_t, eval_msg, eval_edge_p, **kwargs
                    )

                elif anom_type == "temporal-contextual":
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_temporal_contextual_anomalies(
                        num_anom, eval_src, eval_tgt, eval_t, eval_msg, eval_edge_p, eval_first_t, eval_last_t, **kwargs
                    )

                elif anom_type == "temporal-structural-contextual":
                    # Here use all data
                    anom_src, anom_tgt, anom_t, anom_msg = self._generate_temporal_structural_contextual_anomalies(
                        num_anom, src, tgt, t, msg, self.first_t, self.last_t
                    )

                else:
                    raise NotImplementedError(
                        f"Generation for anomaly type '{anom_type}' is not implemented.")

                if anom_src.numel() == 0:
                    self.logger.warning(
                        f"[Attempt {attempts}] Generation returned no anomalies.")
                    continue

                # Keep originally intended structure
                if self.temporal_graph.directionality == "undirected":
                    # Already handles de-duplication
                    anom_src, anom_tgt, undirected_indices = to_undirected(
                        anom_src, anom_tgt)
                    anom_t, anom_msg = anom_t[undirected_indices], anom_msg[undirected_indices]
                else:
                    if self.temporal_graph.directionality == "canonical":
                        anom_src, anom_tgt = to_canonical(anom_src, anom_tgt)

                    # De-dup
                    encoded_edges = self._encode_edges(
                        anom_src, anom_tgt, anom_t, keep_unique=False)
                    _, first_occurence_indices = unique_with_indices(
                        encoded_edges)
                    anom_src, anom_tgt, anom_t, anom_msg = (
                        anom_src[first_occurence_indices],
                        anom_tgt[first_occurence_indices],
                        anom_t[first_occurence_indices],
                        anom_msg[first_occurence_indices],
                    )

                num_unique_anom = anom_src.numel()
                num_generated += num_unique_anom
                self.logger.info(
                    f"[Attempt {attempts}/{max_attempts}] Found {num_unique_anom} new unique anomalies. Total found: {num_generated}/{num_anom}.")

                gen_anom[split] = (anom_src, anom_tgt, anom_t, anom_msg)
                split_gen_anom_list.append(
                    (anom_src, anom_tgt, anom_t, anom_msg))
                # Add to observed edges so that next split anomalies avoid collisions
                self._add_to_encoded_observed_edges(anom_src, anom_tgt, anom_t)

            if num_generated < num_anom:
                self.logger.warning(
                    f"Failed to generate the requested number of anomalies for '{split}' split after {max_attempts} attempts. "
                    f"Proceeding with {num_generated}/{num_anom} anomalies."
                )

            if not split_gen_anom_list:
                continue

            split_anom_src, split_anom_tgt, split_anom_t, split_anom_msg = \
                zip(*split_gen_anom_list)

            split_anom_src = torch.cat(split_anom_src)[:num_anom]
            split_anom_tgt = torch.cat(split_anom_tgt)[:num_anom]
            split_anom_t = torch.cat(split_anom_t)[:num_anom]
            split_anom_msg = torch.cat(split_anom_msg)[:num_anom]

            self.logger.info(
                f"Generated a total of {split_anom_src.numel()} anomalies for '{split}' split.")
            gen_anom[split] = (split_anom_src, split_anom_tgt,
                               split_anom_t, split_anom_msg)

        src_list, tgt_list, t_list, msg_list = [], [], [], []
        train_mask_list, val_mask_list, test_mask_list = [], [], []

        for split, (anom_src, anom_tgt, anom_t, anom_msg) in gen_anom.items():
            num_anom_in_split = anom_src.numel()

            src_list.append(anom_src)
            tgt_list.append(anom_tgt)
            t_list.append(anom_t)
            msg_list.append(anom_msg)

            train_mask_list.append(
                torch.full((num_anom_in_split,), split == "train",
                           dtype=torch.bool, device=self.device)
            )
            val_mask_list.append(torch.full(
                (num_anom_in_split,), split == "val", dtype=torch.bool, device=self.device))
            test_mask_list.append(
                torch.full((num_anom_in_split,), split == "test",
                           dtype=torch.bool, device=self.device)
            )

        all_anom_src = torch.cat(src_list)
        all_anom_tgt = torch.cat(tgt_list)
        all_anom_t = torch.cat(t_list)
        all_anom_msg = torch.cat(msg_list)

        anom_train_mask = torch.cat(train_mask_list)
        anom_val_mask = torch.cat(val_mask_list)
        anom_test_mask = torch.cat(test_mask_list)

        # Combine everything
        normal_labels = torch.zeros_like(edge_labels)
        anom_labels = torch.ones(all_anom_src.numel(),
                                 dtype=torch.long, device=self.device)

        final_src = torch.cat([src, all_anom_src])
        final_tgt = torch.cat([tgt, all_anom_tgt])
        final_t = torch.cat([t, all_anom_t])
        final_msg = torch.cat([msg, all_anom_msg])
        final_label = torch.cat([normal_labels, anom_labels])
        final_train_mask = torch.cat([train_mask, anom_train_mask])
        final_val_mask = torch.cat([val_mask, anom_val_mask])
        final_test_mask = torch.cat([test_mask, anom_test_mask])

        # Sort by 'src' then 't'
        self.logger.info("Sorting final graph by timestamp and source node...")
        src_sort_indices = torch.argsort(final_src, stable=True)
        src_sorted_t = final_t[src_sort_indices]
        final_sort_indices = src_sort_indices[torch.argsort(
            src_sorted_t, stable=True)]

        # Update metadata
        injection_metadata["splits"]["train"]["generated"] = \
            anom_train_mask.sum().item()
        injection_metadata["splits"]["val"]["generated"] = \
            anom_val_mask.sum().item()
        injection_metadata["splits"]["test"]["generated"] = \
            anom_test_mask.sum().item()
        injection_metadata["total"] = final_label.sum().item()
        metadata = copy.deepcopy(self.temporal_graph.metadata)
        metadata["anomaly_injection"] = injection_metadata

        anomalous_temporal_graph = TemporalGraphData(
            src=final_src[final_sort_indices],
            tgt=final_tgt[final_sort_indices],
            t=final_t[final_sort_indices],
            msg=final_msg[final_sort_indices],
            edge_labels=final_label[final_sort_indices],
            train_mask=final_train_mask[final_sort_indices],
            val_mask=final_val_mask[final_sort_indices],
            test_mask=final_test_mask[final_sort_indices],
            w=(self.temporal_graph.w.clone()
               if self.temporal_graph.w is not None else None),
            node_attr=(self.temporal_graph.node_attr.clone()
                       if self.temporal_graph.node_attr is not None else None),
            node_labels=(self.temporal_graph.node_labels.clone()
                         if self.temporal_graph.node_labels is not None else None),
            metadata=metadata,
        )

        self.logger.info(
            f"Edge anomaly injection complete. New graph has {anomalous_temporal_graph.num_edges} total edges."
        )

        return anomalous_temporal_graph
