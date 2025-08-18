import logging
import torch
import torch.nn.functional as F
from math import gcd
from functools import reduce
from typing import Literal, Optional
from .utils import cartesian_sample, compute_unique_inverse_count_probabilities
from src.dgadb.storage import TemporalGraphData


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

        self.max_node_id = (
            torch.max(temporal_graph.src.max(), temporal_graph.tgt.max()) + 1
        ).float()

        self.logger.info(f"Building encoded lookup for (src, tgt)...")
        self.encoded_observed_edges = self._encode_edges(
            temporal_graph.src, temporal_graph.tgt, find_unique=True)
        self.logger.info(
            f"Created lookup tensor with {self.encoded_observed_edges.numel()} unique edges.")

        self.logger.info(f"Building encoded lookup for (src, tgt, t)...")
        self.encoded_observed_edges_t = self._encode_edges(
            temporal_graph.src, temporal_graph.tgt, t=temporal_graph.t, find_unique=True)
        self.logger.info(
            f"Created lookup tensor with {self.encoded_observed_edges_t.numel()} unique timestamped edges.")

        self.logger.info("Computing time properties...")
        self.granularity, self.t_deltas, self.t_deltas_size, self.t_deltas_p = \
            self._compute_time_properties(
                temporal_graph.t[temporal_graph.train_mask])
        self.logger.info(
            f"Time granularity: {self.granularity}, time deltas size: {self.t_deltas_size}.")

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
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        find_unique: bool = False
    ) -> torch.Tensor:
        components = [src, tgt]
        if t is not None:
            components.append(t)

        combined = torch.stack(components, dim=1)

        if find_unique:
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
            encoded_values = (t_ * (base**2) +
                              src_ * base +
                              tgt_)
        else:
            encoded_values = src_ * base + tgt_

        return encoded_values.long()

    def _add_to_encoded_observed_edges(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: Optional[torch.Tensor]
    ) -> None:
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

        return granularity, t_deltas.to(self.device), t_deltas_size, t_deltas_p.to(self.device) if t_deltas_p is not None else None

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
        random_time_walk_max_steps: int = 5
    ) -> torch.Tensor:
        start_base_point = (first_t + self.granularity - 1) // self.granularity
        end_base_point = last_t // self.granularity

        if start_base_point > end_base_point:
            return torch.full((n,), first_t, dtype=torch.int64, device=first_t.device)

        random_base_points = torch.randint(
            start_base_point, end_base_point + 1,
            size=(n,),
            device=first_t.device
        )
        start_ts = random_base_points * self.granularity

        # Fallback if no deltas were learned from training data
        if self.t_deltas_size == 0:
            return start_ts

        num_steps_per_walk = torch.randint(
            1, random_time_walk_max_steps,
            size=(n,),
            device=first_t.device
        )
        total_steps = torch.sum(num_steps_per_walk)

        t_deltas = self._sample_t_deltas(total_steps)

        t_signs = torch.randint(
            0, 2,
            size=(total_steps,),
            device=first_t.device
        ) * 2 - 1  # {-1, 1}

        signed_deltas = t_deltas * t_signs

        walk_ids = torch.arange(n).repeat_interleave(num_steps_per_walk)
        total_displacements = torch.zeros(
            n, dtype=torch.float32, device=first_t.device)
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
            sample_size: Optional[int] = 10
    ):
        device = curr_msg_batch.device
        batch_size, _ = curr_msg_batch.shape
        num_candidates_in_pool = msg_pool.shape[0]

        if sample_size is None or sample_size >= num_candidates_in_pool:
            candidate_indices = torch.arange(
                num_candidates_in_pool, device=device).expand(batch_size, -1)
        else:
            candidate_indices = torch.randint(
                0, num_candidates_in_pool,
                size=(batch_size, sample_size),
                device=device
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
        return candidates[torch.arange(batch_size, device=device), most_dissimilar_indices]

    def _generate_structural_anomalies(
        self,
        size: int,
        src: torch.Tensor,
        tgt: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        window_size: Optional[int | float] = None,
        oversampling_ratio: float = 1.2,
        max_num_candidate: int = 500_000
    ):
        if size == 0:
            return (torch.empty(0, dtype=src.dtype, device=self.device),
                    torch.empty(0, dtype=tgt.dtype, device=self.device),
                    torch.empty(0, dtype=t.dtype, device=self.device),
                    torch.empty(0, msg.shape[1],
                                dtype=msg.dtype, device=self.device))

        oversampling_ratio = max(1.0, min(oversampling_ratio, 2.0))
        num_edges = len(src)

        if window_size is not None:
            if isinstance(window_size, float):
                if not 0 < window_size <= 1:
                    raise ValueError("Invalid 'window_size' value.")

                win_len = int(num_edges * window_size)
            else:
                if window_size > num_edges:
                    self.logger.warning(
                        "'window_size' is larger than the number of provided edges, "
                        "Sampling nodes from the entire split.")
                    window_size = num_edges

                win_len = window_size

            win_len = max(2, win_len)
            start_idx = torch.randint(
                0, num_edges - win_len + 1, size=(1,)).item()
            end_idx = start_idx + win_len

            self.logger.info(
                f"Generating {size} structural anomalies from a temporal window of {win_len}/{num_edges}.")
            window_src, window_tgt, window_t, window_msg = \
                src[start_idx:end_idx], tgt[start_idx:end_idx], t[start_idx:end_idx], msg[start_idx:end_idx]

        else:
            self.logger.info("Sampling nodes from the entire split.")
            window_src, window_tgt, window_t, window_msg = src, tgt, t, msg

        src_pool = torch.unique(window_src)
        tgt_pool = torch.unique(window_tgt)

        num_candidates = src_pool.numel() * tgt_pool.numel()
        if num_candidates > max_num_candidate:
            sample_size = int(max_num_candidate * oversampling_ratio)
            self.logger.info(f"'num_candidates' is too large ({num_candidates}), "
                             f"sampling {sample_size} instead.")
            cand_edges = cartesian_sample(
                [src_pool, tgt_pool], sample_size, self.device)
        else:
            cand_edges = torch.cartesian_prod(src_pool, tgt_pool)

        cand_src, cand_tgt = cand_edges.unbind(dim=1)
        total_candidates = len(cand_src)

        encoded_cand = self._encode_edges(
            src=cand_src, tgt=cand_tgt, find_unique=False)
        is_unique_mask = ~torch.isin(encoded_cand, self.encoded_observed_edges)

        valid_indices = torch.where(is_unique_mask)[0]
        num_found = len(valid_indices)

        self.logger.info(
            f"Structural search found {num_found} unique edges from a pool of {total_candidates}.")

        if num_found == 0:
            self.logger.warning(
                "Found 0 unique structural anomalies. Using random fallbacks.")
            final_indices = torch.arange(
                min(size, total_candidates), device=self.device)
        elif num_found >= size:
            perm = torch.randperm(num_found, device=self.device)[:size]
            final_indices = valid_indices[perm]
        else:  # num_found < size
            self.logger.warning(f"Found only {num_found} unique structural anomalies, "
                                f"but {size} were requested. Using all found and adding fallbacks.")
            num_missing = size - num_found
            non_unique_indices = torch.where(~is_unique_mask)[0]
            num_fallbacks_to_take = min(num_missing, len(non_unique_indices))
            fallback_indices = non_unique_indices[torch.randperm(
                len(non_unique_indices), device=self.device)[:num_fallbacks_to_take]]
            final_indices = torch.cat([valid_indices, fallback_indices])

        anom_src = cand_src[final_indices]
        anom_tgt = cand_tgt[final_indices]

        # Get random t and msg indices
        random_indices = torch.randint(
            0, window_src.numel(), size=(size,), device=self.device)

        return anom_src, anom_tgt, window_t[random_indices], window_msg[random_indices]

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
            e_num_tries: int = 100,
            t_num_tries: int = 100,
            random_time_walk_max_steps: int = 5
    ):
        if size == 0:
            self.logger.warning(
                "Input size 0 in '_generate_temporal_anomalies', returning empty tensors.")
            return (
                torch.empty(0, dtype=src.dtype, device=self.device),
                torch.empty(0, dtype=tgt.dtype, device=self.device),
                torch.empty(0, dtype=t.dtype, device=self.device),
                torch.empty(0, msg.shape[1],
                            dtype=msg.dtype, device=self.device)
            )

        cand_num_edges = max(size, e_num_tries)
        cand_indices = torch.multinomial(
            edge_p, num_samples=cand_num_edges, replacement=True)
        cand_src, cand_tgt, cand_msg = src[cand_indices], tgt[cand_indices], msg[cand_indices]

        cand_t = self._generate_plausible_timestamps(
            t_num_tries,
            first_t,
            last_t,
            random_time_walk_max_steps,
        )

        # Shape: (num_base_edges * timestamp_num_tries)
        expanded_src = cand_src.repeat_interleave(t_num_tries, dim=0)
        expanded_tgt = cand_tgt.repeat_interleave(t_num_tries, dim=0)
        expanded_msg = cand_msg.repeat_interleave(t_num_tries, dim=0)

        expanded_t = cand_t.repeat(cand_num_edges)

        encoded_cand = self._encode_edges(
            expanded_src,
            expanded_tgt,
            t=expanded_t,
            find_unique=False
        )

        is_unique_mask = ~torch.isin(
            encoded_cand, self.encoded_observed_edges_t)
        valid_indices = torch.where(is_unique_mask)[0]
        num_found = len(valid_indices)

        if num_found == 0:
            self.logger.warning(
                f"Found 0 unique temporal anomalies. Using fallbacks for all {size} requested anomalies."
            )
            fallback_indices = torch.arange(size, device=self.device)
            return (expanded_src[fallback_indices], expanded_tgt[fallback_indices],
                    expanded_t[fallback_indices], expanded_msg[fallback_indices])

        valid_encoded = encoded_cand[valid_indices]
        _, unique_indices = torch.unique(valid_encoded, return_inverse=True)
        final_unique_indices = valid_indices[unique_indices]
        num_found_unique = len(final_unique_indices)

        self.logger.info(
            f"Combinatorial search found {num_found_unique} unique temporal anomalies "
            f"from a pool of {len(cand_indices)}."
        )

        if num_found_unique >= size:
            final_indices = final_unique_indices[:size]
        else:
            self.logger.warning(
                f"Found {num_found_unique} unique temporal anomalies, but {size} were requested. "
                "Using fallbacks. Consider increasing `edge_num_tries` or `timestamp_num_tries`."
            )
            num_missing = size - num_found_unique
            fallback_indices = torch.arange(num_missing, device=self.device)
            final_indices = torch.cat([final_unique_indices, fallback_indices])

        anom_src = expanded_src[final_indices]
        anom_tgt = expanded_tgt[final_indices]
        anom_t = expanded_t[final_indices]
        anom_msg = expanded_msg[final_indices]

        return anom_src, anom_tgt, anom_t, anom_msg

    def generate_anomalous_edges(
        self,
        anom_type: str,
        anom_train_ratio: float = 0.0,
        anom_val_ratio: float = 0.0,
        anom_test_ratio: float = 0.05,
        reset_labels: bool = True,
        e_sample_size: int = 100,
        t_sample_size: int = 100,
        random_time_walk_max_steps: int = 5,
        struct_window_size: Optional[int | float] = 1_000
    ) -> TemporalGraphData:
        anom_type = self._validate_and_get_anomaly_type(anom_type)
        anom_ratios = {"train": anom_train_ratio,
                       "test": anom_test_ratio, "val": anom_val_ratio}

        gen_anom: dict[str, tuple[torch.Tensor, ...]] = {}

        for split, anom_ratio in anom_ratios.items():
            if anom_ratios == 0.0:
                continue

            if not 0.0 <= anom_ratio <= 1.0:
                raise ValueError(f"Invalid anomaly ratio for '{split}' split.")

            if split == "train":
                mask = self.temporal_graph.train_mask
            elif split == "test":
                mask = self.temporal_graph.test_mask
            else:
                mask = self.temporal_graph.val_mask

            if not mask.any():
                self.logger.info(
                    f"Split '{split}' is empty, skipping anomaly injection.")
                continue

            src = self.temporal_graph.src[mask]
            tgt = self.temporal_graph.tgt[mask]
            t = self.temporal_graph.t[mask]
            msg = self.temporal_graph.msg[mask]

            num_anom = int(src.shape[0] * anom_ratio)
            if num_anom == 0:
                continue

            self.logger.info(
                f"Generating {num_anom} '{anom_type}' anomalous edges in '{split}' split...")

            first_t, last_t = t.min(), t.max()
            edges = torch.stack([src, tgt], dim=1)
            edge_p = compute_unique_inverse_count_probabilities(
                edges, device=self.device)

            if anom_type == "structural":
                anom_src, anom_tgt, anom_t, anom_msg = self._generate_structural_anomalies(
                    num_anom, src, tgt, t, msg, window_size=struct_window_size)

            elif anom_type == "temporal":
                anom_src, anom_tgt, anom_t, anom_msg = self._generate_temporal_anomalies(
                    num_anom, src, tgt, t, msg, edge_p, first_t, last_t, e_sample_size, t_sample_size, random_time_walk_max_steps)

            elif anom_type == "contextual":
                pass

            elif anom_type == "structural-contextual":
                pass

            elif anom_type == "temporal-contextual":
                pass

            elif anom_type == "temporal-structural-contextual":
                pass

            else:
                raise NotImplementedError(
                    f"Generation for anomaly type '{anom_type}' is not implemented.")

            if anom_src.numel() > 0:
                gen_anom[split] = (
                    anom_src, anom_tgt, anom_t, anom_msg
                )
                self._add_to_encoded_observed_edges(anom_src, anom_tgt, anom_t)

        src_list, tgt_list, t_list, msg_list = [], [], [], []
        train_mask_list, val_mask_list, test_mask_list = [], [], []

        for split, (anom_src, anom_tgt, anom_t, anom_msg) in gen_anom.items():
            num_anom_in_split = anom_msg.numel()

            src_list.append(anom_src)
            tgt_list.append(anom_tgt)
            t_list.append(anom_t)
            msg_list.append(anom_msg)

            train_mask_list.append(torch.full(
                (num_anom_in_split,), split == "train", dtype=torch.bool, device=self.device))
            val_mask_list.append(torch.full(
                (num_anom_in_split,), split == "val", dtype=torch.bool, device=self.device))
            test_mask_list.append(torch.full(
                (num_anom_in_split,), split == "test", dtype=torch.bool, device=self.device))

        all_anom_src = torch.cat(src_list)
        all_anom_tgt = torch.cat(tgt_list)
        all_anom_t = torch.cat(t_list)
        all_anom_msg = torch.cat(msg_list)

        anom_train_mask = torch.cat(train_mask_list)
        anom_val_mask = torch.cat(val_mask_list)
        anom_test_mask = torch.cat(test_mask_list)

        normal_labels = (self.temporal_graph.edge_labels if not reset_labels
                         else torch.zeros(self.temporal_graph.num_edges, dtype=torch.long, device=self.device))
        anom_labels = torch.ones(
            all_anom_src.numel(), dtype=torch.long, device=self.device)

        final_src = torch.cat([self.temporal_graph.src, all_anom_src])
        final_tgt = torch.cat([self.temporal_graph.tgt, all_anom_tgt])
        final_t = torch.cat([self.temporal_graph.t, all_anom_t])
        final_msg = torch.cat([self.temporal_graph.msg, all_anom_msg])

        final_label = torch.cat([normal_labels, anom_labels])

        final_train_mask = torch.cat(
            [self.temporal_graph.train_mask, anom_train_mask])
        final_val_mask = torch.cat(
            [self.temporal_graph.val_mask, anom_val_mask])
        final_test_mask = torch.cat(
            [self.temporal_graph.test_mask, anom_test_mask])

        sort_indices = torch.argsort(final_t)

        output_graph = TemporalGraphData(
            src=final_src[sort_indices],
            tgt=final_tgt[sort_indices],
            t=final_t[sort_indices],
            msg=final_msg[sort_indices],
            edge_labels=final_label[sort_indices],
            node_attr=self.temporal_graph.node_attr,
            node_labels=self.temporal_graph.node_labels,
            train_mask=final_train_mask[sort_indices],
            val_mask=final_val_mask[sort_indices],
            test_mask=final_test_mask[sort_indices],
            metadata=self.temporal_graph.metadata
        )

        self.logger.info(
            f"Edge anomaly injection complete. New graph has {output_graph.num_edges} total edges.")

        return output_graph

    def generate_anomalous_nodes(
            self,
            anom_train_ratio: float = 0.0,
            anom_val_ratio: float = 0.0,
            anom_test_ratio: float = 0.05
    ):
        pass
