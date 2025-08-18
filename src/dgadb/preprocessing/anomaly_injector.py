import logging
import torch
import torch.nn.functional as F
from math import gcd
from functools import reduce
from typing import Literal, Optional
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

    # def _build_encoded_lookup(self, tensors: list[torch.Tensor], base: torch.Tensor):
    #     combined = torch.stack(tensors, dim=1)
    #     unique_rows = torch.unique(combined, dim=0)
    #     encoded_values = torch.zeros(
    #         unique_rows.shape[0], dtype=torch.int64, device=self.device)

    #     num_components = len(tensors)
    #     unpacked_tensors = unique_rows.unbind(dim=1)

    #     for i, component_tensor in enumerate(unpacked_tensors):
    #         power = num_components - 1 - i
    #         encoded_values += (component_tensor.long() * (base**power)).long()

    #     return encoded_values.long()

    # def _build_edge_lookups(self, src: torch.Tensor, tgt: torch.Tensor, t: torch.Tensor) -> tuple[set, set]:
    #     self.logger.info("Building edge lookups...")
    #     unique_edges = set(map(tuple, torch.unique(
    #         torch.stack([src, tgt], dim=1), dim=0).tolist()))
    #     unique_edges_t = set(map(tuple, torch.unique(
    #         torch.stack([src, tgt, t], dim=1), dim=0)))
    #     self.logger.info(
    #         f"Unique edges: {len(unique_edges)}, unique edges with timestamps: {len(unique_edges_t)}.")

    #     return unique_edges, unique_edges_t

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
            t_deltas_p = self._compute_unique_inverse_count_probabilities(
                t_deltas)

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

    def _compute_unique_inverse_count_probabilities(
            self,
            items: torch.Tensor,
            power: float = -1.5
    ) -> torch.Tensor:
        _, inverse_indices, counts = torch.unique(
            items, dim=0, return_inverse=True, return_counts=True)
        unique_weights = torch.pow(counts, power)
        probs = unique_weights / (unique_weights.sum() + 1e-10)
        mapped_probs = probs[inverse_indices]
        return mapped_probs.to(self.device)

    # def _compute_edge_probabilities(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    #     edges = torch.stack([src, tgt], dim=1)
    #     probs, unique_inverse_indices = \
    #         self._compute_unique_inverse_count_probabilities(edges)
        # return probs[unique_inverse_indices]

    # def _compute_edge_probabilities(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    #     # 'inverse_indices' maps each edge in the original 'edges' tensor to its index in 'unique_edges'
    #     edges = torch.stack([src, tgt], dim=1)
    #     _, inverse_indices, counts = torch.unique(
    #         edges, dim=0, return_inverse=True, return_counts=True)
    #     # (1/counts) * (1/sqrt(counts))
    #     unique_weights = torch.pow(counts, -1.5)
    #     edge_weights = unique_weights[inverse_indices]
    #     return edge_weights / (edge_weights.sum() + 1e-10)

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

        # delta_indices = torch.randint(
        #     self.t_deltas_size,
        #     size=(total_steps,),
        #     device=first_t.device
        # )
        # t_deltas = self.t_deltas[delta_indices]

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

    def _generate_temporal_anomalies_new(
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
        _, unique_indices = torch.unique(
            valid_encoded, return_inverse=False, return_counts=False)
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

        # encoded_anom = self._encode_edges(src=anom_src, tgt=anom_tgt, t=anom_t)
        # self.observed_edges_encoded_t = torch.cat([
        #     self.observed_edges_encoded_t,
        #     encoded_anom
        # ])

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
            max_num_tries: int = 100,
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

        indices = torch.multinomial(edge_p, num_samples=size, replacement=True)
        base_src, base_tgt, base_msg = src[indices], tgt[indices], msg[indices]

        num_candidates = size * max_num_tries
        candidate_timestamps = self._generate_plausible_timestamps(
            num_candidates,
            first_t,
            last_t,
            random_time_walk_max_steps,
        )
        # Reshape to: (size, max_num_tries)
        candidate_timestamps_reshaped = candidate_timestamps.view(
            size, max_num_tries)

        # Shape: (size * max_num_tries)
        expanded_src = base_src.repeat_interleave(max_num_tries)
        expanded_tgt = base_tgt.repeat_interleave(max_num_tries)

        candidate_encoded = self._encode_edges(
            expanded_src,
            expanded_tgt,
            t=candidate_timestamps,
            find_unique=False
        )

        is_unique_mask = ~torch.isin(
            candidate_encoded, self.encoded_observed_edges_t)
        is_unique_mask = is_unique_mask.view(size, max_num_tries)

        # Add a fallback column of all 'True's to handle cases where no unique timestamp is found
        fallback_indices = torch.full((size, 1), True, device=self.device)
        mask_with_fallback = torch.cat(
            [is_unique_mask, fallback_indices], dim=1)

        first_valid_indices = torch.argmax(mask_with_fallback.byte(), dim=1)
        failed_mask = (first_valid_indices == max_num_tries)

        if failed_mask.any():
            self.logger.warning(
                f"{failed_mask.sum().item()} temporal anomalies were not unique after {max_num_tries} attempts. Using last candidate as fallback.")
            # 'argmax' points to the fallback column
            # Point to the last candidate
            first_valid_indices[failed_mask] = max_num_tries - 1

        anom_t = candidate_timestamps_reshaped[torch.arange(
            size, device=self.device), first_valid_indices]

        return base_src, base_tgt, anom_t, base_msg

    def generate_anomalous_edges(
        self,
        anom_type: str,
        anom_train_ratio: float = 0.0,
        anom_val_ratio: float = 0.0,
        anom_test_ratio: float = 0.05,
        reset_labels: bool = True,
        e_num_tries: int = 100,
        t_num_tries: int = 100,
        random_time_walk_max_steps: int = 5
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
                f"Generating anomalous {num_anom} '{anom_type}' edges in '{split}' split...")

            first_t, last_t = t.min(), t.max()
            edges = torch.stack([src, tgt], dim=1)
            edge_p = self._compute_unique_inverse_count_probabilities(edges)

            if anom_type == "structural":
                pass

            elif anom_type == "temporal":
                anom_src, anom_tgt, anom_t, anom_msg = self._generate_temporal_anomalies_new(
                    num_anom, src, tgt, t, msg, edge_p, first_t, last_t, e_num_tries, t_num_tries, random_time_walk_max_steps)

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
                    anom_src, anom_tgt, anom_t, anom_msg)
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
