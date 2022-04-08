from __future__ import annotations

import jax
from jax import numpy as jnp, random as jrandom

from rgcn.data.datatypes import BasicModelData


def make_dense_batched_negative_sample(all_data: BasicModelData, pos_idx: jnp.ndarray, num_nodes: int):
    """Perform negative sampling on a batched edge index."""

    num_edges = pos_idx.shape[0]

    @jax.jit
    def perform(key):
        pos_edge_idx = all_data[pos_idx].edge_index
        # Always generate one negative sample per positive sample
        rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

        head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, pos_edge_idx)  # [2, num_edges]
        return maybe_negative_samples

    return perform
