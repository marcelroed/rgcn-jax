from __future__ import annotations

from jax import numpy as jnp, random as jrandom


def make_dense_batched_negative_sample(edge_index, num_nodes, num_edges):
    """Perform negative sampling on a batched edge index."""

    def perform(key):
        # Always generate one negative sample per positive sample
        rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

        head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]
        return maybe_negative_samples

    return perform
