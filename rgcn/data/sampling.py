from __future__ import annotations

import gc
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp, random as jrandom

from rgcn.data.utils import make_dense_relation_neighbors


def make_dense_batched_negative_sample(edge_index, edge_type, num_nodes, num_edges):
    dense_tensor = make_dense_relation_neighbors(edge_index=np.array(edge_index), edge_type=np.array(edge_type), num_nodes=np.array(num_nodes))
    dense_tensor = jnp.array(dense_tensor, dtype=jnp.int32)

    # dense_tensor: [n_relations, num_nodes, max_num_neighbors]

    @partial(jax.vmap, in_axes=1, out_axes=0)
    def isin(triple):
        head, tail, edge_type = triple
        dense_edge_index = dense_tensor[edge_type, head]
        return jnp.isin(tail, dense_edge_index)

    def perform(key):
        # Always generate one negative sample per positive sample
        rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]
        rand2 = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

        head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]

        maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)],
                                                 axis=0)  # [3, num_edges]

        positive_mask = isin(maybe_negative_triples).reshape(1, num_edges)  # [1, num_edges]
        head_or_tail_positive = head_or_tail * positive_mask  # [2, num_edges]

        definitely_negative_samples = jnp.where(head_or_tail_positive, rand2, maybe_negative_samples)  # [2, num_edges]
        return definitely_negative_samples

    return perform