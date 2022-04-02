from __future__ import annotations

import gc
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp, random as jrandom

from rgcn.data.datasets.entity_classification import make_dense_relation_tensor
from rgcn.data.utils import make_dense_relation_neighbors


def make_dense_batched_negative_sample(edge_index, edge_type, num_nodes, num_edges):
    # dense_tensor = make_dense_relation_neighbors(edge_index=np.array(edge_index), edge_type=np.array(edge_type), num_nodes=np.array(num_nodes))
    # dense_tensor = jnp.array(dense_tensor, dtype=jnp.int32)


    # dense_tensor: [n_relations, num_nodes, max_num_neighbors]

    # @partial(jax.vmap, in_axes=1, out_axes=0)
    # def isin(triple):
    #     head, tail, edge_type = triple
    #     dense_edge_index = dense_tensor[edge_type, head]
    #     return jnp.isin(tail, dense_edge_index)

    def perform(key):
        # Always generate one negative sample per positive sample
        rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]
        # rand2 = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

        head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]
        return maybe_negative_samples

        # maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)],
        #                                          axis=0)  # [3, num_edges]
        #
        # positive_mask = isin(maybe_negative_triples).reshape(1, num_edges)  # [1, num_edges]
        # head_or_tail_positive = head_or_tail * positive_mask  # [2, num_edges]
        #
        # definitely_negative_samples = jnp.where(head_or_tail_positive, rand2, maybe_negative_samples)  # [2, num_edges]
        # return definitely_negative_samples

    return perform


def make_dense_batched_negative_sample_dense_rel(edge_index, edge_type, num_nodes):
    num_relations = edge_type.max() + 1
    np_edge_index, np_edge_type = np.array(edge_index), np.array(edge_type)
    dense_tensor = make_dense_relation_neighbors(edge_index=np.array(edge_index), edge_type=np.array(edge_type),
                                                 num_nodes=num_nodes)
    dense_tensor = jnp.array(dense_tensor, dtype=jnp.int32)

    dense_rel_edge_idx, dense_rel_mask = make_dense_relation_tensor(edge_index=np_edge_index, edge_type=np_edge_type, num_relations=num_relations)
    n_relations, _, max_edge_index_length = dense_rel_edge_idx.shape

    # dense_tensor: [n_relations, num_nodes, max_num_neighbors]

    @partial(jax.vmap, in_axes=0, out_axes=0)
    @partial(jax.vmap, in_axes=1, out_axes=0)
    def isin(triple):
        head, tail, edge_type = triple
        dense_edge_index = dense_tensor[edge_type, head]
        return jnp.isin(tail, dense_edge_index)

    def perform(key):
        # Always generate one negative sample per positive sample
        rand = jrandom.randint(key, (n_relations, 1, max_edge_index_length), 0, num_nodes)  # [n_relations, 1, max_edge_index_length]
        rand2 = jrandom.randint(key, (n_relations, 1, max_edge_index_length), 0, num_nodes)  # [n_relations, 1, max_edge_index_length]

        head_or_tail = jrandom.bernoulli(key, 0.5, (n_relations, 1, max_edge_index_length))  # [n_relations, 1, max_edge_index_length]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=1)  # [n_relations, 2, max_edge_index_length]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, dense_rel_edge_idx)  # [n_relations, 2, max_edge_index_length]

        rel_type = jnp.repeat(jnp.arange(n_relations)[:, None], axis=1, repeats=max_edge_index_length).reshape(n_relations, 1, max_edge_index_length)
        maybe_negative_triples = jnp.concatenate(
            [maybe_negative_samples, rel_type],
                 axis=1)  # [n_relations, 3, max_edge_index_length]

        positive_mask = isin(maybe_negative_triples).reshape(n_relations, 1, max_edge_index_length)  # [n_relations, 1, max_edge_index_length]
        head_or_tail_positive = head_or_tail * positive_mask  # [n_relations, 2, max_edge_index_length]

        definitely_negative_samples = jnp.where(head_or_tail_positive, rand2, maybe_negative_samples)  # [n_relations, 2, max_edge_index_length]

        return definitely_negative_samples

    return perform
