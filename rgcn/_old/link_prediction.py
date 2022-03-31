from __future__ import annotations

import equinox as eqx
import jax

from jax import random as jrandom, numpy as jnp, random as random

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.data.utils import encode, encode_with_type
from rgcn.layers.decoder import Decoder


def negative_sample(edge_index, num_nodes, num_negatives, key):
    """Generates negative samples completely randomly (no corruption)"""
    # Generate random edges
    rand = jrandom.randint(key, (2, num_negatives), 0, num_nodes)  # [num_negatives, ]

    real_coding = encode(edge_index, num_nodes)  # [num_edges, ]
    rand_coding = encode(rand, num_nodes)  # [num_negatives, ]

    positive_samples_mask = jnp.isin(rand_coding, real_coding).reshape(1, num_negatives)  # [1, num_negatives]

    negative_samples = jnp.where(positive_samples_mask, jrandom.randint(key, (2, num_negatives), 0, num_nodes), rand)

    return negative_samples  # [2, num_negatives]


def batched_negative_sample(edge_index, edge_type, num_nodes, key):
    # Always generate one negative sample per positive sample
    num_edges = edge_index.shape[1]
    rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]
    rand2 = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

    head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
    head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

    maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]

    real_triples = jnp.concatenate([edge_index, edge_type.reshape(1, -1)], axis=0)  # [3, num_edges]
    encoded_real_triples = encode_with_type(real_triples, num_nodes, edge_type)  # [num_edges, ]

    maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)],
                                             axis=0)  # [3, num_edges]
    maybe_negative_encoded_triples = encode_with_type(maybe_negative_triples, num_nodes, edge_type)  # [num_edges, ]

    positive_samples_mask = jnp.isin(maybe_negative_encoded_triples, encoded_real_triples).reshape(1,
                                                                                                   num_edges)  # [1, num_edges]
    head_tail_adjusted_positive_samples_mask = head_tail_mask * positive_samples_mask  # [2, num_edges]
    definitely_negative_samples = jnp.where(head_tail_adjusted_positive_samples_mask, rand2,
                                            maybe_negative_samples)  # [2, num_edges]

    return definitely_negative_samples


def get_train_epoch_data(dataset: LinkPredictionWrapper, key):
    neg_edge_index = jnp.zeros((2, 0), dtype=jnp.int32)
    neg_edge_type = jnp.zeros(0, dtype=jnp.int32)

    pos_edge_type = dataset.edge_type[dataset.train_idx]
    pos_edge_index = dataset.edge_index[:, dataset.train_idx]
    # pos_count = dataset.train_idx.sum()

    for i in range(0, dataset.num_relations):
        iter_key, key = jrandom.split(key)
        rel_indices = pos_edge_type == i
        rel_edge_index = pos_edge_index[:, rel_indices]
        rel_count = rel_indices.sum()
        rel_neg_edge_index = negative_sample(rel_edge_index, dataset.num_nodes, rel_count, iter_key)
        neg_edge_index = jnp.concatenate((neg_edge_index, rel_neg_edge_index), axis=1)
        neg_edge_type = jnp.concatenate((neg_edge_type, jnp.ones(rel_count, dtype=jnp.int32) * i))

    full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
    full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)
    pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))
    return full_edge_index, full_edge_type, pos_mask


class ComplExComplex(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray

    def __init__(self, n_relations, n_channels, key):
        self.n_relations = n_relations
        key1, key2 = random.split(key, 2)
        init = jax.nn.initializers.normal()
        real, imag = init(key1, (n_relations, n_channels)), init(key2, (n_relations, n_channels))
        self.weights = jnp.stack((real, imag), axis=1)

    def __call__(self, x, edge_index, edge_type):
        x = jax.lax.complex(x[:, 0], x[:, 1])

        s = x[edge_index[0, :], :]  # [n_edges, n_channels]
        r = jax.lax.complex(self.weights[0], self.weights[1])[edge_type, :]  # [n_edges, n_channels]
        o = jnp.conjugate(x[edge_index[1, :], :])  # [n_edges, n_channels]

        return jnp.sum(s * r * o, axis=1).real

    def forward_heads(self, heads, edge_type, tail):
        heads = jax.lax.complex(heads[:, 0], heads[:, 1])
        tail = jax.lax.complex(tail[0], tail[1])
        weights_complex = jax.lax.complex(self.weights[0], self.weights[1])
        r = weights_complex[edge_type, :]
        # return jnp.sum(heads * (r * tail), axis=1)
        return jnp.einsum('ec,c,c->e', heads, r, jnp.conjugate(tail)).real

    def forward_tails(self, head, edge_type, tails):
        head = jax.lax.complex(head[0], head[1])
        tails = jax.lax.complex(tails[:, 0], tails[:, 1])

        weights_complex = jax.lax.complex(self.weights[0], self.weights[1])
        r = weights_complex[edge_type, :]
        # return jnp.sum((head * r) * tails, axis=1)
        return jnp.einsum('c,cd,ed->e', head, jnp.diag(r), jnp.conjugate(tails)).real
        # For some reason, this is faster than the einsum 'c,c,ec->e', which would make much more sense.

    def l2_loss(self):
        return jnp.sum(self.weights.real**2 + self.weights.imag**2)