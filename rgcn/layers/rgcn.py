from functools import partial
from typing import Literal, Union, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom
import optax

jax.config.update('jax_log_compiles', True)


class RelLinear(eqx.Module):
    weights: jnp.ndarray

    def __init__(self, in_channels, out_channels, n_relations, key):
        self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, in_channels, out_channels))

    def __getitem__(self, relation_index: int):
        return self.weights[relation_index]

    def l2_loss(self):
        return jnp.sum(jnp.square(self.weights))


class DecomposedRelLinear(eqx.Module):
    bases: jnp.ndarray
    base_weights: jnp.ndarray

    def __init__(self, in_features, out_features, num_relations, num_bases, key):
        initializer = jax.nn.initializers.glorot_uniform()
        self.bases = initializer(key, (num_bases, in_features, out_features))
        self.base_weights = initializer(key, (num_relations, num_bases))

    def __getitem__(self, relation_index: int):
        return jnp.einsum('b,bio->io', self.base_weights[relation_index], self.bases)

    def l2_loss(self):
        return jnp.sum(jnp.square(self.bases))


class RGCNConv(eqx.Module):
    self_weight: jnp.ndarray
    relation_weights: Union[RelLinear, DecomposedRelLinear]
    in_channels: int
    out_channels: int
    n_relations: int

    def __init__(self, in_channels, out_channels, n_relations,
                 decomposition_method: Literal['none', 'basis', 'block'], n_decomp: Optional[int], key):
        super().__init__()
        sw_key, rel_key = jrandom.split(key)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        glorot_initializer = jnn.initializers.glorot_uniform()

        self.self_weight = glorot_initializer(sw_key, (in_channels, out_channels), jnp.float32)

        if decomposition_method == 'none':
            self.relation_weights = RelLinear(in_channels, out_channels, n_relations, rel_key)
        elif decomposition_method == 'basis':
            self.relation_weights = DecomposedRelLinear(in_channels, out_channels, n_relations, n_decomp, rel_key)
        elif decomposition_method == 'block':
            raise NotImplementedError

    def get_self_transform(self, x: Optional[jnp.ndarray] = None):
        if x is None:
            # x is an identity matrix: [num_nodes, num_nodes] = [in_channels, in_channels]
            self_transform = self.self_weight
        else:
            self_transform = jnp.matmul(x, self.self_weight)
        return self_transform

    def get_work_relation(self, x, edge_type_idcs, edge_masks):
        @jax.jit
        def work_relation(rel, state):
            prev_out = state
            # edge_index[0] is the source node index
            # edge_mask = edge_type == rel
            # rel_edge_index = edge_index[:, edge_type_idcs == rel]
            edge_mask = edge_masks[rel].reshape((-1, 1))
            rel_edge_index = edge_type_idcs[rel]

            if x is None:
                # x is the identity matrix
                out_rel = self.relation_weights[rel]
            else:
                out_rel = jnp.matmul(x, self.relation_weights[rel])

            # Scatter the out_rel to the target nodes
            out_term = jnp.zeros(prev_out.shape).at[rel_edge_index[1], :].add(jnp.where(edge_mask, out_rel[rel_edge_index[0], :], 0), )
            n_input_edges = jnp.zeros((prev_out.shape[0], 1)).at[rel_edge_index[1]].add(jnp.where(edge_mask, 1, 0),)
            out_term = jnp.where(n_input_edges == 0, out_term, out_term / n_input_edges)
            out = prev_out + out_term
            return out
        return work_relation

    def __call__(self, x, edge_type_idcs, edge_masks):
        # x: [num_nodes, in_channels]

        # out: [num_nodes, out_channels]
        out = self.get_self_transform(x)

        # self_transform: [num_nodes, out_channels]

        # edge_index: [2, num_edges]
        work_relation = self.get_work_relation(x, edge_type_idcs, edge_masks)

        # Transform all the nodes using each relation transform
        out = jax.lax.fori_loop(0, self.n_relations, work_relation, out)

        return out

    def single_relation(self, x, edge_type_idcs, edge_masks, rel):
        work_relation = self.get_work_relation(x, edge_type_idcs, edge_masks)
        num_nodes = self.in_channels if x is None else x.shape[0]
        out = jnp.zeros_like(num_nodes, self.out_channels)
        return work_relation(rel, out)

    def l2_loss(self):
        return jnp.sum(jnp.square(self.self_weight)) + self.relation_weights.l2_loss()


def test_rgcn():
    x = jnp.ones((5, 3))
    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_type = jnp.array([0, 0, 1, 1, 1])
    rgcn = RGCNConv(3, 2, 2, jrandom.PRNGKey(0))
    out = rgcn(x, edge_index, edge_type)
    print(out)


def test_rgcn_none():
    x = None
    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_type = jnp.array([0, 0, 1, 1, 1])
    rgcn = RGCNConv(3, 2, 2, jrandom.PRNGKey(0))
    out = rgcn(x, edge_index, edge_type)
    print(out)


def test_rgcn_train():
    x = None

    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_type = jnp.array([0, 0, 1, 1, 1])

    optimizer = optax.adam(1e-3)
    rgcn = RGCNConv(3, 2, 2, jrandom.PRNGKey(0))
    loss = optax.l2_loss

    # @jax.jit
    # @jax.grad
    # def loss_fn():
