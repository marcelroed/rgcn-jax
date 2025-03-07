from typing import Union, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from einops import rearrange
from typing_extensions import Literal


# jax.config.update('jax_log_compiles', True)
from rgcn.data import BasicGraphData


class RelLinear(eqx.Module):
    weights: jnp.ndarray

    def __init__(self, in_channels, out_channels, n_relations, key):
        self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, in_channels, out_channels))

    def __getitem__(self, relation_index: int):
        return self.weights[relation_index]

    def apply(self, rel, x):
        if x is None:
            return self._apply_id(rel)
        return jnp.matmul(x, self[rel])

    def _apply_id(self, rel):
        """Apply to identity matrix."""
        return self[rel]

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

    def apply(self, rel, x):
        if x is None:
            return self._apply_id(rel)
        return jnp.matmul(x, self[rel])

    def _apply_id(self, rel):
        return self[rel]

    def l2_loss(self):
        return jnp.sum(jnp.square(self.bases))


class BlockRelLinear(eqx.Module):
    blocks: jnp.ndarray
    remainder_block: Optional[jnp.ndarray]
    in_features: int
    out_features: int

    def __init__(self, in_features, out_features, num_relations, num_blocks, key):
        initializer = jax.nn.initializers.glorot_uniform()

        assert out_features % num_blocks == 0, 'Block size must divide out_features'

        out_block_size = out_features // num_blocks

        in_features_remainder = in_features % num_blocks
        in_block_size = in_features // num_blocks

        self.in_features = in_features
        self.out_features = out_features

        if in_features_remainder != 0:
            # If block_size doesn't divide in_features we have to make the last block
            key1, key2 = jrandom.split(key)
            self.blocks = initializer(key1, (num_relations, num_blocks - 1, in_block_size, out_block_size))
            self.remainder_block = initializer(key2,
                                               (num_relations, in_block_size + in_features_remainder, out_block_size))
        else:
            self.remainder_block = None
            self.blocks = initializer(key, (num_relations, num_blocks, in_block_size, out_block_size))

    def apply(self, rel, x):
        if x is None:
            return self._apply_id(rel)
        if self.remainder_block is None:
            # x: [num_points, in_features] -> [num_points, num_blocks, in_block_size]
            # where num_blocks * in_block_size = in_features
            x_by_block = rearrange(x, 'num_points (num_blocks in_block_size) -> num_points num_blocks in_block_size',
                                   num_blocks=self.blocks.shape[1])
            relation_blocks = self.blocks[rel]
            transformed = jnp.einsum('nio, pni -> pno', relation_blocks, x_by_block)
            # [1, 2, 3, 4, 5, 6]
            # [[1, 2], [3, 4], [5, 6]] -> [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
            # [0, 1, 2, 1, 2, 3, 2, 3, 4]
            return rearrange(transformed,
                             'num_points num_blocks out_block_size -> num_points (num_blocks out_block_size)')
        else:
            raise NotImplementedError

    def _apply_id(self, rel):
        # Have to construct the block matrix
        block_matrix = jnp.zeros((self.in_features, self.out_features))
        blocks = self.blocks[rel] if self.remainder_block is None else list(self.blocks[rel]) + [
            self.remainder_block[rel]]

        i, j = 0, 0
        for block in blocks:
            brows, bcols = block.shape
            block_matrix = block_matrix.at[i:i + brows, j:j + bcols].set(block)
            i += brows
            j += bcols

        return block_matrix

    def l2_loss(self):
        return jnp.sum(jnp.square(self.blocks)) + \
               (jnp.sum(jnp.square(self.remainder_block)) if self.remainder_block is not None else 0)


class RGCNConv(eqx.Module):
    self_weight: jnp.ndarray
    relation_weights: Union[RelLinear, DecomposedRelLinear, BlockRelLinear]
    in_channels: int
    out_channels: int
    n_relations: int
    dropout_rate: Optional[float]
    normalizing_constant: Literal['per_relation_node', 'per_node', 'none']

    def __init__(self, in_channels, out_channels, n_relations,
                 decomposition_method: Literal['none', 'basis', 'block'], n_decomp: Optional[int],
                 normalizing_constant: Literal['per_relation_node', 'per_node', 'none'], dropout_rate: Optional[float],
                 key):
        sw_key, rel_key = jrandom.split(key)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.normalizing_constant = normalizing_constant
        self.dropout_rate = dropout_rate
        initializer = jnn.initializers.he_normal()

        self.self_weight = initializer(sw_key, (in_channels, out_channels), jnp.float32)

        if decomposition_method == 'none':
            self.relation_weights = RelLinear(in_channels, out_channels, n_relations, rel_key)
        elif decomposition_method == 'basis':
            self.relation_weights = DecomposedRelLinear(in_channels, out_channels,
                                                        n_relations, num_bases=n_decomp, key=rel_key)
        elif decomposition_method == 'block':
            self.relation_weights = BlockRelLinear(in_channels, out_channels, n_relations,
                                                   num_blocks=n_decomp, key=rel_key)

    def get_self_transform(self, x: Optional[jnp.ndarray] = None, key=None):
        if x is None:
            # x is an identity matrix: [num_nodes, num_nodes] = [in_channels, in_channels]
            self_transform = self.self_weight  # [num_nodes=in_channels, out_channels]
        else:
            self_transform = jnp.matmul(x, self.self_weight)  # [num_nodes, out_channels]

        # Ignore the self-transform for some elements if dropout_rate is enabled
        if self.dropout_rate and key is not None:
            self_transform = jnp.where(
                jrandom.bernoulli(key, self.dropout_rate, (self_transform.shape[0], 1)),
                self_transform / self.dropout_rate,
                0
            )

        return self_transform

    def _apply_normalizing_constant(self, edge_mask, out_term, prev_out, rel_edge_index, node_normalizing_constant):
        if self.normalizing_constant == 'per_relation_node':
            n_input_edges = jnp.zeros((prev_out.shape[0], 1)).at[rel_edge_index[1]].add(
                jnp.where(edge_mask, 1, 0), )
            out_term = jnp.where(n_input_edges == 0, out_term, out_term / n_input_edges)
        elif self.normalizing_constant == 'per_node':
            out_term = out_term / node_normalizing_constant.reshape(-1, 1)
        return out_term

    def _setup_normalizing_constant(self, edge_masks, edge_type_idcs, x):
        node_normalizing_constant = None
        if self.normalizing_constant == 'per_node':
            num_nodes = x.shape[0] if x is not None else self.in_channels
            flattened_edge_idcs = rearrange(edge_type_idcs, 'relations node edges -> node (relations edges)')
            flattened_edge_mask = rearrange(edge_masks, 'relations edges -> (relations edges)')
            node_normalizing_constant = jnp.zeros((num_nodes,)).at[flattened_edge_idcs[1]].add(
                jnp.where(flattened_edge_mask, 1, 0))
            node_normalizing_constant = jnp.where(node_normalizing_constant == 0, 1, node_normalizing_constant)
        return node_normalizing_constant

    def get_work_relation(self, x, edge_type_idcs, edge_masks):
        node_normalizing_constant = self._setup_normalizing_constant(edge_masks, edge_type_idcs, x)

        @jax.jit
        def work_relation(rel, state):
            prev_out = state

            edge_mask = edge_masks[rel].reshape((-1, 1))
            rel_edge_index = edge_type_idcs[rel]

            out_rel = self.relation_weights.apply(rel, x)

            # Scatter the out_rel to the target nodes
            out_term = (
                jnp.zeros(prev_out.shape)  # Start off with zeros
                    .at[rel_edge_index[1], :]  # At each object
                    .add(
                        jnp.where(edge_mask, out_rel[rel_edge_index[0], :], 0)  # Scatter the output from the transform
                    )
            )

            # Normalize
            out_term = rel.apply_normalizing_constant(edge_mask, out_term, prev_out, rel_edge_index,
                                                      node_normalizing_constant)

            out = prev_out + out_term
            return out

        return work_relation

    def __call__(self, x, edge_type_idcs, edge_masks, key):
        # x: [num_nodes, in_channels]

        # out: [num_nodes, out_channels]
        out = self.get_self_transform(x, key)  # Self loops

        # Function to add results from a single relation
        work_relation = self.get_work_relation(x, edge_type_idcs, edge_masks)

        # Loop through all relations and add the results to out.
        out = jax.lax.fori_loop(0, self.n_relations, work_relation, out)

        return out

    def l2_loss(self):
        return jnp.sum(jnp.square(self.self_weight)) + self.relation_weights.l2_loss()


class FastRGCNConv(eqx.Module):
    self_weights: jnp.ndarray  # [in_channels, out_channels]
    relation_weights: Union[RelLinear, DecomposedRelLinear, BlockRelLinear]
    n_relations: int
    in_channels: int
    out_channels: int

    def __init__(self, in_channels, out_channels, n_relations,
                 decomposition_method: str = 'basis', n_decomp=2, key=None):
        sw_key, rel_key = jrandom.split(key)

        initializer = jnn.initializers.he_normal()

        self.self_weight = initializer(sw_key, (in_channels, out_channels), jnp.float32)
        self.n_relations = n_relations

        if decomposition_method == 'basis':
            self.relation_weights = DecomposedRelLinear(in_channels, out_channels, n_relations, n_decomp, rel_key)
        else:
            raise NotImplementedError

    def get_self_transform(self, x):
        return self.self_weight if x is None else self.self_weight @ x

    def get_relation_transform(self, x, graph_data: BasicGraphData):
        def work_relation(rel, state):
            """
            Want to scatter points with a transform corresponding to the correct relation

            First we scatter our features onto the tail, adding the results.
            """
            rel_mask = graph_data.edge_type == rel

            # [num_nodes, in_channels]
            scattered_to = (
                jnp.zeros_like(x)
                   .at[graph_data.edge_idx[1, :]].add(jnp.where(rel_mask, x[graph_data.edge_idx[0, :]], 0))
            )

            # Transform
            out = self.relation_weights.apply(rel, scattered_to)  # [num_nodes, out_channels]
            return state + out
        return jax.lax.fori_loop(0, self.n_relations, work_relation, jnp.zeros((x.shape[0], self.out_channels)))

    def __call__(self, x, graph_data: BasicGraphData):
        assert graph_data.sorted_by == 'tail'
        out = self.get_self_transform(x)
        out = out + self.get_relation_transform(x, graph_data)
        return out

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


def test_rgcnconv_block_decomp():
    print()
    rgcn = RGCNConv(3, 2, 2, decomposition_method='block', n_decomp=1, normalizing_constant='per_node',
                    dropout_rate=None, key=jrandom.PRNGKey(0))
    relation_edge_idcs = jnp.array([[[0, 1, 2], [1, 2, 0]], [[1, 2, -1], [2, 0, -1]]])
    result = rgcn(None, relation_edge_idcs, jnp.min(relation_edge_idcs, axis=1) == -1, key=jrandom.PRNGKey(0))
    print(result)


def test_block_rel_linear():
    print()
    n_points = 1
    in_dim = 4
    out_dim = 4
    rel = 0
    n_relations = 2
    n_blocks = 2
    decomp = BlockRelLinear(in_dim, out_dim, n_relations, n_blocks, jrandom.PRNGKey(0))

    print('Blocks for relation 0')
    print(decomp.blocks[0])
    print('\nBlock diag')
    print(decomp.apply_id(0))

    shape = (n_points, in_dim)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    result = decomp.apply(rel, x)

    # Apply the blocks separately and concatenate
    manual_result = jnp.concatenate((x[:, :2] @ decomp.blocks[0, 0], x[:, 2:] @ decomp.blocks[0, 1]), axis=1)

    assert jnp.allclose(manual_result, result)

    print(result)
    print(manual_result)
