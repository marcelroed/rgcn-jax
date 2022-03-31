from __future__ import annotations

import gc
from typing import Tuple

import numpy as np
import torch
import chex
from jax import numpy as jnp
from tqdm import trange

from rgcn.utils import memory


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


def make_dense_relation_edges(edge_index: jnp.ndarray, edge_type: jnp.ndarray, num_relations: int
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Make dense relations from edge_index and edge_type.
    Output is the edge_index matrix per relation padded with zeros, and a mask to indicate where the zeros are.
    Returns (rel_edge_index: [n_relations, max_edge_index_length],
             rel_edge_index_mask: [n_relations, max_edge_index_length])
    """

    max_edge_index_length = max([jnp.sum(edge_type == i) for i in range(num_relations)])

    padded = [np.pad(edge_index[:, i == edge_type], ((0, 0), (0, max_edge_index_length - (i == edge_type).sum()))) for i
              in range(num_relations)]

    # padded: (n_relations, max_edge_index_length, 2)

    mask_lengths = [np.sum(edge_type == i) for i in range(num_relations)]

    # masks: [n_relations, max_edge_index_length]
    masks = [np.concatenate((np.ones(ml, dtype=bool), np.zeros(max_edge_index_length - ml, dtype=bool)), axis=0) for ml
             in mask_lengths]

    return jnp.array(padded), jnp.array(masks)


def test_torch_to_jax():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    jax_array = torch_to_jax(tensor)
    chex.assert_trees_all_equal(jax_array, jnp.array([[1, 2, 3], [4, 5, 6]]))


def encode(edge_index, num_nodes):
    row, col = edge_index[0], edge_index[1]
    return row * num_nodes + col


def encode_with_type(edge_index, num_nodes, edge_type):
    row, col = edge_index[0], edge_index[1]
    return edge_type * num_nodes * num_nodes + row * num_nodes + col


@memory.cache
def make_dense_relation_neighbors(edge_index, edge_type, num_nodes):
    """
    Construct a dense adjacency list from the edge_index and edge_type.
    Cache the results.
    Output is (num_relations, num_nodes, max_num_neighbors)
    """
    n_relations = edge_type.max() + 1
    # Reshape the edge_index matrix (and edge_type) to [n_relations, num_nodes, max_num_neighbors]
    result = []
    max_num_neighbors = 0
    for relation in trange(n_relations, desc='Finding neighbors'):
        gc.collect()
        rel_edge_index = edge_index[:, edge_type == relation]
        rel_result = []
        for head in range(num_nodes):
            head_mask = rel_edge_index[0] == head
            tails_for_head = rel_edge_index[1, head_mask]
            max_num_neighbors = max(max_num_neighbors, tails_for_head.shape[0])
            rel_result.append(tails_for_head)
        result.append(rel_result)

    # Construct a result array
    result_tensor = np.zeros((n_relations, num_nodes, max_num_neighbors), dtype=int)

    # Pad the inner arrays to shape [max_num_neighbors]
    for relation in trange(n_relations, desc='Writing to tensor'):
        gc.collect()
        for head in range(num_nodes):
            head_edges = result[relation][head]
            result_tensor[relation][head] = np.pad(head_edges, (0, max_num_neighbors - head_edges.shape[0]), 'constant',
                                                   constant_values=-1)
    return result_tensor


class BaseConfig:
    name: str

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'