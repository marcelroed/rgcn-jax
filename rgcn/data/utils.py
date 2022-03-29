from typing import Tuple

import numpy as np
import torch
import jax.numpy as jnp
import chex


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


def make_dense_relation_edges(edge_index: jnp.ndarray, edge_type: jnp.ndarray, num_relations: int
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:

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
