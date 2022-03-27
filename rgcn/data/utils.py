import torch
import jax.numpy as jnp
import chex


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


def test_torch_to_jax():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    jax_array = torch_to_jax(tensor)
    chex.assert_trees_all_equal(jax_array, jnp.array([[1, 2, 3], [4, 5, 6]]))