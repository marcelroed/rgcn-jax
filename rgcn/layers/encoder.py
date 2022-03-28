import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random


class DistMult(eqx.Module):
    weights: jnp.ndarray
    n_relations: int

    def __init__(self, n_relations, n_channels, key):
        self.n_relations = n_relations
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels))
        #self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        # O(n_edges * n_channels) to compute
        # x: [n_nodes, n_channels]
        # edge_type_idcs: [n_relations, 2, n_edges_per_relation_max]
        # edge_type_idcs_mask: [n_edges_per_relation_max]

        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
        r = self.weights[edge_type, :]  # [n_edges, n_channels]
        o = x[edge_index[1, :]]  # [n_edges, n_channels]
        o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return jnp.sum(s * r * o, axis=1)


def test_distmult():
    distmult = DistMult(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[1, 2], [3, 4]])
    edge_index = jnp.array([[0, 1], [0, 1]])
    edge_type = jnp.array([0, 1])
    distmult(x, edge_index, edge_type)
