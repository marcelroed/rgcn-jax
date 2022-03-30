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
        # self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, n_channels))

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

    #def call_dense(self, x, rel_edge_index, rel_edge_mask):
    #    n_relations, _, max_edges_per_relation = rel_edge_index.shape
    #    # rel_edge_index: [n_relations, 2, max_edges_per_relation]
    #
    #    expanded_edge_mask = rel_edge_mask.reshape(n_relations, max_edges_per_relation, 1)  # [n_relations, max_edges_per_relation, 1]
    #
    #    s = jnp.where(expanded_edge_mask, x[rel_edge_index[:, 0, :]], 0)  # [n_relations, max_edges_per_relation, n_channels]
    #    s = jnp.where(expanded_edge_mask, s / jnp.linalg.norm(s, axis=2, keepdims=True), 0)  # [n_relations, max_edges_per_relation, n_channels]
    #
    #    o = jnp.where(expanded_edge_mask, x[rel_edge_index[:, 1, :]], 0)
    #    o = jnp.where(expanded_edge_mask, o / jnp.linalg.norm(o, axis=2, keepdims=True), 0)
    #
    #    result = jnp.einsum('rec, rc, rec -> re', s, self.weights, o)
    #    return result


class ComplEx(eqx.Module):
    weights: jnp.ndarray
    n_relations: int

    def __init__(self, n_relations, n_channels, key):
        self.n_relations = n_relations
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels), dtype=jnp.complex64)

    def __call__(self, x, edge_index, edge_type):
        # O(n_edges * n_channels) to compute
        # x: [n_nodes, n_channels]
        # edge_type_idcs: [n_relations, 2, n_edges_per_relation_max]
        # edge_type_idcs_mask: [n_edges_per_relation_max]

        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
        r = self.weights[edge_type, :]  # [n_edges, n_channels]
        o = jnp.conjugate(x[edge_index[1, :]])  # [n_edges, n_channels]
        o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return jnp.sum(s * r * o, axis=1)


def test_distmult():
    distmult = DistMult(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[1, 2], [3, 4]])
    edge_index = jnp.array([[0, 1], [0, 1]])
    edge_type = jnp.array([0, 1])
    distmult(x, edge_index, edge_type)


def test_complex():
    complex = ComplEx(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[1 + 2j, 2 + 3j], [3 - 1j, 4 + 10j]])
    edge_index = jnp.array([[0, 1], [0, 1]])
    edge_type = jnp.array([0, 1])
    print(complex(x, edge_index, edge_type))