import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from abc import ABC, abstractmethod

class Decoder(ABC):
    @abstractmethod
    def __call__(self, x, edge_index, edge_type):
        # [n_nodes, n_channels], [2, num_edges], [num_edges]
        pass

    @abstractmethod
    def forward_heads(self, heads, edge_type, tail):
        # [n_nodes, n_channels], [], [n_channels]
        pass

    @abstractmethod
    def forward_tails(self, head, edge_type, tails):
        # [n_channels], [], [n_nodes, n_channels]
        pass

class DistMult(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray

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
        # return jnp.einsum('ec,ec,ec->e', s, r, o)

    def forward_heads(self, heads, edge_type, tail):
        r = self.weights[edge_type]
        # return jnp.sum(heads * (r * tail), axis=1)
        return jnp.einsum('ec,c,c->e', heads, r, tail)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        # return jnp.sum((head * r) * tails, axis=1)
        return jnp.einsum('c,cd,ed->e', head, jnp.diag(r), tails)
        # For some reason, this is faster than the einsum 'c,c,ec->e', which would make much more sense.


def test_distmult():
    distmult = DistMult(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[1, 2], [3, 4]])
    edge_index = jnp.array([[0, 1], [0, 1]])
    edge_type = jnp.array([0, 1])
    distmult(x, edge_index, edge_type)


class RESCAL(eqx.Module, Decoder):
    n_relations: int
    n_channels: int
    weights: jnp.ndarray

    def __init__(self, n_relations, n_channels, key):
        self.n_relations = n_relations
        self.n_channels = n_channels
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels, n_channels))
        # self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        # O(n_edges * n_channels) to compute
        # x: [n_nodes, n_channels]
        # edge_type_idcs: [n_relations, 2, n_edges_per_relation_max]
        # edge_type_idcs_mask: [n_edges_per_relation_max]

        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
        o = x[edge_index[1, :]]  # [n_edges, n_channels]
        o = o / jnp.linalg.norm(o, axis=1, keepdims=True)
        r = self.weights[edge_type]  # [n_edges, n_channels, n_channels]

        return (s.reshape(-1, self.n_channels, 1) *
                jnp.matmul(r.reshape(-1, self.n_channels, self.n_channels),
                           o.reshape(-1, self.n_channels, 1))).sum(axis=1).reshape(-1)

    def forward_heads(self, heads, edge_type, tail):
        r = self.weights[edge_type]
        return jnp.einsum('ec,cd,d->e', heads, r, tail)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        return jnp.einsum('c,cd,ed->e', head, r, tails)


class TransE(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray

    def __init__(self, n_relations, n_channels, key):
        self.n_relations = n_relations
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels))
        # self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
        r = self.weights[edge_type, :]  # [n_edges, n_channels]
        o = x[edge_index[1, :]]  # [n_edges, n_channels]
        o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return -jnp.linalg.norm(s + r - o, axis=1, ord=1)

    def forward_heads(self, heads, edge_type, tail):
        r = self.weights[edge_type]
        return -jnp.linalg.norm(heads + (r - tail), axis=1, ord=1)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        return -jnp.linalg.norm((head + r) - tails, axis=1, ord=1)
