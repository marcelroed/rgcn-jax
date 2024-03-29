import warnings
from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np


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

    @abstractmethod
    def l2_loss(self):
        pass


class DistMult(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray
    normalize: bool

    def __init__(self, n_relations, n_channels, normalize, key):
        self.n_relations = n_relations
        self.weights = jax.nn.initializers.xavier_normal()(key, (n_relations, n_channels))
        self.normalize = normalize
        # self.weights = jax.nn.initializers.glorot_uniform()(key, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        # O(n_edges * n_channels) to compute
        # x: [n_nodes, n_channels]
        # edge_type_idcs: [n_relations, 2, n_edges_per_relation_max]
        # edge_type_idcs_mask: [n_edges_per_relation_max]

        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        r = self.weights[edge_type, :]  # [n_edges, n_channels]
        o = x[edge_index[1, :]]  # [n_edges, n_channels]

        if self.normalize:
            s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
            o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return jnp.sum(s * r * o, axis=1)
        # return jnp.einsum('ec,ec,ec->e', s, r, o)

    # def call_dense(self, x, rel_edge_index, rel_edge_mask):
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

    def forward_heads(self, heads, edge_type, tail):
        # heads: [num_nodes, n_channels]
        # tail: [n_channels]
        r = self.weights[edge_type]  # [n_channels,]

        # return jnp.sum(heads * (r * tail), axis=1)

        return jnp.einsum('ec,c,c->e', heads, r, tail)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        # return jnp.sum((head * r) * tails, axis=1)
        return jnp.einsum('c,cd,ed->e', head, jnp.diag(r), tails)
        # For some reason, this is faster than the einsum 'c,c,ec->e', which would make much more sense.

    def l2_loss(self):
        return jnp.sum(jnp.square(self.weights))


def test_distmult():
    distmult = DistMult(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[1, 2], [3, 4]])
    edge_index = jnp.array([[0, 1], [0, 1]])
    edge_type = jnp.array([0, 1])
    distmult(x, edge_index, edge_type)


class ComplEx(eqx.Module, Decoder):
    n_relations: int
    weights_r: jnp.ndarray
    weights_i: jnp.ndarray

    def __init__(self, n_relations, n_channels, key, **kwargs):
        self.n_relations = n_relations
        key1, key2 = random.split(key, 2)
        self.weights_r = jax.nn.initializers.glorot_normal()(key1, (n_relations, n_channels))
        self.weights_i = jax.nn.initializers.glorot_normal()(key2, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        # x: [n_nodes, 2, n_channels] -- first real, then imaginary
        s_r, s_i = x[edge_index[0, :], 0, :], x[edge_index[0, :], 1, :]  # [n_edges, n_channels]
        r_r, r_i = self.weights_r[edge_type, :], self.weights_i[edge_type, :]  # [n_edges, n_channels]
        o_r, o_i = x[edge_index[1, :], 0, :], x[edge_index[1, :], 1, :]  # [n_edges, n_channels]

        return (s_r * r_r * o_r +
                s_r * r_i * o_i +
                s_i * r_r * o_i -
                s_i * r_i * o_r).sum(axis=1)

    def forward_heads(self, heads, edge_type, tail):
        r_r, r_i = self.weights_r[edge_type, :], self.weights_i[edge_type, :]
        o_r, o_i = tail[0, :], tail[1, :]
        s_r, s_i = heads[:, 0, :], heads[:, 1, :]
        return (s_r * (r_r * o_r) +
                s_r * (r_i * o_i) +
                s_i * (r_r * o_i) -
                s_i * (r_i * o_r)).sum(axis=1)

    def forward_tails(self, head, edge_type, tails):
        r_r, r_i = self.weights_r[edge_type, :], self.weights_i[edge_type, :]
        s_r, s_i = head[0, :], head[1, :]
        o_r, o_i = tails[:, 0, :], tails[:, 1, :]
        return ((s_r * r_r) * o_r +
                (s_r * r_i) * o_i +
                (s_i * r_r) * o_i -
                (s_i * r_i) * o_r).sum(axis=1)

    def l2_loss(self):
        return jnp.sum(self.weights_r ** 2) + jnp.sum(self.weights_i ** 2)


def test_complex():
    compl = ComplEx(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[[1, 2],
                    [1, -1]],
                   [[3, 5],
                    [2, 3]]])
    x_ = np.array([[1 + 1j, 2 - 1j],
                   [3 + 2j, 5 + 3j]], dtype=complex)
    edge_index = jnp.array([[0, 1],
                            [0, 1]])
    edge_type = jnp.array([0, 1])
    result = compl(x, edge_index, edge_type)
    print(result)
    complex_weights = np.array(compl.weights_i) * 1j + np.array(compl.weights_r)
    s, r, o = x_[edge_index[0, :], :], complex_weights[edge_type, :], x_[edge_index[1, :], :]
    result = np.sum(s * r * np.conj(o), axis=1).real
    print(result)


def test_complex():
    compl = ComplEx(n_relations=2, n_channels=2, key=random.PRNGKey(0))
    x = jnp.array([[[1, 2],
                    [1, -1]],
                   [[3, 5],
                    [2, 3]]])
    x_ = np.array([[1 + 1j, 2 - 1j],
                   [3 + 2j, 5 + 3j]], dtype=complex)
    edge_index = jnp.array([[0, 1],
                            [0, 1]])
    edge_type = jnp.array([0, 1])
    result_jax = compl(x_, edge_index, edge_type)
    complex_weights = np.array(compl.weights.imag) * 1j + np.array(compl.weights.real)

    s, r, o = x_[edge_index[0, :], :], complex_weights[edge_type, :], x_[edge_index[1, :], :]
    result_np = np.sum(s * r * np.conj(o), axis=1).real
    print(result_jax)
    print(result_np)
    assert jnp.allclose(result_jax, result_np)


class SimplE(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray
    weights_inv: jnp.ndarray

    def __init__(self, n_relations, n_channels, key, **kwargs):
        self.n_relations = n_relations
        key1, key2 = random.split(key, 2)
        self.weights = jax.nn.initializers.glorot_normal()(key1, (n_relations, n_channels))
        self.weights_inv = jax.nn.initializers.glorot_normal()(key2, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        # x: [n_nodes, 2, n_channels] -- first head vectors, then tail vectors
        s_h, s_t = x[edge_index[0, :], 0, :], x[edge_index[0, :], 1, :]  # [n_edges, n_channels]
        r, r_inv = self.weights[edge_type, :], self.weights_inv[edge_type, :]  # [n_edges, n_channels]
        o_h, o_t = x[edge_index[1, :], 0, :], x[edge_index[1, :], 1, :]  # [n_edges, n_channels]

        return (s_h * r * o_t + o_h * r_inv * s_t).sum(axis=1) / 2

    def forward_heads(self, heads, edge_type, tail):
        r, r_inv = self.weights[edge_type, :], self.weights_inv[edge_type, :]
        o_h, o_t = tail[0, :], tail[1, :]
        s_h, s_t = heads[:, 0, :], heads[:, 1, :]
        return (s_h * (r * o_t) + (o_h * r_inv) * s_t).sum(axis=1) / 2

    def forward_tails(self, head, edge_type, tails):
        r, r_inv = self.weights[edge_type, :], self.weights_inv[edge_type, :]
        s_h, s_t = head[0, :], head[1, :]
        o_h, o_t = tails[:, 0, :], tails[:, 1, :]
        return ((s_h * r) * o_t + o_h * (r_inv * s_t)).sum(axis=1) / 2

    def l2_loss(self):
        return jnp.sum(self.weights ** 2) + jnp.sum(self.weights_inv ** 2)


class RESCAL(eqx.Module, Decoder):
    n_relations: int
    n_channels: int
    weights: jnp.ndarray
    normalize: bool

    def __init__(self, n_relations, n_channels, normalize, key):
        warnings.warn("RESCAL uses A LOT of memory. Consider using ComplEx instead.", category=ResourceWarning)
        self.n_relations = n_relations
        self.n_channels = n_channels
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels, n_channels))
        self.normalize = normalize

    def __call__(self, x, edge_index, edge_type):
        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        o = x[edge_index[1, :]]  # [n_edges, n_channels]
        r = self.weights[edge_type]  # [n_edges, n_channels, n_channels]

        if self.normalize:
            s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
            o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return (s.reshape(-1, self.n_channels, 1) *
                jnp.matmul(r.reshape(-1, self.n_channels, self.n_channels),
                           o.reshape(-1, self.n_channels, 1))).sum(axis=1).reshape(-1)

    def forward_heads(self, heads, edge_type, tail):
        r = self.weights[edge_type]
        return jnp.einsum('ec,cd,d->e', heads, r, tail)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        return jnp.einsum('c,cd,ed->e', head, r, tails)

    def l2_loss(self):
        return jnp.sum(self.weights ** 2)


class TransE(eqx.Module, Decoder):
    n_relations: int
    weights: jnp.ndarray
    normalize: bool

    def __init__(self, n_relations, n_channels, normalize, key):
        self.n_relations = n_relations
        self.normalize = normalize
        self.weights = jax.nn.initializers.normal()(key, (n_relations, n_channels))

    def __call__(self, x, edge_index, edge_type):
        s = x[edge_index[0, :]]  # [n_edges, n_channels]
        r = self.weights[edge_type, :]  # [n_edges, n_channels]
        o = x[edge_index[1, :]]  # [n_edges, n_channels]

        if self.normalize:
            s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
            o = o / jnp.linalg.norm(o, axis=1, keepdims=True)

        return -jnp.linalg.norm(s + r - o, axis=1, ord=1)

    def forward_heads(self, heads, edge_type, tail):
        r = self.weights[edge_type]
        return -jnp.linalg.norm(heads + (r - tail), axis=1, ord=1)

    def forward_tails(self, head, edge_type, tails):
        r = self.weights[edge_type]
        return -jnp.linalg.norm((head + r) - tails, axis=1, ord=1)

    def l2_loss(self):
        return jnp.sum(self.weights ** 2)


if __name__ == '__main__':
    test_complex()
