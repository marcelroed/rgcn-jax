import equinox as eqx
import jax.random as jrandom
import jax
import jax.numpy as jnp

from rgcn.layers.decoder import TransE, Decoder


def cross_entropy_loss(x, y):
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    return loss.mean()


def margin_ranking_loss(dist_pos, dist_neg, gamma):
    final = jnp.clip(gamma + dist_pos - dist_neg, 0, None)
    return jnp.sum(final)


class GenericModel(eqx.Module):
    n_nodes: int
    n_relations: int
    n_channels: int
    regularization: float
    decoder: Decoder
    initializations: jnp.ndarray

    def __init__(self, decoder, n_nodes, n_relations, n_channels, key, regularization=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_channels = n_channels
        self.regularization = regularization
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key1, (n_nodes, n_channels))
        self.decoder = decoder(n_relations, n_channels, key2)

    def __call__(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)

    def normalize(self):  # Do not JIT
        object.__setattr__(self, 'initializations',
                           self.initializations / jnp.linalg.norm(self.initializations, axis=1, keepdims=True))

    def forward_heads(self, edge_type, tail):
        return self.decoder.forward_heads(self.initializations, edge_type, self.initializations[tail])

    def forward_tails(self, edge_type, head):
        return self.decoder.forward_tails(self.initializations[head], edge_type, self.initializations)

    def loss(self, edge_index, edge_type, mask):
        scores = self(edge_index, edge_type)
        return cross_entropy_loss(scores, mask)

    def l2_loss(self):
        if self.regularization is None:
            return jnp.array(0.)
        return self.regularization * (jnp.square(self.decoder.weights).sum())


class TransEModel(GenericModel):
    margin: int

    def __init__(self, n_nodes, n_relations, n_channels, margin, key):
        super().__init__(TransE, n_nodes, n_relations, n_channels, key)
        self.margin = margin

    def loss(self, edge_index, edge_type, mask):
        neg_start_index = edge_index.shape[1] // 2
        dist_pos = -self(edge_index[:, :neg_start_index], edge_type[:neg_start_index])
        dist_neg = -self(edge_index[:, neg_start_index:], edge_type[neg_start_index:])
        return margin_ranking_loss(dist_pos, dist_neg, self.margin)
