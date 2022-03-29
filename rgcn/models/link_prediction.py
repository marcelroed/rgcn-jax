import equinox as eqx
import jax.random as jrandom
import jax
import jax.numpy as jnp

from rgcn.layers.encoder import DistMult


def compute_loss(x, y):
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    return loss.mean()


class DistMultModel(eqx.Module):
    n_nodes: int
    n_relations: int
    n_channels: int
    decoder: DistMult
    initializations: jnp.ndarray

    def __init__(self, n_nodes, n_relations, n_channels, key):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_channels = n_channels
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key1, (n_nodes, n_channels))
        self.decoder = DistMult(n_relations, n_channels, key2)

    def __call__(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)
