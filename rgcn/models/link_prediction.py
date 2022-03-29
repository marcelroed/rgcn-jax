from jax_dataclasses import pytree_dataclass

import equinox as eqx
import jax.random as jrandom
import jax
import jax.numpy as jnp

from rgcn.layers.decoder import DistMult, ComplEx
from rgcn.layers.rgcn import RGCNConv
from rgcn.data.utils import make_dense_relation_edges


def compute_loss(x, y):
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    return loss.mean()


class DistMultModel(eqx.Module):
    decoder: DistMult
    initializations: jnp.ndarray

    def __init__(self, n_nodes, n_relations, n_channels, key):
        super().__init__()
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key1, (n_nodes, n_channels))
        self.decoder = DistMult(n_relations, n_channels, key2)

    def __call__(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)


class ComplExModel(eqx.Module):
    decoder: ComplEx
    initializations: jnp.ndarray

    def __init__(self, n_nodes, n_relations, n_channels, key):
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key, (n_nodes, n_channels), dtype=jnp.complex64)
        self.decoder = ComplEx(n_relations, n_channels, key2)

    def __call__(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type).real


@pytree_dataclass
class RGCNModelData:
    edge_index: jnp.ndarray
    edge_type: jnp.ndarray
    edge_type_idcs: jnp.ndarray
    edge_masks: jnp.ndarray

    @classmethod
    def from_data(cls, edge_index, edge_type, n_relations):
        edge_type_idcs, edge_masks = make_dense_relation_edges(edge_index, edge_type, n_relations)
        return cls(edge_index, edge_type, edge_type_idcs, edge_masks)


class RGCNModel(eqx.Module):
    rgcns: list[RGCNConv]
    decoder: DistMult

    def __init__(self, n_nodes, n_relations, hidden_channels, key):
        key1, key2 = jrandom.split(key)
        self.rgcns = [
            RGCNConv(in_channels=n_nodes, out_channels=hidden_channels, n_relations=n_relations, decomposition_method='basis', n_decomp=2, key=key1)
        ]
        self.decoder = DistMult(n_relations, hidden_channels, key2)

    def __call__(self, data: RGCNModelData):
        x = None
        for layer in self.rgcns:
            x = layer(x, data.edge_type_idcs, data.edge_masks)
        x = self.decoder(x, data.edge_index, data.edge_type)

        return x


def test_rgcn_model():
    n_nodes = 10
    n_relations = 5
    n_channels = 5
    key = jrandom.PRNGKey(0)

    data = RGCNModelData.from_data(
        jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
        jnp.array([0, 2, 3, 1, 4, 1, 1, 1, 1, 1]),
        n_relations
    )

    model = RGCNModel(n_nodes, n_relations, n_channels, key)
    x = model(data)
    print(x)