from dataclasses import dataclass
from typing import Optional

from jax_dataclasses import pytree_dataclass

import equinox as eqx
import jax.random as jrandom
import jax
import jax.numpy as jnp

from rgcn.layers.decoder import DistMult, ComplEx
from rgcn.layers.rgcn import RGCNConv
from rgcn.data.utils import make_dense_relation_edges, BaseConfig
from warnings import warn


def compute_loss(x, y):
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    return loss.mean()


@pytree_dataclass
class BasicModelData:
    """Only stores edge_index and edge_type"""
    edge_index: jnp.ndarray
    edge_type: jnp.ndarray

    is_dense = False

    @classmethod
    def from_data(cls, edge_index, edge_type, **kwargs):
        warn(f'Not using additional parameters: {", ".join(kwargs.keys())} in {cls.__name__}')
        return cls(edge_index=edge_index, edge_type=edge_type)


class DistMultModel(eqx.Module):
    decoder: DistMult
    initializations: jnp.ndarray
    data_class = BasicModelData

    @dataclass
    class Config(BaseConfig):
        n_channels: int
        name: Optional[str] = None

        def get_model(self, n_nodes, n_relations, key):
            return RGCNModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key1, (n_nodes, config.n_channels))
        self.decoder = DistMult(n_relations, config.n_channels, key2)

    def __call__(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)

    def single_relation(self, edge_index, rel):
        return self.decoder(self.initializations, edge_index, rel)




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
class RGCNModelTrainingData:
    """Has extensions for the RGCN model"""
    edge_type_idcs: jnp.ndarray
    edge_masks: jnp.ndarray

    is_dense = True

    @classmethod
    def from_data(cls, edge_type_idcs, edge_masks):
        # edge_type_idcs, edge_masks = make_dense_relation_edges(edge_index, edge_type, n_relations)
        return cls(edge_type_idcs, edge_masks)


class RGCNModel(eqx.Module):
    rgcns: list[RGCNConv]
    decoder: DistMult
    data_class = RGCNModelTrainingData

    @dataclass
    class Config(BaseConfig):
        hidden_channels: list[int]
        name: Optional[str] = None

        def get_model(self, n_nodes, n_relations, key):
            return RGCNModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        key1, key2 = jrandom.split(key)
        self.rgcns = [
            RGCNConv(in_channels=in_channels, out_channels=out_channels, n_relations=n_relations, decomposition_method='basis', n_decomp=2, key=key1)
            for in_channels, out_channels in zip([n_nodes] + config.hidden_channels[:-1], config.hidden_channels)
        ]
        self.decoder = DistMult(n_relations, config.hidden_channels[-1], key2)

    def __call__(self, data: RGCNModelTrainingData):
        x = None
        for layer in self.rgcns:
            x = layer(x, data.edge_type_idcs, data.edge_masks)
        x = self.decoder.call_dense(x, data.edge_type_idcs, data.edge_masks)

        return x

    def single_relation(self, edge_index, rel):
        x = None
        edge_mask = jnp.ones((edge_index.shape[0], 1), dtype=jnp.bool_)
        for layer in self.rgcns:
            x = layer.single_relation(x, edge_index, edge_mask, rel)
        x = self.decoder(x, edge_index, edge_type=rel)
        return x


def test_rgcn_model():
    n_nodes = 10
    n_relations = 5
    n_channels = 5
    key = jrandom.PRNGKey(0)

    data = RGCNModelTrainingData.from_data(
        jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
        jnp.array([0, 2, 3, 1, 4, 1, 1, 1, 1, 1]),
        n_relations
    )

    model = RGCNModel(n_nodes, n_relations, n_channels, key)
    x = model(data)
    print(x)