from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import Literal

import equinox as eqx
from jax import random as jrandom
import jax
from jax import numpy as jnp

from rgcn.layers.rgcn import RGCNConv


class Encoder(ABC):
    @abstractmethod
    def __call__(self, all_data, key):
        pass

    @abstractmethod
    def get_node_embeddings(self, all_data):
        pass

    @abstractmethod
    def normalize(self):
        pass


class DirectEncoder(eqx.Module, Encoder):
    normalization: bool
    initializations: jnp.ndarray

    def __init__(self, n_nodes, n_channels, key, n_embeddings=1, normalization=True):
        """
        @param n_embeddings: The number of embeddings to learn, i.e. 1 for DistMult and 2 for SimplE / ComplEx
        """
        keys = jrandom.split(key, n_embeddings)
        self.normalization = normalization
        self.initializations = jax.nn.initializers.normal()(keys[0], (n_nodes, n_channels))
        for i in range(1, n_embeddings):
            self.initializations = jnp.stack(
                (self.initializations,
                 jax.nn.initializers.normal()(keys[i], (n_nodes, n_channels))),
                axis=1
            )

    def __call__(self, all_data, key):
        return self.initializations

    def get_node_embeddings(self, all_data):
        return self.initializations

    def normalize(self):  # Do not JIT
        if self.normalization:
            object.__setattr__(self, 'initializations',
                               self.initializations / jnp.linalg.norm(self.initializations, axis=1, keepdims=True))


class RGCNEncoder(eqx.Module, Encoder):
    rgcns: any
    dropout_rate: Optional[float]
    pre_transform: Optional[jnp.ndarray]  # Initializations for x instead of a one-hot matrix

    def __init__(self, hidden_channels, edge_dropout_rate: float, node_dropout_rate: float,
                 normalizing_constant: Literal['per_relation_node', 'per_node', 'none'],
                 decomposition_method: Literal['basis', 'block', 'none'], n_nodes: int, n_relations: int, key):
        super().__init__()
        key1, key2 = jrandom.split(key)
        self.dropout_rate = edge_dropout_rate

        # Use 2 bases or 5 blocks
        n_decomp = 2 if decomposition_method == 'basis' else 100 if decomposition_method == 'block' else None
        self.rgcns = [
            RGCNConv(in_channels=in_channels, out_channels=out_channels, n_relations=2 * n_relations,
                     decomposition_method=decomposition_method, normalizing_constant=normalizing_constant,
                     dropout_rate=node_dropout_rate, n_decomp=n_decomp, key=key1)
            for in_channels, out_channels in zip([500] + hidden_channels[:-1], hidden_channels)
        ]

        if decomposition_method == 'block':
            self.pre_transform = jax.nn.initializers.normal()(key2, (n_nodes, 500))
        else:
            self.pre_transform = None

    def __call__(self, all_data, key):
        dropout_key, key = jrandom.split(key)
        dropout_all_data = all_data.dropout(self.dropout_rate, dropout_key)

        x = self.pre_transform  # May be None

        for layer in self.rgcns:
            layer_key, key = jrandom.split(key)
            x = jax.nn.relu(layer(x, dropout_all_data.edge_type_idcs, dropout_all_data.edge_masks, layer_key))
        return x

    def get_node_embeddings(self, all_data):
        x = None
        for layer in self.rgcns:
            x = jax.nn.relu(layer(x, all_data.edge_type_idcs, all_data.edge_masks, key=None))
        return x

    def normalize(self):
        pass

    # def single_relation(self, edge_index, rel):
    #   x = None
    #    for layer in self.rgcns:
    #        x = layer(x, edge_index, rel)
    #    x = self.decoder(x, edge_index, rel)
    #    return x
