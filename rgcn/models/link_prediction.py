from dataclasses import dataclass
from typing import Optional

import optax
from jax_dataclasses import pytree_dataclass

import equinox as eqx
import jax.random as jrandom
import jax
import jax.numpy as jnp

from rgcn.layers.decoder import Decoder, ComplEx, SimplE, TransE, DistMult
from rgcn.layers.rgcn import RGCNConv
from rgcn.data.utils import BaseConfig
from warnings import warn
from abc import ABC, abstractmethod


def safe_log(x, eps=1e-15):
    return jnp.log(jnp.clip(x, eps, None))


def cross_entropy_loss(x, y):
    # max_val = jnp.clip(x, 0, None)
    # loss = x - x * y + max_val + safe_log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    # return loss.mean()
    return optax.sigmoid_binary_cross_entropy(x, y).mean()


def margin_ranking_loss(scores_pos, scores_neg, gamma):
    final = jnp.clip(gamma - scores_pos + scores_neg, 0, None)
    return jnp.sum(final)


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


class BaseModel(ABC):
    @abstractmethod
    def __call__(self, edge_index, edge_type, all_data, key):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def get_node_embeddings(self, all_data):
        pass

    @abstractmethod
    def forward_heads(self, node_embeddings, relation_type, tail):
        pass

    @abstractmethod
    def forward_tails(self, node_embeddings, relation_type, head):
        pass

    @abstractmethod
    def loss(self, edge_index, edge_type, mask, all_data):
        pass

    @abstractmethod
    def l2_loss(self):
        pass


class GenericShallowModel(eqx.Module, BaseModel):
    @dataclass
    class Config(BaseConfig):
        n_channels: int
        l2_reg: Optional[float] = None
        name: Optional[str] = None

    decoder: Decoder
    initializations: jnp.ndarray
    l2_reg: Optional[float]

    def __init__(self, decoder, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2 = jrandom.split(key, 2)
        self.initializations = jax.nn.initializers.normal()(key1, (n_nodes, config.n_channels))
        self.decoder = decoder(n_relations, config.n_channels, key2)

    def __call__(self, edge_index, edge_type, all_data):
        return self.decoder(self.initializations, edge_index, edge_type)

    def normalize(self):  # Do not JIT
        object.__setattr__(self, 'initializations',
                           self.initializations / jnp.linalg.norm(self.initializations, axis=1, keepdims=True))

    def get_node_embeddings(self, all_data):
        return self.initializations

    def forward_heads(self, node_embeddings, edge_type, tail):
        return self.decoder.forward_heads(node_embeddings, edge_type, node_embeddings[tail])

    def forward_tails(self, node_embeddings, edge_type, head):
        return self.decoder.forward_tails(node_embeddings[head], edge_type, node_embeddings)

    def loss(self, edge_index, edge_type, mask, all_data):
        scores = self(edge_index, edge_type, None)
        return cross_entropy_loss(scores, mask)

    def l2_loss(self):
        if self.l2_reg is None:
            return jnp.array(0.)
        return self.l2_reg * (jnp.square(self.decoder.weights).sum())


class ComplExModel(GenericShallowModel):
    def __init__(self, config: GenericShallowModel.Config, n_nodes, n_relations, key):
        key1, key2, key3 = jrandom.split(key, 3)
        super().__init__(decoder=ComplEx, config=config,
                         n_nodes=n_nodes, n_relations=n_relations, key=key3)

        # [n_nodes, 2, n_channels] -- first real, then imaginary
        self.initializations = jnp.stack(
            (jax.nn.initializers.normal()(key1, (n_nodes, config.n_channels)),
             jax.nn.initializers.normal()(key2, (n_nodes, config.n_channels))), axis=1
        )

    def normalize(self):
        pass

    def l2_loss(self):
        if self.l2_reg is None:
            return jnp.array(0.)

        return self.l2_reg * (self.decoder.l2_loss() + jnp.square(self.initializations).sum())


class SimplEModel(GenericShallowModel):

    def __init__(self, config: GenericShallowModel.Config, n_nodes, n_relations, key):
        key1, key2, key3 = jrandom.split(key, 3)
        super().__init__(SimplE, config, n_nodes, n_relations, key3)

        # [n_nodes, 2, n_channels] -- first real, then imaginary
        self.initializations = jnp.stack([jax.nn.initializers.normal()(key1, (n_nodes, config.n_channels)),
                                          jax.nn.initializers.normal()(key2, (n_nodes, config.n_channels))], 1)

    def normalize(self):
        pass

    def l2_loss(self):
        if self.l2_reg is None:
            return jnp.array(0.)
        return self.l2_reg * (
                jnp.square(self.decoder.weights).sum() +
                jnp.square(self.decoder.weights_inv).sum() +
                jnp.square(self.initializations).sum()
        )


class TransEModel(GenericShallowModel):
    @dataclass
    class Config(GenericShallowModel.Config):
        margin: int = 2

    margin: int

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__(TransE, config, n_nodes, n_relations, key)
        self.margin = config.margin

    def normalize(self):
        pass

    def loss(self, edge_index, edge_type, mask, all_data):
        neg_start_index = edge_index.shape[1] // 2
        scores_pos = self(edge_index[:, :neg_start_index], edge_type[:neg_start_index], all_data)
        scores_neg = self(edge_index[:, neg_start_index:], edge_type[neg_start_index:], all_data)
        return margin_ranking_loss(scores_pos, scores_neg, self.margin)


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

    def dropout(self, p: float, key):
        if p is None:
            return self
        return self.__class__(self.edge_type_idcs,
                              self.edge_masks * jrandom.bernoulli(key, p, self.edge_masks.shape))


class RGCNModel(eqx.Module, BaseModel):
    rgcns: list[RGCNConv]
    decoder: DistMult
    dropout_rate: Optional[float]
    l2_reg: Optional[float]

    @dataclass
    class Config(BaseConfig):
        hidden_channels: list[int]
        dropout_rate: Optional[float] = None  # None -> 1.0, meaning no dropout
        normalizing_constant: str = 'per_node'
        l2_reg: Optional[float] = None
        name: Optional[str] = None
        epochs: int = 10
        learning_rate: float = 0.5
        seed: int = 42

        def get_model(self, n_nodes, n_relations, key):
            return RGCNModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        key1, key2 = jrandom.split(key)
        self.dropout_rate = config.dropout_rate
        self.l2_reg = config.l2_reg
        self.rgcns = [
            RGCNConv(in_channels=in_channels, out_channels=out_channels, n_relations=n_relations,
                     decomposition_method='basis', normalizing_constant=config.normalizing_constant, n_decomp=2, key=key1)
            for in_channels, out_channels in zip([n_nodes] + config.hidden_channels[:-1], config.hidden_channels)
        ]
        self.decoder = DistMult(n_relations, config.hidden_channels[-1], key2)

    def __call__(self, edge_index, rel, all_data: RGCNModelTrainingData, key):
        dropout_all_data = all_data.dropout(self.dropout_rate, key)
        x = None
        for layer in self.rgcns:
            x = jax.nn.relu(layer(x, dropout_all_data.edge_type_idcs, dropout_all_data.edge_masks))
        x = self.decoder(x, edge_index, rel)
        return x

    def get_node_embeddings(self, all_data):
        x = None
        for layer in self.rgcns:
            x = jax.nn.relu(layer(x, all_data.edge_type_idcs, all_data.edge_masks))
        return x

    def normalize(self):
        pass

    def forward_heads(self, node_embeddings, relation_type, tail):
        return self.decoder.forward_heads(node_embeddings, relation_type, node_embeddings[tail])

    def forward_tails(self, node_embeddings, relation_type, head):
        return self.decoder.forward_tails(node_embeddings[head], relation_type, node_embeddings)

    def loss(self, edge_index, edge_type, mask, all_data, key):
        scores = self(edge_index, edge_type, all_data, key=key)
        return cross_entropy_loss(scores, mask)

    def l2_loss(self):
        if self.l2_reg:
            # Don't regularize the weights in the RGCN
            return self.l2_reg * jnp.sum(self.decoder.l2_loss())
        else:
            return jnp.array(0.)

    # def single_relation(self, edge_index, rel):
    #   x = None
    #    for layer in self.rgcns:
    #        x = layer(x, edge_index, rel)
    #    x = self.decoder(x, edge_index, rel)
    #    return x


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
