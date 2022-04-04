from dataclasses import dataclass
from typing import Optional, Literal, Union

import jax
import optax
from jax_dataclasses import pytree_dataclass

import equinox as eqx
import jax.random as jrandom
import jax.numpy as jnp

from rgcn.layers.decoder import Decoder, TransE, SimplE, ComplEx
from rgcn.layers.encoder import DirectEncoder, RGCNEncoder, Encoder
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
    def loss(self, edge_index, edge_type, mask, all_data, key):
        pass

    @abstractmethod
    def l2_loss(self):
        pass


class GenericShallowModel(eqx.Module, BaseModel):
    @dataclass
    class Config(BaseConfig):
        n_channels: int
        n_embeddings: int
        normalization: bool

        def get_model(self, n_nodes, n_relations, key):
            return GenericShallowModel(self, n_nodes, n_relations, key)

    encoder: DirectEncoder
    decoder: Decoder
    l2_reg: Optional[float]

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2 = jrandom.split(key, 2)
        self.encoder = DirectEncoder(n_nodes, config.n_channels, key1, config.n_embeddings, config.normalization)
        self.decoder = config.decoder_class(n_relations, config.n_channels, key2)

    def __call__(self, edge_index, edge_type, all_data, key):
        embeddings = self.encoder(all_data, key)
        return self.decoder(embeddings, edge_index, edge_type)

    def normalize(self):  # Do not JIT
        self.encoder.normalize()

    def get_node_embeddings(self, all_data):
        return self.encoder.get_node_embeddings(all_data)

    def forward_heads(self, node_embeddings, edge_type, tail):
        return self.decoder.forward_heads(node_embeddings, edge_type, node_embeddings[tail])

    def forward_tails(self, node_embeddings, edge_type, head):
        return self.decoder.forward_tails(node_embeddings[head], edge_type, node_embeddings)

    def loss(self, edge_index, edge_type, mask, all_data, key):
        scores = self(edge_index, edge_type, None, key)
        return cross_entropy_loss(scores, mask)

    def l2_loss(self):
        if self.l2_reg is None:
            return jnp.array(0.)
        return self.l2_reg * self.decoder.l2_loss()


class TransEModel(GenericShallowModel):
    @dataclass
    class Config(GenericShallowModel.Config):
        margin: int

        def get_model(self, n_nodes, n_relations, key):
            return TransEModel(self, n_nodes, n_relations, key)

    margin: int

    def __init__(self, config: Config, n_nodes, n_relations, key):
        assert (config.decoder_class == TransE)
        super().__init__(config, n_nodes, n_relations, key)
        self.margin = config.margin

    def loss(self, edge_index, edge_type, mask, all_data, key):
        neg_start_index = edge_index.shape[1] // 2
        scores_pos = self(edge_index[:, :neg_start_index], edge_type[:neg_start_index], all_data, key)
        scores_neg = self(edge_index[:, neg_start_index:], edge_type[neg_start_index:], all_data, key)
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
    l2_reg: Optional[float]
    encoder: RGCNEncoder
    decoder: Decoder

    @dataclass
    class Config(BaseConfig):
        hidden_channels: list[int]
        edge_dropout_rate: Optional[float]  # None -> 1.0, meaning no dropout
        node_dropout_rate: Optional[float]  # None -> 1.0, meaning no dropout
        normalizing_constant: Literal['per_relation_node', 'per_node', 'none']
        decomposition_method: Literal['basis', 'block', 'none']

        def get_model(self, n_nodes, n_relations, key):
            return RGCNModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2 = jrandom.split(key)
        self.encoder = RGCNEncoder(config.hidden_channels, config.edge_dropout_rate, config.node_dropout_rate,
                                   config.normalizing_constant, config.decomposition_method, n_nodes, n_relations, key1)
        self.decoder = config.decoder_class(n_relations, config.hidden_channels[-1], key2)

    def __call__(self, edge_index, rel, all_data: RGCNModelTrainingData, key):
        embeddings = self.encoder(all_data, key)
        return self.decoder(embeddings, edge_index, rel)

    def get_node_embeddings(self, all_data):
        return self.encoder.get_node_embeddings(all_data)

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
            return self.l2_reg * self.decoder.l2_loss()
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


class CombinedModel(eqx.Module, BaseModel):
    l2_reg: Optional[float]
    encoder: RGCNEncoder
    decoder: Union[SimplE, ComplEx]

    @dataclass
    class Config(RGCNModel.Config):
        def get_model(self, n_nodes, n_relations, key):
            return CombinedModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2 = jrandom.split(key)
        self.encoder = RGCNEncoder(config.hidden_channels, config.edge_dropout_rate, config.node_dropout_rate,
                                   config.normalizing_constant, config.decomposition_method, n_nodes, n_relations, key1)
        last_layer_channels = config.hidden_channels[-1]
        assert (last_layer_channels % 2 == 0)
        assert (config.decoder_class in [SimplE, ComplEx])
        self.decoder = config.decoder_class(n_relations, last_layer_channels // 2, key2)

    def __call__(self, edge_index, rel, all_data: RGCNModelTrainingData, key):
        embeddings = self.encoder(all_data, key)  # [n_nodes, num_channels]
        num_channels = embeddings.shape[1]
        combined = jnp.stack(
            (embeddings[:, :num_channels // 2],
             embeddings[:, num_channels // 2:]),
            axis=1
        )
        return self.decoder(combined, edge_index, rel)

    def get_node_embeddings(self, all_data):
        embeddings = self.encoder.get_node_embeddings(all_data)
        num_channels = embeddings.shape[1]
        combined = jnp.stack(
            (embeddings[:, :num_channels // 2],
             embeddings[:, num_channels // 2:]),
            axis=1
        )
        return combined

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
            return self.l2_reg * self.decoder.l2_loss()
        else:
            return jnp.array(0.)


class DoubleRGCNModel(eqx.Module, BaseModel):
    l2_reg: Optional[float]
    encoder1: RGCNEncoder
    encoder2: RGCNEncoder
    decoder: Union[SimplE, ComplEx]

    @dataclass
    class Config(RGCNModel.Config):
        def get_model(self, n_nodes, n_relations, key):
            return DoubleRGCNModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2, key3 = jrandom.split(key, 3)
        self.encoder1 = RGCNEncoder(config.hidden_channels, config.edge_dropout_rate, config.node_dropout_rate,
                                    config.normalizing_constant, config.decomposition_method, n_nodes, n_relations,
                                    key1)
        self.encoder2 = RGCNEncoder(config.hidden_channels, config.edge_dropout_rate, config.node_dropout_rate,
                                    config.normalizing_constant, config.decomposition_method, n_nodes, n_relations,
                                    key2)
        assert (config.decoder_class in [SimplE, ComplEx])
        self.decoder = config.decoder_class(n_relations, config.hidden_channels[-1], key3)

    def __call__(self, edge_index, rel, all_data: RGCNModelTrainingData, key):
        key1, key2 = jrandom.split(key, 2)
        embeddings1 = self.encoder1(all_data, key1)
        embeddings2 = self.encoder2(all_data, key2)
        combined = jnp.stack([embeddings1, embeddings2], axis=1)
        return self.decoder(combined, edge_index, rel)

    def get_node_embeddings(self, all_data):
        embeddings1 = self.encoder1.get_node_embeddings(all_data)
        embeddings2 = self.encoder2.get_node_embeddings(all_data)
        combined = jnp.stack([embeddings1, embeddings2], axis=1)
        return combined

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
            return self.l2_reg * self.decoder.l2_loss()
        else:
            return jnp.array(0.)


class LearnedEnsembleModel(eqx.Module, BaseModel):
    l2_reg: Optional[float]
    encoder1: Encoder
    encoder2: Encoder
    decoder1: Decoder
    decoder2: Decoder
    alpha: jnp.array

    @dataclass
    class Config(RGCNModel.Config, GenericShallowModel.Config):
        def get_model(self, n_nodes, n_relations, key):
            return LearnedEnsembleModel(self, n_nodes, n_relations, key)

    def __init__(self, config: Config, n_nodes, n_relations, key):
        super().__init__()
        self.l2_reg = config.l2_reg
        key1, key2, key3, key4 = jrandom.split(key, 4)
        self.encoder1 = RGCNEncoder(config.hidden_channels, config.edge_dropout_rate, config.node_dropout_rate,
                                    config.normalizing_constant, config.decomposition_method, n_nodes, n_relations,
                                    key1)
        self.encoder2 = DirectEncoder(n_nodes, config.n_channels, key2, config.n_embeddings, config.normalization)
        self.decoder1 = config.decoder_class(n_relations, config.hidden_channels[-1] // 2, key3)
        self.decoder2 = config.decoder_class(n_relations, config.n_channels, key4)
        self.alpha = jnp.array(0.5)

    def __call__(self, edge_index, rel, all_data: RGCNModelTrainingData, key):
        key1, key2 = jrandom.split(key)
        embeddings1 = self.encoder1(all_data, key1)
        num_channels = embeddings1.shape[1]
        combined = jnp.stack(
            (embeddings1[:, :num_channels // 2],
             embeddings1[:, num_channels // 2:]),
            axis=1
        )
        embeddings2 = self.encoder2(all_data, key2)
        scores1 = self.decoder1(combined, edge_index, rel)
        scores2 = self.decoder2(embeddings2, edge_index, rel)
        return self.alpha * scores1 + (1 - self.alpha) * scores2

    def get_node_embeddings(self, all_data):
        embeddings1 = self.encoder1.get_node_embeddings(all_data)
        num_channels = embeddings1.shape[1]
        combined = jnp.stack(
            (embeddings1[:, :num_channels // 2],
             embeddings1[:, num_channels // 2:]),
            axis=1
        )
        embeddings2 = self.encoder2.get_node_embeddings(all_data)
        return jnp.stack([combined, embeddings2], axis=0)

    def normalize(self):
        self.encoder2.normalize()

    def forward_heads(self, node_embeddings, relation_type, tail):
        scores1 = self.decoder1.forward_heads(node_embeddings[0, ...], relation_type, node_embeddings[0, tail, ...])
        scores2 = self.decoder2.forward_heads(node_embeddings[1, ...], relation_type, node_embeddings[1, tail, ...])
        return self.alpha * scores1 + (1 - self.alpha) * scores2

    def forward_tails(self, node_embeddings, relation_type, head):
        scores1 = self.decoder1.forward_tails(node_embeddings[0, head, ...], relation_type, node_embeddings[0, ...])
        scores2 = self.decoder2.forward_tails(node_embeddings[1, head, ...], relation_type, node_embeddings[1, ...])
        return self.alpha * scores1 + (1 - self.alpha) * scores2

    def loss(self, edge_index, edge_type, mask, all_data, key):
        scores = self(edge_index, edge_type, all_data, key=key)
        return cross_entropy_loss(scores, mask)

    def l2_loss(self):
        if self.l2_reg:
            # Don't regularize the weights in the RGCN
            return self.l2_reg * (self.decoder1.l2_loss() + self.decoder2.l2_loss())
        else:
            return jnp.array(0.)
