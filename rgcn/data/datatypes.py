from _warnings import warn

from jax import numpy as jnp, random as jrandom
from jax_dataclasses import pytree_dataclass


@pytree_dataclass
class BasicModelData:
    """Only stores edge_index and edge_type"""
    edge_index: jnp.ndarray
    edge_type: jnp.ndarray

    is_dense = False

    @classmethod
    def from_data(cls, edge_index, edge_type, **kwargs):
        if kwargs:
            warn(f'Not using additional parameters: {", ".join(kwargs.keys())} in {cls.__name__}')
        return cls(edge_index=edge_index, edge_type=edge_type)

    def __getitem__(self, idcs) -> 'BasicModelData':
        return self.__class__(edge_index=self.edge_index[:, idcs], edge_type=self.edge_type[idcs])

    def dropout(self, p: float, key) -> 'BasicModelData':
        if p is None:
            return self
        n_edges = self.edge_index.shape[1]
        dropout_shape = int(n_edges * p)
        idcs = jrandom.choice(key, jnp.arange(n_edges), (dropout_shape,), replace=False)

        return self[idcs]

    @property
    def fused(self) -> jnp.ndarray:
        """n_edges triples of shape [3, n_edges], each triple is (head, tail, edge_type)"""
        return jnp.concatenate((self.edge_index, self.edge_type.reshape(1, -1)), axis=0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# @pytree_dataclass
# class RGCNModelTrainingData:
#     """Has extensions for the RGCN model"""
#
#     edge_type_idcs: jnp.ndarray
#     "A dense tensor of shape (n_relations, 2, max_edges_per_relation) containing padded edge indices for each relation"
#
#     edge_masks: jnp.ndarray
#     "Masks to indicate where the edge indices are valid, of shape (n_relations, max_edges_per_relation)"
#
#     is_dense = True
#
#     @classmethod
#     def from_data(cls, edge_type_idcs, edge_masks):
#         # edge_type_idcs, edge_masks = make_dense_relation_edges(edge_index, edge_type, n_relations)
#         return cls(edge_type_idcs, edge_masks)
#
#     def dropout(self, p: float, key):
#         if p is None:
#             return self
#         return self.__class__(self.edge_type_idcs,
#                               self.edge_masks * jrandom.bernoulli(key, p, self.edge_masks.shape))