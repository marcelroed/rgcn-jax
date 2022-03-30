from attrs import define
import jax.numpy as jnp
from torch_geometric.datasets import WordNet18
from rgcn.data.utils import torch_to_jax
from jax_dataclasses import pytree_dataclass


@pytree_dataclass
class LinkPredictionData:
    edge_index: jnp.ndarray
    edge_type: jnp.ndarray


@define
class LinkPredictionWrapper:
    name: str
    num_nodes: int
    num_edges: int
    num_relations: int
    edge_index: jnp.ndarray
    edge_type: jnp.ndarray
    # edge_index_by_type: jnp.ndarray
    # edge_masks_by_type: jnp.ndarray
    train_idx: jnp.ndarray
    val_idx: jnp.ndarray
    test_idx: jnp.ndarray

    @classmethod
    def load_wordnet18(cls, root='data/'):
        dataset = WordNet18(f'{root}wordnet18')[0]
        edge_index = torch_to_jax(dataset.edge_index)
        edge_type = torch_to_jax(dataset.edge_type)
        num_relations = jnp.max(edge_type).item() + 1
        num_nodes = dataset.num_nodes
        num_edges = edge_type.shape[0]
        train_idx = torch_to_jax(dataset.train_mask)
        val_idx = torch_to_jax(dataset.val_mask)
        test_idx = torch_to_jax(dataset.test_mask)
        return cls(
            name='WN18',
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_edges=num_edges,
            edge_index=edge_index,
            edge_type=edge_type,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )