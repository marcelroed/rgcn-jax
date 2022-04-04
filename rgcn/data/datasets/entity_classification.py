import gc
from typing import Optional

import numpy as np
from torch_geometric.datasets.entities import Entities
from attrs import define
import jax.numpy as jnp
from rgcn.data.utils import torch_to_jax


@define
class EntityClassificationWrapper:
    name: str
    num_nodes: int
    num_edges: int
    num_relations: int
    num_classes: int
    node_features: Optional[jnp.ndarray]
    # edge_index: jnp.ndarray
    edge_type: jnp.ndarray
    edge_index_by_type: jnp.ndarray
    edge_masks_by_type: jnp.ndarray
    train_idx: jnp.ndarray
    train_y: jnp.ndarray
    val_idx: jnp.ndarray
    val_y: jnp.ndarray
    test_idx: jnp.ndarray
    test_y: jnp.ndarray

    @classmethod
    def load_dataset(cls, dataset_name, root='data/'):
        dataset = Entities(root=root + dataset_name.lower(), name=dataset_name)
        edge_index = dataset.data.edge_index.numpy()
        edge_type = torch_to_jax(dataset.data.edge_type)
        test_idx_original = torch_to_jax(dataset.data.test_idx)
        test_y_original = torch_to_jax(dataset.data.test_y)

        val_idx, test_idx = jnp.split(test_idx_original, [len(test_idx_original) // 5])
        val_y, test_y = jnp.split(test_y_original, [len(test_y_original) // 5])

        np_padded, np_masks = make_dense_relation_tensor(dataset.num_relations, edge_index, edge_type)

        instance = cls(
            name=dataset_name,
            num_nodes=dataset.data.num_nodes,
            num_edges=dataset.data.num_edges,
            num_relations=dataset.num_relations,
            num_classes=dataset.num_classes,
            node_features=getattr(dataset.data, 'node_features', None),
            # edge_index=edge_index,
            edge_index_by_type=jnp.array(np_padded),
            edge_masks_by_type=jnp.array(np_masks),
            edge_type=edge_type,
            train_idx=torch_to_jax(dataset.data.train_idx),
            train_y=torch_to_jax(dataset.data.train_y),
            val_idx=val_idx,
            val_y=val_y,
            test_idx=test_idx,
            test_y=test_y,
        )
        del np_padded; del np_masks
        gc.collect()

        return instance


def make_dense_relation_tensor(num_relations, edge_index, edge_type):
    max_edge_index_length = max([jnp.sum(edge_type == i) for i in range(num_relations)])

    # padded: [n_relations, 2, max_edge_index_length]
    # 1300 * 2 * 20_000
    padded = [np.pad(edge_index[:, i == edge_type], ((0, 0), (0, max_edge_index_length - (i == edge_type).sum())))
              for i in range(num_relations)]
    mask_lengths = [np.sum(edge_type == i) for i in range(num_relations)]

    # masks: [n_relations, max_edge_index_length]
    masks = [np.concatenate((np.ones(ml, dtype=bool), np.zeros(max_edge_index_length - ml, dtype=bool)), axis=0) for
             ml in mask_lengths]
    np_mask = np.stack(masks, axis=0)
    np_padded = np.stack(padded, axis=0)
    del padded; del masks; gc.collect()
    return np_padded, np_mask


def test_dataset_structure():
    dataset = Entities(name='AIFB', root='data/aifb')

    print(dataset)


def test_dataset_object():
    dataset = EntityClassificationWrapper.load_dataset('BGS')

    print(dataset)