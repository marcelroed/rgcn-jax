import os
from typing import Optional, Callable, List

import jax.numpy as jnp
import torch
from attrs import define
from jax_dataclasses import pytree_dataclass
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.datasets import WordNet18, WordNet18RR

from rgcn.data.utils import torch_to_jax


class CustomDataset(InMemoryDataset):
    urls = {
        'FB15k-237': 'https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova',
        'FB15k': 'https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB15k'
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self):
        with open(os.path.join(self.raw_dir, 'entities.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(os.path.join(self.raw_dir, 'relations.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

        size = {}
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(self.raw_dir, f'{split}.txt'), 'r') as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                edge_index = torch.cat((edge_index, torch.tensor([src, dst])), dim=-1)
                edge_type = torch.cat((edge_type, torch.tensor(rel)), dim=-1)
                size[split] = len(lines)

        data = Data(num_nodes=len(entities_dict),
                    edge_index=edge_index,
                    edge_type=edge_type,
                    train_mask=torch.cat((torch.ones(size['train']), torch.zeros(size['valid'] + size['test']))).bool(),
                    val_mask=torch.cat(
                        (torch.zeros(size['train']), torch.ones(size['valid']), torch.zeros(size['test']))).bool(),
                    test_mask=torch.cat((torch.zeros(size['train'] + size['valid']), torch.ones(size['test']))).bool())

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((self.collate([data])), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


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
    train_idx: jnp.ndarray
    val_idx: jnp.ndarray
    test_idx: jnp.ndarray

    @classmethod
    def load_dataset(cls, dataset, name):
        edge_index = torch_to_jax(dataset.edge_index)
        edge_type = torch_to_jax(dataset.edge_type)
        num_relations = jnp.max(edge_type).item() + 1
        num_nodes = dataset.num_nodes
        num_edges = edge_type.shape[0]
        train_idx = torch_to_jax(torch.arange(dataset.train_mask.shape[0])[dataset.train_mask])
        val_idx = torch_to_jax(torch.arange(dataset.train_mask.shape[0])[dataset.val_mask])
        test_idx = torch_to_jax(torch.arange(dataset.test_mask.shape[0])[dataset.test_mask])
        return cls(
            name=name,
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_edges=num_edges,
            edge_index=edge_index,
            edge_type=edge_type,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )

    @classmethod
    def load_wordnet18(cls, root='data/'):
        dataset = WordNet18(f'{root}wordnet18')[0]
        return cls.load_dataset(dataset, 'WN18')

    @classmethod
    def load_wordnet18rr(cls, root='data/'):
        dataset = WordNet18RR(f'{root}wordnet18rr')[0]
        return cls.load_dataset(dataset, 'WN18RR')

    @classmethod
    def load_fb15k(cls, root='data/'):
        dataset = CustomDataset(f'{root}fb15k', 'FB15k')[0]
        return cls.load_dataset(dataset, 'FB15k')

    @classmethod
    def load_fb15k_237(cls, root='data/'):
        dataset = CustomDataset(f'{root}fb15k-237', 'FB15k-237')[0]
        return cls.load_dataset(dataset, 'FB15k-237')

    @classmethod
    def load_str(cls, name, root='data/'):
        name = name.lower()
        if name == 'wordnet18':
            return cls.load_wordnet18(root)
        elif name == 'wordnet18rr':
            return cls.load_wordnet18rr(root)
        elif name == 'fb15k':
            return cls.load_fb15k(root)
        elif name == 'fb15k-237':
            return cls.load_fb15k_237(root)
        else:
            raise ValueError(f'{name} is not a valid dataset name')
