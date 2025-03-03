import numpy as np
from jax_dataclasses import pytree_dataclass
import jax.numpy as jnp
from typing import Literal, Optional

__all__ = ['BasicGraphData']


@pytree_dataclass
class BasicGraphData:
    edge_idx: jnp.ndarray
    edge_type: jnp.ndarray
    relation_sizes: np.ndarray
    sorted_by: Optional[Literal['head', 'tail']] = None

    @classmethod
    def from_dataset(cls, edge_idx, edge_type):
        assert isinstance(edge_idx, np.ndarray) and isinstance(edge_type, np.ndarray)
        relation_sizes = np.bincount(edge_type, minlength=edge_type.max() + 1)
        return cls(edge_idx, edge_type, relation_sizes)

    def relation_sizes(self):
        pass

    def as_sorted_by(self, /, x: Literal['head', 'tail']):
        pass

    def __getitem__(self, mask):
        """Mask the graph by a boolean mask"""
        return self.__class__(self.edge_idx[2, mask], self.edge_type[mask])