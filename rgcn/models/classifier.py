from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing_extensions import Literal

from rgcn.layers.rgcn import RGCNConv, FastRGCNConv
from rgcn.data import BasicGraphData


class RGCNClassifier(eqx.Module):
    n_nodes: int
    n_relations: int
    hidden_channels: int
    n_classes: int
    layers: any
    l2_reg: Optional[float]

    def __init__(self, n_nodes, n_relations, hidden_channels, n_classes,
                 decomposition_method: Literal['none', 'basis', 'block'], n_decomp, l2_reg, key):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.l2_reg = l2_reg
        key1, key2 = jrandom.split(key, 2)
        self.layers = [
            FastRGCNConv(in_channels=n_nodes, out_channels=hidden_channels, n_relations=n_relations, n_decomp=n_decomp, key=key1),
            FastRGCNConv(in_channels=n_nodes, out_channels=hidden_channels, n_relations=n_relations, n_decomp=n_decomp, key=key2)
        ]

    @eqx.filter_jit
    def __call__(self, x, graph_data: BasicGraphData):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x, graph_data))
        x = self.layers[-1](x, graph_data)
        return x

    def l2_loss(self):
        if self.l2_reg is None:
            return jnp.array(0.)
        return self.layers[0].l2_loss()


def test_rgcn_classifier_inference():
    n_nodes = 10
    n_relations = 2
    hidden_channels = 16
    n_classes = 3
    key = jrandom.PRNGKey(0)
    model = RGCNClassifier(n_nodes, n_relations, hidden_channels, n_classes, key)
    x = None
    edge_index = jax.random.randint(key, (2, n_relations), 0, 10)
    edge_type = jax.random.randint(key, (n_relations,), 0, 10)
    y = model(x, edge_index, edge_type)
    print(y)
