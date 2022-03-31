from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax

import numpy as np
from jax import numpy as jnp
from tqdm import trange

from rgcn.utils import memory
from rgcn.utils.algorithms import parallel_argsort_last


@memory.cache
def generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data):
    n_test_edges = test_data.shape[1]
    result_head = np.empty((n_test_edges, num_nodes), dtype=bool)
    result_tail = np.empty((n_test_edges, num_nodes), dtype=bool)
    for i in trange(n_test_edges):
        head = test_data[0, i]
        tail = test_data[1, i]

        mask_head = np.zeros(num_nodes, dtype=bool)
        mask_head[edge_index[0, (edge_type == test_data[2, i]) & (edge_index[1, :] == tail)]] = True
        mask_head[head] = False
        result_head[i, :] = mask_head

        mask_tail = np.zeros(num_nodes, dtype=bool)
        mask_tail[edge_index[1, (edge_type == test_data[2, i]) & (edge_index[0, :] == head)]] = True
        mask_tail[tail] = False
        result_tail[i, :] = mask_tail

    return result_head, result_tail


def test_generate_mrr_filter_mask_numba():
    print()
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_type = np.array([0, 0, 1, 1, 2])
    num_nodes = 5
    test_data = np.array([[0, 1, 0], [1, 2, 0], [0, 3, 1], [3, 4, 1], [2, 4, 2]])
    head_result, tail_result = generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data)
    print(head_result)
    print(tail_result)


def test_generate_mrr2():
    print()
    edge_index = np.array([[0, 1, 2, 0, 3, 3, 4],
                           [3, 4, 0, 1, 1, 2, 3]])
    edge_type = np.array([0, 0, 0, 1, 1, 1, 0])
    test_edge_index = np.array([[0, 2, 3],
                                [3, 0, 1]])
    test_edge_type = np.array([0, 0, 1])
    test_data = np.concatenate((test_edge_index, test_edge_type[None, :]), axis=0)
    num_nodes = 5

    head_mask, tail_mask = generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data)

    print(head_mask)
    print(tail_mask)


@dataclass
class MRRResults:
    mrr: float
    hits_at_10: float
    hits_at_3: float
    hits_at_1: float

    def average_with(self, other: MRRResults):
        return MRRResults(
            mrr=(self.mrr + other.mrr) / 2,
            hits_at_10=(self.hits_at_10 + other.hits_at_10) / 2,
            hits_at_3=(self.hits_at_3 + other.hits_at_3) / 2,
            hits_at_1=(self.hits_at_1 + other.hits_at_1) / 2
        )

    @classmethod
    def generate_from(cls, head_hrt_scores, tail_hrt_scores, test_edge_index):
        head_results = mean_reciprocal_rank_and_hits(head_hrt_scores, test_edge_index, 'head')
        tail_results = mean_reciprocal_rank_and_hits(tail_hrt_scores, test_edge_index, 'tail')
        return head_results.average_with(tail_results)


def mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index, corrupt: Literal['head', 'tail']) -> MRRResults:
    hrt_scores, test_edge_index = map(np.array, (hrt_scores, test_edge_index))
    assert corrupt in ['head', 'tail']

    # hrt_scores: (n_test_edges, n_nodes)
    perm = parallel_argsort_last(-hrt_scores)
    # Find the location of the true edges in the sorted list
    if corrupt == 'head':
        mask = perm == test_edge_index[0, :].reshape((-1, 1))
    else:
        mask = perm == test_edge_index[1, :].reshape((-1, 1))

    # Get the index of the true edges in the sorted list
    print(mask.shape)
    true_index = np.argmax(mask, axis=1) + 1
    print(true_index.shape)
    # Get the reciprocal rank of the true edges
    rr = 1.0 / true_index.astype(np.float32)

    # Get the mean reciprocal rank
    mrr = np.mean(rr)

    # Get the hits@10 of the true edges
    hits10 = np.sum(mask[:, :10], axis=1, dtype=np.float32).mean()
    # Get the hits@3 of the true edges
    hits3 = np.sum(mask[:, :3], axis=1, dtype=np.float32).mean()
    # Get the hits@1 of the true edges
    hits1 = np.sum(mask[:, :1], axis=1, dtype=np.float32).mean()
    return MRRResults(mrr, hits10, hits3, hits1)


def generate_unfiltered_mrr(dataset, model, test_data, test_edge_index, all_data):
    print('Computing scores')
    head_corrupt_scores = make_generate_logits(model, dataset.num_nodes, all_data, mode='head')(test_data)
    print('Computed head scores')
    tail_corrupt_scores = make_generate_logits(model, dataset.num_nodes, all_data, mode='tail')(test_data)
    print('Computed tail scores')
    unfiltered_results = MRRResults.generate_from(head_corrupt_scores, tail_corrupt_scores, test_edge_index)
    return head_corrupt_scores, tail_corrupt_scores, unfiltered_results


def generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data, test_edge_index):
    mask_head, mask_tail = generate_mrr_filter_mask(np.array(dataset.edge_index), np.array(dataset.edge_type),
                                                    num_nodes, np.array(test_data))
    head_filtered_scores = filter_scores(scores=head_corrupt_scores, mask=mask_head)
    tail_filtered_scores = filter_scores(scores=tail_corrupt_scores, mask=mask_tail)
    filtered_results = MRRResults.generate_from(head_filtered_scores, tail_filtered_scores, test_edge_index)
    return filtered_results


def filter_scores(scores, mask):
    return scores.at[mask].set(-jnp.inf)


def make_generate_logits(model, num_nodes, all_data, batch_dim=50, mode: Literal['head', 'tail'] = 'head'):
    node_embeddings = model.get_node_embeddings(all_data)

    @eqx.filter_jit
    def generate_logits(test_data):
        test_data = jnp.transpose(test_data)

        # test_data: [n_test_edges, 3]
        @jax.vmap  # [n_test_edges, 3] -> [n_test_edges, n_nodes]
        def loop(x):  # [3,] -> [n_nodes,]
            head, tail, relation_type = x[0], x[1], x[2]
            if mode == 'head':
                return model.forward_heads(node_embeddings, relation_type, tail)
            else:
                return model.forward_tails(node_embeddings, relation_type, head)

        # Batch the test data
        # batched_test_data = rearrange(test_data, 'tuple (batch_size batch_dim) -> batch_size tuple batch_dim', batch_size=batch_size)
        batched_test_data = test_data.reshape((-1, batch_dim, 3))  # [batch_size, n_test_edges, 3]

        return jax.lax.map(loop, batched_test_data).reshape((-1, num_nodes))

    return generate_logits
