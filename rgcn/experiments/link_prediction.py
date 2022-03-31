from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Literal

import jax
import optax
from tqdm import trange

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.models.link_prediction import GenericModel, TransEModel, ComplExModel, SimplEModel
from rgcn.layers.decoder import DistMult, RESCAL
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from rgcn.utils.algorithms import parallel_argsort_last, generate_mrr_filter_mask

import numpy as np
from functools import partial

from joblib import Memory

memory = Memory('/tmp/joblib')

jax.log_compiles(True)


# WordNet18: {n_nodes: 40_000, n_test_edges: 5000}


def wrapper(model, num_nodes, batch_dim=50, mode: Literal['head', 'tail'] = 'head'):
    @eqx.filter_jit
    def generate_logits(test_data):
        test_data = jnp.transpose(test_data)

        # test_data: [n_test_edges, 3]
        @jax.vmap  # [n_test_edges, 3] -> [n_test_edges, n_nodes]
        def loop(x):  # [3,] -> [n_nodes,]
            head, tail, relation_type = x[0], x[1], x[2]
            if mode == 'head':
                return model.forward_heads(relation_type, tail)
            else:
                return model.forward_tails(relation_type, head)

        # Batch the test data
        # batched_test_data = rearrange(test_data, 'tuple (batch_size batch_dim) -> batch_size tuple batch_dim', batch_size=batch_size)
        batched_test_data = test_data.reshape((-1, batch_dim, 3))  # [batch_size, n_test_edges, 3]

        return jax.lax.map(loop, batched_test_data).reshape((-1, num_nodes))

    return generate_logits


def encode(edge_index, num_nodes):
    row, col = edge_index[0], edge_index[1]
    return row * num_nodes + col


def encode_with_type(edge_index, num_nodes, edge_type):
    row, col = edge_index[0], edge_index[1]
    return edge_type * num_nodes * num_nodes + row * num_nodes + col


def negative_sample(edge_index, num_nodes, num_negatives, key):
    """Generates negative samples completely randomly (no corruption)"""
    # Generate random edges
    rand = jrandom.randint(key, (2, num_negatives), 0, num_nodes)  # [num_negatives, ]

    real_coding = encode(edge_index, num_nodes)  # [num_edges, ]
    rand_coding = encode(rand, num_nodes)  # [num_negatives, ]

    positive_samples_mask = jnp.isin(rand_coding, real_coding).reshape(1, num_negatives)  # [1, num_negatives]

    negative_samples = jnp.where(positive_samples_mask, jrandom.randint(key, (2, num_negatives), 0, num_nodes), rand)

    return negative_samples  # [2, num_negatives]


# @memory.cache
def make_dense_relation_edges(edge_index, edge_type, num_nodes):
    n_relations = edge_type.max() + 1
    # Reshape the edge_index matrix (and edge_type) to [n_relations, num_nodes, max_num_neighbors]
    result = []
    max_num_neighbors = 0
    for relation in trange(n_relations, desc='Finding neighbors'):
        gc.collect()
        rel_edge_index = edge_index[:, edge_type == relation]
        rel_result = []
        for head in range(num_nodes):
            head_mask = rel_edge_index[0] == head
            tails_for_head = rel_edge_index[1, head_mask]
            max_num_neighbors = max(max_num_neighbors, tails_for_head.shape[0])
            rel_result.append(tails_for_head)
        result.append(rel_result)

    # Construct a result array
    result_tensor = np.zeros((n_relations, num_nodes, max_num_neighbors), dtype=int)

    # Pad the inner arrays to shape [max_num_neighbors]
    for relation in trange(n_relations, desc='Writing to tensor'):
        for head in range(num_nodes):
            head_edges = result[relation][head]
            result_tensor[relation][head] = np.pad(head_edges, (0, max_num_neighbors - head_edges.shape[0]), 'constant',
                                                   constant_values=-1)
    return result_tensor


def make_dense_batched_negative_sample(edge_index, edge_type, num_nodes):
    # dense_tensor = make_dense_relation_edges(*map(np.array, (edge_index, edge_type, num_nodes)))
    # dense_tensor = jnp.array(dense_tensor, dtype=jnp.int32)
    # gc.collect()

    # dense_tensor: [n_relations, num_nodes, max_num_neighbors]

    # @partial(jax.vmap, in_axes=1, out_axes=0)
    # def isin(triple):
    #     head, tail, edge_type = triple
    #     dense_edge_index = dense_tensor[edge_type, head]
    #     return jnp.isin(tail, dense_edge_index)

    def perform(key):
        # Always generate one negative sample per positive sample
        num_edges = edge_index.shape[1]
        rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]
        rand2 = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

        head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
        head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

        maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]

        # maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)],
        #                                          axis=0)  # [3, num_edges]
        #
        # positive_mask = isin(maybe_negative_triples).reshape(1, num_edges)  # [1, num_edges]
        # head_or_tail_positive = head_or_tail * positive_mask  # [2, num_edges]
        #
        # definitely_negative_samples = jnp.where(head_or_tail_positive, rand2, maybe_negative_samples)  # [2, num_edges]
        return maybe_negative_samples

    return perform


def batched_negative_sample(edge_index, edge_type, num_nodes, key):
    # Always generate one negative sample per positive sample
    num_edges = edge_index.shape[1]
    rand = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]
    rand2 = jrandom.randint(key, (1, num_edges), 0, num_nodes)  # [1, num_edges]

    head_or_tail = jrandom.bernoulli(key, 0.5, (1, num_edges))  # [1, num_edges]
    head_tail_mask = jnp.concatenate([head_or_tail, ~head_or_tail], axis=0)  # [2, num_edges]

    maybe_negative_samples = jnp.where(head_tail_mask, rand, edge_index)  # [2, num_edges]

    real_triples = jnp.concatenate([edge_index, edge_type.reshape(1, -1)], axis=0)  # [3, num_edges]
    encoded_real_triples = encode_with_type(real_triples, num_nodes, edge_type)  # [num_edges, ]

    maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)],
                                             axis=0)  # [3, num_edges]
    maybe_negative_encoded_triples = encode_with_type(maybe_negative_triples, num_nodes, edge_type)  # [num_edges, ]

    positive_samples_mask = jnp.isin(maybe_negative_encoded_triples, encoded_real_triples).reshape(1,
                                                                                                   num_edges)  # [1, num_edges]
    head_tail_adjusted_positive_samples_mask = head_tail_mask * positive_samples_mask  # [2, num_edges]
    definitely_negative_samples = jnp.where(head_tail_adjusted_positive_samples_mask, rand2,
                                            maybe_negative_samples)  # [2, num_edges]

    return definitely_negative_samples


def get_train_epoch_data(dataset: LinkPredictionWrapper, key):
    neg_edge_index = jnp.zeros((2, 0), dtype=jnp.int32)
    neg_edge_type = jnp.zeros(0, dtype=jnp.int32)

    pos_edge_type = dataset.edge_type[dataset.train_idx]
    pos_edge_index = dataset.edge_index[:, dataset.train_idx]
    # pos_count = dataset.train_idx.sum()

    for i in range(0, dataset.num_relations):
        iter_key, key = jrandom.split(key)
        rel_indices = pos_edge_type == i
        rel_edge_index = pos_edge_index[:, rel_indices]
        rel_count = rel_indices.sum()
        rel_neg_edge_index = negative_sample(rel_edge_index, dataset.num_nodes, rel_count, iter_key)
        neg_edge_index = jnp.concatenate((neg_edge_index, rel_neg_edge_index), axis=1)
        neg_edge_type = jnp.concatenate((neg_edge_type, jnp.ones(rel_count, dtype=jnp.int32) * i))

    full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
    full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)
    pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))
    return full_edge_index, full_edge_type, pos_mask


def make_get_train_epoch_data_fast(pos_edge_index, pos_edge_type, num_nodes):
    dense_batched_negative_sample = make_dense_batched_negative_sample(edge_index=pos_edge_index,
                                                                       edge_type=pos_edge_type, num_nodes=num_nodes)

    @jax.jit
    def perform(key):
        neg_edge_index = dense_batched_negative_sample(key=key)
        neg_edge_type = pos_edge_type

        full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
        full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)

        pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))
        return full_edge_index, full_edge_type, pos_mask

    return perform


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, edge_index, edge_type, mask):
    return model.loss(edge_index, edge_type, mask) + model.l2_loss() / (2 * edge_index.shape[1])
    # 50: Filtered: MRRResults(mrr=0.5456010103225708, hits_at_10=0.8876000046730042, hits_at_3=0.708899974822998, hits_at_1=0.34860000014305115)
    # 200, oldsampling: Filtered: MRRResults(mrr=0.7419325709342957, hits_at_10=0.9191000461578369, hits_at_3=0.8622000217437744, hits_at_1=0.6100000143051147)
    # 200: 0.8109014630317688, loss=0.00489
    # 200 (norm): 0.357095, loss=0.15281

    #50 RESCAL: Filtered: MRRResults(mrr=0.6268602013587952, hits_at_10=0.8105000257492065, hits_at_3=0.6930999755859375, hits_at_1=0.5256999731063843)
    #50: RESCAL: Filtered: MRRResults(mrr=0.6216883659362793, hits_at_10=0.8100999593734741, hits_at_3=0.6886999607086182, hits_at_1=0.5184999704360962)

@dataclass
class MRRResults:
    mrr: float
    hits_at_10: float
    hits_at_3: float
    hits_at_1: float

    def average_with(self, other: MRRResults):
        print('Averaging', self)
        print('with', other)
        return MRRResults(
            mrr=(self.mrr + other.mrr) / 2,
            hits_at_10=(self.hits_at_10 + other.hits_at_10) / 2,
            hits_at_3=(self.hits_at_3 + other.hits_at_3) / 2,
            hits_at_1=(self.hits_at_1 + other.hits_at_1) / 2
        )

    @classmethod
    def generate_from(cls, hrt_scores, test_edge_index):
        head_results = mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index, 'head')
        tail_results = mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index, 'tail')
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


def train():
    seed = 42
    key = jrandom.PRNGKey(seed)
    dataset = LinkPredictionWrapper.load_wordnet18()
    model = GenericModel(RESCAL, dataset.num_nodes, dataset.num_relations, 100, key)  # same settings for DistMult and RESCAL
    optimizer = optax.adam(learning_rate=0.5)
    # model = ComplExModel(dataset.num_nodes, dataset.num_relations, 150, key)
    # optimizer = optax.adam(learning_rate=0.05)  # ComplEx
    # model = SimplEModel(dataset.num_nodes, dataset.num_relations, 150, key)  # same settings for SimplE and ComplEx
    # optimizer = optax.adam(learning_rate=0.05)  # ComplEx
    # model = TransEModel(dataset.num_nodes, dataset.num_relations, 50, 2, key)
    # optimizer = optax.adam(learning_rate=0.01)  # TransE
    opt_state = optimizer.init(model)

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    num_epochs = 500  # TransE: 1000 epochs, ComplEx: 100–200 epochs, RESCAL: 500 epochs

    t = trange(num_epochs)
    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes
    get_train_epoch_data_fast = make_get_train_epoch_data_fast(pos_edge_index, pos_edge_type, num_nodes)
    try:
        for i in t:
            use_key, key = jrandom.split(key)
            edge_index, edge_type, pos_mask = get_train_epoch_data_fast(key=use_key)
            loss, grads = loss_fn(model, edge_index, edge_type, pos_mask)
            updates, opt_state = optimizer.update(grads, opt_state)
            # mean_test_score = model(test_edge_index, test_edge_type).mean()
            # t.set_description(f'\tLoss: {loss}, Mean Test Score: {mean_test_score}')
            t.set_description(f'\tLoss: {loss}')
            t.refresh()
            model = eqx.apply_updates(model, updates)
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    # key1, key2 = jrandom.split(jrandom.PRNGKey(seed))

    model.normalize()
    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]
    print('Starting test scores')
    head_corrupt_scores = wrapper(model, dataset.num_nodes, mode='head')(test_data)
    print('Computed head scores')
    tail_corrupt_scores = wrapper(model, dataset.num_nodes, mode='tail')(test_data)
    print('Finished test scores')
    # print(test_data.shape)
    # print(test_scores.shape)
    # print(test_scores)
    # print(test_edge_index)
    # print(test_edge_type)
    # print(test_edge_index[1, :].shape)
    # print(test_scores.shape)
    # print(test_edge_index[1, :].min())
    # print(test_edge_index[1, :].max())
    # print(test_edge_index[1, :].dtype)

    head_mrr = mean_reciprocal_rank_and_hits(head_corrupt_scores, test_edge_index, 'head')
    tail_mrr = mean_reciprocal_rank_and_hits(tail_corrupt_scores, test_edge_index, 'tail')

    print('Unfiltered:', head_mrr.average_with(tail_mrr))

    mask_head, mask_tail = generate_mrr_filter_mask(np.array(dataset.edge_index), np.array(dataset.edge_type), num_nodes, np.array(test_data))

    head_filtered_scores = head_corrupt_scores.at[mask_head].set(-jnp.inf)
    tail_filtered_scores = tail_corrupt_scores.at[mask_tail].set(-jnp.inf)

    head_mrr = mean_reciprocal_rank_and_hits(head_filtered_scores, test_edge_index, 'head')
    tail_mrr = mean_reciprocal_rank_and_hits(tail_filtered_scores, test_edge_index, 'tail')

    print('Filtered:', head_mrr.average_with(tail_mrr))

    # f = wrapper(model, dataset.num_nodes, 5000)
    # test_data = jnp.concatenate((test_edge_index.reshape((-1, 2)), test_edge_type.reshape((-1, 1))), axis=1)
    # num_nodes = dataset.num_nodes

    # def do():
    #    res = f(test_data)
    #    print(res.shape)

    # print()
    # print(jnp.unique(mask, return_counts=True))
    # print(rand[:, mask])

    # print(timeit.timeit(do, number=1))
    # print(dataset.edge_type.dtype)
    # print(dataset.edge_index.dtype)
    # print(jnp.concatenate((test_edge_index.reshape((-1, 2)), test_edge_type.reshape((-1, 1))), axis=1).shape)


if __name__ == '__main__':
    train()
