from dataclasses import dataclass
from typing import Literal

import jax
import optax
from jax import jit
from tqdm import trange

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.models.link_prediction import DistMultModel, compute_loss
import jax.random as jrandom
import jax.numpy as jnp
from jax import lax, jit
import timeit
import equinox as eqx
from einops import rearrange

jax.log_compiles(True)


def get_head_corrupted(head, tail, num_nodes):
    range_n = jnp.arange(0, num_nodes, dtype=jnp.int32)
    return jnp.stack((range_n, jnp.ones(num_nodes, dtype=jnp.int32) * tail))


def get_tail_corrupted(head, tail, num_nodes):
    range_n = jnp.arange(0, num_nodes, dtype=jnp.int32)
    return jnp.stack((jnp.ones(num_nodes, dtype=jnp.int32) * head, range_n))


# WordNet18: {n_nodes: 40_000, n_test_edges: 5000}


def wrapper(model, num_nodes, num_relations, batch_dim=50):
    @eqx.filter_jit
    def generate_logits(test_data):
        # test_data: [n_test_edges, 3]
        @jax.vmap  # [n_test_edges, 3] -> [n_test_edges, n_nodes]
        def loop(x):  # [3,] -> [n_nodes,]
            # x: [3, ]
            head = x[0] # []
            tail = x[1] # []
            corrupted_edge_type = x[2].repeat(num_nodes)  # [num_nodes, ]
            corrupted_edge_index = get_head_corrupted(head, tail, num_nodes) # [2, num_nodes]
            scores = model(corrupted_edge_index, corrupted_edge_type)  # [num_nodes, ]
            #return jnp.array([head, tail, x[2]])
            return scores

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

    maybe_negative_triples = jnp.concatenate([maybe_negative_samples, edge_type.reshape(1, -1)], axis=0)  # [3, num_edges]
    maybe_negative_encoded_triples = encode_with_type(maybe_negative_triples, num_nodes, edge_type)  # [num_edges, ]

    positive_samples_mask = jnp.isin(maybe_negative_encoded_triples, encoded_real_triples).reshape(1, num_edges)  # [1, num_edges]
    head_tail_adjusted_positive_samples_mask = head_tail_mask * positive_samples_mask  # [2, num_edges]
    definitely_negative_samples = jnp.where(head_tail_adjusted_positive_samples_mask, rand2, maybe_negative_samples)  # [2, num_edges]

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


@jax.jit
def get_train_epoch_data_fast(pos_edge_index, pos_edge_type, num_nodes, key):
    neg_edge_index = batched_negative_sample(edge_index=pos_edge_index, edge_type=pos_edge_type, num_nodes=num_nodes, key=key)
    neg_edge_type = pos_edge_type

    full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
    full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)

    pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))
    return full_edge_index, full_edge_type, pos_mask


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, edge_index, edge_type, mask):
    scores = model(edge_index, edge_type)
    return compute_loss(scores, mask)


@dataclass
class MRRResults:
    mrr: float
    hits_at_10: float
    hits_at_3: float
    hits_at_1: float


@eqx.filter_jit
def mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index, corrupt: Literal['head', 'tail']):
    assert corrupt in ['head', 'tail']

    # hrt_scores: (n_test_edges, n_nodes)
    perm = jnp.argsort(-hrt_scores, axis=1)
    # Find the location of the true edges in the sorted list
    if corrupt == 'head':
        mask = perm == test_edge_index[0, :].reshape((-1, 1))
    else:
        mask = perm == test_edge_index[1, :].reshape((-1, 1))

    # Get the index of the true edges in the sorted list
    print(mask.shape)
    true_index = jnp.argmax(mask, axis=1) + 1
    print(true_index.shape)
    # Get the reciprocal rank of the true edges
    rr = 1.0 / true_index.astype(jnp.float32)

    # Get the mean reciprocal rank
    mrr = jnp.mean(rr)

    # Get the hits@10 of the true edges
    hits10 = jnp.sum(mask[:, :10], axis=1, dtype=jnp.float32).mean()
    # Get the hits@3 of the true edges
    hits3 = jnp.sum(mask[:, :3], axis=1, dtype=jnp.float32).mean()
    # Get the hits@1 of the true edges
    hits1 = jnp.sum(mask[:, :1], axis=1, dtype=jnp.float32).mean()
    return mrr, hits10, hits3, hits1 #  MRRResults(mrr, hits10, hits3, hits1)


def train():
    seed = 42
    key = jrandom.PRNGKey(seed)
    dataset = LinkPredictionWrapper.load_wordnet18()
    model = DistMultModel(dataset.num_nodes, dataset.num_relations, 100, key)
    optimizer = optax.adam(learning_rate=5e-1)
    opt_state = optimizer.init(model)

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    num_epochs = 50

    t = trange(num_epochs)
    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes
    try:
        for i in t:
            edge_index, edge_type, pos_mask = get_train_epoch_data_fast(pos_edge_index, pos_edge_type, num_nodes, key)
            loss, grads = loss_fn(model, edge_index, edge_type, pos_mask)
            updates, opt_state = optimizer.update(grads, opt_state)
            mean_test_score = model(test_edge_index, test_edge_type).mean()
            t.set_description(f'\tLoss: {loss}, Mean Test Score: {mean_test_score}')
            t.refresh()
            model = eqx.apply_updates(model, updates)
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    # key1, key2 = jrandom.split(jrandom.PRNGKey(seed))

    test_data = jnp.concatenate((test_edge_index.T, test_edge_type.reshape((-1, 1))), axis=1)  # [3, n_test_edges]
    test_scores = wrapper(model, dataset.num_nodes, dataset.test_idx.sum())(test_data)
    #print(test_data.shape)
    #print(test_scores.shape)
    #print(test_scores)
    #print(test_edge_index)
    #print(test_edge_type)
    #print(test_edge_index[1, :].shape)
    #print(test_scores.shape)
    #print(test_edge_index[1, :].min())
    #print(test_edge_index[1, :].max())
    #print(test_edge_index[1, :].dtype)
    print(test_scores.shape)
    print(jnp.choose(test_edge_index[0, :], test_scores.T).mean())
    print(test_scores)
    print(mean_reciprocal_rank_and_hits(test_scores, test_edge_index, 'head'))

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
