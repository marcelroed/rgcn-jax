from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from einops import rearrange
from tqdm import trange

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.data.sampling import make_dense_batched_negative_sample, make_dense_batched_negative_sample_dense_rel
from rgcn.evaluation.mrr import generate_unfiltered_mrr, generate_filtered_mrr
from rgcn.models.link_prediction import compute_loss, RGCNModel, RGCNModelTrainingData, BasicModelData
from rgcn.data.datasets.entity_classification import make_dense_relation_tensor

jax.config.update('jax_log_compiles', True)


# WordNet18: {n_nodes: 40_000, n_test_edges: 5000}


def make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes):
    dense_batched_negative_sample = make_dense_batched_negative_sample(edge_index=pos_edge_index,
                                                                       edge_type=pos_edge_type, num_nodes=num_nodes,
                                                                       num_edges=pos_edge_index.shape[1])

    @jax.jit
    def perform(key):
        neg_edge_index = dense_batched_negative_sample(key=key)
        neg_edge_type = pos_edge_type

        full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
        full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)

        pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))

        return BasicModelData(full_edge_index, full_edge_type), pos_mask

        # return full_edge_index, full_edge_type, pos_mask

    return perform


def make_get_epoch_train_data_dense(pos_edge_index, pos_edge_type, num_nodes):
    # Generate the dense representation of the positive edges once to determine the shape
    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=pos_edge_type.max() + 1, edge_index=pos_edge_index, edge_type=pos_edge_type)
    dense_relation_shape = dense_relation.shape

    dense_batched_negative_sample = make_dense_batched_negative_sample_dense_rel(edge_index=pos_edge_index,
                                                                       edge_type=pos_edge_type, num_nodes=num_nodes,
                                                                       )

    dense_mask = jnp.array(dense_mask, dtype=jnp.bool_)
    dense_relation = jnp.array(dense_relation)
    doubled_dense_mask = jnp.repeat(dense_mask, repeats=2, axis=-1)

    @jax.jit
    def perform(key):
        neg_dense_edge_index = dense_batched_negative_sample(key=key)

        full_dense_edge_index = jnp.concatenate((dense_relation, neg_dense_edge_index), axis=-1)

        pos_mask = jnp.concatenate((jnp.ones_like(dense_mask), jnp.zeros_like(dense_mask)), axis=1)
        return RGCNModelTrainingData(edge_type_idcs=full_dense_edge_index, edge_masks=doubled_dense_mask,), pos_mask

    return perform


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, data, mask):
    scores = model(data)
    return compute_loss(scores, mask)


model_configs = [
    RGCNModel.Config(hidden_channels=[16], name='Basic'),
]


def train():
    config = model_configs[0]
    print('Using model', config)

    seed = 42
    key = jrandom.PRNGKey(seed)
    dataset = LinkPredictionWrapper.load_wordnet18()
    model = config.get_model(dataset.num_nodes, dataset.num_relations, key)

    optimizer = optax.adam(learning_rate=5e-1)
    opt_state = optimizer.init(model)

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    num_epochs = 200

    t = trange(num_epochs)
    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes

    i = None

    if model.data_class.is_dense:
        get_train_epoch_data_fast = make_get_epoch_train_data_dense(pos_edge_index, pos_edge_type, num_nodes)
    else:
        get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)

    try:
        for i in t:
            use_key, key = jrandom.split(key)
            train_data, pos_mask = get_train_epoch_data_fast(key=use_key)
            loss, grads = loss_fn(model, train_data, pos_mask)
            updates, opt_state = optimizer.update(grads, opt_state)
            # mean_test_score = model(test_edge_index, test_edge_type).mean()
            # t.set_description(f'\tLoss: {loss}, Mean Test Score: {mean_test_score}')
            t.set_description(f'\tLoss: {loss}')
            t.refresh()
            model = eqx.apply_updates(model, updates)
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    # Generate MRR results

    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset, model, test_data,
                                                                                           test_edge_index)

    print('Unfiltered:', unfiltered_results)

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data,
                                             test_edge_index)

    print('Filtered:', filtered_results)


if __name__ == '__main__':
    train()
