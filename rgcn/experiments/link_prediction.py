from __future__ import annotations

import logging
import sys

import jax
import optax
from tqdm import trange

logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.models.link_prediction import GenericShallowModel, TransEModel, RGCNModel, \
    RGCNModelTrainingData, BasicModelData
from rgcn.layers.decoder import DistMult, ComplEx, SimplE, TransE, RESCAL
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from rgcn.data.sampling import make_dense_batched_negative_sample, make_dense_batched_negative_sample_dense_rel
from rgcn.evaluation.mrr import generate_unfiltered_mrr, generate_filtered_mrr
from rgcn.data.datasets.entity_classification import make_dense_relation_tensor


# jax.config.update('jax_log_compiles', True)


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
    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=pos_edge_type.max() + 1,
                                                            edge_index=pos_edge_index, edge_type=pos_edge_type)

    dense_batched_negative_sample = make_dense_batched_negative_sample_dense_rel(edge_index=pos_edge_index,
                                                                                 edge_type=pos_edge_type,
                                                                                 num_nodes=num_nodes,
                                                                                 )

    dense_mask = jnp.array(dense_mask, dtype=jnp.bool_)
    dense_relation = jnp.array(dense_relation)
    doubled_dense_mask = jnp.repeat(dense_mask, repeats=2, axis=-1)

    @jax.jit
    def perform(key):
        neg_dense_edge_index = dense_batched_negative_sample(key=key)

        full_dense_edge_index = jnp.concatenate((dense_relation, neg_dense_edge_index), axis=-1)

        pos_mask = jnp.concatenate((dense_mask, jnp.zeros_like(dense_mask)), axis=1)
        return RGCNModelTrainingData(edge_type_idcs=full_dense_edge_index, edge_masks=doubled_dense_mask, ), pos_mask

    return perform


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, all_data: RGCNModelTrainingData, data: BasicModelData, mask, key):
    return model.loss(data.edge_index, data.edge_type, mask, all_data, key=key) + model.l2_loss() / (
            2 * data.edge_index.shape[1])


model_configs = {
    'distmult': GenericShallowModel.Config(decoder_class=DistMult, n_channels=100, l2_reg=None, name='DistMult',
                                           n_embeddings=1, normalization=True,
                                           epochs=600, learning_rate=0.5, seed=42),
    'complex': GenericShallowModel.Config(decoder_class=ComplEx, n_channels=200, l2_reg=None, name='ComplEx',
                                   n_embeddings=2, normalization=False,
                                   epochs=100, learning_rate=0.05, seed=42),
    'simple': GenericShallowModel.Config(decoder_class=SimplE, n_channels=150, l2_reg=None, name='SimplE',
                                 n_embeddings=2, normalization=False,
                                 epochs=100, learning_rate=0.05, seed=42),
    'rescal': GenericShallowModel.Config(decoder_class=RESCAL, n_channels=100, l2_reg=None, name='RESCAL',
                                         n_embeddings=1, normalization=True,
                                         epochs=200, learning_rate=0.5, seed=42),
    'transe': TransEModel.Config(decoder_class=TransE, n_channels=50, margin=2, l2_reg=None, name='TransE',
                                 n_embeddings=1, normalization=True,
                                 epochs=1000, learning_rate=0.01, seed=42),
    'rgcn': RGCNModel.Config(decoder_class=DistMult, hidden_channels=[500, 500], normalizing_constant='per_node',
                             edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=0.01, name='RGCN',
                             epochs=250, learning_rate=0.05, seed=42, decomposition_method='block')
}


def train():
    logging.info('-'*50)
    dataset = LinkPredictionWrapper.load_fb15k_237()
    # dataset = LinkPredictionWrapper.load_wordnet18()

    model_config = model_configs['rgcn']
    model_init_key, key = jrandom.split(jrandom.PRNGKey(model_config.seed))
    model = model_config.get_model(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations, key=model_init_key)

    logging.info(str(model_config))
    logging.info(str(model))

    optimizer = optax.adam(learning_rate=model_config.learning_rate)  # ComplEx

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    num_epochs = model_config.epochs  # and 1

    pbar = trange(num_epochs)
    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes

    complete_pos_edge_index = jnp.concatenate((pos_edge_index, jnp.flip(pos_edge_index, axis=0)), axis=1)
    complete_pos_edge_type = jnp.concatenate((pos_edge_type, pos_edge_type + dataset.num_relations))
    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=2 * dataset.num_relations,
                                                            edge_index=complete_pos_edge_index,
                                                            edge_type=complete_pos_edge_type)
    all_data = RGCNModelTrainingData(jnp.asarray(dense_relation), jnp.asarray(dense_mask))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # if model.data_class.is_dense:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_dense(pos_edge_index, pos_edge_type, num_nodes)
    # else:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)
    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)

    opt_update = jax.jit(optimizer.update)

    i = None
    try:
        for i in pbar:
            data_key, model_key, key = jrandom.split(key, 3)
            train_data, pos_mask = get_train_epoch_data_fast(key=data_key)

            loss, grads = loss_fn(model, all_data, train_data, pos_mask, key=model_key)
            updates, opt_state = opt_update(grads, opt_state)

            pbar.set_description(f'\tLoss: {loss}')
            pbar.refresh()
            model = eqx.apply_updates(model, updates)
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    # Generate MRR results

    model.normalize()
    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset, model, test_data,
                                                                                           test_edge_index, all_data)

    logging.info(f'Unfiltered: {unfiltered_results}')

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data,
                                             test_edge_index)

    logging.info(f'Filtered: {filtered_results}')


if __name__ == '__main__':
    train()
