from __future__ import annotations

import gc
import logging
import sys
import pickle
from typing import Optional

import jax
import numpy as np

import optax
from tqdm import trange


from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.models.link_prediction import GenericShallowModel, TransEModel, RGCNModel, \
    RGCNModelTrainingData, BasicModelData, CombinedModel, DoubleRGCNModel, LearnedEnsembleModel
from rgcn.layers.decoder import DistMult, ComplEx, SimplE, TransE, RESCAL
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from rgcn.data.sampling import make_dense_batched_negative_sample, make_dense_batched_negative_sample_dense_rel
from rgcn.evaluation.mrr import generate_unfiltered_mrr, generate_filtered_mrr, MRRResults
from rgcn.data.datasets.entity_classification import make_dense_relation_tensor

jax.config.update('jax_log_compiles', False)

# jax.config.update('jax_platform_name', 'cpu')


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


def save_model(model, model_config):
    flattened_model = jax.tree_util.tree_flatten(model)
    with open(f'{model_config.name}_model', 'wb') as f:
        pickle.dump(flattened_model[0], f)


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
    'rgcn': RGCNModel.Config(decoder_class=DistMult, hidden_channels=[200], normalizing_constant='per_node',
                             edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=0.01, name='RGCN',
                             epochs=50, learning_rate=0.05, seed=42, n_decomp=2, decomposition_method='basis'),
    'combined': CombinedModel.Config(decoder_class=SimplE, hidden_channels=[400], normalizing_constant='per_node',
                                     edge_dropout_rate=0.5, node_dropout_rate=None, l2_reg=None, name='Combined',
                                     epochs=500, learning_rate=0.01, seed=42, decomposition_method='basis', n_decomp=2),
    'doublergcn': DoubleRGCNModel.Config(decoder_class=SimplE, hidden_channels=[150], normalizing_constant='per_node',
                                         edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=None, name='DoubleRGCN',
                                         epochs=250, learning_rate=0.05, seed=42, decomposition_method='basis', n_decomp=2),
    'learnedensemble': LearnedEnsembleModel.Config(decoder_class=DistMult, hidden_channels=[200],
                                                   normalizing_constant='per_node',
                                                   edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=None,
                                                   name='LearnedEnsemble',
                                                   epochs=600, learning_rate=0.05, seed=42,
                                                   decomposition_method='basis',
                                                   n_channels=200, n_embeddings=1, normalization=False, n_decomp=2),
    'simpleensemble': LearnedEnsembleModel.Config(decoder_class=SimplE, hidden_channels=[300],
                                                  normalizing_constant='per_node',
                                                  edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=None,
                                                  name='SimplEnsemble',
                                                  epochs=600, learning_rate=0.05, seed=42,
                                                  decomposition_method='basis',
                                                  n_channels=150, n_embeddings=2, normalization=False, n_decomp=2)
}


def make_train_step(num_nodes, all_data, optimizer, val_data, get_train_epoch_data_fast):
    @eqx.filter_jit
    def train_step(*, model, opt_state, key):
        data_key, model_key = jrandom.split(key)
        train_data, pos_mask = get_train_epoch_data_fast(key=data_key)

        loss, grads = loss_fn(model, all_data, train_data, pos_mask, key=model_key)
        updates, opt_state = optimizer.update(grads, opt_state)

        model = eqx.apply_updates(model, updates)
        _, _, val_mrr_results = generate_unfiltered_mrr(num_nodes=num_nodes, model=model, test_data=val_data,
                                                        all_data=all_data)
        return model, opt_state, loss, val_mrr_results.mrr

    return train_step


def train(model: Optional[str] = None, dataset: Optional[str] = None):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('-' * 50)
    dataset = LinkPredictionWrapper.load_fb15k_237()
    # dataset = LinkPredictionWrapper.load_wordnet18()
    logging.info(dataset.name)

    model_config = model_configs['rgcn']
    model_init_key, key = jrandom.split(jrandom.PRNGKey(model_config.seed))
    model = model_config.get_model(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations, key=model_init_key)

    logging.info(str(model_config))
    logging.info(str(model))

    optimizer = optax.adam(learning_rate=model_config.learning_rate)  # ComplEx

    val_edge_index = dataset.edge_index[:, dataset.val_idx]
    val_edge_type = dataset.edge_type[dataset.val_idx]

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
    del dense_relation, dense_mask, complete_pos_edge_index, complete_pos_edge_type

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # if model.data_class.is_dense:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_dense(pos_edge_index, pos_edge_type, num_nodes)
    # else:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)
    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)

    val_data = jnp.concatenate((val_edge_index, val_edge_type.reshape(1, -1)), axis=0)

    loss = None
    i = None

    train_step = make_train_step(dataset.num_nodes, all_data, optimizer, val_data, get_train_epoch_data_fast)

    best_model = None
    best_val_mrr = 0.0

    try:
        for i in pbar:
            train_key, key = jrandom.split(key)
            model, opt_state, loss, val_mrr = train_step(model=model, opt_state=opt_state, key=train_key)

            if val_mrr > best_val_mrr:
                logging.info(f'New best model found at epoch {i} with val_mrr {val_mrr}')
                best_val_mrr = val_mrr
                best_model = model

            pbar.set_description(f'\tLoss: {loss}, val_mrr: {val_mrr}')
            pbar.refresh()
            if hasattr(model, 'alpha'):
                object.__setattr__(model, 'alpha', jnp.clip(model.alpha, 0, 1))
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    model = best_model

    logging.info(f'Final loss: {loss}')

    del val_data, train_step, get_train_epoch_data_fast, opt_state, loss
    gc.collect()

    # Generate MRR results

    model.normalize()

    if hasattr(model, 'alpha'):
        logging.info(f'Alpha: {model.alpha}')

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset.num_nodes, model,
                                                                                           test_data, all_data,
                                                                                           force_cpu=True)

    logging.info(f'Unfiltered: {unfiltered_results}')

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data,
                                             test_edge_index, force_cpu=True)

    logging.info(f'Filtered: {filtered_results}')


def test_ensemble():
    logging.info('-' * 50)
    # dataset = LinkPredictionWrapper.load_fb15k_237()
    dataset = LinkPredictionWrapper.load_wordnet18()
    logging.info(dataset.name)

    with open(f'DistMult_model', 'rb') as f:
        distmult_load_model = pickle.load(f)
        distmult_load_model = jax.tree_map(jnp.array, distmult_load_model, is_leaf=lambda x: isinstance(x, np.ndarray))

    with open(f'RGCN_model', 'rb') as f:
        rgcn_load_model = pickle.load(f)
        rgcn_load_model = jax.tree_map(jnp.array, rgcn_load_model, is_leaf=lambda x: isinstance(x, np.ndarray))

    distmult_config = model_configs['distmult']
    rgcn_config = model_configs['rgcn']

    model_init_key, key = jrandom.split(jrandom.PRNGKey(distmult_config.seed), 2)
    distmult_model = distmult_config.get_model(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations,
                                               key=model_init_key)
    distmult_treedef = jax.tree_util.tree_flatten(distmult_model)[1]
    distmult_model = jax.tree_util.tree_unflatten(distmult_treedef, distmult_load_model)

    model_init_key, key = jrandom.split(jrandom.PRNGKey(rgcn_config.seed), 2)
    rgcn_model = rgcn_config.get_model(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations,
                                       key=model_init_key)
    rgcn_treedef = jax.tree_util.tree_flatten(rgcn_model)[1]
    rgcn_model = jax.tree_util.tree_unflatten(rgcn_treedef, rgcn_load_model)

    ensemble_model = EnsembleModel(distmult_model, rgcn_model, key)

    # logging.info(str(distmult_config))
    logging.info(str(ensemble_model))

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes

    complete_pos_edge_index = jnp.concatenate((pos_edge_index, jnp.flip(pos_edge_index, axis=0)), axis=1)
    complete_pos_edge_type = jnp.concatenate((pos_edge_type, pos_edge_type + dataset.num_relations))
    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=2 * dataset.num_relations,
                                                            edge_index=complete_pos_edge_index,
                                                            edge_type=complete_pos_edge_type)
    all_data = RGCNModelTrainingData(jnp.asarray(dense_relation), jnp.asarray(dense_mask))

    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset, ensemble_model,
                                                                                           test_data,
                                                                                           test_edge_index, all_data)

    logging.info(f'Unfiltered: {unfiltered_results}')

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data,
                                             test_edge_index)

    logging.info(f'Filtered: {filtered_results}')


if __name__ == '__main__':
    train()
