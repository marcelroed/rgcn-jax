from __future__ import annotations

import gc
import logging
import sys
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from tqdm import trange

from rgcn.data.datasets.entity_classification import make_dense_relation_tensor
from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.data.sampling import make_dense_batched_negative_sample
from rgcn.data.utils import get_data_triples
from rgcn.evaluation.mrr import generate_unfiltered_mrr, generate_filtered_mrr
from rgcn.layers.decoder import DistMult, ComplEx, SimplE, TransE, RESCAL
from rgcn.models.link_prediction import GenericShallowModel, TransEModel, RGCNModel, \
    RGCNModelTrainingData, BasicModelData, CombinedModel, DoubleRGCNModel, LearnedEnsembleModel

jax.config.update('jax_log_compiles', False)


def make_rgcn_data(num_relations, pos_edge_index, pos_edge_type):
    """
    Construct training data for RGCN models.
    See `RGCNModelTrainingData` for more details.
    """
    complete_pos_edge_index = jnp.concatenate((pos_edge_index, jnp.flip(pos_edge_index, axis=0)), axis=1)
    complete_pos_edge_type = jnp.concatenate((pos_edge_type, pos_edge_type + num_relations))
    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=2 * num_relations,
                                                            edge_index=complete_pos_edge_index,
                                                            edge_type=complete_pos_edge_type)
    all_data = RGCNModelTrainingData(jnp.asarray(dense_relation), jnp.asarray(dense_mask))
    return all_data


def make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes):
    """Setup sampling for training data."""
    dense_batched_negative_sample = make_dense_batched_negative_sample(edge_index=pos_edge_index,
                                                                       num_nodes=num_nodes,
                                                                       num_edges=pos_edge_index.shape[1])

    @jax.jit
    def perform(key):
        neg_edge_index = dense_batched_negative_sample(key=key)
        neg_edge_type = pos_edge_type

        full_edge_index = jnp.concatenate((pos_edge_index, neg_edge_index), axis=-1)
        full_edge_type = jnp.concatenate((pos_edge_type, neg_edge_type), axis=-1)

        pos_mask = jnp.concatenate((jnp.ones_like(pos_edge_type), jnp.zeros_like(neg_edge_type)))

        return BasicModelData(full_edge_index, full_edge_type), pos_mask

    return perform


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, all_data: RGCNModelTrainingData, data: BasicModelData, mask, key):
    """Complete loss function with gradients computed for all parameters of the model."""
    return model.loss(data.edge_index, data.edge_type, mask, all_data, key=key) + model.l2_loss() / (
            2 * data.edge_index.shape[1])


model_configs = {
    'distmult': GenericShallowModel.Config(decoder_class=DistMult, n_channels=200, l2_reg=None, name='DistMult',
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
    'rgcn-basis': RGCNModel.Config(decoder_class=DistMult, hidden_channels=[100],
                                   normalizing_constant='per_node',
                                   edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=0.01, name='RGCN',
                                   epochs=500, learning_rate=0.05, seed=42, n_decomp=2, decomposition_method='basis'),
    'rgcn-block': RGCNModel.Config(decoder_class=DistMult, hidden_channels=[500, 500],
                                   normalizing_constant='per_node',
                                   edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=0.01, name='RGCN',
                                   epochs=500, learning_rate=0.05, seed=42, n_decomp=100, decomposition_method='block'),
    'rgcn-simpl-e': CombinedModel.Config(decoder_class=SimplE, hidden_channels=[400], normalizing_constant='per_node',
                                         edge_dropout_rate=0.5, node_dropout_rate=None, l2_reg=None, name='Combined',
                                         epochs=500, learning_rate=0.01, seed=42, decomposition_method='basis',
                                         n_decomp=2),
    'doublergcn': DoubleRGCNModel.Config(decoder_class=SimplE, hidden_channels=[150], normalizing_constant='per_node',
                                         edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=None, name='DoubleRGCN',
                                         epochs=250, learning_rate=0.05, seed=42, decomposition_method='basis',
                                         n_decomp=2),
    'learnedensemble': LearnedEnsembleModel.Config(decoder_class=DistMult, hidden_channels=[200],
                                                   normalizing_constant='per_node',
                                                   edge_dropout_rate=0.4, node_dropout_rate=None, l2_reg=None,
                                                   name='LearnedEnsemble',
                                                   epochs=600, learning_rate=0.05, seed=42,
                                                   decomposition_method='basis',
                                                   n_channels=200, n_embeddings=1, normalization=False, n_decomp=2),
}


def make_train_step(num_nodes, all_data, optimizer, val_data, get_train_epoch_data_fast, validation):
    @eqx.filter_jit
    def train_step(*, model, opt_state, key):
        """
        Perform one epoch of training, return new model, and optimizer state along with the computed loss and
        validation MRR.
        """
        data_key, model_key = jrandom.split(key)
        train_data, pos_mask = get_train_epoch_data_fast(key=data_key)

        loss, grads = loss_fn(model, all_data, train_data, pos_mask, key=model_key)
        updates, opt_state = optimizer.update(grads, opt_state)

        model = eqx.apply_updates(model, updates)
        if validation:
            _, _, val_mrr_results = generate_unfiltered_mrr(num_nodes=num_nodes, model=model, test_data=val_data,
                                                            all_data=all_data)
            mrr = val_mrr_results.mrr
        else:
            mrr = None
        return model, opt_state, loss, mrr

    return train_step


def train(model_name: Optional[str] = None, dataset_name: Optional[str] = None, validation: bool = True):
    """
    Train a link prediction model on a dataset.
    Args:
        model_name: Name of the model to train. The available models are the keys of the model_configs dictionary.
        dataset_name: Name of the dataset to train on.
        validation: Whether to compute and select using validation data.
    """

    # Setup logging
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('-' * 50)

    if dataset_name is None:
        dataset = LinkPredictionWrapper.load_fb15k()
    else:
        dataset = LinkPredictionWrapper.load_str(dataset_name)

    logging.info(f'Dataset: {dataset.name}')

    if model_name is None:
        model_config = model_configs['distmult']
    else:
        model_config = model_configs[model_name]

    model_init_key, key = jrandom.split(jrandom.PRNGKey(model_config.seed))
    model = model_config.get_model(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations, key=model_init_key)
    optimizer = optax.adam(learning_rate=model_config.learning_rate)  # ComplEx

    logging.info(str(model_config))
    logging.info(str(model))

    val_data = get_data_triples(dataset, dataset.val_idx)

    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]

    # Data used for learning node embeddings transforms for RGCN. This format is different for efficiency reasons.
    all_data = make_rgcn_data(dataset.num_relations, pos_edge_index, pos_edge_type)

    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, dataset.num_nodes)

    train_step = make_train_step(dataset.num_nodes, all_data, optimizer, val_data, get_train_epoch_data_fast,
                                 validation)

    best_model = None
    best_val_mrr = 0.0
    loss = None
    i = None

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    try:
        pbar = trange(model_config.epochs, dynamic_ncols=True)
        for i in pbar:
            train_key, key = jrandom.split(key)
            model, opt_state, loss, val_mrr = train_step(model=model, opt_state=opt_state, key=train_key)

            if validation and val_mrr > best_val_mrr:
                logging.info(f'\nNew best model found at epoch {i} with val_mrr {val_mrr}')
                best_val_mrr = val_mrr
                best_model = model

            pbar.set_description(f'Loss: {loss}, val_mrr: {val_mrr}')
            pbar.refresh()
            if hasattr(model, 'alpha'):
                object.__setattr__(model, 'alpha', jnp.clip(model.alpha, 0, 1))
    except KeyboardInterrupt:  # Allow interrupt
        print(f'Interrupted training at epoch {i}')

    logging.info(f'Final loss: {loss}')

    del val_data, train_step, get_train_epoch_data_fast, opt_state, loss, val_mrr
    gc.collect()

    if validation:
        model = best_model  # Use the model with the best validation MRR

    # Generate MRR results

    model.normalize()

    if hasattr(model, 'alpha'):
        logging.info(f'Alpha: {model.alpha}')

    test_data = get_data_triples(dataset, dataset.test_idx)

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset.num_nodes, model,
                                                                                           test_data, all_data,
                                                                                           force_cpu=True)

    logging.info(f'Unfiltered: {unfiltered_results}')

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, dataset.num_nodes, tail_corrupt_scores,
                                             test_data,
                                             force_cpu=True)

    logging.info(f'Filtered: {filtered_results}')


if __name__ == '__main__':
    train()
