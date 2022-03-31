from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from einops import rearrange
from tqdm import trange

from rgcn.data.datasets.link_prediction import LinkPredictionWrapper
from rgcn.models.link_prediction import GenericShallowModel, TransEModel, ComplExModel, SimplEModel, RGCNModel, \
    RGCNModelTrainingData, BasicModelData
from rgcn.layers.decoder import DistMult
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from rgcn.data.sampling import make_dense_batched_negative_sample, make_dense_batched_negative_sample_dense_rel
from rgcn.evaluation.mrr import generate_unfiltered_mrr, generate_filtered_mrr
from rgcn.data.datasets.entity_classification import make_dense_relation_tensor

import numpy as np
from functools import partial

from joblib import Memory

memory = Memory('/tmp/joblib')

jax.config.update('jax_log_compiles', False)


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
    dense_relation_shape = dense_relation.shape

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
def loss_fn(model, all_data: RGCNModelTrainingData, data: BasicModelData, mask):
    return model.loss(data.edge_index, data.edge_type, mask, all_data) + model.l2_loss() / (
                2 * data.edge_index.shape[1])
    # 50: Filtered: MRRResults(mrr=0.5456010103225708, hits_at_10=0.8876000046730042, hits_at_3=0.708899974822998, hits_at_1=0.34860000014305115)
    # 200, oldsampling: Filtered: MRRResults(mrr=0.7419325709342957, hits_at_10=0.9191000461578369, hits_at_3=0.8622000217437744, hits_at_1=0.6100000143051147)
    # 200: 0.8109014630317688, loss=0.00489
    # 200 (norm): 0.357095, loss=0.15281

    # 50 RESCAL: Filtered: MRRResults(mrr=0.6268602013587952, hits_at_10=0.8105000257492065, hits_at_3=0.6930999755859375, hits_at_1=0.5256999731063843)
    # 50: RESCAL: Filtered: MRRResults(mrr=0.6216883659362793, hits_at_10=0.8100999593734741, hits_at_3=0.6886999607086182, hits_at_1=0.5184999704360962)


model_configs = {
    'distmult': GenericShallowModel.Config(n_channels=100, name='DistMult'),  # 600 epochs
    'complex': ComplExModel.Config(n_channels=150, name='ComplEx'),
    'simple': SimplEModel.Config(n_channels=150, name='SimplE'),
    'transe': TransEModel.Config(n_channels=50, margin=2, name='TransE'),
    'rgcn': RGCNModel.Config(hidden_channels=[100], name='RGCN')
}


def train():
    # config = model_configs[1]
    # print('Using model', config)

    seed = 42
    key = jrandom.PRNGKey(seed)
    dataset = LinkPredictionWrapper.load_wordnet18()
    # same settings for DistMult and RESCAL
    # model = GenericShallowModel(DistMult, model_configs['distmult'], dataset.num_nodes, dataset.num_relations, key)
    # optimizer = optax.adam(learning_rate=0.5)
    # model = ComplExModel(model_configs['complex'], dataset.num_nodes, dataset.num_relations, key)
    # optimizer = optax.adam(learning_rate=0.05)  # ComplEx
    # model = SimplEModel(model_configs['simple'], dataset.num_nodes, dataset.num_relations, key)  # same settings for SimplE and ComplEx
    # optimizer = optax.adam(learning_rate=0.05)  # SimplE
    # model = TransEModel(model_configs['transe'], dataset.num_nodes, dataset.num_relations, key)
    # optimizer = optax.adam(learning_rate=0.01)  # TransE
    # opt_state = optimizer.init(model)

    test_edge_index = dataset.edge_index[:, dataset.test_idx]
    test_edge_type = dataset.edge_type[dataset.test_idx]

    num_epochs = 100

    t = trange(num_epochs)
    pos_edge_index, pos_edge_type = dataset.edge_index[:, dataset.train_idx], dataset.edge_type[dataset.train_idx]
    num_nodes = dataset.num_nodes

    dense_relation, dense_mask = make_dense_relation_tensor(num_relations=dataset.num_relations,
                                                            edge_index=dataset.edge_index, edge_type=dataset.edge_type)
    all_data = RGCNModelTrainingData(jnp.asarray(dense_relation), jnp.asarray(dense_mask))

    model = RGCNModel(model_configs['rgcn'], dataset.num_nodes, dataset.num_relations, key)
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(model)

    i = None
    # if model.data_class.is_dense:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_dense(pos_edge_index, pos_edge_type, num_nodes)
    # else:
    #    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)
    get_train_epoch_data_fast = make_get_epoch_train_data_edge_index(pos_edge_index, pos_edge_type, num_nodes)

    try:
        for i in t:
            use_key, key = jrandom.split(key)
            train_data, pos_mask = get_train_epoch_data_fast(key=use_key)
            # print(train_data)
            # print(pos_mask)
            # print(all_data)
            loss, grads = loss_fn(model, all_data, train_data, pos_mask)
            updates, opt_state = optimizer.update(grads, opt_state)
            # scores = model(train_data)
            # x = scores[train_data.edge_masks].sum()
            # y = scores[~train_data.edge_masks].sum()
            # t.set_description(f'\tLoss: {loss}, Mean Test Score: {mean_test_score}')
            t.set_description(f'\tLoss: {loss}')
            t.refresh()
            model = eqx.apply_updates(model, updates)
    except KeyboardInterrupt:
        print(f'Interrupted training at epoch {i}')

    # Generate MRR results

    model.normalize()
    test_data = jnp.concatenate((test_edge_index,  # (2, n_test_edges)
                                 test_edge_type.reshape(1, -1)), axis=0)  # [3, n_test_edges]

    head_corrupt_scores, tail_corrupt_scores, unfiltered_results = generate_unfiltered_mrr(dataset, model, test_data,
                                                                                           test_edge_index, all_data)

    print('Unfiltered:', unfiltered_results)

    filtered_results = generate_filtered_mrr(dataset, head_corrupt_scores, num_nodes, tail_corrupt_scores, test_data,
                                             test_edge_index)

    print('Filtered:', filtered_results)


if __name__ == '__main__':
    train()
