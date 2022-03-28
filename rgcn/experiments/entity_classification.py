import json
import pickle
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from statistics import mean

from rgcn.models.classifier import RGCNClassifier
import optax

from rgcn.data.datasets.entity_classification import EntityClassificationWrapper

# Set jax device to CPU
jax.config.update("jax_platform_name", "cpu")


def make_model(dataset, seed):
    if dataset.name == 'AIFB':
        l2_reg = None
        decomposition_method = 'none'
        hidden_channels = 16
        n_decomp = None
    elif dataset.name == 'MUTAG':
        l2_reg = 5e-4
        decomposition_method = 'basis'
        n_decomp = 30
        hidden_channels = 16
    elif dataset.name == 'BGS':
        l2_reg = 5e-4
        decomposition_method = 'basis'
        n_decomp = 40
        hidden_channels = 16
    elif dataset.name == 'AM':
        l2_reg = 5e-4
        decomposition_method = 'basis'
        n_decomp = 40
        hidden_channels = 10
    else:
        raise ValueError(f'Unknown dataset: {dataset.name}')

    classifier = RGCNClassifier(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations, hidden_channels=hidden_channels,
                                n_classes=dataset.num_classes, decomposition_method=decomposition_method, n_decomp=n_decomp, l2_reg=l2_reg, key=random.PRNGKey(seed))
    optimizer = optax.adam(learning_rate=1e-2)

    return classifier, optimizer


@jax.jit
def softmax_loss(logits, y):
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jnp.zeros_like(log_softmax)
    one_hot = one_hot.at[jnp.arange(y.shape[0]), y].set(1)
    loss = - jnp.sum(one_hot * log_softmax)
    return loss


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, x, edge_type_idcs, edge_masks, y_idx, y):
    logits = model(x, edge_type_idcs, edge_masks)[y_idx]  # (num_nodes, num_classes)
    loss = softmax_loss(logits, y)
    loss = loss + model.l2_loss()
    return loss


@eqx.filter_jit
def test_results(model, x, edge_type_idcs, edge_masks, test_idx, test_y):
    logits = model(x, edge_type_idcs, edge_masks)[test_idx]
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == test_y)
    test_loss = softmax_loss(logits, test_y)
    test_loss = test_loss + model.l2_loss()
    return test_loss, accuracy


@eqx.filter_jit
def train_loop(model, x, edge_type_idx, y_idx, y):
    pass


def train_model(model, optimizer: optax.GradientTransformation, dataset: EntityClassificationWrapper):
    epochs = 50

    x = dataset.node_features
    y_idx = dataset.train_idx
    y = dataset.train_y
    edge_type_idcs = dataset.edge_index_by_type

    opt_state = optimizer.init(model)

    pbar = trange(epochs)
    losses, test_losses, test_accs = [], [], []
    best_model = None
    min_val_loss = float('inf')
    # global loss_fn
    # loss_fn = jax.jit(loss_fn)
    try:
        for _ in pbar:
            loss, grads = loss_fn(model, x, edge_type_idcs, dataset.edge_masks_by_type, y_idx, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            val_loss, val_acc = test_results(model, x, edge_type_idcs, dataset.edge_masks_by_type, dataset.val_idx, dataset.val_y)
            losses.append(loss), test_losses.append(val_loss); test_accs.append(val_acc)
            model = eqx.apply_updates(model, updates)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = model
            pbar.set_description(f"Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    except KeyboardInterrupt:
        print('Training cancelled by KeyboardInterrupt')
        raise KeyboardInterrupt()
        pass

    test_loss, test_acc = test_results(best_model, x, edge_type_idcs, dataset.edge_masks_by_type, dataset.test_idx, dataset.test_y)
    print(test_acc)

    return model, losses, test_losses, test_accs, test_acc


def run_experiment(dataset, seed):
    model, optimizer = make_model(dataset, seed)

    trained_model, losses, val_losses, val_accs, test_acc = train_model(model, optimizer, dataset)

    n = len(losses)
    x = np.arange(n)
    plt.plot(x, losses, label="train_loss")
    plt.plot(x, val_losses, label="test_loss")
    plt.legend()
    plt.savefig(f'{dataset.name.lower()}_{seed}_losses.png')
    plt.close()

    plt.plot(x, val_accs, label="val_acc")
    plt.legend()
    plt.savefig(f'{dataset.name.lower()}_{seed}_accs.png')
    plt.close()

    return {
        'seed': seed,
        'losses': losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
    }


if __name__ == '__main__':
    dataset = EntityClassificationWrapper.load_dataset('AIFB')
    results = []
    try:
        for i in trange(2, desc='Running models with different seeds'):
            result = run_experiment(dataset, seed=i)
            results.append(result)
    except KeyboardInterrupt:
        print('Training cancelled by KeyboardInterrupt')

    results = {k: [results[i][k].tolist() if isinstance(results[i][k], jnp.ndarray) else results[i][k] for i in range(len(results))] for k in results[0].keys()}
    print(f"Mean: {mean(results['test_acc'])}")

    with open(f'{dataset.name.lower()}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
