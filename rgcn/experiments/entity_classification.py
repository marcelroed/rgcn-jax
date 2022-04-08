import logging
import pickle
import sys
import warnings
from pprint import pformat
from statistics import mean, stdev

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import trange

from rgcn.data.datasets.entity_classification import EntityClassificationWrapper
from rgcn.models.classifier import RGCNClassifier

warnings.simplefilter('ignore', FutureWarning)  # Ignore tree_multimap deprecation warnings
warnings.simplefilter('ignore', UserWarning)  # Ignore tree_multimap deprecation warnings


def make_model(dataset, seed):
    """Get the appropriate model for the dataset."""
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

    # Always use the same RGCNClassifier class
    classifier = RGCNClassifier(n_nodes=dataset.num_nodes, n_relations=dataset.num_relations,
                                hidden_channels=hidden_channels,
                                n_classes=dataset.num_classes, decomposition_method=decomposition_method,
                                n_decomp=n_decomp, l2_reg=l2_reg, key=random.PRNGKey(seed))

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
    """Complete loss function returning both loss and gradients with respect to the model parameters."""
    logits = model(x, edge_type_idcs, edge_masks)[y_idx]  # (num_nodes, num_classes)
    loss = softmax_loss(logits, y)
    loss = loss + model.l2_loss()
    return loss


@eqx.filter_jit
def test_results(model, x, edge_type_idcs, edge_masks, test_idx, test_y):
    """
    Get loss and accuracy for a dataset split.
    """
    logits = model(x, edge_type_idcs, edge_masks)[test_idx]
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == test_y)
    test_loss = softmax_loss(logits, test_y)
    test_loss = test_loss + model.l2_loss()
    return test_loss, accuracy


def train_model(model, optimizer: optax.GradientTransformation, dataset: EntityClassificationWrapper):
    """Train for several epochs and get test results for the resulting model."""

    epochs = 50

    x = dataset.node_features
    y_idx = dataset.train_idx
    y = dataset.train_y
    edge_type_idcs = dataset.edge_index_by_type

    # opt_state = optimizer.init(model)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    pbar = trange(epochs)
    losses, val_losses, val_accs = [], [], []
    best_model = None
    min_val_loss = float('inf')
    # global loss_fn
    # loss_fn = jax.jit(loss_fn)
    try:
        for _ in pbar:
            loss, grads = loss_fn(model, x, edge_type_idcs, dataset.edge_masks_by_type, y_idx, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            val_loss, val_acc = test_results(model, x, edge_type_idcs, dataset.edge_masks_by_type, dataset.val_idx,
                                             dataset.val_y)
            losses.append(loss);
            val_losses.append(val_loss);
            val_accs.append(val_acc)
            model = eqx.apply_updates(model, updates)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = model
            pbar.set_description(f"Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    except KeyboardInterrupt:
        print('Training cancelled by KeyboardInterrupt')
        # raise KeyboardInterrupt()

    test_loss, test_acc = test_results(best_model, x, edge_type_idcs, dataset.edge_masks_by_type, dataset.test_idx,
                                       dataset.test_y)
    logging.info('Test loss: %.4f, test accuracy: %.4f', test_loss, test_acc)

    return model, jnp.array(losses), jnp.array(val_losses), jnp.array(val_accs), jnp.array(test_acc)


def run_experiment(dataset, seed):
    """Run training, evaluation and plot results."""
    logging.info('Training seed: %d', seed)
    model, optimizer = make_model(dataset, seed)

    trained_model, losses, val_losses, val_accs, test_acc = train_model(model, optimizer, dataset)

    n = len(losses)
    x = np.arange(n)
    plt.plot(x, losses, label="train_loss")
    plt.plot(x, val_losses, label="val_loss")
    plt.legend()
    plt.savefig(f'{dataset.name.lower()}_{seed}_losses.png')
    plt.show()
    plt.close()

    plt.plot(x, val_accs, label="val_acc")
    plt.legend()
    plt.show()
    plt.savefig(f'{dataset.name.lower()}_{seed}_accs.png')
    plt.close()

    return {
        'seed': seed,
        'losses': losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
    }


def main(dataset_name: str, seed):
    """Entry point for CLI"""
    jax.config.update("jax_platform_name", "cpu")  # Always run on CPU
    logging.basicConfig(filename='entity_classification.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f'Running experiment with dataset: {dataset_name.upper()}, seed: {seed}')
    dataset = EntityClassificationWrapper.load_dataset(dataset_name.upper())
    results = run_experiment(dataset, seed)
    logging.info(pformat(results))
    logging.info(f'Test accuracy is {results["test_acc"]:.4f}')


if __name__ == '__main__':
    # When running as a script we run several instances of the experiment specified below and report aggregates
    jax.config.update("jax_platform_name", "cpu")  # Always run on CPU

    logging.basicConfig(filename='entity_classification.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    dataset = EntityClassificationWrapper.load_dataset('BGS')
    results = []
    try:
        for i in trange(10, desc='Running models with different seeds'):
            result = run_experiment(dataset, seed=i)
            results.append(result)
    except KeyboardInterrupt:
        print('Training cancelled by KeyboardInterrupt')

    results = {k: [results[i][k].tolist() if isinstance(results[i][k], jnp.ndarray) else results[i][k] for i in
                   range(len(results))] for k in results[0].keys()}
    logging.info(f'Mean of test accuracies {mean(results["test_acc"])}')
    logging.info(f'Std of test accuracies {stdev(results["test_acc"])}')
    logging.info(f'Max of test accuracies {max(results["test_acc"])}')
    logging.info(f'Min of test accuracies {min(results["test_acc"])}')
    logging.info(results)

    with open(f'{dataset.name.lower()}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
