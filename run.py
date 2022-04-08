import click
from rgcn.experiments.link_prediction import train as train_link_prediction, model_configs
from rgcn.experiments.entity_classification import main as train_entity_classification


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model', default='distmult', type=click.Choice(list(model_configs.keys())),
              help='Model to use. Also decides epochs and learning rate.')
@click.option('--dataset', default='wordnet18', type=click.Choice(['wordnet18', 'wordnet18rr', 'fb15k-237', 'fb15k']),
              help='Dataset to use. Note that FB15k is huge.')
@click.option('--validation/--no-validation', default=False, help='Whether to use validation set for model selection.')
def link_prediction(model, dataset, validation):
    train_link_prediction(model, dataset, validation)


# @click.option('--model', default='distmult', help='Model to use. Also decides epochs and learning rate.')
@cli.command()
@click.option('--dataset', default='aifb', type=click.Choice(['aifb', 'mutag', 'bgs', 'am']),
              help='Dataset to use. The model to use is determined by the dataset. Default is aifb.')
@click.option('--seed', default=0, help='Seed for random number generator.')
def entity_classification(dataset, seed):
    train_entity_classification(dataset, seed=seed)


if __name__ == '__main__':
    cli()