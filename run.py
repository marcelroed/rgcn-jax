import click
from rgcn.experiments.link_prediction import train as train_link_prediction
from rgcn.experiments.entity_classification import main as train_entity_classification


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model', default='distmult', help='Model to use. Also decides epochs and learning rate.')
@click.option('--dataset', default='FB15k', help='Dataset to use.')
def link_prediction(model, dataset):
    train_link_prediction(model, dataset)


# @click.option('--model', default='distmult', help='Model to use. Also decides epochs and learning rate.')
@cli.command()
@click.option('--dataset', default='AIFB', help='Dataset to use. The model to use is determined by the dataset.')
@click.option('--seed', default=0, help='Random seed.')
def entity_classification(dataset, seed):
    train_entity_classification(dataset, seed=seed)


if __name__ == '__main__':
    cli()