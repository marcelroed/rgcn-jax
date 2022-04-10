# RGCN-JAX

Full implementation of Relational Graph Convolutional Networks (R-GCNs) in JAX.

PyTorch and PyTorch Geometric are still required for fetching the datasets, and binaries are needed to install these packages.
Install the requirements with anaconda by using the `environment.yml` file.

```
$ conda env create -f environment.yml
# OR
$ conda env create -f environment.yml -n <custom-name>
```

Now experiments can be ran by using the CLI.
```
$ python run.py --help
Usage: run.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  entity-classification
  link-prediction
```

The commands are `entity-classification` and `link-prediction`. See the help for each command for more information.
```
Usage: run.py entity-classification [OPTIONS]

Options:
  --dataset [aifb|mutag|bgs|am]  Dataset to use. The model to use is
                                 determined by the dataset. Default is aifb.
  --seed INTEGER                 Seed for random number generator.
  --help                         Show this message and exit.
```

```
Usage: run.py link-prediction [OPTIONS]

Options:
  --model [distmult|complex|simple|rescal|transe|rgcn-basis|rgcn-block|rgcn-simpl-e|doublergcn|learnedensemble]
                                  Model to use. Also decides epochs and
                                  learning rate.
  --dataset [wordnet18|wordnet18rr|fb15k-237|fb15k]
                                  Dataset to use. Note that FB15k is huge.
  --validation / --no-validation  Whether to use validation set for model
                                  selection.
  --help                          Show this message and exit.
```


