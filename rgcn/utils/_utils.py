import time
from contextlib import contextmanager
from typing import Union

import jax.numpy as jnp
from joblib import Memory

__all__ = ['memory']

memory = Memory('/tmp/joblib')


@contextmanager
def time_block(name=None):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        if name is not None:
            print(f'{name} took {end_time - start_time:.4f}s')
        else:
            print(f'Took {end_time - start_time:.4f}s')


def _broadcast_dims(args, dims: Union[int, tuple, list]):
    if isinstance(dims, int):
        return (dims,) * len(args)
    elif isinstance(dims, list):
        assert len(dims) == len(args)
        return tuple(dims)
    assert len(dims) == len(args)
    return dims


def batch_function(batch_size, dims=0):
    def wrapper(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            return jnp.concatenate(res, axis=0)
