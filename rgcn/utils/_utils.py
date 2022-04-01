from contextlib import contextmanager

from joblib import Memory
import time

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