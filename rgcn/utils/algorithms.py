import numba
import numpy as np


@numba.njit(parallel=True)
def parallel_argsort_last(arr):
    # Run argsort in parallel for each row on CPU
    n = arr.shape[0]
    argsort_arr = np.empty_like(arr, dtype=numba.int64)
    for i in numba.prange(n):
        argsort_arr[i] = np.argsort(arr[i])
    return argsort_arr


if __name__ == '__main__':
    arr = np.random.rand(5000, 40_000)
    parallel_argsort_last(arr)
    print('Started')
    argsort_arr = parallel_argsort_last(arr)
    print(argsort_arr)


def test_parallel_argsort():
    print()
    arr = np.random.rand(5000, 40_000)
    argsort_arr = parallel_argsort_last(arr)
    standard_argsort_arr = np.argsort(arr, axis=1)

    assert np.all(argsort_arr == standard_argsort_arr)