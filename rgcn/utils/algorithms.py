import numba
import numpy as np
from tqdm import trange

from rgcn.utils import memory


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


@memory.cache
def generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data):
    n_test_edges = test_data.shape[1]
    result_head = np.empty((n_test_edges, num_nodes), dtype=bool)
    result_tail = np.empty((n_test_edges, num_nodes), dtype=bool)
    for i in trange(n_test_edges):
        head = test_data[0, i]
        tail = test_data[1, i]

        mask_head = np.zeros(num_nodes, dtype=bool)
        mask_head[edge_index[0, (edge_type == test_data[2, i]) & (edge_index[1, :] == tail)]] = True
        mask_head[head] = False
        result_head[i, :] = mask_head

        mask_tail = np.zeros(num_nodes, dtype=bool)
        mask_tail[edge_index[1, (edge_type == test_data[2, i]) & (edge_index[0, :] == head)]] = True
        mask_tail[tail] = False
        result_tail[i, :] = mask_tail

    return result_head, result_tail


def test_generate_mrr_filter_mask_numba():
    print()
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_type = np.array([0, 0, 1, 1, 2])
    num_nodes = 5
    test_data = np.array([[0, 1, 0], [1, 2, 0], [0, 3, 1], [3, 4, 1], [2, 4, 2]])
    head_result, tail_result = generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data)
    print(head_result)
    print(tail_result)


def test_generate_mrr2():
    print()
    edge_index = np.array([[0, 1, 2, 0, 3, 3, 4],
                           [3, 4, 0, 1, 1, 2, 3]])
    edge_type = np.array([0, 0, 0, 1, 1, 1, 0])
    test_edge_index = np.array([[0, 2, 3],
                                [3, 0, 1]])
    test_edge_type = np.array([0, 0, 1])
    test_data = np.concatenate((test_edge_index, test_edge_type[None, :]), axis=0)
    num_nodes = 5


    head_mask, tail_mask = generate_mrr_filter_mask(edge_index, edge_type, num_nodes, test_data)

    print(head_mask)
    print(tail_mask)