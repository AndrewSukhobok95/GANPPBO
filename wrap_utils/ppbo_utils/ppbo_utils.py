import numpy as np
from typing import List, Tuple


def get_bounds(D: int,
               min_bound: float,
               max_bound: float) -> Tuple[Tuple[float, float]]:
    bounds = tuple([(min_bound, max_bound)] * D)
    return bounds


def adaptive_init_Xi_iterator(n_comp_in_use: int):
    init_xi_query = np.eye(n_comp_in_use)
    queries = iter(init_xi_query)
    while True:
        for _ in range(n_comp_in_use):
            xi = next(queries)
            yield xi
        queries = iter(init_xi_query)


def adaptive_init_Xi_iterator_def_comp(n_comp_in_use: int, components: list):
    n_def_comp = len(components)
    init_xi_query = np.zeros((n_def_comp, n_comp_in_use))
    for i, c in enumerate(components):
        init_xi_query[i, c] = 1
    queries = iter(init_xi_query)
    while True:
        for _ in range(n_def_comp):
            xi = next(queries)
            yield xi
        queries = iter(init_xi_query)

if __name__=="__main__":

    # aq = adaptive_init_Xi_iterator(3)

    aqd = adaptive_init_Xi_iterator_def_comp(5, [0,2,4])

    print()
