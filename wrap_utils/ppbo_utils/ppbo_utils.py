import numpy as np
from typing import List, Tuple


def get_bounds(D: int,
               min_bound: float,
               max_bound: float) -> Tuple[Tuple[float, float]]:
    bounds = tuple([(min_bound, max_bound)] * D)
    return bounds


def adaptive_init_Xi_iterator(n_comp_in_use):
    init_xi_query = np.eye(n_comp_in_use)
    queries = iter(init_xi_query)
    while True:
        for _ in range(n_comp_in_use):
            xi = next(queries)
            yield xi
        queries = iter(init_xi_query)

if __name__=="__main__":

    AIiter = adaptive_init_Xi_iterator(3)

    print()
