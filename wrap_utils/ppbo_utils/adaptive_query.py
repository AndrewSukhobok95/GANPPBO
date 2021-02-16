import numpy as np

from wrap_utils.ppbo_utils import ppbo_utils


class AdaptiveQuery(object):

    def __init__(self,
                 n_comp_in_use: int,
                 components_to_query: list = None):
        self.N_comp_in_use = n_comp_in_use
        if components_to_query is None:
            self.init_Xi_queries = ppbo_utils.adaptive_init_Xi_iterator(self.N_comp_in_use)
        else:
            _components_to_query = components_to_query.copy()
            for _c in components_to_query:
                if (_c >= self.N_comp_in_use) or (_c < 0):
                    _components_to_query.remove(_c)
            self.init_Xi_queries = ppbo_utils.adaptive_init_Xi_iterator_def_comp(self.N_comp_in_use, _components_to_query)
        self.adaptiveX = np.zeros(self.N_comp_in_use)

    def next_Xi(self):
        return next(self.init_Xi_queries)

    def get_X(self):
        return self.adaptiveX

    def set_X(self, x):
        self.adaptiveX = x


if __name__=="__main__":

    aq = AdaptiveQuery(5)

    print("done")


