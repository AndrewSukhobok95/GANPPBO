import numpy as np

from wrap_utils.ppbo_utils import ppbo_utils


class AdaptiveQuery(object):

    def __init__(self, n_comp_in_use):
        self.N_comp_in_use = n_comp_in_use
        self.init_Xi_queries = ppbo_utils.adaptive_init_Xi_iterator(self.N_comp_in_use)
        self.adaptiveX = np.zeros(self.N_comp_in_use)

    def next_Xi(self):
        return next(self.init_Xi_queries)

    def get_X(self):
        return self.adaptiveX

    def set_X(self, x):
        self.adaptiveX = x
