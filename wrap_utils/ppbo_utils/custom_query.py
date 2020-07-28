import numpy as np

from wrap_utils.ppbo_utils import ppbo_utils

# class CustomQuery(object):
#
#     def __init__(self, n_comp_in_use):
#         self.N_comp_in_use = n_comp_in_use
#         self.init_Xi_queries = ppbo_utils.adaptive_init_Xi_iterator(self.N_comp_in_use)
#         self.adaptiveX = np.zeros(self.N_comp_in_use)
#         self.currentXi = np.zeros(self.N_comp_in_use)
#         self.custom_Xi_strategy = False
#
#     def next_Xi(self):
#         return next(self.init_Xi_queries)
#
#     def switch_Xi_stragtegy(self):
#         if self.custom_Xi_strategy:
#             self.custom_Xi_strategy = False
#         else:
#             self.custom_Xi_strategy = True
#
#     def update_X(self, x):
#         self.adaptiveX = x
#
#     def get_X(self):
#         return self.adaptiveX
#
#     def update_Xi(self, xi):
#         self.currentXi = xi
#
#     def get_Xi(self):
#         return self.currentXi



    # def switch_Xi_strategy(self):
    #     self.AQ.switch_Xi_stragtegy()
    #
    # def Xi_from_user(self, comp_index):
    #     xi = np.zeros(self.AQ.N_comp_in_use)
    #     xi[comp_index] = 1
    #     self.AQ.update_Xi(xi)


