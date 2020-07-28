import pandas as pd
import numpy as np


class FeedbackStore(object):
    def __init__(self, D):
        self.D = D
        self.columns = ['alpha_xi_x' + str(i) for i in range(1,self.D+1)] + \
                       ['xi' + str(i) for i in range(1,self.D+1)] + \
                       ['alpha_star']
        self.df_feedback = pd.DataFrame(columns=(self.columns), dtype=np.float64)
    
    def save_feedback(self,
                      x: np.array,
                      xi: np.array,
                      alpha: float):
        res = pd.DataFrame(columns=(self.columns), dtype=np.float64)
        alpha_xi_x = self.compute_alpha_xi_x(x, xi, alpha)
        alpha_star = np.nanmin(alpha_xi_x[x==0] / xi[x==0])
        new_row = list(alpha_xi_x) + list(xi) + [alpha_star]
        res.loc[0,:] = new_row
        self.df_feedback = self.df_feedback.append(res, ignore_index=True)

    def compute_alpha_xi_x(self,
                           x: np.array,
                           xi: np.array,
                           alpha: float):
        return alpha * xi + x
    
    def get_feedback(self):
        return self.df_feedback
    
    def get_np_feedback(self):
        return np.array(self.df_feedback)



#    alpha_xi_x1  alpha_xi_x2  alpha_xi_x3  alpha_xi_x4  alpha_xi_x5  alpha_xi_x6  xi1  xi2  xi3  xi4  xi5  xi6  alpha_star
# 0    -0.207148         -0.5          5.0        -84.4        142.8          2.7  1.0  0.0  0.0  0.0  0.0  0.0   -0.207148