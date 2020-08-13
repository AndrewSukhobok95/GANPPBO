import sys
import os
import numpy as np
import torch

PROJECTDIR = os.getcwd()
PPBO_DIR = os.path.join(PROJECTDIR, "PPBO")
GAN_DIR = os.path.join(PROJECTDIR, "ganspace")

sys.path.insert(0, PPBO_DIR)
sys.path.insert(0, GAN_DIR)

from PPBO.gp_model import GPModel
from PPBO.ppbo_settings import PPBO_settings
from PPBO.acquisition import next_query

from wrap_utils.ppbo_utils.feedback_storage import FeedbackStore
from wrap_utils.ppbo_utils.adaptive_query import AdaptiveQuery
from wrap_utils.gan_utils.gan_wrapper import GANSpaceModel
from wrap_utils.ppbo_utils import ppbo_utils


class GANPfinder(object):
    def __init__(self,
                 model_name: str = 'StyleGAN2',
                 class_name: str = 'ffhq',
                 layer_name: str = 'style',
                 device: str = "cpu",
                 n_comp: int = 80,
                 n_comp_in_use: int = None,
                 acquisition_strategy: str = "PCD",        # PCD, EXP, EI
                 adaptive_init: bool = True,
                 ppbo_m: int = 20,                         # number of pseudo comparisons for GP fitting# number of pseudo comparisons for GP fitting
                 ppbo_user_feedback_grid_size: int = 40,   # grid
                 ppbo_EI_EXR_mc_samples: int = 200,        # number of points for the integrals to solve
                 ppbo_EI_EXR_BO_maxiter: int = 30,         # max number of iterations for BO
                 ppbo_max_iter_fMAP_estimation: int = 500,
                 gan_sample_seed: int = None,
                 gan_sample_zero_w: bool = False,
                 strength_left_bound: float = -30,
                 strength_right_bound: float = 30,
                 verbose: bool = True):

        self.verbose = verbose
        self.verbose_endl = "   "

        self.USING_CUDA = "cuda" in device

        self.ACQUISITION_STRATEGY = acquisition_strategy  # PCD EI
        self.ADAPTIVE_INITIALIZATION = adaptive_init
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION = False
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION = False
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER = 1000

        if n_comp_in_use is None:
            self.N_comp_in_use = n_comp
        else:
            self.N_comp_in_use = n_comp_in_use
        comp_range = (0, self.N_comp_in_use)
        self.GANsm = GANSpaceModel(model_name, class_name, layer_name, device, n_comp, comp_range)
        self.init_W, self.init_img = self.GANsm.sample_image(seed=gan_sample_seed,
                                                             zero_w=gan_sample_zero_w)

        self.left_bound = strength_left_bound
        self.right_bound = strength_right_bound

        self.PPBOsettings = PPBO_settings(
            D=self.N_comp_in_use,
            bounds=ppbo_utils.get_bounds(self.N_comp_in_use, self.left_bound, self.right_bound),
            xi_acquisition_function=self.ACQUISITION_STRATEGY,
            m=ppbo_m,
            user_feedback_grid_size=ppbo_user_feedback_grid_size,
            EI_EXR_mc_samples=ppbo_EI_EXR_mc_samples,
            EI_EXR_BO_maxiter=ppbo_EI_EXR_BO_maxiter,
            max_iter_fMAP_estimation=ppbo_max_iter_fMAP_estimation,
            verbose=False)

        self.GP_model = GPModel(self.PPBOsettings)

        self.fs_ses = FeedbackStore(D=self.N_comp_in_use)
        self.AQ = AdaptiveQuery(self.N_comp_in_use)

        self.x_star_hist = []
        self.mu_star_hist = []

    def get_init_W(self):
        return self.init_W

    def get_init_img(self):
        return self.init_img

    def get_X_star_history(self):
        return self.x_star_hist

    def get_last_X_star(self):
        return self.x_star_hist[-1]

    def get_MU_star_history(self):
        return self.mu_star_hist

    def get_next_query(self):
        nq_unscale = False
        if self.ADAPTIVE_INITIALIZATION:
            xi = self.AQ.next_Xi()
            x = self.AQ.get_X()
            x[xi != 0] = 0
        else:
            if self.verbose: print("BO query sampling", end=self.verbose_endl)
            xi, x = next_query(self.PPBOsettings, self.GP_model, unscale=nq_unscale)
        return x, xi

    def calculate_pref_vector(self,
                              x: np.array,
                              xi: np.array,
                              alpha: float):
        return x + xi * alpha

    def update_image(self, prefVec: np.array):
        w, img = self.GANsm.modify_image_by_prefVec(w=self.init_W, prefVec=prefVec)
        return img

    def update_adaptive_query(self,
                              x: np.array,
                              xi: np.array,
                              alpha: float):
        x = self.calculate_pref_vector(x, xi, alpha)
        self.AQ.set_X(x)

    def updateGP(self,
                 x: np.array,
                 xi: np.array,
                 alpha: float):
        self.fs_ses.save_feedback(x, xi, alpha)
        feedback = self.fs_ses.get_np_feedback()

        self.GP_model.update_feedback_processing_object(feedback)
        self.GP_model.update_data()
        self.GP_model.update_model()

        self.x_star_hist.append(self.GP_model.x_star_)
        self.mu_star_hist.append(self.GP_model.mu_star_)

    def optimizeGP(self):
        self.GP_model.optimize_theta()

    def switch_adaptive_initialization(self):
        if self.ADAPTIVE_INITIALIZATION:
            self.ADAPTIVE_INITIALIZATION = False
        else:
            self.ADAPTIVE_INITIALIZATION = True


if __name__ == "__main__":

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda:0"

    print("Device used:", DEVICE)

    gpf = GANPfinder(class_name="car",
                     gan_sample_seed=0,
                     n_comp_in_use=5,
                     strength_left_bound=-5,
                     strength_right_bound=5,
                     ppbo_max_iter_fMAP_estimation=5000,
                     device=DEVICE)

    print("Next Query")
    X, Xi = gpf.get_next_query()

    print("Update Image")
    prefVec = gpf.calculate_pref_vector(X, Xi, alpha=3)
    img = gpf.update_image(prefVec=prefVec)

    print("Update GP")
    gpf.updateGP(X, Xi, alpha=3)

    print("Next Query")
    X, Xi = gpf.get_next_query()

    print("X Star")
    gpf.get_last_X_star()




