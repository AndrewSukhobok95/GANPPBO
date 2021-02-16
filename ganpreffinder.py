import sys
import os
import numpy as np
import torch

PROJECTDIR = os.getcwd()

GAN_DIR = os.path.join(PROJECTDIR, "ganspace")
sys.path.insert(0, GAN_DIR)

from base_modules import GPModel
from base_modules import PPBO_settings
from base_modules import next_query

from wrap_utils.ppbo_utils.feedback_storage import FeedbackStore
from wrap_utils.ppbo_utils.adaptive_query import AdaptiveQuery
from wrap_utils.gan_utils.gan_wrapper import GANSpaceModel
from wrap_utils.ppbo_utils import ppbo_utils


class GANPrefFinder(object):
    def __init__(self,
                 model_name: str = 'StyleGAN2',
                 class_name: str = 'ffhq',
                 layer_name: str = 'style',
                 device: str = "cpu",
                 n_comp: int = 80,
                 n_comp_in_use: int = None,
                 comp_layers_dict: dict = None,
                 adaptive_init: bool = True,
                 adaptive_components: list = None,
                 acquisition_strategy: str = "PCD",   # PCD, EXP, EI, EI_fixed_x
                 ppbo_m: int = 18,                           # number of pseudo comparisons for GP fitting
                 ppbo_user_feedback_grid_size: int = 40,     # grid
                 ppbo_EI_EXR_mc_samples: int = 150,          # number of points for the integrals to solve
                 ppbo_EI_EXR_BO_maxiter: int = 20,           # max number of iterations for BO
                 ppbo_max_iter_fMAP_estimation: int = 5000,
                 ppbo_mu_star_finding_trials: int = 4,
                 gan_sample_seed: int = None,
                 gan_sample_zero_w: bool = False,
                 strength_left_bound: float = -30,
                 strength_right_bound: float = 30,
                 verbose: bool = True):
        """
        :param model_name: GAN model
            Possible models: ProGAN, BigGAN-512, BigGAN-256, BigGAN-128, StyleGAN, StyleGAN2
        :param class_name: Dataset used to train GAN
            Possible datasets: ffhq, car, cat, church, horse, bedrooms, kitchen, places
        :param layer_name: style
        :param device:
            Possible devices: cpu / cuda:0
        :param n_comp: Number of components that PCA uses in GANSpace
        :param n_comp_in_use: Number of components that is used (first n_comp_in_use out of n_comp)
        :param comp_layers_dict: Dictionary with layers to which apply modification when add a particular component
            Structure:
            component: (start layer, end layer)
            {
                0: (5,8),
                2: (13,16),
                ....
            }
        :param adaptive_init: Adaptive initialization flag
        :param adaptive_components: Components to which apply modifications when adaptive_components is True
        :param acquisition_strategy:
            Explored options: PCD, EXP, EI, EI_fixed_x
        :param ppbo_m: Number of pseudo comparisons for GP fitting
        :param ppbo_user_feedback_grid_size: Grid for slider
        :param ppbo_EI_EXR_mc_samples: Number of points for the integrals to solve (mc)
        :param ppbo_EI_EXR_BO_maxiter: Max number of iterations for BO
        :param ppbo_max_iter_fMAP_estimation:
        :param ppbo_mu_star_finding_trials:
        :param gan_sample_seed: Random seed for GAN
        :param gan_sample_zero_w: Use w as a vector of zeros
        :param strength_left_bound: Left bound for strength slider
        :param strength_right_bound: Right bound for strength slider
        :param verbose: Verbose flag
        """

        self.verbose = verbose
        self.verbose_endl = "   "

        self.USING_CUDA = "cuda" in device

        self.ACQUISITION_STRATEGY = acquisition_strategy  # PCD EI
        self.ADAPTIVE_INITIALIZATION = adaptive_init
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION = False
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION = False
        # self.OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER = 1000

        self.comp_layers_dict = comp_layers_dict

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
            theta_initial=[0.01,0.5,0.1],
            user_feedback_grid_size=ppbo_user_feedback_grid_size,
            EI_EXR_mc_samples=ppbo_EI_EXR_mc_samples,
            EI_EXR_BO_maxiter=ppbo_EI_EXR_BO_maxiter,
            mustar_finding_trials=ppbo_mu_star_finding_trials,
            verbose=False)

        self.GP_model = GPModel(self.PPBOsettings)

        self.fs_ses = FeedbackStore(D=self.N_comp_in_use)
        self.AQ = AdaptiveQuery(n_comp_in_use=self.N_comp_in_use,
                                components_to_query=adaptive_components)

        self.x_star_hist = []
        self.mu_star_hist = []

    def get_init_W(self):
        return self.init_W

    def get_init_img(self):
        return self.init_img

    def get_X_star_history(self):
        return self.x_star_hist

    def get_last_X_star_scaled(self):
        x_star = self.x_star_hist[-1]
        x_star_unscaled = self.GP_model.FP.unscale(x_star)
        return x_star_unscaled

    def get_MU_star_history(self):
        return self.mu_star_hist

    def get_next_query(self):
        if self.ADAPTIVE_INITIALIZATION:
            xi = self.AQ.next_Xi()
            x = self.AQ.get_X()
            x[xi != 0] = 0
        else:
            if self.verbose: print("BO query sampling", end=self.verbose_endl)
            nq_unscale = True
            xi, x = next_query(self.PPBOsettings, self.GP_model, unscale=nq_unscale)
        xi = np.abs(xi) / np.max(np.abs(xi))
        return x, xi

    def calculate_pref_vector(self,
                              x: np.array,
                              xi: np.array,
                              alpha: float):
        return x + xi * alpha

    def get_comp_layers_range(self, xi: np.array):
        layers_range = None
        if (self.comp_layers_dict is not None) & (self.ADAPTIVE_INITIALIZATION):
            current_component_num = np.argmax(xi)
            layers_range = self.comp_layers_dict.get(current_component_num)
        return layers_range

    def update_image(self,
                     prefVec: np.array,
                     layers_range: tuple = None):
        w, img = self.GANsm.modify_image_by_prefVec(w=self.init_W,
                                                    prefVec=prefVec,
                                                    layers_range=layers_range)
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

        self.x_star_hist.append(self.GP_model.xstar)
        self.mu_star_hist.append(self.GP_model.mustar)

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

    gpf = GANPrefFinder(class_name="car",
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
    gpf.get_last_X_star_scaled()
