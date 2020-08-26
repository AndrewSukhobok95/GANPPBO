import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os
import sys
from typing import List, Tuple

if __name__ == "__main__":
    PROJECTDIR = os.path.normpath(os.path.join(os.getcwd(), "../../"))
    GAN_DIR = os.path.join(PROJECTDIR, "ganspace")
    sys.path.insert(0, GAN_DIR)
    sys.path.insert(0, PROJECTDIR)

from ganspace.config import Config
from ganspace.models import get_instrumented_model
from ganspace.decomposition import get_or_compute


class GANSpaceModel(object):
    def __init__(self,
                 model_name: str = 'StyleGAN2',
                 class_name: str = 'car',
                 layer_name: str = 'style',
                 device: str = "cpu",
                 n_comp: int = 80,
                 comp_range: Tuple[int, int] = None):

        torch.autograd.set_grad_enabled(False)

        self.model_name = model_name
        self.class_name = class_name
        self.layer_name = layer_name
        self.use_w = 'StyleGAN' in self.model_name
        self.device = torch.device(device)

        self.inst = get_instrumented_model(self.model_name,
                                           self.class_name,
                                           self.layer_name,
                                           self.device,
                                           use_w=self.use_w)
        self.gan_model = self.inst.model

        self.n_comp = n_comp
        self.n_layers = self.gan_model.get_max_latents()

        pc_config = Config(
            components=self.n_comp,
            n=1_000_000,
            batch_size=200,
            layer=self.layer_name,
            model=self.model_name,
            output_class=self.class_name,
            use_w=self.use_w)
        dump_name = get_or_compute(pc_config, self.inst)
        with np.load(dump_name) as data:
            self.lat_comp = data['lat_comp']
            self.lat_mean = data['lat_mean']
            self.lat_std = data['lat_stdev']

        if comp_range is None:
            comp_range = (0, self.n_comp)
        else:
            if comp_range[0] < 0:
                raise ValueError("The number of components should be positive number.")
            if comp_range[1] > self.n_comp:
                raise ValueError("You can't choose more components then", n_comp)

        lcb, rcb = comp_range[0], comp_range[1]
        self.V = self.lat_comp[lcb:rcb, 0, :]

    def sample_image(self,
                     seed: int = None,
                     zero_w: bool = False):
        if zero_w:
            w = np.zeros((1, 512), dtype=np.dtype('float32'))
        else:
            w = self.gan_model.sample_latent(1, seed=seed).cpu().numpy()
        w_list = [w] * self.n_layers
        img = self.gan_model.sample_np(w_list)
        return w, img

    def modify_image(self,
                     w: np.array,
                     idx: int,
                     strength: int,
                     start: int = None,
                     end: int = None):
        if (start is None) | (end is None):
            start = 0
            end = self.n_layers - 1
        w_list = [w] * self.n_layers
        for l in range(start, end):
            w_list[l] = w_list[l] + self.V[idx] * strength
        img = self.gan_model.sample_np(w_list)
        return w, img

    def _weighted_latent_modification(self, prefVec: np.array):
        lat_modification = np.matmul(prefVec, self.V, dtype=np.dtype("float32"))
        return lat_modification

    def modify_image_by_prefVec(self,
                                w: np.array,
                                prefVec: np.array,
                                layers_range: tuple = None):
        lat_modification = self._weighted_latent_modification(prefVec)
        if layers_range is None:
            w_list = [w + lat_modification] * self.n_layers
        else:
            w_list = [w] * self.n_layers
            for l in range(layers_range[0], layers_range[1]):
                print("APPLY LAYER", l)
                w_list[l] = w_list[l] + lat_modification
        img = self.gan_model.sample_np(w_list)
        return w, img


if __name__ == "__main__":

    g = GANSpaceModel()

    g.modify_image_by_prefVec(np.array([]), np.array([]))
