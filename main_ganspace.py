import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os
import sys

PROJECTDIR = os.getcwd()
GAN_DIR = os.path.join(PROJECTDIR, "ganspace")
sys.path.insert(0, GAN_DIR)

from ganspace.config import Config
from ganspace.models import get_instrumented_model
from ganspace.decomposition import get_or_compute



configs = {
    # StyleGAN2 ffhq
    'frizzy_hair':             (31,  2,  6,  20, False),
    'background_blur':         (49,  6,  9,  20, False),
    'bald':                    (21,  2,  5,  20, False),
    'big_smile':               (19,  4,  5,  20, False),
    'caricature_smile':        (26,  3,  8,  13, False),
    'scary_eyes':              (33,  6,  8,  20, False),
    'curly_hair':              (47,  3,  6,  20, False),
    'dark_bg_shiny_hair':      (13,  8,  9,  20, False),
    'dark_hair_and_light_pos': (14,  8,  9,  20, False),
    'dark_hair':               (16,  8,  9,  20, False),
    'disgusted':               (43,  6,  8, -30, False),
    'displeased':              (36,  4,  7,  20, False),
    'eye_openness':            (54,  7,  8,  20, False),
    'eye_wrinkles':            (28,  6,  8,  20, False),
    'eyebrow_thickness':       (37,  8,  9,  20, False),
    'face_roundness':          (37,  0,  5,  20, False),
    'fearful_eyes':            (54,  4, 10,  20, False),
    'hairline':                (21,  4,  5, -20, False),
    'happy_frizzy_hair':       (30,  0,  8,  20, False),
    'happy_elderly_lady':      (27,  4,  7,  20, False),
    'head_angle_up':           (11,  1,  4,  20, False),
    'huge_grin':               (28,  4,  6,  20, False),
    'in_awe':                  (23,  3,  6, -15, False),
    'wide_smile':              (23,  3,  6,  20, False),
    'large_jaw':               (22,  3,  6,  20, False),
    'light_lr':                (15,  8,  9,  10, False),
    'lipstick_and_age':        (34,  6, 11,  20, False),
    'lipstick':                (34, 10, 11,  20, False),
    'mascara_vs_beard':        (41,  6,  9,  20, False),
    'nose_length':             (51,  4,  5, -20, False),
    'elderly_woman':           (34,  6,  7,  20, False),
    'overexposed':             (27,  8, 18,  15, False),
    'screaming':               (35,  3,  7, -15, False),
    'short_face':              (32,  2,  6, -20, False),
    'show_front_teeth':        (59,  4,  5,  40, False),
    'smile':                   (46,  4,  5, -20, False),
    'straight_bowl_cut':       (20,  4,  5, -20, False),
    'sunlight_in_face':        (10,  8,  9,  10, False),
    'trimmed_beard':           (58,  7,  9,  20, False),
    'white_hair':              (57,  7, 10, -24, False),
    'wrinkles':                (20,  6,  7, -18, False),
    'boyishness':              (8,   2,  5,  20, False),
}



outdir = Path('out/')
outdir.mkdir(parents=True, exist_ok=True)

torch.autograd.set_grad_enabled(False)

seed = 1

model_name = 'StyleGAN2'
class_name = 'ffhq'
layer_name = 'style'
use_w = 'StyleGAN' in model_name
device = torch.device('cpu')

modification = "big_smile"
(idx, start, end, strength, invert) = configs[modification]


if __name__=="__main__":
    print("START")

    inst = get_instrumented_model(model_name, class_name, layer_name, device, use_w=use_w)
    model = inst.model

    print("+ Config Processing")

    pc_config = Config(
        components=80,
        n=1_000_000,
        batch_size=200,
        layer=layer_name,
        model=model_name,
        output_class=class_name,
        use_w=use_w)
    
    dump_name = get_or_compute(pc_config, inst)

    with np.load(dump_name) as data:
        lat_comp = data['lat_comp']
        lat_mean = data['lat_mean']
        lat_std = data['lat_stdev']
    
    print("+   path to config:", dump_name)

    print("+ Original Image Generation")

    w = model.sample_latent(1, seed=seed).cpu().numpy()
    w = [w] * model.get_max_latents()
    img = model.sample_np(w)
    Image.fromarray((img*255).astype(np.uint8)).save(outdir / f'exp_{model_name}_{layer_name}_{class_name}.png')

    print("+ Modified Image Generation")
    
    for l in range(start, end):
        w[l] = w[l] + lat_comp[idx] * strength
    
    mod_img = model.sample_np(w)
    Image.fromarray((mod_img * 255).astype(np.uint8)).save(outdir / f'exp_mod_{model_name}_{layer_name}_{class_name}.png')

    print("END")
