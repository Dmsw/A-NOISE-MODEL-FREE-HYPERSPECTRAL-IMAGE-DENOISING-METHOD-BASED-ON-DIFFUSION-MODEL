import torch
from guided_diffusion.dist_util import dev
import numpy as np
import cv2 as cv


def measurement_fn(x_t, ref, bar_alpha_t, var, alpha_t):
    alpha_t = torch.tensor(alpha_t).to(dev())
    over_sqrt_bar_alpha_t = torch.tensor(1 / np.sqrt(bar_alpha_t)).to(dev())
    coeff = torch.tensor(bar_alpha_t / (var * bar_alpha_t + 1 - bar_alpha_t)).to(dev())
    return coeff * over_sqrt_bar_alpha_t * (ref - over_sqrt_bar_alpha_t * x_t) * (1 - alpha_t) / torch.sqrt(alpha_t)


