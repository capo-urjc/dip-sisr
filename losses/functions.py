import time
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
import numpy as np


def calculate_gradients(tensor: Tensor, fadf: str) -> tuple:
    assert fadf in ["forward", "central", "backward"], "FADF not valid"

    delta_x = torch.zeros_like(tensor)
    delta_y = torch.zeros_like(tensor)

    if fadf == "central":

        delta_x[..., :, 1: -1] = 0.5 * tensor[..., :, 2:] - 0.5 * tensor[..., :, :-2]
        delta_y[..., 1: -1, :] = 0.5 * tensor[..., 2:, :] - 0.5 * tensor[..., :-2, :]

    elif fadf == "forward":

        delta_x[..., :, :-1] = tensor[..., :, 1:] - tensor[..., :, :-1]
        delta_y[..., :-1, :] = tensor[..., 1:, :] - tensor[..., :-1, :]

    elif fadf == "backward":

        delta_x[..., :, 1:] = tensor[..., :, 1:] - tensor[..., :, :-1]
        delta_y[..., 1:, :] = tensor[..., 1:, :] - tensor[..., :-1, :]

    return delta_x, delta_y


def p_tv_norm_isotropic(tensor: Tensor, p: float, fadf: str) -> Tensor:
    assert fadf in ["forward", "central", "backward"], "FADF not valid"
    assert p > 0, "p value must be greater than 0"

    delta_x, delta_y = calculate_gradients(tensor, fadf)

    # tv_loss = torch.pow((torch.sum(torch.pow(torch.abs(delta_x), p)) +
    #                      torch.sum(torch.pow(torch.abs(delta_y), p))), 1 / p)

    mod_p = torch.pow(delta_x**2 + delta_y**2 + 1e-6, p/2)
    # tv_loss = mod_p.sum(dim=(2, 3)).mean()
    tv_loss = mod_p.mean()

    return tv_loss


def p_tv_norm_anisotropic(tensor: Tensor, p: float, fadf: str) -> Tensor:
    assert fadf in ["forward", "central", "backward"], "FADF not valid"
    assert p > 0, "p value must be greater than 0"

    delta_x, delta_y = calculate_gradients(tensor, fadf)

    mod_p = torch.abs(delta_x) + torch.abs(delta_y) + 1e-6
    tv_loss = mod_p.sum(dim=(2, 3)).mean()

    return tv_loss


def schatten_norm(tensor: Tensor, p: float) -> Tensor:
    """
    With p=1 we have nuclear norm
    """
    assert p > 0, "p value must be greater than 0"

    b, c, w, h = tensor.size()
    omega = c * w * h

    nuclear_norms = torch.norm(tensor, p='nuc', dim=(2, 3))  # returns 1 x 35
    # nuclear_norms = torch.norm(tensor, p='nuc', dim=(1, 3))  # returns 1 x 512
    # nuclear_norms = torch.linalg.svdvals(tensor)
    nuclear_loss = torch.norm(nuclear_norms, p=p, dim=1) / omega

    return nuclear_loss


def SSAHTV_norm(tensor: Tensor, mu: float, fadf: str):
    assert fadf in ["forward", "central", "backward"], "FADF not valid"

    delta_x, delta_y = calculate_gradients(tensor, fadf=fadf)

    b, c, h, w = tensor.size()

    squared_mod = delta_x**2 + delta_y**2
    G = torch.sqrt(squared_mod.sum(1, keepdims=False) + 1e-6)

    tau = (1 / (1 + mu * G))  # (14)

    W = tau  # (15) ??

    ssah_tv = torch.sum(W * G)  # (16)
    ssah_tv = ssah_tv/(c * h * w)

    return ssah_tv


def schiavi_norm(tensor, alpha: float, beta: float, p: float = 1):

    delta_x, delta_y = calculate_gradients(tensor, fadf="central")


    b, c, h, w = tensor.size()
    omega = c * h * w

    squared_mod = delta_x**2 + delta_y**2
    mod = torch.sqrt(squared_mod.sum(1, keepdims=False) + 1e-6)

    # G = torch.pow(torch.sum(G, 1), 1/2)

    # G_ = p_tv_norm_isotropic(tensor, p=p, fadf="central") / omega

    schiavi_term = (alpha * torch.exp(-mod) - beta)  # (14)


    schiavi = torch.sum(schiavi_term * mod)  # (16)

    schiavi = schiavi / (c * h * w)

    return schiavi


if __name__ == "__main__":
    input_tensor = torch.tensor([[[[1.333333, 35.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                  [[2.0, 13.0, 4.0], [5.0, 7.5, 7.0], [8.0, 9.0, 10.0]],
                                  [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                                  [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0]]
                                  ]], dtype=torch.float32)
    input_tensor = torch.randn(1,35,512,512)
    # a = p_tv_norm_isotropic(input_tensor, p=1, fadf="forward")

    # ssahtv = SSAHTV_norm(tensor=input_tensor, p=1, mu=0.4, fadf="forward")
    tic = time.time()
    ssahtv = schatten_norm(tensor=input_tensor, p=1)
    print(time.time()-tic)
    # print(ssahtv)
