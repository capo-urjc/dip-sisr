from losses.functions import p_tv_norm_isotropic, schatten_norm, SSAHTV_norm, schiavi_norm
import numpy as np
import os
import random
import torch

# from quality_measures import sam_metric


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def loss_fn(hr, low_res, dw: list, method: int, tv_param: float = 0, nn_param: float = 0, ssahtv_param: float = 0,
            schiavi_param: float = 0, p_tv: float = 0, p_nn: float = 0, alpha: float = 0, beta: float = 0):

    if method == 0:
        dw = dw[0]
        loss = torch.abs(dw(hr) - low_res).mean()
        # loss = torch.square(dw(hr) - low_res).mean()

    else:
        dw1, dw2 = dw[0], dw[1]
        down_hr_1 = dw1(hr)
        down_hr_2 = dw2(hr)

        if method == 1:
            down_hr = (1/2) * down_hr_1 + (1/2) * down_hr_2

            loss = torch.abs(down_hr - low_res).mean()

        elif method == 2:
            loss = (1/2) * torch.abs(down_hr_1 - low_res).mean() + (1/2) * torch.abs(down_hr_2 - low_res).mean()

    if tv_param != 0:
        loss = loss + tv_param * p_tv_norm_isotropic(hr, p=p_tv, fadf="forward")
    if nn_param != 0:
        loss = loss + nn_param * schatten_norm(hr, p=p_nn)
    if ssahtv_param != 0:
        loss = loss + ssahtv_param * SSAHTV_norm(hr, mu=0.01, fadf="central")
    if schiavi_param != 0 and (alpha != 0 or beta != 0):
        loss = loss + schiavi_param * schiavi_norm(hr, alpha=alpha, beta=beta)
    return loss

