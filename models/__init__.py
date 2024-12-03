from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet

import torch.nn as nn
# cambio n_channels que estaba a 3

import torch
import numpy as np
import random
import os


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
    # print(f"Random seed set as {seed}")

set_seed()

def get_net(input_depth, NET_TYPE, pad, upsample_mode, num_output_channels, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    set_seed()
    if NET_TYPE == 'ResNet':
        # TODO
        net = ResNet(input_depth, 1, 10, 16, 1, nn.BatchNorm2d, False)

    elif NET_TYPE == 'skip':
        net = skip(input_depth, num_output_channels=num_output_channels,
                   num_channels_down=[skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                   num_channels_up=[skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                   num_channels_skip=[skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11,
                   upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False, pad=pad)

    elif NET_TYPE == 'UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=num_output_channels,
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)

    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()

    else:
        assert False

    return net
