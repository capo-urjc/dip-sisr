from datasets.NaturalColor import NaturalColor
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.normalize import Normalize
from utils.cast_to_precision import CastToPrecision


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, max]"""

    def __init__(self, max=1.0):
        self.name = "PSNR"
        self.max = max

    def __call__(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(self.max / torch.sqrt(mse))


def get_opt(model: object, opt: str, lr: float, **kwargs: object) -> torch.optim:
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif opt == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, **kwargs)
    elif opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, **kwargs)
    elif opt == "asgd":
        return torch.optim.ASGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=lr, **kwargs)
    elif opt == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer {opt} not supported.")


def get_loss(loss: str, **kwargs: object) -> nn.Module:
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss == "mse":
        return nn.MSELoss(**kwargs)
    elif loss == "l1":
        return nn.L1Loss(**kwargs)
    elif loss == "poisson":
        return nn.PoissonNLLLoss(**kwargs)
    elif loss == "bce":
        return nn.BCELoss(**kwargs)
    elif loss == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss == "margin_ranking":
        return nn.MarginRankingLoss(**kwargs)
    elif loss == "hinge_embedding":
        return nn.HingeEmbeddingLoss(**kwargs)
    elif loss == "multi_margin":
        return nn.MultiMarginLoss(**kwargs)
    elif loss == "smooth_l1":
        return nn.SmoothL1Loss(**kwargs)
    else:
        raise ValueError(f"Loss function {loss} not supported.")


def get_transforms(dataset: str, precision: int) -> tuple:
    assert dataset in ["natural", "color", "Set5", "Set14", "KODAK", "cave"], "The dataset is not valid"
    if dataset in ["natural", "color", "Set5", "Set14", "KODAK"]:
        train_composed = transforms.Compose([Normalize(param=255),
                                             CastToPrecision(precision=precision),
                                             transforms.ToTensor(),
                                             ])

        valid_composed = transforms.Compose([Normalize(param=255),
                                             CastToPrecision(precision=precision),
                                             transforms.ToTensor()
                                             ])

        test_composed = transforms.Compose([Normalize(param=255),
                                            CastToPrecision(precision=precision),
                                            ])

    elif dataset == "cave":
        train_composed = transforms.Compose([Normalize(param=2 ** 16),
                                             CastToPrecision(precision=precision),
                                             transforms.ToTensor(),
                                             ])

        valid_composed = transforms.Compose([Normalize(param=2 ** 16),
                                             CastToPrecision(precision=precision),
                                             transforms.ToTensor()
                                             ])

        test_composed = transforms.Compose([Normalize(param=2 ** 16),
                                            CastToPrecision(precision=precision),
                                            ])

    return train_composed, valid_composed, test_composed


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(128, 1)
    )

    opt = get_opt(model, "adam", 0.01, weight_decay=0.01)
    print(1)

