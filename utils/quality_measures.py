import numpy as np
import torch
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def PSNR_manual(img1, img2):
    psnrs: list = []
    for i in range(31):
        mse = torch.mean((img1[:, i, ...] - img2[:, i, ...]) ** 2)
        #psnr_i = 10 * torch.log10(torch.max(img2)**2 / mse)
        psnr_i = 10 * torch.log10(1**2 / mse)
        psnrs.append(psnr_i)
        # mse = ((img1 - img2) ** 2).sum()/(512*512)
    # return 10 * torch.log10(torch.max(img2)**2 / mse)
    return np.mean(psnrs)


def sam_metric(infered, gt):
    assert infered.shape == gt.shape

    # result = np.arccos((infered * gt).sum(1) / (np.linalg.norm(infered, axis=1) * np.linalg.norm(gt, axis=1))).mean()
    inner_product = torch.sum(infered * gt, dim=1)
    norm_infered = torch.linalg.norm(infered, dim=1)
    norm_gt = torch.linalg.norm(gt, dim=1)

    result = torch.arccos(inner_product / (norm_infered * norm_gt)).mean()
    return result


def compute_qlt_measures(infered: torch.Tensor, gt: torch.Tensor):
    mae = torchmetrics.functional.mean_absolute_error(infered, gt)
    mse = torchmetrics.functional.mean_squared_error(infered, gt)

    infered = infered.to("cpu")
    gt = gt.to("cpu")

    ssim = SSIM(data_range=1.0)
    ssim_metric = ssim(infered, gt)

    psnr = PSNR()
    # psnr = PSNR_manual
    psnr_metric = psnr(infered, gt)

    sam = sam_metric(infered, gt)

    return mae.item(), mse.item(), ssim_metric.item(), psnr_metric.item(), sam.item()


if __name__ == "__main__":
    torch.manual_seed(42)

    dimensions = (1, 31, 128, 128)

    tensor_1 = torch.rand(dimensions)
    tensor_2 = torch.rand(dimensions)

    mae, mse, ssim, psnr, sam = compute_qlt_measures(tensor_1, tensor_2)

    print(mae, mse, ssim, psnr, sam)
