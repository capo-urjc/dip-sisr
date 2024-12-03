import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision


def get_folder_number(directory: str = "logs") -> int:
    return len(os.listdir(directory)) + 1


def get_identifier(args: dict) -> str:
    result: str = ''
    for key, value in args.items():
        result = result + str(value) + '_'

    result = result[:-1]

    return result


def remove_files_in_directory(directory_path) -> None:
    # Get the list of files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through the files and remove only files, not directories
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: {e}")


def save_imgs_tb(gt: torch.Tensor, cleaned: torch.Tensor, noisy: torch.Tensor, mae_hr: float, mse_hr: float,
                 psnr: float, loss: float, it: int, writer: torch.utils.tensorboard.SummaryWriter, w: int, h: int,
                 dataset: str) -> None:

    org_noisy = torchvision.transforms.Resize(size=(h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(noisy)
    list_idxs = [30, 20, 10]

    if dataset == "natural":
        gt_ = gt
        cleaned_ = cleaned
        noisy_ = org_noisy
        dif = 10 * torch.abs(gt_ - cleaned_)

        all_ = torch.concat([noisy_, cleaned_, dif, gt_], dim=-1)

        writer.add_image('LR, HR, DIF, GT', all_[:, 0, ...], global_step=it)
    elif dataset == "color":
        gt_ = gt[0, :, ...]
        cleaned_ = cleaned[0, :, ...]
        noisy_ = org_noisy[0, :, ...]
        dif = 10 * torch.abs(gt_ - cleaned_)

        all_ = torch.concat([noisy_, cleaned_, dif, gt_], dim=-1)
        writer.add_image('LR, HR, DIF, GT', all_, global_step=it)
    else:
        gt_ = gt[0, list_idxs, ...]
        cleaned_ = cleaned[0, list_idxs, ...]
        noisy_ = org_noisy[0, list_idxs, ...]
        dif = 10 * torch.abs(gt_ - cleaned_)

        all_ = torch.concat([noisy_, cleaned_, dif, gt_], dim=-1)
        writer.add_image('LR, HR, DIF, GT', all_, global_step=it)


