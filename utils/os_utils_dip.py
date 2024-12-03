import os
from utils.results_saver import get_name_2_save
import torch
from utils.os_utils import get_folder_number


def save_outputs(folder: str, infered: torch.tensor, gt: torch.tensor, args_dict: dict, dataset: str, identifier: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    name_2_save = get_name_2_save(args_dict)

    if not os.path.exists(folder + dataset + '/' + identifier):
        os.makedirs(folder + dataset + '/' + identifier)

    number_folder: int = get_folder_number(folder + dataset + '/' + identifier)
    folder_2_save: str = folder + dataset + '/' + identifier + "/version_" + str(number_folder)

    os.makedirs(folder_2_save)

    torch.save(infered, folder_2_save + '/' + name_2_save)
    torch.save(gt, folder_2_save + '/gt_' + name_2_save)


def logs_folder_structure(main_folder: str, dataset: str, identifier: str) -> str:

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    if not os.path.exists(main_folder + "/" + dataset):
        os.makedirs(main_folder + "/" + dataset)

    if not os.path.exists(main_folder + "/" + dataset + '/' + identifier):
        os.makedirs(main_folder + "/" + dataset + '/' + identifier)

    number_folder: int = get_folder_number(main_folder + "/" + dataset + '/' + identifier)
    log_dir: str = main_folder + "/" + dataset + '/' + identifier + "/version_" + str(number_folder)
    os.makedirs(log_dir + "/checkpoints/")

    return log_dir
