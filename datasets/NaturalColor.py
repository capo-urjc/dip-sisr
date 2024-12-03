import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


def find_all_files_in_folder_cave(directory: str) -> list:
    files: list = []

    for file in os.listdir(directory):
        files.append(directory + file)

    return files


class NaturalColor(Dataset):
    """NaturalColor images dataset."""

    def __init__(self, root_dir: str = "../DATA/NaturalColor/", transform=None, sigma=0):

        self.root_dir = root_dir
        self.transform = transform
        self.sigma = sigma

        images_paths = find_all_files_in_folder_cave(self.root_dir)

        self.images_paths = images_paths


    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images_paths[idx]
        sample: np.ndarray = np.array(Image.open(image_path))

        if self.transform:
            sample = self.transform(sample)

        return {'x': sample + self.sigma*torch.randn_like(sample), 'y': sample}


if __name__ == "__main__":

    print(1)
    fds = NaturalColor()
    elem = fds[0]
    # print(1)



