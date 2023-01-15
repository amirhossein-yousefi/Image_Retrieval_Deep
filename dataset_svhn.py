import numpy as np
from torch.utils.data import Dataset
from utils import load_data
import torchvision
import torch
import torchvision.transforms as transforms


class SVHNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 root_dir,
                 split,
                 transform=None):
        torchvision.datasets.SVHN('./data', 'train', download=True)
        torchvision.datasets.SVHN('./data', 'test', download=True)
        self.images, self.labels = load_data(root_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, target = self.images[:, :, :, index], int(self.labels[index])
        if self.transform:
            img = self.transform(img.astype(np.uint8))
        return img, target - 1


def transform(MEAN: torch.Tensor = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
              STD: torch.Tensor = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)):
    trans = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN.tolist(), STD.tolist()),
         ]
    )
    return trans
    # trans = transforms.Compose(
    #     [transforms.ToPILImage(),
    #      transforms.ToTensor(),
    #      ]
    # )
