import os
import pickle
from typing import *
import torch

from torchvision.datasets.cifar import CIFAR100


class CIFAR100(CIFAR100):
    MAP = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.extra_targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.extra_targets.extend(entry['coarse_labels'])
    def map(self, fine_output:torch.Tensor) -> torch.Tensor:
        mapper = torch.tensor(self.MAP, device=fine_output.device)
        return mapper[fine_output]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        extra_target = self.extra_targets[index]
        if self.target_transform is not None:
            extra_target = self.target_transform(extra_target)
        return *super().__getitem__(index), extra_target