from torchvision.datasets import CIFAR10
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten

import torch
import torchvision.transforms as transforms


class CIFAR10_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    def get_dataloader(self, download = False, split = None, is_spiky: bool = False, **kwargs):
        test_dataset = CIFAR10(
            './data',
            train=False,
            transform=self.transform,
            download=download
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        
        return test_loader