from torchvision.datasets import MNIST

from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten

import torch
import torchvision.transforms as transforms


class MNIST_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64):
        super().__init__(batch_size=batch_size)
        
        resize = transforms.Resize((28, 28))
        self.transform = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_ttemp_flatten)
        ])
        
        self.transform_spiky = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Lambda(_ttemp_flatten)
        ])
    
    
    def load_dataloader(self, download = False, split = None, subset = None, is_spiky = True, **kwargs):
        dataset = MNIST(
            './data',
            transform=self.transform,
            train=(split == "train"),
            download=download
        )
        
        if subset is not None:
            idx = torch.tensor([label in subset for label in dataset.targets])

            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]
            
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        