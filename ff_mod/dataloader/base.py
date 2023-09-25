
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

def _ttemp_flatten(x):
    return torch.flatten(x)

class DataLoaderExtractor:
    
    transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                Lambda(_ttemp_flatten)])
    
    transform_spiky = Compose([
                ToTensor(),
                Lambda(_ttemp_flatten)])
    
    def __init__(self, batch_size = 64):
        self.batch_size = batch_size
        
        self.dataloader = None
        
    @abstractmethod
    def load_dataloader(self, download = False, split = None, is_spiky: bool = False, **kwargs):
        pass
    
    def get_dataloader(self, is_spiky: bool = False,**kwargs) -> DataLoader:
        if self.dataloader is None:
            self.load_dataloader(**kwargs)
        
        return self.dataloader