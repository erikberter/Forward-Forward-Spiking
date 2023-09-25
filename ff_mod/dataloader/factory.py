from ff_mod.dataloader.not_mnist import NotMNIST
from ff_mod.dataloader.omniglot import Omniglot_Dataloader
from ff_mod.dataloader.KMNIST import KMNIST_Dataloader
from ff_mod.dataloader.EMNIST import EMNIST_Dataloader
from ff_mod.dataloader.FashionMNIST import FashionMNIST_Dataloader
from ff_mod.dataloader.CIFAR import CIFAR10_Dataloader
from ff_mod.dataloader.MNIST import MNIST_Dataloader

class DataloaderFactory:
    # TODO Refactor this name
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, dataset : str = None, batch_size : int = 512):
        
        loader = None
        
        if dataset is None:
            raise ValueError("dataset is None")
        

        if dataset == "kmnist":
            return KMNIST_Dataloader(batch_size=batch_size)
        elif dataset == "omniglot":
            return Omniglot_Dataloader(batch_size=batch_size)
        elif dataset == "not_mnist":
            return NotMNIST(batch_size=batch_size)
        elif dataset == "fashion_mnist":
            return FashionMNIST_Dataloader(batch_size=batch_size)
        elif dataset == "emnist":
            return EMNIST_Dataloader(batch_size=batch_size)
        elif dataset == "mnist":
            return MNIST_Dataloader(batch_size=batch_size)
        elif dataset == "cifar10":
            return CIFAR10_Dataloader(batch_size=batch_size)
        
        raise ValueError("The dataloader is not valid.")
    
    def get_valid_dataloaders(self):
        # TODO Add to enum
        return ["kmnist", "omniglot", "not_mnist", "fashion_mnist", "emnist", "mnist"]
    
    def get_mix_dataloader(self, datasets : list[str], batch_size : int = 512):
        raise NotImplementedError("Mix dataloader is not implemented yet.")