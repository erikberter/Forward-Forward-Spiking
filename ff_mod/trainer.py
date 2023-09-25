
import torch
from tqdm import tqdm

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet


from ff_mod.callbacks.callback import CallbackList, Callback

from ff_mod.overlay import corner_overlay

from ff_mod.dataloader.factory import DataloaderFactory

class Trainer:
    
    def __init__(self, unsupervised = False, device = "cuda:0") -> None:
        
        self.unsupervised = unsupervised
        
        self.device = device
        # TODO Assert the whole code is prepared for assert_cuda
        
        self.overlay_function = None
        self.__net = None
        
        self.callbacks = CallbackList()
        
        self.train_loader = None
        self.test_loader = None
    
    def set_network(self, net) -> None:
        self.__net = net
        self.overlay_function = net.overlay_function

    def get_network(self) -> AbstractForwardForwardNet:
        return self.__net

    def add_callback(self, callback : Callback):
        self.callbacks.add(callback)
    
    def load_data_loaders(self, in_data = "mnist", batch_size: int = 512, test_batch_size: int = 512, is_spiky: bool = False):
        
        dataloader = DataloaderFactory.get_instance().get(in_data)
        
        self.train_loader = dataloader.get_dataloader(split = "train", batch_size = batch_size, is_spiky = is_spiky)
        self.test_loader = dataloader.get_dataloader(split = "test", batch_size = test_batch_size, is_spiky = is_spiky)
        
        self.num_classes = len(self.train_loader.dataset.classes) 
        
        print(self.num_classes)

    def train_epoch(self, break_step : int = -1, verbose: int = 1) -> float:
        if verbose > 0:
            print("Train epoch")
            
        self.__net.train()
        """ Returns the mean accuracy """
        for step, (x,y) in  tqdm(enumerate(iter(self.train_loader)), total = len(self.train_loader), leave=True, disable=verbose<2):
            
            if break_step > 0 and step > break_step:
                break
            
            self.callbacks.on_train_batch_start()

            x, y = x.to(self.device), y.to(self.device)
            
            # Prepare positive data
            if not self.unsupervised:
                x_pos = self.overlay_function(x, label = y)
            else:
                x_pos = x.clone().detach()

            # Prepare negative data
            rnd_extra = torch.randint(1, self.num_classes, size = (y.size(0),), device = self.device)
            y_rnd = (y+rnd_extra)%self.num_classes
            
            x_neg = self.overlay_function(x, label = y_rnd)
            
            # Train the network
            self.__net.train_network(x_pos, x_neg, labels = y)
            
            # Get the predictions
            predicts = self.__net.predict(x)
            
            
            self.callbacks.on_train_batch_end(predictions = predicts.cpu(), y = y.cpu())

    def test_epoch(self, verbose: int = 1):
        if verbose > 1: print("Test epoch")
        
        accuracy = 0.0
        
        self.__net.eval()
        for step, (x,y) in enumerate(iter(self.test_loader)):

            self.callbacks.on_test_batch_start()
            
            x, y = x.to(self.device), y.to(self.device)
            
            predicts = self.__net.predict(x)
            
            accuracy += predicts.eq(y).float().sum().item()
            
            self.callbacks.on_test_batch_end(predictions = predicts.cpu(), y = y.cpu())

        torch.cuda.empty_cache()
        
        return accuracy / len(self.test_loader.dataset)

    def train(self, epochs: int = 2, verbose: int = 1):
        for epoch in range(epochs):
            if verbose > 0: print(f"Epoch {epoch}")
                
            self.train_epoch(verbose=verbose)
            self.callbacks.on_train_epoch_end()
            
            self.test_epoch(verbose=verbose)
            self.callbacks.on_test_epoch_end()
            
            
            self.callbacks.next_epoch()
            self.__net.reset_activity()
            
            torch.cuda.empty_cache()