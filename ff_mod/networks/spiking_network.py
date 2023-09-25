from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

import torch
from torch.optim import Adam

import snntorch as snn
from snntorch import surrogate

from ff_mod.overlay import corner_overlay

from tqdm import tqdm

class SpikingNetwork(AbstractForwardForwardNet):
    
    def __init__(
            self,
            overlay_function,
            loss_function,
            dims = [784, 400, 400],
            num_steps = 20,
            batch_size = 32,
            internal_epoch = 20,
            num_classes = 10,
            first_prediction_layer = 0,
            save_activity = False
        ) -> None:
        super().__init__(overlay_function, num_classes, first_prediction_layer, save_activity=save_activity)
              
        self.BATCH_SIZE = batch_size
        
        self.num_steps = num_steps
        self.internal_epoch = internal_epoch
        
        self.num_classes = num_classes
        
        
        for d in range(len(dims) - 1):
            self.layers.append(Layer(dims[d], dims[d + 1], internal_epoch = internal_epoch, num_steps = num_steps, loss_function = loss_function, save_activity=save_activity))
        
    
    def adjust_data(self, data):
        """
            Returns the data in the correct shape based on the Net properties.
            
            E.g. Spiking Neural Networks may need to adjust the data for temporal information.
        """
        
        data = torch.reshape(data, [data.shape[0],1,data.shape[1]])
        
        return data.repeat(1, self.num_steps, 1)
            
        
        

class Layer(AbstractForwardForwardLayer):
    def __init__(self, in_features, out_features, loss_function,
                bias=False, device=None, dtype=None, internal_epoch = 10, num_steps = 20, save_activity = False):
        super().__init__(in_features, out_features, loss_function, bias, device, dtype, save_activity=save_activity)
        
        if in_features == 784:
            beta_n = 0.4
            threshold = 0.4
        else:
            beta_n = 0.3
            threshold = 0.2
        
        # Tiempo de decay de la actividad
        # A ver si da tiempo a que se descarge por completo en tiempo de simulacion
        self.relu = snn.Leaky(beta=beta_n, threshold = threshold, init_hidden=True, spike_grad=surrogate.fast_sigmoid())
        self.opt = Adam(self.parameters(), lr=0.0001)
        #self.bn = torch.nn.BatchNorm1d(in_features)
        self.threshold = 1.0
        self.num_epochs = internal_epoch
        self.num_steps = num_steps

    def get_goodness(self, x):
        #return (1/(1-x.mean(1).pow(2)+0.033) - 1).mean(1)
        #return -torch.log(1-x.mean(1) + 0.033).mean(1)
        return x.mean(1).mean(1)

    def forward(self, x, positivity_type = None):
        # X should have a shape of [batch, num_steps, dim]
        # x_direction = x / (x.norm(2, 2, keepdim=True) + 1e-4)
        # I think normalization like this won't work
        result = []
        
        self.relu.reset_hidden()
        
        for t in range(self.num_steps):
            if self.bias is not None:
                result += [self.relu(
                    torch.mm(x[:,t,:], self.weight.T) +
                    self.bias.unsqueeze(0)).reshape([x.shape[0], 1, self.weight.T.shape[1]])]
            else:
                result += [self.relu(
                    torch.mm(x[:,t,:], self.weight.T)).reshape([x.shape[0], 1, self.weight.T.shape[1]])]
        
        result = torch.cat(result, dim=1)
        
        self.add_activity(result, positivity_type)
        
        return result
    
    
    
    
    