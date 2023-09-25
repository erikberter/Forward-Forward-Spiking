import torch
from torch.optim import Adam

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import corner_overlay

class BaseNet(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function,
            dims,
            loss_function,
            batch_size=32,
            num_classes = 10,
            internal_epoch = 20,
            first_prediction_layer = 0,
            save_activity = False
        ):
        
        super().__init__(
            overlay_function=overlay_function,
            num_classes=num_classes,
            first_prediction_layer = first_prediction_layer,
            save_activity=False)

        self.BATCH_SIZE = batch_size
        
        self.internal_epoch = internal_epoch
        
        self.num_classes = num_classes
        
        for d in range(len(dims) - 1):
            self.layers.append(Layer(dims[d], dims[d + 1], loss_function, internal_epoch=internal_epoch, save_activity=save_activity))

    def adjust_data(self, data):
        return data

class Layer(AbstractForwardForwardLayer):
    def __init__(self, in_features, out_features, loss_function,
                 bias=False, device=None, dtype=None, save_activity = False, internal_epoch = 10):
        super().__init__(in_features, out_features, loss_function = loss_function, bias = bias, device = device, dtype = dtype, save_activity = save_activity)
        
        self.relu = torch.nn.ReLU()
        #self.bn = torch.nn.BatchNorm1d(in_features)
        self.opt = Adam(self.parameters(), lr=0.001)
        self.threshold = 2.0
        self.num_epochs = internal_epoch

    def get_goodness(self, x):
        return x.pow(2).mean(1)
    
    def forward(self, x, positivity_type = None):
        
        if self.bias is not None:
            result =  self.relu(torch.mm(x, self.weight.T) + self.bias.unsqueeze(0))
        else:
            result = self.relu(torch.mm(x, self.weight.T))
            
        self.add_activity(result, positivity_type)
        
        return result
