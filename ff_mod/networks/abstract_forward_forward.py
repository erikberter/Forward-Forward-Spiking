import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class ActivitySaver:
    def __init__(self) -> None:
        self.avg_activity = []
        self.pos_activity = []
        self.neg_activity = []
        self.res_activity = []
        
    def reset_activity(self):
        self.avg_activity = []
        self.pos_activity = []
        self.neg_activity = []
        self.res_activity = []
        
    def add_activity(self, activity, positivity_type = None):
        
        self.avg_activity += [activity]
        
        if positivity_type is not None:
            if positivity_type == "pos":
                self.pos_activity += [activity]
            elif positivity_type == "neg":
                self.neg_activity += [activity]
        else:
            self.res_activity += []

class AbstractForwardForwardNet(ABC, nn.Module):
    """
    Abstract class representing a FeedForward Forward-Forward Neural Network.
    
    Ref:
        - The Forward-Forward Algorithm: Some Preliminary Investigations - G. Hinton (https://arxiv.org/pdf/2212.13345.pdf)
    """
    
    def __init__(self, overlay_function, num_classes, first_prediction_layer = 0, save_activity = False):
        """
        Args:
            overlay_function (_type_): _description_
            num_classes (_type_): Number of classes of the dataset.
            save_activity (bool, optional): True if the activity state of the network should be stored. Defaults to False.
        """
        super().__init__()
        self.overlay_function = overlay_function
        self.num_classes = num_classes
        
        self.first_prediction_layer = first_prediction_layer
        
        self._save_activity = save_activity
        
        self.layers = torch.nn.ModuleList()
        
    
    @abstractmethod
    def adjust_data(self, data):
        """
        Adjust the data based on the network's properties.
        This method needs to be implemented by a subclass.
        """
        
    def train_network(self, x_pos, x_neg, labels = None):
        """
        Train the network based on the positive and negative examples.
        This method needs to be implemented by a subclass.
        """
        
        x_pos = self.adjust_data(x_pos)
        x_neg = self.adjust_data(x_neg)
        
        for i, layer in enumerate(self.layers):
            x_pos, x_neg = layer.train_network(x_pos, x_neg, labels = labels)
    
    @torch.no_grad()
    def predict(self, x):
        """
        Predict the labels of the input data.

        Args:
            x (torch.Tensor): Input data. 
                Shape: [batch_size, input_size] if the network is not spiking, [batch_size, num_steps, input_size] otherwise.

        Returns:
            torch.Tensor: Predicted labels.
        """
        # TODO Assert first_prediction_layer < len(self.layers)
        goodness_scores = []
        
        for label in range(self.num_classes):
            h = self.overlay_function(x, torch.full((x.shape[0],), label, dtype=torch.long))
            h = self.adjust_data(h)
            
            goodness = torch.zeros(h.shape[0], 1).to(x.device)
            
            for j, layer in enumerate(self.layers):
                h = layer(h)
                if j >= self.first_prediction_layer:
                    goodness += layer.get_goodness(h).unsqueeze(1)
            
            goodness_scores += [goodness]
        
        return torch.cat(goodness_scores, 1).argmax(1)
        
    def reset_activity(self):
        #for layer in self.layers:
        #    layer.reset_activity()
        pass

class AbstractForwardForwardLayer(ABC, nn.Linear):
    """
    Abstract class representing a layer in a Feed-Forward Neural Network.
    """
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            bias=False,
            device="cuda:0",
            dtype=None,
            save_activity = False
        ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.save_activity = save_activity
        self.activity_saver = ActivitySaver()
        
        self.loss_function = loss_function
        
    
    def add_activity(self, result, positivity_type = None):
        if not self.save_activity:
            return
        
        self.activity_saver.add_activity(float(result.mean().clone().detach().cpu()), positivity_type)
    
    def reset_activity(self):
        self.activity_saver.reset_activity()
    
    @abstractmethod
    def get_goodness(self, x):
        pass
    
    @abstractmethod
    def forward(self, x, positivity_type = None):
        pass
    
    def train_network(self, x_pos, x_neg, labels = None):
        
        for epoch in range(self.num_epochs):
            
            self.opt.zero_grad()
            
            latent_pos = self.forward(x_pos, positivity_type="pos")
            g_pos = self.get_goodness(latent_pos)
            
            latent_neg = self.forward(x_neg, positivity_type="neg")
            g_neg = self.get_goodness(latent_neg)

            loss = self.loss_function(g_pos, g_neg, latent_pos = latent_pos, latent_neg = latent_neg, labels = labels)
            loss.backward()
            self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()