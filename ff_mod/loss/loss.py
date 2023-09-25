from typing import Any, Optional
import torch

class Base0Loss:
    def __init__(self, threshold = 0.5, alpha = 1.0, beta: Optional[float] = None):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        
        if beta is None:
            self.beta = alpha
    
    def __call__(self, g_pos, g_neg, **kwargs):
        loss = torch.log(1 + torch.exp(torch.cat([
                    self.alpha*(-g_pos + self.threshold),
                    self.beta*(g_neg - self.threshold)]))).mean()
        
        #print(f" G_pos {g_pos.mean()} | G_neg {g_neg.mean()} | Loss {loss}")
        
        return loss

class Base1Loss:
    def __init__(self, threshold: float = 0.5, alpha = 1.0, beta: Optional[float] = None):
        super().__init__()
        
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        
        if beta is None:
            self.beta = alpha
        
    def __call__(self, g_pos, g_neg, **kwargs):
        loss_pos = torch.log(1 + torch.exp(self.alpha * (-g_pos.mean() + self.threshold)))
        loss_neg = torch.log(1 + torch.exp(self.beta * (g_neg.mean() - self.threshold)))
        return loss_pos+loss_neg

class Base2Loss:
    def __init__(self, threshold: float = 0.5, alpha : float = 1.0, beta: Optional[float] = None):
        super().__init__()
        
        self.threshold = threshold
        self.alpha = alpha
        if beta is None:
            self.beta = alpha
        
    def __call__(self, g_pos, g_neg, **kwargs):
        loss_pos = torch.log(1 + torch.exp(self.alpha * (-g_pos + self.threshold)).mean())
        loss_neg = torch.log(1 + torch.exp(self.beta * (g_neg - self.threshold)).mean())
        return loss_pos+loss_neg

class SymbaLoss:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, g_pos, g_neg, **kwargs):
        loss = torch.log(1+torch.exp(self.alpha * (g_neg.mean() - g_pos.mean())))
        #print(f"G_neg {g_neg.mean()} | G_pos {g_pos.mean()} | Loss {loss}")
        return loss
    
class SymVarLoss:
    def __init__(self, alpha: float = 1.0, beta : float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, g_pos, g_neg, **kwargs):
        return torch.log(1+torch.exp(self.alpha * (g_neg.mean() - g_pos.mean()+self.beta * g_pos.var())))
    
class SymBCELoss:
    def __init__(self) -> None:
        pass
    
    def __call__(self, g_pos, g_neg, **kwargs):
        #print(g_pos.shape, g_neg.shape)
        return torch.nn.BCELoss()(torch.cat([g_pos, g_neg]), torch.cat([torch.ones_like(g_pos), torch.zeros_like(g_neg)]))

class ContrativeFF:
    
    def __init__(self, threshold = 0.5, alpha = 1.0, beta : Optional[float] = None, ratio = 0.1) -> None:
        self.base_loss = Base0Loss(threshold = threshold, alpha = alpha, beta = beta)
        
        self.ratio = ratio
    
    def __call__(self, g_pos, g_neg, latent_pos = None, latent_neg = None, labels = None, **kwargs):
        
        if len(latent_pos.shape) == 3:
            # SNN Check
            # TODO Implement Loss for SNN
            latent_pos = latent_pos.mean(dim=1)
        
        # Sum of pairwise distance between different classes
        dists = torch.cdist(latent_pos, latent_pos, p=2)/latent_pos.shape[1]

        labels = labels.view(-1, 1)
        mask = labels != labels.t()

        dists_masked = dists * mask.float()
        
        dists_masked = torch.log(1 + torch.exp(-dists_masked))
        dists_masked *= mask.float() # Remove the same label pairs
        
        loss1 = dists_masked.mean()
        
        loss2 = self.base_loss(g_pos, g_neg)
        
        return self.ratio * loss1 + loss2