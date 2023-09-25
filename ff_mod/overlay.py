from abc import abstractmethod
import torch

from torchvision.transforms import GaussianBlur


class MultiOverlay:
    def __init__(self):
        self.overlays = []
    
    def add_overlay(self, overlay):
        self.overlays.append(overlay)
    
    def __call__(self, data, label, **kwargs):
        for overlay in self.overlays:
            data = overlay(data, label, **kwargs)
        return data

def corner_overlay(data, label = None, **kwargs):
    """ Returns the data with the label as a one hot encoding in the left corner of the image"""
    data_ = data.clone()
    data_[..., :10] *= 0.0
    if len(data_.shape) == 2:
        # Image data
        data_[range(data.shape[0]), label] = data.max()
    else:
        # Spiking data
        data_[range(data.shape[0]),:, label] = data.max()
    return data_

def fussion_overlay(batch, label, steps = 7, **kwargs):
    # TODO Improve in future versions
    batch = batch.reshape(batch.shape[0], 1, 28, 28)
    bitmap = torch.randint_like(batch, 0, 2).long()
    
    gauss = GaussianBlur(kernel_size = (5,5))
    
    
    
    for _ in range(steps):
        bitmap = gauss(bitmap)
    
    permu = torch.randperm(batch.shape[0])
    
    result = batch * bitmap + batch[permu] * (1-bitmap)
    
    result = result.reshape(batch.shape[0], 784)
    return result


# TODO Move the whole overlay to its own folder for better readability
class ClassPatternGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def create_pattern_set(self, pattern_size, classes, num_vectors):
        pass
    
class BinaryPatternGenerator(ClassPatternGenerator):
    def __init__(self):
        super().__init__()
        
    def create_pattern_set(self, pattern_size, classes, num_vectors, p=0.2):
        # TODO Add distance assertions to guarantee that the patterns are not too close to each other
        return torch.bernoulli(p*torch.ones(classes, num_vectors, pattern_size))


class FloatPatternGenerator(ClassPatternGenerator):
    """

    Ideas:
     * Maybe pick random in a sphere
     * Maybe pick evenly spaced in a sphere
     * Maybe pick random with a gaussian distribution
     * Maybe pick with no length restrictions
     * Choose from different disributions of random variables
    """
    def __init__(self):
        super().__init__()
        
    def create_pattern_set(self, pattern_size, classes, num_vectors, scale = 1):
        # TODO Add distance assertions to guarantee that the patterns are not too close to each other
        return scale * torch.abs(torch.rand((classes, num_vectors, pattern_size)))


class AppendToEndOverlay:
    def __init__(
            self,
            pattern_size: int,
            classes: int,
            num_vectors: int = 1,
            p: float = 0.2,
            device: str = "cuda:0"
        ):
        """

        Args:
            pattern_size (int): Size of each pattern
            classes (int): Number of classes
            num_vectors (int): Number of vectors per class. Defaults to 1.
            p (float, optional): Percentaje of ones in the vectors. Defaults to 0.2.
            device (str, optional): Device of the patterns. Defaults to "cuda:0".
        """
        self.pattern_size = pattern_size
        self.classes = classes
        self.num_vectors = num_vectors
        
        self.device = device

        self.label_vectors = torch.bernoulli(p*torch.ones(classes, num_vectors, pattern_size)).to(device)

    def save(self, file_path):
        # Save the label_vectors tensor to a file
        torch.save(self.label_vectors, file_path)
    
    def load(self, file_path):
        # Load the label_vectors tensor from a file
        self.label_vectors = torch.load(file_path,  map_location=torch.device(self.device))
        
        self.classes = self.label_vectors.shape[0]
        self.num_vectors = self.label_vectors.shape[1]
        self.pattern_size = self.label_vectors.shape[2]

    def get_null(self):
        
        def null_overlay(x, y):
            if len(x.shape) == 2:
                return torch.cat([x, torch.zeros(x.shape[0],self.pattern_size).to(x.device)], 1).to(x.device)
            else:
                end_vals = torch.zeros(x.shape[0], x.shape[1],self.pattern_size).to(x.device)
                return torch.cat([x, end_vals], 2).to(x.device)

        return null_overlay
        
    def __call__(self, data, label = None, **kwargs):
        """
        Given a data in dimension (batch, N), a label vector set with dimension (M, P) and a expected label Y,
        return new data with shape (batch, N+M) with the label Y as the vector Y in the label_vectors matrix
        """
        if len(data.shape) == 2:
            # Image Data
            batch_label_vectors = self.label_vectors[label].squeeze(1) # Remove the random pattern dimension
            data_ = torch.cat([data, batch_label_vectors], 1).to(data.device)
        else:
            # Spiking Data
            batch_label_vectors = self.label_vectors[label] # (b,1,p) We will reuse the dimension for the time steps
            batch_label_vectors = batch_label_vectors.repeat(1,data.shape[1],1) # (b, t, p)
            
            data_ = torch.cat([data, batch_label_vectors], 2).to(data.device)
        
        return data_