import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class LabelPredictor(nn.Module):
    def __init__(self, input_dim=1024*7*7 , num_classes=200): #1024*7*7 , 200
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=1024*7*7 ):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReversalLayer()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 2)
    
    def forward(self, x):
        x = self.grl(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReverseLayerF(Function):
    """ Gradient reversal layer
    
    During forward propagation, this function leaves the input unchanged.
    During backpropagation, it reverses the gradient by multiplying it by a negative scalar.

    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        # use .vew_as() to make a copy of x 
        # otherwise backward is not being called
        
        return x.view_as(x) 

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return ReverseLayerF.apply(x, self.lambda_)