import torch.nn as nn
from torch.autograd import Function

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024 * 7 * 7, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=200),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=1024 * 7 * 7, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
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
