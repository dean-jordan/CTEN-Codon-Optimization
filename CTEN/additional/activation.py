import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self):
        x = torch.maximum(x, 0.0)
        return x
    
class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def forward(self, x, negative_slope=0.1):
        if x > 0:
            return x
        else:
            return x * negative_slope
        
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return 1/(1 + torch.exp(-x))
    
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        return torch.exp(x) / sum(torch.exp(x))