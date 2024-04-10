"""
File where to put our custom losses.
"""

import torch.nn as nn
    
class CustomLoss(nn.Module):
    """
    Custom loss class. We have to implement both :func: `__init__` and :func: `forward` methods.
    """
    
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
        pass
    