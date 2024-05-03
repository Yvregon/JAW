"""
File where to put our custom models.
"""

import torch.nn as nn

class CustomModel(nn.Module):
    """
    Custom models class. Like the custom losses we have to implement both :func: `__init__` and :func: `forward` methods.
    """

    def __init__(self, input_size, output_size):
        super(CustomModel, self).__init__()
        pass

    def forward(self, x):
        x = ...
        y = ...
        return y


    