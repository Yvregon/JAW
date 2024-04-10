"""
File where to put our custom losses.
"""

import torch.nn as nn

class RelativeL1(nn.Module):
    """
    Comparing to the regular L1, introducing the division by \|c\| + epsilon
    (where epsilon = 0.01, to prevent values of 0 in the denominator).
    """
    def __init__(self, eps=.01, reduction='mean'):
        
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):

        base = target.sum() + self.eps

        return self.criterion(input.sum()/base, target.sum()/base)
    
    
class RelativeL2(nn.Module):
    """
    Idem but for MSE.
    """
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):

        base = target.sum() + self.eps

        return self.criterion(input.sum()/base, target.sum()/base)
    
    
def loss_name_to_class(loss_name : str):
    """
    Build the right loss class in function of the given name.

    :parameter loss_name: Name of the loss used inside the calling context.
    :type loss_name: str.

    :returns: The matching loss class.
    :rtype: nn.Module.
    """
    loss : nn.Module = None

    if(loss_name == "CrossEntropy"):
        loss = nn.CrossEntropyLoss()

    elif(loss_name == "RelativeL1"):
        loss = RelativeL1()

    elif(loss_name == "RelativeL2"):
        loss = RelativeL2()

    else:
        raise NotImplementedError("Unknown loss {}".format(loss_name))
    
    return loss
    