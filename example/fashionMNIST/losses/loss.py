# File where to put our custom losses

import torch.nn as nn

class RelativeL1(nn.Module):
    '''
    Comparing to the regular L1, introducing the division by |c| + epsilon
    (where epsilon = 0.01, to prevent values of 0 in the denominator).
    '''
    def __init__(self, eps=.01, reduction='mean'):
        
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):

        base = target + self.eps

        return self.criterion(input/base, target/base)
    
    
class RelativeL2(nn.Module):
    '''
    Idem but for MSE.
    '''
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):

        base = target + self.eps

        return self.criterion(input/base, target/base)
    
    
def loss_name_to_class(loss_name : str):
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
    