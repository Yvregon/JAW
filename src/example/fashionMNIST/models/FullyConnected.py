import torch.nn as nn

"""
File where to put our custom fully connected models.
"""

def linear_relu(dim_in, dim_out):
    """
    Create a linear reLU layer.

    :parameter dim_in: Input dimension of the layer.
    :type dim_in: int
    :parameter dim_out: Output dimension of the layer.
    :type dim_out: int

    :returns: A linear reLU layer.
    :rtype: nn.Linear.
    """
    return [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


class FullyConnected(nn.Module):
    """
    Definition simple fully connected classification model.
    """

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier = nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnectedRegularized(nn.Module):
    """
    Definition simple fully connected classification model with network regularization.
    """

    def __init__(self, input_size, num_classes, l2_reg):
        super(FullyConnectedRegularized, self).__init__()
        self.l2_reg = l2_reg
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_classes)

    def penalty(self):
        return self.l2_reg * (
            self.lin1.weight.norm(2)
            + self.lin2.weight.norm(2)
            + self.lin3.weight.norm(2)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y
    
    
def build_model(model_name):
    """
    Build the right model in function of the given name.

    :parameter model_name: Name of the model used inside the calling context.
    :type model_name: str.

    :returns: The matching model.
    :rtype: nn.Module.
    """

    model = None

    if(model_name == "FC"):
        model = FullyConnected(1 * 28 * 28, 10)

    elif(model_name == "FCReg"):
        model = FullyConnectedRegularized(1 * 28 * 28, 10, 1e-3)
        
    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
    
    return model
    