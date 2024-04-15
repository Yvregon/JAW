import torch


def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    :param model: A torch.nn.Module object.
    :type model: torch.nn.Module.
    :param loader: A torch.utils.data.DataLoader.
    :type loader: torch.utils.data.DataLoader.
    :param f_loss: The loss function, i.e. a loss Module.
    :type f_loss: torch.nn.Module.
    :param optimizer: Optimisation algorithm used for the gradient retropropagation.
    :type optimizer: torch.optim.
    :param device: The device to use for computation.
    :type device: torch.device.
    
    :returns: None.
    """

    model.train()

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Other step here :

        optimizer.step()
