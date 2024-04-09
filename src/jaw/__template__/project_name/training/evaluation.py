import torch


def test(model, loader, f_loss, device):
    """Test a model by iterating over the loader.

    :param model: A torch.nn.Module object.
    :type model: torch.nn.Module.
    :param loader: A torch.utils.data.DataLoader.
    :type loader: torch.utils.data.DataLoader.
    :param f_loss: The loss function, i.e. a loss Module.
    :type f_loss: torch.nn.Module.
    :param device: The device to use for computation.
    :type device: torch.device.
    
    :returns: A tuple with the mean loss and mean accuracy.
    :rtype: tuple of two floats.

    """

    return 0.0, 0.0