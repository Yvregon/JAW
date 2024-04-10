import torch


# GPU usage for model calculation.
def get_device():
    """
    Wrap the pytorch device selection.

    :returns: a torch device with GPU calculation if cuda is available, with CPU calculation else.
    :ret type: torch.device.
    """
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
