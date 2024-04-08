import torch


# GPU usage for model calculation.
def get_device():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
