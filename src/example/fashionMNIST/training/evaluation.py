import torch
from jaw.utils.progress_bar import progress_bar

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
    
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
            
            progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (tot_loss/(i+1), 100.*correct/N, correct, N))
            
        return tot_loss / N, correct / N