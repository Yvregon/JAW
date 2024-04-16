Training functions
==================

Now it's time to write the method that will be called inside every loop turn. We need two of them : one for train the model and another for validate is accuracy, 
then test it.

.. important::

    It's possible to use two different function for evaluate the model during validation and test phase, but isn't recommended since use the same evaluation method 
    allow us to compute a confidence for our network.

Train
-----

All we have to do here is to put our sample (now turned into tensor, remember `here <data_preprocessing.html>`_) inside a device to be calculated, compute a loss and 
retropropagate the gradient through our network. In Pytorch we can do that easily with one loop and few instructions. Here we write this function inside 
``training/train.py``.

.. code-block:: python

    def train(model, loader, f_loss, optimizer, device):

        # Switch the model to "train mode"
        model.train()

        for (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Wait ... where is my model?
***************************

If you go further in this tutorial, you will notices that we never give at the loss object a reference to the model or its parameters. Moreover we use two unexplained 
method calls : ``model.train()`` and ``optimizer.zero_grad()``. For have a good understanding of all of this, we must talk about **Autograd**.

.. note::

    This part is important only if you keep using Pytorch in your project and only concern an implementation problem. 

Autograd
********

Autograd is a Pytorch module of automatic differentiation, which allow us to compute tensor's gradient. At the loading of the torch core modules, it create a graph 
of functions called ``grad_fn``, which represent the data of our tensor (inside our device) in the form of an acyclic graph where the inputs are the leaves and the 
outputs the roots. Everytime a tensor with the flag ``requires_grad`` (true by default), activated is submitted to an operation, it's updated inside the graph.
That is why we have to put our tensor to devices before computation. All our ``nn.Module`` object have a reference to this graph, so when we use 
``optimizer.zero_grad()``, all the previously computed gradients are sets to 0, here in order to not accumulate the gradient through our iterations. Same for the loss 
object, when call ``backward()`` method, all computed gradient is retropropagated through the previously used tensors. You can take a look at this good 
`example <https://amsword.medium.com/understanding-pytorchs-autograd-with-grad-fn-and-next-functions-b2c4836daa00>`_ if you want to know more about Autograd and 
``graph_fn``.

.. note::

    ``model.train()`` give as instructions to the model to consider is special layers, such as *batchNorm* or *Dropout* layers, useful for the training but not for 
    inference. This mode is activated by default, but later we will deactivate it inside our evaluation loop, so we must ensure that the train mode is switched on 
    before the train loop.

Eval
----

For the evaluation function, we will keep a similar structure. But remember just above: when doing inference, we want avoid non-useful computation. So we will 
deactivate the model's training mode and indicate to the **autograd** module that we don't want to track our next operations and calculate the gradient of our tensors. 
For that, we write our piece of code under **no_grad** context.

We write this function inside ``training/evaluation.py``.

.. code-block:: python

    def test(model, loader, f_loss, device):
    
        model.eval()
        with torch.no_grad():


            N = 0
            tot_loss, correct = 0.0, 0.0
            for i, (inputs, targets) in enumerate(loader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # We accumulate the exact number of processed samples
                N += inputs.shape[0]

                # We accumulate the loss considering
                # The multipliation by inputs.shape[0] is due to the fact
                # that our loss criterion is averaging over its samples
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

                predicted_targets = outputs.argmax(dim=1)
                correct += (predicted_targets == targets).sum().item()

            return tot_loss / N, correct / N

When we evaluate a model, we want known more than only the loss score. Here we also compute the accuracy score.

.. tip::
    
    For more clarity when the training is running, we can add a progress bar as below, with a built-in JAW method.

    .. code-block:: python

        from jaw.utils.progress_bar import progress_bar

        ...
        for i, (inputs, targets) in enumerate(loader):

            ...
            progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (tot_loss/(i+1), 100.*correct/N, correct, N))

        return tot_loss / N, correct / N
