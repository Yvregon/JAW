Custom loss
===========

Many losses are already implemented in Pytorch, but if you read this tutorial, is probably because you need to reimplement many part by yourself, and loss calculation 
is a very important one. For the example, we will implement a relative L2 loss, i.e. a classical L2 (or MSE) loss but divided by a the inputs sum plus a little epsilon 
coefficient for avoiding by-zero divisions.

.. note::

    In the case of a simple article classifier, we obviously don't need such loss function. It's only for a demonstrative purpose.

As remember, a Pytorch loss inherit the ``nn.module`` just like a model object. So we need to implement the ``forward`` method which is, in fact, will call the 
``backward()`` method of the Pytorch MSE loss. We can also skip this method and write our derivation method from scratch by implementing the ``nn.Module.backward()`` 
method.

.. code-block:: python

    class RelativeL2(nn.Module):
    """
    Comparing to the regular MSE, introducing the division by \|c\| + epsilon
    (where epsilon = 0.01, to prevent values of 0 in the denominator).
    """

    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):

        base = target.sum() + self.eps

        return self.criterion(input.sum()/base, target.sum()/base)

Or:

.. code-block:: python

    class RelativeL2(nn.Module):

    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.eps = eps

    def backward(self, input, target):
        gradient = ... # Our derivation formula

        return gradient
    
