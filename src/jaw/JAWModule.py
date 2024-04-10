from abc import ABC, abstractmethod
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset
from types import FunctionType
import sys
from datetime import date, datetime


class JAWTrainer(ABC):
    """
    Abstract class representing a JAW's style pytorch trainer. Implement this class is the only requirement of a JAW project.
    """

    def __init__(self, model : torch.nn.Module,
                 loss : torch.nn.Module,
                 train_loader : torch.utils.data.dataloader,
                 val_loader : torch.utils.data.dataloader,
                 test_loader : torch.utils.data.dataloader,
                 train_func : FunctionType,
                 eval_func : FunctionType) -> None:
        """
        JAWTrainer constructor. Here is registered all parameters that can most likely not change between two training.

        .. note::
            Please notice that in the litterature it's usually the validation dataset that is use for the testing process and the test dataset for the
            validation process, not the opposite. It's also tgenerally the same method that is use for the validation and the testing process.

        :parameter model: Model to train.
        :parameter loss: Loss used for train and evaluate the model.
        :parameter train_loader: Dataloader used for load the training data.
        :parameter val_loader: Dataloader used for load the data that will be used for testing the training process.
        :parameter test_loader: Dataloader used for load the data that will be used for validate the model.
        :parameter train_func: the method that handle the train loop.
        :parameter test_func: the method that handle the validation loop.
        """
        
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_func = train_func
        self.eval_func = eval_func
        

    @abstractmethod
    def launch_training(self, epochs : int, device : torch.device, logdir : str, prefix : str) -> None:
        """Declaration of a JAW's style training workflow.

        :param epochs: Number of total training complete epochs.
        :type epochs: int.
        :param device: Pytorch device used for this training.
        :type device: torch.device.
        :param logdir: The name of the directory where your models and training info will be saved.
        :type logdir: str.
        :param prefix: Prefix of the training saving directory.
        :type prefix: str.
        
        :returns:  None.
        """
        pass


    def get_summary_text(self, optimizer : torch.optim):
        """
        Return the text that will written inside the training summary file.
        """
        summary_text = """Executed command
================
{}

Date
====
{} ({})

Dataset
=======
FashionMNIST

Model summary
=============
{}

{} trainable parameters

Optimizer
========
{}

""".format(
            " ".join(sys.argv),
            date.today(),
            datetime.now().strftime('%H:%M:%S'),
            self.model,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            optimizer,
        )

        return summary_text
