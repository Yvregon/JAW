from abc import ABC, abstractmethod
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset
from types import FunctionType
import sys
from datetime import date, datetime
import argparse
from jaw.utils.tracking import ModelCheckpoint


class JAWTrainer(ABC):

    def __init__(self, model : torch.nn.Module,
                 loss : torch.nn.Module,
                 train_loader : torch.utils.data.dataloader,
                 val_loader : torch.utils.data.dataloader,
                 test_loader : torch.utils.data.dataloader,
                 train_func : FunctionType,
                 eval_func : FunctionType) -> None:
        
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_func = train_func
        self.eval_func = eval_func
        

    @abstractmethod
    def launch_training(self, epochs : int, device : torch.device, logdir : str, prefix : str) -> None:
        pass


    def get_summary_text(self, optimizer : torch.optim):
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
