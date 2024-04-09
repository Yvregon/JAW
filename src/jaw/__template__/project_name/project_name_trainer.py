from jaw.JAWModule import JAWTrainer
from jaw.utils.tracking import ModelCheckpoint
import torch
import os
import argparse
from jaw.utils.tracking import generate_unique_logpath
from jaw.utils.computation import get_device


class ProjectNameTrainer(JAWTrainer):
    """Typical example implementation of a JAWTrainer. We want train a mini classifier with the fashion MNIST dataset (https://github.com/zalandoresearch/fashion-mnist).

    Here we want be able to launch a training with our previously written classes (dataloader, models, losses, training and evaluation processes).

    .. note::

       Here we only use the necessary methods for implement a JAWTrainer, but you are free to declare new parameters.

    """

    def launch_training(self, epochs : int, device : torch.device, logdir : str, prefix : str) -> None:
        """Method where you define your custom training workflow.

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


def main(args :  dict) -> None:
    """Training definition. It's here where we give our custom classes at our previous written :func: `launch_training`.

        :param args: Argument given in the command line.
        :type args: dict.
        
        :returns:  None.

    """

    project_name_trainer : ProjectNameTrainer = ProjectNameTrainer(model=...,
                                                               loss=...,
                                                               train_loader=...,
                                                               val_loader=...,
                                                               test_loader=...,
                                                               train_func=...,
                                                               eval_func=...,
                                                               )
    

    project_name_trainer.launch_training(args["epochs"], get_device(), args["logdir"], args["prefix"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Parse arguments

    main(vars(parser.parse_args()))
