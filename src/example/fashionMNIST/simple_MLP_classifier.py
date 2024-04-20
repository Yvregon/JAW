from jaw.JAWModule import JAWTrainer
from jaw.utils.tracking import ModelCheckpoint
import torch
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from example.fashionMNIST.models.FullyConnected import *
from example.fashionMNIST.losses.loss import *
from example.fashionMNIST.data_preprocessing.fashion_MNIST_dataset import *
from example.fashionMNIST.training import evaluation, train
from jaw.utils.tracking import generate_unique_logpath
from jaw.utils.computation import get_device


class SimpleMLPClassifier(JAWTrainer):
    """Typical example implementation of a JAWTrainer. We want train a mini classifier with the
    `fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ dataset.

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

        # Create the directory "./logs" if it does not exist
        top_logdir = "./logs"
        if not os.path.exists(top_logdir):
            os.mkdir(top_logdir)

        logdir = generate_unique_logpath(top_logdir, prefix)
        print("Logging to {}".format(logdir))
        # -> Prints out     Logging to   ./logs/simple_MLP_classifier_X
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        

        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.to(device)
        # We only keeping the best model (depends of the validation loss)
        checkpoint = ModelCheckpoint(logdir + "/best_model.pt", self.model)

        tensorboard_writer = SummaryWriter(log_dir=logdir)

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            self.train_func(self.model, self.train_loader, self.loss, optimizer, device)
            train_loss, train_acc = self.eval_func(self.model, self.train_loader, self.loss, device)
            val_loss, val_acc = self.eval_func(self.model, self.val_loader, self.loss, device)

            print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

            # Check if the current model is better than the previous best.
            checkpoint.update(val_loss)

            # Add the score of the training and validation loss and accuracy inside the tensorboard logs. You can decide to add more information
            # or keep only losses (it doesn't make sense to show the accuracy of a regression model).
            tensorboard_writer.add_scalar("metrics/train_loss", train_loss, epoch)
            tensorboard_writer.add_scalar("metrics/train_acc", train_acc, epoch)
            tensorboard_writer.add_scalar("metrics/val_loss", val_loss, epoch)
            tensorboard_writer.add_scalar("metrics/val_acc", val_acc, epoch)

        # Write the summary file
        summary_text = self.get_summary_text(optimizer)
        summary_file = open(logdir + "/summary.txt", "w")
        summary_file.write(summary_text)
        summary_file.close()

        tensorboard_writer.add_text("Experiment summary", summary_text)


def main(args :  dict) -> None:
    """Training definition. It's here where we give our custom classes at our previous written :func: `launch_training`.

        :param args: Argument given in the command line.
        :type args: dict.
        
        :returns:  None.

    """
    train_loader, val_loader, test_loader = load_dataset_FashionMNIST_with_standardization("dataset/")
    model : torch.nn.Module = build_model(args["model"])
    loss : torch.nn.Module = loss_name_to_class(args["loss"])

    MNIST_training : SimpleMLPClassifier = SimpleMLPClassifier(model=model,
                                                               loss=loss,
                                                               train_loader=train_loader,
                                                               val_loader=val_loader,
                                                               test_loader=test_loader,
                                                               train_func=train.train,
                                                               eval_func=evaluation.test,
                                                               )
    

    MNIST_training.launch_training(args["epochs"], get_device(), args["logdir"], args["prefix"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epoch",
        default=None,
        required=True
    )

    parser.add_argument(
        "--loss",
        choices=["CrossEntropy", "RelativeL1", "RelativeL2"],
        help="Choose between CrossEntropy, RelativeL1 and RelativeL2.",
        action="store",
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=["FC", "FCReg"],
        help="Choose between FC (Fully Connected) and FCReg (Regularized).",
        action="store",
        required=True,
    )

    parser.add_argument(
        "--logdir",
        type=str,
        help="Where write the result of the training and save the best model",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix of the folder where you want to save your results.",
        default="my_model",
        required=False,
    )

    main(vars(parser.parse_args()))
