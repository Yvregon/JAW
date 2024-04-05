import torch
import os

def generate_unique_logpath(logdir, prefix):
    """
    Generate a unique log file for a new model saving

    Arguments :

        logdir    -- The path of the directory where the logs will be saved
        prefix    -- Prefix name of the log file

    Returns :
    """
    i = 0
    while True:
        run_name = prefix + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint:
    """
    Utility class for saving a model when a training produce a new better.
    """

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            self.min_loss = loss
