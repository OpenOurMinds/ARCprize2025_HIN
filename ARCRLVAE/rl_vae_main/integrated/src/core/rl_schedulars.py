import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import logging

from src.utils.logger_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LinearDecayScheduler(_LRScheduler):
    """
    A learning rate scheduler that linearly decays the learning rate
    from its initial value to zero over a specified number of epochs.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, last_epoch: int = -1, verbose: bool = False):
        """
        Initializes the linear decay scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to which this scheduler will be attached.
            total_epochs (int): The total number of epochs for the decay. The learning rate will reach zero at this epoch.
            last_epoch (int): The index of the last epoch. Defaults to -1.
            verbose (bool): If True, prints a message for each update. Defaults to False.
        """
        self.total_epochs = total_epochs
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super(LinearDecayScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """
        Computes the new learning rate for the current epoch.

        The formula is: lr = initial_lr * (1 - (current_epoch / total_epochs))

        Returns:
            list[float]: A list of new learning rates, one for each parameter group in the optimizer.
        """
        if self.last_epoch >= self.total_epochs:
            return [0.0 for _ in self.base_lrs]
        
        return [
            base_lr * (1.0 - self.last_epoch / self.total_epochs)
            for base_lr in self.base_lrs
        ]

# A placeholder for future schedulers if needed
# class ExponentialDecayScheduler(_LRScheduler):
#     def get_lr(self):
#         ...