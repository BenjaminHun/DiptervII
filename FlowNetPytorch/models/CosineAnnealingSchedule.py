import math


class CosineAnnealingSchedule:
    def __init__(self, initial_lr, T_max, eta_min=0):
        """
        Initialize the cosine annealing learning rate scheduler.

        Args:
        - initial_lr: Initial learning rate.
        - T_max: Number of epochs for a full cycle of the cosine annealing schedule.
        - eta_min: Minimum learning rate. Defaults to 0.
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch):
        """
        Calculate the learning rate for a given epoch.

        Args:
        - epoch: Current epoch.

        Returns:
        - lr: Learning rate for the given epoch.
        """
        if epoch >= self.T_max:
            return self.eta_min
        else:
            return self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
