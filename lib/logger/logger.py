"""
logger.py
============
Logging utilities for network training.
"""

from torch.utils.tensorboard import SummaryWriter

class NetLogger(object):
    """
    Simple Logger class for easy tensorboard logging.

    :param log_dir: Directory the log gets written to
    :type log_dir: string
    :param losses: List of loss names, when updating loss make sure to use
        same order as here. If None both lists of length one and floats can be
        used to update loss
    :type losses: list
    """
    def __init__(self, log_dir, losses = None):
        self.trainig_loss_steps = 0
        self.validation_loss_steps = 0
        self.trainig_acc_steps = 0
        self.validation_acc_steps = 0
        self.writer = SummaryWriter(log_dir=log_dir)
        if losses == None:
            self.losses = ['Loss']
        else:
            self.losses = losses


    def update_train_loss(self, loss):
        """
        Update training loss tensorboard log.

        :param loss: List of losses or single loss value, see above for details
        :type loss: float/list
        """
        if not isinstance(loss, list):
            loss = [loss]
        assert len(loss) == len(self.losses), \
                    "Number of given losses doesn't match loss list..."
        for loss, loss_name in zip(loss, self.losses):
            self.writer.add_scalar("Train " + loss_name, loss,
                                   self.trainig_loss_steps)
        self.trainig_loss_steps += 1


    def update_val_loss(self, loss):
        """
        Update validation loss tensorboard log.

        :param loss: List of losses or single loss value, see above for details
        :type loss: float/list
        """
        if not isinstance(loss, list):
            loss = [loss]
        assert len(loss) == len(self.losses), \
                    "Number of given losses doesn't match loss list..."
        for loss, loss_name in zip(loss, self.losses):
            self.writer.add_scalar("Validation " + loss_name,
                                   loss, self.validation_loss_steps)
        self.validation_loss_steps += 1


    def update_train_accuracy (self, acc):
        """
        Update training accuracy tesnorboard log.

        :param acc: current accuracy
        :type acc: float
        """
        self.writer.add_scalar('Train Accuracy', acc, self.trainig_acc_steps)
        self.trainig_acc_steps += 1


    def update_val_accuracy(self, acc):
        """
        Update validation accuracy tesnorboard log.

        :param acc: current accuracy
        :type acc: float
        """
        self.writer.add_scalar('Validation Accuracy', acc,
                               self.validation_acc_steps)
        self.validation_acc_steps += 1



class AverageMeter():
    """
    Simple average metering class to better monitor training parameters like
    loss and accuracy.
    """
    def __init__(self):
        self.value = 0
        self.step = 0


    def update(self, val):
        """
        Update metered value.

        :param val: Value to add to average
        :type val: float
        """
        self.value += val
        self.step += 1


    def read(self):
        """
        Read metered value.

        :returns: Metered value or 0 if no value added to meter
        :rtype: float
        """
        if self.step != 0:
            return (self.value/self.step)
        else:
            return 0


    def reset(self):
        """
        Resets metered value.
        """
        self.value = 0
        self.step = 0
