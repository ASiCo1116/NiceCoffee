import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------------------------------------------------------
#loss.py contain the new loss function of the research
#--------------------------------------------------------------------------------


__all__ = ['BinaryCrossEntropyLoss', 'FocalLoss', 'HingeLoss', 'loss_select']


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight = None, size_average = None, reduce = None, reduction = 'mean'):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.loss_function = nn.BCEWithLogitsLoss(weight = weight, size_average = size_average,
                reduce = reduce, reduction = reduction)

    def forward(self, pred, label):
        loss = self.loss_function(pred, label)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 0.25, logits = True, reduce = None, reduction = 'mean'):
        #focalloss implement with sigmoid output
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduction = reduction
        if self.reduction == 'mean' and reduce == None:
            self.reduce = True
        elif self.reduction == 'all' and reduce == None:
            self.reduce = False
        elif reduce is not None:
            if type(reduce) == bool:
                self.reduce = reduce
            else:
                raise TypeError('Please check the type of args reduce.')
        else:
            raise RuntimeError('Please check all the args in FocalLoss is correct.')

    def forward(self, pred, label):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction = 'none')
        else:
            BCE_loss = F.binary_cross_entropy(pred, label, reduction = 'none')

        pt = torch.exp(-BCE_loss)
        modulating_factor = torch.pow((1 - pt), self.gamma)
        FocalLoss = torch.mul(torch.mul(self.alpha, modulating_factor), BCE_loss)

        if self.reduce:
            return torch.mean(FocalLoss)
        else:
            return FocalLoss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        pass

    def forward(self, pred, label):
        pass


def loss_select(loss_config):
    try:
        argument_dict = loss_config['argument']
    except KeyError:
        argument_dict = {}

    if loss_config['select'].lower() == 'binarycrossentropy':
        return BinaryCrossEntropyLoss(**argument_dict)
    elif loss_config['select'].lower() == 'focalloss':
        return FocalLoss(**argument_dict)


