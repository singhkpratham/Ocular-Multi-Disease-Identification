import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BCELoss(nn.Module):
    def __init__(self, n_classes, batch_weights):
        super(BCELoss, self).__init__()
        if n_classes > 1:
            self.criterion = nn.CrossEntropyLoss() # included Softmax activation
        else:
            self.criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(batch_weights).float().cuda()) # included Sigmoid activation

    def forward(self, output, labels, train=True):
        return self.criterion(output, labels)


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.00001
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        # print(DiceCoeff().forward(c[0], c[1]))
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
