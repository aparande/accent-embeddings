import torch
import torch.nn.functional as F


def softmax_and_accuracy(pred, target):
    assert pred.shape == target.shape, "Shapes are not equal"
    return (torch.argmax(F.softmax(pred, dim=1), dim=1) == target).float().sum() / pred.shape[0]
