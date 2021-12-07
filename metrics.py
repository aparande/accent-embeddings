import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

class Metric(ABC):
  def __init__(self, name):
    self.name = name

  @abstractmethod
  def __call__(self, pred, target):
    pass

class SoftmaxAccuracy(Metric):
  def __init__(self):
    super().__init__("Accuracy")

  def __call__(self, pred, target):
    assert pred.shape[0] == target.shape[0], f"Prediction and target output length does not match: pred shape {pred.shape} != target shape {target.shape}"
    logits = F.softmax(pred, dim=1)
    return (torch.argmax(logits, dim=1) == target).float().sum() / pred.shape[0]
