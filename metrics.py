import torch
import torch.nn.functional as F
import numpy as np

from abc import ABC, abstractmethod
from transformers import Wav2Vec2Processor
from datasets import load_metric

class Metric(ABC):
  def __init__(self, name):
    self.name = name

  @abstractmethod
  def __call__(self, pred, target):
    pass

class SoftmaxAccuracy(Metric):
  def __init__(self):
    super().__init__("Accuracy")

  def __call__(self, preds, targets):
    pred = torch.cat(preds)
    target = torch.cat(targets)
    assert pred.shape[0] == target.shape[0], f"Prediction and target output length does not match: pred shape {pred.shape} != target shape {target.shape}"
    logits = F.softmax(pred, dim=1)
    return (torch.argmax(logits, dim=1) == target).float().sum() / pred.shape[0]

class WERAccuracy(Metric):
  def __init__(self, model_name = "facebook/wav2vec2-large-960h"):
    super().__init__("WER")
    self.processor = Wav2Vec2Processor.from_pretrained(model_name)
    self.metric = load_metric("wer")

  def __call__(self, preds, targets):
    pred_str, label_str = [], []
    for i in range(len(preds)):
      pred, target = preds[i], targets[i]
      pred_logits = pred["logits"]
      pred_ids = pred_logits.argmax(axis=-1)
      target[target == -100] = self.processor.tokenizer.pad_token_id
      pred_str.extend(self.processor.batch_decode(pred_ids))
      label_str.extend(self.processor.batch_decode(target, group_tokens=False))
    wer = self.metric.compute(predictions=pred_str, references=label_str)
    return wer

class MSE(Metric):
  def __init__(self):
    super().__init__("MSE")

  def __call__(self, pred, target):
    running_loss = 0
    n_batches = 0
    for x, y in zip(pred, target):
      mel_out_postnet = x['mel_out_postnet']
      mel_target = y['mfcc']

      running_loss += F.mse_loss(mel_out_postnet, mel_target) * mel_target.shape[0]
      n_batches += mel_target.shape[0]

    return running_loss / n_batches