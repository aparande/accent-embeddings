from typing import NamedTuple, List, Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pytorch_lightning as pl
from collections import defaultdict

from hyper_params import MultiTaskParams
from utils import build_mlp

class Task(NamedTuple):
  model: nn.Module
  loss: nn.Module
  learning_rate: float
  weight_decay: float
  loss_weight: float
  name: str
  metrics: List[Callable]

class AccentedMultiTaskNetwork(pl.LightningModule):
  def __init__(self, params: MultiTaskParams, tasks: [Task], lr=1e-3, weight_decay=0.0):
    super().__init__()

    self.params = params
    self.bottleneck = build_mlp([self.params.in_dim, *self.params.hidden_dim, self.params.out_dim])
    self.tasks = tasks
    # Makes network aware of other model parameters.
    self.models = nn.ModuleList([task.model for task in self.tasks])
    self.wav2vec_model = Wav2Vec2Model.from_pretrained(params.wav2vec)

    if self.params.wav2vec_freeze_feature_extractor:
      self.wav2vec_model.feature_extractor._freeze_parameters()

    # for param in self.wav2vec_model.parameters():
    #   param.requires_grad = False

    self.lr = lr
    self.weight_decay = weight_decay

    self.task_idx = 0
    self.epoch = 0

    self.iteration = 0
    self.loss_history = np.zeros((len(self.tasks), 2))
    self.loss_weights = np.array([task.loss_weight for task in self.tasks])


  def get_accent_embed(self, batch):
    input_values = batch["wav2vec_input"]
    outputs = self.wav2vec_model(input_values).last_hidden_state
    outputs = torch.mean(outputs, 1)
    return self.bottleneck(outputs)

  def get_wav2vec_features(self, batch):
    input_values = batch["wav2vec_input"]
    outputs = self.wav2vec_model(input_values).last_hidden_state
    batch["wav2vec_hidden"] = outputs
    return torch.mean(outputs, 1)

  def forward(self, inputs):
    wav2vec_feats = self.get_wav2vec_features(inputs)
    accent_embed = self.bottleneck(wav2vec_feats)

    outs = dict()
    for task in self.tasks:
      x = task.model.parse_batch(inputs)
      outs[task.name] = task.model(x, accent_embed)

    return outs

  def training_step(self, batch, batch_idx):
    self.iteration += 1
    wav2vec_feats = self.get_wav2vec_features(batch)
    accent_embed = self.bottleneck(wav2vec_feats)
    loss_vals = []

    if self.params.alternate_epoch_interval > 0:
      task = self.tasks[self.task_idx]
      x = task.model.parse_batch(batch, train=True)
      y_pred = task.model.training_step(x, accent_embed)
      targets = task.model.get_targets(batch)

      loss_vals.append(task.loss(y_pred, targets))
    else:
      for task in self.tasks:
        x = task.model.parse_batch(batch, train=True)
        y_pred = task.model.training_step(x, accent_embed)
        targets = task.model.get_targets(batch)

        loss_vals.append(task.loss(y_pred, targets))

      for i, loss in enumerate(loss_vals):
        self.loss_history[i, 0], self.loss_history[i, 1] = loss.item(), self.loss_history[i, 0]

      if self.params.loss_weighting_strategy == "dwa":
        weights = F.softmax(torch.from_numpy(self.loss_weights), dim=0).numpy()
      else:
        weights = self.loss_weights

      loss_vals = [weight * val for weight, val in zip(weights, loss_vals)]

    total_loss = sum(loss_vals)

    if self.iteration >= 2 and self.params.loss_weighting_strategy == "dwa":
      self.loss_weights = self.loss_history[:, 0] / self.loss_history[:, 1]

    for task, loss_val, w in zip(self.tasks, loss_vals, weights):
      self.log(f"train_loss_{task.name}", loss_val)
      self.log(f"loss_weight_{task.name}", w)

    self.log("train_loss", total_loss)

    return total_loss

  def validation_step(self, batch, batch_idx):
    wav2vec_feats = self.get_wav2vec_features(batch)
    accent_embed = self.bottleneck(wav2vec_feats)
    val_out = {}

    loss_vals = []

    if self.params.loss_weighting_strategy == "dwa":
      weights = F.softmax(torch.from_numpy(self.loss_weights), dim=0).numpy()
    else:
      weights = self.loss_weights

    for task, weight in zip(self.tasks, weights):
      x = task.model.parse_batch(batch, train=True)
      y_pred = task.model(x, accent_embed)

      targets = task.model.get_targets(batch)

      val_out[task.name] = (y_pred, targets)

      loss_vals.append(weight * task.loss(y_pred, targets))

    total_loss = sum(loss_vals)

    for task, loss_val in zip(self.tasks, loss_vals):
      self.log(f"val_loss_{task.name}", loss_val)

    self.log("val_loss", total_loss)
    return val_out

  def training_epoch_end(self, train_outs):
    if self.params.alternate_epoch_interval > 0 and (self.epoch + 1) % self.params.alternate_epoch_interval == 0:
      print(f"Turning off {self.tasks[self.task_idx].name}")
      self.task_idx = (self.task_idx + 1) % len(self.tasks)
      print(f"Turning on {self.tasks[self.task_idx].name}")
    self.epoch += 1


  def validation_epoch_end(self, val_outs):
    preds_dict = { task.name : [] for task in self.tasks }
    targets_dict = { task.name : [] for task in self.tasks }
    for batch_out in val_outs:
      for task_name in batch_out:
        y_pred, target = batch_out[task_name]
        preds_dict[task_name].append(y_pred)
        targets_dict[task_name].append(target)

    for task in self.tasks:
      y_preds, targets = preds_dict[task.name], targets_dict[task.name]
      for metric in task.metrics:
        self.log(f"{metric.name}_on_{task.name}", metric(y_preds, targets))

  def configure_optimizers(self):
    optim_args = [ {
      "params" : task.model.parameters(),
      "lr": task.learning_rate,
      "weight_decay": task.weight_decay
    } for task in self.tasks ]
    optim_args.append( {
      "params" : self.wav2vec_model.parameters(),
      "lr": self.params.wav2vec_learning_rate
    } )
    optim_args.append( { "params" : self.bottleneck.parameters() } )
    return torch.optim.Adam(optim_args, lr=self.lr, weight_decay=self.weight_decay)

