from typing import NamedTuple, List

import torch
from torch import nn
import torch.nn.functional as F

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pytorch_lightning as pl

from hyper_params import MultiTaskParams

class Task(NamedTuple):
  model: nn.Module
  loss: nn.Module
  learning_rate: float
  weight_decay: float
  name: str

class AccentedMultiTaskNetwork(pl.LightningModule):
  def __init__(self, params: MultiTaskParams, tasks: [Task], lr=1e-3, weight_decay=0.0):
    super().__init__()

    self.params = params
    self.bottleneck = self._build_bottleneck()
    self.tasks = tasks
    # Makes network aware of other model parameters.
    self.models = nn.ModuleList([task.model for task in self.tasks])

    # self.bottleneck = Bottleneck(params.in_dim, params.out_dim, params.hidden_dim)
    # self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(params.wav2vec)
    # self.wav2vec_model = Wav2Vec2Model.from_pretrained(params.wav2vec)

    for param in self.wav2vec_model.parameters():
      param.requires_grad = False

    self.lr = lr
    self.weight_decay = weight_decay

  def _build_bottleneck(self):
    layers = []
    layers.append(nn.Linear(self.params.in_dim, self.params.hidden_dim[0]))
    for i, dim in enumerate(self.params.hidden_dim[:-1]):
      layers.append(nn.ReLU())
      layers.append(nn.Linear(dim, self.params.hidden_dim[i + 1]))

    layers.append(nn.ReLU())
    layers.append(nn.Linear(self.params.hidden_dim[-1], self.params.out_dim))

    return nn.Sequential(*layers)

  def get_wav2vec_features(self, batch):
    waveforms = batch["waveform"]
    features = self.wav2vec_processor(waveforms, sampling_rate=16000, return_tensors = 'pt')
    outputs = self.wav2vec_model(features.input_values[0]).last_hidden_state
    return torch.mean(outputs, 1)

  def forward(self, inputs):
    # wav2vec_feats = self.get_wav2vec_features(inputs)
    # accent_embed = self.bottleneck(wav2vec_feats)

    outs = dict()
    for task in self.tasks:
      x = task.model.parse_batch(inputs)
      outs[task.name] = task.model(x)

    return outs

  def training_step(self, batch, batch_idx):
    wav2vec_feats = self.get_wav2vec_features(batch)
    accent_embed = self.bottleneck(wav2vec_feats)
    loss_vals = []
    for task in self.tasks:
      x = task.model.parse_batch(batch, train=True)
      y_pred = task.model.training_step(x, accent_embed)
      targets = task.model.get_targets(batch)

      loss_vals.append(task.loss(y_pred, targets))

    total_loss = loss_vals[0]
    for val in loss_vals[1:]:
      total_loss += val

    for task, loss_val in zip(self.tasks, loss_vals):
      self.log(f"train_loss_{task.name}", loss_val)

    self.log("train_loss", total_loss)
    return total_loss

  def validation_step(self, batch, batch_idx):
    wav2vec_feats = self.get_wav2vec_features(batch)
    accent_embed = self.bottleneck(wav2vec_feats)

    loss_vals = []
    for task in self.tasks:
      x = task.model.parse_batch(batch)
      y_pred = task.model(x, accent_embed)

      targets = task.model.get_targets(batch)

      loss_vals.append(task.loss(y_pred, targets))

    total_loss = loss_vals[0]
    for val in loss_vals[1:]:
      total_loss += val

    for task, loss_val in zip(self.tasks, loss_vals):
      self.log(f"val_loss_{task.name}", loss_val)

    self.log("val_loss", total_loss)

  def configure_optimizers(self):
    optim_args = [ {
      "params" : task.model.parameters(),
      "lr": task.learning_rate,
      "weight_decay": task.weight_decay
    } for task in self.tasks ]
    # optim_args.append( { "params" : self.bottleneck.parameters() } )
    return torch.optim.Adam(optim_args, lr=self.lr, weight_decay=self.weight_decay)

