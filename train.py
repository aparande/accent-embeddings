import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from data_utils import VCTK, ASRCollate
from models.tacotron2 import Tacotron2, Tacotron2Loss
from models.wav2vec_asr import Wav2VecASR, Wav2VecASRLoss
from multitask import AccentedMultiTaskNetwork, Task

from pytorch_lightning import Trainer

def load_data(params, data_params):
  if data_params.speaker is not None:
    dataset = VCTK_092_Speaker(data_params, data_params.speaker)
  else:
    dataset = VCTK(data_params)

  collate_fn = ASRCollate()

  val_size = int(params.val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(params.random_seed))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  val_loader = DataLoader(val, shuffle=False, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  return train_loader, val_loader, collate_fn

def train(params, multitask_params, model_params, train_loader, val_loader):
  
  # Multitask network without bottleneck, just acts as a wrapper to run model
  learning_rate, weight_decay = params.lr, params.weight_decay
  model = Wav2VecASR(model_params)
  loss = Wav2VecASRLoss()
  tasks = [Task(model, loss, lr, weight_decay, 'asr')]

  multitask_model = AccentedMultiTaskNetwork(multitask_params, tasks, lr=lr, weight_decay=weight_decay)
  trainer = Trainer()
  trainer.fit(multitask_model, train_loader, val_loader)