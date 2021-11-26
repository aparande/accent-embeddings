import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from hyper_params import TrainingParams, TacotronParams, DataParams
from data_utils import VCTK, TTSCollate
from tacotron2 import Tacotron2, Tacotron2Loss

def load_data(params: TrainingParams, data_params: DataParams, n_frames_per_step: int):
  dataset = VCTK(data_params)

  collate_fn = TTSCollate(n_frames_per_step)

  val_size = int(params.val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(params.random_seed))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  val_loader = DataLoader(val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
  return train_loader, val_loader

