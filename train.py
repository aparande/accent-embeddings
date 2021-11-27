import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from data_utils import VCTK, Collate
from models.tacotron2 import Tacotron2, Tacotron2Loss
from models.wav2vec_asr import Wav2VecASR, Wav2VecASRLoss
from models.wav2vec_id import Wav2VecID, Wav2VecIDLoss
from multitask import AccentedMultiTaskNetwork, Task

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from hyper_params import TrainingParams, DataParams, MultiTaskParams, Wav2VecASRParams


def load_data(params: TrainingParams, data_params: DataParams):
  dataset = VCTK(data_params)
  collate_fn = Collate()

  val_size = int(params.val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(params.random_seed))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  val_loader = DataLoader(val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
  return train_loader, val_loader

def train(params, multitask_params, model_params, train_loader, val_loader):
  checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=".", filename=params.model_path, save_top_k=1)
  wandb_logger = WandbLogger(project="accent_embeddings")

  # Multitask network without bottleneck, just acts as a wrapper to run model
  lr, weight_decay = params.learning_rate, params.weight_decay
  model = Wav2VecASR(model_params)
  loss = Wav2VecASRLoss()
  tasks = [Task(model, loss, lr, weight_decay, 'asr')]

  multitask_model = AccentedMultiTaskNetwork(multitask_params, tasks, lr=lr, weight_decay=weight_decay)
  trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
  trainer.fit(multitask_model, train_loader, val_loader)

if __name__ == "__main__":
  train_loader, val_loader, collate_fn = load_data(TrainingParams, DataParams)
  train(TrainingParams, MultiTaskParams, Wav2VecASRParams, train_loader, val_loader)
