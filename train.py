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

from hyper_params import *


def load_data(params: TrainingParams, data_params: DataParams, precompute_features=True):
  dataset = VCTK(data_params, precompute_features=True)
  collate_fn = Collate()

  val_size = int(params.val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(params.random_seed))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  val_loader = DataLoader(val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
  return train_loader, val_loader

def train():
  tp = TrainingParams(val_size=0.1)
  dp = DataParams(filter_length=800, sample_rate=16000, win_length=800, hop_length=200)
  mp = MultiTaskParams(hidden_dim=[], in_dim=1024)

  train_loader, val_loader = load_data(tp, dp)
  checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=".", filename=tp.model_path, save_top_k=1)
  wandb_logger = WandbLogger(name=tp.run_name, project="accent_embeddings")


  tacotron = Tacotron2(TacotronParams())
  tacotron_loss = Tacotron2Loss()
  tts_task = Task(model=tacotron, loss=tacotron_loss, learning_rate=1e-3, weight_decay=1e-6, name='TTS', loss_weight=0.5)

  asr = Wav2VecASR(Wav2VecASRParams())
  asr_loss = Wav2VecASRLoss()
  asr_task = Task(model=asr, loss=asr_loss, learning_rate=1e-5, weight_decay=0, name='ASR', loss_weight=0.5)

  accent_id = Wav2VecID(Wav2VecIDParams())
  accent_id_loss = Wav2VecIDLoss()
  accent_id_task = Task(model=accent_id, loss=accent_id_loss, learning_rate=1e-5, weight_decay=0, name='ID', loss_weight=2)

  model = AccentedMultiTaskNetwork(mp, [accent_id_task, tts_task, asr_task])

  trainer = Trainer(gradient_clip_val=tp.grad_clip_thresh, max_epochs=tp.epochs, gpus=1, logger=wandb_logger, accumulate_grad_batches=tp.accumulate, callbacks=[checkpoint_callback])
  trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
  train()
