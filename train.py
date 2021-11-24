import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from hyper_params import TrainingParams, TacotronParams, DataParams
from data_utils import VCTK, TTSCollate
from tacotron2 import Tacotron2, Tacotron2Loss

def load_data(params: TrainingParams, data_params: DataParams, n_frames_per_step: int):
  if data_params.speaker is not None:
    dataset = VCTK_092_Speaker(data_params, data_params.speaker)
  else:
    dataset = VCTK(data_params)

  collate_fn = TTSCollate(n_frames_per_step)

  val_size = int(params.val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size), generator=torch.Generator().manual_seed(params.random_seed))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  return train_loader, val, collate_fn

def validate(model, criterion, val_set, batch_size, collate_fn):
  model.eval()
  with torch.no_grad():
    val_loader = DataLoader(val_set, batch_size, shuffle=False, collate_fn=collate_fn)

    val_loss = 0.0
    for i, batch in enumerate(val_loader):
      x, y = model.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      val_loss += loss.item()

  model.train()
  return val_loss / (i + 1)

def train(params: TrainingParams, model_params: TacotronParams, data_params: DataParams, 
          train_loader: DataLoader, val_set, collate_fn, model: Tacotron2=None, optimizer: torch.optim.Adam = None):
  assert model_params.n_mel_channels == data_params.n_mel_channels, "MFCC output does not match data"

  if model is None:
    model = Tacotron2(model_params)
    if os.path.exists(params.model_path):
      print("Loading pre-trained model")
      model.load_state_dict(torch.load(params.model_path, map_location=torch.device('cpu')))

  if optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

  criterion = Tacotron2Loss()

  if torch.cuda.is_available():
    model = model.cuda()

  iteration = 0
  train_losses = []
  val_losses = []
  for epoch in range(params.epochs):
    model.train()
    print(f"Starting Epoch {epoch}")
    epoch_loss = 0
    report_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
      iteration += 1

      model.zero_grad()
      x, y = model.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      loss.backward()

      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_thresh)

      optimizer.step()

      epoch_loss += loss.item()
      report_loss += loss.item()

      if iteration % params.report_interval == 0:
        print(f"Loss: {report_loss / params.report_interval}")
        report_loss = 0

      if iteration % params.save_interval == 0:
        print(f"Saving Model")
        torch.save(model.state_dict(), params.model_path)

    val_loss = validate(model, criterion, val_set, params.batch_size, collate_fn)
    train_loss = epoch_loss / len(train_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Finished Epoch {epoch} with (train, val): {train_loss}, {val_loss}")
    torch.save(model.state_dict(), params.model_path)

  return model, optimizer, train_losses, val_losses

