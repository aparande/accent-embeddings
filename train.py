from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from hyper_params import TrainingParams, TacotronParams
from data_utils import VCTK, TTSCollate
from tacotron2 import Tacotron2, Tacotron2Loss

def load_data(params: TrainingParams, n_frames_per_step: int, mfcc_num: int, val_size = 0.1):
  dataset = VCTK(mfcc_num)
  collate_fn = TTSCollate(n_frames_per_step)

  val_size = int(val_size * len(dataset))
  train_size = len(dataset) - val_size
  train, val = random_split(dataset, (train_size, val_size))

  train_loader = DataLoader(train, shuffle=True, batch_size=params.batch_size, drop_last=True, collate_fn=collate_fn)
  return train_loader, val, collate_fn

def train(params: TrainingParams, model_params: TacotronParams):
  model = Tacotron2(model_params)
  optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

  criterion = Tacotron2Loss()

  train_loader, valset, collate_fn = load_data(params, model_params.n_frames_per_step, model_params.n_mel_channels)

  iteration = 0

  if torch.cuda.is_available():
    model = model.cuda()

  model.train()

  iteration = 0
  for epoch in range(params.epochs):
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

    print(f"Finished Epoch {epoch} with average loss: {epoch_loss / len(train_loader)}")
    torch.save(model.state_dict(), params.model_path)
