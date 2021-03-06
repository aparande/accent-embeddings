import numpy as np
import torch
import torch.nn as nn
import librosa

def get_mask_from_lengths(lengths):
  max_len = torch.max(lengths).item()
  dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
  ids = torch.arange(0, max_len, out=dtype(max_len))
  ids = ids.to(lengths.device)
  mask = (ids < lengths.unsqueeze(1)).bool()
  return mask

def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)

def build_mlp(dims):
  layers = []
  for i in range(len(dims) - 2):
    layers.append(nn.Linear(dims[i], dims[i+1]))
    layers.append(nn.ReLU())

  layers.append(nn.Linear(dims[-2], dims[-1]))
  return nn.Sequential(*layers)

def log_mel_spec_to_audio(mel_spec, dp):
  mel_spec = np.exp(mel_spec)
  gl_audio = librosa.feature.inverse.mel_to_audio(mel_spec, n_fft=dp.filter_length, hop_length=dp.hop_length, win_length=dp.win_length, fmin=dp.fmin, fmax=dp.fmax)

  return gl_audio
