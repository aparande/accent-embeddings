import numpy as np
import torch

def get_mask_from_lengths(lengths):
  max_len = torch.max(lengths).item()
  dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
  ids = torch.arange(0, max_len, out=dtype(max_len))
  mask = (ids < lengths.unsqueeze(1)).bool()
  return mask

def to_gpu(x):
  x = x.contiguous()

  if torch.cuda.is_available():
      x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)

