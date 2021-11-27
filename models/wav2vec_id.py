from torch import nn
from utils import build_mlp

class Wav2VecIDLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, model_output, targets):
    return nn.functional.cross_entropy(model_output, targets)


class Wav2VecID(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.network = build_mlp([params.accent_embed_dim, *params.hidden_dim, params.num_accents]) 

  def parse_batch(self, batch, train=True):
    return None

  def get_targets(self, batch):
    return batch["accents"]

  def forward(
      self,
      inputs,
      accent_embed
  ):
    output = self.network(accent_embed)

    return output

  def training_step(self, inputs, accent_embed):
    return self.forward(inputs, accent_embed)