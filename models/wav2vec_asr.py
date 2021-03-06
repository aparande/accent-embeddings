import torch
from torch import nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Config

class Wav2VecASRLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, model_output, targets):
    labels = targets
    logits = model_output["logits"]
    input_lengths = model_output["input_lengths"]
    
    # assuming that padded tokens are filled with -100
    # when not being attended to
    labels_mask = labels >= 0
    target_lengths = labels_mask.sum(-1)
    flattened_targets = labels.masked_select(labels_mask)
    log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    
    # hardcode settings
    with torch.backends.cudnn.flags(enabled=False):
      loss = nn.functional.ctc_loss(
        log_probs,
        flattened_targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
      )
    return loss

class Wav2VecASR(Wav2Vec2PreTrainedModel):
  def __init__(self, hparams):
    config = Wav2Vec2Config.from_pretrained(hparams.model_name)

    super().__init__(config)
    # self.wav2vec2 = Wav2Vec2Model.from_pretrained(hparams.model_name)
    self.dropout = nn.Dropout(config.final_dropout)
    self.lm_head = nn.Linear(config.hidden_size + hparams.accent_embed_dim, config.vocab_size)
    self.init_weights()

  # def freeze_feature_extractor(self):
  #   self.wav2vec2.feature_extractor._freeze_parameters()

  def parse_batch(self, batch, train=True):
    return batch["wav2vec_hidden"], batch["wav2vec_input"]

  def get_targets(self, batch):
    return batch["wav2vec_text"]

  def forward(
    self,
    inputs,
    accent_embed,
    attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    labels=None
  ):

    hidden_states, input_values = inputs
    hidden_states = self.dropout(hidden_states)

    accent_embed = accent_embed.unsqueeze(1)
    accent_embed = accent_embed.expand(-1, hidden_states.size(1), -1)
    lm_inputs = torch.cat([hidden_states, accent_embed], dim=2)

    logits = self.lm_head(lm_inputs)

    attention_mask = (
      attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
    )
    input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    
    return {
      'logits': logits,
      'input_lengths': input_lengths
    }

  def training_step(self, inputs, accent_embed):
    return self.forward(inputs, accent_embed)