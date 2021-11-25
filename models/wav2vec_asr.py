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
    self.wav2vec2 = Wav2Vec2Model.from_pretrained(hparams.model_name)
    self.dropout = nn.Dropout(config.final_dropout)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    self.init_weights()

  def freeze_feature_extractor(self):
    self.wav2vec2.feature_extractor._freeze_parameters()

  def parse_batch(self, batch, train=True):
    return batch["input_values"]

  def get_targets(self, batch):
    return batch["labels"]

  def forward(
    self,
    input_values,
    attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    labels=None,
  ):

    outputs = self.wav2vec2(
      input_values,
      attention_mask=attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    hidden_states = outputs[0]
    hidden_states = self.dropout(hidden_states)

    # TODO: Concat hidden_states
    logits = self.lm_head(hidden_states)

    attention_mask = (
      attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
    )
    input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    
    return {
      'logits': logits,
      'input_lengths': input_lengths,
      'hidden_states': outputs.hidden_states, 
      'attentions': outputs.attentions
    }

  def train_step(self, input_values):
    return self.forward(input_values)