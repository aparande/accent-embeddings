# https://github.com/huggingface/transformers/issues/10497
import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

class Wav2VecASRLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, model_output, targets):
    return model_output["loss"]

class Wav2VecASR(Wav2Vec2ForCTC):
  def __init__(self, hparams):
    # Check that model used pretrained weights
    config = Wav2Vec2Config.from_pretrained(
              hparams.model_name,
              ctc_loss_reduction=hparams.ctc_loss_reduction,
              pad_token_id=hparams.pad_token_id
            )
    super().__init__(config)

  def parse_batch(self, batch):
    return batch, torch.empty(0)

  def forward(self, batch):
    causal_lm_outputs = super().forward(
      batch.get("input_values", None), 
      batch.get("attention_mask", None), 
      batch.get("output_attentions", None),
      batch.get("output_hidden_states", None),
      batch.get("return_dict", None),
      batch.get("labels", None)
    )

    outputs = {}
    outputs["loss"] = causal_lm_outputs["loss"]
    outputs["logits"] = causal_lm_outputs["logits"]
    # outputs["hidden_states"] = causal_lm_outputs["hidden_states"]
    # outputs["attentions"] = causal_lm_outputs["attentions"]
    outputs["label_ids"] = batch.get("labels", None)
    return outputs
