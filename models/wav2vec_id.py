import torch
from torch import nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Config
from transformers import AutoConfig, AutoModelForAudioClassification


class Wav2VecIDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output):
        return model_output['loss']


class Wav2VecID(Wav2Vec2PreTrainedModel):
    def __init__(self, hparams):
        config = Wav2Vec2Config.from_pretrained(hparams.model_name)

        super().__init__(config)
        self.wav2vec2 = AutoModelForAudioClassification.from_config(config)

    def parse_batch(self, batch, train=True):
        return batch["input_values"]

    def get_targets(self, batch):
        return batch["labels"]

    # Labels should be either list of possible accents or gender
    def forward(
            self,
            input_values,
            labels,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )

        loss, logits = outputs[0], outputs[1]

        return {
            'logits': logits,
            'loss': loss
        }

    def train_step(self, input_values):
        return self.forward(input_values)