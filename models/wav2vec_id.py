from torch import nn


class Wav2VecIDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output, targets):
        return nn.functional.cross_entropy(model_output, targets)


class Wav2VecID(nn.Module):
    def __init__(self, embed_size, num_accents):
        super().__init__()
        self.output_layer = nn.Linear(embed_size, num_accents)

    def parse_batch(self, batch, train=True):
        return batch["input_values"]

    def get_targets(self, batch):
        return batch["labels"]

    def forward(
            self,
            accent_embed
    ):
        output = nn.output_layer(accent_embed)

        return output

    def train_step(self, input_values):
        return self.forward(input_values)