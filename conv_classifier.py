import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, BatchNorm1d, ReLU, MaxPool1d, Linear
import pytorch_lightning as pl


class Conv1DClassifier(pl.LightningModule):
    def __init__(self, num_channels_in, num_classes):
        super().__init__()
        # TODO: how to choose these params?
        self.cnn_layers = nn.Sequential(
            # Defining a 1D convolution layer
            Conv1d(num_channels_in, 32, kernel_size=3, stride=2),
            BatchNorm1d(32),
            ReLU(inplace=True),
            # Defining another 1D convolution layer
            Conv1d(32, 64, kernel_size=3, stride=2),
            BatchNorm1d(64),
            ReLU(inplace=True)
        )

        self.linear_layer = nn.Sequential(
            Linear(64, num_classes)
        )

    # Takes in the mel_spectrogram, which is (batch_size, n_mels, time)
    def forward(self, x):
        x = self.cnn_layers(x)
        # Average along the time dimension
        x = torch.mean(x, 2)
        x = self.linear_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        # y is a categorical label for the accent
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)





