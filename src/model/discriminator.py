import torch
import torch.nn as nn

from src.model.stft_discriminator import STFTDiscriminator
from src.model.wave_discriminator import WaveDiscriminator


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_discriminator = STFTDiscriminator()
        self.wave_discriminator = WaveDiscriminator()

    def forward(self, x):
        stft_logits = self.stft_discriminator(x)
        wave_logits = self.wave_discriminator(x)
        return stft_logits, wave_logits
