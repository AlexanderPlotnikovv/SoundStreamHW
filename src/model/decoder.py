import torch.nn as nn
from torch.nn.utils import weight_norm

from src.model.residual_unit import ResidualUnit


class DecoderBlock(nn.Module):
    def __init__(self, N, S, length):
        super().__init__()
        self.target_length = length * S
        self.net = nn.Sequential(
            nn.ELU(),
            weight_norm(
                nn.ConvTranspose1d(
                    N, N // 2, kernel_size=2 * S, stride=S, padding=S // 2
                )
            ),
            ResidualUnit(N // 2, 1),
            ResidualUnit(N // 2, 3),
            ResidualUnit(N // 2, 9),
        )

    def forward(self, x):
        return self.net(x)[:, :, : self.target_length]


class Decoder(nn.Module):
    def __init__(self, C=32, K=128, length=8000, strides=None):
        super().__init__()

        if strides is None:
            strides = [2, 4, 5, 5]

        layers = [
            weight_norm(nn.Conv1d(K, (2 ** len(strides)) * C, kernel_size=7, padding=3))
        ]

        channels = (2 ** len(strides)) * C
        target_lengths = []
        reverse_strides = list(reversed(strides))
        target_length = length
        for stride in strides:
            target_length //= stride
            target_lengths.append(target_length)
        target_lengths = list(reversed(target_lengths))
        for stride, cur_length in zip(reverse_strides, target_lengths):
            layers.append(DecoderBlock(N=channels, S=stride, length=cur_length))
            channels //= 2

        layers += [
            nn.ELU(),
            weight_norm(nn.Conv1d(C, 1, kernel_size=7, padding=3)),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
