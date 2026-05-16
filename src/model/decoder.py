import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.model.residual_unit import ResidualUnit


class DecoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
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
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, C=32, K=128, strides=None):
        super().__init__()

        if strides is None:
            strides = [2, 4, 5, 5]

        layers = [
            weight_norm(nn.Conv1d(K, (2 ** len(strides)) * C, kernel_size=7, padding=3))
        ]

        channels = (2 ** len(strides)) * C
        for stride in reversed(strides):
            layers.append(DecoderBlock(N=channels, S=stride))
            channels //= 2

        layers += [
            nn.ELU(),
            weight_norm(nn.Conv1d(C, 1, kernel_size=7, padding=3)),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x, target_length=None):
        out = self.net(x)
        if target_length is not None:
            if out.shape[-1] > target_length:
                out = out[:, :, :target_length]
            elif out.shape[-1] < target_length:
                out = F.pad(out, (0, target_length - out.shape[-1]))
        return out
