import torch.nn as nn

from src.model.residual_unit import ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(N // 2, dilation=1),
            ResidualUnit(N // 2, dilation=3),
            ResidualUnit(N // 2, dilation=9),
            nn.ELU(),
            nn.Conv1d(N // 2, N, kernel_size=2 * S, stride=S, padding=S - 1),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, C=32, K=128, strides=None):
        super().__init__()

        if strides is None:
            strides = [2, 4, 5, 5]
        layers = [nn.Conv1d(1, C, kernel_size=7, padding=3)]

        channels = C
        for stride in strides:
            channels *= 2
            layers.append(EncoderBlock(N=channels, S=stride))

        layers += [nn.ELU(), nn.Conv1d(channels, K, kernel_size=3, padding=1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
