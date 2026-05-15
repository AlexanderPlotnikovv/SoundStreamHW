import torch.nn as nn
from torch.nn.utils import weight_norm


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, kernel_size=15, padding=7)),
                weight_norm(
                    nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)
                ),
                weight_norm(
                    nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)
                ),
                weight_norm(
                    nn.Conv1d(
                        256, 1024, kernel_size=41, stride=4, groups=64, padding=20
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        1024, 1024, kernel_size=41, stride=4, groups=256, padding=20
                    )
                ),
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, padding=2)),
                weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1)),
            ]
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
            feature_maps.append(x)
        return x.squeeze(1), feature_maps


class WaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        logits = []
        feature_maps = []

        for i, disc in enumerate(self.discriminators):
            logit, features = disc(x)
            logits.append(logit)
            feature_maps.append(features)
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)

        return logits, feature_maps
