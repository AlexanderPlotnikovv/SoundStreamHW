import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class STFTResidualBlock(nn.Module):
    def __init__(self, N, m, stride):
        super().__init__()
        st, sf = stride
        kernel = (st + 2, sf + 2)

        self.net = nn.Sequential(
            weight_norm(nn.Conv2d(N, N, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            weight_norm(
                nn.Conv2d(
                    N,
                    m * N,
                    kernel_size=kernel,
                    stride=stride,
                    padding=(st // 2, sf // 2),
                )
            ),
            nn.LeakyReLU(0.2),
        )
        self.skip = nn.Conv2d(N, m * N, kernel_size=1, stride=stride)

    def forward(self, x):
        h = self.net(x)
        s = self.skip(x)
        min_t = min(h.shape[2], s.shape[2])
        min_f = min(h.shape[3], s.shape[3])
        return h[:, :, :min_t, :min_f] + s[:, :, :min_t, :min_f]


class STFTDiscriminator(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.register_buffer("window", torch.hann_window(1024))
        self.start_transform = nn.Sequential(
            weight_norm(nn.Conv2d(2, C, kernel_size=7, padding=3)),
            nn.LeakyReLU(0.2),
        )
        self.residual_blocks = nn.ModuleList(
            [
                STFTResidualBlock(N=C, m=2, stride=(1, 2)),
                STFTResidualBlock(N=2 * C, m=2, stride=(2, 2)),
                STFTResidualBlock(N=4 * C, m=1, stride=(1, 2)),
                STFTResidualBlock(N=4 * C, m=2, stride=(2, 2)),
                STFTResidualBlock(N=8 * C, m=1, stride=(1, 2)),
                STFTResidualBlock(N=8 * C, m=2, stride=(2, 2)),
            ]
        )
        self.end_transform = weight_norm(nn.Conv2d(16 * C, 1, kernel_size=(1, 8)))

    def compute_stft(self, x):
        x = x.squeeze(1)
        stft = torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.window,
            return_complex=True,
        )
        x = torch.stack([stft.real, stft.imag], dim=1)
        return x.permute(0, 1, 3, 2)

    def forward(self, x):
        x = self.compute_stft(x)
        x = self.start_transform(x)
        feature_maps = []
        for block in self.residual_blocks:
            x = block(x)
            feature_maps.append(x)
        x = self.end_transform(x)
        logit = x.squeeze(-1).squeeze(1)
        return logit, feature_maps
