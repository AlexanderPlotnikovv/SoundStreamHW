import torch
import torch.nn as nn


class STFTResidualBlock(nn.Module):
    def __init__(self, N, m, stride):
        super().__init__()
        st, sf = stride
        kernel = (st + 2, sf + 2)

        self.net = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                N, m * N, kernel_size=kernel, stride=stride, padding=(st // 2, sf // 2)
            ),
            nn.LeakyReLU(0.2),
        )
        self.skip = nn.Conv2d(N, m * N, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.start_transform = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
        )

        self.residual_net = nn.Sequential(
            STFTResidualBlock(N=C, m=2, stride=(1, 2)),
            STFTResidualBlock(N=2 * C, m=2, stride=(2, 2)),
            STFTResidualBlock(N=4 * C, m=1, stride=(1, 2)),
            STFTResidualBlock(N=4 * C, m=2, stride=(2, 2)),
            STFTResidualBlock(N=8 * C, m=1, stride=(1, 2)),
            STFTResidualBlock(N=8 * C, m=2, stride=(2, 2)),
        )

        self.end_transform = nn.Conv2d(16 * C, 1, kernel_size=(1, 8))

    def compute_shift(self, x):
        x = x.squeeze(1)
        stft = torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024).to(x.device),
            return_complex=True,
        )

        x = torch.stack([stft.real, stft.imag], dim=1).permute(0, 1, 3, 2)
        return x

    def forward(self, x):
        x = self.compute_shift(x)
        x = self.start_transform(x)
        x = self.residual_net(x)
        x = self.end_transform(x)
        x = x.squeeze(-1).squeeze(1)
        return x
