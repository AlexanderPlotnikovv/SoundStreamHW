import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, N, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(N, N, kernel_size=7, dilation=dilation, padding=3 * dilation),
            nn.ELU(),
            nn.Conv1d(N, N, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)
