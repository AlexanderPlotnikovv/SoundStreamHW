import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.quantizer import ResidualVectorQuantizer


class SoundStream(nn.Module):
    def __init__(self, C=32, K=128, N=1024, Nq=8, strides=None):
        super().__init__()
        if strides is None:
            strides = [2, 4, 5, 5]

        self.encoder = Encoder(C=C, K=K, strides=strides)
        self.rvq = ResidualVectorQuantizer(N=N, K=K, Nq=Nq)
        self.decoder = Decoder(C=C, K=K, strides=strides)

    def forward(self, x):
        z = self.encoder(x)
        z_q, idx, commit_loss = self.rvq(z)
        x_hat = self.decoder(z_q)
        return x_hat, commit_loss
