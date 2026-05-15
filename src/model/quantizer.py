import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.K = K
        self.codebook = nn.Embedding(N, K)

    def forward(self, z):
        B, K, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, K)

        z_norm = (z_flat**2).sum(dim=1, keepdim=True)
        codebook_norm = (self.codebook.weight**2).sum(dim=1)
        dot = z_flat @ self.codebook.weight.T
        dist = z_norm + codebook_norm - 2 * dot

        idx = dist.argmin(dim=1)
        quantized = self.codebook(idx)
        quantized = quantized.reshape(B, T, K).permute(0, 2, 1)

        quantized_st = z + (quantized - z).detach()

        return quantized_st, quantized, idx.reshape(B, T)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, N=1024, K=128, Nq=8):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer(N, K) for _ in range(Nq)])

    def forward(self, y):
        y_hat = torch.zeros_like(y)
        residual = y.clone()
        all_indices = []
        commitment_loss = 0.0

        for quantizer in self.quantizers:
            quantized_st, quantized, idx = quantizer(residual)
            y_hat += quantized_st
            residual = residual - quantized
            all_indices.append(idx)
            commitment_loss += F.mse_loss(residual.detach(), quantized)

        all_indices = torch.stack(all_indices, dim=1)
        commitment_loss = commitment_loss / len(self.quantizers)

        return y_hat, all_indices, commitment_loss
