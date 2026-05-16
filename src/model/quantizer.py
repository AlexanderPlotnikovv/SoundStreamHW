import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, N, K, decay=0.99, threshold=2.0):
        super().__init__()
        self.N = N
        self.K = K
        self.decay = decay
        self.threshold = threshold

        self.register_buffer("codebook", torch.randn(N, K))
        self.register_buffer("ema_count", torch.ones(N))
        self.register_buffer("ema_weight", torch.randn(N, K))
        self.register_buffer("initialized", torch.tensor(False))

    def _init_codebook(self, z_flat):
        n_samples = z_flat.shape[0]
        with torch.no_grad():
            if n_samples >= self.N:
                perm = torch.randperm(n_samples, device=z_flat.device)
                init = z_flat[perm[: self.N]]
            else:
                repeats = (self.N + n_samples - 1) // n_samples
                init = z_flat.repeat(repeats, 1)[: self.N]
                init = init + 0.01 * torch.randn_like(init)
            self.codebook.copy_(init)
            self.ema_weight.copy_(init)
            self.ema_count.fill_(1.0)
            self.initialized.fill_(True)

    def forward(self, z):
        B, K, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, K)
        if self.training and not self.initialized:
            self._init_codebook(z_flat)

        dist = torch.cdist(z_flat, self.codebook)
        idx = dist.argmin(dim=1)
        quantized = self.codebook[idx]
        quantized = quantized.reshape(B, T, K).permute(0, 2, 1)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(idx, minlength=self.N).float()
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * counts
                one_hot = F.one_hot(idx, self.N).float()
                new_weight = one_hot.T @ z_flat
                self.ema_weight = (
                    self.decay * self.ema_weight + (1 - self.decay) * new_weight
                )
                self.codebook = self.ema_weight / self.ema_count.unsqueeze(1).clamp(
                    min=1e-5
                )

                dead = self.ema_count < self.threshold
                if dead.any():
                    n_dead = int(dead.sum().item())
                    dists_min = torch.cdist(z_flat, self.codebook).min(dim=1).values
                    _, far_idx = torch.topk(dists_min, k=min(n_dead, len(z_flat)))
                    new_codes = z_flat[far_idx[:n_dead]]
                    self.codebook[dead] = new_codes
                    self.ema_weight[dead] = new_codes
                    self.ema_count[dead] = 1.0

        quantized_st = z + (quantized - z).detach()
        return quantized_st, quantized, idx.reshape(B, T)

    def perplexity(self, indices):
        counts = torch.bincount(indices, minlength=self.N)
        probs = counts.float() / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return torch.exp(entropy)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, N=1024, K=128, Nq=8):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer(N, K) for _ in range(Nq)])

    def forward(self, y):
        y_hat = torch.zeros_like(y)
        residual = y.clone()
        all_indices = []
        commitment_loss = 0.0
        per_layer_perplexity = []

        for quantizer in self.quantizers:
            quantized_st, quantized, idx = quantizer(residual)
            y_hat += quantized_st
            commitment_loss += F.mse_loss(residual, quantized.detach())
            residual = residual - quantized
            all_indices.append(idx)
            per_layer_perplexity.append(quantizer.perplexity(idx.flatten()))

        all_indices = torch.stack(all_indices, dim=1)
        commitment_loss = commitment_loss / len(self.quantizers)
        total_perplexity = sum(per_layer_perplexity) / len(per_layer_perplexity)
        per_layer_perplexity = torch.stack(per_layer_perplexity)

        return (
            y_hat,
            all_indices,
            commitment_loss,
            total_perplexity,
            per_layer_perplexity,
        )
