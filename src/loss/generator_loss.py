import torch
import torch.nn as nn
import torch.nn.functional as F


def adv_loss(fake_logits):
    loss = 0.0
    for fake in fake_logits:
        loss += torch.mean(torch.relu(1 - fake))
    return loss / len(fake_logits)


def feat_loss(real_features, fake_features):
    loss = 0.0
    K = len(real_features)
    for real_maps, fake_maps in zip(real_features, fake_features):
        L = len(real_maps)
        disc_loss = 0.0
        for real_feat, fake_feat in zip(real_maps, fake_maps):
            disc_loss += torch.mean(torch.abs(real_feat.detach() - fake_feat))
        loss += disc_loss / L
    return loss / K


def rec_loss(x_real, x_fake, alpha=1.0):
    fft_sizes = [2**i for i in range(6, 12)]
    device = x_real.device
    loss = 0.0

    for s in fft_sizes:
        hop = s // 4
        window = torch.hann_window(s, device=device)

        real_spec = torch.stft(
            x_real.squeeze(1),
            n_fft=s,
            hop_length=hop,
            win_length=s,
            window=window,
            return_complex=True,
        ).abs()
        fake_spec = torch.stft(
            x_fake.squeeze(1),
            n_fft=s,
            hop_length=hop,
            win_length=s,
            window=window,
            return_complex=True,
        ).abs()

        loss += F.l1_loss(real_spec, fake_spec)
        loss += alpha * F.mse_loss(
            torch.log(real_spec + 1e-7),
            torch.log(fake_spec + 1e-7),
        )

    return loss / len(fft_sizes)


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_commit=1.0, lambda_feat=100.0):
        super().__init__()
        self.lambda_commit = lambda_commit
        self.lambda_feat = lambda_feat

    def forward(
        self, x_real, x_fake, fake_logits, real_features, fake_features, commit_loss
    ):
        l_rec = rec_loss(x_real, x_fake)
        l_adv = adv_loss(fake_logits)
        l_feat = feat_loss(real_features, fake_features)

        loss = (
            l_adv + self.lambda_feat * l_feat + l_rec + self.lambda_commit * commit_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": l_rec,
            "adversarial_loss": l_adv,
            "feature_loss": l_feat,
            "commitment_loss": commit_loss,
        }
