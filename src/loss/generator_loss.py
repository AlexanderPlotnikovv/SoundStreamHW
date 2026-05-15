import torch
import torch.nn as nn
import torch.nn.functional as F


def adv_loss(fake_logits):
    loss = 0.0
    for fake in fake_logits:
        loss += torch.mean(torch.relu(1 - fake))

    loss /= len(fake_logits)
    return loss


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
    loss = 0.0

    for s in fft_sizes:
        hop = s // 4
        win = s
        window = torch.hann_window(win)
        real_cpu = x_real.squeeze(1).cpu()
        fake_cpu = x_fake.squeeze(1).cpu()

        real_spec = torch.stft(
            real_cpu,
            n_fft=s,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
        ).abs()

        fake_spec = torch.stft(
            fake_cpu,
            n_fft=s,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
        ).abs()

        real_spec = real_spec.to(x_real.device)
        fake_spec = fake_spec.to(x_fake.device)
        loss += torch.mean(torch.abs(real_spec - fake_spec))
        loss += alpha * torch.mean(
            (torch.log(real_spec + 1e-7) - torch.log(fake_spec + 1e-7)) ** 2
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
