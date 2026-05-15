import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(real_logits, fake_logits):
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss += torch.mean(torch.relu(1 - real)) + torch.mean(torch.relu(1 + fake))

    loss /= len(real_logits)
    return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_logits, fake_logits):
        loss = discriminator_loss(real_logits, fake_logits)

        return {"loss": loss, "discriminator_loss": loss}
