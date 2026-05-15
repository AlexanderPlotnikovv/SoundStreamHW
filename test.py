import torch

from src.model.soundstream import SoundStream

model = SoundStream(C=32, K=128, N=1024, Nq=8)
x = torch.randn(2, 1, 8000)
x_hat, commit_loss = model(x)
print(x_hat.shape)
print(commit_loss)
