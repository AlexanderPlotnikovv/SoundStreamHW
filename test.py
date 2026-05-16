import torch

from src.model.soundstream import SoundStream

model = SoundStream()
model.eval()

for L in [8000, 16000, 32000, 80000]:
    x = torch.randn(1, 1, L)
    with torch.no_grad():
        x_hat, _, _ = model(x)
    print(f"input {L:>6} → output {x_hat.shape[-1]:>6}, diff={x_hat.shape[-1] - L}")
