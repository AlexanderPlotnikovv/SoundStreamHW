import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)

    def __call__(self, x_real, x_fake, **batch):
        x_real = x_real.squeeze(1)
        x_fake = x_fake.squeeze(1)
        scores = []
        for real, fake in zip(x_real, x_fake):
            scores.append(self.metric(fake.cpu(), real.cpu()).item())
        return sum(scores) / len(scores)
