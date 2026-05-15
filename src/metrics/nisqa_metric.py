import torch
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

from src.metrics.base_metric import BaseMetric


class NISQAMetric(BaseMetric):
    def __init__(self, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate)

    def __call__(self, x_fake, **batch):
        x_fake = x_fake.squeeze(1)
        scores = []
        for fake in x_fake:
            result = self.metric(fake.cpu())
            scores.append(result[0].item())
        return sum(scores) / len(scores)
