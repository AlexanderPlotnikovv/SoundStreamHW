import torch
import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    def __init__(
        self, root_dir, url, crop=0.5, sample_rate=16000, download=False, limit=None
    ):
        """
        Args:
            root (str): path to dataset root
            url (str): which split, e.g. "train-clean-100" or "test-clean"
            crop_seconds (float): length of random crop in seconds
            sample_rate (int): target sample rate
            download (bool): download if not present
        """
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root_dir,
            url=url,
            download=download,
        )

        self.crop_samples = int(crop * sample_rate)
        self.sample_rate = sample_rate
        self.crop = crop
        self.resampler = {}

        if limit is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, range(limit))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, _, _, _, _ = self.dataset[idx]

        if sample_rate != self.sample_rate:
            if sample_rate not in self.resampler:
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )
            waveform = self.resampler[sample_rate](waveform)

        if waveform.shape[1] >= self.crop_samples:
            start = torch.randint(
                0, waveform.shape[1] - self.crop_samples + 1, (1,)
            ).item()
            waveform = waveform[:, start : start + self.crop_samples]
        else:
            repeats = (self.crop_len // waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeats)
            waveform = waveform[:, : self.crop_samples]

        return {"audio": waveform, "sample_rate": sample_rate}
