import torch
import torchaudio
from tqdm.auto import tqdm

from src.trainer.inferencer import Inferencer


class CodecInferencer(Inferencer):
    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        x_real = batch["audio"]
        x_fake, _, _, _ = self.model(x_real)
        batch["x_real"] = x_real
        batch["x_fake"] = x_fake

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(x_real=x_real, x_fake=x_fake))

        if self.save_path is not None:
            batch_size = x_real.shape[0]
            current_id = batch_idx * batch_size
            for i in range(batch_size):
                output_id = current_id + i
                real_path = self.save_path / part / f"real_{output_id}.wav"
                fake_path = self.save_path / part / f"fake_{output_id}.wav"
                torchaudio.save(str(real_path), x_real[i].cpu(), 16000)
                torchaudio.save(str(fake_path), x_fake[i].cpu(), 16000)

        return batch
