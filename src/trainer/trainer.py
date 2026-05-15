import torch

from src.loss.discriminator_loss import DiscriminatorLoss
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        discriminator,
        criterion,
        metrics,
        optimizer,
        optimizer_d,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        **kwargs,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            **kwargs,
        )
        self.discriminator = discriminator.to(device)
        self.optimizer_d = optimizer_d
        self.disc_loss = DiscriminatorLoss()

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        x_real = batch["audio"]

        if self.is_train:
            x_fake, commit_loss, perplexity = self.model(x_real)
            batch["x_fake"] = x_fake
            batch["x_real"] = x_real
            self.optimizer_d.zero_grad()

            wave_logits_real, wave_feats_real = self.discriminator.wave_discriminator(
                x_real
            )
            stft_logits_real = self.discriminator.stft_discriminator(x_real)
            wave_logits_fake, _ = self.discriminator.wave_discriminator(x_fake.detach())
            stft_logits_fake = self.discriminator.stft_discriminator(x_fake.detach())
            real_logits = wave_logits_real + [stft_logits_real]
            fake_logits = wave_logits_fake + [stft_logits_fake]

            d_losses = self.disc_loss(real_logits, fake_logits)
            d_losses["loss"].backward()
            self.optimizer_d.step()
            self.optimizer.zero_grad()

            wave_logits_fake, wave_feats_fake = self.discriminator.wave_discriminator(
                x_fake
            )
            stft_logits_fake = self.discriminator.stft_discriminator(x_fake)
            fake_logits = wave_logits_fake + [stft_logits_fake]
            g_losses = self.criterion(
                x_real=x_real,
                x_fake=x_fake,
                fake_logits=fake_logits,
                real_features=wave_feats_real,
                fake_features=wave_feats_fake,
                commit_loss=commit_loss,
            )
            g_losses["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()

            all_losses = {**g_losses, **d_losses}
            for loss_name, loss_val in all_losses.items():
                if loss_name in metrics.keys():
                    metrics.update(loss_name, loss_val.item())

            metrics.update("codebook_perplexity", perplexity.detach().item())
            batch["loss"] = g_losses["loss"]

        else:
            with torch.no_grad():
                x_fake, commit_loss, perplexity = self.model(x_real)
            batch["x_fake"] = x_fake
            batch["x_real"] = x_real
            batch["loss"] = torch.tensor(0.0)

            for met in self.metrics["inference"]:
                metrics.update(met.name, met(x_real=x_real, x_fake=x_fake))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        self.writer.add_audio(
            "real_audio",
            batch["x_real"][0].squeeze(0).detach().cpu(),
            sample_rate=16000,
        )
        self.writer.add_audio(
            "fake_audio",
            batch["x_fake"][0].squeeze(0).detach().cpu(),
            sample_rate=16000,
        )
