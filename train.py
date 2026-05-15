import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    """
    Main script for training SoundStream neural audio codec.
    Instantiates the model, discriminator, optimizers, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    logger.info(model)

    discriminator = instantiate(config.discriminator).to(device)
    criterion = instantiate(config.criterion).to(device)
    metrics = instantiate(config.metrics)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1
    )
    trainable_params_d = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_d = instantiate(config.optimizer_d, params=trainable_params_d)

    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        discriminator=discriminator,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
