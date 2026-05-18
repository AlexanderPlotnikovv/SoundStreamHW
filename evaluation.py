import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer.codec_inferencer import CodecInferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.inferencer.device == "auto"
        else config.inferencer.device
    )

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    metrics = instantiate(config.metrics)

    save_path = ROOT_PATH / "predictions" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = CodecInferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    print("\n=== Final metrics ===")
    for part, part_logs in logs.items():
        for key, value in part_logs.items():
            print(f"  {part}_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
