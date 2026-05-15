import torch


def collate_fn(dataset_items: list[dict]):
    return {"audio": torch.stack([item["audio"] for item in dataset_items])}
