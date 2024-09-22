import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, class_name, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, class_name, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, class_name, distributed=True):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, class_name, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, class_name, training=False, distributed=distributed)

    logger.info("build dataset done")
    return train_loader, test_loader
