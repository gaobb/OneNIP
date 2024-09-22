import logging
import json

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader
from datasets.custom_dataset import build_onenip_dataloader


logger = logging.getLogger("global")


def build(cfg, prompt_dic, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "onenip":
        data_loader = build_onenip_dataloader(cfg, prompt_dic, training, distributed) 
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    train_loader = None
    if cfg_dataset.get("train", None):      
        meta_file = cfg_dataset["train"]["meta_file"]
        train_prompt = {} 
        with open(meta_file, "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                if meta["clsname"] not in train_prompt:
                    train_prompt[meta["clsname"]] = []
                train_prompt[meta["clsname"]].append(meta)
                
        train_loader = build(cfg_dataset, train_prompt, training=True, distributed=distributed)
                    
    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset,  train_prompt, training=False, distributed=distributed)

    logger.info("build dataset done")
    return train_loader, test_loader
