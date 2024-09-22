from __future__ import division

import json
import logging

import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.base_dataset import TestSupportTransform, TrainSupportTransform 
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from datasets.cutpaste import CutPaste, PerlinPaste

logger = logging.getLogger("global_logger")


def build_custom_dataloader(cfg, training, distributed=True):
    if "class_names" in cfg:
        cls2label_dict = {class_name: i for i, class_name in enumerate(cfg["class_names"])}
    else:
        cls2label_dict = {}

    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        cls2label_dict,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        cls2label_dict,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        self.cls2label_dict = cls2label_dict

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]
 
        if input["clsname"] in self.cls2label_dict:
            clslabel = self.cls2label_dict[input["clsname"]] 
        else:
            clslabel = 0

        input.update({"class_labels": clslabel})
        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask})
        return input


def build_onenip_dataloader(cfg, prompt_dict, training, distributed=True):
    if "class_names" in cfg:
        cls2label_dict = {class_name: i for i, class_name in enumerate(cfg["class_names"])}
    else:
        cls2label_dict = {}
    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = OneNIPDataset(
        cls2label_dict,
        image_reader,
        prompt_dict,
        cfg["meta_file"],
        cfg["dtd_dir"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class OneNIPDataset(BaseDataset):
    #cutpaste_style: 'hybrid', 'cutpaste', "hybrid"
    def __init__(
        self,
        cls2label_dict,
        image_reader,
        prompt_dict,
        meta_file,
        dtd_dir,
        training,
        transform_fn,
        normalize_fn,
        cutpaste_style='hybrid', 
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.prompt_dict = prompt_dict
        self.meta_file = meta_file
        self.dtd_dir = dtd_dir
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        self.cls2label_dict = cls2label_dict
        self.cutpaste_style = cutpaste_style
        self.cutpaste_transform = CutPaste()
        self.perlinpaste_transform = PerlinPaste(self.dtd_dir)
        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]
        
        if input["clsname"] in self.cls2label_dict:
            clslabel = self.cls2label_dict[input["clsname"]] 
        else:
            clslabel = 0

        input.update({"class_labels": clslabel})

        image = Image.fromarray(image, "RGB")

        # read prompt image
        prompt_meta = random.choice(self.prompt_dict[meta["clsname"]])
        # fixed
        # prompt_meta = self.support_dict[meta["clsname"]][0]

        #prompt_meta = self.support_dict["metal_nut"][0]

        #prompt_meta = random.choice(self.support_dict["metal_nut"])
        #prompt_meta = self.support_dict["screw"][0]

        prompt = self.image_reader(prompt_meta["filename"])
        prompt = Image.fromarray(prompt, "RGB")
         
        if prompt_meta.get("maskname", None):
            pmask = self.image_reader(prompt_meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                pmask = np.zeros((prompt.height, prompt.width)).astype(np.uint8)
            elif label == 1:  # defective
                pmask = (np.ones((prompt.height, prompt.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")
        pmask = Image.fromarray(pmask, "L")

        # read / generate mask
        if meta.get("maskname", None):
            imask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                imask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                imask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")
        imask = Image.fromarray(imask, "L")

        if self.transform_fn:
            image, imask = self.transform_fn(image, imask)
            prompt, pmask = self.transform_fn(prompt, pmask)

        if self.training:
           # cutpaste paper
            if self.cutpaste_style == "cutpaste":
                aimage, amask = self.cutpaste_transform(image, imask)
           # dream paper
            if self.cutpaste_style == "dream":
                aimage, amask = self.perlinpaste_transform(np.array(image), np.array(imask))
                aimage, amask = Image.fromarray(np.uint8(aimage)), Image.fromarray(np.uint8(amask))

           # hybrid cutpaste and dream
            if self.cutpaste_style == "hybrid":
                if torch.rand(1) > 0.5:
                    aimage, amask = self.cutpaste_transform(image, imask)
                else:
                    aimage, amask = self.perlinpaste_transform(np.array(image), np.array(imask))
                    aimage, amask = Image.fromarray(np.uint8(aimage)), Image.fromarray(np.uint8(amask))

        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
            prompt = self.colorjitter_fn(prompt)
        if self.training and self.colorjitter_fn:
            aimage = self.colorjitter_fn(aimage)

        prompt = transforms.ToTensor()(prompt)    
        image = transforms.ToTensor()(image)
        imask = transforms.ToTensor()(imask)

        if self.training:
            aimage = transforms.ToTensor()(aimage)
            amask = transforms.ToTensor()(amask)

        if self.normalize_fn:
            image = self.normalize_fn(image)
            prompt = self.normalize_fn(prompt) 
        if self.training and self.normalize_fn:
            aimage = self.normalize_fn(aimage)

        if self.training:
            input.update({"prompt": prompt, "image": image, "mask": imask, "aimage": aimage, "amask": amask})
        else:
            input.update({"prompt": prompt, "image": image, "mask": imask})

        return input
