import torch
import torch.nn as nn
import torchvision.ops as tops  #.drop_block as
# MFCN: multi-scale feature concat network
__all__ = ["MFCN"]

import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.


    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() / mask.sum()  # TODO input * mask * self.keep_prob ?
    

class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]


    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        feature_list = []
        # resize & concatenate
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            feature_list.append(feature_resize)

         
        features = torch.cat(feature_list, dim=1)
        
        if "prompt" in input and "aimage" not in input:
            B = features.shape[0]
            query_feat = features[:int(B/2), :, :, :]
            prompt_feat = features[int(B/2):, :, :, :]

            return {"query_feat": query_feat, 
                    "prompt_feat": prompt_feat, 
                    "outplane": self.get_outplanes()}

        if "aimage" in input and "prompt" in input:
            B = features.shape[0]
            query_feat = features[:int(B/3), :, :, :]
            synthetic_feat = features[int(B/3):int(2*B/3), :, :, :]
            prompt_feat = features[int(2*B/3):, :, :, :]

            return {"query_feat": query_feat, 
                    "synthetic_feat": synthetic_feat, 
                    "prompt_feat": prompt_feat, 
                    "outplane": self.get_outplanes()}
        
        if "aimage" not in input and "prompt" not in input:
            query_feat = features
            return {"query_feat": query_feat, 
                    "outplane": self.get_outplanes()}


    def get_outplanes(self):
        return self.outplanes


    def get_outstrides(self):
        return self.outstrides
