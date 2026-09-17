"""Vendored R(2+1)D-34 from moabitcoin/ig65m-pytorch (MIT).

Source: https://github.com/moabitcoin/ig65m-pytorch (commit used by public CUHK-X 0.711 notebook).
Architecture is VideoResNet BasicBlock layers [3,4,6,3] = R(2+1)D-34, not r2plus1d_18.
"""
from __future__ import annotations

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet

MODEL_URLS = {
    "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",
    "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",
    "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",
    "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",
}


def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
    )
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    # Caffe2 / IG-65M midplanes (see pytorch/vision#1265, facebookresearch/VMZ#89)
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def r2plus1d_34_32_kinetics(num_classes, pretrained=False, progress=False):
    if pretrained and num_classes != 400:
        raise ValueError("kinetics pretrained head is 400 classes")
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_kinetics",
                       pretrained=pretrained, progress=progress)


def r2plus1d_34_8_kinetics(num_classes, pretrained=False, progress=False):
    if pretrained and num_classes != 400:
        raise ValueError("kinetics pretrained head is 400 classes")
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_8_kinetics",
                       pretrained=pretrained, progress=progress)
