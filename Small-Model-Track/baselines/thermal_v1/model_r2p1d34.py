"""4-channel Depth_Color+IR R(2+1)D-34 (IG-65M/Kinetics stem, 40-class head)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ig65m_models import r2plus1d_34_32_kinetics

ARCH_NAME = "r2plus1d_34"
IN_CH = 4
NUM_CLASSES = 40
LAYERS = (3, 4, 6, 3)


def adapt_stem_to_4ch(model: nn.Module) -> nn.Module:
    conv = model.stem[0]
    if conv.in_channels == IN_CH:
        return model
    if conv.in_channels != 3:
        raise ValueError(f"expected 3ch stem, got {conv.in_channels}")
    new = nn.Conv3d(
        IN_CH, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    model.stem[0] = new
    return model


def build_r2p1d34_4ch(pretrained: bool = True, progress: bool = True) -> nn.Module:
    """R(2+1)D-34, 4-channel input, 40-class head. Not r2plus1d_18."""
    model = r2plus1d_34_32_kinetics(num_classes=400, pretrained=pretrained, progress=progress)
    model = adapt_stem_to_4ch(model)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, NUM_CLASSES)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)
    return model


def assert_r2p1d34_4ch(model: nn.Module) -> dict:
    stem_in = int(model.stem[0].in_channels)
    n_layer1 = len(model.layer1)
    n_layer2 = len(model.layer2)
    n_layer3 = len(model.layer3)
    n_layer4 = len(model.layer4)
    layers = (n_layer1, n_layer2, n_layer3, n_layer4)
    nparams = sum(p.numel() for p in model.parameters())
    if stem_in != IN_CH:
        raise AssertionError(f"stem in_channels={stem_in} want {IN_CH}")
    if layers != LAYERS:
        raise AssertionError(f"layers={layers} want {LAYERS} (34-layer, not 18)")
    if int(model.fc.out_features) != NUM_CLASSES:
        raise AssertionError(f"fc={model.fc.out_features}")
    if nparams < 50_000_000:
        raise AssertionError(f"param count {nparams} looks like r2plus1d_18 not 34")
    return {
        "arch": ARCH_NAME,
        "in_ch": stem_in,
        "layers": list(layers),
        "nparams": int(nparams),
        "fc": int(model.fc.out_features),
    }


def pack_int8(state: dict) -> dict:
    packed = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.numel() >= 64:
            scale = v.detach().abs().amax().clamp_min(1e-8) / 127.0
            q = torch.clamp((v.detach() / scale).round(), -128, 127).to(torch.int8)
            packed[k] = {"q": q.cpu(), "scale": scale.cpu().float(), "shape": list(v.shape)}
        else:
            packed[k] = v.cpu() if torch.is_tensor(v) else v
    return packed


def unpack_int8(packed: dict) -> dict:
    out = {}
    for k, v in packed.items():
        if isinstance(v, dict) and "q" in v and "scale" in v:
            out[k] = v["q"].float() * v["scale"]
        else:
            out[k] = v
    return out
