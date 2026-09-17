from pathlib import Path

path = Path("train_earlyfuse_v1.py")
text = path.read_text(encoding="utf-8")

old_doc = '"""Early-fuse IR+Depth: dual-stem R(2+1)D-18 or 6ch shared backbone.\nReuses cache/ir_yolo_v4 + cache/depth_color_yolo_v4_irbox (index-aligned).\nGate vs ir_v7: hold AND nested >= 0.763 and >=20 disagrees; else status JSON only.\n"""'
new_doc = '"""Early-fuse IR+Depth: dual / ch6 / shared(siamese) / hybrid(IR+compact-depth).\nReuses cache/ir_yolo_v4 + cache/depth_color_yolo_v4_irbox (index-aligned).\nSize: dual~120MB ILLEGAL; shared/hybrid/ch6 fp16 ~60MB (+yolo~66) LEGAL.\nGate vs ir_v7: hold AND nested >= 0.763 and >=20 disagrees; else status JSON only.\n"""'
if old_doc not in text:
    raise SystemExit("docstring not found")
text = text.replace(old_doc, new_doc, 1)

needle = "from dataset import DEFAULT_HOLD_OUT_USERS, NUM_CLASSES\n"
insert_imp = needle + "from model_r2p1d import CompactR2Plus1D\n"
if "from model_r2p1d import CompactR2Plus1D" not in text:
    if needle not in text:
        raise SystemExit("import needle missing")
    text = text.replace(needle, insert_imp, 1)

shared_cls = '''
class SharedStemR2P1D(nn.Module):
    """Weight-tied R(2+1)D-18: same stem for IR+Depth, concat feats -> head.
    fp16 pack ~60MB (legal). Optional 1x1 modality adapters (~few KB).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4, adapters: bool = True):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=w)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.use_adapters = adapters
        if adapters:
            self.ir_adapt = nn.Sequential(nn.Conv3d(3, 3, 1, bias=False), nn.BatchNorm3d(3))
            self.dp_adapt = nn.Sequential(nn.Conv3d(3, 3, 1, bias=False), nn.BatchNorm3d(3))
            with torch.no_grad():
                self.ir_adapt[0].weight.zero_()
                self.dp_adapt[0].weight.zero_()
                for i in range(3):
                    self.ir_adapt[0].weight[i, i, 0, 0, 0] = 1.0
                    self.dp_adapt[0].weight[i, i, 0, 0, 0] = 1.0
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat * 2, NUM_CLASSES)

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        if self.use_adapters:
            x_ir = self.ir_adapt(x_ir)
            x_dp = self.dp_adapt(x_dp)
        f = torch.cat([self.backbone(x_ir), self.backbone(x_dp)], dim=1)
        return self.head(self.drop(f))

    def freeze_early(self, which: str = "stem_l1") -> None:
        for name, p in self.backbone.named_parameters():
            if which == "all_but_head":
                p.requires_grad = False
            elif which == "stem_l1":
                p.requires_grad = not any(k in name for k in ["stem", "layer1"])
            elif which == "stem_l1_l2":
                p.requires_grad = not any(k in name for k in ["stem", "layer1", "layer2"])
            else:
                p.requires_grad = True
        for mod in (getattr(self, "ir_adapt", None), getattr(self, "dp_adapt", None), self.head):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class HybridIRCompactDepth(nn.Module):
    """Full Kinetics IR stem + tiny CompactR2Plus1D depth branch; feature concat.
    fp16 pack ~61MB (legal). Depth capacity limited on purpose for size/VRAM.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4, depth_base: int = 32):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.ir = r2plus1d_18(weights=w)
        feat_ir = self.ir.fc.in_features
        self.ir.fc = nn.Identity()
        self.dp = CompactR2Plus1D(num_classes=NUM_CLASSES, in_ch=3, base=depth_base, dropout=dropout)
        feat_dp = depth_base * 8
        self.dp.head = nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_ir + feat_dp, NUM_CLASSES)

    def _dp_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dp.stem(x)
        x = self.dp.layer1(x)
        x = self.dp.layer2(x)
        x = self.dp.layer3(x)
        return self.dp.pool(x).flatten(1)

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.ir(x_ir), self._dp_feat(x_dp)], dim=1)
        return self.head(self.drop(f))

    def freeze_early(self, which: str = "stem_l1") -> None:
        for name, p in self.ir.named_parameters():
            if which == "all_but_head":
                p.requires_grad = False
            elif which == "stem_l1":
                p.requires_grad = not any(k in name for k in ["stem", "layer1"])
            elif which == "stem_l1_l2":
                p.requires_grad = not any(k in name for k in ["stem", "layer1", "layer2"])
            else:
                p.requires_grad = True
        for p in self.dp.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


'''

anchor = "def build_model(mode: str, pretrained: bool = True) -> nn.Module:\n"
if "class SharedStemR2P1D" not in text:
    if anchor not in text:
        raise SystemExit("build_model anchor missing")
    text = text.replace(anchor, shared_cls + anchor, 1)

old_build = '''def build_model(mode: str, pretrained: bool = True) -> nn.Module:
    if mode == "dual":
        return DualStemR2P1D(pretrained=pretrained)
    if mode == "ch6":
        return Channel6R2P1D(pretrained=pretrained)
    raise ValueError(mode)
'''
new_build = '''def build_model(mode: str, pretrained: bool = True) -> nn.Module:
    if mode == "dual":
        return DualStemR2P1D(pretrained=pretrained)
    if mode == "ch6":
        return Channel6R2P1D(pretrained=pretrained)
    if mode == "shared":
        return SharedStemR2P1D(pretrained=pretrained, adapters=True)
    if mode == "hybrid":
        return HybridIRCompactDepth(pretrained=pretrained, depth_base=32)
    raise ValueError(mode)
'''
if old_build not in text:
    raise SystemExit("build_model body not found")
text = text.replace(old_build, new_build, 1)

text = text.replace(
    'ap.add_argument("--mode", choices=["dual", "ch6"], default="dual")',
    'ap.add_argument("--mode", choices=["dual", "ch6", "shared", "hybrid"], default="shared")',
    1,
)
text = text.replace(
    '"fp16_pack_note": "dual ~2x r2plus1d18 fp16; check size_ok after pack"',
    '"fp16_pack_note": "shared/hybrid/ch6 ~60MB fp16 legal; dual~120MB illegal — prefer shared"',
    1,
)
text = text.replace(
    'ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "earlyfuse_ir_depth_v1"))',
    'ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "sharedstem_ir_depth_v1"))',
    1,
)

path.write_text(text, encoding="utf-8")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("patched ok", path.stat().st_size)
