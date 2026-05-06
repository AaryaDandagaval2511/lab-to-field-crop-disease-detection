"""
model.py
========
Backbone model definitions for the Lab-to-Field Crop Disease
Domain-Generalisation Benchmark (Idea 4).

Supported architectures
-----------------------
  • efficientnet_b0   — CNN backbone (lightweight, strong ImageNet baseline)
  • mobilenet_v3_small — CPU-friendly backbone (~2.5M params, ~10x faster than efnet on CPU)
  • swin_t            — Transformer backbone (Swin-Tiny, strong spatial reasoning)

CPU vs GPU guidance
-------------------
  • On GPU (T4/A100): use efficientnet_b0 or swin_t
  • On CPU (no GPU available): use mobilenet_v3_small — it is ~10-15x faster
    than efficientnet_b0 on CPU and still achieves strong accuracy

Usage
-----
    from model import build_model, SUPPORTED_ARCHS, recommend_arch

    arch = recommend_arch()   # auto-selects based on device
    model = build_model(arch=arch, num_classes=17)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    Swin_T_Weights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
SUPPORTED_ARCHS = ("efficientnet_b0", "mobilenet_v3_small", "swin_t")

ArchType = Literal["efficientnet_b0", "mobilenet_v3_small", "swin_t"]


def recommend_arch() -> str:
    """Return the best architecture for the current hardware."""
    if torch.cuda.is_available():
        return "efficientnet_b0"
    else:
        logger.info(
            "No GPU detected — recommending mobilenet_v3_small for CPU training. "
            "It is ~10-15x faster than efficientnet_b0 on CPU."
        )
        return "mobilenet_v3_small"


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------
class PlantDiseaseClassifier(nn.Module):
    """
    Thin wrapper around a torchvision backbone that:
      1. Replaces the original classification head with a task-specific one.
      2. Exposes `feature_dim` — the penultimate embedding width.
      3. Provides `get_features()` for domain-gap computation.
      4. Provides `freeze_backbone()` / `unfreeze_backbone()` helpers.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_extractor: nn.Module,
        feature_dim: int,
        num_classes: int,
        arch: str,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._backbone = backbone

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings — shape (B, feature_dim)."""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits — shape (B, num_classes)."""
        feats = self.get_features(x)
        return self.classifier(feats)

    def freeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("[%s] Backbone frozen.", self.arch)

    def unfreeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        logger.info("[%s] Backbone unfrozen.", self.arch)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"PlantDiseaseClassifier("
            f"arch={self.arch}, "
            f"num_classes={self.num_classes}, "
            f"feature_dim={self.feature_dim}, "
            f"trainable_params={counts['trainable']:,}, "
            f"total_params={counts['total']:,})"
        )


# ---------------------------------------------------------------------------
# Architecture-specific constructors
# ---------------------------------------------------------------------------
def _build_efficientnet_b0(num_classes: int) -> PlantDiseaseClassifier:
    """
    EfficientNet-B0 backbone.
    Feature dim: 1280. Best for GPU training.
    """
    base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    feature_dim = base.classifier[1].in_features  # 1280
    feature_extractor = nn.Sequential(
        base.features,
        base.avgpool,
        nn.Flatten(start_dim=1),
    )
    return PlantDiseaseClassifier(
        backbone=base,
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=num_classes,
        arch="efficientnet_b0",
    )


def _build_mobilenet_v3_small(num_classes: int) -> PlantDiseaseClassifier:
    """
    MobileNetV3-Small backbone.
    Feature dim: 576. ~10-15x faster than EfficientNet-B0 on CPU.
    Recommended for CPU-only training (no GPU available).
    ~2.5M parameters vs EfficientNet-B0's 5.3M.
    """
    base = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # MobileNetV3-Small: features -> avgpool -> flatten -> classifier[0] (Linear+BN+Hardswish) -> classifier[3]
    # We extract after avgpool + flatten, before the classifier
    feature_dim = base.classifier[0].in_features  # 576

    feature_extractor = nn.Sequential(
        base.features,
        base.avgpool,
        nn.Flatten(start_dim=1),
    )
    return PlantDiseaseClassifier(
        backbone=base,
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=num_classes,
        arch="mobilenet_v3_small",
    )


def _build_swin_t(num_classes: int) -> PlantDiseaseClassifier:
    """
    Swin-Tiny transformer backbone.
    Feature dim: 768. Best for GPU training.
    """
    base = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    feature_dim = base.head.in_features  # 768
    feature_extractor = nn.Sequential(
        base.features,
        base.norm,
        base.permute,
        base.avgpool,
        nn.Flatten(start_dim=1),
    )
    return PlantDiseaseClassifier(
        backbone=base,
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=num_classes,
        arch="swin_t",
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def build_model(
    arch: ArchType = "efficientnet_b0",
    num_classes: int = 17,
    freeze_backbone: bool = False,
    device: torch.device | str | None = None,
) -> PlantDiseaseClassifier:
    """
    Build and return a PlantDiseaseClassifier.

    Parameters
    ----------
    arch : str
        One of SUPPORTED_ARCHS. Use recommend_arch() to auto-select.
    num_classes : int
        Output classes. Default 17 (aligned PV/PlantDoc classes).
    freeze_backbone : bool
        If True, only the head trains.
    device : torch.device | str | None
        Target device. None → auto-detect.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {SUPPORTED_ARCHS}.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info("Building model: arch=%s  num_classes=%d  device=%s", arch, num_classes, device)

    if arch == "efficientnet_b0":
        model = _build_efficientnet_b0(num_classes=num_classes)
    elif arch == "mobilenet_v3_small":
        model = _build_mobilenet_v3_small(num_classes=num_classes)
    else:
        model = _build_swin_t(num_classes=num_classes)

    if freeze_backbone:
        model.freeze_backbone()

    model = model.to(device)

    counts = model.count_parameters()
    logger.info(
        "Model ready — total: %s | trainable: %s | frozen: %s",
        f"{counts['total']:,}",
        f"{counts['trainable']:,}",
        f"{counts['frozen']:,}",
    )

    return model


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Smoke-test model.py")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=SUPPORTED_ARCHS)
    parser.add_argument("--num_classes", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Recommended arch: {recommend_arch()}")

    model = build_model(arch=args.arch, num_classes=args.num_classes, device=device)
    print(model)

    dummy = torch.randn(4, 3, 224, 224, device=device)
    logits = model(dummy)
    features = model.get_features(dummy)
    print(f"Logits shape:   {logits.shape}")
    print(f"Features shape: {features.shape}")
    print("Smoke test passed.")
