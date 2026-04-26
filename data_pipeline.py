"""
data_pipeline.py
================
Production-quality data pipeline for the Lab-to-Field Crop Disease
Domain-Generalisation Benchmark (Idea 4).

Source domain : PlantVillage  (lab / controlled background)
Target domain : PlantDoc      (semi-field / noisy background)
Shared labels : 21 aligned canonical disease classes

ROOT CAUSE FIX (v2)
-------------------
The 4931-second epoch was caused by reading images DIRECTLY FROM GOOGLE DRIVE
at training time. Google Drive has ~50-100 ms latency per file open. With
~42k training images, that's 35-70 minutes of pure I/O per epoch.

Fix: cache_dataset_to_local() copies the dataset from Drive to Colab's local
SSD (/content/data_cache/) ONCE at startup. All subsequent reads hit fast
local disk (<1ms latency).

IMPORTANT: If the source path is already under /content/ (i.e. already on
local SSD — as is the case when Cell 6 builds final_dataset directly into
/content/final_dataset/), the copy is SKIPPED and the source path is returned
directly. This prevents the FileNotFoundError that occurred when train() tried
to copy /content/final_dataset/train from Google Drive (where it doesn't exist).

Expected epoch time after fix:
  * EfficientNet-B0, batch_size=64, T4 GPU: ~1.5-2.5 min/epoch
  * 15 epochs total: ~25-40 min
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224

# 21 canonical disease classes
CANONICAL_CLASSES: list[str] = sorted([
    "apple_cedar_rust",
    "apple_black_rot",
    "apple_scab",
    "corn_common_rust",
    "corn_gray_leaf_spot",
    "corn_northern_leaf_blight",
    "grape_black_rot",
    "grape_leaf_blight",
    "potato_early_blight",
    "potato_late_blight",
    "squash_powdery_mildew",
    "strawberry_leaf_scorch",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_yellow_leaf_curl_virus",
])

CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
NUM_CLASSES:  int = len(CANONICAL_CLASSES)

_SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini", ".thumbs"}


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL SSD CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _is_local_path(path: Path) -> bool:
    """
    Return True if path is already on Colab's local filesystem (not Drive).

    Google Drive is always mounted under /content/drive/.
    Anything else under /content/ is local SSD.
    """
    path_str = str(path.resolve())
    # If it's NOT under /content/drive, it's already local
    return not path_str.startswith("/content/drive")


def cache_dataset_to_local(
    src_train_dir: str | Path,
    src_val_dir:   str | Path,
    local_cache:   str | Path = "/content/data_cache",
    force_recopy:  bool = False,
) -> tuple[Path, Path]:
    """
    Ensure dataset is on Colab's local SSD for fast training I/O.

    SMART BEHAVIOUR:
    ----------------
    - If src paths are already local (e.g. /content/final_dataset/train built
      by Cell 6), returns them AS-IS — no copy needed, no FileNotFoundError.
    - If src paths are on Google Drive (/content/drive/...), copies to
      local_cache once per session (skips on subsequent calls).

    WHY THIS EXISTS:
    ----------------
    Google Drive has ~50-100ms open-latency per file. Colab local SSD has
    <1ms. With ~42k images, Drive costs 35-70 min/epoch; local SSD costs
    ~1.5-2.5 min/epoch.

    Parameters
    ----------
    src_train_dir : str | Path   Training directory (Drive or local path).
    src_val_dir   : str | Path   Validation directory (Drive or local path).
    local_cache   : str | Path   Where to copy if source is on Drive.
    force_recopy  : bool         Re-copy even if local cache exists.

    Returns
    -------
    (local_train_dir, local_val_dir) — guaranteed to be on local SSD.
    """
    src_train = Path(src_train_dir)
    src_val   = Path(src_val_dir)

    # ── FAST PATH: source is already local ────────────────────────────────────
    # Cell 6 builds final_dataset directly into /content/ — no copy needed.
    if _is_local_path(src_train) and not force_recopy:
        if not src_train.exists():
            raise FileNotFoundError(
                f"Training directory not found: {src_train}\n"
                "  → Re-run Cell 6 to rebuild final_dataset (session may have reset)."
            )
        if not src_val.exists():
            raise FileNotFoundError(
                f"Validation directory not found: {src_val}\n"
                "  → Re-run Cell 6 to rebuild final_dataset (session may have reset)."
            )
        n_train = sum(1 for _ in src_train.rglob("*") if _.is_file())
        n_val   = sum(1 for _ in src_val.rglob("*")   if _.is_file())
        logger.info(
            "Source dataset is already on local SSD — skipping copy.\n"
            "  train: %s  (%d files)\n"
            "  val  : %s  (%d files)",
            src_train, n_train, src_val, n_val,
        )
        return src_train, src_val

    # ── COPY PATH: source is on Google Drive ──────────────────────────────────
    cache_dir   = Path(local_cache)
    local_train = cache_dir / "train"
    local_val   = cache_dir / "val"

    def _needs_copy(src: Path, dst: Path) -> bool:
        if force_recopy:
            return True
        if not dst.exists():
            return True
        src_count = sum(1 for _ in src.rglob("*") if _.is_file())
        dst_count = sum(1 for _ in dst.rglob("*") if _.is_file())
        if dst_count < src_count:
            logger.warning(
                "Cache incomplete (%d/%d files). Re-copying.", dst_count, src_count,
            )
            return True
        return False

    for src, dst, split in [
        (src_train, local_train, "train"),
        (src_val,   local_val,   "val"),
    ]:
        if _needs_copy(src, dst):
            if dst.exists():
                shutil.rmtree(dst)
            logger.info("Copying %s split: %s → %s", split, src, dst)
            t0 = time.time()
            shutil.copytree(src, dst)
            elapsed = time.time() - t0
            n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
            logger.info("  Done: %d files in %.1fs", n_files, elapsed)
        else:
            n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
            logger.info("Cache hit for %s (%d files) — skipping copy.", split, n_files)

    return local_train, local_val


def cache_single_dir_to_local(
    src_dir:      str | Path,
    local_cache:  str | Path = "/content/data_cache",
    subdir_name:  str = "plantdoc_val",
    force_recopy: bool = False,
) -> Path:
    """
    Copy a single dataset directory (e.g. PlantDoc val) to local SSD.
    If source is already local, returns it as-is.
    """
    src = Path(src_dir)
    dst = Path(local_cache) / subdir_name

    if _is_local_path(src) and not force_recopy:
        if not src.exists():
            raise FileNotFoundError(
                f"Source directory not found: {src}\n"
                "  → Re-run Cell 6 to rebuild datasets."
            )
        logger.info("Source already local: %s — skipping copy.", src)
        return src

    def _needs_copy() -> bool:
        if force_recopy or not dst.exists():
            return True
        src_count = sum(1 for _ in src.rglob("*") if _.is_file())
        dst_count = sum(1 for _ in dst.rglob("*") if _.is_file())
        return dst_count < src_count

    if _needs_copy():
        if dst.exists():
            shutil.rmtree(dst)
        logger.info("Copying %s → %s ...", src, dst)
        t0 = time.time()
        shutil.copytree(src, dst)
        logger.info("  Done in %.1fs", time.time() - t0)
    else:
        logger.info("Cache hit: %s — skipping copy.", dst)

    return dst


# ──────────────────────────────────────────────────────────────────────────────
# Transform factories
# ──────────────────────────────────────────────────────────────────────────────


def get_train_transform() -> transforms.Compose:
    """
    Domain-robust augmentation pipeline (v3 — targeted for PV→PD gap).

    Key changes from v2:
      - RandomErasing REMOVED (was destroying disease texture signal)
      - RandomPerspective ADDED (simulates oblique field camera angles)
      - GaussianBlur strengthened (sigma max 3.0, p=0.4)
      - RandomAutocontrast ADDED (global outdoor lighting variance)
      - RandomAdjustSharpness ADDED (phone camera variation)
      - ElasticTransform alpha reduced 50.0→30.0 (preserves disease texture)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.5, 1.0),
            ratio=(0.67, 1.5),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        # KEY NEW: simulates oblique handheld camera angles in field images
        # PV is always top-down/flat; PlantDoc is angled — this directly bridges that.
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.4,
            hue=0.1,
        ),
        # KEY NEW: handles global contrast from overcast vs sunny outdoor conditions
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomGrayscale(p=0.05),
        # Strengthened: sigma 3.0 (was 2.0), p=0.4 (was 0.3) — field blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        ], p=0.4),
        # Phone camera sharpness variation
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        # alpha reduced 50→30: preserves disease morphology, still simulates leaf droop
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=30.0, sigma=5.0)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # RandomErasing deliberately removed — see docstring.
        # Background robustness comes from RandomPerspective + RandomResizedCrop.
    ])


def get_eval_transform() -> transforms.Compose:
    """Deterministic preprocessing for VALIDATION / TEST splits."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PlantDiseaseDataset(Dataset):
    """
    ImageFolder-style dataset restricted to CANONICAL_CLASSES.

    Expected directory layout:
        root/
            <canonical_class_name>/
                img001.jpg
                img002.png
    """

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose,
        class_to_idx: dict[str, int] = CLASS_TO_IDX,
        allow_extra_classes: bool = True,
    ) -> None:
        self.root              = Path(root)
        self.transform         = transform
        self.class_to_idx      = class_to_idx
        self.allow_extra_classes = allow_extra_classes
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        n_skipped_class = 0
        n_skipped_file  = 0

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name.lower() in _SKIP_NAMES or class_dir.name.startswith("."):
                continue

            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                n_skipped_class += 1
                if not self.allow_extra_classes:
                    logger.warning("Unknown class folder (skipped): %s", class_name)
                continue

            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.name.lower() in _SKIP_NAMES or img_path.name.startswith("."):
                    continue
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
                }:
                    n_skipped_file += 1
                    continue
                self.samples.append((img_path, label))

        logger.info(
            "Loaded %d samples from %s  |  classes: %d  |  "
            "skipped class folders: %d  |  skipped non-image files: %d",
            len(self.samples), self.root,
            len(self.class_to_idx) - n_skipped_class,
            n_skipped_class, n_skipped_file,
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {self.root}. "
                "Check that class folder names match CANONICAL_CLASSES."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping corrupt image (%s): %s", exc.__class__.__name__, img_path)
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_distribution(self) -> dict[str, int]:
        from collections import Counter
        idx_to_cls = {v: k for k, v in self.class_to_idx.items()}
        counts: Counter = Counter(label for _, label in self.samples)
        return {idx_to_cls[idx]: count for idx, count in sorted(counts.items())}


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def _optimal_num_workers(cap: int = 2) -> int:
    """
    Return a safe worker count for Colab.
    Cap at 2: more workers don't improve throughput when the GPU is the
    bottleneck and data is on local SSD.
    """
    try:
        cpus = os.cpu_count() or 1
    except Exception:
        cpus = 1
    return min(cpus, cap)


def build_dataloaders(
    train_dir:           Optional[str | Path],
    val_dir:             str | Path,
    batch_size:          int  = 64,
    num_workers:         Optional[int] = None,
    train_shuffle:       bool = True,
    drop_last_train:     bool = True,
    class_to_idx:        dict[str, int] = CLASS_TO_IDX,
    allow_extra_classes: bool = True,
    persistent_workers:  bool = True,
) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    Build train and validation DataLoaders.

    Pass LOCAL SSD paths (from cache_dataset_to_local()), not Drive paths.
    Reading directly from Drive causes 35-70 min/epoch.
    """
    use_gpu    = torch.cuda.is_available()
    pin_memory = use_gpu
    nw         = num_workers if num_workers is not None else _optimal_num_workers()
    pw         = persistent_workers and (nw > 0)

    logger.info(
        "DataLoader — batch_size: %d | num_workers: %d | pin_memory: %s | device: %s",
        batch_size, nw, pin_memory, "CUDA" if use_gpu else "CPU",
    )

    train_loader: Optional[DataLoader] = None
    if train_dir is not None:
        train_dataset = PlantDiseaseDataset(
            root=train_dir, transform=get_train_transform(),
            class_to_idx=class_to_idx, allow_extra_classes=allow_extra_classes,
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=train_shuffle,
            num_workers=nw, pin_memory=pin_memory, drop_last=drop_last_train,
            persistent_workers=pw, prefetch_factor=2 if nw > 0 else None,
        )
        logger.info("Train loader: %d batches × %d", len(train_loader), batch_size)

    val_dataset = PlantDiseaseDataset(
        root=val_dir, transform=get_eval_transform(),
        class_to_idx=class_to_idx, allow_extra_classes=allow_extra_classes,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin_memory, drop_last=False,
        persistent_workers=pw, prefetch_factor=2 if nw > 0 else None,
    )
    logger.info("Val loader:   %d batches × %d", len(val_loader), batch_size)

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build both source and target loaders in one call
# ──────────────────────────────────────────────────────────────────────────────

def build_benchmark_loaders(
    pv_train_dir: str | Path,
    pv_val_dir:   str | Path,
    pd_val_dir:   str | Path,
    batch_size:   int = 64,
    num_workers:  Optional[int] = None,
) -> dict[str, DataLoader]:
    """Build all loaders for the source-only baseline experiment."""
    pv_train_loader, pv_val_loader = build_dataloaders(
        train_dir=pv_train_dir, val_dir=pv_val_dir,
        batch_size=batch_size, num_workers=num_workers,
    )
    _, pd_eval_loader = build_dataloaders(
        train_dir=None, val_dir=pd_val_dir,
        batch_size=batch_size, num_workers=num_workers,
    )
    return {"pv_train": pv_train_loader, "pv_val": pv_val_loader, "pd_eval": pd_eval_loader}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline verification utility
# ──────────────────────────────────────────────────────────────────────────────

def verify_pipeline(train_dir: str | Path, val_dir: str | Path, batch_size: int = 8) -> None:
    """Quick sanity check — pass LOCAL SSD paths."""
    logger.info("=" * 60)
    logger.info("PIPELINE VERIFICATION")
    logger.info("=" * 60)

    train_loader, val_loader = build_dataloaders(
        train_dir=train_dir, val_dir=val_dir, batch_size=batch_size,
        num_workers=0, drop_last_train=False, persistent_workers=False,
    )

    for name, loader in [("TRAIN", train_loader), ("VAL", val_loader)]:
        if loader is None:
            continue
        images, labels = next(iter(loader))
        logger.info(
            "[%s]  shape: %s  min: %.3f  max: %.3f  mean: %.3f",
            name, tuple(images.shape),
            images.min().item(), images.max().item(), images.mean().item(),
        )
        assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert labels.min().item() >= 0
        assert labels.max().item() < NUM_CLASSES

    logger.info("Class distribution (train):")
    for cls, cnt in sorted(train_loader.dataset.class_distribution().items()):
        logger.info("  [%2d] %-45s  %5d", CLASS_TO_IDX[cls], cls, cnt)

    logger.info("All assertions passed.")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/content/final_dataset/train")
    parser.add_argument("--val",   default="/content/final_dataset/val")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    verify_pipeline(train_dir=args.train, val_dir=args.val, batch_size=args.batch_size)
