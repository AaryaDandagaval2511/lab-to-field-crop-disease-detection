"""
finetune_plantdoc.py
====================
Fine-tunes a pretrained source-only (PlantVillage) checkpoint on the
PlantDoc dataset using the existing training pipeline infrastructure.

Key design decisions
---------------------
  • Loads the saved PlantVillage checkpoint (best_<arch>.pth or final_<arch>.pth).
  • Replaces the final classifier head if num_classes differs (supports
    17-class PV→PD transfer as-is, so no replacement is needed by default).
  • Uses discriminative learning rates:
      backbone LR = base_lr * 0.1   (preserve ImageNet / PV features)
      head     LR = base_lr         (train head at full speed)
  • Two-phase training (optional):
      Phase 1 — freeze backbone, train head only for `freeze_epochs` epochs.
      Phase 2 — unfreeze backbone, discriminative LRs for remaining epochs.
  • Validates on the PlantDoc val split after every epoch.
  • Saves best checkpoint (lowest val loss) + periodic checkpoints.
  • Logs metrics to CSV and prints a summary table at the end.
  • Fully self-contained: drop this file next to model.py / data_pipeline.py /
    train.py / evaluate.py and run it.  No other modifications required.

Usage (Colab cell)
------------------
    import importlib, finetune_plantdoc as _ft
    importlib.reload(_ft)
    from finetune_plantdoc import finetune_plantdoc

    history = finetune_plantdoc(
        checkpoint_path = '/content/drive/MyDrive/idea4drive/checkpoints/best_efficientnet_b0.pth',
        pd_train_dir    = '/content/plantdoc_aligned/train',   # ← create this split (see notes below)
        pd_val_dir      = '/content/plantdoc_aligned/val',
        output_dir      = '/content/drive/MyDrive/idea4drive/checkpoints_ft',
        epochs          = 15,
        freeze_epochs   = 3,
        base_lr         = 1e-4,
    )

Notes on the PlantDoc train/val split
--------------------------------------
The existing pipeline (Cell 6 in the notebook) puts ALL PlantDoc images
into /content/plantdoc_aligned/val for zero-shot domain-gap evaluation.
For supervised fine-tuning you need a *train* split.

Before running this script, create a 80/20 train/val split of PlantDoc:

    from finetune_plantdoc import split_plantdoc
    split_plantdoc(
        src_dir  = '/content/plantdoc_aligned/val',   # existing combined dir
        out_dir  = '/content/plantdoc_ft',
        val_frac = 0.20,
        seed     = 42,
    )
    # Then use:
    #   pd_train_dir = '/content/plantdoc_ft/train'
    #   pd_val_dir   = '/content/plantdoc_ft/val'

CPU vs GPU
----------
  GPU: efficientnet_b0, batch_size=32, AMP=True, ~1-2 min/epoch
  CPU: mobilenet_v3_small, batch_size=16, AMP=False, ~8-20 min/epoch
  (The script auto-detects and adjusts.)
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import queue
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── Project imports ────────────────────────────────────────────────────────
from data_pipeline import (
    NUM_CLASSES,
    PlantDiseaseDataset,
    CLASS_TO_IDX,
    build_dataloaders,
    get_eval_transform,
    get_train_transform,
)
from model import SUPPORTED_ARCHS, PlantDiseaseClassifier, build_model, recommend_arch
from evaluate import load_model_from_checkpoint

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
# Utility: split PlantDoc into train / val
# ──────────────────────────────────────────────────────────────────────────────

def split_plantdoc(
    src_dir:  str | Path,
    out_dir:  str | Path,
    val_frac: float = 0.20,
    seed:     int   = 42,
    force:    bool  = False,
) -> tuple[Path, Path]:
    """
    Create an 80/20 (default) stratified train/val split of PlantDoc.

    Parameters
    ----------
    src_dir  : directory with one subfolder per class (e.g. plantdoc_aligned/val)
    out_dir  : output root; creates <out_dir>/train and <out_dir>/val
    val_frac : fraction of images per class to put in val
    seed     : RNG seed for reproducibility
    force    : if True, delete & re-create even if out_dir already exists

    Returns
    -------
    (train_dir, val_dir)
    """
    src     = Path(src_dir)
    out     = Path(out_dir)
    tr_dir  = out / "train"
    v_dir   = out / "val"

    if out.exists() and not force:
        n_tr = sum(1 for _ in tr_dir.rglob("*") if _.is_file())
        n_v  = sum(1 for _ in v_dir.rglob("*")  if _.is_file())
        logger.info(
            "Split already exists — skipping.  train=%d  val=%d  (use force=True to redo)",
            n_tr, n_v,
        )
        return tr_dir, v_dir

    if out.exists() and force:
        shutil.rmtree(out)

    rng = random.Random(seed)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    total_tr = total_v = 0

    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = [
            f for f in sorted(cls_dir.iterdir())
            if f.is_file() and f.suffix.lower() in valid_ext
        ]
        rng.shuffle(images)
        n_val  = max(1, round(len(images) * val_frac))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dst = out / split_name / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dst / img.name)

        total_tr += len(train_imgs)
        total_v  += len(val_imgs)
        logger.info("  %-40s  train=%3d  val=%3d", cls_dir.name, len(train_imgs), len(val_imgs))

    logger.info("PlantDoc split complete — train=%d  val=%d → %s", total_tr, total_v, out)
    return tr_dir, v_dir


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
    logger.info("Seed set to %d", seed)


# ──────────────────────────────────────────────────────────────────────────────
# Background Drive writer (copied from train.py pattern)
# ──────────────────────────────────────────────────────────────────────────────

class _DriveWriter:
    _SENTINEL = None

    def __init__(self, drive_dir: Optional[Path]) -> None:
        self.drive_dir = drive_dir
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._errors: list[str] = []

    def start(self) -> None:
        if self.drive_dir is None:
            return
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("[DriveWriter] started → %s", self.drive_dir)

    def queue_copy(self, local_path: Path) -> None:
        if self.drive_dir is None or self._thread is None:
            return
        self._q.put(local_path)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._q.put(self._SENTINEL)
        self._thread.join()
        if self._errors:
            logger.warning("[DriveWriter] %d errors: %s", len(self._errors), self._errors)
        else:
            logger.info("[DriveWriter] All copies done.")

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                self._q.task_done()
                break
            try:
                dst = self.drive_dir / item.name
                shutil.copy2(item, dst)
                logger.info("[DriveWriter] %s → %s", item.name, dst)
            except Exception as exc:
                self._errors.append(str(exc))
            finally:
                self._q.task_done()


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule (identical to train.py)
# ──────────────────────────────────────────────────────────────────────────────

def _build_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
) -> SequentialLR:
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=optimizer.param_groups[0]["lr"] / 100,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser factories
# ──────────────────────────────────────────────────────────────────────────────

def _optimizer_phase1(model: PlantDiseaseClassifier, lr: float, wd: float) -> AdamW:
    """Phase 1: head only.  Backbone must be frozen before calling."""
    return AdamW(model.classifier.parameters(), lr=lr * 10.0, weight_decay=wd)


def _optimizer_phase2(model: PlantDiseaseClassifier, lr: float, wd: float) -> AdamW:
    """Phase 2: discriminative LRs — backbone 10× slower than head."""
    head_ids    = {id(p) for p in model.classifier.parameters()}
    backbone_ps = [p for p in model.parameters() if id(p) not in head_ids]
    return AdamW(
        [
            {"params": backbone_ps,                  "lr": lr * 0.1, "name": "backbone"},
            {"params": list(model.classifier.parameters()), "lr": lr,       "name": "head"},
        ],
        weight_decay=wd,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-epoch helpers (train + validate)
# ──────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.Module,
    optimizer:    torch.optim.Optimizer,
    scaler:       GradScaler,
    device:       torch.device,
    max_grad_norm: float,
    use_amp:      bool,
    epoch:        int,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss = total_correct = total_samples = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += bs

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "Epoch %3d | step %4d/%4d | loss %.4f | acc %.3f | lr %.2e | %.1fs",
                epoch, batch_idx + 1, len(loader),
                total_loss / total_samples,
                total_correct / total_samples,
                optimizer.param_groups[0]["lr"],
                time.time() - start_time,
            )

    return {
        "loss":         total_loss    / max(total_samples, 1),
        "accuracy":     total_correct / max(total_samples, 1),
        "lr":           optimizer.param_groups[0]["lr"],
        "epoch_time_s": time.time() - start_time,
    }


@torch.no_grad()
def _validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool,
) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += bs

    return {
        "loss":     total_loss    / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

_LOCAL_FT_CKPT_DIR = Path("/content/checkpoints_ft_local")

_CSV_FIELDS = ["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc",
               "lr", "epoch_time_s"]


def _save_checkpoint(
    filename:     str,
    epoch:        int,
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    scaler:       GradScaler,
    val_loss:     float,
    val_acc:      float,
    config:       dict,
    drive_writer: Optional[_DriveWriter] = None,
) -> Path:
    _LOCAL_FT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _LOCAL_FT_CKPT_DIR / filename
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "val_loss":        val_loss,
        "val_acc":         val_acc,
        "config":          config,
    }, local_path)
    logger.info("Checkpoint saved (local): %s", local_path)
    if drive_writer is not None:
        drive_writer.queue_copy(local_path)
    return local_path


def _init_csv(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()


def _append_csv(log_path: Path, row: dict) -> None:
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ──────────────────────────────────────────────────────────────────────────────

def finetune_plantdoc(
    checkpoint_path: str | Path,
    pd_train_dir:    str | Path,
    pd_val_dir:      str | Path,
    arch:            Optional[str]       = None,
    num_classes:     int                 = NUM_CLASSES,
    epochs:          int                 = 15,
    freeze_epochs:   int                 = 3,
    base_lr:         float               = 1e-4,
    weight_decay:    float               = 1e-4,
    warmup_epochs:   int                 = 2,
    max_grad_norm:   float               = 5.0,
    batch_size:      Optional[int]       = None,
    num_workers:     Optional[int]       = None,
    output_dir:      str | Path          = "./checkpoints_ft",
    save_every:      int                 = 5,
    seed:            int                 = 42,
) -> dict:
    """
    Fine-tune a pretrained PlantVillage checkpoint on PlantDoc.

    Parameters
    ----------
    checkpoint_path : path to the source-only .pth checkpoint
    pd_train_dir    : PlantDoc training split (class subfolders)
    pd_val_dir      : PlantDoc validation split (class subfolders)
    arch            : backbone name; None → auto-detect from checkpoint or device
    num_classes     : number of output classes (default 17, same as PV)
    epochs          : total training epochs
    freeze_epochs   : Phase-1 head-only epochs before unfreezing backbone
    base_lr         : learning rate for the head in Phase 2
                      (Phase 1 head LR = base_lr * 10; backbone LR = base_lr * 0.1)
    weight_decay    : AdamW weight decay
    warmup_epochs   : cosine-schedule warmup length (Phase 2)
    max_grad_norm   : gradient clipping
    batch_size      : None → 32 on GPU, 16 on CPU
    num_workers     : None → 2 on GPU, 0 on CPU
    output_dir      : where to save checkpoints (Drive path → async copy)
    save_every      : save a periodic checkpoint every N epochs
    seed            : RNG seed

    Returns
    -------
    dict with keys train_loss, train_acc, val_loss, val_acc (each a list)
    """
    seed_everything(seed)

    checkpoint_path = Path(checkpoint_path)
    output_dir      = Path(output_dir)

    # ── Device ────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Auto-settings ─────────────────────────────────────────────────────────
    if arch is None:
        arch = recommend_arch()
    if batch_size is None:
        batch_size = 32 if device.type == "cuda" else 16
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0

    freeze_epochs = max(0, min(freeze_epochs, epochs))

    logger.info("=" * 65)
    logger.info("FINE-TUNING on PlantDoc")
    logger.info("DEVICE: %s | AMP: %s | arch: %s", device, use_amp, arch)
    logger.info("batch_size: %d | num_workers: %d | epochs: %d", batch_size, num_workers, epochs)
    logger.info("base_lr: %.2e | freeze_epochs: %d", base_lr, freeze_epochs)
    logger.info("checkpoint: %s", checkpoint_path)
    logger.info("=" * 65)

    # ── Background Drive writer ────────────────────────────────────────────────
    out_str   = str(output_dir.resolve())
    drive_dir = output_dir if out_str.startswith("/content/drive") else None
    writer    = _DriveWriter(drive_dir)
    writer.start()

    _LOCAL_FT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pretrained model ─────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        arch=arch,
        num_classes=num_classes,
        device=device,
    )

    # ── If the checkpoint has a different num_classes, replace head ───────────
    # (By default PV and the aligned PlantDoc set share the same 17-class label
    #  space, so no replacement is needed.  This guard handles edge cases.)
    if model.classifier.out_features != num_classes:
        logger.info(
            "Replacing classifier head: %d → %d classes",
            model.classifier.out_features, num_classes,
        )
        model.classifier = nn.Linear(model.feature_dim, num_classes).to(device)
        # Re-initialise the new head with kaiming uniform
        nn.init.kaiming_uniform_(model.classifier.weight, nonlinearity="relu")
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)

    # ── Data loaders ──────────────────────────────────────────────────────────
    logger.info("Building PlantDoc DataLoaders …")
    train_loader, val_loader = build_dataloaders(
        train_dir=pd_train_dir,
        val_dir=pd_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info("Train: %d batches | Val: %d batches", len(train_loader), len(val_loader))

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # ── Config ────────────────────────────────────────────────────────────────
    config = dict(
        arch=arch, num_classes=num_classes, epochs=epochs,
        freeze_epochs=freeze_epochs, batch_size=batch_size,
        base_lr=base_lr, weight_decay=weight_decay,
        warmup_epochs=warmup_epochs, max_grad_norm=max_grad_norm,
        seed=seed, device=str(device),
        checkpoint_path=str(checkpoint_path),
        pd_train_dir=str(pd_train_dir), pd_val_dir=str(pd_val_dir),
    )

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = _LOCAL_FT_CKPT_DIR / f"ft_log_{arch}.csv"
    _init_csv(log_path)

    # ── History ───────────────────────────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    scaler = GradScaler(device=device.type, enabled=use_amp)
    val_metrics: dict = {}

    # =========================================================================
    # PHASE 1 — head-only (epochs 0 .. freeze_epochs-1)
    # =========================================================================
    if freeze_epochs > 0:
        model.freeze_backbone()
        optimizer = _optimizer_phase1(model, lr=base_lr, wd=weight_decay)
        scheduler = _build_scheduler(
            optimizer,
            warmup_epochs=min(1, freeze_epochs),
            total_epochs=freeze_epochs,
        )

        logger.info("=" * 65)
        logger.info("PHASE 1 (head-only): %d epoch(s)  head_lr=%.2e", freeze_epochs, base_lr * 10.0)
        logger.info("=" * 65)

        for epoch in range(freeze_epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs}  [Phase 1 — head only] =====")
            t0 = time.time()

            train_m = _train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, scaler=scaler, device=device,
                max_grad_norm=max_grad_norm, use_amp=use_amp, epoch=epoch + 1,
            )
            val_metrics = _validate(
                model=model, loader=val_loader, criterion=criterion,
                device=device, use_amp=use_amp,
            )
            scheduler.step()

            wall = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d | P1 | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f | lr=%.2e | %.1fs",
                epoch+1, train_m["loss"], train_m["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], lr_now, wall,
            )
            print(f"  Train — loss: {train_m['loss']:.4f}  acc: {train_m['accuracy']:.4f}")
            print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}")

            history["train_loss"].append(train_m["loss"])
            history["train_acc"].append(train_m["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            _append_csv(log_path, {
                "epoch": epoch+1, "phase": "P1",
                "train_loss": round(train_m["loss"], 6),
                "train_acc":  round(train_m["accuracy"], 6),
                "val_loss":   round(val_metrics["loss"], 6),
                "val_acc":    round(val_metrics["accuracy"], 6),
                "lr":         round(lr_now, 8),
                "epoch_time_s": round(wall, 2),
            })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ckpt = _save_checkpoint(
                    f"ft_best_{arch}.pth", epoch, model, optimizer, scheduler,
                    scaler, val_metrics["loss"], val_metrics["accuracy"], config, writer,
                )
                logger.info("  ★ New best val_loss=%.4f → %s", best_val_loss, ckpt)

            if (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    f"ft_{arch}_epoch{epoch+1:03d}.pth", epoch, model, optimizer,
                    scheduler, scaler, val_metrics["loss"], val_metrics["accuracy"],
                    config, writer,
                )

    # =========================================================================
    # PHASE 2 — full model, discriminative LRs (epochs freeze_epochs .. end)
    # =========================================================================
    remaining = epochs - freeze_epochs
    if remaining > 0:
        if freeze_epochs > 0:
            model.unfreeze_backbone()

        optimizer = _optimizer_phase2(model, lr=base_lr, wd=weight_decay)
        scheduler = _build_scheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=remaining,
        )

        logger.info("=" * 65)
        logger.info(
            "PHASE 2 (full model): %d epoch(s)  backbone_lr=%.2e  head_lr=%.2e",
            remaining, base_lr * 0.1, base_lr,
        )
        logger.info("=" * 65)

        for epoch in range(freeze_epochs, epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs}  [Phase 2 — full model] =====")
            t0 = time.time()

            train_m = _train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, scaler=scaler, device=device,
                max_grad_norm=max_grad_norm, use_amp=use_amp, epoch=epoch + 1,
            )
            val_metrics = _validate(
                model=model, loader=val_loader, criterion=criterion,
                device=device, use_amp=use_amp,
            )
            scheduler.step()

            wall = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d | P2 | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f | lr=%.2e | %.1fs",
                epoch+1, train_m["loss"], train_m["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], lr_now, wall,
            )
            print(f"  Train — loss: {train_m['loss']:.4f}  acc: {train_m['accuracy']:.4f}")
            print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}")

            history["train_loss"].append(train_m["loss"])
            history["train_acc"].append(train_m["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            _append_csv(log_path, {
                "epoch": epoch+1, "phase": "P2",
                "train_loss": round(train_m["loss"], 6),
                "train_acc":  round(train_m["accuracy"], 6),
                "val_loss":   round(val_metrics["loss"], 6),
                "val_acc":    round(val_metrics["accuracy"], 6),
                "lr":         round(lr_now, 8),
                "epoch_time_s": round(wall, 2),
            })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ckpt = _save_checkpoint(
                    f"ft_best_{arch}.pth", epoch, model, optimizer, scheduler,
                    scaler, val_metrics["loss"], val_metrics["accuracy"], config, writer,
                )
                logger.info("  ★ New best val_loss=%.4f → %s", best_val_loss, ckpt)

            if (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    f"ft_{arch}_epoch{epoch+1:03d}.pth", epoch, model, optimizer,
                    scheduler, scaler, val_metrics["loss"], val_metrics["accuracy"],
                    config, writer,
                )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if val_metrics:
        _save_checkpoint(
            f"ft_final_{arch}.pth", epochs - 1, model, optimizer, scheduler,
            scaler, val_metrics["loss"], val_metrics["accuracy"], config, writer,
        )

    writer.queue_copy(log_path)
    logger.info("Flushing Drive copies …")
    writer.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Fine-tuning complete.")
    if history["val_acc"]:
        logger.info(
            "Best val_loss: %.4f | Final val_acc: %.3f (%.2f%%)",
            best_val_loss, history["val_acc"][-1], 100 * history["val_acc"][-1],
        )
    logger.info("Checkpoints (local): %s", _LOCAL_FT_CKPT_DIR)
    if drive_dir:
        logger.info("Checkpoints (Drive): %s", drive_dir)
    logger.info("=" * 65)

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: print training curves after fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def plot_ft_curves(history: dict, save_path: Optional[str | Path] = None) -> None:
    """Plot and optionally save training curves from finetune_plantdoc()."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_acc"], "o-", label="train_acc")
    ax1.plot(epochs, history["val_acc"],   "s-", label="val_acc")
    ax1.set_title("Accuracy — PlantDoc fine-tuning")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_loss"], "o-", label="train_loss")
    ax2.plot(epochs, history["val_loss"],   "s-", label="val_loss")
    ax2.set_title("Loss — PlantDoc fine-tuning")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Curves saved: %s", save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a source-only checkpoint on PlantDoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True, help="Path to source .pth checkpoint")
    p.add_argument("--pd_train_dir",  required=True, help="PlantDoc train split directory")
    p.add_argument("--pd_val_dir",    required=True, help="PlantDoc val split directory")
    p.add_argument("--arch",          default=None,  choices=list(SUPPORTED_ARCHS) + [None])
    p.add_argument("--num_classes",   type=int,   default=NUM_CLASSES)
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--freeze_epochs", type=int,   default=3)
    p.add_argument("--base_lr",       type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int,   default=2)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--num_workers",   type=int,   default=None)
    p.add_argument("--output_dir",
                   default="/content/drive/MyDrive/idea4drive/checkpoints_ft")
    p.add_argument("--save_every",    type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument(
        "--split_first", action="store_true",
        help=(
            "If set, treat --pd_val_dir as the *combined* PlantDoc dir and split it "
            "80/20 into train/val before fine-tuning.  Output goes to --pd_split_dir."
        ),
    )
    p.add_argument("--pd_split_dir",
                   default="/content/plantdoc_ft",
                   help="Where to write the train/val split (only used with --split_first).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    pd_train = args.pd_train_dir
    pd_val   = args.pd_val_dir

    # Optional: auto-split the combined PlantDoc dir
    if args.split_first:
        pd_train, pd_val = split_plantdoc(
            src_dir=args.pd_val_dir,
            out_dir=args.pd_split_dir,
            val_frac=0.20,
            seed=args.seed,
        )

    history = finetune_plantdoc(
        checkpoint_path=args.checkpoint,
        pd_train_dir=pd_train,
        pd_val_dir=pd_val,
        arch=args.arch,
        num_classes=args.num_classes,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        save_every=args.save_every,
        seed=args.seed,
    )

    # Print summary table
    print("\n" + "=" * 65)
    print("FINE-TUNING SUMMARY")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10}")
    print("-" * 65)
    for i, (tl, ta, vl, va) in enumerate(zip(
        history["train_loss"], history["train_acc"],
        history["val_loss"],   history["val_acc"],
    )):
        print(f"{i+1:<8} {tl:<12.4f} {ta:<12.4f} {vl:<12.4f} {va:<10.4f}")
    print("=" * 65)
