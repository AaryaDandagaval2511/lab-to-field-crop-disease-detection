"""
train.py
========
Baseline source-only training pipeline for the Lab-to-Field Crop Disease
Domain-Generalisation Benchmark (Idea 4).

CPU vs GPU
----------
This script auto-detects the device and applies appropriate settings:

  GPU (T4/A100):
    arch=efficientnet_b0, batch_size=64, num_workers=2, AMP=True
    Expected: ~2-3 min/epoch, 15 epochs in ~35-45 min

  CPU (no GPU):
    arch=mobilenet_v3_small, batch_size=32, num_workers=0, AMP=False
    Expected: ~8-15 min/epoch

TWO-PHASE TRAINING (freeze_epochs)
-----------------------------------
Phase 1 (epochs 0 .. freeze_epochs-1):
  - Backbone is frozen; only the classification head trains.
  - Uses a single AdamW optimiser over head parameters only.
  - Head LR = lr * 10.0

Phase 2 (epochs freeze_epochs .. total_epochs-1):
  - Backbone unfrozen with discriminative learning rates:
      backbone LR = lr * 0.1   (preserves ImageNet features)
      head LR     = lr         (normal)
  - A fresh optimiser and scheduler are created at the phase boundary.

Set freeze_epochs=0 to disable freezing entirely (original behaviour).

Usage (Colab Cell)
------------------
    import importlib, train as _t, data_pipeline as _dp
    importlib.reload(_dp)
    importlib.reload(_t)
    from train import train

    history = train(
        train_dir     = '/content/final_dataset/train',
        val_dir       = '/content/final_dataset/val',
        epochs        = 15,
        freeze_epochs = 3,
        output_dir    = '/content/drive/MyDrive/idea4drive/checkpoints',
    )
"""

from __future__ import annotations

import argparse
import csv
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

from data_pipeline import NUM_CLASSES, build_dataloaders, cache_dataset_to_local
from model import SUPPORTED_ARCHS, build_model, recommend_arch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        logger.info("Global seed set to %d (cuDNN benchmark enabled)", seed)
    else:
        logger.info("Global seed set to %d (CPU mode — cuDNN not used)", seed)


# ---------------------------------------------------------------------------
# Background Drive writer
# ---------------------------------------------------------------------------
class DriveWriter:
    """Copies files to Google Drive in a background thread (non-blocking)."""

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
        logger.info("[DriveWriter] Background Drive-sync thread started → %s", self.drive_dir)

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
            logger.warning("[DriveWriter] %d copy errors: %s", len(self._errors), self._errors)
        else:
            logger.info("[DriveWriter] All Drive copies complete.")

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                self._q.task_done()
                break
            try:
                dst = self.drive_dir / item.name
                shutil.copy2(item, dst)
                logger.info("[DriveWriter] Copied %s → %s", item.name, dst)
            except Exception as exc:
                self._errors.append(str(exc))
                logger.warning("[DriveWriter] Failed to copy %s: %s", item, exc)
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=optimizer.param_groups[0]["lr"] / 100,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Single-epoch training pass
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
    epoch: int,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    start_time    = time.time()

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

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            running_acc  = total_correct / total_samples
            running_loss = total_loss    / total_samples
            lr_now       = optimizer.param_groups[0]["lr"]
            elapsed      = time.time() - start_time
            logger.info(
                "Epoch %3d | step %4d/%4d | loss %.4f | acc %.3f | lr %.2e | %.1fs",
                epoch, batch_idx + 1, len(loader),
                running_loss, running_acc, lr_now, elapsed,
            )

    epoch_time = time.time() - start_time
    return {
        "loss":         total_loss    / max(total_samples, 1),
        "accuracy":     total_correct / max(total_samples, 1),
        "lr":           optimizer.param_groups[0]["lr"],
        "epoch_time_s": epoch_time,
    }


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return {
        "loss":     total_loss    / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
_LOCAL_CKPT_DIR = Path("/content/checkpoints_local")


def save_checkpoint(
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    val_loss: float,
    val_acc: float,
    config: dict,
    drive_writer: Optional[DriveWriter] = None,
) -> Path:
    _LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _LOCAL_CKPT_DIR / filename

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

    logger.info("Checkpoint saved (local SSD): %s", local_path)
    if drive_writer is not None:
        drive_writer.queue_copy(local_path)

    return local_path


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    logger.info(
        "Resumed from %s (epoch %d, val_loss=%.4f, val_acc=%.3f)",
        ckpt_path, ckpt["epoch"], ckpt["val_loss"], ckpt["val_acc"],
    )
    return ckpt["epoch"] + 1, ckpt["val_loss"]


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
_CSV_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"]

def init_csv_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

def append_csv_log(log_path: Path, row: dict) -> None:
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# Optimiser factory — one per phase
# ---------------------------------------------------------------------------
def _build_optimizer_phase1(model, lr: float, weight_decay: float) -> AdamW:
    """Phase 1: optimise head only. Backbone must already be frozen."""
    head_params = list(model.classifier.parameters())
    return AdamW(head_params, lr=lr * 10.0, weight_decay=weight_decay)


def _build_optimizer_phase2(model, lr: float, weight_decay: float) -> AdamW:
    """Phase 2: discriminative LR — backbone slow, head fast."""
    head_params     = list(model.classifier.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
            {"params": head_params,     "lr": lr,       "name": "head"},
        ],
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    train_dir: str | Path,
    val_dir: str | Path,
    arch: Optional[str] = None,
    num_classes: int = NUM_CLASSES,
    epochs: int = 15,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    max_grad_norm: float = 5.0,
    num_workers: Optional[int] = None,
    output_dir: str | Path = "./checkpoints",
    resume: Optional[str | Path] = None,
    seed: int = 42,
    save_every: int = 5,
    local_cache_dir: str | Path = "/content/data_cache",
    force_recopy: bool = False,
    freeze_epochs: int = 3,
) -> dict:
    """
    Train a PlantDiseaseClassifier on PlantVillage (source domain only).

    Parameters
    ----------
    freeze_epochs : int
        Number of epochs to freeze the backbone (Phase 1 — head only).
        After this many epochs the backbone is unfrozen and Phase 2
        starts with discriminative learning rates.
        Set to 0 to skip Phase 1 entirely (standard fine-tuning).
    """
    seed_everything(seed)

    output_dir = Path(output_dir)

    # ── Device ────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Auto-select arch / batch_size / num_workers ───────────────────────────
    if arch is None:
        arch = recommend_arch()
    if batch_size is None:
        batch_size = 64 if device.type == "cuda" else 32
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0

    # Clamp so freeze_epochs never exceeds total epochs
    freeze_epochs = max(0, min(freeze_epochs, epochs))

    logger.info("=" * 65)
    logger.info("DEVICE: %s | AMP: %s | arch: %s", device, use_amp, arch)
    logger.info("batch_size: %d | num_workers: %d | epochs: %d", batch_size, num_workers, epochs)
    logger.info(
        "freeze_epochs: %d  (Phase 1: epochs 1-%d head-only | Phase 2: epochs %d-%d full model)",
        freeze_epochs,
        freeze_epochs,
        freeze_epochs + 1,
        epochs,
    )
    if device.type == "cpu":
        logger.info(
            "⚠ CPU MODE: Training will be slow (~8-15 min/epoch). "
            "Switch to GPU for full training."
        )
    logger.info("=" * 65)

    # ── Background Drive writer ────────────────────────────────────────────────
    output_str = str(output_dir.resolve())
    drive_dir: Optional[Path] = output_dir if output_str.startswith("/content/drive") else None
    drive_writer = DriveWriter(drive_dir)
    drive_writer.start()

    local_log_dir = Path("/content/checkpoints_local")
    local_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Cache dataset to local SSD ────────────────────────────────────────────
    logger.info("Caching dataset to local SSD...")
    local_train_dir, local_val_dir = cache_dataset_to_local(
        src_train_dir=train_dir,
        src_val_dir=val_dir,
        local_cache=local_cache_dir,
        force_recopy=force_recopy,
    )
    logger.info("Dataset ready. Building DataLoaders...")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_dir=local_train_dir,
        val_dir=local_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(arch=arch, num_classes=num_classes, device=device)
    logger.info("%s", model)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # ── Config ────────────────────────────────────────────────────────────────
    config = {
        "arch": arch, "num_classes": num_classes,
        "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs, "max_grad_norm": max_grad_norm,
        "seed": seed, "device": str(device),
        "train_dir": str(train_dir), "val_dir": str(val_dir),
        "freeze_epochs": freeze_epochs,
    }

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = local_log_dir / f"train_log_{arch}.csv"
    init_csv_log(log_path)
    logger.info("Training log: %s", log_path)

    # ── History ───────────────────────────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss  = float("inf")
    val_metrics: dict = {}

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # =========================================================================
    # PHASE 1 — head-only training  (epochs 0 .. freeze_epochs-1)
    # =========================================================================
    if freeze_epochs > 0:
        # --- Freeze backbone: only head parameters will be updated ---
        model.freeze_backbone()

        optimizer = _build_optimizer_phase1(model, lr=lr, weight_decay=weight_decay)
        scheduler = build_scheduler(
            optimizer,
            warmup_epochs=min(1, freeze_epochs),
            total_epochs=freeze_epochs,
        )

        logger.info("=" * 65)
        logger.info("PHASE 1: head-only training for %d epoch(s)", freeze_epochs)
        logger.info("  head LR = %.2e  (backbone is frozen)", lr * 10.0)
        logger.info("=" * 65)

        for epoch in range(freeze_epochs):
            phase_tag = f"[P1 {epoch + 1}/{freeze_epochs}]"
            print(f"\n===== Epoch {epoch + 1}/{epochs} — HEAD ONLY (backbone frozen) =====")
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, scaler=scaler, device=device,
                max_grad_norm=max_grad_norm, use_amp=use_amp,
                epoch=epoch + 1,
            )
            val_metrics = validate(
                model=model, loader=val_loader, criterion=criterion,
                device=device, use_amp=use_amp,
            )
            scheduler.step()

            lr_now     = optimizer.param_groups[0]["lr"]
            epoch_wall = time.time() - epoch_start

            logger.info(
                "Epoch %3d/%d %s | train_loss %.4f  train_acc %.3f | "
                "val_loss %.4f  val_acc %.3f | lr %.2e | %.1fs",
                epoch + 1, epochs, phase_tag,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"],   val_metrics["accuracy"],
                lr_now, epoch_wall,
            )
            print(f"  Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}")
            print(f"  Time: {epoch_wall:.1f}s")

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            append_csv_log(log_path, {
                "epoch":        epoch + 1,
                "train_loss":   round(train_metrics["loss"],     6),
                "train_acc":    round(train_metrics["accuracy"], 6),
                "val_loss":     round(val_metrics["loss"],       6),
                "val_acc":      round(val_metrics["accuracy"],   6),
                "lr":           round(lr_now,                    8),
                "epoch_time_s": round(epoch_wall,                2),
            })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ckpt_path = save_checkpoint(
                    filename=f"best_{arch}.pth",
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"],
                    config=config, drive_writer=drive_writer,
                )
                logger.info("  * New best val_loss=%.4f → %s", best_val_loss, ckpt_path)

            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    filename=f"{arch}_epoch{epoch + 1:03d}.pth",
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"],
                    config=config, drive_writer=drive_writer,
                )

    # =========================================================================
    # PHASE 2 — full model with discriminative LRs (epochs freeze_epochs .. end)
    # =========================================================================
    remaining = epochs - freeze_epochs

    if remaining > 0:
        # --- Unfreeze backbone: all parameters are now trainable ---
        if freeze_epochs > 0:
            model.unfreeze_backbone()

        optimizer = _build_optimizer_phase2(model, lr=lr, weight_decay=weight_decay)
        scheduler = build_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=remaining,
        )

        logger.info("=" * 65)
        logger.info("PHASE 2: full-model fine-tuning for %d epoch(s)", remaining)
        logger.info("  backbone LR = %.2e  |  head LR = %.2e", lr * 0.1, lr)
        logger.info("=" * 65)

        for epoch in range(freeze_epochs, epochs):
            phase_tag = f"[P2 {epoch - freeze_epochs + 1}/{remaining}]"
            print(f"\n===== Epoch {epoch + 1}/{epochs} — FULL MODEL (discriminative LR) =====")
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, scaler=scaler, device=device,
                max_grad_norm=max_grad_norm, use_amp=use_amp,
                epoch=epoch + 1,
            )
            val_metrics = validate(
                model=model, loader=val_loader, criterion=criterion,
                device=device, use_amp=use_amp,
            )
            scheduler.step()

            lr_now     = optimizer.param_groups[0]["lr"]
            epoch_wall = time.time() - epoch_start

            logger.info(
                "Epoch %3d/%d %s | train_loss %.4f  train_acc %.3f | "
                "val_loss %.4f  val_acc %.3f | lr %.2e | %.1fs",
                epoch + 1, epochs, phase_tag,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"],   val_metrics["accuracy"],
                lr_now, epoch_wall,
            )
            print(f"  Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}")
            print(f"  Time: {epoch_wall:.1f}s")

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            append_csv_log(log_path, {
                "epoch":        epoch + 1,
                "train_loss":   round(train_metrics["loss"],     6),
                "train_acc":    round(train_metrics["accuracy"], 6),
                "val_loss":     round(val_metrics["loss"],       6),
                "val_acc":      round(val_metrics["accuracy"],   6),
                "lr":           round(lr_now,                    8),
                "epoch_time_s": round(epoch_wall,                2),
            })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                ckpt_path = save_checkpoint(
                    filename=f"best_{arch}.pth",
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"],
                    config=config, drive_writer=drive_writer,
                )
                logger.info("  * New best val_loss=%.4f → %s", best_val_loss, ckpt_path)

            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    filename=f"{arch}_epoch{epoch + 1:03d}.pth",
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"],
                    config=config, drive_writer=drive_writer,
                )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if val_metrics:
        save_checkpoint(
            filename=f"final_{arch}.pth",
            epoch=epochs - 1, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"],
            config=config, drive_writer=drive_writer,
        )

    drive_writer.queue_copy(log_path)

    logger.info("Flushing Drive copies...")
    drive_writer.stop()

    logger.info("=" * 65)
    logger.info("Training complete.")
    if history["val_acc"]:
        logger.info(
            "Best val_loss: %.4f | Final val_acc: %.3f",
            best_val_loss, history["val_acc"][-1],
        )
    logger.info("Checkpoints (local SSD): %s", _LOCAL_CKPT_DIR)
    if drive_dir:
        logger.info("Checkpoints (Drive):     %s", drive_dir)
    logger.info("=" * 65)

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source-only baseline for Idea 4 benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_dir",       default="/content/final_dataset/train")
    parser.add_argument("--val_dir",         default="/content/final_dataset/val")
    parser.add_argument("--arch",            default=None, choices=list(SUPPORTED_ARCHS) + [None])
    parser.add_argument("--num_classes",     type=int,   default=NUM_CLASSES)
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",   type=int,   default=2)
    parser.add_argument("--max_grad_norm",   type=float, default=5.0)
    parser.add_argument("--num_workers",     type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--save_every",      type=int,   default=5)
    parser.add_argument("--output_dir",      default="/content/drive/MyDrive/idea4drive/checkpoints")
    parser.add_argument("--resume",          default=None)
    parser.add_argument("--local_cache_dir", default="/content/data_cache")
    parser.add_argument("--force_recopy",    action="store_true")
    parser.add_argument("--freeze_epochs",   type=int,   default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        train_dir       = args.train_dir,
        val_dir         = args.val_dir,
        arch            = args.arch,
        num_classes     = args.num_classes,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        max_grad_norm   = args.max_grad_norm,
        num_workers     = args.num_workers,
        output_dir      = args.output_dir,
        resume          = args.resume,
        seed            = args.seed,
        save_every      = args.save_every,
        local_cache_dir = args.local_cache_dir,
        force_recopy    = args.force_recopy,
        freeze_epochs   = args.freeze_epochs,
    )
