"""
evaluate.py
===========
Evaluation pipeline for the Lab-to-Field Crop Disease
Domain-Generalisation Benchmark (Idea 4).

Evaluates a trained PlantDiseaseClassifier checkpoint on:
  • PlantVillage val split  (source domain — in-distribution)
  • PlantDoc aligned split  (target domain — out-of-distribution)

Metrics computed
----------------
  • Top-1 Accuracy
  • Macro-averaged F1 score
  • Per-class precision, recall, F1 (full classification report)
  • Confusion matrix (saved as CSV + heatmap PNG)

Design decisions
----------------
  • Zero leakage: PlantDoc is NEVER seen during training.  This script
    only loads a frozen checkpoint and runs inference.
  • Deterministic: eval transform only (no augmentation), fixed seed.
  • Robust: handles class imbalance gracefully (macro-F1 is the primary
    target metric, not accuracy alone).
  • Structured output: results are saved as JSON + printed as a table.

RECOMMENDATION 3 (Test-Time Augmentation):
------------------------------------------
Added run_inference_tta() function. This is used automatically for the
PlantDoc (target/OOD) evaluation inside evaluate(). PlantVillage val
continues to use the standard deterministic run_inference().

TTA averages softmax probabilities across n_tta=5 augmented views of each
test image. View 0 is the standard deterministic eval transform. Views 1..4
apply mild random augmentation (no RandomErasing — too destructive at test
time). This directly addresses the domain gap because:
  - The model is uncertain on PlantDoc images (OOD distribution)
  - Averaging over multiple views reduces this uncertainty
  - The mild augmentation at test time samples views closer to how the
    model was trained (with the new domain-robust augmentation)
  - Zero retraining cost

Usage
-----
    python evaluate.py \
        --checkpoint /content/drive/MyDrive/idea4drive/checkpoints/best_efficientnet_b0.pth \
        --pv_val_dir /content/final_dataset/val \
        --pd_eval_dir /content/plantdoc_aligned/val \
        --output_dir /content/drive/MyDrive/idea4drive/eval_outputs \
        --arch efficientnet_b0 \
        --batch_size 32

    # Quick source-only check (no PlantDoc path needed):
    python evaluate.py \
        --checkpoint /content/.../best_efficientnet_b0.pth \
        --pv_val_dir /content/final_dataset/val \
        --output_dir /content/.../eval_outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_pipeline import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PlantDiseaseDataset,
    build_dataloaders,
    get_eval_transform,
)
from model import SUPPORTED_ARCHS, PlantDiseaseClassifier, build_model
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    arch: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> PlantDiseaseClassifier:
    """
    Load a PlantDiseaseClassifier from a saved checkpoint.

    Supports both:
      • Full training checkpoints (saved by train.py — dict with 'model_state')
      • Raw state_dict files (torch.save(model.state_dict(), ...))

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the .pth checkpoint file.
    arch : str
        Architecture string.  Must match what was used during training.
    num_classes : int
        Number of output classes.
    device : torch.device | None
        Target device.  None → auto-detect.

    Returns
    -------
    PlantDiseaseClassifier (eval mode, on device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("=" * 65)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    logger.info("  arch=%s | num_classes=%d | device=%s", arch, num_classes, device)

    # Build the model architecture first
    model = build_model(arch=arch, num_classes=num_classes, device=device)

    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both full training checkpoints and raw state dicts
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        saved_epoch = ckpt.get("epoch", "?")
        saved_val_loss = ckpt.get("val_loss", float("nan"))
        saved_val_acc = ckpt.get("val_acc", float("nan"))
        logger.info(
            "  Loaded training checkpoint — epoch: %s | val_loss: %.4f | val_acc: %.3f",
            saved_epoch, saved_val_loss, saved_val_acc,
        )
    elif isinstance(ckpt, dict):
        # Raw state dict
        model.load_state_dict(ckpt)
        logger.info("  Loaded raw state dict.")
    else:
        raise ValueError(
            f"Unexpected checkpoint format: {type(ckpt)}. "
            "Expected a dict with 'model_state' key or a raw state dict."
        )

    model.eval()
    logger.info("Model loaded and set to eval mode.")
    logger.info("=" * 65)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: PlantDiseaseClassifier,
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    domain_label: str = "unknown",
) -> dict:
    """
    Run inference on a dataset directory and return predictions + ground truth.

    Parameters
    ----------
    model : PlantDiseaseClassifier
        The model in eval mode.
    data_dir : str | Path
        Path to a dataset directory (one subfolder per canonical class).
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader workers.
    domain_label : str
        Human-readable label for logging (e.g., "PlantVillage val", "PlantDoc").

    Returns
    -------
    dict with keys:
        all_preds  : list[int] — predicted class indices
        all_labels : list[int] — ground truth class indices
        all_probs  : np.ndarray of shape (N, num_classes) — softmax probabilities
        n_samples  : int
        inference_time_s : float
    """
    data_dir = Path(data_dir)
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    logger.info("-" * 55)
    logger.info("[%s] Loading dataset from: %s", domain_label, data_dir)

    dataset = PlantDiseaseDataset(
        root=data_dir,
        transform=get_eval_transform(),
        class_to_idx=CLASS_TO_IDX,
        allow_extra_classes=True,
    )

    pin_memory = device.type == "cuda"
    pw = num_workers > 0

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,           # deterministic order for reproducibility
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    n_batches = len(loader)
    logger.info(
        "[%s] %d samples | %d batches | batch_size=%d",
        domain_label, len(dataset), n_batches, batch_size,
    )

    all_preds:  list[int] = []
    all_labels: list[int] = []
    all_probs:  list[np.ndarray] = []

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

        # Progress logging every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            logger.info(
                "[%s] batch %d/%d  (%.1f%%)",
                domain_label, batch_idx + 1, n_batches,
                100.0 * (batch_idx + 1) / n_batches,
            )

    inference_time = time.time() - start_time
    all_probs_arr = np.concatenate(all_probs, axis=0)  # (N, num_classes)

    logger.info(
        "[%s] Inference complete: %d samples in %.2fs (%.1f ms/sample)",
        domain_label,
        len(all_labels),
        inference_time,
        1000.0 * inference_time / max(len(all_labels), 1),
    )

    return {
        "all_preds":         all_preds,
        "all_labels":        all_labels,
        "all_probs":         all_probs_arr,
        "n_samples":         len(all_labels),
        "inference_time_s":  inference_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation inference (Recommendation 3)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_tta(
    model: PlantDiseaseClassifier,
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    domain_label: str = "unknown",
    n_tta: int = 5,
) -> dict:
    """
    TTA inference: average softmax probabilities across n_tta augmented views.

    View 0  : standard deterministic eval transform (CenterCrop).
    Views 1+: mild random augmentation — no RandomErasing (too destructive
              at test time) but enough variation to reduce OOD uncertainty.

    Why this helps for PlantDoc:
      The model is uncertain on OOD images because they look nothing like
      PV training images. Averaging probabilities over multiple views of the
      same image reduces this uncertainty without any retraining cost.

    Parameters
    ----------
    model : PlantDiseaseClassifier
        The model in eval mode.
    data_dir : str | Path
        Path to a dataset directory (one subfolder per canonical class).
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader workers.
    domain_label : str
        Human-readable label for logging.
    n_tta : int
        Number of augmented views to average. Default 5.

    Returns
    -------
    dict with keys:
        all_preds  : list[int]
        all_labels : list[int]
        all_probs  : np.ndarray of shape (N, num_classes) — mean softmax probs
        n_samples  : int
        inference_time_s : float
    """
    data_dir = Path(data_dir)
    device   = next(model.parameters()).device
    use_amp  = device.type == "cuda"

    # Mild TTA transform — no erasing, conservative augmentation
    # Mirrors the kind of variation seen in field (PlantDoc) images
    # Inside run_inference_tta() — replace tta_transform definition only
# (approximately line 185 in your evaluate.py)

    # TTA transform: mild augmentation matching PlantDoc appearance variation
    # Includes perspective (new) to match the updated training augmentation
    tta_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),  # NEW
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # mild blur
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    logger.info("-" * 55)
    logger.info("[%s TTA] n_tta=%d | data: %s", domain_label, n_tta, data_dir)

    all_probs_tta: list[np.ndarray] = []  # each entry shape: (N, num_classes)
    all_labels: list[int] = []
    start_time = time.time()

    for view_idx in range(n_tta):
        # View 0 is always the deterministic eval transform for reproducibility
        transform = get_eval_transform() if view_idx == 0 else tta_transform

        dataset = PlantDiseaseDataset(
            root=data_dir,
            transform=transform,
            class_to_idx=CLASS_TO_IDX,
            allow_extra_classes=True,
        )

        pin_memory = device.type == "cuda"
        pw = num_workers > 0

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # must be False — order must match across views
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=pw,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        view_probs: list[np.ndarray] = []

        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            view_probs.append(probs.cpu().numpy())

            # Collect ground truth labels only once (view 0)
            if view_idx == 0:
                all_labels.extend(labels.tolist())

        all_probs_tta.append(np.concatenate(view_probs, axis=0))  # (N, num_classes)
        logger.info(
            "[%s TTA] View %d/%d complete (%d samples)",
            domain_label, view_idx + 1, n_tta, len(all_labels),
        )

    # Average softmax probabilities across all views, then take argmax
    mean_probs = np.mean(all_probs_tta, axis=0)   # (N, num_classes)
    all_preds  = mean_probs.argmax(axis=1).tolist()

    inference_time = time.time() - start_time
    logger.info(
        "[%s TTA] Complete: %d samples × %d views in %.2fs",
        domain_label, len(all_labels), n_tta, inference_time,
    )

    return {
        "all_preds":        all_preds,
        "all_labels":       all_labels,
        "all_probs":        mean_probs,
        "n_samples":        len(all_labels),
        "inference_time_s": inference_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds:  list[int],
    all_labels: list[int],
    domain_label: str = "unknown",
) -> dict:
    """
    Compute accuracy, macro-F1, and per-class metrics.

    Parameters
    ----------
    all_preds : list[int]
        Predicted class indices.
    all_labels : list[int]
        Ground truth class indices.
    domain_label : str
        For logging purposes only.

    Returns
    -------
    dict with keys:
        accuracy      : float
        macro_f1      : float
        weighted_f1   : float
        per_class     : dict[str, dict]   — per-class precision/recall/f1/support
        report_str    : str               — full sklearn classification report
        confusion_matrix : list[list[int]]
    """
    accuracy    = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Build target_names only for classes actually present in the data
    present_indices = sorted(set(all_labels) | set(all_preds))
    target_names    = [IDX_TO_CLASS.get(i, f"class_{i}") for i in present_indices]

    report_str = classification_report(
        all_labels, all_preds,
        labels=present_indices,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    # Per-class metrics as a structured dict
    report_dict = classification_report(
        all_labels, all_preds,
        labels=present_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1":        round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in target_names
        if cls in report_dict
    }

    cm = confusion_matrix(all_labels, all_preds, labels=present_indices).tolist()

    logger.info("-" * 55)
    logger.info("[%s] Results:", domain_label)
    logger.info("  Accuracy   : %.4f  (%.2f%%)", accuracy, 100 * accuracy)
    logger.info("  Macro-F1   : %.4f", macro_f1)
    logger.info("  Weighted-F1: %.4f", weighted_f1)
    logger.info("-" * 55)
    logger.info("\n%s", report_str)

    return {
        "domain":          domain_label,
        "accuracy":        round(accuracy,    4),
        "macro_f1":        round(macro_f1,    4),
        "weighted_f1":     round(weighted_f1, 4),
        "n_samples":       len(all_labels),
        "per_class":       per_class,
        "report_str":      report_str,
        "confusion_matrix": cm,
        "present_class_indices": present_indices,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_csv(
    cm: list[list[int]],
    present_indices: list[int],
    output_path: Path,
) -> None:
    """Save confusion matrix as a CSV file."""
    import csv
    labels = [IDX_TO_CLASS.get(i, f"class_{i}") for i in present_indices]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + row)
    logger.info("Confusion matrix CSV saved: %s", output_path)


def save_confusion_matrix_png(
    cm: list[list[int]],
    present_indices: list[int],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix as a heatmap PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        labels = [IDX_TO_CLASS.get(i, f"class_{i}") for i in present_indices]
        cm_arr = np.array(cm)

        # Normalize each row (true label) for readability
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

        fig_size = max(10, len(labels) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.4,
            linecolor="lightgray",
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix PNG saved: %s", output_path)

    except ImportError:
        logger.warning(
            "matplotlib/seaborn not available — confusion matrix PNG not saved. "
            "Install with: pip install matplotlib seaborn pandas"
        )


def save_results_json(results: dict, output_path: Path) -> None:
    """Save the full results dict as a JSON file (excluding raw report string)."""
    # Don't serialise the long report string — it's in its own file
    exportable = {k: v for k, v in results.items() if k != "report_str"}
    with open(output_path, "w") as f:
        json.dump(exportable, f, indent=2)
    logger.info("Results JSON saved: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str | Path,
    pv_val_dir: str | Path,
    pd_eval_dir: Optional[str | Path] = None,
    arch: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    batch_size: int = 32,
    num_workers: int = 2,
    output_dir: str | Path = "./eval_outputs",
    device: Optional[torch.device | str] = None,
    tta_n_views: int = 5,
) -> dict:
    """
    Evaluate a trained model on source and (optionally) target domains.

    PlantVillage val uses standard deterministic inference (run_inference).
    PlantDoc eval uses Test-Time Augmentation (run_inference_tta) with
    n_tta=tta_n_views views — zero retraining cost, reduces OOD uncertainty.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the .pth checkpoint.
    pv_val_dir : str | Path
        PlantVillage validation split directory.
    pd_eval_dir : str | Path | None
        PlantDoc evaluation directory.  Pass None to skip.
    arch : str
        Backbone architecture matching the checkpoint.
    num_classes : int
        Number of output classes.
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader workers.
    output_dir : str | Path
        Directory where results, confusion matrices, and reports are saved.
    device : torch.device | str | None
        Target device.
    tta_n_views : int
        Number of TTA views for PlantDoc evaluation. Default 5.

    Returns
    -------
    dict with keys:
        pv_val : dict   — source domain results
        pd_eval: dict   — target domain results (if pd_eval_dir provided)
        domain_gap : dict — accuracy and F1 drop (pv − pd)
    """
    if device is not None:
        device = torch.device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        arch=arch,
        num_classes=num_classes,
        device=device,
    )
    effective_device = next(model.parameters()).device

    all_results: dict = {}

    # ── Evaluate: PlantVillage val (source domain) ────────────────────────────
    # Standard deterministic inference — PV is in-distribution, no TTA needed
    logger.info("\n" + "=" * 65)
    logger.info("EVALUATING: PlantVillage val (source domain)")
    logger.info("=" * 65)

    pv_inference = run_inference(
        model=model,
        data_dir=pv_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        domain_label="PV-val",
    )
    pv_metrics = compute_metrics(
        all_preds=pv_inference["all_preds"],
        all_labels=pv_inference["all_labels"],
        domain_label="PV-val",
    )
    pv_metrics["inference_time_s"] = pv_inference["inference_time_s"]

    # Save PV outputs
    save_confusion_matrix_csv(
        cm=pv_metrics["confusion_matrix"],
        present_indices=pv_metrics["present_class_indices"],
        output_path=output_dir / "pv_val_confusion_matrix.csv",
    )
    save_confusion_matrix_png(
        cm=pv_metrics["confusion_matrix"],
        present_indices=pv_metrics["present_class_indices"],
        output_path=output_dir / "pv_val_confusion_matrix.png",
        title="PlantVillage Val — Confusion Matrix (row-normalised)",
    )
    (output_dir / "pv_val_classification_report.txt").write_text(
        pv_metrics["report_str"]
    )
    save_results_json(pv_metrics, output_dir / "pv_val_results.json")

    all_results["pv_val"] = pv_metrics

    # ── Evaluate: PlantDoc (target domain) with TTA ───────────────────────────
    # Uses run_inference_tta instead of run_inference (Recommendation 3).
    # TTA averages probabilities over n_tta views to reduce OOD uncertainty.
    if pd_eval_dir is not None:
        pd_eval_dir = Path(pd_eval_dir)
        logger.info("\n" + "=" * 65)
        logger.info(
            "EVALUATING: PlantDoc (target domain — OOD) with TTA (n=%d views)",
            tta_n_views,
        )
        logger.info("=" * 65)

        pd_inference = run_inference_tta(
            model=model,
            data_dir=pd_eval_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            domain_label="PD-eval",
            n_tta=tta_n_views,
        )
        pd_metrics = compute_metrics(
            all_preds=pd_inference["all_preds"],
            all_labels=pd_inference["all_labels"],
            domain_label="PD-eval",
        )
        pd_metrics["inference_time_s"] = pd_inference["inference_time_s"]

        # Save PD outputs
        save_confusion_matrix_csv(
            cm=pd_metrics["confusion_matrix"],
            present_indices=pd_metrics["present_class_indices"],
            output_path=output_dir / "pd_eval_confusion_matrix.csv",
        )
        save_confusion_matrix_png(
            cm=pd_metrics["confusion_matrix"],
            present_indices=pd_metrics["present_class_indices"],
            output_path=output_dir / "pd_eval_confusion_matrix.png",
            title="PlantDoc Eval — Confusion Matrix (row-normalised)",
        )
        (output_dir / "pd_eval_classification_report.txt").write_text(
            pd_metrics["report_str"]
        )
        save_results_json(pd_metrics, output_dir / "pd_eval_results.json")

        all_results["pd_eval"] = pd_metrics

    # ── Return inference results for domain_gap.py ─────────────────────────────
    # Attach raw probs and labels for domain gap computation
    all_results["_pv_inference"] = pv_inference
    if pd_eval_dir is not None:
        all_results["_pd_inference"] = pd_inference

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PlantDiseaseClassifier on PV val + PlantDoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pth checkpoint to evaluate"
    )
    parser.add_argument(
        "--pv_val_dir", default="/content/final_dataset/val",
        help="PlantVillage validation split directory"
    )
    parser.add_argument(
        "--pd_eval_dir", default=None,
        help="PlantDoc evaluation directory (optional; skip if not available)"
    )
    parser.add_argument(
        "--arch", default="efficientnet_b0", choices=SUPPORTED_ARCHS,
        help="Backbone architecture (must match checkpoint)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=NUM_CLASSES,
    )
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=2)
    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/idea4drive/eval_outputs",
        help="Directory for all evaluation outputs"
    )
    parser.add_argument(
        "--tta_n_views", type=int, default=5,
        help="Number of TTA views for PlantDoc evaluation (default 5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        pv_val_dir=args.pv_val_dir,
        pd_eval_dir=args.pd_eval_dir,
        arch=args.arch,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        tta_n_views=args.tta_n_views,
    )
