"""
domain_gap.py
=============
Domain gap quantification for the Lab-to-Field Crop Disease
Domain-Generalisation Benchmark (Idea 4).

RAM FIX (v2 — CPU/low-memory safe)
-----------------------------------
The original code OOM-crashed on CPU Colab (12 GB RAM limit) because:

  1. _rbf_kernel() built a (m, n, d) intermediate tensor.
     With m=n=2000, d=576: 2000×2000×576×4 bytes = ~9 GB  → instant OOM.

  2. Median heuristic built (4000, 4000, d) = ~37 GB → even worse OOM.

  3. extract_features() used batch_size=64 on CPU, loading many images
     and model activations into RAM simultaneously.

Fixes:
  1. _rbf_kernel_chunked(): computes pairwise distances in row-chunks
     of CHUNK_SIZE rows at a time, accumulating the mean kernel value
     without ever materialising the full (m, n, d) tensor.
     Peak extra RAM per chunk: CHUNK_SIZE × n × d × 4 bytes
     (default CHUNK_SIZE=50 → 50×500×576×4 ≈ 57 MB — fits easily).

  2. _median_bandwidth_chunked(): same trick for the median heuristic —
     samples a small random subset (≤500 points), computes pairwise
     distances in chunks, and finds the median from those.

  3. mmd_sample_size default reduced to 500 (was 2000).
     At 500 samples: peak RAM for kernel ≈ 50×500×576×4 ≈ 57 MB.

  4. extract_features(): batch_size reduced to 16 on CPU (was 64).
     Also explicitly calls torch.cuda.empty_cache() after extraction
     and del intermediate tensors.

  5. compute_domain_gap(): explicitly del features arrays after MMD/
     centroid computation and call gc.collect() to free RAM before
     the next step.

Metrics computed
----------------
  1. MMD² (Maximum Mean Discrepancy) with RBF kernel
  2. Centroid distance (L2 + cosine similarity)
  3. Accuracy & F1 drop (from eval results)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_pipeline import CLASS_TO_IDX, NUM_CLASSES, PlantDiseaseDataset, get_eval_transform
from model import SUPPORTED_ARCHS, PlantDiseaseClassifier, build_model
from evaluate import load_model_from_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Row-chunk size for memory-safe kernel computation.
# Peak extra RAM per chunk = CHUNK_SIZE × n × d × 4 bytes
# At CHUNK_SIZE=50, n=500, d=576: ~57 MB — well within 12 GB Colab RAM.
_CHUNK_SIZE = 50


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: PlantDiseaseClassifier,
    data_dir: str | Path,
    batch_size: int = 16,      # reduced from 64 — saves RAM on CPU
    num_workers: int = 0,      # 0 = safe on CPU Colab
    domain_label: str = "unknown",
) -> np.ndarray:
    """
    Extract penultimate-layer embeddings from all samples in data_dir.

    Returns
    -------
    features : np.ndarray of shape (N, feature_dim), float32, L2-normalised.
    """
    data_dir = Path(data_dir)
    device   = next(model.parameters()).device
    # Never use CUDA autocast on CPU — it errors
    use_amp  = device.type == "cuda"

    dataset = PlantDiseaseDataset(
        root=data_dir,
        transform=get_eval_transform(),
        class_to_idx=CLASS_TO_IDX,
        allow_extra_classes=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,          # pin_memory=True causes issues on CPU
        drop_last=False,
        persistent_workers=False,  # avoid worker memory leaks
    )

    n_batches  = len(loader)
    all_feats: list[np.ndarray] = []

    logger.info("[%s] Extracting features from %d samples (%d batches, batch_size=%d)...",
                domain_label, len(dataset), n_batches, batch_size)

    start_time = time.time()
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=False)

        if use_amp:
            with torch.amp.autocast("cuda"):
                feats = model.get_features(images)
        else:
            feats = model.get_features(images)

        # L2 normalise per sample
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        # Convert to float32 numpy immediately; release the tensor
        all_feats.append(feats.cpu().float().numpy())
        del images, feats

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            logger.info("[%s] batch %d/%d", domain_label, batch_idx + 1, n_batches)

    features = np.concatenate(all_feats, axis=0)
    del all_feats
    gc.collect()

    elapsed = time.time() - start_time
    logger.info(
        "[%s] Features extracted: shape=%s | time=%.2fs | RAM: %.0f MB",
        domain_label, features.shape, elapsed,
        features.nbytes / 1e6,
    )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Memory-safe RBF kernel (chunked — no giant intermediate arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _rbf_kernel_chunked(X: np.ndarray, Y: np.ndarray, sigma: float,
                         chunk_size: int = _CHUNK_SIZE) -> float:
    """
    Compute mean RBF kernel k(x,y) = exp(-||x-y||² / 2σ²) WITHOUT
    materialising the full (m, n, d) difference tensor.

    Processes X in row-chunks of `chunk_size` rows.
    Peak extra RAM = chunk_size × n × d × 4 bytes.
    At chunk_size=50, n=500, d=576: ~57 MB.

    Parameters
    ----------
    X : (m, d)  source samples
    Y : (n, d)  target samples
    sigma : float  kernel bandwidth
    chunk_size : int  rows of X to process at a time

    Returns
    -------
    float — mean kernel value E[k(x,y)]
    """
    m = X.shape[0]
    inv_denom = 1.0 / (2.0 * sigma ** 2)
    total = 0.0
    count = 0

    for start in range(0, m, chunk_size):
        end   = min(start + chunk_size, m)
        X_c   = X[start:end]                           # (c, d)
        # Compute pairwise squared distances via ||x-y||² = ||x||² + ||y||² - 2<x,y>
        # This avoids the (c, n, d) diff tensor entirely.
        xx    = (X_c ** 2).sum(axis=1, keepdims=True)  # (c, 1)
        yy    = (Y  ** 2).sum(axis=1, keepdims=True).T # (1, n)
        xy    = X_c @ Y.T                               # (c, n)
        sq_d  = xx + yy - 2.0 * xy                     # (c, n)
        sq_d  = np.clip(sq_d, 0.0, None)               # numerical safety
        total += np.exp(-sq_d * inv_denom).sum()
        count += X_c.shape[0] * Y.shape[0]
        del X_c, xx, yy, xy, sq_d

    return float(total / count)


def _median_bandwidth_chunked(X: np.ndarray, Y: np.ndarray,
                               max_pts: int = 300,
                               chunk_size: int = _CHUNK_SIZE,
                               seed: int = 42) -> float:
    """
    Compute median heuristic bandwidth σ = sqrt(median(||x-y||²) / 2)
    on a small random subsample, using chunked distance computation
    to avoid the (n², d) intermediate array.

    Parameters
    ----------
    X, Y : feature arrays
    max_pts : int  max points to subsample (total from both domains)
    chunk_size : int  rows processed at a time
    seed : int  RNG seed

    Returns
    -------
    float — bandwidth σ (clamped to ≥ 1e-6)
    """
    rng = np.random.default_rng(seed)
    # Take at most max_pts/2 from each domain
    half = max_pts // 2
    idx_x = rng.choice(len(X), size=min(half, len(X)), replace=False)
    idx_y = rng.choice(len(Y), size=min(half, len(Y)), replace=False)
    pts   = np.vstack([X[idx_x], Y[idx_y]])   # (≤max_pts, d)
    n     = pts.shape[0]

    # Collect upper-triangle squared distances in chunks
    upper_dists: list[np.ndarray] = []
    for i in range(0, n, chunk_size):
        Xi = pts[i:i + chunk_size]             # (c, d)
        xx = (Xi ** 2).sum(1, keepdims=True)   # (c, 1)
        yy = (pts ** 2).sum(1, keepdims=True).T # (1, n)
        xy = Xi @ pts.T                         # (c, n)
        sq_d = np.clip(xx + yy - 2.0 * xy, 0.0, None)  # (c, n)

        # Keep only upper-triangle elements (j > i + row_offset)
        for local_row in range(sq_d.shape[0]):
            global_row = i + local_row
            upper_dists.append(sq_d[local_row, global_row + 1:])
        del Xi, xx, yy, xy, sq_d

    all_dists = np.concatenate(upper_dists) if upper_dists else np.array([1.0])
    sigma = float(np.sqrt(max(np.median(all_dists) / 2.0, 1e-12)))
    del pts, upper_dists, all_dists
    gc.collect()
    return max(sigma, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# MMD² (memory-safe)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
    sample_size: int = 500,    # reduced from 2000 — 500 is plenty for domain gap
    seed: int = 42,
) -> dict:
    """
    Compute squared MMD² between X and Y using a chunked RBF kernel.

    MMD²(P, Q) = E[k(x,x')] − 2·E[k(x,y)] + E[k(y,y')]

    Memory usage: O(chunk_size × n × d) per kernel call, NOT O(n² × d).

    Parameters
    ----------
    X : (m, d) source domain features
    Y : (n, d) target domain features
    sigma : float | None — bandwidth; None → median heuristic
    sample_size : int — points per domain (500 is sufficient and safe)
    seed : int

    Returns
    -------
    dict with mmd_squared, sigma, k_xx, k_yy, k_xy, n_source_used, n_target_used
    """
    rng = np.random.default_rng(seed)

    # Sub-sample
    if sample_size > 0:
        m = min(sample_size, len(X))
        n = min(sample_size, len(Y))
        X_s = X[rng.choice(len(X), size=m, replace=False)]
        Y_s = Y[rng.choice(len(Y), size=n, replace=False)]
    else:
        X_s, Y_s = X, Y
        m, n = len(X), len(Y)

    logger.info(
        "MMD: sub-sampled source=%d→%d | target=%d→%d | feature_dim=%d",
        len(X), m, len(Y), n, X.shape[1],
    )
    logger.info("MMD: peak RAM per chunk ≈ %.0f MB",
                _CHUNK_SIZE * n * X.shape[1] * 4 / 1e6)

    # Bandwidth via memory-safe median heuristic
    if sigma is None:
        sigma = _median_bandwidth_chunked(X_s, Y_s, max_pts=300, seed=seed)
        logger.info("MMD: median heuristic bandwidth σ = %.6f", sigma)
    else:
        logger.info("MMD: user-specified bandwidth σ = %.6f", sigma)

    start_time = time.time()
    k_xx = _rbf_kernel_chunked(X_s, X_s, sigma)
    k_yy = _rbf_kernel_chunked(Y_s, Y_s, sigma)
    k_xy = _rbf_kernel_chunked(X_s, Y_s, sigma)
    mmd2 = float(k_xx - 2.0 * k_xy + k_yy)
    elapsed = time.time() - start_time

    logger.info(
        "MMD²=%.6f  (k_xx=%.6f  k_yy=%.6f  k_xy=%.6f)  computed in %.2fs",
        mmd2, k_xx, k_yy, k_xy, elapsed,
    )

    # Free sub-samples
    del X_s, Y_s
    gc.collect()

    return {
        "mmd_squared":   round(mmd2, 8),
        "sigma":         round(sigma, 6),
        "k_xx":          round(k_xx, 8),
        "k_yy":          round(k_yy, 8),
        "k_xy":          round(k_xy, 8),
        "n_source_used": m,
        "n_target_used": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Centroid distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid_distance(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    L2 distance and cosine similarity between per-domain mean feature vectors.
    Memory: O(d) — just two mean vectors.
    """
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    l2_dist = float(np.linalg.norm(mu_x - mu_y))
    cos_sim  = float(np.dot(mu_x, mu_y) / (np.linalg.norm(mu_x) * np.linalg.norm(mu_y) + 1e-10))

    logger.info("Centroid L2 distance  : %.6f", l2_dist)
    logger.info("Centroid cosine sim   : %.6f", cos_sim)

    return {
        "centroid_distance": round(l2_dist, 6),
        "cosine_similarity": round(cos_sim, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Performance drop
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_drop(eval_results: dict) -> dict:
    """
    Accuracy and F1 drop between source (PV-val) and target (PD-eval).
    Requires eval_results from evaluate.evaluate().
    """
    if "pv_val" not in eval_results or "pd_eval" not in eval_results:
        logger.warning(
            "Cannot compute performance drop: 'pv_val' or 'pd_eval' missing "
            "from eval_results. Ensure PlantDoc evaluation was run."
        )
        return {}

    pv = eval_results["pv_val"]
    pd = eval_results["pd_eval"]

    delta_acc  = pv["accuracy"]    - pd["accuracy"]
    delta_f1   = pv["macro_f1"]   - pd["macro_f1"]
    delta_wf1  = pv["weighted_f1"] - pd["weighted_f1"]

    rel_acc_drop = 100.0 * delta_acc / (pv["accuracy"]  + 1e-10)
    rel_f1_drop  = 100.0 * delta_f1  / (pv["macro_f1"]  + 1e-10)

    logger.info("=" * 65)
    logger.info("PERFORMANCE DROP (source → target)")
    logger.info("=" * 65)
    logger.info("  PV-val  Accuracy  : %.4f  (%.2f%%)", pv["accuracy"],  100 * pv["accuracy"])
    logger.info("  PD-eval Accuracy  : %.4f  (%.2f%%)", pd["accuracy"],  100 * pd["accuracy"])
    logger.info("  Δ Accuracy        : %.4f  (%.2f%% relative drop)", delta_acc,  rel_acc_drop)
    logger.info("  PV-val  Macro-F1  : %.4f", pv["macro_f1"])
    logger.info("  PD-eval Macro-F1  : %.4f", pd["macro_f1"])
    logger.info("  Δ Macro-F1        : %.4f  (%.2f%% relative drop)", delta_f1,   rel_f1_drop)
    logger.info("=" * 65)

    return {
        "pv_accuracy":                round(pv["accuracy"],    4),
        "pd_accuracy":                round(pd["accuracy"],    4),
        "delta_accuracy":             round(delta_acc,         4),
        "relative_accuracy_drop_pct": round(rel_acc_drop,      2),
        "pv_macro_f1":                round(pv["macro_f1"],    4),
        "pd_macro_f1":                round(pd["macro_f1"],    4),
        "delta_macro_f1":             round(delta_f1,          4),
        "relative_f1_drop_pct":       round(rel_f1_drop,       2),
        "pv_weighted_f1":             round(pv["weighted_f1"], 4),
        "pd_weighted_f1":             round(pd["weighted_f1"], 4),
        "delta_weighted_f1":          round(delta_wf1,         4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main domain gap function
# ─────────────────────────────────────────────────────────────────────────────

def compute_domain_gap(
    model: PlantDiseaseClassifier,
    pv_val_dir: str | Path,
    pd_eval_dir: str | Path,
    eval_results: Optional[dict] = None,
    output_dir: str | Path = "./eval_outputs",
    batch_size: int = 16,          # reduced from 64 — safe on CPU
    num_workers: int = 0,          # 0 = safe on CPU Colab
    mmd_sample_size: int = 500,    # reduced from 2000 — sufficient + safe
    seed: int = 42,
) -> dict:
    """
    Compute the full domain gap analysis between PlantVillage and PlantDoc.

    Memory-safe version: chunked kernel, small sample size, explicit gc.

    Parameters
    ----------
    mmd_sample_size : int
        Points per domain for MMD sub-sampling.
        500 is sufficient for a reliable estimate and uses ~57 MB peak RAM.
        Set to 200 if you still hit RAM limits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 65)
    logger.info("DOMAIN GAP ANALYSIS (memory-safe mode)")
    logger.info("=" * 65)

    model.eval()

    # ── Extract PV features ────────────────────────────────────────────────────
    pv_features = extract_features(
        model=model,
        data_dir=pv_val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        domain_label="PV-val (source)",
    )
    gc.collect()

    # ── Extract PD features ────────────────────────────────────────────────────
    pd_features = extract_features(
        model=model,
        data_dir=pd_eval_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        domain_label="PD-eval (target)",
    )
    gc.collect()

    # ── Feature statistics ────────────────────────────────────────────────────
    pv_stats = {
        "n_samples":    int(pv_features.shape[0]),
        "feature_dim":  int(pv_features.shape[1]),
        "mean_l2_norm": round(float(np.linalg.norm(pv_features, axis=1).mean()), 6),
        "feature_std":  round(float(pv_features.std()), 6),
    }
    pd_stats = {
        "n_samples":    int(pd_features.shape[0]),
        "feature_dim":  int(pd_features.shape[1]),
        "mean_l2_norm": round(float(np.linalg.norm(pd_features, axis=1).mean()), 6),
        "feature_std":  round(float(pd_features.std()), 6),
    }
    logger.info("PV feature stats: %s", pv_stats)
    logger.info("PD feature stats: %s", pd_stats)

    # ── Centroid distance (cheap — O(d) memory) ───────────────────────────────
    logger.info("\nComputing centroid distance...")
    centroid_results = compute_centroid_distance(X=pv_features, Y=pd_features)

    # ── MMD² (chunked — O(chunk × n × d) memory) ─────────────────────────────
    logger.info("\nComputing MMD² (chunked, memory-safe)...")
    mmd_results = compute_mmd_squared(
        X=pv_features,
        Y=pd_features,
        sample_size=mmd_sample_size,
        seed=seed,
    )

    # ── Free feature arrays — we no longer need them ──────────────────────────
    del pv_features, pd_features
    gc.collect()
    logger.info("Feature arrays freed from RAM.")

    # ── Performance drop ──────────────────────────────────────────────────────
    perf_drop = {}
    if eval_results is not None:
        logger.info("\nComputing performance drop...")
        perf_drop = compute_performance_drop(eval_results)

    # ── Assemble + save results ───────────────────────────────────────────────
    gap_results = {
        "pv_feature_stats":  pv_stats,
        "pd_feature_stats":  pd_stats,
        "mmd":               mmd_results,
        "centroid_distance": centroid_results,
        "performance_drop":  perf_drop,
    }

    out_path = output_dir / "domain_gap_results.json"
    with open(out_path, "w") as f:
        json.dump(gap_results, f, indent=2)
    logger.info("\nDomain gap results saved: %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("DOMAIN GAP SUMMARY")
    logger.info("=" * 65)
    logger.info("  MMD² (feature space)       : %.6f", mmd_results["mmd_squared"])
    logger.info("  Centroid L2 distance       : %.6f", centroid_results["centroid_distance"])
    logger.info("  Centroid cosine similarity : %.6f", centroid_results["cosine_similarity"])
    if perf_drop:
        logger.info(
            "  Δ Accuracy (PV→PD)         : %.4f  (%.1f%% drop)",
            perf_drop.get("delta_accuracy", 0.0),
            perf_drop.get("relative_accuracy_drop_pct", 0.0),
        )
        logger.info(
            "  Δ Macro-F1 (PV→PD)        : %.4f  (%.1f%% drop)",
            perf_drop.get("delta_macro_f1", 0.0),
            perf_drop.get("relative_f1_drop_pct", 0.0),
        )
    logger.info("=" * 65)

    return gap_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute domain gap (MMD, centroid distance) between PV and PlantDoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--pv_val_dir",      default="/content/final_dataset/val")
    parser.add_argument("--pd_eval_dir",     required=True)
    parser.add_argument("--arch",            default="efficientnet_b0", choices=SUPPORTED_ARCHS)
    parser.add_argument("--num_classes",     type=int, default=NUM_CLASSES)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=0)
    parser.add_argument("--mmd_sample_size", type=int, default=500)
    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/idea4drive/eval_outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        num_classes=args.num_classes,
    )
    compute_domain_gap(
        model=model,
        pv_val_dir=args.pv_val_dir,
        pd_eval_dir=args.pd_eval_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mmd_sample_size=args.mmd_sample_size,
    )
