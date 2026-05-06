"""
benchmark_comparison.py
=======================
Standalone script to compare baseline vs. fine-tuned model results.

Loads pre-saved JSON evaluation outputs from Google Drive, extracts key metrics,
and produces a formatted comparison table saved as JSON and CSV.

Usage (Colab cell):
    exec(open('/content/drive/MyDrive/idea4drive/benchmark_comparison.py').read())

Or as a standalone cell — just paste the contents.
"""

import json
import os
import csv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust paths if needed
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_BASE = "/content/drive/MyDrive/idea4drive"

PATHS = {
    "baseline": {
        "eval":     os.path.join(DRIVE_BASE, "eval_outputs",    "benchmark_summary.json"),
        "gap":      os.path.join(DRIVE_BASE, "eval_outputs",    "domain_gap_results.json"),
    },
    "finetuned": {
        "eval":     os.path.join(DRIVE_BASE, "eval_outputs_ft", "benchmark_summary.json"),
        "gap":      os.path.join(DRIVE_BASE, "eval_outputs_ft", "domain_gap_results.json"),
    },
}

OUTPUT_DIR = os.path.join(DRIVE_BASE, "eval_outputs")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "comparison_table.json")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "comparison_table.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    """Load a JSON file; return None if file does not exist."""
    if not os.path.isfile(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def pct(value):
    """Convert fraction to percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt(value, decimals=4):
    """Format a float or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# METRIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(eval_json, gap_json, label):
    """
    Pull the key benchmark metrics from loaded JSON dicts.

    Returns a dict with all metrics for one method (baseline or finetuned).
    """
    # ── Evaluation metrics ────────────────────────────────────────────────────
    pv_acc   = safe_get(eval_json, "pv_val",  "accuracy")
    pd_acc   = safe_get(eval_json, "pd_eval", "accuracy")
    pv_f1    = safe_get(eval_json, "pv_val",  "macro_f1")
    pd_f1    = safe_get(eval_json, "pd_eval", "macro_f1")
    pv_wf1   = safe_get(eval_json, "pv_val",  "weighted_f1")
    pd_wf1   = safe_get(eval_json, "pd_eval", "weighted_f1")
    pv_n     = safe_get(eval_json, "pv_val",  "n_samples")
    pd_n     = safe_get(eval_json, "pd_eval", "n_samples")

    # ── Domain gap metrics ────────────────────────────────────────────────────
    delta_acc   = None
    delta_f1    = None
    rel_acc_pct = None
    rel_f1_pct  = None
    mmd2        = None
    centroid_l2 = None
    cos_sim     = None

    # Try loading from the gap JSON first (more authoritative)
    if gap_json is not None:
        drop = safe_get(gap_json, "performance_drop", default={})
        delta_acc   = safe_get(drop, "delta_accuracy")
        delta_f1    = safe_get(drop, "delta_macro_f1")
        rel_acc_pct = safe_get(drop, "relative_accuracy_drop_pct")
        rel_f1_pct  = safe_get(drop, "relative_f1_drop_pct")
        mmd2        = safe_get(gap_json, "mmd",               "mmd_squared")
        centroid_l2 = safe_get(gap_json, "centroid_distance", "centroid_distance")
        cos_sim     = safe_get(gap_json, "centroid_distance", "cosine_similarity")

    # Fallback: compute delta from eval JSON if not in gap file
    if delta_acc is None and pv_acc is not None and pd_acc is not None:
        delta_acc = round(pv_acc - pd_acc, 4)
    if delta_f1 is None and pv_f1 is not None and pd_f1 is not None:
        delta_f1 = round(pv_f1 - pd_f1, 4)
    if rel_acc_pct is None and pv_acc and delta_acc is not None:
        rel_acc_pct = round(100.0 * delta_acc / (pv_acc + 1e-10), 2)
    if rel_f1_pct is None and pv_f1 and delta_f1 is not None:
        rel_f1_pct = round(100.0 * delta_f1 / (pv_f1 + 1e-10), 2)

    return {
        "method":           label,
        "pv_accuracy":      pv_acc,
        "pd_accuracy":      pd_acc,
        "delta_accuracy":   delta_acc,
        "rel_acc_drop_pct": rel_acc_pct,
        "pv_macro_f1":      pv_f1,
        "pd_macro_f1":      pd_f1,
        "delta_macro_f1":   delta_f1,
        "rel_f1_drop_pct":  rel_f1_pct,
        "pv_weighted_f1":   pv_wf1,
        "pd_weighted_f1":   pd_wf1,
        "mmd_squared":      mmd2,
        "centroid_l2":      centroid_l2,
        "cosine_similarity": cos_sim,
        "pv_n_samples":     pv_n,
        "pd_n_samples":     pd_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(rows):
    """Print a clean side-by-side comparison table."""
    labels   = [r["method"] for r in rows]
    col_w    = max(20, max(len(l) for l in labels) + 2)
    hdr_w    = 32

    def row_line(metric_label, values):
        vals = "".join(f"{v:>{col_w}}" for v in values)
        print(f"  {metric_label:<{hdr_w}}{vals}")

    sep = "=" * (hdr_w + 2 + col_w * len(rows))
    print()
    print(sep)
    print("  BENCHMARK COMPARISON — Source-Only vs. Fine-Tuned")
    print(sep)

    # Header
    hdr = "".join(f"{l:>{col_w}}" for l in labels)
    print(f"  {'Metric':<{hdr_w}}{hdr}")
    print("-" * (hdr_w + 2 + col_w * len(rows)))

    row_line("PV Accuracy (source, in-dist)",
             [pct(r["pv_accuracy"]) for r in rows])
    row_line("PD Accuracy (target, OOD)",
             [pct(r["pd_accuracy"]) for r in rows])
    row_line("Δ Accuracy (domain gap)",
             [fmt(r["delta_accuracy"]) for r in rows])
    row_line("Relative Accuracy Drop (%)",
             [f"{r['rel_acc_drop_pct']:.1f}%" if r['rel_acc_drop_pct'] is not None else "N/A"
              for r in rows])

    print()
    row_line("PV Macro-F1 (source)",
             [fmt(r["pv_macro_f1"]) for r in rows])
    row_line("PD Macro-F1 (target)",
             [fmt(r["pd_macro_f1"]) for r in rows])
    row_line("Δ Macro-F1",
             [fmt(r["delta_macro_f1"]) for r in rows])
    row_line("Relative F1 Drop (%)",
             [f"{r['rel_f1_drop_pct']:.1f}%" if r['rel_f1_drop_pct'] is not None else "N/A"
              for r in rows])

    print()
    row_line("MMD² (feature-space gap)",
             [fmt(r["mmd_squared"], 6) for r in rows])
    row_line("Centroid L2 distance",
             [fmt(r["centroid_l2"], 6) for r in rows])
    row_line("Centroid cosine similarity",
             [fmt(r["cosine_similarity"], 6) for r in rows])

    print()
    row_line("PV samples",
             [str(r["pv_n_samples"]) if r["pv_n_samples"] else "N/A" for r in rows])
    row_line("PD samples",
             [str(r["pd_n_samples"]) if r["pd_n_samples"] else "N/A" for r in rows])

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_json(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  JSON saved → {path}")


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading evaluation JSONs...")

    results = []
    for method_key, method_label in [("baseline", "Source-Only (Baseline)"),
                                      ("finetuned", "Fine-Tuned (PlantDoc)")]:
        eval_path = PATHS[method_key]["eval"]
        gap_path  = PATHS[method_key]["gap"]

        eval_data = load_json(eval_path)
        gap_data  = load_json(gap_path)

        if eval_data is None:
            print(f"  [SKIP] No eval data found for '{method_label}' — skipping.")
            continue

        metrics = extract_metrics(eval_data, gap_data, label=method_label)
        results.append(metrics)

    if not results:
        print("\nERROR: No evaluation data loaded. Check that JSON files exist in Drive.")
        return

    # Print table
    print_comparison_table(results)

    # Save outputs
    print("Saving outputs...")
    save_json(results, OUTPUT_JSON)
    save_csv(results,  OUTPUT_CSV)
    print("\nDone.")


if __name__ == "__main__":
    main()

# Also run immediately when exec()'d in a Colab cell
