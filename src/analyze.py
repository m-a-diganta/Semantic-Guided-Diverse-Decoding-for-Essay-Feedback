"""Produce the figures and tables for the report from results/summary.json.

Outputs into report/figures/ and report/tables/ — directly slottable into
the LaTeX document via \\includegraphics and \\input.

Usage:
    python -m src.analyze                          # default paths
    python -m src.analyze --summary results/summary.json --out-dir report

What it produces:
    report/figures/method_comparison.pdf   — 3-panel bar chart (main figure)
    report/figures/coverage_by_score.pdf   — coverage broken down by essay score
    report/tables/method_comparison.tex    — booktabs LaTeX table
    report/tables/method_comparison.csv    — same table as CSV
    stdout — a markdown-formatted summary for pasting into chat/messages

Plots use a colour-blind-safe palette and Times-like font sizing to match
the rendered LaTeX document (11 pt body text, A4 column width).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---- Methods + display order ---------------------------------------------


METHOD_ORDER = ["greedy", "temperature", "diverse_beam", "semdid"]
METHOD_LABELS = {
    "greedy": "Greedy",
    "temperature": "Temperature",
    "diverse_beam": "Diverse Beam",
    "semdid": "SemDiD",
}
# Coordinated palette — each method has a stable colour across every figure.
# Greedy gray (neutral), temperature blue, diverse_beam orange (highlight
# the failure), SemDiD green (hero).
METHOD_COLOURS = {
    "greedy":       "#888888",
    "temperature":  "#1f77b4",
    "diverse_beam": "#ff7f0e",
    "semdid":       "#2ca02c",
}


# ---- Style ---------------------------------------------------------------


def set_plot_style() -> None:
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


# ---- Data loading + aggregation -------------------------------------------


def load_summary(path: Path) -> pd.DataFrame:
    """Load results/summary.json into a flat DataFrame."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.json_normalize(payload["rows"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method aggregate stats. Reindexed to METHOD_ORDER for stable display."""
    agg = df.groupby("method").agg(
        n=("essay_id", "count"),
        div_mean=("diversity_mean_sim", "mean"),
        div_std=("diversity_mean_sim", "std"),
        cov_mean=("n_dimensions_union", "mean"),
        cov_std=("n_dimensions_union", "std"),
        judge_mean=("judge_mean", "mean"),
        judge_std=("judge_mean", "std"),
        gen_mean=("t_gen_s", "mean"),
    )
    return agg.reindex(METHOD_ORDER).round(3)


# ---- Figures -------------------------------------------------------------


def fig_method_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Three-panel bar chart: diversity, rubric coverage, LLM-judge.

    Greedy has no diversity (k=1) — its bar is omitted from panel 1 and
    annotated as 'k=1' so the omission is intentional, not missing data.
    """
    agg = aggregate(df)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    # --- Panel 1: intrinsic diversity (mean pairwise cosine similarity)
    methods_div = [m for m in METHOD_ORDER if not np.isnan(agg.loc[m, "div_mean"])]
    means = [agg.loc[m, "div_mean"] for m in methods_div]
    errs = [agg.loc[m, "div_std"] for m in methods_div]
    cols = [METHOD_COLOURS[m] for m in methods_div]
    bars = axes[0].bar(range(len(methods_div)), means, yerr=errs, capsize=4, color=cols)
    axes[0].set_xticks(range(len(methods_div)))
    axes[0].set_xticklabels([METHOD_LABELS[m] for m in methods_div], rotation=20, ha="right")
    axes[0].set_ylabel("Mean pairwise cosine similarity")
    axes[0].set_title("(a) Intrinsic diversity\n(lower = more diverse)")
    axes[0].set_ylim(0, 1.08)
    axes[0].axhline(0.85, ls="--", color="black", alpha=0.4, lw=0.8)
    axes[0].text(0.02, 0.86, "paraphrase threshold", fontsize=8, alpha=0.6,
                 transform=axes[0].get_yaxis_transform())
    # Greedy callout
    if "greedy" not in methods_div:
        axes[0].text(-0.5, 0.05, "Greedy: k=1\n(diversity undefined)",
                     fontsize=8, alpha=0.6)
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                     f"{m:.2f}", ha="center", fontsize=8)

    # --- Panel 2: rubric coverage (union of dimensions across k variants)
    means = [agg.loc[m, "cov_mean"] for m in METHOD_ORDER]
    errs = [agg.loc[m, "cov_std"] for m in METHOD_ORDER]
    cols = [METHOD_COLOURS[m] for m in METHOD_ORDER]
    bars = axes[1].bar(range(len(METHOD_ORDER)), means, yerr=errs, capsize=4, color=cols)
    axes[1].set_xticks(range(len(METHOD_ORDER)))
    axes[1].set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    axes[1].set_ylabel("Dimensions covered (of 4)")
    axes[1].set_title("(b) Rubric coverage\n(higher = more)")
    axes[1].set_ylim(0, 4.4)
    axes[1].axhline(4.0, ls=":", color="black", alpha=0.3, lw=0.8)
    for bar, m in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, m + 0.08,
                     f"{m:.2f}", ha="center", fontsize=8)

    # --- Panel 3: LLM-judge mean score
    means = [agg.loc[m, "judge_mean"] for m in METHOD_ORDER]
    errs = [agg.loc[m, "judge_std"] for m in METHOD_ORDER]
    cols = [METHOD_COLOURS[m] for m in METHOD_ORDER]
    bars = axes[2].bar(range(len(METHOD_ORDER)), means, yerr=errs, capsize=4, color=cols)
    axes[2].set_xticks(range(len(METHOD_ORDER)))
    axes[2].set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    axes[2].set_ylabel("Judge mean score (1–5)")
    axes[2].set_title("(c) Quality (LLM-judge)\n(higher = better)")
    axes[2].set_ylim(0, 5)
    for bar, m in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, m + 0.08,
                     f"{m:.2f}", ha="center", fontsize=8)

    fig.suptitle("Per-method comparison over $n=100$ essays", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_coverage_by_score(df: pd.DataFrame, out_path: Path) -> None:
    """Rubric coverage stratified by holistic essay score, per method.

    Tests whether SemDiD's coverage advantage is consistent across the
    score range or concentrated in one region.
    """
    grouped = (
        df.groupby(["score", "method"])["n_dimensions_union"]
        .mean()
        .unstack("method")
        .reindex(columns=METHOD_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(grouped.index))
    width = 0.2
    for i, m in enumerate(METHOD_ORDER):
        if m not in grouped.columns:
            continue
        ax.bar(x + (i - 1.5) * width, grouped[m].values, width,
               label=METHOD_LABELS[m], color=METHOD_COLOURS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Score {int(s)}" for s in grouped.index])
    ax.set_ylabel("Mean dimensions covered (of 4)")
    ax.set_title("Rubric coverage by holistic essay score")
    ax.set_ylim(0, 4.2)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---- Tables --------------------------------------------------------------


def write_method_table(agg: pd.DataFrame, tex_path: Path, csv_path: Path) -> None:
    """Save the per-method aggregate as a LaTeX booktabs table + CSV."""

    # CSV — flat, easy for Andy / spreadsheets
    agg.to_csv(csv_path)
    print(f"  saved {csv_path}")

    # LaTeX booktabs — drop straight into Results section
    rows = []
    for m in METHOD_ORDER:
        r = agg.loc[m]
        div = (
            f"{r['div_mean']:.2f} ({r['div_std']:.2f})"
            if not np.isnan(r["div_mean"]) else "---"
        )
        cov = f"{r['cov_mean']:.2f} ({r['cov_std']:.2f})"
        judge = f"{r['judge_mean']:.2f} ({r['judge_std']:.2f})"
        rows.append(f"  {METHOD_LABELS[m]:<14} & {int(r['n']):>3} & {div} & {cov} & {judge} \\\\")

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Per-method aggregate metrics over the $n=100$ stratified Exploring Venus sample. "
        "Cells show \\emph{mean (SD)}. Diversity is mean pairwise cosine similarity; "
        "lower indicates more diverse outputs. Coverage is the count of pedagogical dimensions "
        "addressed by at least one of the $k=3$ variants. Judge is the LLM-judge mean of "
        "usefulness, specificity, and actionability on a 1--5 scale.}\n"
        "\\label{tab:method_comparison}\n"
        "\\small\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & $n$ & Diversity & Coverage & Judge \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  saved {tex_path}")


def print_markdown_summary(agg: pd.DataFrame) -> None:
    """Print a markdown summary suitable for pasting into chat / messages."""
    print("\n=== Method comparison (markdown) ===\n")
    print("| Method | n | Diversity (mean ± SD) | Coverage (mean) | Judge (mean ± SD) |")
    print("|---|---|---|---|---|")
    for m in METHOD_ORDER:
        r = agg.loc[m]
        div = (
            f"{r['div_mean']:.3f} ± {r['div_std']:.3f}"
            if not np.isnan(r["div_mean"]) else "—"
        )
        cov = f"{r['cov_mean']:.2f}"
        judge = f"{r['judge_mean']:.3f} ± {r['judge_std']:.3f}"
        print(f"| {METHOD_LABELS[m]} | {int(r['n'])} | {div} | {cov} | {judge} |")


# ---- Driver --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="results/summary.json",
                        help="Path to the summary.json produced by run_local.py")
    parser.add_argument("--out-dir", default="report",
                        help="Output dir; figures/ and tables/ created inside it")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    df = load_summary(summary_path)
    agg = aggregate(df)
    print(f"Loaded {len(df)} rows, {df['method'].nunique()} methods, "
          f"{df['essay_id'].nunique()} unique essays.")

    print("\n=== Per-method aggregate ===")
    print(agg.to_string())

    print("\n=== Saving artefacts ===")
    fig_method_comparison(df, fig_dir / "method_comparison.pdf")
    fig_coverage_by_score(df, fig_dir / "coverage_by_score.pdf")
    write_method_table(agg, tbl_dir / "method_comparison.tex",
                       tbl_dir / "method_comparison.csv")

    print_markdown_summary(agg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
