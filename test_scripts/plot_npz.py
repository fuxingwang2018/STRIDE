#!/usr/bin/env python3
"""
plot_npz.py — Read and visualise the contents of a .npz file.

Usage:
    python plot_npz.py path/to/file.npz [--keys key1 key2] [--out output_dir]

Examples:
    python plot_npz.py data.npz
    python plot_npz.py data.npz --keys precip temperature
    python plot_npz.py data.npz --out ./plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}
    print(f"\n📦  Loaded '{path}'  —  {len(arrays)} array(s) found:")
    for k, v in arrays.items():
        print(f"   {k!r:30s}  shape={str(v.shape):20s}  dtype={v.dtype}")
    print()
    return arrays


def select_keys(arrays: dict, requested: list[str] | None) -> dict:
    if not requested:
        return arrays
    missing = [k for k in requested if k not in arrays]
    if missing:
        print(f"⚠️  Keys not found and skipped: {missing}")
    return {k: arrays[k] for k in requested if k in arrays}


def infer_plot_type(arr: np.ndarray) -> str:
    """Choose the best plot type based on array shape."""
    if arr.ndim == 0:
        return "scalar"
    if arr.ndim == 1:
        return "line" if arr.size > 1 else "scalar"
    if arr.ndim == 2:
        return "heatmap"
    if arr.ndim == 3:
        # (C, H, W) or (T, H, W) — treat as stack of 2-D slices
        return "slices"
    if arr.ndim == 4:
        # (T, C, H, W) — show first few time steps
        return "slices4d"
    return "unsupported"


# ─────────────────────────────────────────────────────────────────────────────
# Per-type plotters
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalar(ax, arr, title):
    ax.axis("off")
    ax.text(0.5, 0.5, f"{arr.item():.6g}",
            ha="center", va="center", fontsize=22, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title(title, fontsize=11)


def plot_line(ax, arr, title):
    ax.plot(arr, linewidth=1.5, color="#2196F3")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _add_stats(ax, arr)


def plot_heatmap(ax, arr, title):
    vmin, vmax = arr.min(), arr.max()
    norm = CenteredNorm(vcenter=0, halfrange=max(abs(vmin), abs(vmax))) if vmin < 0 < vmax else None
    cmap = "RdBu_r" if norm else "viridis"
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else vmin,
                   vmax=None if norm else vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    ax.set_title(f"{title}  [{arr.shape[0]}×{arr.shape[1]}]", fontsize=11)
    _add_stats_title(ax, arr)


def plot_slices(fig, axes_row, arr, title):
    """Plot up to len(axes_row) slices from a 3-D array."""
    n_show = min(len(axes_row), arr.shape[0])
    vmin, vmax = arr.min(), arr.max()
    norm = CenteredNorm(vcenter=0, halfrange=max(abs(vmin), abs(vmax))) if vmin < 0 < vmax else None
    cmap = "RdBu_r" if norm else "viridis"
    for i in range(len(axes_row)):
        ax = axes_row[i]
        if i < n_show:
            im = ax.imshow(arr[i], aspect="auto", cmap=cmap, norm=norm,
                           vmin=None if norm else vmin,
                           vmax=None if norm else vmax,
                           interpolation="nearest")
            ax.set_title(f"{title}[{i}]", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.axis("off")


def plot_slices4d(fig, axes_rows, arr, title):
    """Plot arr[t, 0] for the first few time steps."""
    for t_idx, axes_row in enumerate(axes_rows):
        if t_idx >= arr.shape[0]:
            for ax in axes_row:
                ax.axis("off")
            continue
        plot_slices(fig, axes_row, arr[t_idx], f"{title}[t={t_idx}]")


# ─────────────────────────────────────────────────────────────────────────────
# Stats annotations
# ─────────────────────────────────────────────────────────────────────────────

def _add_stats(ax, arr):
    a = arr.ravel()
    stats = f"min={a.min():.3g}  max={a.max():.3g}  mean={a.mean():.3g}  std={a.std():.3g}"
    ax.annotate(stats, xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=7, color="gray", va="bottom")


def _add_stats_title(ax, arr):
    a = arr.ravel()
    extra = f"min={a.min():.3g}  max={a.max():.3g}  mean={a.mean():.3g}  std={a.std():.3g}"
    current = ax.get_title()
    ax.set_title(f"{current}\n{extra}", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

MAX_SLICES = 4   # max 2-D panels to show per 3-D array

def build_figure(arrays: dict) -> list[plt.Figure]:
    """Return one Figure per array key (easier to save / inspect)."""
    figs = []

    for key, arr in arrays.items():
        ptype = infer_plot_type(arr)

        if ptype == "scalar":
            fig, ax = plt.subplots(figsize=(3, 2))
            plot_scalar(ax, arr, key)

        elif ptype == "line":
            fig, ax = plt.subplots(figsize=(8, 3))
            plot_line(ax, arr, key)

        elif ptype == "heatmap":
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_heatmap(ax, arr, key)

        elif ptype == "slices":
            n_show = min(MAX_SLICES, arr.shape[0])
            fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4),
                                     squeeze=False)
            plot_slices(fig, axes[0], arr, key)

        elif ptype == "slices4d":
            n_t = min(MAX_SLICES, arr.shape[0])
            n_c = min(MAX_SLICES, arr.shape[1])
            fig, axes = plt.subplots(n_t, n_c,
                                     figsize=(4 * n_c, 4 * n_t),
                                     squeeze=False)
            plot_slices4d(fig, axes, arr[:n_t, :n_c], key)

        else:
            print(f"⚠️  '{key}' has unsupported ndim={arr.ndim}, skipping.")
            continue

        fig.suptitle(f"Key: '{key}'  |  shape={arr.shape}  dtype={arr.dtype}",
                     fontsize=10, y=1.01)
        fig.tight_layout()
        figs.append((key, fig))

    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Read and plot a .npz file.")
    parser.add_argument("npz_path", help="Path to the .npz file")
    parser.add_argument("--keys", nargs="*", default=None,
                        help="Specific array key(s) to plot (default: all)")
    parser.add_argument("--out", default=None,
                        help="Output directory for PNG files (default: show interactively)")
    args = parser.parse_args()

    arrays = load_npz(args.npz_path)
    arrays = select_keys(arrays, args.keys)

    if not arrays:
        print("No arrays to plot. Exiting.")
        sys.exit(1)

    figs = build_figure(arrays)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.npz_path).stem
        for key, fig in figs:
            safe_key = key.replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"{stem}__{safe_key}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  ✅  Saved: {out_path}")
        print(f"\nAll plots saved to '{args.out}'")
    else:
        plt.show()


if __name__ == "__main__":
    main()
