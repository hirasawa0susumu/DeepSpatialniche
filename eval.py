#!/usr/bin/env python3
"""DeepSpatialniche evaluation — tutorial visualizations + paper quantitative metrics.

Metrics from the paper:
  1. Patch-level cell-type cosine similarity (3D patches, GT vs recon)
  2. Moran's I correlation (top HVGs, GT vs recon)
  3. JS divergence, spatial continuity, label transfer, spatial correlation (supplementary)

Usage:
    python eval.py
    python eval.py --recon ./output/result.h5ad --ground_truth ./data/brain.h5ad --label_key cell_type
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from deepspatial.vis_utils import (
    interactive_3d_expression,
    interactive_3d_labels,
    plot_orthogonal_projections,
    plot_z_distribution,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _v(args_val, cfg_val, default):
    if args_val is not None: return args_val
    if cfg_val is not None: return cfg_val
    return default


def _get_3d_coords(adata):
    """Return (N, 3) array of [x, y, z] in physical coordinates."""
    xy = adata.obsm["spatial"]
    z = adata.obs["z_coord"].to_numpy().reshape(-1, 1)
    return np.column_stack([xy, z])


def _to_dense(adata, gene):
    x = adata[:, gene].X
    return x.toarray().flatten() if issparse(x) else np.asarray(x).flatten()


def _make_palette(adata_recon, adata_gt, label_key):
    all_labels = sorted(
        set(adata_gt.obs[label_key].astype(str).unique())
        | set(adata_recon.obs[label_key].astype(str).unique())
    )
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#3182bd', '#31a354', '#756bb1', '#636363', '#e6550d',
    ]
    return {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(all_labels)}


def _normalize_coords(adata):
    """Return a copy of adata with x, y, z each normalized to [0, 1] for fair visualization."""
    import copy
    v = adata.copy()
    xy = v.obsm["spatial"].copy()
    z = v.obs["z_coord"].to_numpy().copy()
    for i in range(2):
        lo, hi = xy[:, i].min(), xy[:, i].max()
        xy[:, i] = (xy[:, i] - lo) / (hi - lo + 1e-8)
    lo_z, hi_z = z.min(), z.max()
    z = (z - lo_z) / (hi_z - lo_z + 1e-8)
    v.obsm["spatial"] = xy
    v.obs["z_coord"] = z
    return v


# ---------------------------------------------------------------------------
# paper metrics
# ---------------------------------------------------------------------------

def patch_cell_type_cosine(adata_recon, adata_gt, label_key, patch_size=(50, 50, 30)):
    """
    Patch-level cell-type cosine similarity (paper Fig 2c-e).

    1. Grid 3D space into patches of shape `patch_size` (in physical units).
    2. For each patch, build cell-type proportion vectors for GT and recon.
    3. Compute cosine similarity per patch, return mean ± std.
    """
    coords_r = _get_3d_coords(adata_recon)
    coords_g = _get_3d_coords(adata_gt)

    all_types = sorted(set(adata_recon.obs[label_key].astype(str).unique())
                       | set(adata_gt.obs[label_key].astype(str).unique()))
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    n_types = len(all_types)

    labels_r = np.array([type_to_idx[t] for t in adata_recon.obs[label_key].astype(str)])
    labels_g = np.array([type_to_idx[t] for t in adata_gt.obs[label_key].astype(str)])

    # Shared grid origin
    origin = np.minimum(coords_r.min(axis=0), coords_g.min(axis=0))
    dx, dy, dz = patch_size

    similarities = []
    for coords, labels in [(coords_r, labels_r), (coords_g, labels_g)]:
        # Bin each cell to a patch
        ijk = np.floor((coords - origin) / patch_size).astype(int)
        for cell_ijk, cell_label in zip(ijk, labels):
            key = tuple(cell_ijk)
            # Will fill later
        break  # just getting unique keys...

    # Actually let me use a dict-based approach
    patches_r = {}
    patches_g = {}
    for coords, labels, storage in [(coords_r, labels_r, patches_r),
                                     (coords_g, labels_g, patches_g)]:
        ijk = np.floor((coords - origin) / patch_size).astype(int)
        for cell_ijk, cell_label in zip(ijk, labels):
            key = tuple(cell_ijk)
            storage.setdefault(key, np.zeros(n_types))[cell_label] += 1

    common_keys = set(patches_r) & set(patches_g)
    if not common_keys:
        return float("nan"), float("nan")

    cos_sims = []
    for key in common_keys:
        p = patches_r[key] + 1e-9
        q = patches_g[key] + 1e-9
        p, q = p / p.sum(), q / q.sum()
        cos = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
        cos_sims.append(cos)

    return float(np.mean(cos_sims)), float(np.std(cos_sims))


def morans_i_correlation(adata_recon, adata_gt, n_top_genes=100, n_neighbors=8):
    """
    Moran's I correlation (paper Fig 2c-e).

    1. Pick top `n_top_genes` highly variable genes from GT.
    2. For each gene, compute Moran's I in GT and in recon (k-NN spatial weights).
    3. Report Pearson R between the two Moran's I vectors.
    """
    # Find common genes
    common_genes = list(set(adata_recon.var_names) & set(adata_gt.var_names))
    if len(common_genes) < n_top_genes:
        n_top_genes = len(common_genes)

    # Select top HVGs from GT
    gt_sub = adata_gt[:, common_genes].copy()
    if issparse(gt_sub.X):
        var = np.array(gt_sub.X.power(2).mean(0) - np.square(gt_sub.X.mean(0))).flatten()
    else:
        var = gt_sub.X.var(axis=0)
    top_idx = np.argsort(var)[-n_top_genes:]
    top_genes = [common_genes[i] for i in top_idx]

    # Spatial weights: k-NN for each dataset
    coords_r = _get_3d_coords(adata_recon)
    coords_g = _get_3d_coords(adata_gt)

    def _morans_i(values, coords, k=n_neighbors):
        """Univariate Moran's I with k-NN binary weights. O(N·k)."""
        n = len(values)
        if n < k + 1:
            return float("nan")
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        _, indices = nn.kneighbors(coords)
        # Only use neighbors (skip self at column 0)
        nbrs = indices[:, 1:]  # (n, k)

        z = values - values.mean()
        denom = (z * z).sum()
        if denom == 0:
            return 0.0

        # O(N·k): each cell has exactly k neighbors
        num = (z[:, None] * z[nbrs]).sum()  # sum over all neighbor pairs
        W = n * k
        return (n / W) * (num / denom) if W > 0 else 0.0

    moran_gt, moran_recon = [], []
    for gene in top_genes:
        expr_g = _to_dense(adata_gt, gene)
        expr_r = _to_dense(adata_recon, gene)
        moran_gt.append(_morans_i(expr_g, coords_g))
        moran_recon.append(_morans_i(expr_r, coords_r))

    valid = ~(np.isnan(moran_gt) | np.isnan(moran_recon))
    if valid.sum() < 3:
        return float("nan"), top_genes[:5]

    r = float(np.corrcoef(np.array(moran_gt)[valid], np.array(moran_recon)[valid])[0, 1])
    return r, top_genes


# ---------------------------------------------------------------------------
# supplementary metrics
# ---------------------------------------------------------------------------

def cell_type_consistency(adata_recon, adata_gt, label_key, z_bins=20):
    z_min = max(adata_recon.obs["z_coord"].min(), adata_gt.obs["z_coord"].min())
    z_max = min(adata_recon.obs["z_coord"].max(), adata_gt.obs["z_coord"].max())
    bins = np.linspace(z_min, z_max, z_bins + 1)

    all_types = sorted(set(adata_recon.obs[label_key].astype(str).unique())
                       | set(adata_gt.obs[label_key].astype(str).unique()))
    n_types = len(all_types)

    js_divs = []
    for i in range(z_bins):
        mask_r = (adata_recon.obs["z_coord"] >= bins[i]) & (adata_recon.obs["z_coord"] < bins[i + 1])
        mask_g = (adata_gt.obs["z_coord"] >= bins[i]) & (adata_gt.obs["z_coord"] < bins[i + 1])
        if mask_r.sum() == 0 or mask_g.sum() == 0:
            js_divs.append(np.nan)
            continue
        p = np.array([(adata_recon.obs[label_key][mask_r].astype(str) == ct).mean()
                       for ct in all_types])
        q = np.array([(adata_gt.obs[label_key][mask_g].astype(str) == ct).mean()
                       for ct in all_types])
        p, q = p / (p.sum() + 1e-9), q / (q.sum() + 1e-9)
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log((p + 1e-9) / (m + 1e-9))) + \
             0.5 * np.sum(q * np.log((q + 1e-9) / (m + 1e-9)))
        js_divs.append(js)
    per_bin = pd.DataFrame({"z_mid": 0.5 * (bins[:-1] + bins[1:]), "js_div": js_divs})
    return per_bin, float(np.nanmean(js_divs))


def spatial_continuity(adata, label_key, n_neighbors=5):
    coords = _get_3d_coords(adata)
    labels = adata.obs[label_key].astype(str).values
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nn.kneighbors(coords)
    neighbor_labels = labels[indices[:, 1:]]
    return float((neighbor_labels == labels[:, None]).mean(axis=1).mean())


def spatial_label_transfer(adata_recon, adata_gt, label_key):
    coords_r = _get_3d_coords(adata_recon)
    coords_g = _get_3d_coords(adata_gt)
    tree = cKDTree(coords_g)
    _, nn_idx = tree.query(coords_r, k=1)
    gt_labels = adata_gt.obs[label_key].astype(str).values
    recon_labels = adata_recon.obs[label_key].astype(str).values
    return float((recon_labels == gt_labels[nn_idx]).mean())


def spatial_expression_correlation(adata_recon, adata_gt, gene_name=None, n_samples=5000):
    if gene_name is None:
        common = list(set(adata_recon.var_names) & set(adata_gt.var_names))
        gene_name = common[0] if common else adata_recon.var_names[0]
    coords_r = _get_3d_coords(adata_recon)
    coords_g = _get_3d_coords(adata_gt)
    idx_r = np.random.choice(len(adata_recon), min(n_samples, len(adata_recon)), replace=False)
    tree = cKDTree(coords_g)
    _, nn_idx = tree.query(coords_r[idx_r], k=1)
    expr_r, expr_g = _to_dense(adata_recon, gene_name), _to_dense(adata_gt, gene_name)
    expr_r, expr_g = expr_r[idx_r], expr_g[nn_idx]
    corr = float(np.corrcoef(expr_r, expr_g)[0, 1]) if expr_r.std() > 0 and expr_g.std() > 0 else 0.0
    return gene_name, corr


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _plot_js_div(per_bin, output_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(per_bin["z_mid"], per_bin["js_div"],
           width=per_bin["z_mid"][1] - per_bin["z_mid"][0], edgecolor="white")
    ax.set_xlabel("Z (μm)"); ax.set_ylabel("JS divergence")
    ax.set_title("Per-Z-slice cell type distribution divergence (lower = better)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "js_divergence.pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_continuity(cont_recon, cont_gt, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Reconstruction", "Ground Truth"], [cont_recon, cont_gt],
           color=["#d62728", "#1f77b4"])
    ax.set_ylabel("Spatial Continuity"); ax.set_title("Neighbor label agreement")
    for i, v in enumerate([cont_recon, cont_gt]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spatial_continuity.pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DeepSpatialniche Evaluation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--recon", type=str, default=None)
    p.add_argument("--ground_truth", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--label_key", type=str, default=None)
    p.add_argument("--gene", type=str, default=None)
    p.add_argument("--z_range", type=float, nargs=2, default=None)
    p.add_argument("--no_plots", action="store_true", default=None)
    p.add_argument("--no_metrics", action="store_true", default=None,
                   help="Skip quantitative metrics (visualization only)")
    args = p.parse_args()

    # --- load config ---
    yaml_path = args.config or (Path(__file__).parent / "configs" / "default.yaml")
    cfg = yaml.safe_load(open(yaml_path)) if os.path.exists(yaml_path) else {}
    ec = cfg.get("eval", {})

    # --- load data ---
    recon_path = _v(args.recon, ec.get("recon"), None)
    gt_path = _v(args.ground_truth, ec.get("ground_truth"), None)
    if not recon_path or not gt_path:
        print("Error: --recon and --ground_truth are required"); sys.exit(1)

    output_dir = _v(args.output_dir, ec.get("output_dir"), "./eval_results")
    label_key = _v(args.label_key, ec.get("label_key"), "cell_class")
    gene_name = _v(args.gene, ec.get("gene"), None)
    no_plots = _v(args.no_plots, ec.get("no_plots"), False)
    no_metrics = _v(args.no_metrics, ec.get("no_metrics"), False)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading reconstruction: {recon_path}")
    adata_recon = sc.read_h5ad(recon_path)
    print(f"  cells={adata_recon.n_obs}  genes={adata_recon.n_vars}")

    print(f"Loading ground truth: {gt_path}")
    adata_gt_all = sc.read_h5ad(gt_path)

    # --- subset GT to reconstructed Z range (matches tutorial) ---
    z_gt = adata_gt_all.obsm["spatial"][:, 2].astype(float)
    if args.z_range is not None:
        z_min, z_max = args.z_range
    else:
        z_min = adata_recon.obs["z_coord"].min() - 5
        z_max = adata_recon.obs["z_coord"].max() + 5
    mask = (z_gt >= z_min) & (z_gt <= z_max)
    adata_gt = adata_gt_all[mask].copy()

    # --- build comparable views (matches tutorial) ---
    adata_gt.obs["z_coord"] = adata_gt.obsm["spatial"][:, 2].astype(float)
    adata_gt.obsm["spatial"] = adata_gt.obsm["spatial"][:, :2].copy()
    adata_recon_vis = adata_recon.copy()
    print(f"  GT subset: {adata_gt.n_obs} cells (Z=[{z_min:.0f}, {z_max:.0f}])")

    # --- validate label key ---
    for name, adata in [("reconstruction", adata_recon_vis), ("ground truth", adata_gt)]:
        if label_key not in adata.obs:
            print(f"Error: '{label_key}' not in {name}.obs. Available: {list(adata.obs.columns)}")
            sys.exit(1)

    # --- shared palette ---
    palette = _make_palette(adata_recon_vis, adata_gt, label_key)

    # --- metrics ---
    if not no_metrics:
        print("\nComputing metrics...")

        # Paper metrics
        print("  Patch-level cell-type cosine similarity ...")
        patch_cos_mean, patch_cos_std = patch_cell_type_cosine(
            adata_recon_vis, adata_gt, label_key, patch_size=(50, 50, 30))

        print("  Moran's I correlation (top 100 HVGs) ...")
        moran_r, _ = morans_i_correlation(adata_recon_vis, adata_gt, n_top_genes=100)

        # Supplementary metrics
        per_bin, mean_js = cell_type_consistency(adata_recon_vis, adata_gt, label_key)
        cont_recon = spatial_continuity(adata_recon_vis, label_key)
        cont_gt = spatial_continuity(adata_gt, label_key)
        label_match = spatial_label_transfer(adata_recon_vis, adata_gt, label_key)
        gene_used, expr_corr = spatial_expression_correlation(adata_recon_vis, adata_gt, gene_name)

        print(f"  Patch cosine similarity:  {patch_cos_mean:.4f} ± {patch_cos_std:.4f}  (1 = perfect)")
        print(f"  Moran's I correlation:    {moran_r:.4f}  (1 = perfect)")
        print(f"  Mean JS divergence:       {mean_js:.4f}  (0 = identical)")
        print(f"  Spatial continuity:        recon={cont_recon:.3f}  gt={cont_gt:.3f}")
        print(f"  Spatial label transfer:   {label_match:.4f}  (1 = perfect)")
        print(f"  Spatial expr corr ({gene_used}): {expr_corr:.4f}")

        metrics = pd.DataFrame({
            "metric": [
                "patch_cosine_similarity_mean", "patch_cosine_similarity_std",
                "morans_i_correlation",
                "mean_js_divergence",
                "spatial_continuity_recon", "spatial_continuity_gt",
                "spatial_label_transfer",
                f"spatial_expr_corr_{gene_used}",
            ],
            "value": [
                patch_cos_mean, patch_cos_std, moran_r, mean_js,
                cont_recon, cont_gt, label_match, expr_corr,
            ],
        })
        metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # --- plots (with normalized coordinates for fair visualization) ---
    if not no_plots:
        print("\nGenerating plots...")

        # Normalize coords so XY (thousands) and Z (hundreds) are all [0,1]
        gt_norm = _normalize_coords(adata_gt)
        recon_norm = _normalize_coords(adata_recon_vis)

        if not no_metrics:
            _plot_js_div(per_bin, output_dir)
            _plot_continuity(cont_recon, cont_gt, output_dir)

        plot_orthogonal_projections(gt_norm, color_col=label_key, palette=palette,
                                    show=False, save_png=os.path.join(output_dir, "ortho_gt.png"))
        plot_orthogonal_projections(recon_norm, color_col=label_key, palette=palette,
                                    show=False, save_png=os.path.join(output_dir, "ortho_recon.png"))
        plt.close("all")

        try:
            plot_z_distribution(gt_norm, color_col=label_key, palette=palette,
                                show=False, save_pdf=os.path.join(output_dir, "zdist_gt.pdf"))
            plot_z_distribution(recon_norm, color_col=label_key, palette=palette,
                                show=False, save_pdf=os.path.join(output_dir, "zdist_recon.pdf"))
        except ValueError:
            print("  (skipping Z distribution plots — data shape mismatch)")
        plt.close("all")

        interactive_3d_labels(gt_norm, color_col=label_key, palette=palette,
                              title="Ground Truth: 3D Label Distribution",
                              width=1000, height=900,
                              save_html=os.path.join(output_dir, "3d_gt.html"))
        interactive_3d_labels(recon_norm, color_col=label_key, palette=palette,
                              title="Reconstruction: 3D Label Distribution",
                              width=1000, height=900,
                              save_html=os.path.join(output_dir, "3d_recon.html"))

        gene_vis = 'Reln' if 'Reln' in adata_recon_vis.var_names and 'Reln' in adata_gt.var_names \
                   else adata_recon_vis.var_names[0]
        interactive_3d_expression(gt_norm, gene_name=gene_vis,
                                  title=f"Ground Truth: Expression {gene_vis}",
                                  width=1000, height=900,
                                  save_html=os.path.join(output_dir, "3d_expr_gt.html"))
        interactive_3d_expression(recon_norm, gene_name=gene_vis,
                                  title=f"Reconstruction: Expression {gene_vis}",
                                  width=1000, height=900,
                                  save_html=os.path.join(output_dir, "3d_expr_recon.html"))

        print(f"Plots saved to {output_dir}/")

    print(f"\nDone. Metrics → {output_dir}/metrics.csv")


if __name__ == "__main__":
    main()
