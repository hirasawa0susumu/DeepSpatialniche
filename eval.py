#!/usr/bin/env python3
"""DeepSpatialniche evaluation — tutorial visualizations + paper quantitative metrics.

Metrics from the paper:
  1. Global cell-type proportion cosine similarity (GT vs recon)
  2. Patch-level cell-type cosine similarity (3D patches, GT vs recon)
  3. Moran's I correlation (top HVGs, GT vs recon)
  4. JS divergence, spatial continuity, label transfer, spatial correlation (supplementary)

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
from scipy.stats import spearmanr
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


def _cell_type_vector(adata, label_key, all_types):
    labels = adata.obs[label_key].astype(str)
    return np.array([(labels == ct).mean() for ct in all_types], dtype=float)


def _safe_cosine(p, q):
    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    if p_norm == 0 and q_norm == 0:
        return np.nan
    if p_norm == 0 or q_norm == 0:
        return 0.0
    return float(np.dot(p, q) / (p_norm * q_norm))


def _sanitize_expression_matrix(adata):
    """Replace NaN/Inf values before HVG/variance calculations."""
    if issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)


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

def cell_type_proportion_cosine(adata_recon, adata_gt, label_key):
    """
    Global cell-type proportion agreement.

    The paper first checks that cell-type proportions are consistent between
    ground-truth and reconstructed volumes before local patch evaluation.
    """
    all_types = sorted(set(adata_recon.obs[label_key].astype(str).unique())
                       | set(adata_gt.obs[label_key].astype(str).unique()))
    p = _cell_type_vector(adata_recon, label_key, all_types)
    q = _cell_type_vector(adata_gt, label_key, all_types)
    return _safe_cosine(p, q), pd.DataFrame({
        "cell_type": all_types,
        "reconstruction": p,
        "ground_truth": q,
    })


def patch_cell_type_cosine(adata_recon, adata_gt, label_key, patch_size=(50, 50, 50), patch_scope="gt_occupied"):
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

    # Determine fixed 50um voxels and cell membership in original physical space.
    origin = coords_g.min(axis=0)
    gt_max = coords_g.max(axis=0)
    patch_size = np.asarray(patch_size, dtype=float)
    extent = gt_max - origin
    grid_shape = np.maximum(np.ceil((extent + 1e-8) / patch_size).astype(int), 1)

    patches_r = {}
    patches_g = {}
    for coords, labels, storage in [(coords_r, labels_r, patches_r),
                                     (coords_g, labels_g, patches_g)]:
        ijk = np.floor((coords - origin) / patch_size).astype(int)
        in_grid = ((ijk >= 0) & (ijk < grid_shape)).all(axis=1)
        ijk = ijk[in_grid]
        labels = labels[in_grid]
        for cell_ijk, cell_label in zip(ijk, labels):
            key = tuple(cell_ijk)
            storage.setdefault(key, np.zeros(n_types))[cell_label] += 1

    if patch_scope == "gt_occupied":
        occupied_keys = set(patches_g)
    elif patch_scope == "common":
        occupied_keys = set(patches_r) & set(patches_g)
    elif patch_scope == "union":
        occupied_keys = set(patches_r) | set(patches_g)
    else:
        raise ValueError("patch_scope must be one of: gt_occupied, common, union")
    if not occupied_keys:
        return float("nan"), float("nan")

    cos_sims = []
    empty = np.zeros(n_types)
    for key in occupied_keys:
        p = patches_r.get(key, empty).astype(float)
        q = patches_g.get(key, empty).astype(float)
        if p.sum() > 0:
            p = p / p.sum()
        if q.sum() > 0:
            q = q / q.sum()
        cos = _safe_cosine(p, q)
        if np.isnan(cos):
            continue
        cos_sims.append(cos)

    if not cos_sims:
        return float("nan"), float("nan")
    return float(np.mean(cos_sims)), float(np.std(cos_sims))


def _select_top_hvgs(adata_gt, common_genes, n_top_genes):
    gt_sub = adata_gt[:, common_genes].copy()
    _sanitize_expression_matrix(gt_sub)
    try:
        sc.pp.highly_variable_genes(
            gt_sub,
            n_top_genes=n_top_genes,
            flavor="seurat",
            inplace=True,
        )
        hvgs = gt_sub.var_names[gt_sub.var["highly_variable"]].tolist()
        if len(hvgs) >= min(3, n_top_genes):
            return hvgs[:n_top_genes]
    except Exception as exc:
        print(f"  (Scanpy HVG selection failed; falling back to variance: {exc})")

    if issparse(gt_sub.X):
        mean = np.asarray(gt_sub.X.mean(0)).ravel()
        mean_sq = np.asarray(gt_sub.X.power(2).mean(0)).ravel()
        var = mean_sq - np.square(mean)
    else:
        var = np.asarray(gt_sub.X).var(axis=0)
    var = np.nan_to_num(var, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    top_idx = np.argsort(var)[-n_top_genes:]
    return [common_genes[i] for i in top_idx]


def morans_i_correlation(adata_recon, adata_gt, n_top_genes=100, n_neighbors=8):
    """
    Moran's I correlation (paper Fig 2c-e).

    1. Pick top `n_top_genes` highly variable genes from GT.
    2. For each gene, compute Moran's I in GT and in recon (k-NN spatial weights).
    3. Report Pearson R between the two Moran's I vectors.
    """
    # Find common genes
    common_genes = sorted(set(adata_recon.var_names) & set(adata_gt.var_names))
    if not common_genes:
        return float("nan"), [], None, None
    if len(common_genes) < n_top_genes:
        n_top_genes = len(common_genes)

    # Select top HVGs from GT
    top_genes = _select_top_hvgs(adata_gt, common_genes, n_top_genes)

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

    moran_gt = np.array(moran_gt)
    moran_recon = np.array(moran_recon)
    valid = ~(np.isnan(moran_gt) | np.isnan(moran_recon))
    if valid.sum() < 3:
        return float("nan"), top_genes[:5], None, None

    r = float(spearmanr(moran_gt[valid], moran_recon[valid]).correlation)
    return r, top_genes, moran_gt, moran_recon


def _plot_morans_i(moran_gt, moran_recon, r, top_genes, output_dir):
    """Scatter plot of GT vs Recon Moran's I for top HVGs (paper Figs. 2c–e)."""
    if moran_gt is None or moran_recon is None:
        return
    valid = ~(np.isnan(moran_gt) | np.isnan(moran_recon))
    mg, mr = moran_gt[valid], moran_recon[valid]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mg, mr, s=20, alpha=0.6, edgecolors='none', c='#1f77b4')
    ax.set_xlabel("Ground Truth Moran's I")
    ax.set_ylabel("Reconstruction Moran's I")
    ax.set_title(f"Moran's I Correlation (top {valid.sum()} HVGs)")

    lo = min(mg.min(), mr.min()) - 0.05
    hi = max(mg.max(), mr.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.text(0.05, 0.95, f'Pearson R = {r:.4f}', transform=ax.transAxes,
            fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "morans_i.pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def spatial_expression_correlation(adata_recon, adata_gt, gene_name=None, n_samples=5000, random_state=0):
    if gene_name is None:
        common = sorted(set(adata_recon.var_names) & set(adata_gt.var_names))
        gene_name = common[0] if common else adata_recon.var_names[0]
    coords_r = _get_3d_coords(adata_recon)
    coords_g = _get_3d_coords(adata_gt)
    rng = np.random.default_rng(random_state)
    idx_r = rng.choice(len(adata_recon), min(n_samples, len(adata_recon)), replace=False)
    tree = cKDTree(coords_g)
    _, nn_idx = tree.query(coords_r[idx_r], k=1)
    expr_r, expr_g = _to_dense(adata_recon, gene_name), _to_dense(adata_gt, gene_name)
    expr_r, expr_g = expr_r[idx_r], expr_g[nn_idx]
    corr = float(np.corrcoef(expr_r, expr_g)[0, 1]) if expr_r.std() > 0 and expr_g.std() > 0 else 0.0
    return gene_name, corr


def marker_gene_correlations(adata_recon, adata_gt, marker_genes, n_samples=5000):
    rows = []
    for gene in marker_genes:
        if gene not in adata_recon.var_names or gene not in adata_gt.var_names:
            continue
        _, corr = spatial_expression_correlation(
            adata_recon,
            adata_gt,
            gene_name=gene,
            n_samples=n_samples,
            random_state=0,
        )
        rows.append({"gene": gene, "spatial_expr_corr": corr})
    return pd.DataFrame(rows)


def _plot_patch_cosine(patch_cos_mean, patch_cos_std, output_dir):
    """Bar plot of patch-level cell-type cosine similarity (paper Figs. 2c–e)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["Mean", "Std"], [patch_cos_mean, patch_cos_std],
                  color=['#d62728', '#7f7f7f'], width=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel("Patch Cosine Similarity")
    ax.set_title("Patch-level Cell-type Cosine Similarity\n(1 = perfect)")
    for bar, val in zip(bars, [patch_cos_mean, patch_cos_std]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.4f}', ha='center', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "patch_cosine.pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    p.add_argument("--patch_size", type=float, nargs=3, default=None,
                   help="3D patch size. Mouse brain paper setting: 50 50 50")
    p.add_argument("--patch_scope", type=str, choices=["gt_occupied", "common", "union"], default=None,
                   help="Voxel averaging scope for patch cosine")
    p.add_argument("--moran_neighbors", type=int, default=None)
    p.add_argument("--marker_genes", type=str, nargs="*", default=None,
                   help="Marker genes to quantify. Mouse brain default: Gad1 Gad2 Slc17a7")
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
    z_range = _v(args.z_range, ec.get("z_range"), None)
    patch_size = tuple(_v(args.patch_size, ec.get("patch_size"), (50, 50, 50)))
    patch_scope = _v(args.patch_scope, ec.get("patch_scope"), "gt_occupied")
    moran_neighbors = int(_v(args.moran_neighbors, ec.get("moran_neighbors"), 8))
    marker_genes = _v(args.marker_genes, ec.get("marker_genes"), ["Gad1", "Gad2", "Slc17a7"])
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
    if z_range is not None:
        z_min, z_max = z_range
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
        print("  Global cell-type proportions ...")
        prop_cos, prop_table = cell_type_proportion_cosine(adata_recon_vis, adata_gt, label_key)
        prop_table.to_csv(os.path.join(output_dir, "cell_type_proportions.csv"), index=False)

        print("  Patch-level cell-type cosine similarity ...")
        patch_cos_mean, patch_cos_std = patch_cell_type_cosine(
            adata_recon_vis, adata_gt, label_key, patch_size=patch_size, patch_scope=patch_scope)

        print("  Moran's I correlation (top 100 HVGs) ...")
        moran_r, _, moran_gt, moran_recon = morans_i_correlation(
            adata_recon_vis, adata_gt, n_top_genes=100, n_neighbors=moran_neighbors)

        # Supplementary metrics
        per_bin, mean_js = cell_type_consistency(adata_recon_vis, adata_gt, label_key)
        cont_recon = spatial_continuity(adata_recon_vis, label_key)
        cont_gt = spatial_continuity(adata_gt, label_key)
        label_match = spatial_label_transfer(adata_recon_vis, adata_gt, label_key)
        gene_used, expr_corr = spatial_expression_correlation(adata_recon_vis, adata_gt, gene_name)
        marker_df = marker_gene_correlations(adata_recon_vis, adata_gt, marker_genes)
        if not marker_df.empty:
            marker_df.to_csv(os.path.join(output_dir, "marker_gene_correlations.csv"), index=False)

        print(f"  Global cell-type proportion cosine: {prop_cos:.4f}  (1 = perfect)")
        print(f"  Patch cosine similarity:  {patch_cos_mean:.4f} ± {patch_cos_std:.4f}  (1 = perfect)")
        print(f"  Moran's I correlation:    {moran_r:.4f}  (1 = perfect)")
        print(f"  Mean JS divergence:       {mean_js:.4f}  (0 = identical)")
        print(f"  Spatial continuity:        recon={cont_recon:.3f}  gt={cont_gt:.3f}")
        print(f"  Spatial label transfer:   {label_match:.4f}  (1 = perfect)")
        print(f"  Spatial expr corr ({gene_used}): {expr_corr:.4f}")

        metrics = pd.DataFrame({
            "metric": [
                "cell_type_proportion_cosine",
                "patch_cosine_similarity_mean", "patch_cosine_similarity_std",
                "morans_i_correlation",
                "mean_js_divergence",
                "spatial_continuity_recon", "spatial_continuity_gt",
                "spatial_label_transfer",
                f"spatial_expr_corr_{gene_used}",
            ],
            "value": [
                prop_cos,
                patch_cos_mean, patch_cos_std, moran_r, mean_js,
                cont_recon, cont_gt, label_match, expr_corr,
            ],
        })
        if not marker_df.empty:
            marker_metrics = pd.DataFrame({
                "metric": "marker_spatial_expr_corr_" + marker_df["gene"].astype(str),
                "value": marker_df["spatial_expr_corr"].astype(float),
            })
            metrics = pd.concat([metrics, marker_metrics], ignore_index=True)
        metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # --- plots (with normalized coordinates for fair visualization) ---
    if not no_plots:
        print("\nGenerating plots...")

        # Normalize coords so XY (thousands) and Z (hundreds) are all [0,1]
        gt_norm = _normalize_coords(adata_gt)
        recon_norm = _normalize_coords(adata_recon_vis)

        if not no_metrics:
            _plot_patch_cosine(patch_cos_mean, patch_cos_std, output_dir)
            _plot_morans_i(moran_gt, moran_recon, moran_r, None, output_dir)
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
