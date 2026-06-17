"""Wasserstein distance between ground truth and reconstruction.

Direct 2-Wasserstein distance on the joint (coords + PCA-reduced gene expression)
distribution.  PCA compresses ~1000 genes → 50 dims so Sinkhorn is numerically stable.

Usage: just edit the paths below and run `python wasserstein.py`
"""

import ot
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════════════════
# config — edit here
# ═══════════════════════════════════════════════════════════════════════════

RECON_PATH = "/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_starmap_brain_crossattn_noniche_new.h5ad"
GT_PATH = "/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad"

N_SUBSAMPLE = 10000
PCA_DIMS = 50               # compress gene expression to this many dims
SEED = 42
SINKHORN_REG = 0.1


# ═══════════════════════════════════════════════════════════════════════════

def _extract(adata, is_gt, pca=None, fit_pca=False):
    """Extract [x, y, z, pca_gene] for all cells.

    Coords are z-scored, gene expression is PCA-reduced.
    """
    if is_gt:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)  # (N,3)
    else:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        z = adata.obs["z_coord"].to_numpy(dtype=np.float32).reshape(-1, 1)
        coords = np.hstack([xy, z])

    g = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    g = g.astype(np.float64)

    # z-score normalisation
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)

    if fit_pca:
        pca = PCA(n_components=PCA_DIMS, random_state=SEED)
        g_pca = pca.fit_transform(g)
    else:
        g_pca = pca.transform(g)

    # scale PCA output to match coords variance
    g_pca = (g_pca - g_pca.mean(axis=0)) / (g_pca.std(axis=0) + 1e-8)

    return np.concatenate([coords, g_pca], axis=1).astype(np.float64), pca


def wasserstein_2(a, b):
    """Sinkhorn-approximated 2-Wasserstein distance."""
    rng = np.random.default_rng(SEED)

    if N_SUBSAMPLE and N_SUBSAMPLE < len(a):
        a = a[rng.choice(len(a), N_SUBSAMPLE, replace=False)]
    if N_SUBSAMPLE and N_SUBSAMPLE < len(b):
        b = b[rng.choice(len(b), N_SUBSAMPLE, replace=False)]

    wa = np.ones(len(a)) / len(a)
    wb = np.ones(len(b)) / len(b)

    M = ot.dist(a, b, metric="sqeuclidean")
    M = M / (M.max() + 1e-16)           # scale to [0, 1] so exp(-M/reg) doesn't underflow
    w2 = ot.sinkhorn2(wa, wb, M, reg=SINKHORN_REG, numItermax=2000, stopThr=1e-7)
    # Rescale back: W2 scales as sqrt(M_max) when M is normalised
    w2_raw = max(w2, 0.0)
    return float(np.sqrt(w2_raw))


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading reconstruction: {RECON_PATH}")
    adata_recon = sc.read_h5ad(RECON_PATH)
    print(f"  cells={adata_recon.n_obs}  genes={adata_recon.n_vars}")

    print(f"Loading GT: {GT_PATH}")
    adata_gt_all = sc.read_h5ad(GT_PATH)

    # subset GT to recon Z range
    z_gt = adata_gt_all.obsm["spatial"][:, 2].astype(float)
    z_min = adata_recon.obs["z_coord"].min() - 5
    z_max = adata_recon.obs["z_coord"].max() + 5
    adata_gt = adata_gt_all[(z_gt >= z_min) & (z_gt <= z_max)].copy()
    adata_gt.obs["z_coord"] = adata_gt.obsm["spatial"][:, 2].astype(float)
    adata_gt.obsm["spatial"] = adata_gt.obsm["spatial"][:, :2].copy()
    print(f"  GT subset: {adata_gt.n_obs} cells (Z=[{z_min:.0f}, {z_max:.0f}])")

    # common genes
    common_genes = sorted(set(adata_recon.var_names) & set(adata_gt.var_names))
    adata_recon = adata_recon[:, common_genes].copy()
    adata_gt = adata_gt[:, common_genes].copy()
    print(f"  Common genes: {len(common_genes)}")

    print(f"PCA: {len(common_genes)} genes → {PCA_DIMS} dims")
    print(f"Joint dim: 3 + {PCA_DIMS} = {3 + PCA_DIMS}")

    # Fit PCA on GT, apply to both
    print(f"Extracting features (subsample={N_SUBSAMPLE})...")
    X_gt, pca = _extract(adata_gt, is_gt=False, fit_pca=True)
    X_recon, _ = _extract(adata_recon, is_gt=False, pca=pca, fit_pca=False)

    print(f"Computing Wasserstein-2 distance (reg={SINKHORN_REG})...")
    w2 = wasserstein_2(X_gt, X_recon)

    print(f"\n  Wasserstein-2 distance: {w2:.4f}  (lower = better)")
