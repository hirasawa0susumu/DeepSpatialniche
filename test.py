import anndata as ad
import matplotlib.pyplot as plt

_PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
            '#393b79','#637939','#8c6d31','#843c39','#7b4173',
            '#3182bd','#31a354','#756bb1','#636363','#e6550d']
ad1 = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_imc_breastcancer_heldout.h5ad")
ad2 = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/Datas/imc_human_breastcancer/imc_human_breastcancer/imc_10.h5ad")
ad3 = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/deepspatial_gt]/output/deepspatial_3d_imc_breastcancer_1.h5ad")
# print(ad2)
# ad3 = ad1[ad1.obs["z_coord"]==]
# ad_1 = ad1[ad1.obs["z_coord"]==20]
# print(ad_1)
# print(ad_1.shape)
print(ad1.obs["spatial_z"].unique())
mid = 200
thick = 1
mask1 = (ad1.obs["z_coord"] >= mid - thick) & (ad1.obs["z_coord"] <= mid +
                                           thick)
mask2 = (ad3.obs["z_coord"] >= mid - thick) & (ad3.obs["z_coord"] <= mid +
                                           thick)
cells1 = ad1[mask1]
cells1.obs["z_coord"][:] = mid
print(f"slice : Z=20μm ±{thick} → {cells1.n_obs} cells")
cells2 = ad3[mask2]
cells2.obs["z_coord"][:] = mid
print(f"slice : Z=20μm ±{thick} → {cells2.n_obs} cells")


def plot_slices(adata_a, adata_b, label_key="cell_type",
                title_a="A", title_b="B", point_size=3):
    """Plot two AnnData slices side by side, colored by cell type.

    Both must have XY coords in obsm['spatial'] and label_key in obs.
    """
    all_labs = sorted(set(adata_a.obs[label_key].astype(str))
                      | set(adata_b.obs[label_key].astype(str)))
    palette = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(all_labs)}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, adata, title in [(ax0, adata_a, title_a), (ax1, adata_b, title_b)]:
        coords = adata.obsm["spatial"]
        labels = adata.obs[label_key].astype(str).values
        for lab in all_labs:
            m = labels == lab
            if m.any():
                ax.scatter(coords[m, 0], coords[m, 1], s=point_size,
                           c=palette[lab], alpha=0.6, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Line2D([0],[0], marker='o', color=palette[lab], linestyle='', markersize=5)
               for lab in all_labs]
    fig.legend(handles, all_labs, loc='center right', bbox_to_anchor=(1.14, 0.5), fontsize=7)
    fig.tight_layout()
    plt.show()




# ---------------------------------------------------------------------------
# Wasserstein-2 evaluation — three-level design
#   L1  global spatial W2  (MinMax‑normalised XY)
#   L2  per‑cell‑type spatial W2  (unweighted mean — rare types count equally)
#   L3  gene‑expression W2  (joint PCA + joint z‑score, cross‑slice mean kept)
# ---------------------------------------------------------------------------

def _minmax_coords(adata):
    """MinMax‑normalise XY to [0, 1] — preserves shape, no variance rescaling."""
    import numpy as np
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xs = (xy[:, 0] - xmin) / (xmax - xmin + 1e-8)
    ys = (xy[:, 1] - ymin) / (ymax - ymin + 1e-8)
    return np.column_stack([xs, ys])


def _sinkhorn_w2(wa, wb, M, reg_frac=0.05, maxiter=200, stopThr=1e-6):
    """Sinkhorn 𝒲₂ with reg adapted to cost-matrix scale.

    Standard Sinkhorn — XY is MinMax-normalised so costs are well-behaved
    and log-domain overhead is unnecessary.
    """
    import ot
    import numpy as np

    reg_abs = max(float(reg_frac) * float(np.median(M)), 1e-12)
    w2_sq = ot.sinkhorn2(wa, wb, M, reg=reg_abs,
                         numItermax=maxiter, stopThr=stopThr, warn=False)
    return float(np.sqrt(max(w2_sq, 0)))


def _w2_spatial(xy_a, xy_b, reg_frac):
    """L1 — global spatial 𝒲₂ between two 2D point sets."""
    import ot
    import numpy as np

    wa = np.ones(len(xy_a)) / len(xy_a)
    wb = np.ones(len(xy_b)) / len(xy_b)
    M = ot.dist(xy_a, xy_b, metric="sqeuclidean")
    return _sinkhorn_w2(wa, wb, M, reg_frac=reg_frac)


def _w2_per_type(adata_a, adata_b, label_key, reg_frac):
    """L2 — unweighted mean per‑cell‑type spatial 𝒲₂.

    Each cell type contributes equally regardless of abundance, so rare
    but biologically important populations (stem cells, specific immune
    subsets) are not drowned out by dominant types.
    """
    import numpy as np

    xy_a_all = _minmax_coords(adata_a)
    xy_b_all = _minmax_coords(adata_b)
    labs_a = adata_a.obs[label_key].astype(str).values
    labs_b = adata_b.obs[label_key].astype(str).values

    per_type = {}
    for lab in sorted(set(labs_a) | set(labs_b)):
        mask_a = labs_a == lab
        mask_b = labs_b == lab
        # skip types with < 5 cells in either slice — W2 is unreliable
        if mask_a.sum() < 5 or mask_b.sum() < 5:
            continue
        per_type[lab] = _w2_spatial(xy_a_all[mask_a], xy_b_all[mask_b], reg_frac)

    if not per_type:
        return None, None

    mean_w2 = float(np.mean(list(per_type.values())))
    return mean_w2, per_type


def _w2_gene(adata_a, adata_b, pca_dims=50, reg_frac=0.05, seed=42):
    """L3 — gene‑expression 𝒲₂ (joint PCA + joint z‑score)."""
    import ot
    import numpy as np
    from scipy.sparse import issparse
    from sklearn.decomposition import PCA

    common = sorted(set(adata_a.var_names) & set(adata_b.var_names))
    if len(common) == 0:
        return None

    def _get_g(adata, genes):
        g = (adata[:, genes].X.toarray() if issparse(adata[:, genes].X)
             else np.asarray(adata[:, genes].X, dtype=np.float64))
        return g.astype(np.float64)

    g_a = _get_g(adata_a, common)
    g_b = _get_g(adata_b, common)
    pca = PCA(n_components=min(pca_dims, len(common)), random_state=seed)
    g_all = pca.fit_transform(np.vstack([g_a, g_b]))
    g_a, g_b = g_all[:len(g_a)], g_all[len(g_a):]

    # joint z‑score — preserves between‑slice mean differences
    g_mean = g_all.mean(0, keepdims=True)
    g_std  = g_all.std(0, keepdims=True) + 1e-8
    g_a = (g_a - g_mean) / g_std
    g_b = (g_b - g_mean) / g_std

    wa = np.ones(len(g_a)) / len(g_a)
    wb = np.ones(len(g_b)) / len(g_b)
    M = ot.dist(g_a, g_b, metric="sqeuclidean")
    return _sinkhorn_w2(wa, wb, M, reg_frac=reg_frac)


def slice_w2(adata_a, adata_b, label_key="cell_type",
             use_gene=True, pca_dims=50,
             reg_frac=0.2, seed=42):
    """Three‑level Wasserstein‑2 between two 2D slices.

    Uses all cells — no subsampling.
    """
    import numpy as np, time

    # ---- L1 — global spatial -----------------------------------------------

    xy_a = _minmax_coords(adata_a)
    xy_b = _minmax_coords(adata_b)

    w2_spatial = _w2_spatial(xy_a, xy_b, reg_frac=reg_frac)


    # ---- L2 — per‑cell‑type spatial ----------------------------------------

    w2_per_type_mean, w2_per_type = _w2_per_type(
        adata_a, adata_b, label_key, reg_frac=reg_frac,
    )


    # ---- L3 — gene expression ----------------------------------------------
    w2_gene = None
    if use_gene:

        w2_gene = _w2_gene(adata_a, adata_b, pca_dims=pca_dims,
                           reg_frac=reg_frac, seed=seed)


    return {
        "w2_spatial":       w2_spatial,
        "w2_per_type_mean": w2_per_type_mean,
        "w2_per_type":      w2_per_type,
        "w2_gene":          w2_gene,
    }


plot_slices(cells1, ad2)
plot_slices(cells2, ad2)
import time

def _print_w2(label, r, elapsed):
    pt = r.get("w2_per_type") or {}
    n_types = len(pt)
    top = sorted(pt.items(), key=lambda x: x[1], reverse=True)[:5]
    t = r.get("timing", {})
    print(f"{label}  ({elapsed:.1f}s total):")
    for k in ["L1_spatial", "L2_per_type", "L3_gene"]:
        if k in t:
            print(f"  [{k}]        {t[k]:.2f}s")
    print(f"  L1 spatial           = {r['w2_spatial']:.4f}")
    print(f"  L2 per-type mean     = {r['w2_per_type_mean']:.4f}  ({n_types} types)")
    if r["w2_gene"] is not None:
        print(f"  L3 gene              = {r['w2_gene']:.4f}")
    if top:
        print(f"  top-5 worst types:   {', '.join(f'{l}={v:.3f}' for l,v in top)}")
    print()

t0 = time.time()
r1 = slice_w2(cells1, ad2, label_key="cell_type")
t1 = time.time()
_print_w2("cells1 vs ad2", r1, t1 - t0)

t0 = time.time()
r2 = slice_w2(cells2, ad2, label_key="cell_type")
t2 = time.time()
_print_w2("cells2 vs ad2", r2, t2 - t0)