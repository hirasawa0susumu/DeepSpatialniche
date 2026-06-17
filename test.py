import anndata as ad
import matplotlib.pyplot as plt

_PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
            '#393b79','#637939','#8c6d31','#843c39','#7b4173',
            '#3182bd','#31a354','#756bb1','#636363','#e6550d']
ad1 = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_imc_breastcancer_heldout_smb.h5ad")
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
print(f"slice : Z=20μm ±5 → {cells1.n_obs} cells")
cells2 = ad3[mask2]
cells2.obs["z_coord"][:] = mid
print(f"slice : Z=20μm ±5 → {cells2.n_obs} cells")


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




def slice_w2(adata_a, adata_b, use_gene=True, pca_dims=50, n_subsample=5000, reg=0.05):
    """Wasserstein-2 distance between two 2D slices in joint (xy + gene) space."""
    import ot
    import numpy as np
    from scipy.sparse import issparse
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(42)

    def _get_xy(adata):
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        return (xy - xy.mean(0)) / (xy.std(0) + 1e-8)

    def _get_g(adata, genes):
        g = adata[:, genes].X.toarray() if issparse(adata[:, genes].X) else np.asarray(adata[:, genes].X)
        return g.astype(np.float64)

    if use_gene:
        common = sorted(set(adata_a.var_names) & set(adata_b.var_names))
        g_a = _get_g(adata_a, common)
        g_b = _get_g(adata_b, common)
        pca = PCA(n_components=min(pca_dims, len(common)), random_state=42)
        g_all = pca.fit_transform(np.vstack([g_a, g_b]))
        g_a, g_b = g_all[:len(g_a)], g_all[len(g_a):]
        g_a = (g_a - g_a.mean(0)) / (g_a.std(0) + 1e-8)
        g_b = (g_b - g_b.mean(0)) / (g_b.std(0) + 1e-8)
        A = np.hstack([_get_xy(adata_a), g_a])
        B = np.hstack([_get_xy(adata_b), g_b])
    else:
        A = _get_xy(adata_a)
        B = _get_xy(adata_b)

    if n_subsample and n_subsample < len(A):
        A = A[rng.choice(len(A), n_subsample, replace=False)]
    if n_subsample and n_subsample < len(B):
        B = B[rng.choice(len(B), n_subsample, replace=False)]

    wa = np.ones(len(A)) / len(A)
    wb = np.ones(len(B)) / len(B)
    M = ot.dist(A, B, metric="sqeuclidean")
    M = M / (M.max() + 1e-16)
    w2 = ot.sinkhorn2(wa, wb, M, reg=reg, numItermax=1000, stopThr=1e-6)
    return float(np.sqrt(max(w2, 0)))


plot_slices(cells1, ad2)
plot_slices(cells2, ad2)
print(f"W2 distance: {slice_w2(cells1, ad2):.4f}")
print(f"W2 distance: {slice_w2(cells2, ad2):.4f}")