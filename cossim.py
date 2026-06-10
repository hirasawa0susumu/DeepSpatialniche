"""Patch-level cell-type cosine similarity (paper Fig 2c-e).

Patch extraction with raw physical coordinates (no normalization).
Matches the paper description:
  "divided each 3D volume into local patches (50 x 50 x 50 for mouse brain)
   and computed the cosine similarity of cell-type distributions within each patch."
"""

import numpy as np
import anndata as ad


# =========================
# 1. coordinates
# =========================
def get_3d_coords(adata, is_gt=False):
    if is_gt:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    else:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        z = adata.obs["z_coord"].to_numpy(dtype=np.float32).reshape(-1, 1)
        return np.hstack([xy, z])


# =========================
# 2. label encoding
# =========================
def encode_labels(labels_r, labels_g):
    all_types = sorted(set(labels_r) | set(labels_g))
    label_to_idx = {t: i for i, t in enumerate(all_types)}
    return label_to_idx, len(all_types)


# =========================
# 3. patch assignment (raw physical coords)
# =========================
def assign_patch(coords, origin, patch_size):
    return np.floor((coords - origin) / patch_size).astype(int)


# =========================
# 4. patch composition
# =========================
def build_patch_composition(patch_idx, labels, grid_shape, n_types):
    P = np.zeros((*grid_shape, n_types), dtype=np.float32)

    for (i, j, k), l in zip(patch_idx, labels):
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
            P[i, j, k, l] += 1

    P = P.reshape(-1, n_types)
    s = P.sum(axis=1, keepdims=True)
    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)

    return P


# =========================
# 5. patch cosine metric
# =========================
def patch_cosine_score(P_pred, P_gt):
    """Only evaluate patches where GT has cells (per paper).
    GT-empty patches are ignored; pred-empty patches get score=0."""
    scores = []
    for a, b in zip(P_gt, P_pred):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 and nb == 0:
            continue          # both empty → skip
        if na == 0 and nb != 0:
            scores.append(0.0)
            continue          #     GT empty, pred hallucinated → skip
        if na != 0 and nb == 0:
            # scores.append(0.0)  # GT occupied, pred empty → score 0
            continue

        scores.append(float(np.dot(a, b) / (na * nb)))

    return float(np.mean(scores)) if len(scores) > 0 else 0.0


# =========================
# 6. MAIN PIPELINE
# =========================
def patch_level_evaluation(
    adata_recon,
    adata_gt,
    label_key,
    patch_size=(50, 50, 50),
    isgt1 = False,
    isgt2 = False
):
    # -------- coords (raw physical, no normalization) --------
    coords_r = get_3d_coords(adata_recon, is_gt=isgt1)
    coords_g = get_3d_coords(adata_gt, is_gt=isgt2)

    # Shared origin = GT minimum
    origin = coords_g.min(axis=0).astype(np.float32)
    patch_size = np.array(patch_size, dtype=np.float32)

    # -------- labels --------
    labels_r_raw = adata_recon.obs[label_key].astype(str).values
    labels_g_raw = adata_gt.obs[label_key].astype(str).values

    label_to_idx, n_types = encode_labels(labels_r_raw, labels_g_raw)

    labels_r = np.array([label_to_idx[t] for t in labels_r_raw])
    labels_g = np.array([label_to_idx[t] for t in labels_g_raw])

    # -------- patch indexing --------
    patch_r = assign_patch(coords_r, origin, patch_size)
    patch_g = assign_patch(coords_g, origin, patch_size)

    # Grid shape covers both GT and recon
    all_patch = np.vstack([patch_r, patch_g])
    grid_shape = all_patch.max(axis=0) + 1

    # -------- build composition --------
    P_r = build_patch_composition(patch_r, labels_r, grid_shape, n_types)
    P_g = build_patch_composition(patch_g, labels_g, grid_shape, n_types)

    # -------- final metric --------
    return patch_cosine_score(P_g, P_r)

def out_fun(adatagt, adatarecon, isgt1, isgt2):
    score = patch_level_evaluation(
        adatagt,
        adatarecon,
        label_key="Harmony_labels",
        isgt1=isgt1,
        isgt2=isgt2
    )

    print(score)

# =========================
# 7. RUN
# =========================
if __name__ == "__main__":
    adata_recon_csa = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_starmap_brain_crossattn.h5ad")

    adata_recon_noni1 = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_starmap_brain_crossattn_noniche_new.h5ad"
    )
    adata_dsgt = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/deepspatial_gt]/output/deepspatial_3d_starmap_brain.h5ad")
    adata_dsgt_new = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/deepspatial_gt]/output/deepspatial_3d_starmap_brain_new.h5ad"
    )
    adata_recon_csa03drop = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/output/deepspatial_3d_starmap_brain_crossattn_03drop.h5ad"
    )
    adata_gt = ad.read_h5ad(
        "/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad"
    )

    print("Computing patch-level cosine similarity (patch_size=50x50x50 μm, raw coords)...")

    out_fun(adata_gt, adata_dsgt, isgt1=True, isgt2=False)
    out_fun(adata_gt, adata_dsgt_new, isgt1=True, isgt2=False)
    out_fun(adata_gt, adata_recon_csa03drop, isgt1=True, isgt2=False)