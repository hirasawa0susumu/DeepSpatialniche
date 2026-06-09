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
# 2. coordinate normalization (CRITICAL FIX)
# =========================
def normalize_coords(coords, min_, max_):
    return (coords - min_) / (max_ - min_ + 1e-8)


# =========================
# 3. label encoding
# =========================
def encode_labels(labels_r, labels_g):
    all_types = sorted(set(labels_r) | set(labels_g))
    label_to_idx = {t: i for i, t in enumerate(all_types)}
    return label_to_idx, len(all_types)


# =========================
# 4. cosine
# =========================
def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


# =========================
# 5. patch assignment
# =========================
def assign_patch(coords, patch_size):
    return np.floor(coords / patch_size).astype(int)


# =========================
# 6. patch composition
# =========================
def build_patch_composition(coords, labels, patch_idx, grid_shape, n_types):
    P = np.zeros((*grid_shape, n_types), dtype=np.float32)

    for (i, j, k), l in zip(patch_idx, labels):
        P[i, j, k, l] += 1

    P = P.reshape(-1, n_types)
    s = P.sum(axis=1, keepdims=True)

    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)

    return P


# =========================
# 7. patch cosine metric
# =========================
def patch_cosine_score(P_gt, P_pred):
    scores = []

    for a, b in zip(P_gt, P_pred):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        # =========================
        # CASE 1: both empty → ignore
        # =========================
        if na == 0 and nb == 0:
            continue

        # =========================
        # CASE 2: one empty → score = 0
        # =========================
        if na != 0 and nb == 0:
            scores.append(0.0)
            continue

        if na == 0 and nb != 0:
            continue

        # =========================
        # CASE 3: both non-empty → cosine
        # =========================
        scores.append(float(np.dot(a, b) / (na * nb)))

    return float(np.mean(scores)) if len(scores) > 0 else 0.0


# =========================
# 8. MAIN PIPELINE (PAPER FINAL)
# =========================
def patch_level_evaluation(
    adata_recon,
    adata_gt,
    label_key,
    patch_size=(50, 50, 50)
):
    # -------- coords --------
    coords_r = get_3d_coords(adata_recon, is_gt=False)
    coords_g = get_3d_coords(adata_gt, is_gt=True)

    # =========================
    # ✔ GLOBAL MIN-MAX NORMALIZATION (KEY FIX)
    # =========================
    all_coords = np.vstack([coords_r, coords_g])

    c_min = all_coords.min(axis=0)
    c_max = all_coords.max(axis=0)

    coords_r = normalize_coords(coords_r, c_min, c_max)
    coords_g = normalize_coords(coords_g, c_min, c_max)

    # -------- labels --------
    mask_r = np.ones(len(coords_r), dtype=bool)
    mask_g = np.ones(len(coords_g), dtype=bool)

    labels_r_raw = adata_recon.obs[label_key].astype(str).values
    labels_g_raw = adata_gt.obs[label_key].astype(str).values

    label_to_idx, n_types = encode_labels(labels_r_raw, labels_g_raw)

    labels_r = np.array([label_to_idx[t] for t in labels_r_raw])
    labels_g = np.array([label_to_idx[t] for t in labels_g_raw])

    # =========================
    # patch size in normalized space
    # =========================
    patch_size = np.array(patch_size, dtype=np.float32) / 50.0  # scale-aware

    # -------- patch indexing --------
    patch_r = assign_patch(coords_r, patch_size)
    patch_g = assign_patch(coords_g, patch_size)

    # -------- grid shape --------
    all_patch = np.vstack([patch_r, patch_g])
    grid_shape = all_patch.max(axis=0) + 1

    # -------- build composition --------
    P_r = build_patch_composition(coords_r, labels_r, patch_r, grid_shape, n_types)
    P_g = build_patch_composition(coords_g, labels_g, patch_g, grid_shape, n_types)

    # -------- final metric --------
    return patch_cosine_score(P_g, P_r)


# =========================
# 9. RUN
# =========================
adata_recon = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/DeepSpatialniche/logs/mouse_starmap_01drop/mouse_starmap_reconstructed_3d.h5ad")
adata_gt = ad.read_h5ad("/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad")

score = patch_level_evaluation(
    adata_recon,
    adata_gt,
    label_key="Harmony_labels",
    patch_size=(50, 50, 50)
)

print("Patch-level cosine similarity (normalized):", score)