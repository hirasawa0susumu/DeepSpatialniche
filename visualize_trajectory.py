"""Visualize ODE trajectory animation between two slices.

Usage:
    python visualize_trajectory.py
"""
import torch
import numpy as np
import scanpy as sc
from deepspatial.vis_utils import animate_trajectory

# ==== CONFIG — edit these ====
DATA_DIR = "/root/autodl-tmp/wangjiaxiang/Datas/imc_human_breastcancer/held_out/"
SLICE_PAIR = (6, 7)           # which two slices to use
N_CELLS = 3000                 # how many cells to animate
STEPS = 100                   # ODE integration steps
CKPT_PATH = "/root/autodl-tmp/wangjiaxiang/DeepSpatialniche/logs/deepspatial_run_imc1/deepspatial-epoch=34-loss=0.1014.ckpt"
SAVE_GIF = "trajectory.gif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== LOAD ====
import glob, re, os
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "imc_*.h5ad")),
    key=lambda x: int(re.search(r'imc_(\d+)', os.path.basename(x)).group(1)),
)
print(f"Found {len(files)} slices")

# Load the two slices
i0, i1 = SLICE_PAIR
ad0 = sc.read_h5ad(files[i0])
ad1 = sc.read_h5ad(files[i1])

# Prepare spatial
for adata in (ad0, ad1):
    if 'spatial_3d' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial_3d'][:, :2].copy()
        adata.obs['z_coord'] = adata.obsm['spatial_3d'][:, 2].astype(float)
print(f"Slice {i0}: {ad0.n_obs} cells, z={ad0.obs['z_coord'].iloc[0]}")
print(f"Slice {i1}: {ad1.n_obs} cells, z={ad1.obs['z_coord'].iloc[0]}")

# ==== LOAD MODEL ====
from deepspatial import DeepSpatial
label_key = 'cell_type' if 'cell_type' in ad0.obs else 'cell_class'

model = DeepSpatial()
model.setup_data(
    adata_list=[ad0, ad1],
    spatial_key='spatial', z_key='z_coord', label_key=label_key,
    batch_size=128, mode='predict',
    use_niche=False
)
model.load_checkpoint(CKPT_PATH)
model.module.to(DEVICE)
model.module.eval()

# ==== BUILD BATCH ====
dev = torch.device(DEVICE)
(x0, z0, g0, c0, x1, z1, g1, c1,
 _, _, niche_ref_0, niche_ref_1) = model._setup_and_extract(ad0, ad1, thickness=1.0, dev=dev)

# Take a small subset
n = min(N_CELLS, x0.shape[0])
idx = torch.randperm(x0.shape[0])[:n]
x0_s, g0_s, c0_s = x0[idx], g0[idx], c0[idx]

c_onehot = torch.nn.functional.one_hot(c0_s, num_classes=model.num_classes).float()

batch = {
    'x0': x0_s, 'g0': g0_s, 'c0': c_onehot,
    'z0': torch.full((n, 1), z0, device=dev),
    'z1': torch.full((n, 1), z1, device=dev),
    'delta_z': torch.full((n, 1), z1 - z0, device=dev),
}

# Compute niche tokens (single-scale)
has_niche = (niche_ref_0 is not None and model.niche_encoder is not None)
if has_niche:
    _ne = model.niche_encoder
    nbr_idx = niche_ref_0['neighbors'][idx]
    g_nbr = g0[nbr_idx]
    batch['niche_tokens'] = _ne(
        g_center=g0_s, pos_center=x0_s,
        g_nbrs=g_nbr,
        delta_nbrs=niche_ref_0['deltas'][idx],
        dist_nbrs=niche_ref_0['dists'][idx],
        mask_nbr=niche_ref_0['mask'][idx],
    )
    batch['niche_nbr_data'] = {
        'g_nbr': g_nbr,
        'delta_nbr': niche_ref_0['deltas'][idx],
        'dist_nbr': niche_ref_0['dists'][idx],
        'mask_nbr': niche_ref_0['mask'][idx],
    }

ct_labels = c0_s.cpu().numpy()

# ==== VELOCITY CHECK ====
print("=== Velocity check ===")
_z_mid = (z0 + z1) / 2
_z_mid_t = torch.full((n, 1), _z_mid, device=dev)
_t_half = torch.full((n,), 0.5, device=dev)
with torch.no_grad():
    vx_train, vg_train, vc_train = model.module.model(
        xt=x0_s, gt=g0_s, t=_t_half,
        zt=_z_mid_t, delta_z=batch['delta_z'], ct=c_onehot,
        niche_tokens=batch.get('niche_tokens'),
    )
    vx_ema, vg_ema, vc_ema = model.module.ema_model(
        xt=x0_s, gt=g0_s, t=_t_half,
        zt=_z_mid_t, delta_z=batch['delta_z'], ct=c_onehot,
        niche_tokens=batch.get('niche_tokens'),
    )

idx1 = torch.randperm(x1.shape[0])[:n]
print(f"true Δx max={(x1[idx1] - x0_s).abs().max():.6f}")
print(f"train model: |vx| max={vx_train.abs().max():.6f}  mean={vx_train.abs().mean():.6f}")
print(f"EMA model:   |vx| max={vx_ema.abs().max():.6f}  mean={vx_ema.abs().mean():.6f}")
print(f"EMA/train ratio: {vx_ema.abs().max() / max(vx_train.abs().max(), 1e-8):.4f}")

# ==== RUN ODE ====
print(f"Running ODE integration ({STEPS} steps, {n} cells)...")
res = model.module.sample(batch, mode="ODE", steps=STEPS)
x_traj = res['x_traj'].cpu().numpy()

delta = np.abs(x_traj[-1] - x_traj[0]).max()
print(f"Max displacement over trajectory: {delta:.4f} (norm), "
      f"{delta * model.spatial_stats['x_range']:.2f}um (physical)")

# ==== ANIMATE ====
animate_trajectory(
    x_traj,
    cell_types=ct_labels,
    adata0=ad0, adata1=ad1,
    spatial_stats=model.spatial_stats,
    title=f"Trajectory: slice {i0} -> {i1} (Δx_max={delta:.3f})",
    save_path=SAVE_GIF,
    frame_interval=100,
    point_size=2.0,
)
