import os
import random

import anndata as ad
import numpy as np
import torch

from deepspatial import DeepSpatial

# --- seed ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
from deepspatial.vis_utils import (
    interactive_3d_labels,
    interactive_3d_expression,
    plot_z_distribution,
    plot_orthogonal_projections
)

# Set this to your local true-3D reference file
gt_path = '/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad'
adata_gt = ad.read_h5ad(gt_path)
dropout_rate = 0.3
print("dropout_rate:", dropout_rate)
# Extract 2D slice inputs from true 3D data using segmented Z ranges
z_coords = adata_gt.obsm['spatial'][:, 2]

ranges = [
    (5, 15),
    (35, 45),
    (65, 75),
    (95, 105),
    (125, 135),
    (155, 165),
    (185, 195)
]

adata_list = []

for i, (z_start, z_end) in enumerate(ranges, start=1):
    mask = (z_coords >= z_start) & (z_coords <= z_end)
    sub_adata = adata_gt[mask].copy()

    # Use XY as 2D coordinates for model input
    sub_adata.obsm['spatial'] = sub_adata.obsm['spatial'][:, :2]

    midpoint = float((z_start + z_end) / 2)
    sub_adata.obs['z_coord'] = midpoint

    adata_list.append(sub_adata)

model = DeepSpatial()

model.setup_data(
    adata_list=adata_list,
    spatial_key='spatial',
    z_key='z_coord',
    label_key='Harmony_labels',
    batch_size=512
)
model.build_model(
    patch_size=8,
    hidden_size=256,
    depth=6,
    num_heads=8,
    lr=2e-4,
    use_niche_encoder=True,
    niche_hidden_dim=256,
    niche_num_heads=4,
    niche_num_tokens=4,
    niche_dropout=dropout_rate
)
model.fit(
    max_epochs=10,
    accelerator='gpu',
    devices=[0],
    save_ckpt=True
)

adata_3d = model.reconstruct_full_volume(
    adata_list,
    thickness=10,
    use_niche=True
)
os.makedirs('output', exist_ok=True)
adata_3d.write_h5ad('output/deepspatial_3d_starmap_brain_crossattn_03drop_new.h5ad')
