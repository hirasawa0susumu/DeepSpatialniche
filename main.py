import os
import numpy as np
import pandas as pd
import anndata as ad
from sympy import false

from deepspatial import DeepSpatial
from deepspatial.vis_utils import (
    interactive_3d_labels,
    interactive_3d_expression,
    plot_z_distribution,
    plot_orthogonal_projections
)

# Set this to your local true-3D reference file
gt_path = '/root/autodl-tmp/wangjiaxiang/Datas/deepstarmap_mouse_brain.h5ad'
adata_gt = ad.read_h5ad(gt_path)
adata_gt
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
    lr=2e-4,
    use_niche_encoder=false,
    niche_hidden_dim=256,
    niche_num_heads=4
)
model.fit(
    max_epochs=10,
    accelerator='gpu',
    devices=[0],
    save_ckpt=False
)

adata_3d = model.reconstruct_full_volume(
    adata_list,
    thickness=10
)
adata_3d
os.makedirs('output', exist_ok=True)
adata_3d.write_h5ad('output/deepspatial_3d_starmap_brain_noniche.h5ad')