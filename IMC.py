import os
import re
import glob
import torch
import scanpy as sc

from deepspatial import DeepSpatial
from deepspatial.vis_utils import interactive_3d_labels, interactive_3d_expression, plot_z_distribution, plot_orthogonal_projections

# 2. Data preparation and path parsing
# Set this to your local dataset directory before running
data_dir = "/root/autodl-tmp/wangjiaxiang/Datas/imc_human_breastcancer/imc_human_breastcancer/"
file_paths = sorted(
    glob.glob(os.path.join(data_dir, "imc_*.h5ad")),
    key=lambda x: int(re.search(r'imc_(\d+)', os.path.basename(x)).group(1)),
)

if len(file_paths) == 0:
    raise FileNotFoundError(f"No files found under: {data_dir}")

# Load slices and split spatial_3d into spatial(x, y) + z_coord
adata_list = []
for p in file_paths:
    adata = sc.read_h5ad(p)

    if 'spatial_3d' not in adata.obsm or adata.obsm['spatial_3d'].shape[1] < 3:
        raise ValueError(
            f"{os.path.basename(p)} is missing valid obsm['spatial_3d'] with at least 3 columns"
        )

    coords_3d = adata.obsm['spatial_3d']
    adata.obsm['spatial'] = coords_3d[:, :2].copy()
    adata.obs['z_coord'] = coords_3d[:, 2].astype(float)

    adata_list.append(adata)

if len(adata_list) == 0:
    raise ValueError("No valid IMC slices loaded.")

print(f"Loaded {len(adata_list)} slices with z_coord derived from spatial_3d")

model = DeepSpatial()

# 3. Data setup: normalization is handled internally without overwriting raw coordinates
candidate_label_keys = ['cell_type', 'cell_class', 'annotation', 'Harmony_labels']
label_key = next((k for k in candidate_label_keys if k in adata_list[0].obs.columns), None)
if label_key is None:
    raise ValueError(f'No valid label key found. Tried: {candidate_label_keys}')
print('Using label key:', label_key)

model.setup_data(
    adata_list=adata_list,
    spatial_key='spatial',
    z_key='z_coord',
    label_key=label_key,
    batch_size=2048
)

# Build model: configure architecture hyperparameters
model.build_model(
    patch_size=8,
    hidden_size=256,
    depth=6,
    lr=2e-4,
    use_niche_encoder=True,
    niche_hidden_dim=256,
    niche_num_heads=4,
    niche_num_tokens=4,
    niche_dropout=0.3
)

# Train: set checkpoint directory and device options
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
devices = [0] if accelerator == 'gpu' else 1
print('Training accelerator:', accelerator)

model.fit(
    max_epochs=10,
    save_dir="/root/autodl-tmp/wangjiaxiang/DeepSpatialniche/logs/deepspatial_run_imc1",
    accelerator=accelerator,
    devices=devices,
    save_ckpt=True
)

adata_3d = model.reconstruct_full_volume(
    adata_list,
    thickness=2
)

os.makedirs("output", exist_ok=True)
adata_3d.write_h5ad("output/deepspatial_3d_imc_breastcancer_1.h5ad")
print("Saved reconstruction to output/deepspatial_3d_imc_breastcancer.h5ad")