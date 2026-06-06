#!/usr/bin/env python3
"""DeepSpatialniche training entry point.

All parameters have defaults in configs/default.yaml.
CLI args override config values.

Usage:
    python train.py
    python train.py --config configs/custom.yaml
    python train.py --true_3d --data_path ./brain.h5ad --max_epochs 50
    python train.py --data_dir ./slices/ --hidden_size 128 --depth 3
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml

import deepspatial as ds


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _slice_3d(data_path, slice_thickness, slice_gap, ranges=None,
               z_column=2, spatial_key="spatial"):
    """Slice a single 3D h5ad into 2D slices by Z ranges.

    Args:
        ranges: optional list of (start, end) tuples. If given, ignores
                slice_thickness/slice_gap and uses exact ranges (matches tutorial).
    """
    print(f"Loading 3D data: {data_path}")
    adata_3d = sc.read_h5ad(data_path)
    z_all = adata_3d.obsm[spatial_key][:, z_column].astype(float)

    if ranges is None:
        z_min = np.floor(z_all.min())
        z_max = np.ceil(z_all.max())
        ranges = []
        z_start = z_min
        while z_start < z_max:
            ranges.append((z_start, z_start + slice_thickness))
            z_start += slice_gap

    adata_list = []
    for z_start, z_end in ranges:
        mask = (z_all >= z_start) & (z_all <= z_end)
        sub = adata_3d[mask].copy()
        if sub.n_obs == 0:
            continue

        sub.obsm["spatial"] = adata_3d[mask].obsm[spatial_key][:, :2].copy()
        sub.obs["z_coord"] = float((z_start + z_end) / 2)

        print(f"  z=[{z_start:.0f}, {z_end:.0f}]  mid={sub.obs['z_coord'].iloc[0]:.0f}  "
              f"cells={sub.n_obs}")
        adata_list.append(sub)

    if len(adata_list) < 2:
        print(f"Error: need >=2 slices, got {len(adata_list)}.")
        sys.exit(1)
    return adata_list


def _v(args_val, cfg_val, default):
    """Resolve value: CLI > config > hardcoded default."""
    if args_val is not None:
        return args_val
    if cfg_val is not None:
        return cfg_val
    return default


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DeepSpatialniche Training")
    p.add_argument("--config", type=str, default=None)

    # data
    p.add_argument("--true_3d", action="store_true", default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--slice_thickness", type=float, default=None)
    p.add_argument("--slice_gap", type=float, default=None)
    p.add_argument("--spatial_key", type=str, default=None)
    p.add_argument("--z_key", type=str, default=None)
    p.add_argument("--label_key", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_samples_base", type=int, default=None)
    p.add_argument("--alpha_spatial", type=float, default=None)
    p.add_argument("--uot_reg", type=float, default=None)
    p.add_argument("--uot_tau", type=float, default=None)

    # model
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--mlp_ratio", type=float, default=None)
    p.add_argument("--path_type", type=str, default=None)
    p.add_argument("--use_niche_encoder", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--niche_hidden_dim", type=int, default=None)
    p.add_argument("--niche_num_heads", type=int, default=None)

    # train
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lambda_g", type=float, default=None)
    p.add_argument("--lambda_c", type=float, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--accelerator", type=str, default=None)
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--save_ckpt", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--resume_ckpt_path", type=str, default=None)

    args = p.parse_args()

    # --- load config ---
    yaml_path = args.config or (Path(__file__).parent / "configs" / "default.yaml")
    cfg = yaml.safe_load(open(yaml_path)) if os.path.exists(yaml_path) else {}
    dc = cfg.get("data", {})
    mc = cfg.get("model", {})
    tc = cfg.get("train", {})

    # --- load data ---
    true_3d = _v(args.true_3d, dc.get("true_3d"), False)

    if true_3d:
        data_path = _v(args.data_path, dc.get("data_path"), None)
        if not data_path:
            print("Error: true_3d=true requires data_path")
            sys.exit(1)
        st = _v(args.slice_thickness, dc.get("slice_thickness"), 10.0)
        sg = _v(args.slice_gap, dc.get("slice_gap"), 20.0)
        sr = dc.get("slice_ranges")
        adatas = _slice_3d(data_path, st, sg, ranges=sr)
    else:
        data_dir = _v(args.data_dir, dc.get("data_dir"), None)
        if not data_dir:
            print("Error: true_3d=false requires data_dir")
            sys.exit(1)
        files = sorted(glob.glob(os.path.join(data_dir, "*.h5ad")))
        if not files:
            print(f"Error: no .h5ad files in {data_dir}")
            sys.exit(1)
        print(f"Found {len(files)} slice files")
        for f in files:
            print(f"  {f}")
        adatas = [sc.read_h5ad(f) for f in files]

    for i, a in enumerate(adatas):
        print(f"  slice {i}: {a.n_obs} cells, {a.n_vars} genes")

    # --- setup ---
    model = ds.DeepSpatial()
    model.setup_data(
        adatas,
        spatial_key=_v(args.spatial_key, dc.get("spatial_key"), "spatial"),
        z_key=_v(args.z_key, dc.get("z_key"), "z_coord"),
        label_key=_v(args.label_key, dc.get("label_key"), "cell_class"),
        batch_size=_v(args.batch_size, dc.get("batch_size"), 128),
        num_workers=_v(args.num_workers, dc.get("num_workers"), 4),
        n_samples_base=_v(args.n_samples_base, dc.get("n_samples_base"), 50000),
        alpha_spatial=_v(args.alpha_spatial, dc.get("alpha_spatial"), 0.5),
        uot_reg=_v(args.uot_reg, dc.get("uot_reg"), 0.8),
        uot_tau=_v(args.uot_tau, dc.get("uot_tau"), 0.05),
    )
    print(f"Gene dim: {model.gene_dim}  Classes: {model.num_classes}")

    # --- build ---
    model.build_model(
        patch_size=_v(args.patch_size, mc.get("patch_size"), 8),
        hidden_size=_v(args.hidden_size, mc.get("hidden_size"), 256),
        depth=_v(args.depth, mc.get("depth"), 6),
        num_heads=_v(args.num_heads, mc.get("num_heads"), 8),
        mlp_ratio=_v(args.mlp_ratio, mc.get("mlp_ratio"), 4.0),
        path_type=_v(args.path_type, mc.get("path_type"), "Linear"),
        lr=_v(args.lr, tc.get("lr"), 2e-4),
        weight_decay=_v(args.weight_decay, tc.get("weight_decay"), 1e-5),
        lambda_g=_v(args.lambda_g, tc.get("lambda_g"), 0.1),
        lambda_c=_v(args.lambda_c, tc.get("lambda_c"), 10.0),
        use_niche_encoder=_v(args.use_niche_encoder, mc.get("use_niche_encoder"), True),
        niche_hidden_dim=_v(args.niche_hidden_dim, mc.get("niche_hidden_dim"), 128),
        niche_num_heads=_v(args.niche_num_heads, mc.get("niche_num_heads"), 4),
        niche_dropout=_v(None, mc.get("niche_dropout"), 0.3),
    )
    print(f"Parameters: {sum(p.numel() for p in model.module.parameters()):,}")

    # --- train ---
    model.fit(
        max_epochs=_v(args.max_epochs, tc.get("max_epochs"), 100),
        save_dir=_v(args.save_dir, tc.get("save_dir"), "./checkpoints"),
        accelerator=_v(args.accelerator, tc.get("accelerator"), "auto"),
        devices=_v(args.devices, tc.get("devices"), "auto"),
        save_ckpt=_v(args.save_ckpt, tc.get("save_ckpt"), True),
        resume_ckpt_path=_v(args.resume_ckpt_path, tc.get("resume_ckpt_path"), None),
    )


if __name__ == "__main__":
    main()
