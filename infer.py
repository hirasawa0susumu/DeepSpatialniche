#!/usr/bin/env python3
"""DeepSpatialniche inference entry point.

All parameters have defaults in configs/default.yaml.
CLI args override config values.

Usage:
    python infer.py
    python infer.py --config configs/custom.yaml
    python infer.py --checkpoint ./ckpt/last.ckpt --true_3d --data_path ./brain.h5ad
    python infer.py --checkpoint ./ckpt/last.ckpt --data_dir ./slices/ --thickness 5.0
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import yaml

import deepspatial as ds
from deepspatial.models.niche_encoder import precompute_neighbors


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
    p = argparse.ArgumentParser(description="DeepSpatialniche 3D Reconstruction")
    p.add_argument("--config", type=str, default=None)

    # data
    p.add_argument("--true_3d", action="store_true", default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--slice_thickness", type=float, default=None)
    p.add_argument("--slice_gap", type=float, default=None)

    # infer
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--sampling_method", type=str, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--thickness", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_niche", type=lambda x: x.lower() == "true", default=None,
                   help="Whether to use niche encoder (default True). Set false for ablation.")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()

    # --- load config ---
    yaml_path = args.config or (Path(__file__).parent / "configs" / "default.yaml")
    cfg = yaml.safe_load(open(yaml_path)) if os.path.exists(yaml_path) else {}
    dc = cfg.get("data", {})
    ic = cfg.get("infer", {})

    # --- seed ---
    seed = _v(args.seed, ic.get("seed"), 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        adatas = [sc.read_h5ad(f) for f in files]

    # --- checkpoint ---
    ckpt = _v(args.checkpoint, ic.get("checkpoint"), None)
    if not ckpt:
        print("Error: checkpoint is required (via --checkpoint or config infer.checkpoint)")
        sys.exit(1)
    if not os.path.exists(ckpt):
        print(f"Error: checkpoint not found: {ckpt}")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt}")

    sampling_method = _v(args.sampling_method, ic.get("sampling_method"), "dopri5")
    atol = _v(args.atol, ic.get("atol"), None)
    rtol = _v(args.rtol, ic.get("rtol"), None)
    model = ds.DeepSpatial()
    model.load_checkpoint(ckpt, sampling_method=sampling_method,
                          atol=atol, rtol=rtol)

    # --- normalize + precompute niche ---
    model.spatial_key = model.spatial_key or "spatial"
    model.z_key = model.z_key or "z_coord"
    model.label_key = model.label_key or "cell_class"
    model._normalize_spatial(adatas)

    ckpt_cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(ckpt_cfg_path):
        with open(ckpt_cfg_path) as f:
            ckpt_cfg = json.load(f)
        nec = ckpt_cfg.get("niche_encoder_config", {})
        if nec.get("use_niche_encoder", False):
            print("Precomputing spatial neighbors for niche encoder...")
            precompute_neighbors(adatas, spatial_key="spatial_norm", K=32)

    # --- reconstruct ---
    print(f"Reconstructing 3D volume ({len(adatas)} slices)...")
    adata_3d = model.reconstruct_full_volume(
        adatas,
        thickness=_v(args.thickness, ic.get("thickness"), 10.0),
        steps=_v(args.steps, ic.get("steps"), 100),
        chunk_size=_v(args.chunk_size, ic.get("chunk_size"), 2048),
        use_niche=_v(args.use_niche, ic.get("use_niche"), True),
        seed=seed,
        device=_v(args.device, ic.get("device"), "auto"),
    )

    # --- save ---
    output = _v(args.output, ic.get("output"), "reconstructed_3d.h5ad")
    adata_3d.write(output)
    print(f"Saved: {output}")
    print(f"  Cells: {adata_3d.n_obs}  "
          f"Genes: {adata_3d.n_vars}  "
          f"Z: {adata_3d.obs[model.z_key].min():.1f} - "
          f"{adata_3d.obs[model.z_key].max():.1f} um")


if __name__ == "__main__":
    main()
