import gc
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from .uot_solver import compute_uot_coupling
from ..models.niche_encoder import precompute_multiscale_neighbors

class DeepSpatialDataset(Dataset):
    """
    DeepSpatialniche Global Trajectory Dataset.
    Constructs cross-slice cell pairs using Unbalanced Optimal Transport (UOT) 
    to provide continuous training samples for Flow Matching.
    """
    def __init__(self, 
                 adata_list: list, 
                 spatial_key: str = 'spatial_norm', 
                 z_key: str = 'z_norm',
                 label_key: str = 'cell_class',
                 n_samples_base: int = 50000, 
                 alpha_spatial: float = 0.5,
                 uot_reg: float = 0.8, 
                 uot_tau: float = 0.05,
                 mode: str = 'fit'):
        """
        Args:
            adata_list: List of AnnData objects.
            spatial_key: Key in .obsm containing normalized XY coordinates.
            z_key: Key in .obs containing normalized Z coordinates.
            label_key: Key in .obs for cell labels/types.
            n_samples_base: Target total number of trajectory pairs.
            alpha_spatial: Balance between spatial and gene distances for UOT.
            uot_reg: Entropy regularization for UOT solver.
            uot_tau: Marginal relaxation for UOT solver.
            mode: 'fit' for full multi-slice training, 'predict' for limited slice pairs.
        """
        self.adata_list = adata_list
        self.spatial_key = spatial_key
        self.z_key = z_key
        self.label_key = label_key
        self.n_samples_base = n_samples_base
        self.alpha_spatial = alpha_spatial
        self.uot_reg = uot_reg
        self.uot_tau = uot_tau

        # --- 1. Global Label Encoding ---
        self.label_encoder = LabelEncoder()
        all_labels = []
        for adata in adata_list:
            if label_key in adata.obs:
                all_labels.extend(adata.obs[label_key].astype(str).tolist())
        
        if all_labels:
            self.label_encoder.fit(all_labels)
            self.num_classes = len(self.label_encoder.classes_)
            self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        else:
            self.num_classes = 1
            self.id2label = {0: "unknown"}

        # Precompute multi-scale spatial KNN neighbors for niche encoding
        precompute_multiscale_neighbors(adata_list, spatial_key=spatial_key)

        if mode != 'fit':
            self.adata_list = self.adata_list[:2]

        # Containers for trajectory pairs
        self.trajectory_pairs = {
            'x0': [], 'g0': [], 'c0': [], 'z0': [],
            'x1': [], 'g1': [], 'c1': [], 'z1': [],
            'delta_z': [],
            # Multi-scale source neighbor data
            'g_nbr_local': [], 'delta_nbr_local': [], 'dist_nbr_local': [], 'mask_nbr_local': [],
            'g_nbr_mid':   [], 'delta_nbr_mid':   [], 'dist_nbr_mid':   [], 'mask_nbr_mid': [],
            'g_nbr_global':[], 'delta_nbr_global':[], 'dist_nbr_global':[], 'mask_nbr_global':[],
            # Dynamic niche: UOT-mapped target data per scale
            'g_nbr_target_local': [],  'pos_nbr_target_local': [],
            'g_nbr_target_mid':   [],  'pos_nbr_target_mid': [],
            'g_nbr_target_global':[],  'pos_nbr_target_global': [],
        }
        
        self._build_trajectory_dataset()
        self._convert_to_tensors()

    def _get_data_arrays(self, adata):
        """Extracts spatial, gene expression, and one-hot labels from AnnData."""
        x = adata.obsm[self.spatial_key].astype(np.float32)
        g = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else adata.X.astype(np.float32)
            
        if self.label_key in adata.obs:
            raw_labels = adata.obs[self.label_key].astype(str).values
            indices = self.label_encoder.transform(raw_labels)
            c_onehot = np.eye(self.num_classes)[indices].astype(np.float32)
        else:
            c_onehot = np.zeros((adata.n_obs, self.num_classes), dtype=np.float32)
            
        # Z coordinate is assumed uniform across a single slice
        z = float(adata.obs[self.z_key].iloc[0])
            
        return x, g, c_onehot, z

    def _build_trajectory_dataset(self):
        """Pairs cells across adjacent slices using UOT coupling."""
        num_slices = len(self.adata_list)
        
        # Allocate samples based on slice product weights
        pair_sizes = [self.adata_list[k].n_obs * self.adata_list[k+1].n_obs for k in range(num_slices - 1)]
        total_weight = sum(pair_sizes)
        sampling_counts = [int(self.n_samples_base * (w / total_weight)) for w in pair_sizes]

        for k in tqdm(range(num_slices - 1), desc="DeepSpatialniche: Building Trajectories"):
            n_to_sample = sampling_counts[k]
            if n_to_sample <= 0:
                continue

            x0, g0, c0, z0 = self._get_data_arrays(self.adata_list[k])
            x1, g1, c1, z1 = self._get_data_arrays(self.adata_list[k+1])
            delta_z = z1 - z0

            # Compute Unbalanced Optimal Transport (UOT)
            pi = compute_uot_coupling(
                x0, g0, c0, x1, g1, c1, 
                alpha_spatial=self.alpha_spatial,
                uot_reg=self.uot_reg,
                uot_tau=self.uot_tau
            )
            
            # Sampling logic
            pi_flat = pi.ravel()
            pi_sum = pi_flat.sum()
            
            if pi_sum > 0:
                pi_prob = pi_flat / pi_sum
                idx_flat = np.random.choice(len(pi_flat), size=n_to_sample, p=pi_prob, replace=True)
                idx0, idx1 = np.unravel_index(idx_flat, pi.shape)

                # --- UOT target mapping (cell-level, shared across scales) ---
                pi_norm = pi / (pi.sum(axis=1, keepdims=True) + 1e-16)
                g_mapped = pi_norm @ g1   # (N0, G)
                pos_mapped = pi_norm @ x1  # (N0, 2)

                def _extract_scale(scale_key, K_default):
                    nk = f'niche_{scale_key}_neighbors'
                    dk = f'niche_{scale_key}_deltas'
                    dsk = f'niche_{scale_key}_dists'
                    mk = f'niche_{scale_key}_mask'
                    nbr_idx = self.adata_list[k].uns[nk][idx0]
                    g_nbr = g0[nbr_idx]
                    delta = self.adata_list[k].uns[dk][idx0]
                    dist = self.adata_list[k].uns[dsk][idx0]
                    msk = self.adata_list[k].uns.get(
                        mk, np.ones((g0.shape[0], K_default), dtype=bool))[idx0]
                    g_tgt = g_mapped[nbr_idx]
                    pos_tgt = pos_mapped[nbr_idx]
                    return g_nbr, delta, dist, msk, g_tgt, pos_tgt

                (gl, dl, dstl, ml, gtl, ptl) = _extract_scale('local', 8)
                (gm_, dm_, dstm, mm, gtm, ptm) = _extract_scale('mid', 32)
                (gg, dg, dstg, mg, gtg, ptg) = _extract_scale('global', 128)

                # Store endpoints
                self.trajectory_pairs['x0'].append(x0[idx0])
                self.trajectory_pairs['g0'].append(g0[idx0])
                self.trajectory_pairs['c0'].append(c0[idx0])
                self.trajectory_pairs['z0'].append(np.full((n_to_sample, 1), z0, dtype=np.float32))

                # Store multi-scale source niche
                self.trajectory_pairs['g_nbr_local'].append(gl)
                self.trajectory_pairs['delta_nbr_local'].append(dl)
                self.trajectory_pairs['dist_nbr_local'].append(dstl)
                self.trajectory_pairs['mask_nbr_local'].append(ml)
                self.trajectory_pairs['g_nbr_mid'].append(gm_)
                self.trajectory_pairs['delta_nbr_mid'].append(dm_)
                self.trajectory_pairs['dist_nbr_mid'].append(dstm)
                self.trajectory_pairs['mask_nbr_mid'].append(mm)
                self.trajectory_pairs['g_nbr_global'].append(gg)
                self.trajectory_pairs['delta_nbr_global'].append(dg)
                self.trajectory_pairs['dist_nbr_global'].append(dstg)
                self.trajectory_pairs['mask_nbr_global'].append(mg)

                # Store dynamic target-mapped data
                self.trajectory_pairs['g_nbr_target_local'].append(gtl)
                self.trajectory_pairs['pos_nbr_target_local'].append(ptl)
                self.trajectory_pairs['g_nbr_target_mid'].append(gtm)
                self.trajectory_pairs['pos_nbr_target_mid'].append(ptm)
                self.trajectory_pairs['g_nbr_target_global'].append(gtg)
                self.trajectory_pairs['pos_nbr_target_global'].append(ptg)

                self.trajectory_pairs['x1'].append(x1[idx1])
                self.trajectory_pairs['g1'].append(g1[idx1])
                self.trajectory_pairs['c1'].append(c1[idx1])
                self.trajectory_pairs['z1'].append(np.full((n_to_sample, 1), z1, dtype=np.float32))
                self.trajectory_pairs['delta_z'].append(np.full((n_to_sample, 1), delta_z, dtype=np.float32))

            del pi, pi_flat, x0, g0, c0, x1, g1, c1
            gc.collect()

    def _convert_to_tensors(self):
        """Aggregates list of arrays into final PyTorch tensors."""
        self.tensors = {}
        for key in self.trajectory_pairs:
            if self.trajectory_pairs[key]:
                concatenated = np.concatenate(self.trajectory_pairs[key], axis=0)
                if key.startswith('mask_nbr'):
                    self.tensors[key] = torch.from_numpy(concatenated).bool()
                else:
                    self.tensors[key] = torch.from_numpy(concatenated).float()

        self.trajectory_pairs.clear()
        gc.collect()

        if 'x0' in self.tensors:
            self.num_samples = self.tensors['x0'].shape[0]
        else:
            self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}

    def decode_label(self, one_hot_vec):
        """Converts model one-hot output back to original string label."""
        idx = torch.argmax(one_hot_vec, dim=-1).item()
        return self.id2label[idx]