"""Graph attention pooling niche encoder.

Spatial KNN → joint neighbor encoding → attention-weighted sum → niche token.
Can be called at any time t with any (g_center, pos_center).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# KNN precompute
# ---------------------------------------------------------------------------

def precompute_neighbors(adata_list, spatial_key='spatial_norm', K=32):
    """Precompute K nearest spatial neighbors for every cell in each 2D slice."""
    for adata in adata_list:
        coords = adata.obsm[spatial_key].astype(np.float64)
        n_cells = len(coords)
        k = min(K + 1, n_cells)
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=k)

        nbr_idx = indices[:, 1:].astype(np.int64)
        nbr_dists = distances[:, 1:].astype(np.float32)

        center_coords = coords[:, np.newaxis, :]
        nbr_coords = coords[nbr_idx]
        deltas = (nbr_coords - center_coords).astype(np.float32)

        valid_mask = np.ones((n_cells, k - 1), dtype=bool)
        pad = K - (k - 1)
        if pad > 0:
            nbr_idx = np.pad(nbr_idx, ((0, 0), (0, pad)), constant_values=0)
            nbr_dists = np.pad(nbr_dists, ((0, 0), (0, pad)), constant_values=1e9)
            deltas = np.pad(deltas, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
            valid_mask = np.pad(valid_mask, ((0, 0), (0, pad)), constant_values=False)

        adata.uns['niche_neighbors'] = nbr_idx
        adata.uns['niche_deltas'] = deltas
        adata.uns['niche_dists'] = nbr_dists
        adata.uns['niche_mask'] = valid_mask


# ---------------------------------------------------------------------------
# NicheEncoder
# ---------------------------------------------------------------------------

class NicheEncoder(nn.Module):
    """Graph attention pooling: KNN → joint MLP → attention → weighted sum + residual.

    Output: niche token (B, 1, hidden_dim), ready to concat into GiT token sequence.
    """

    def __init__(self, gene_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Neighbor joint encoding: [g_j, Δx, Δy, dist]
        self.nbr_encoder = nn.Sequential(
            nn.Linear(gene_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Center cell projection
        self.center_proj = nn.Linear(gene_dim + 2, hidden_dim)

        # Multi-head attention
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Residual
        self.res_proj = nn.Sequential(
            nn.Linear(gene_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, g_center, pos_center, g_nbrs, delta_nbrs, dist_nbrs, mask_nbr=None):
        """
        Args:
            g_center:   (B, G)         center cell gene expression (any t)
            pos_center: (B, 2)         center cell position (any t)
            g_nbrs:     (B, K, G)      neighbor gene expressions
            delta_nbrs: (B, K, 2)      relative (Δx, Δy) from center
            dist_nbrs:  (B, K)         Euclidean distances
            mask_nbr:   (B, K) bool    True = valid neighbor
        Returns:
            n_token:    (B, 1, D)      niche token
        """
        B, K, _ = g_nbrs.shape
        D = self.hidden_dim
        H = self.num_heads
        d = self.head_dim

        # --- neighbor encoding ---
        nbr_feat = torch.cat([g_nbrs, delta_nbrs, dist_nbrs.unsqueeze(-1)], dim=-1)  # (B,K,G+3)
        h_nbr = self.nbr_encoder(nbr_feat)                                            # (B,K,D)

        # --- center query ---
        c_feat = torch.cat([g_center, pos_center], dim=-1)                            # (B,G+2)
        h_ctr = self.center_proj(c_feat)                                              # (B,D)

        # --- multi-head attention ---
        q = self.W_q(h_ctr).view(B, H, d)              # (B,H,d)
        k = self.W_k(h_nbr).view(B, K, H, d)            # (B,K,H,d)
        v = self.W_v(h_nbr).view(B, K, H, d)            # (B,K,H,d)

        attn = torch.einsum('bhd,bkhd->bhk', q, k) * (d ** -0.5)  # (B,H,K)

        if mask_nbr is not None:
            attn = attn.masked_fill(~mask_nbr.unsqueeze(1), float('-inf'))

        attn = attn.softmax(dim=-1)                                            # (B,H,K)

        n_pooled = torch.einsum('bhk,bkhd->bhd', attn, v)                     # (B,H,d)
        n_pooled = n_pooled.reshape(B, D)                                      # (B,D)

        # --- residual ---
        n_token = self.res_proj(c_feat) + self.out_proj(n_pooled)              # (B,D)

        return n_token.unsqueeze(1)  # (B,1,D) — ready to concat
