"""Multi-token niche encoder with learnable queries.

Spatial KNN → neighbor encoding → K learnable queries × cross-attention →
K niche tokens.  Each learnable query probes a different microenvironment facet.
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
    """K learnable query vectors cross-attend over the KNN neighborhood.

    Each query is a global parameter that learns to probe a specific
    microenvironment aspect (e.g. immune, stromal, vascular).
    """

    def __init__(self, gene_dim, hidden_dim=128, num_heads=4, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Neighbor encoding: [g_j, dx, dy, dist, c_j]
        self.nbr_encoder = nn.Sequential(
            nn.Linear(gene_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # K learnable query vectors (shared across all cells)
        self.niche_queries = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        # K/V projections
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.niche_queries, std=0.02)

    def forward(self, g_center, pos_center, g_nbrs, delta_nbrs, dist_nbrs, mask_nbr=None):
        """
        Args:
            g_center:   (B, G)      center cell gene expression
            pos_center: (B, 2)      center cell position
            g_nbrs:     (B, N, G)   neighbor gene expressions
            delta_nbrs: (B, N, 2)   relative (dx, dy) from center
            dist_nbrs:  (B, N)      Euclidean distances
            mask_nbr:   (B, N) bool True = valid neighbor
        Returns:
            niche_tokens: (B, K, D) K niche tokens
        """
        B, N, _ = g_nbrs.shape
        K = self.num_tokens
        H = self.num_heads
        d = self.head_dim

        # --- neighbor encoding ---
        nbr_feat = torch.cat([g_nbrs, delta_nbrs, dist_nbrs.unsqueeze(-1)], dim=-1)
        h_nbr = self.nbr_encoder(nbr_feat)                                # (B, N, D)

        # --- K learnable queries ---
        q = self.niche_queries.unsqueeze(0).expand(B, -1, -1)             # (B, K, D)

        k = self.W_k(h_nbr)                                               # (B, N, D)
        v = self.W_v(h_nbr)                                               # (B, N, D)

        # --- multi-head reshape ---
        q = q.view(B, K, H, d)                                            # (B, K, H, d)
        k = k.view(B, N, H, d)                                            # (B, N, H, d)
        v = v.view(B, N, H, d)

        # --- cross-attention ---
        attn = torch.einsum('bkhd,bnhd->bhkn', q, k) * (d ** -0.5)        # (B, H, K, N)

        if mask_nbr is not None:
            attn = attn.masked_fill(~mask_nbr[:, None, None, :], float('-inf'))

        attn = attn.softmax(dim=-1)                                       # (B, H, K, N)

        n_tokens = torch.einsum('bhkn,bnhd->bkhd', attn, v)              # (B, K, H, d)
        n_tokens = n_tokens.reshape(B, K, -1)                            # (B, K, D)

        return n_tokens
