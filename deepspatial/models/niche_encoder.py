"""Multi-scale niche encoder with out_key support for KNN precomputation.

Spatial KNN × 3 scales → per-scale neighbor encoding → per-scale attention →
3 niche tokens concatenated.

local  (K=8):   direct cell-cell contact (~5-10μm)
mid    (K=32):  paracrine signalling (~15-30μm)
global (K=128): tissue architecture (~50-80μm)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# KNN precompute
# ---------------------------------------------------------------------------

def precompute_neighbors(adata_list, spatial_key='spatial_norm', K=32,
                        out_key='niche'):
    """Precompute K nearest spatial neighbors, stored under adata.uns[f'{out_key}_*']."""
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

        adata.uns[f'{out_key}_neighbors'] = nbr_idx
        adata.uns[f'{out_key}_deltas'] = deltas
        adata.uns[f'{out_key}_dists'] = nbr_dists
        adata.uns[f'{out_key}_mask'] = valid_mask


def precompute_multiscale_neighbors(adata_list, spatial_key='spatial_norm'):
    """Precompute three KNN layers for multi-scale niche encoding."""
    for K, key in [(8, 'niche_local'), (32, 'niche_mid'), (128, 'niche_global')]:
        precompute_neighbors(adata_list, spatial_key=spatial_key, K=K, out_key=key)


# ---------------------------------------------------------------------------
# NicheEncoder (original, single-scale)
# ---------------------------------------------------------------------------

class NicheEncoder(nn.Module):
    """Original graph attention pooling niche encoder. Returns (B, 1, D)."""

    def __init__(self, gene_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.nbr_encoder = nn.Sequential(
            nn.Linear(gene_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.center_proj = nn.Linear(gene_dim + 2, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
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
        B, K, _ = g_nbrs.shape
        D = self.hidden_dim
        H = self.num_heads
        d = self.head_dim

        nbr_feat = torch.cat([g_nbrs, delta_nbrs, dist_nbrs.unsqueeze(-1)], dim=-1)
        h_nbr = self.nbr_encoder(nbr_feat)

        c_feat = torch.cat([g_center, pos_center], dim=-1)
        h_ctr = self.center_proj(c_feat)

        q = self.W_q(h_ctr).view(B, H, d)
        k = self.W_k(h_nbr).view(B, K, H, d)
        v = self.W_v(h_nbr).view(B, K, H, d)

        attn = torch.einsum('bhd,bkhd->bhk', q, k) * (d ** -0.5)
        if mask_nbr is not None:
            attn = attn.masked_fill(~mask_nbr.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)

        n_pooled = torch.einsum('bhk,bkhd->bhd', attn, v).reshape(B, D)
        n_token = self.res_proj(c_feat) + self.out_proj(n_pooled)
        return n_token.unsqueeze(1)


# ---------------------------------------------------------------------------
# MultiScaleNicheEncoder
# ---------------------------------------------------------------------------

class MultiScaleNicheEncoder(nn.Module):
    """Multi-scale niche encoder with cell type embedding and pooled global.

    local  (K=8):   cross-attention over nearest neighbors
    mid    (K=32):  cross-attention over mid-range neighbors
    global (K=128): mean-pool gene + cell type histogram → MLP (no cross-attn)

    Center and neighbor cell types embedded via learnable Embedding for
    natural Q·K interaction in attention, or as histogram in pooled global.
    """

    def __init__(self, gene_dim, num_classes, hidden_dim=128, num_heads=4, ct_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_classes = num_classes

        # Cell type embedding (shared)
        self.ct_emb = nn.Embedding(num_classes, ct_embed_dim)

        # --- Shared modules for local/mid (cross-attention) ---
        self.nbr_encoder = nn.Sequential(
            nn.Linear(gene_dim + 3 + ct_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.center_proj = nn.Linear(gene_dim + 2 + ct_embed_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Per-scale query projections
        self.W_q_local = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q_mid = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Per-scale output projections
        self.out_proj_local = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_mid = nn.Linear(hidden_dim, hidden_dim)

        # Shared residual (includes ct_emb)
        self.res_proj = nn.Sequential(
            nn.Linear(gene_dim + 2 + ct_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Global scale: mean-pooled gene + cell type histogram → MLP ---
        self.global_pool_encoder = nn.Sequential(
            nn.Linear(gene_dim + num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for proj in [self.out_proj_local, self.out_proj_mid]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.global_pool_encoder[-1].weight)
        nn.init.zeros_(self.global_pool_encoder[-1].bias)
        nn.init.normal_(self.ct_emb.weight, std=0.02)

    def _cross_attn_scale(self, h_ctr, W_q, h_nbr, mask_nbr):
        B, K_nbr, D = h_nbr.shape
        H = self.num_heads
        d = self.head_dim

        q = W_q(h_ctr).view(B, H, d)
        k = self.W_k(h_nbr).view(B, K_nbr, H, d)
        v = self.W_v(h_nbr).view(B, K_nbr, H, d)

        attn = torch.einsum('bhd,bkhd->bhk', q, k) * (d ** -0.5)
        if mask_nbr is not None:
            attn = attn.masked_fill(~mask_nbr.unsqueeze(1), float('-inf'))
        attn = attn.softmax(dim=-1)

        return torch.einsum('bhk,bkhd->bhd', attn, v).reshape(B, D)

    def _encode(self, g_nbrs, delta_nbrs, dist_nbrs, ct_nbrs):
        ct_emb = self.ct_emb(ct_nbrs)
        print("awdwa", g_nbrs.shape, delta_nbrs.shape, dist_nbrs.shape, ct_emb.shape)
        feat = torch.cat([g_nbrs, delta_nbrs, dist_nbrs.unsqueeze(-1), ct_emb], dim=-1)
        return self.nbr_encoder(feat)

    def _center_feat(self, g_center, pos_center, ct_center):
        ct_emb = self.ct_emb(ct_center)  # (B, embed_dim)
        return torch.cat([g_center, pos_center, ct_emb], dim=-1)

    def forward_scale(self, scale, g_center, pos_center, ct_center,
                      g_nbrs, delta_nbrs, dist_nbrs, mask_nbr, ct_nbrs,
                      h_ctr=None, residual=None):
        """Process one scale, return (B, D). h_ctr/residual precomputed and reused."""
        if h_ctr is None or residual is None:
            c_feat = self._center_feat(g_center, pos_center, ct_center)
            h_ctr = self.center_proj(c_feat)
            residual = self.res_proj(c_feat)

        if scale == 'global':
            g_pooled = g_nbrs.mean(dim=1)  # (B, G)
            ct_hist = torch.zeros(g_center.shape[0], self.num_classes,
                                  device=g_center.device)
            ct_hist.scatter_add_(1, ct_nbrs,
                                 torch.ones_like(ct_nbrs, dtype=torch.float32))
            ct_hist = ct_hist / (ct_hist.sum(dim=1, keepdims=True) + 1e-8)
            h = torch.cat([g_pooled, ct_hist], dim=-1)
            return residual + self.global_pool_encoder(h)
        else:
            W_q = {'local': self.W_q_local, 'mid': self.W_q_mid}[scale]
            out_proj = {'local': self.out_proj_local, 'mid': self.out_proj_mid}[scale]
            h_nbr = self._encode(g_nbrs, delta_nbrs, dist_nbrs, ct_nbrs)
            pooled = self._cross_attn_scale(h_ctr, W_q, h_nbr, mask_nbr)
            return residual + out_proj(pooled)

    def forward(self, g_center, pos_center, ct_center,
                g_nbrs_local, delta_local, dist_local, mask_local, ct_local,
                g_nbrs_mid, delta_mid, dist_mid, mask_mid, ct_mid,
                g_nbrs_global, delta_global, dist_global, mask_global, ct_global):
        B = g_center.shape[0]
        D = self.hidden_dim

        c_feat = self._center_feat(g_center, pos_center, ct_center)
        h_ctr = self.center_proj(c_feat)
        residual = self.res_proj(c_feat)

        # Local (cross-attention)
        h_local = self._encode(g_nbrs_local, delta_local, dist_local, ct_local)
        tok_local = residual + self.out_proj_local(
            self._cross_attn_scale(h_ctr, self.W_q_local, h_local, mask_local))

        # Mid (cross-attention)
        h_mid = self._encode(g_nbrs_mid, delta_mid, dist_mid, ct_mid)
        tok_mid = residual + self.out_proj_mid(
            self._cross_attn_scale(h_ctr, self.W_q_mid, h_mid, mask_mid))

        # Global (pooled)
        g_pooled = g_nbrs_global.mean(dim=1)
        ct_hist = torch.zeros(B, self.num_classes, device=g_center.device)
        ct_hist.scatter_add_(1, ct_global,
                             torch.ones_like(ct_global, dtype=torch.float32))
        ct_hist = ct_hist / (ct_hist.sum(dim=1, keepdims=True) + 1e-8)
        tok_global = residual + self.global_pool_encoder(
            torch.cat([g_pooled, ct_hist], dim=-1))

        return torch.stack([tok_local, tok_mid, tok_global], dim=1)  # (B, 3, D)
