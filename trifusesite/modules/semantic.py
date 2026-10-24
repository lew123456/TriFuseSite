from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class SemanticResidualFusion(nn.Module):
    """Fuse frozen ESM2-650M residue embeddings with 62D residue features.

    ESM2 itself is not trained here. Its precomputed per-residue embedding is
    normalized, projected back to 62 dimensions, and injected through one
    learnable scalar residual gate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.esm_norm = nn.LayerNorm(config.esm_dim)
        self.esm_projection = nn.Sequential(
            nn.Linear(config.esm_dim, config.semantic_projection_dim),
            nn.GELU(),
            nn.Linear(config.semantic_projection_dim, config.residue_feature_dim),
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(config.esm_gate_logit_init, dtype=torch.float32)
        )

    def forward(
        self,
        residue_features: torch.Tensor,
        esm_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused residue features and the scalar ESM residual weight.

        Args:
            residue_features: [L, 62] handcrafted residue descriptors.
            esm_embedding: [L, 1280] frozen ESM2-650M embeddings.
        """
        if residue_features.ndim != 2 or esm_embedding.ndim != 2:
            raise ValueError("residue_features and esm_embedding must be 2D")
        if residue_features.shape[0] != esm_embedding.shape[0]:
            raise ValueError("residue and ESM sequence lengths do not match")
        if residue_features.shape[1] != self.config.residue_feature_dim:
            raise ValueError("unexpected residue feature dimension")
        if esm_embedding.shape[1] != self.config.esm_dim:
            raise ValueError("unexpected ESM embedding dimension")

        # Cached float16 embeddings are fixed inputs, not trainable parameters.
        esm_embedding = esm_embedding.detach().to(residue_features.dtype)
        esm_residual = self.esm_projection(self.esm_norm(esm_embedding))
        gate = torch.sigmoid(self.gate_logit)
        fused = residue_features + gate * esm_residual
        return fused, gate
