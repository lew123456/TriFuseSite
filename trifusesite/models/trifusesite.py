from __future__ import annotations
import torch
from torch import nn
from ..config import ModelConfig
from ..modules.semantic import SemanticResidualFusion
from ..modules.graph import DualScaleGraphEncoder
from ..modules.geometry import DistanceGeometryEncoder, build_distance_geometry_graph

class TriFuseSite(nn.Module):
    """One shared architecture; separately trained weights for four tasks.

    No CRF, directional/rotation features, or joint multitask training.
    PPI retains the supplied Patch/Global graphs when C-alpha atoms are missing.
    Other tasks scale the geometry gate by whole-protein structural coverage.
    """
    def __init__(self, config=None, task='ppi'):
        super().__init__()
        if task not in ('ppi', 'smpi', 'dpi', 'rpi'):
            raise ValueError('Unknown interaction task')
        self.task = task
        self.config = config or ModelConfig()
        c = self.config
        self.semantic_fusion = SemanticResidualFusion(c)
        self.graph_encoder = DualScaleGraphEncoder(c)
        self.geometry_encoder = DistanceGeometryEncoder(c)
        self.geometry_gate_logit = nn.Parameter(torch.tensor(c.geometry_gate_logit_init))
        self.fusion_norm = nn.LayerNorm(c.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(c.hidden_dim, c.classifier_dim),
                                        nn.ReLU(), nn.Dropout(c.dropout),
                                        nn.Linear(c.classifier_dim, c.num_classes))

    def forward(self, residue_features, esm_embedding, patch_adjacency,
                global_edge_index, global_edge_features, ca, structure_mask,
                confidence, *, return_intermediates=False):
        x0, esm_gate = self.semantic_fusion(residue_features, esm_embedding)
        hgraph, parts = self.graph_encoder(x0, patch_adjacency,
                                           global_edge_index, global_edge_features)
        geometry = build_distance_geometry_graph(ca, structure_mask, confidence, self.config)
        hgeo = self.geometry_encoder(x0, geometry)
        coverage = structure_mask.to(x0.dtype).mean()
        gate = torch.sigmoid(self.geometry_gate_logit)
        if self.task != 'ppi':
            gate = gate * coverage
        h = self.fusion_norm(hgraph + gate * hgeo)
        logits = self.classifier(h)
        if return_intermediates:
            return logits, {'X0': x0, 'Hpatch': parts['patch'],
                            'Hglobal': parts['global'], 'Hatt': parts['attended_patch'],
                            'Hgraph': hgraph, 'Hgeo': hgeo, 'H': h,
                            'esm_gate': esm_gate, 'geometry_gate': gate,
                            'structural_coverage': coverage}
        return logits
