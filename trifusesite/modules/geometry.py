from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..config import ModelConfig
from .graph import segment_softmax

def build_distance_geometry_graph(ca, mask, confidence, config=None):
    c = config or ModelConfig()
    n = len(ca)
    if ca.shape != (n, 3) or mask.shape != (n,) or confidence.shape != (n,):
        raise ValueError('Expected ca [L,3], mask [L], confidence [L]')
    valid = mask.bool() & torch.isfinite(ca).all(-1)
    if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('Confidence must be finite and in [0,1]')
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if len(valid_ids) < 2:
        return {'edge_index': torch.empty((2, 0), device=ca.device, dtype=torch.long),
                'edge_features': ca.new_empty((0, c.geometry_edge_dim))}
    xyz = ca[valid_ids]
    dist = torch.cdist(xyz, xyz)
    dist.fill_diagonal_(torch.inf)
    k = min(c.geometry_neighbors, len(valid_ids) - 1)
    nearest = torch.argsort(dist, dim=1, stable=True)[:, :k]
    destination = valid_ids.repeat_interleave(k)
    source = valid_ids[nearest.flatten()]
    distances = (ca[source] - ca[destination]).norm(dim=-1)
    centers = torch.linspace(c.rbf_min, c.rbf_max, c.rbf_channels,
                             device=ca.device, dtype=ca.dtype)
    rbf = torch.exp(-((distances[:, None] - centers) / c.rbf_width).square())
    gap = (source - destination).abs().to(ca.dtype)
    separation = torch.stack([torch.log1p(gap) / math.log1p(max(n, 2)),
                              (gap > 12).to(ca.dtype)], -1)
    pair_confidence = torch.minimum(confidence[source], confidence[destination])[:, None]
    edge = torch.cat([rbf, ca.new_zeros((len(source), 12)), separation,
                      pair_confidence, ca.new_ones((len(source), 1))], dim=-1)
    return {'edge_index': torch.stack([source, destination]), 'edge_features': edge}

class DistanceGeometryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.heads = config.geometry_attention_heads
        self.head_dim = d // self.heads
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.edge_encoder = nn.Sequential(nn.Linear(config.geometry_edge_dim, d),
                                           nn.GELU(), nn.Linear(d, d))
        self.edge_bias = nn.Linear(d, self.heads, bias=False)
        self.message_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(2*d, d))
        self.output_norm = nn.LayerNorm(d)

    def forward(self, hidden, geometry):
        edge = geometry['edge_features']
        keep = edge[:, -1] > 0.5
        source, destination = geometry['edge_index'][:, keep].long()
        edge = edge[keep]
        n = len(hidden)
        aggregate = torch.zeros_like(hidden)
        if source.numel():
            z = self.edge_encoder(edge)
            q = self.query(hidden).view(n, self.heads, self.head_dim)
            k = self.key(hidden).view(n, self.heads, self.head_dim)
            v = self.value(hidden).view(n, self.heads, self.head_dim)
            scores = (q[destination] * k[source]).sum(-1) / math.sqrt(self.head_dim)
            scores = scores + self.edge_bias(z)
            alpha = segment_softmax(scores, destination, n)
            message = (v[source] + z.view(-1, self.heads, self.head_dim)) * alpha[..., None]
            aggregate.index_add_(0, destination, message.flatten(1))
        u = self.message_norm(hidden + aggregate)
        return self.output_norm(u + self.ffn(u))

class DistanceGeometryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_projection = nn.Linear(config.residue_feature_dim, config.hidden_dim)
        self.layers = nn.ModuleList([DistanceGeometryLayer(config) for _ in range(config.geometry_layers)])

    def forward(self, features, geometry):
        h = F.gelu(self.input_projection(features))
        for layer in self.layers:
            h = layer(h, geometry)
        return h
