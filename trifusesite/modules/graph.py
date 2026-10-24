from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


def segment_softmax(
    scores: torch.Tensor,
    destination: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax over incoming edges for every destination node."""
    squeeze = scores.ndim == 1
    if squeeze:
        scores = scores[:, None]
    index = destination[:, None].expand_as(scores)
    maxima = torch.full(
        (num_nodes, scores.shape[1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[destination])
    denominator = torch.zeros_like(maxima).index_add_(0, destination, weights)
    result = weights / denominator[destination].clamp_min(1e-12)
    return result.squeeze(-1) if squeeze else result


def adjacency_matmul(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if adjacency.layout != torch.strided:
        return torch.sparse.mm(adjacency, features)
    return adjacency @ features


class ResidualGraphConvolution(nn.Module):
    """Residual GCNII-style propagation used inside the patch branch."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * hidden_dim, hidden_dim))
        nn.init.uniform_(
            self.weight,
            -1.0 / math.sqrt(hidden_dim),
            1.0 / math.sqrt(hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        initial: torch.Tensor,
        layer_index: int,
        mapping: float = 1.5,
        residual: float = 0.7,
    ) -> torch.Tensor:
        theta = min(1.0, math.log(mapping / layer_index + 1.0))
        propagated = adjacency_matmul(adjacency, features)
        support = torch.cat([propagated, initial], dim=-1)
        initial_mix = (1.0 - residual) * propagated + residual * initial
        update = theta * (support @ self.weight) + (1.0 - theta) * initial_mix
        return update + features


class InterfacePatchEncoder(nn.Module):
    """GCN-scored top-k interface patches with two-hop propagation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ratio = config.patch_pooling_ratio
        self.input_projection = nn.Linear(
            config.residue_feature_dim,
            config.hidden_dim,
        )
        self.score = nn.Linear(config.hidden_dim, 1)
        self.convolutions = nn.ModuleList(
            [ResidualGraphConvolution(config.hidden_dim) for _ in range(config.patch_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.relu(self.input_projection(features))
        degree = normalized_adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        propagated = normalized_adjacency @ hidden / degree
        scores = torch.sigmoid(self.score(propagated).squeeze(-1))

        count = min(len(hidden), max(2, int(self.ratio * len(hidden))))
        selected = torch.topk(scores, count).indices
        pooled = hidden[selected] * scores[selected, None]

        binary = normalized_adjacency.bool().float()
        two_hop = (binary @ binary).bool().float()
        pooled_adjacency = two_hop[selected][:, selected]
        initial = pooled
        for layer_index, convolution in enumerate(self.convolutions, start=1):
            pooled = F.dropout(pooled, self.config.dropout, training=self.training)
            pooled = F.relu(
                pooled
                + convolution(
                    pooled,
                    pooled_adjacency,
                    initial,
                    layer_index,
                    self.config.identity_mapping_strength,
                    self.config.initial_residual_weight,
                )
            )

        restored = torch.zeros_like(hidden)
        restored[selected] = pooled
        return self.norm(hidden + restored)


class EdgeAwareGraphAttention(nn.Module):
    """Global residue-graph attention with distance/cosine edge features."""

    def __init__(self, hidden_dim: int, edge_dim: int = 2, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_attention_projection = nn.Linear(edge_dim, edge_dim, bias=False)
        self.edge_message_projection = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 2 * self.head_dim + edge_dim))

        gain = nn.init.calculate_gain("relu")
        for module in (
            self.node_projection,
            self.edge_attention_projection,
            self.edge_message_projection,
        ):
            nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.xavier_normal_(self.attention, gain=gain)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, destination = edge_index.long()
        projected = self.node_projection(hidden).view(len(hidden), self.heads, self.head_dim)
        edge_for_attention = self.edge_attention_projection(edge_features)
        joined = torch.cat(
            [projected[source], projected[destination],
             edge_for_attention[:, None, :].expand(-1, self.heads, -1)],
            dim=-1,
        )
        scores = F.leaky_relu((joined * self.attention[None]).sum(-1))
        weights = segment_softmax(scores, destination, len(hidden))
        messages = (
            projected[source]
            + self.edge_message_projection(edge_for_attention).view(-1, self.heads, self.head_dim)
        ) * weights[..., None]
        # Attention parameters may promote scores to FP32 under BF16 autocast.
        # Accumulate in the message dtype rather than mixing index_add_ dtypes.
        aggregate = torch.zeros(projected.shape, device=projected.device, dtype=messages.dtype)
        return aggregate.index_add_(0, destination, messages).flatten(1)


class GraphPropagationBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.attention = EdgeAwareGraphAttention(hidden_dim, heads=heads)
        self.weight = nn.Parameter(torch.empty(2 * hidden_dim, hidden_dim))
        nn.init.uniform_(
            self.weight,
            -1.0 / math.sqrt(hidden_dim),
            1.0 / math.sqrt(hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        initial: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        layer_index: int,
        mapping: float,
        residual: float,
    ) -> torch.Tensor:
        theta = min(1.0, math.log(mapping / layer_index + 1.0))
        propagated = self.attention(hidden, edge_index, edge_features)
        support = torch.cat([propagated, initial], dim=-1)
        initial_mix = (1.0 - residual) * propagated + residual * initial
        update = theta * (support @ self.weight) + (1.0 - theta) * initial_mix
        return update + hidden


class GlobalGraphEncoder(nn.Module):
    """Eight-layer global AGAT branch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.residue_feature_dim,
            config.hidden_dim,
        )
        self.blocks = nn.ModuleList(
            [
                GraphPropagationBlock(config.hidden_dim, config.global_attention_heads)
                for _ in range(config.global_graph_layers)
            ]
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.dropout(
            features,
            self.config.dropout,
            training=self.training,
        )
        hidden = F.relu(self.input_projection(hidden))
        initial = hidden
        for layer_index, block in enumerate(self.blocks, start=1):
            hidden = F.dropout(
                hidden,
                self.config.dropout,
                training=self.training,
            )
            hidden = F.relu(
                block(
                    hidden,
                    initial,
                    edge_index,
                    edge_features,
                    layer_index,
                    self.config.identity_mapping_strength,
                    self.config.initial_residual_weight,
                )
            )
        hidden = F.dropout(
            hidden,
            self.config.dropout,
            training=self.training,
        )
        return self.output_projection(hidden)


class BranchCrossAttention(nn.Module):
    """One-head patch-to-global cross-branch attention."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        patch_hidden: torch.Tensor,
        global_hidden: torch.Tensor,
    ) -> torch.Tensor:
        query = F.relu(self.query(patch_hidden))
        key = F.relu(self.key(patch_hidden))
        value = F.relu(self.value(global_hidden))
        # Full-chain attention in Eq. (7), not a radius-neighbor approximation.
        attended = F.scaled_dot_product_attention(
            query[None, None], key[None, None], value[None, None], dropout_p=0.0
        )[0, 0]
        return F.relu(self.output(attended))


class DualScaleGraphEncoder(nn.Module):
    """Patch branch + global AGAT branch + cross-branch fusion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_encoder = InterfacePatchEncoder(config)
        self.global_encoder = GlobalGraphEncoder(config)
        self.cross_attention = BranchCrossAttention(config.hidden_dim)
        self.fusion = nn.Linear(2 * config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        patch_adjacency: torch.Tensor,
        global_edge_index: torch.Tensor,
        global_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patch = self.patch_encoder(features, patch_adjacency)
        global_hidden = self.global_encoder(
            features,
            global_edge_index,
            global_edge_features,
        )
        attended_patch = self.cross_attention(patch, global_hidden)
        graph_hidden = F.relu(
            self.fusion(torch.cat([attended_patch, global_hidden], dim=-1))
        )
        return graph_hidden, {
            "patch": patch,
            "global": global_hidden,
            "attended_patch": attended_patch,
        }
