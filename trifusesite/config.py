from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class ModelConfig:
    residue_feature_dim: int = 62
    esm_dim: int = 1280
    semantic_projection_dim: int = 256
    hidden_dim: int = 256
    classifier_dim: int = 128
    num_classes: int = 2
    graph_radius_angstrom: float = 14.0
    radius_max_neighbors: int = 64
    patch_pooling_ratio: float = 0.7
    patch_layers: int = 3
    global_graph_layers: int = 8
    global_attention_heads: int = 4
    geometry_neighbors: int = 24
    geometry_layers: int = 2
    geometry_attention_heads: int = 4
    geometry_edge_dim: int = 24
    rbf_channels: int = 8
    rbf_min: float = 0.0
    rbf_max: float = 20.0
    rbf_width: float = 2.5
    dropout: float = 0.1
    initial_residual_weight: float = 0.7
    identity_mapping_strength: float = 1.5
    esm_gate_logit_init: float = -4.0
    geometry_gate_logit_init: float = -1.3862944

    def __post_init__(self):
        if self.hidden_dim % self.global_attention_heads or self.hidden_dim % self.geometry_attention_heads:
            raise ValueError('hidden_dim must be divisible by both attention head counts')
        if self.rbf_channels != 8 or self.geometry_edge_dim != 24:
            raise ValueError('This source release uses 8 RBF + 12 reserved zeros + 4 auxiliary channels')
        if not 0 < self.patch_pooling_ratio <= 1 or self.rbf_width <= 0:
            raise ValueError('Invalid pooling ratio or RBF width')

    def to_dict(self):
        return asdict(self)

def load_config(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        config = json.loads(path.read_text(encoding='utf-8'))
    else:
        import yaml
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if 'base' in config:
        import yaml
        base = yaml.safe_load((path.parent / config.pop('base')).read_text(encoding='utf-8'))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
    if config['task'] not in ('ppi', 'smpi', 'dpi', 'rpi'):
        raise ValueError('Only the four manuscript tasks are supported')
    ModelConfig(**config.get('model', {}))
    training = config.get('training', {})
    required = {'optimizer': 'AdamW', 'learning_rate': 2e-4,
                'weight_decay': 1e-4, 'batch_size': 1, 'gradient_clip': 1.0}
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f'Table S2 requires {key}={expected}')
    return config
