from __future__ import annotations
import numpy as np
from ..config import ModelConfig

def radius_graph(coordinates, mask, config=None, distance_map=None):
    """14-Angstrom graph with at most 64 non-self neighbors per target.

    PPI supplies a precomputed distance map; its graph is independent of the
    local backbone mask. Other tasks use only valid C-alpha coordinates.
    Fallback edges are sequence edges and carry no fabricated geometry.
    """
    c = config or ModelConfig()
    xyz = np.asarray(coordinates, dtype=np.float32)
    n = len(xyz)
    valid = np.asarray(mask, bool) & np.isfinite(xyz).all(1)
    supplied = distance_map is not None
    if supplied:
        distances = np.asarray(distance_map, dtype=np.float32).copy()
        if distances.shape != (n, n):
            raise ValueError('Precomputed distance map must be [L,L]')
        distances[(distances < 0) | ~np.isfinite(distances)] = np.inf
    else:
        # Invalid placeholders never enter distance calculations.
        distances = np.full((n, n), np.inf, np.float32)
        ids = np.flatnonzero(valid)
        d = xyz[ids, None] - xyz[None, ids]
        distances[np.ix_(ids, ids)] = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(distances, np.inf)
    directed = np.zeros((n, n), bool)
    for target in range(n):
        source = np.flatnonzero(distances[target] <= c.graph_radius_angstrom)
        source = source[np.argsort(distances[target, source], kind='stable')[:c.radius_max_neighbors]]
        directed[target, source] = True
    # Mutual cap preserves symmetry and the documented maximum degree.
    adjacency = directed & directed.T
    fallback = False
    if not adjacency.any() and not supplied:
        fallback = True
        i = np.arange(max(n - 1, 0))
        adjacency[i, i + 1] = adjacency[i + 1, i] = True
    np.fill_diagonal(adjacency, True)
    degree = adjacency.sum(1).astype(np.float32)
    norm = adjacency.astype(np.float32) / np.sqrt(degree[:, None] * degree[None, :])
    target, source = np.nonzero(adjacency)
    edge = np.zeros((len(source), 2), np.float32)
    geometry_valid = (valid[source] & valid[target]) if not supplied else np.ones(len(source), bool)
    if fallback:
        geometry_valid[:] = False
    if geometry_valid.any():
        # Relative position vectors share the first valid residue as origin.
        origin = xyz[np.flatnonzero(valid)[0]] if valid.any() else np.zeros(3)
        relative = xyz - origin
        a, b = relative[source[geometry_valid]], relative[target[geometry_valid]]
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide((a * b).sum(1), den, out=np.zeros_like(den), where=den > 1e-8)
        edge[geometry_valid, 0] = np.linalg.norm(a - b, axis=1) / c.graph_radius_angstrom
        edge[geometry_valid, 1] = (np.clip(cosine, -1, 1) + 1) / 2
    return norm.astype(np.float32), np.stack([source, target]).astype(np.int64), edge, fallback
