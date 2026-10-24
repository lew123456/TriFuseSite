from __future__ import annotations
from pathlib import Path
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1

def map_structure(sequence, path, chain_hint=None, confidence_mode='observed'):
    n = len(sequence)
    empty = (np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), bool),
             np.zeros(n, np.float32), '')
    if path is None:
        return empty
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Structure not found: {path}; use null only for declared missing structures')
    if confidence_mode not in ('observed', 'plddt'):
        raise ValueError('confidence_mode must be observed or plddt')
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in ('.cif', '.mmcif') else PDBParser(QUIET=True)
    structure = parser.get_structure('target', str(path))
    chains = list(next(structure.get_models()).get_chains())
    if chain_hint is not None:
        preferred = [c for c in chains if c.id == chain_hint]
        if not preferred:
            raise ValueError(f'Requested target chain {chain_hint!r} not found in {path}')
        chains = preferred
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                             open_gap_score=-4, extend_gap_score=-0.5)
    best, best_coverage = empty, -1
    for chain in chains:
        residues = [r for r in chain if seq1(r.resname) in 'ACDEFGHIKLMNPQRSTVWY']
        observed = ''.join(seq1(r.resname) for r in residues)
        if not observed:
            continue
        alignment = aligner.align(sequence, observed)[0]
        xyz, mask, confidence, _ = (a.copy() if isinstance(a, np.ndarray) else a for a in empty)
        target_blocks, source_blocks = alignment.aligned
        for (t0, t1), (s0, s1) in zip(target_blocks, source_blocks):
            for i, j in zip(range(t0, t1), range(s0, s1)):
                if sequence[i] != observed[j]:
                    continue
                residue = residues[j]
                for atom_id, atom_name in enumerate(('N', 'CA', 'C')):
                    if atom_name in residue:
                        coords = np.asarray(residue[atom_name].coord, np.float32)
                        if np.isfinite(coords).all():
                            xyz[i, atom_id] = coords
                            mask[i, atom_id] = True
                if mask[i, 1]:
                    confidence[i] = (np.clip(residue['CA'].bfactor / 100, 0, 1)
                                     if confidence_mode == 'plddt' else 1.0)
        coverage = mask[:, 1].sum()
        if coverage > best_coverage:
            best, best_coverage = (xyz, mask, confidence, chain.id), coverage
    return best
