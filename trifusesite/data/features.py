from __future__ import annotations

import math
import numpy as np
from Bio.Align import substitution_matrices

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy, approximate volume, polarity, charge, aromatic,
# flexibility and helix propensity. Columns are standardized below.
PROPS = {
    "A": [1.8, 88.6, 8.1, 0, 0, 0.36, 1.45], "C": [2.5, 108.5, 5.5, 0, 0, 0.35, 0.77],
    "D": [-3.5, 111.1, 13.0, -1, 0, 0.51, 0.98], "E": [-3.5, 138.4, 12.3, -1, 0, 0.50, 1.53],
    "F": [2.8, 189.9, 5.2, 0, 1, 0.31, 1.12], "G": [-0.4, 60.1, 9.0, 0, 0, 0.54, 0.53],
    "H": [-3.2, 153.2, 10.4, 0.1, 1, 0.32, 1.24], "I": [4.5, 166.7, 5.2, 0, 0, 0.46, 1.00],
    "K": [-3.9, 168.6, 11.3, 1, 0, 0.47, 1.07], "L": [3.8, 166.7, 4.9, 0, 0, 0.37, 1.34],
    "M": [1.9, 162.9, 5.7, 0, 0, 0.30, 1.20], "N": [-3.5, 114.1, 11.6, 0, 0, 0.46, 0.73],
    "P": [-1.6, 112.7, 8.0, 0, 0, 0.51, 0.59], "Q": [-3.5, 143.8, 10.5, 0, 0, 0.49, 1.17],
    "R": [-4.5, 173.4, 10.5, 1, 0, 0.53, 0.79], "S": [-0.8, 89.0, 9.2, 0, 0, 0.51, 0.79],
    "T": [-0.7, 116.1, 8.6, 0, 0, 0.44, 0.82], "V": [4.2, 140.0, 5.9, 0, 0, 0.39, 1.14],
    "W": [-0.9, 227.8, 5.4, 0, 1, 0.31, 1.14], "Y": [-1.3, 193.6, 6.2, 0, 1, 0.42, 0.61],
}

# Index both BLOSUM62 axes by the declared amino-acid alphabet.
_standard_blosum = substitution_matrices.load('BLOSUM62')
BLOSUM = np.array([[_standard_blosum[a, b] for b in AA] for a in AA], dtype=np.float32)
PROP = np.array([PROPS[a] for a in AA], dtype=np.float32)
PROP = (PROP - PROP.mean(0)) / (PROP.std(0) + 1e-6)


def handcrafted62(sequence: str) -> np.ndarray:
    """The 62D sequence descriptor for SMPI/DPI/RPI (Methods 4.2.1).

    PPI must use its precomputed evolutionary/structural descriptors instead.
    """
    sequence = sequence.upper()
    if not sequence:
        raise ValueError('Empty sequences are not supported')
    n = len(sequence)
    one = np.zeros((n, 20), np.float32)
    blo = np.zeros((n, 20), np.float32)
    prop = np.zeros((n, 7), np.float32)
    for i, aa in enumerate(sequence):
        j = AA_INDEX.get(aa)
        if j is not None:
            one[i, j] = 1
            blo[i] = BLOSUM[j] / 11.0
            prop[i] = PROP[j]
    pos = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    positional = np.stack([pos, 1-pos, np.sin(math.pi*pos), np.cos(math.pi*pos),
                           np.minimum(pos, 1-pos)], axis=1)
    local = np.zeros((n, 10), np.float32)
    hydro = set("AILMFWVY")
    charge = {"K": 1, "R": 1, "H": .1, "D": -1, "E": -1}
    for i in range(n):
        s = sequence[max(0, i-4):min(n, i+5)]
        counts = np.array([s.count(a) for a in AA], np.float32) / len(s)
        nz = counts[counts > 0]
        local[i] = [
            sum(a in hydro for a in s)/len(s),
            sum(charge.get(a, 0) for a in s)/len(s),
            -float((nz*np.log(nz)).sum())/math.log(20),
            s.count("G")/len(s), s.count("P")/len(s), s.count("C")/len(s),
            sum(a in "STNQ" for a in s)/len(s),
            sum(a in "FWY" for a in s)/len(s),
            len(s)/9.0, 1.0,
        ]
    out = np.concatenate([one, blo, prop, positional, local], axis=1)
    assert out.shape == (n, 62)
    return out
