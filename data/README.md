# Model input preparation

Prepared caches are written to `data/<task>/<split>/`, matching the task YAML
files. `main_prepare_datasets.py` reads the fixed raw manifests in `dataset/`.

## Structures

Default coordinate locations are `dataset/structures/PDB/<pdb_id>.pdb` (or
`.cif`) and `dataset/structures/AlphaFoldDB_v6/<accession>.pdb`. Alternatively,
pass `--structure-index structures.jsonl`, containing one record per override:

```json
{"task":"ppi","id":"2c0lA","path":"coordinates/2c0l.pdb","chain":"A"}
```

Paths are relative to the index file. A verified unavailable structure is
specified by `path: null` and a `reason`. Otherwise missing files are reported
by `--check-only`. Sequence and label positions are retained under masking.
The first model is used; chain selection and exact-residue alignment determine
the valid coordinates. Experimental coordinates use observed-position confidence;
AlphaFold coordinates use pLDDT/100.

## PPI geometry

Provide the original benchmark arrays under `dataset/geometry/ppi/` or pass
`--ppi-geometry-root` pointing to the following layout:

```text
pseudopositions/<protein_id>.npy   # [L,3], Angstrom
distance_map/<protein_id>.npy      # [L,L], Angstrom
```

PPI features concatenate PSSM (20), HMM (20), DSSP (14), resAF (7), and
`norm(position_i-position_0)/15` (1). SMPI/DPI/RPI construct their 62-dimensional
descriptors from the sequence. ESM inputs have shape `[L,1280]`.

## Prepared files

Each partition contains `manifest.jsonl`, protein `.npz` files, and generated
`.esm.npy` embeddings. The NPZ stores features, labels, CA coordinates,
structural confidence/masks and graph arrays. A manifest entry is:

```json
{"id":"example","npz":"example.npz","esm":"example.esm.npy"}
```

`main_prepare.py` also accepts a custom raw JSONL with `id`, `sequence`,
`labels`, `structure_path`, `chain` and `structure_source`; PPI additionally
requires `ppi_features` paths for the four descriptor arrays and two geometry
arrays. Use `scripts/from_triplet_fasta.py` to convert three-line benchmark files.
