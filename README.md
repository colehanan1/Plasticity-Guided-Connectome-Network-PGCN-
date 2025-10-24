# Plasticity-Guided Connectome Network (PGCN)

The Plasticity-Guided Connectome Network repository provides a reproducible
pipeline for extracting and analysing the FlyWire projection-neuron
subgraph (PN→KC→MBON core plus DAN ancillary pathways). The codebase pins
FlyWire materialization versions, writes schema-stable cache artefacts, and
exposes command line interfaces for cache generation and structural metrics.

## Quickstart

1. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate PGCN
   ```

2. **Provision your FlyWire token**

   First (re)install the project in editable mode so the latest console scripts
   are registered on your `$PATH`:

   ```bash
   python -m pip install -e .[dev]
   ```

   The `[dev]` extra is optional but convenient if you plan to run tests. When
   you prefer not to install development dependencies, drop the extras suffix.

   You can now write the token to the expected secret path either via the
   installed entry point **or** via the repository-local runner that works even
   before installation:

   ```bash
   pgcn-auth --token "<paste-your-flywire-token-here>"
   # or
   ./scripts/pgcn-auth --token "<paste-your-flywire-token-here>"
   ```

   Replace the placeholder with the token string copied from the
   [FlyWire account portal](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html).
   The command will create both `~/.cloudvolume/secrets/cave-secret.json` and
   the FlyWire-preferred
   `~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json` if they do not
   exist. When you keep the token in a file, point the helper at it instead:

   ```bash
   pgcn-auth --token-file /path/to/my_token.txt
   # or
   ./scripts/pgcn-auth --token-file /path/to/my_token.txt
   ```

   Use `--force` whenever you intentionally rotate credentials and need to
   overwrite the existing JSON.

3. **Preflight your FlyWire permissions**

   Run the diagnostic CLI before attempting an expensive cache build. The tool
   checks every supported secret location, validates the token against the
   InfoService endpoint, and enumerates available materialization versions so
   you know the datastack is readable:

   ```bash
   pgcn-access-check --datastack flywire_fafb_production
   # or
   ./scripts/pgcn-access-check --datastack flywire_fafb_production
   ```

   A successful probe prints something akin to:

   ```text
   Datastack: flywire_fafb_production
   Token source: file:/home/<user>/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json
   InfoService: OK (dataset=fafb)
   Materialization: OK (27 versions discovered)
   ```

   Anything returning `HTTP 401` means the secret is malformed or expired—mint
   a fresh token at
   `https://global.daf-apis.com/auth/api/v1/user/create_token` and rerun
   `pgcn-auth`. An `HTTP 403` indicates the token is valid but the associated
   FlyWire account lacks FAFB *view* permission; request access via the official
   [FlyWire setup guide](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html)
   before proceeding. While you wait for approval you can still exercise the
   downstream tooling in offline mode by passing `--use-sample-data` to the
   cache command.

4. **Build the connectome cache**

   ```bash
   pgcn-cache --datastack flywire_fafb_production --mv 783 --out data/cache/
   ```

   > **HTTP 403?** The FlyWire API rejected the token because it lacks `view`
   > permission for the FAFB dataset. Revisit the
   > [FlyWire setup guide](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html),
   > confirm the correct Google/ORCID identity has access, rerun the
   > `pgcn-access-check` diagnostic to verify, and then repeat the cache build.
   > Until approval lands you can keep moving by supplying `--use-sample-data`.

   Use `--use-sample-data` for an offline deterministic cache when a FlyWire
   account or network access is unavailable:

   ```bash
   pgcn-cache --use-sample-data --out data/cache/
   ```

5. **Compute structural metrics**

   ```bash
   pgcn-metrics --cache-dir data/cache/
   ```

6. **Run the unit tests**

   ```bash
   pytest -q
   ```

7. **Validate reservoir weight hydration**

   ```bash
   pytest tests/test_reservoir.py -q
   ```

   The dedicated reservoir tests fabricate a minimal connectome cache and a
   precomputed PN→KC matrix to confirm that weights are normalised, masks are
   preserved, and gradients remain frozen.

## Chemical Odor Generalisation Toolkit

The repository ships an optional chemical modelling stack for reproducing
odor-generalisation analyses. The modules live under `pgcn.chemical` and
`pgcn.models` and are fully importable once PyTorch is available.

1. **Install the project in editable mode (adds PyTorch via the `models` extra)**

   Run this command from the repository root so `pip` can resolve the local
   package:

   ```bash
   pip install -e ".[models]" --find-links https://download.pytorch.org/whl/cpu
   ```

   The project is not published on PyPI; editable installation ensures the
   `pgcn` package is importable from your working tree. The extra pulls in a
   CPU build of PyTorch. Swap the `--find-links` URL for the CUDA-specific index
   if you require GPU acceleration.

2. **Run the chemical unit tests**

   ```bash
   pytest tests/test_chemical.py -q
   ```

3. **Execute the reference modelling snippet**

   ```bash
   python - <<'PY'
   from pgcn.models import ChemicallyInformedDrosophilaModel

   model = ChemicallyInformedDrosophilaModel()
   prediction = model.predict("ethyl_butyrate", "hexanol")
   print(f"Predicted generalisation probability: {prediction:.3f}")
   PY
   ```

   Substitute any other training/testing odor combination that appears in
   `pgcn.chemical.COMPLETE_ODOR_MAPPINGS` to probe cross-generalisation
   behaviour. The model automatically loads the curated chemical descriptors and
   similarity priors included in the repository.

4. **Hydrate the Drosophila reservoir from a connectome cache**

   With the cache generated via `pgcn-cache`, the reservoir will ingest the
   PN→KC weights directly and respect the native sparsity mask:

   ```bash
   python - <<'PY'
   from pathlib import Path

   from pgcn.models.reservoir import DrosophilaReservoir

   reservoir = DrosophilaReservoir(cache_dir=Path("data/cache"))
   print("PN→KC weight shape:", tuple(reservoir.pn_to_kc.weight.shape))
   density = reservoir.pn_kc_mask.float().mean().item()
   print("Mask density:", density)
   print("Gradients frozen:", not reservoir.pn_to_kc.weight.requires_grad)
   PY
   ```

   The reported mask density reflects the Kenyon cell sparsity encoded in the
   cache (≈5 % for hemibrain-derived datasets), and the PN→KC parameters should
   report frozen gradients.

5. **Load a pre-parsed PN→KC matrix**

   When working with precomputed connectivity matrices (for example, custom
   normalisations or ablation studies), feed them directly into the reservoir:

   ```bash
   python - <<'PY'
   import numpy as np

   # Replace this with your own PN→KC weight matrix construction logic.
   # The saved array must have shape (n_kc, n_pn) and non-negative weights.
   matrix = np.random.rand(2000, 50)
   np.save("custom_pn_to_kc.npy", matrix)
   PY

   python - <<'PY'
   import numpy as np

   from pgcn.models.reservoir import DrosophilaReservoir

   matrix = np.load("custom_pn_to_kc.npy")  # shape = (n_kc, n_pn)
   reservoir = DrosophilaReservoir(pn_kc_matrix=matrix)
   weights = reservoir.pn_to_kc.weight.detach().numpy()
   row_sums = weights.sum(axis=1)
   nonzero = row_sums > 0
   print("Weights sum to 1 per KC row:",
         np.allclose(row_sums[nonzero], 1.0))
   print("Sparsity mask preserved:",
         np.array_equal(reservoir.pn_kc_mask.numpy(), (matrix > 0).astype(float)))
   PY
   ```

   The reservoir auto-resolves `n_pn`/`n_kc` dimensions from the matrix and
   keeps absent connections masked without resampling new sparsity patterns.

## Troubleshooting common setup errors

- **`Authentication secret not found at ~/.cloudvolume/secrets/cave-secret.json`** –
  The FlyWire CLI credentials are missing. Run
  `pgcn-auth --token "<paste-your-flywire-token-here>"` to provision the JSON
  file automatically, or rerun `pgcn-cache` with the `--use-sample-data` flag to
  fabricate an offline cache for testing.

- **`FileNotFoundError: 'data/cache/nodes.parquet'` when running `pgcn-metrics` or
  instantiating `DrosophilaReservoir(cache_dir=...)`** – The connectome cache has
  not been generated yet. Execute `pgcn-cache --out data/cache/` (optionally with
  `--use-sample-data`) before invoking downstream commands.

- **`FileNotFoundError: 'custom_pn_to_kc.npy'`** – The matrix file was not saved
  prior to reservoir initialisation. Save the NumPy array with `np.save()` as
  shown above before loading it into `DrosophilaReservoir`.

## Repository Structure

```
PGCN/
├── src/pgcn/                     # Python package
│   ├── __init__.py
│   ├── connectome_pipeline.py     # Cache construction and CLI
│   └── metrics.py                 # Structural metrics and CLI
├── tests/                        # Pytest suite
│   └── test_cache.py
├── data/
│   └── cache/                    # Cache output directory (git-kept, empty)
├── environment.yml               # Conda specification (name=PGCN)
├── pyproject.toml                # Packaging, formatting, and entry points
├── Makefile                      # Common workflows
├── README.md                     # This document
└── data_schema.md                # Cache and metrics schema reference
```

## Cache Outputs

`pgcn-cache` writes the following artefacts into the selected cache directory:

- `nodes.parquet`: node-level metadata with `node_id`, `type`, `glomerulus`,
  `synapse_count`, and centroid coordinates (`x`, `y`, `z`).
- `edges.parquet`: PN→KC and KC→MBON edges with synapse weights.
- `dan_edges.parquet`: DAN→KC and DAN→MBON edges with synapse weights.
- `meta.json`: datastack, materialization version, and table provenance.

All parquet files follow the schemas defined in `data_schema.md`.

## Metrics Outputs

`pgcn-metrics` expects a cache directory and produces:

- `kc_overlap.parquet`: glomerulus-wise Kenyon-cell Jaccard overlaps.
- `pn_kc_mbon_paths.parquet`: PN→KC→MBON two-hop path summary.
- `weighted_centrality.parquet`: weighted in/out degree and betweenness.
- `dan_valence.parquet`: PAM/PPL1/DAN-other valence labels.
- `metrics_meta.json`: row counts for quick validation.

## Authentication Notes

FlyWire access requires a valid CAVE token stored at both
`~/.cloudvolume/secrets/cave-secret.json` (legacy tools) and
`~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json` (preferred by the
FlyWire API). `pgcn-auth` keeps both files in sync, `pgcn-access-check`
preflights permissions, and the cache pipeline exits early with actionable
guidance if either the token is missing, malformed (`HTTP 401`), or the account
lacks FAFB *view* rights (`HTTP 403`). The pipeline still pins materialization
version 783 when available and records the selected version in `meta.json`. When
version 783 is missing the latest available version is selected and the fallback
is logged explicitly.

## Testing Strategy

Unit tests fabricate a deterministic cache to verify schema integrity,
positive weights, and the absence of direct PN→MBON edges. The `--use-sample-data`
flag mirrors this behaviour for developers working offline.
