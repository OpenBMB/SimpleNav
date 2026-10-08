# NavVLA LeRobot v3 Tools

`tool/navvla` provides dataset validation, repair, statistics, compact BATS context, visual-token cache, and dataloader support used directly by the main project's training and evaluation workflows.

Raw-data conversion, trajectory augmentation, AirSim four-view image collection, and enhanced-data conversion are available in this repository's [`data_pipeline/`](../../data_pipeline/README.md).

## Data Preparation

```bash
cd data_pipeline/dataset_conversion
uv run --no-sync vln-convert --help
```

Choose the component for your task:

- [`dataset_conversion`](../../data_pipeline/dataset_conversion/README.md): Convert raw data or enhanced packages with completed image collection to NavVLA LeRobot v3.
- [`trajectory_augmentation`](../../data_pipeline/trajectory_augmentation/README.md): Recover world poses, smooth and resample trajectories, and generate rendering requests.
- [`image_collection`](../../data_pipeline/image_collection/README.md): Collect front, rear, left, and right RGB views in AirSim and publish collection metadata.

After conversion, place or symlink the final split under the main project's `local/data/`, then use the commands below to validate it and build model-side derived artifacts.

## Stable Entry Points

Run from the repository root:

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset ...
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset ...
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.generate_visual_cache ...
```

## Validation

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset \
  <dataset_split_root> \
  --visual-token-mode online_images \
  --smoke-load 8
```

Full checks:

- Required metadata and manifests exist, are readable, and have nonempty required fields.
- Each Parquet shard's schema and metadata row count.
- Counts and references across data, episodes, tasks, video indexes, and context.
- Compact BATS context manifests, metadata, frame arrays, and mask arrays.
- Visual-cache manifests, index schemas, and index row counts.

Deterministic sampling checks:

- State, action, and timestamps in Parquet files.
- Decoded video frames.
- Cache indexes and tensor slices.
- Dataloader samples.

The JSON report records `scope`, `checked`, `total`, sampled indexes, and the seed for each artifact type.

## Repair

Inspect the repair plan first:

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root>
```

Apply the plan after reviewing it:

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root> \
  --apply
```

Specify context budgets:

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root> \
  --token-budget 512 \
  --token-budget 1024 \
  --token-budget 2048 \
  --budget-num-cameras 4 \
  --history-camera-names front left right rear \
  --apply
```

Repair currently supports:

- Missing or incomplete context budgets in the current format.
- Missing `dataset_statistics.json`.
- Memory-mapped visual-cache `index.parquet` files that can be deterministically recovered from checkpoint/rank indexes.

Repair does not write files by default and automatically runs the validator when `--apply` is used. Data Parquet files, video indexes, or semantic fields that cannot be deterministically recovered cause an error.

## Visual-token cache

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.generate_visual_cache \
  <dataset_split_root> \
  --skip-existing \
  --all-token-budgets \
  ...
```

Specify the corresponding profile when validating the cache:

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset \
  <dataset_split_root> \
  --visual-token-mode cached_history_online_current \
  --visual-token-profile <profile_name>
```

## Validating Development Changes

```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest \
  tests/test_navvla_artifact_validation.py \
  tests/test_navvla_repair.py \
  tests/test_navvla_lerobot_context_validation.py \
  tests/test_navvla_visual_cache_cli.py \
  tests/test_navvla_cpm_dataset.py \
  tests/test_navvla_cpm_context_index.py \
  tests/test_navvla_cpm_visual_cache.py \
  -q
```

Data construction and augmentation tests are in `data_pipeline/*/tests/`. The main project's tests cover validation, repair, context, cache, and dataloaders used directly by training. See [Data Structure and State/Action Protocol](../../docs/guides/DATA_STRUCTURE.md) for data structures and semantic boundaries.
