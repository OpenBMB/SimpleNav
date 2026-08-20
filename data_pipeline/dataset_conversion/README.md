# VLN Dataset Conversion

[Chinese](README_ZH.md)

This component converts heterogeneous navigation datasets to the common NavVLA
LeRobot v3 format. It is the first stage of SimpleNAV's `data_pipeline`, and it is also
used after trajectory augmentation and image collection to turn a completed
`vln_train_enhanced` package into a trainable enhanced LeRobot split.

```text
raw dataset
  -> dataset_conversion
  -> NavVLA LeRobot v3 vln_train
  -> trajectory_augmentation
  -> image_collection
  -> dataset_conversion --adapter enhanced_vln
  -> complete enhanced LeRobot split
```

Source datasets are treated as read-only inputs. Existing targets are rejected
unless `--overwrite` is explicitly provided; the `enhanced_vln` adapter never
overwrites a target.

## Installation

Python 3.10, FFmpeg, and FFprobe are required:

```bash
cd dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
```

Or install into an existing Python 3.10 environment:

```bash
python -m pip install -e .
```

VLN-CE RGB rendering requires a separately prepared compatible Habitat/VLN-CE
environment. AerialVLN LMDB image decoding uses `lmdb`, `msgpack`, and
`msgpack-numpy`, which are declared dependencies.

## Common output contract

Each converted split contains data and episode Parquet shards, H.264 videos,
task/camera/frame metadata, a video index, dataset statistics, a conversion
report, and compact BATS context artifacts. `observation.state` always has four
values `[x, y, z, yaw]`, but its coordinate semantics are explicitly declared
by `navvla.state_mode` in `meta/info.json`. Actions are future body-frame
`[dx, dy, dz, dyaw]` waypoint chunks anchored at the current frame.

The standalone converter does not generate visual-token caches. A four-value
state must never be assumed to be a canonical world pose. Before trajectory
augmentation, verify episode/trajectory/scene identity, exact lengths, finite
poses, axes, z convention, and yaw units.

## Unified conversion

```bash
vln-convert \
  --adapter aerialvln \
  --source-root /path/to/raw/AerialVLN \
  --output-root /path/to/AerialVLN_lerobot \
  --dataset-name vln_train \
  --split train \
  --write-workers 8 \
  --validate
```

The equivalent module entry point is:

```bash
python -m navvla_conversion.cli.convert_dataset --help
```

`--cache-workers` remains an alias for `--write-workers`. `--repair-existing`
only reuses episode shards that are provably complete; ambiguous data or
semantic fields fail closed.

### Supported datasets

| Adapter | Image FPS | Control rate | Main contents under `--source-root` |
| --- | ---: | ---: | --- |
| `traveluav` | 0.2 | 1 Hz | TravelUAV root, one episode directory, or a compatible JSON layout |
| `uav_flow` | 5 | 5 Hz | UAV-Flow Real/Sim root; family roots use `--source-root-is-family-root` |
| `aerialvln` | 1 | 1 Hz | AerialVLN annotations and RGB LMDB roots |
| `vlnce_rendered` | 1 | 1 Hz | RGB manifest root produced by `vln-render-vlnce` |
| `flight` | 1 | 2 Hz | FLIGHT annotations, videos, and trajectories |
| `indooruav` | 10 | 10 Hz | IndoorUAV dataset or extracted root |
| `huge` | 5 | 5 Hz | HUGE dataset root |
| `embodiednav` | 1 | 1 Hz | EmbodiedNav dataset root |
| `openfly` | 5 | 5 Hz | OpenFly images, Annotation, and trajectory roots |
| `openscene` | 2 | 2 Hz | OpenScene dataset root |
| `nuscenes` | 2 | 2 Hz | nuScenes dataroot |
| `enhanced_vln` | 1 | 1 Hz | Image-complete `vln_train_enhanced` package |

Run `vln-convert --help` for dataset-specific options such as `--variant`,
`--media-cache-root`, `--annotation-root`, `--traj-root`, and
`--dataset-version`.

### Convert a completed enhanced package

```bash
vln-convert \
  --adapter enhanced_vln \
  --source-root /path/to/Dataset_lerobot/vln_train_enhanced \
  --output-root /path/to/enhanced_lerobot/Dataset \
  --dataset-name vln_train_enhanced_lerobot \
  --split train \
  --load-workers 8 \
  --validate
```

The enhanced manifest, trajectories, four-view videos, waypoint metadata, and
episode camera parameters must agree. OpenFly rendered poses are already AirSim
NED and must not receive another z reflection.

## CosFly

CosFly preserves paired ori/aug samples through a deterministic two-step flow:

```bash
vln-cosfly prepare-manifest \
  --source-root /path/to/CosFly \
  --output /path/to/cosfly_splits.json \
  --seed 42

vln-cosfly convert \
  --source-root /path/to/CosFly \
  --output-root /path/to/CosFly_lerobot \
  --split-manifest /path/to/cosfly_splits.json \
  --split train \
  --write-workers 8 \
  --validate
```

Metadata finalization is part of conversion; no separate repair command is exposed.

## VLN-CE RGB rendering

```bash
vln-render-vlnce \
  --vlnce-root /path/to/VLN-CE \
  --output-root /path/to/vlnce_rendered_rgb \
  --family r2r \
  --split train \
  --gpu-id 0
```

Both paths are required. Feed the result to `vln-convert --adapter vlnce_rendered`.

## Validation

```bash
vln-validate /path/to/Dataset_lerobot/vln_train \
  --all-token-budgets \
  --check-media-decode sampled
```

The validator checks the full metadata and Parquet inventories, row counts,
episode/task/index relationships, statistics, compact BATS arrays, and video
indexes. Row payloads and video decoding are sampled deterministically. Its
`world_pose_assumed` field is always false because format validation alone does
not prove canonical world-pose semantics.

## Common failures

- `output exists`: choose a new destination, or use `--overwrite` only after
  confirming that the existing destination may be removed.
- `scene mismatch` or `length mismatch`: fix the source episode mapping instead
  of pairing records by their incidental order.
- Mirrored paths or reversed turns: verify y/yaw signs, yaw units, and coordinate
  handedness.
- Trajectories below ground: verify z-up versus z-down and keep training-alignment
  transforms separate from rendering transforms.
- FFmpeg errors: confirm `ffmpeg` and `ffprobe` are on `PATH`, then inspect the
  source images or videos for missing/corrupt media.
