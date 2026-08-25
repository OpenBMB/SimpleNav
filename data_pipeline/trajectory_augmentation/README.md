# VLN Trajectory Augmentation

[Chinese](README_ZH.md)

This component scans a LeRobot-format `vln_train` split without modifying it, recovers canonical absolute world poses, smooths and resamples trajectories at 1 Hz, and generates a trajectory-only package with direct render requests for an external simulator.

It does not launch AirSim or re-encode collected images into a complete LeRobot split. See the [image collection guide](../image_collection/README.md) for the rendering stage.

## 1. Installation

Python 3.10 is required. From the repository root, create the dedicated Conda environment:

```bash
cd trajectory_augmentation
conda env create -f environment.yml
conda activate vln-trajectory-augmentation
vln-augment --help
```

To install in an existing Python 3.10 environment:

```bash
python -m pip install .
python -m vln_aug.cli --help
```

## 2. Input Data Location and Format

The dataset is external to this repository. Store it anywhere and pass its root through `--dataset-root`.

```text
/path/to/Dataset_lerobot/
├── vln_train/
│   ├── meta/
│   │   ├── info.json
│   │   └── episodes/chunk-XXX/part-XXX.parquet
│   └── data/chunk-XXX/part-XXX.parquet
└── <world-pose annotation path referenced by the profile>
```

The source split must provide:

- `meta/info.json`, including the data-path template;
- episode metadata columns `episode_index`, `episode_id`, `scene_id`, `length`, `data/chunk_index`, and `data/file_index`;
- data columns `episode_index`, contiguous `frame_index`, and `observation.state`;
- one canonical absolute world pose for every source frame.

Supported world-pose sources include a profile adapter, `vln_train/meta/navvla_frame_metadata.jsonl`, or an `observation.state` explicitly declared as absolute in `meta/info.json`. An unknown `observation.state` may be episode-local and must not be treated as a simulator pose.

Built-in layouts:

| Dataset | Required content under `--dataset-root` | Profile |
| --- | --- | --- |
| AerialVLN | `vln_train/` and `aerialvln_json/train.json` | `profiles/aerialvln.json` |
| OpenFly | `vln_train/` and `openfly_env/Annotation/train.json` | `profiles/openfly.json` |
| Custom | `vln_train/` and the annotation path declared by a custom profile | `profiles/new-dataset-template.json` |

The AerialVLN profile reads the original `reference_path` with identity alignment and render transforms. The OpenFly profile reads annotation `pos`/`yaw`, uses `reflect-y-yaw` for local alignment, and uses `reflect-y-z-yaw` for rendering.

For a custom annotation schema, implement a pose adapter that returns a finite array with shape `[episode_length, 4]` and columns `[x, y, z, yaw_radians]`.

## 3. Profile Configuration

All profile paths are relative to `--dataset-root` and may not escape that root. The relevant fields are:

- `paths.train_split`: source LeRobot split, normally `vln_train`;
- `paths.output_dir`: new output directory, normally `vln_train_enhanced`;
- `world_pose`: pose source, adapter, annotation path, and coordinate transforms;
- `sampling`: image-waypoint stride policy;
- `render`: requested image width and height;
- `selection`: optional episode and scene filtering;
- `trajectory`: smoothing and 1 Hz retiming parameters.

To add another dataset:

```bash
cp profiles/new-dataset-template.json /path/to/my-dataset-profile.json
```

Fill in the annotation path, pose adapter, coordinate transforms, scene selection, trajectory parameters, and sampling policy. For the first run, use `selection.include_episode_indices` to select a small set of episodes covering every scene. `sample_episode_indices` controls sample plots only and does not limit the formal export.

## 4. Complete Augmentation Workflow

### Validate the profile

```bash
vln-augment validate-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot
```

### Run a dry export

```bash
vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot \
  --dry-run
```

### Export the package

After checking representative episodes and rendered samples, run:

```bash
vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot
```

### Check the generated package

```bash
vln-augment validate-trajectory-package \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced
```

The exporter keeps the source `vln_train` read-only and refuses to overwrite existing targets.

## 5. Output Location and Format

For `/path/to/Dataset_lerobot/vln_train`, the formal output is `/path/to/Dataset_lerobot/vln_train_enhanced`:

```text
Dataset_lerobot/
├── vln_train/                         # read-only source
└── vln_train_enhanced/                # generated package
    ├── manifest.json
    ├── validation.json
    ├── trajectory_metrics.jsonl
    ├── trajectories/
    │   ├── train.json
    │   ├── episodes.jsonl
    │   └── augmentation_metadata.jsonl
    ├── render/
    │   └── render_requests.jsonl
    └── validation/
        ├── summary.json
        ├── failures.jsonl
        └── samples/
```

The output directory must not exist before export. The package contains trajectories and simulator requests, not videos or a complete trainable LeRobot split. See [`docs/final-trajectory-package-format.md`](docs/final-trajectory-package-format.md) for field definitions.

Image waypoints always include the start and the true terminal waypoint. The control frequency is fixed at 1 Hz, and no synthetic terminal hover is added.

Render requests use `224x224x3` by default. A uniform positive even resolution is also supported:

```json
{
  "render": {
    "image_width": 448,
    "image_height": 448
  }
}
```

The image collector must use exactly the same width and height.

## 6. Troubleshooting

| Symptom | Check first |
| --- | --- |
| Start pose is outside the scene | Scene mapping, world origin, and annotation mapping |
| Most of the trajectory is underground | Z-axis convention and `render_transform` |
| Turns are mirrored | Y/yaw signs and the alignment transform |
| Path is correct but heading is wrong | Yaw units, rotation order, and quaternion convention |
| Profile passes but an episode fails | Adapter mapping, path length, or local/world alignment |
| Output is rejected immediately | Existing output, output inside the source split, or an invalid sibling directory |
