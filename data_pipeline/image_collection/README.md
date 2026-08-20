# AirSim Waypoint-Based Four-View RGB Collection

[Chinese](README_ZH.md)

This component reads `render/render_requests.jsonl` from a trajectory package, places the drone at each absolute waypoint, and captures front, back, left, and right RGB views in one step. It is independent of AirVLN training, models, tokenizers, LMDB, and TF/DAgger workflows.

The component publishes videos and collection metadata into the trajectory package. It does not generate observation/action Parquet files, so the result is not yet a complete trainable LeRobot split.

## 1. Installation and System Requirements

Requirements:

- Linux, NVIDIA GPUs, and an available hardware Vulkan device;
- GLX/EGL/Vulkan user-space libraries compatible with the host driver;
- AirSim scene directories or ZIP archives;
- `ffmpeg`, `ffprobe`, and `vulkaninfo`;
- sufficient space for scene caches and encoded videos.

Confirm that the required system tools are available:

```bash
ffmpeg -version
ffprobe -version
vulkaninfo --summary
```

AirSim RGB rendering cannot run with CUDA or `nvidia-smi` alone when no hardware Vulkan graphics device is available.

The collector uses Python 3.10 with AirSim 1.8.1, msgpack-rpc-python 0.4.1, NumPy 1.24.4, PyArrow 12.0.1, and Pillow 10.0.0. From the repository root:

```bash
cd image_collection
conda env create -f environment.yml
conda activate vln-image-collection
vln-collect --help
```

To install in an existing Python 3.10 environment:

```bash
python -m pip install .
python -m waypoint_collector --help
```

If the host requires a separate graphics-library activation script, specify it when using the generic launcher:

```bash
export VLN_GRAPHICS_ACTIVATE=/path/to/nvidia-graphics/activate.sh
./scripts/collect_waypoints.sh --help
```

## 2. Input Data Location and Format

The collector uses three external paths. None of them needs to be inside this repository:

| Argument | Purpose |
| --- | --- |
| `--package-dir` | Trajectory package produced by `vln-augment` |
| `--env-archive-root` | Root containing AirSim scene directories or ZIP files |
| `--env-cache-root` | Writable root for prepared scenes and default resume state |

### Trajectory package

`--package-dir` must point to `vln_train_enhanced` and contain at least:

```text
/path/to/Dataset_lerobot/vln_train_enhanced/
├── manifest.json
├── trajectories/episodes.jsonl
└── render/render_requests.jsonl
```

Each request must contain contiguous episode-local `image_index` values, strictly increasing `waypoint_index` values, a scene identifier, absolute `position_xyz`, a unit `orientation_quaternion_wxyz`, and expected image dimensions.

The dimensions in the package manifest and render requests must match `--image-width` and `--image-height`. For native 448x448 collection, both stages must use 448x448. The collector does not upscale 224x224 images after capture.

### AirSim scenes

`--env-archive-root` accepts extracted scene directories and AerialVLN scene ZIP archives:

```text
/path/to/AirSim_scenes/
├── env_1/
│   └── LinuxNoEditor/
│       └── AirVLN.sh
├── env_2.zip
├── env_airsim_16/
│   └── LinuxNoEditor/
│       └── start.sh
```

For an AerialVLN scene, `LinuxNoEditor/AirVLN.sh` and the packaged executable/assets must be present; numeric AerialVLN scenes may be supplied as ZIP archives. A custom/OpenFly scene uses an extracted directory with `LinuxNoEditor/start.sh` and one runtime `settings.json` under its packaged binaries. Scene names must match render-request `scene_id` values, using forms such as `env_1` or `env_airsim_16`.

Prepared scenes are extracted or linked under `--env-cache-root`. Do not point the cache root at the source package or Git repository.

## 3. Complete Collection Workflow

The stages are:

```text
preflight -> prepare-envs -> pilot -> render -> assemble -> validate -> publish
```

| Stage | Purpose |
| --- | --- |
| `preflight` | Check the package, scenes, ports, graphics runtime, disk space, and task index |
| `prepare-envs` | Inspect and atomically prepare scene caches |
| `pilot` | Capture previews at a small set of waypoints |
| `render` | Collect temporary episode videos with multiple workers |
| `assemble` | Merge videos into the LeRobot chunk/part layout and write metadata |
| `validate` | Check collection artifacts for completeness |
| `publish` | Publish `videos/` and `meta/` into the package and update the manifest |
| `run` | Execute all stages in order |

Use the same `--run-id` and common arguments for every stage.

### Step 1: Preflight

Start with one GPU:

```bash
vln-collect preflight \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 \
  --workers 1 \
  --skip-scene 1 \
  --image-width 224 \
  --image-height 224 \
  --run-id waypoint-v1
```

### Step 2: Prepare scenes

```bash
vln-collect prepare-envs \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --skip-scene 1 \
  --image-width 224 --image-height 224 \
  --run-id waypoint-v1 --resume
```

### Step 3: Run and inspect the pilot

```bash
vln-collect pilot \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --skip-scene 1 \
  --image-width 224 --image-height 224 \
  --run-id waypoint-v1 --resume
```

Inspect:

```text
<package-dir>/.render_staging/<run-id>/pilot/contact_sheet.png
<package-dir>/.render_staging/<run-id>/pilot/pilot_result.json
```

Confirm that the drone is in the correct scene, above the ground, and facing a plausible direction, and that all four views and color channels are correct.

### Step 4: Resume the complete run

After approving the pilot, keep the same `run-id` and collection configuration. `--resume` reuses the completed stages:

```bash
vln-collect run \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 \
  --workers 1 \
  --skip-scene 1 \
  --views front,back,left,right \
  --image-width 224 \
  --image-height 224 \
  --run-id waypoint-v1 \
  --resume
```

The generic launcher accepts the same arguments:

```bash
./scripts/collect_waypoints.sh run --package-dir ... --env-archive-root ... --env-cache-root ...
```

It only activates an optional graphics runtime and forwards the provided arguments.

For multi-GPU collection, choose the final `--gpus` and `--workers` values before `preflight` and keep them unchanged for all stages using that `run-id`.

## 4. Output Location and Format

Temporary output is written inside the package staging area:

```text
<package-dir>/.render_staging/<run-id>/
├── pilot/
├── rendered_episodes/
├── final/
├── logs/
└── reports/
```

The default resume-state database is outside the package:

```text
<env-cache-root>/.waypoint_collector_state/<run-id>/state.sqlite3
```

After `validate` succeeds, `publish` adds the following final artifacts to the same trajectory package:

```text
<package-dir>/
├── videos/
│   ├── front_image/chunk-NNN/part-NNN.mp4
│   ├── back_image/chunk-NNN/part-NNN.mp4
│   ├── left_image/chunk-NNN/part-NNN.mp4
│   └── right_image/chunk-NNN/part-NNN.mp4
└── meta/
    ├── navvla_video_index.parquet
    ├── navvla_multiview_frame_metadata.jsonl
    ├── navvla_episode_camera_parameters.jsonl
    └── navvla_cameras.json
```

The video layout uses the source episode index: 20 episodes per part and 50 parts per chunk. Existing formal `videos/` are never overwritten.

## 5. Multi-GPU Execution, Resume, and Failure Handling

- `--workers` may not exceed the number of GPUs in `--gpus`. Each worker uses a separate control port.
- If custom UE4 scenes cannot share runtime settings, use `--worker-env-cache-roots` to assign an independent prepared scene root to every worker.
- `--resume` reuses the SQLite state and collected episode videos for the same `run-id`, then requeues interrupted or failed tasks.
- Use a new `--run-id` for an independent collection run. Do not reuse another run's state.
- A frame is retried according to `--frame-attempts`; failed episodes are retried according to `--failed-episode-retry-rounds`. Exhausted failures block assembly and publishing.
- Episodes selected by `--skip-scene` remain in the index as unavailable, and no black frames are written.

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| `vulkaninfo` lists only llvmpipe/CPU | NVIDIA Vulkan user-space libraries and ICD configuration |
| Control port is occupied | Change `--base-control-port` or stop the previous worker |
| Scene cannot be opened | Scene naming, ZIP/directory structure, permissions, cache integrity, and GPU graphics environment |
| Returned image dimensions do not match | Profile/manifest/request dimensions and collector `--image-width`/`--image-height` |
| Image colors are incorrect | Use `--channel-order rgb`, or explicitly select `bgr` when required |
| Pose is underground or mirrored | Return to trajectory augmentation and check world coordinates and `render_transform` |
| Disk preflight fails | Change the output location or set `--estimated-output-gib` from representative encoding results |
