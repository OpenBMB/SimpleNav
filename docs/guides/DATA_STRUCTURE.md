# Data Structure and State/Action Protocol

[Main README](../../README.md) · [中文](DATA_STRUCTURE_ZH.md)

## LeRobot v3 split

```text
<root>/
├── data/chunk-*/part-*.parquet
├── meta/info.json
├── meta/episodes/chunk-*/part-*.parquet
├── videos/<camera>/chunk-*/part-*.mp4
├── dataset_statistics.json
├── meta/navvla_frame_metadata.jsonl
├── meta/navvla_video_index.parquet
└── meta/navvla_context_*.{npy,npz,json}
```

`meta/info.json` declares feature shapes, camera keys, dataset names, and the `navvla` protocol block. Episode metadata maps each episode to scene, task, length, and data shards. Frame rows contain episode/frame indexes, timestamps, `observation.state`, `action`, and media references.

## Three different tensors

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Stored `observation.state` | `[4]` | One pose `[x, y, z, yaw]` in the adapter-declared storage frame. |
| Model state | `[K, 4]` when enabled | Consecutive relative motions over the BATS-selected history, ending at the current frame. `K` varies with selected history. |
| Action target | `[H, 4]` | Future body-frame waypoint chunk predicted from the current frame. The primary horizon is `H=8`. |

Do not feed stored `observation.state` directly as the model state unless a separate model/config explicitly defines that behavior.

## Primary action contract

The primary training action mode is `anchor_relative_body_frame_xyz_yaw`.

For anchor pose `(p0, yaw0)` and each future world pose `(pi, yawi)`:

1. compute `pi - p0` in the world frame;
2. rotate it once into the anchor body frame;
3. apply the declared axis convention to obtain forward/right/down components;
4. compute wrapped yaw difference relative to `yaw0`.

Every future waypoint is relative to the same current anchor pose. It is not relative to the previous future waypoint.

```text
action[t, h] = [dx_forward, dy_right, dz_down, dyaw]
               for future offset h, anchored at pose t
```

Benchmark configs may declare another action mode or horizon. The benchmark adapter must convert between its runtime motion contract and the model/checkpoint contract explicitly.

## Model state contract

When `include_state: true`, the main dataloader:

1. obtains the poses selected by BATS;
2. appends the current pose;
3. computes consecutive relative body-frame motions;
4. normalizes that variable-length motion history;
5. passes it as `variable_bats_history_relative_body_frame_actions`.

This history state is neither an absolute pose sequence nor a copy of the future action target. Most current Qwen3.5 configs use `include_state: false` and `state_dim: 0`.

## Coordinate metadata

Each adapter must state:

- source coordinate handedness and axis order;
- world z direction;
- yaw units, zero direction, and positive direction;
- stored-state frame and origin;
- action frame, axis order, anchor, and horizon;
- camera order, orientation, resolution, and timestamp relation;
- source and target sampling rates.

A four-value pose does not prove that it is a simulator world pose. Trajectory augmentation must use a verified absolute-world-pose source declared by its dataset profile.

## Normalization

`dataset_statistics.json` is authoritative for training and inference.

- Use per-dimension action `q01` and `q99` for all four action dimensions.
- Normalize model state with the matching state statistics when state is enabled.
- Apply padding after normalization so padded action rows are exact zero.
- Unnormalize predictions with the same statistics key before execution.
- Save `dataset_statistics.json` next to the checkpoint.

Do not silently recompute or substitute statistics during evaluation.

## Derived artifacts

| Artifact | Purpose |
| --- | --- |
| `dataset_statistics.json` | State/action normalization and protocol metadata |
| BATS context files | Candidate and selected history indexes |
| Visual-token cache | Encoder-specific history tokens and cache metadata |
| Video index | Frame-to-camera/video/timestamp lookup |
| Conversion/validation report | Counts, source mapping, schema, and media checks |

Cache metadata must match the encoder checkpoint, preprocessing profile, token stage, image size, camera layout, and source dataset revision.

## Adapter acceptance checks

Before training or evaluation:

1. verify episode, scene, trajectory, and task identity;
2. verify exact frame and pose lengths;
3. inspect representative axes, z, yaw, and turn direction;
4. inspect camera order and timestamps;
5. check action anchoring against known poses;
6. validate all metadata and sampled media;
7. smoke-load dataloader samples and inspect state/action shapes;
8. confirm checkpoint and evaluation use the same statistics key.
