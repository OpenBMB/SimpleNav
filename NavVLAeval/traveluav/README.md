# TravelUAV Eval Notes

TravelUAV follows the shared `EvalEpisode` contract. Input adapters create episode payloads; `TravelUAVBenchmarkSpec` validates the payload and creates the runtime used by workers.

## Inputs

Supported input adapters:

- `TravelUAVJsonInputAdapter` for explicit eval JSON records.
- `TravelUAVLeRobotV3InputAdapter` for NavVLA LeRobot v3 roots such as
  `local/data/TravelUAV/navvla_lerobot_full/vln_val_seen` and
  `local/data/TravelUAV/navvla_lerobot_full/vln_val_unseen`.

Every input source must provide an explicit namespace. Episode ids are `"{namespace}:{source_episode_id}"`.

## Collision policy

The portable configuration uses depth-only collision stopping:

```yaml
depth_collision_policy: stop
ignore_movement_collision: true
```

AirSim movement collisions remain available in step diagnostics but do not terminate the episode. The TravelUAV depth-collision rule determines collision termination.

## Step Artifacts

The common worker owns the episode directory:

```text
logs/<scene_id>/<input_namespace>/<source_episode_id>/
  eval_info.json
  frontcamera/000000.png
  downcamera/000000.png
  data/000000.json
```

TravelUAV-specific step files are written through `EpisodeArtifactWriter`. TravelUAV logging does not choose or sanitize episode directory names.

`eval_info.json` stores episode-level fields such as `scene_id`, instruction, backend env name, step count, SR/OSR/NE/SPL, termination reason, failure details, and artifact paths. `data/*.json` stores step-local pose, waypoint, distance, and AirSim diagnostics.
