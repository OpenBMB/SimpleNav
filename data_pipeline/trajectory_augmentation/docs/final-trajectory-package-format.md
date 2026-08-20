# Final Enhanced Trajectory Package Format

The package is a trajectory-and-render-request interchange format. It is not a
complete LeRobot split until simulator images are collected and packaged.

## Files

- `trajectories/train.json`: AerialVLN-compatible top-level `episodes` object.
- `trajectories/episodes.jsonl`: the same episodes, one JSON object per line for streaming.
- `trajectories/augmentation_metadata.jsonl`: source identity, actual and cruise speed,
  waypoint count, sampling policy, actual image collection indices, and gaps.
- `render/render_requests.jsonl`: direct simulator image requests. Each request contains the
  scene, absolute position, WXYZ quaternion, XYZ/RPY pose, camera, image size, and output path.
- `validation/summary.json`: package-level validation.
- `validation/failures.jsonl`: trajectories rejected during conversion.
- `validation/samples/`: only the small representative sample selected for human inspection.

## Motion and image assignment

- Controls remain absolute `xyz+yaw` poses sampled at 1 Hz.
- Straight sections use a deterministic cruise speed near 1 m/s.
- Turn curvature lowers local speed and waypoint spacing, with a default minimum factor of 0.55.
- Random-interval packages draw each complete gap independently from the
  package-declared stride choices using a stable seed derived from dataset key,
  episode ID, and base seed.
- Every render request records the explicit `waypoint_index`, sampling policy,
  and actual gap from the previous requested waypoint.
- Every render request uses the package-declared common RGB shape. The default is
  `224x224x3`; positive even dimensions such as `448x448x3` are supported. The
  source image feature shape is preserved in `camera_metadata.source_feature_shape`.
- The real terminal waypoint is always requested even when it does not align with the stride.

## Direct rendering loop

```python
from pathlib import Path
from vln_aug.trajectory_package import iter_render_requests

package = Path("OpenFly_lerobot/vln_train_enhanced")
for request in iter_render_requests(package):
    scene_id = request["scene_id"]
    x, y, z = request["position_xyz"]
    w, qx, qy, qz = request["orientation_quaternion_wxyz"]
    camera = request["camera_key"]
    output = package / request["expected_image_relpath"]
    # renderer.set_scene(scene_id)
    # renderer.set_absolute_pose(position=(x, y, z), quaternion_wxyz=(w, qx, qy, qz))
    # renderer.capture(camera, output)
```

AirSim has no universal trajectory-file standard. The renderer adapter must map the request's
declared dataset coordinate convention to the loaded simulator scene, then set the absolute pose.
No discrete AirVLN action conversion is permitted for these enhanced trajectories.
