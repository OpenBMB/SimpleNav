from airsim_plugin.camera_pose_noise import sample_camera_pose_records
from airsim_plugin.camera_views import camera_specs, parse_camera_views


def _pose_payload(pose):
    return {
        "x": pose.x, "y": pose.y, "z": pose.z,
        "yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll,
    }


def episode_camera_rows(episode_id, scene_id, seed=1, views="all",
                        render_status="pending", zero_position_delta_views=()):
    selected_views = parse_camera_views(views) if isinstance(views, str) else tuple(views)
    records = episode_camera_records(
        episode_id=episode_id,
        seed=seed,
        views=selected_views,
        zero_position_delta_views=zero_position_delta_views,
    )
    return tuple({
        "episode_id": str(episode_id),
        "scene_id": str(scene_id),
        "view": record.view,
        "camera_name": record.name,
        "seed": int(seed),
        "base_pose": _pose_payload(record.base),
        "pose_delta": _pose_payload(record.delta),
        "final_pose": _pose_payload(record.final),
        "fov_degrees": record.fov_degrees,
        "render_status": str(render_status),
    } for record in records)


def episode_camera_records(episode_id, seed=1, views="all",
                           zero_position_delta_views=()):
    selected_views = parse_camera_views(views) if isinstance(views, str) else tuple(views)
    records = sample_camera_pose_records(
        camera_specs(selected_views), mode="episode", seed=int(seed),
        episode_id=episode_id, fov_min_degrees=90.0, fov_max_degrees=120.0,
    )
    # Kept only to read state produced by earlier collector versions.  A
    # black/constant scene image can be legitimate (for example, inside a
    # building or through a narrow window), so legacy fallback markers must
    # never alter the randomized camera extrinsics.
    del zero_position_delta_views
    return records


def static_camera_metadata(views="all", image_width=224, image_height=224):
    selected_views = parse_camera_views(views) if isinstance(views, str) else tuple(views)
    return {
        "schema_version": "1.0",
        "calibration_status": "episode_randomized",
        "actual_episode_parameters": "navvla_episode_camera_parameters.jsonl",
        "cameras": [
            {
                "view": spec.view,
                "camera_name": spec.name,
                "video_key": "{}_image".format(spec.view),
                "base_pose": {
                    "x": spec.x, "y": spec.y, "z": spec.z,
                    "yaw": spec.yaw, "pitch": spec.pitch, "roll": spec.roll,
                },
                "base_fov_degrees": spec.fov_degrees,
                "width": int(image_width),
                "height": int(image_height),
                "channels": 3,
            }
            for spec in camera_specs(selected_views)
        ],
    }
