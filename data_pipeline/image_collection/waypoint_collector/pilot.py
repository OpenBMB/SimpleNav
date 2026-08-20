import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from airsim_plugin.camera_pose_noise import sample_camera_pose_records
from airsim_plugin.camera_views import camera_specs
from waypoint_collector.airsim_session import AirSimServerRuntime
from waypoint_collector.cameras import episode_camera_records
from waypoint_collector.renderer import validate_rgb_frames
from waypoint_collector.requests import iter_render_requests


class PilotError(RuntimeError):
    pass


def fixed_camera_records(views):
    return sample_camera_pose_records(
        camera_specs(tuple(views)), mode="episode", seed=0,
        episode_id="pilot-fixed", xyz_max=0.0, yaw_pitch_max=0.0,
        roll_max=0.0, fov_min_degrees=90.0, fov_max_degrees=90.0,
    )


def _save_contact_sheet(samples, output_path, image_width=224, image_height=224):
    if not samples:
        raise PilotError("pilot produced no samples")
    cell_width, cell_height = int(image_width), int(image_height) + 26
    columns = 4
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, frame) in enumerate(samples):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(Image.fromarray(frame), (x, y))
        draw.text((x + 4, y + int(image_height) + 4), label[:34], fill="black")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run_pilot(state, request_path, repository_root, env_cache_root,
              output_root, views, gpu, base_control_port,
              channel_order="rgb",
              camera_seed=1, skipped_scenes=("1",), image_width=224,
              image_height=224):
    jobs = []
    seen_scenes = set()
    for job in state.iter_jobs(
        include_skipped=False, skipped_scenes=skipped_scenes
    ):
        if job.scene_id not in seen_scenes:
            jobs.append(job)
            seen_scenes.add(job.scene_id)
        if len(jobs) == 2:
            break
    if len(jobs) < 2:
        raise PilotError("pilot requires two available episodes from different scenes")
    output_root = Path(output_root)
    samples = []
    pose_checks = []
    records = fixed_camera_records(views)
    for pilot_index, job in enumerate(jobs):
        requests = tuple(iter_render_requests(
            request_path, byte_start=job.byte_start, byte_end=job.byte_end,
            start_index=job.start_index,
            expected_width=image_width, expected_height=image_height,
        ))
        selected = [requests[0], requests[len(requests) // 2], requests[-1]]
        runtime = AirSimServerRuntime(
            repository_root, env_cache_root, gpu,
            base_control_port + pilot_index * 100,
            output_root / "logs/pilot-{}.log".format(pilot_index),
            image_width=image_width, image_height=image_height,
        )
        session = runtime.open_scene(job.scene_id, channel_order="rgb")
        try:
            session.apply_camera_records(records)
            for request in selected:
                session.set_vehicle_pose(request)
                position_error, rotation_error = session.verify_vehicle_pose(request)
                if position_error > 0.01 or rotation_error > 0.1:
                    raise PilotError(
                        "pose readback failed for {} waypoint {}: position={}, rotation={}".format(
                            job.episode_id, request.waypoint_index,
                            position_error, rotation_error,
                        )
                    )
                frames = session.capture_rgb(views)
                validate_rgb_frames(
                    frames, views, image_width=image_width,
                    image_height=image_height,
                )
                pose_checks.append({
                    "episode_id": job.episode_id,
                    "waypoint_index": request.waypoint_index,
                    "position_error": position_error,
                    "rotation_error_degrees": rotation_error,
                })
                for view in views:
                    samples.append((
                        "{} wp{} {}".format(job.episode_id[:12], request.waypoint_index, view),
                        frames[view],
                    ))
        finally:
            session.close()
            runtime.close()
    randomized = [
        episode_camera_records(job.episode_id, seed=camera_seed, views=views)
        for job in jobs
    ]
    if randomized[0] == randomized[1]:
        raise PilotError("different pilot episodes received identical camera parameters")
    if randomized[0] != episode_camera_records(
        jobs[0].episode_id, seed=camera_seed, views=views
    ):
        raise PilotError("episode camera parameters are not reproducible")
    display_samples = (
        [(label, frame[:, :, ::-1]) for label, frame in samples]
        if channel_order == "bgr"
        else samples
    )
    _save_contact_sheet(
        display_samples, output_root / "contact_sheet.png",
        image_width=image_width, image_height=image_height,
    )
    payload = {
        "channel_order": channel_order,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "episodes": [job.episode_id for job in jobs],
        "scene_ids": [job.scene_id for job in jobs],
        "pose_checks": pose_checks,
        "randomized_camera_parameters_differ": True,
        "randomized_camera_parameters_reproducible": True,
    }
    result_path = output_root / "pilot_result.json"
    temporary = result_path.with_name(".pilot_result.json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(result_path))
    return payload
