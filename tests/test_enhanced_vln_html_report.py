# ruff: noqa: RUF001

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_render_episode_html_is_offline_and_contains_trajectory_coordinate_and_four_views() -> None:
    from tool.navvla.enhanced_vln_report import render_episode_html

    payload = {
        "dataset_label": "AerialVLN",
        "episode_id": "episode-0",
        "scene_id": "scene-5",
        "instruction": "fly forward",
        "dense_states": [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, -0.25, 0.1],
            [2.0, 1.0, -0.5, 0.2],
        ],
        "sampled_waypoint_indices": [0, 2],
        "image_interval_seconds": 5.0,
        "video_fps": 1.0,
        "camera_parameters": {
            view: {
                "pose": [0.1, 0.0, 0.0, yaw, 0.0, 0.0, 1.7],
                "pose_degrees": [0.1, 0.0, 0.0, yaw * 57.2958, 0.0, 0.0, 97.4],
            }
            for view, yaw in {"front": 0.0, "back": 3.14, "left": -1.57, "right": 1.57}.items()
        },
        "summary": {
            "dense_waypoints": 3,
            "video_frames": 2,
            "trajectory_duration_seconds": 2.0,
        },
        "transform_note": "identity",
    }
    video_sources = {view: f"videos/{view}.mp4" for view in ("front", "back", "left", "right")}

    html = render_episode_html(payload, video_sources=video_sources)

    assert "X：前为正" in html
    assert "Y：右为正" in html
    assert "Z：下为正" in html
    assert 'id="trajectory-top"' in html
    assert 'id="trajectory-side"' in html
    assert 'id="trajectory-iso"' in html
    assert "const trajectory = [[0.0, 0.0, 0.0, 0.0]" in html
    for view in ("front", "back", "left", "right"):
        assert f'data-view="{view}"' in html
        assert f'src="videos/{view}.mp4"' in html
    assert "const imageIntervalSeconds = 5.0" in html
    assert "https://" not in html
    assert "http://" not in html


def test_load_episode_report_uses_dense_source_trajectory_and_video_frame_span(tmp_path) -> None:
    from tool.navvla.enhanced_vln_report import load_episode_report

    source = tmp_path / "OpenFly_lerobot" / "vln_train_enhanced"
    (source / "trajectories").mkdir(parents=True)
    trajectory = {
        "episode_id": "000000__enhanced_v1",
        "trajectory_id": "trajectory-0",
        "scene_id": "env_airsim_18",
        "reference_path": [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 9.0, 0.0, 0.0, 0.1],
            [2.0, 2.0, 8.0, 0.0, 0.0, 0.2],
        ],
    }
    (source / "trajectories" / "episodes.jsonl").write_text(json.dumps(trajectory) + "\n")

    root = tmp_path / "converted"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    for view in ("front", "back", "left", "right"):
        (root / "videos" / f"{view}_image" / "chunk-000").mkdir(parents=True)

    columns = {
        "episode_index": [0, 0],
        "timestamp": [0.0, 5.0],
        "task_index": [0, 0],
        "source_frame_index": [0, 2],
        "index": [0, 1],
    }
    for view, yaw in {"front": 0.0, "back": 3.14, "left": -1.57, "right": 1.57}.items():
        columns[f"observation.camera_pose.{view}"] = pa.array(
            [[0.1, 0.0, 0.0, yaw, 0.0, 0.0, 1.7]] * 2,
            type=pa.list_(pa.float32(), list_size=7),
        )
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "part-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["fly to the building"]}), root / "meta" / "tasks.parquet")
    video_rows = []
    for index in (0, 1):
        for view in ("front", "back", "left", "right"):
            video_rows.append(
                {
                    "index": index,
                    "video_key": f"{view}_image",
                    "available": True,
                    "video_frame_index": index,
                    "chunk_index": 0,
                    "file_index": 0,
                }
            )
    pq.write_table(pa.Table.from_pylist(video_rows), root / "meta" / "navvla_video_index.parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 1.0,
                "video_path": {
                    f"{view}_image": f"videos/{view}_image/chunk-{{chunk_index:03d}}/part-{{file_index:03d}}.mp4"
                    for view in ("front", "back", "left", "right")
                },
            }
        )
    )
    frame_meta = {
        "index": 0,
        "source_frame_index": 0,
        "source_metadata": {
            "source_dataset": "OpenFly_lerobot",
            "source_package": str(source),
            "source_episode_id": "000000__enhanced_v1",
            "source_episode_index": 0,
            "scene_id": "env_airsim_18",
        },
    }
    (root / "meta" / "navvla_frame_metadata.jsonl").write_text(json.dumps(frame_meta) + "\n")

    payload, videos = load_episode_report(root, episode_index=0)

    assert payload["dense_states"] == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1], [2.0, 2.0, 2.0, 0.2]]
    assert payload["sampled_waypoint_indices"] == [0, 2]
    assert payload["image_interval_seconds"] == 5.0
    assert payload["instruction"] == "fly to the building"
    assert payload["transform_note"] == "OpenFly render Z 反射后，再按首帧机体朝向转换为 FRD 局部坐标"
    assert videos["front"]["start_frame"] == 0
    assert videos["front"]["frame_count"] == 2
    assert videos["front"]["source_path"].name == "part-000.mp4"


def test_build_ffmpeg_extract_command_cuts_exact_episode_frames(tmp_path) -> None:
    from tool.navvla.enhanced_vln_report import build_ffmpeg_extract_command

    command = build_ffmpeg_extract_command(
        {
            "source_path": tmp_path / "source.mp4",
            "start_frame": 123,
            "frame_count": 21,
            "fps": 1.0,
        },
        output_path=tmp_path / "front.mp4",
    )

    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert "trim=start_frame=123:end_frame=144,setpts=PTS-STARTPTS" in command
    assert command[-1] == str(tmp_path / "front.mp4")


def test_media_file_data_uri_embeds_video_bytes(tmp_path) -> None:
    from tool.navvla.enhanced_vln_report import media_file_data_uri

    video = tmp_path / "front.mp4"
    video.write_bytes(b"\x00\x01\x02\xff")

    assert media_file_data_uri(video, mime_type="video/mp4") == "data:video/mp4;base64,AAEC/w=="


def test_embedded_video_payload_is_not_duplicated_in_visible_source_label() -> None:
    from tool.navvla.enhanced_vln_report import _video_cards

    sources = {view: f"data:video/mp4;base64,{view.upper()}PAYLOAD" for view in ("front", "back", "left", "right")}

    cards = _video_cards(sources)

    for source in sources.values():
        assert cards.count(source) == 1
    assert cards.count("内嵌 MP4") == 4


def test_generate_standalone_report_writes_one_html_with_four_embedded_videos(tmp_path, monkeypatch) -> None:
    import tool.navvla.enhanced_vln_report as report_module

    payload = {
        "dataset_label": "OpenFly",
        "episode_id": "episode-0",
        "scene_id": "scene-0",
        "instruction": "fly forward",
        "dense_states": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        "sampled_waypoint_indices": [0, 1],
        "image_interval_seconds": 5.0,
        "video_fps": 1.0,
        "camera_parameters": {
            view: {"pose": [0.0] * 7, "pose_degrees": [0.0] * 7} for view in ("front", "back", "left", "right")
        },
        "summary": {"dense_waypoints": 2, "video_frames": 2, "trajectory_duration_seconds": 1.0},
        "transform_note": "identity",
    }
    specs = {
        view: {"source_path": tmp_path / f"{view}-source.mp4", "start_frame": 0, "frame_count": 2, "fps": 1.0}
        for view in ("front", "back", "left", "right")
    }
    monkeypatch.setattr(report_module, "load_episode_report", lambda *_args, **_kwargs: (payload, specs))

    def fake_run(command, *, check):
        assert check is True
        output = tmp_path / Path(command[-1]).name
        Path(command[-1]).write_bytes(f"video:{output.stem}".encode())

    monkeypatch.setattr(report_module.subprocess, "run", fake_run)
    output_html = tmp_path / "openfly.html"

    result = report_module.generate_standalone_episode_report(
        tmp_path / "dataset", output_html=output_html, episode_index=0
    )

    rendered = output_html.read_text()
    assert rendered.count("data:video/mp4;base64,") == 4
    assert not (tmp_path / "videos").exists()
    assert result["html"] == str(output_html)
    assert result["standalone"] is True
