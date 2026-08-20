# ruff: noqa: E501, RUF001

from __future__ import annotations

import base64
import html
import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tool.navvla.adapters.enhanced_vln import render_poses_to_training_local

VIEWS = ("front", "back", "left", "right")
VIEW_LABELS = {
    "front": "前视 Front",
    "back": "后视 Back",
    "left": "左视 Left",
    "right": "右视 Right",
}


def media_file_data_uri(path: str | Path, *, mime_type: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_parquet_tree(path: Path) -> pa.Table:
    shards = sorted(path.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found below {path}")
    return pa.concat_tables([pq.read_table(shard) for shard in shards], promote_options="default")


def _read_frame_metadata(path: Path, *, global_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["index"]) == global_index:
                return record
    raise ValueError(f"frame metadata does not contain global index {global_index}")


def _read_source_trajectory(
    source_package: Path, *, source_episode_index: int, source_episode_id: str
) -> dict[str, Any]:
    path = source_package / "trajectories" / "episodes.jsonl"
    fallback: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            if index == source_episode_index:
                fallback = record
                if record.get("episode_id") == source_episode_id:
                    return record
            if record.get("episode_id") == source_episode_id:
                return record
    if fallback is not None:
        raise ValueError(
            f"source episode index {source_episode_index} contains {fallback.get('episode_id')!r}, "
            f"expected {source_episode_id!r}"
        )
    raise ValueError(f"source trajectory does not contain episode {source_episode_id!r}")


def _clean_float(value: float) -> float:
    cleaned = round(float(value), 7)
    return 0.0 if abs(cleaned) < 1.0e-7 else cleaned


def _camera_payload(table: pa.Table) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for view in VIEWS:
        pose = [_clean_float(value) for value in table[f"observation.camera_pose.{view}"][0].as_py()]
        pose_degrees = pose.copy()
        for index in range(3, 7):
            pose_degrees[index] = _clean_float(math.degrees(pose[index]))
        result[view] = {"pose": pose, "pose_degrees": pose_degrees}
    return result


def _video_specs(dataset_root: Path, *, info: Mapping[str, Any], global_indices: list[int]) -> dict[str, dict[str, Any]]:
    index_table = pq.read_table(dataset_root / "meta" / "navvla_video_index.parquet")
    specs: dict[str, dict[str, Any]] = {}
    selected_indices = set(global_indices)
    for view in VIEWS:
        key = f"{view}_image"
        rows = [
            row
            for row in index_table.to_pylist()
            if row["video_key"] == key and bool(row["available"]) and int(row["index"]) in selected_indices
        ]
        rows.sort(key=lambda row: global_indices.index(int(row["index"])))
        if len(rows) != len(global_indices):
            raise ValueError(f"video index for {key} covers {len(rows)} of {len(global_indices)} episode frames")
        locations = {(int(row["chunk_index"]), int(row["file_index"])) for row in rows}
        if len(locations) != 1:
            raise ValueError(f"example episode for {key} spans multiple video shards: {sorted(locations)}")
        frame_indices = [int(row["video_frame_index"]) for row in rows]
        expected = list(range(frame_indices[0], frame_indices[0] + len(frame_indices)))
        if frame_indices != expected:
            raise ValueError(f"video frames for {key} are not contiguous: {frame_indices[:4]} ...")
        chunk_index, file_index = next(iter(locations))
        relative = str(info["video_path"][key]).format(chunk_index=chunk_index, file_index=file_index)
        specs[view] = {
            "source_path": dataset_root / relative,
            "start_frame": frame_indices[0],
            "frame_count": len(frame_indices),
            "fps": float(info["fps"]),
        }
    return specs


def load_episode_report(
    dataset_root: str | Path, *, episode_index: int = 0
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    dataset_root = Path(dataset_root).resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    all_data = _read_parquet_tree(dataset_root / "data")
    episode_data = all_data.filter(pc.equal(all_data["episode_index"], pa.scalar(int(episode_index))))
    if episode_data.num_rows == 0:
        raise ValueError(f"dataset does not contain episode_index={episode_index}")

    global_indices = [int(value) for value in episode_data["index"].to_pylist()]
    frame_meta = _read_frame_metadata(
        dataset_root / "meta" / "navvla_frame_metadata.jsonl", global_index=global_indices[0]
    )
    source_meta = frame_meta["source_metadata"]
    dataset_key = str(source_meta["source_dataset"])
    source_trajectory = _read_source_trajectory(
        Path(source_meta["source_package"]),
        source_episode_index=int(source_meta["source_episode_index"]),
        source_episode_id=str(source_meta["source_episode_id"]),
    )
    render_poses = [[pose[0], pose[1], pose[2], pose[5]] for pose in source_trajectory["reference_path"]]
    dense = render_poses_to_training_local(render_poses, dataset_key=dataset_key)
    dense_states = [[_clean_float(value) for value in row] for row in dense.tolist()]

    timestamps = np.asarray(episode_data["timestamp"].to_pylist(), dtype=np.float64)
    image_interval = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 5.0
    task_index = int(episode_data["task_index"][0].as_py())
    tasks = pq.read_table(dataset_root / "meta" / "tasks.parquet")
    task_rows = tasks.filter(pc.equal(tasks["task_index"], pa.scalar(task_index))).to_pylist()
    if len(task_rows) != 1:
        raise ValueError(f"expected one task row for task_index={task_index}, got {len(task_rows)}")

    dataset_label = {"AerialVLN_lerobot": "AerialVLN", "OpenFly_lerobot": "OpenFly"}.get(
        dataset_key, dataset_key.removesuffix("_lerobot")
    )
    transform_note = (
        "OpenFly render Z 反射后，再按首帧机体朝向转换为 FRD 局部坐标"
        if dataset_key == "OpenFly_lerobot"
        else "AerialVLN render 世界位姿直接按首帧机体朝向转换为 FRD 局部坐标"
    )
    payload = {
        "dataset_label": dataset_label,
        "episode_id": str(source_meta["source_episode_id"]),
        "scene_id": str(source_meta["scene_id"]),
        "instruction": str(task_rows[0]["task"]),
        "dense_states": dense_states,
        "sampled_waypoint_indices": [int(value) for value in episode_data["source_frame_index"].to_pylist()],
        "image_interval_seconds": image_interval,
        "video_fps": float(info["fps"]),
        "camera_parameters": _camera_payload(episode_data),
        "summary": {
            "dense_waypoints": len(dense_states),
            "video_frames": episode_data.num_rows,
            "trajectory_duration_seconds": float(max(0, len(dense_states) - 1)),
        },
        "transform_note": transform_note,
    }
    return payload, _video_specs(dataset_root, info=info, global_indices=global_indices)


def build_ffmpeg_extract_command(spec: Mapping[str, Any], *, output_path: str | Path) -> list[str]:
    start_frame = int(spec["start_frame"])
    frame_count = int(spec["frame_count"])
    if start_frame < 0 or frame_count <= 0:
        raise ValueError(f"invalid frame span: start={start_frame}, count={frame_count}")
    end_frame = start_frame + frame_count
    fps = float(spec["fps"])
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(spec["source_path"]),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-an",
        "-r",
        str(fps),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def generate_episode_report(
    dataset_root: str | Path, *, output_dir: str | Path, episode_index: int = 0
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"report output already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(f"report staging output already exists: {staging}")

    payload, video_specs = load_episode_report(dataset_root, episode_index=episode_index)
    try:
        video_dir = staging / "videos"
        video_dir.mkdir(parents=True)
        video_sources: dict[str, str] = {}
        for view in VIEWS:
            output_video = video_dir / f"{view}.mp4"
            subprocess.run(build_ffmpeg_extract_command(video_specs[view], output_path=output_video), check=True)
            video_sources[view] = f"videos/{view}.mp4"
        rendered = render_episode_html(payload, video_sources=video_sources)
        (staging / "index.html").write_text(rendered, encoding="utf-8")
        (staging / "report_payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output_dir),
        "html": str(output_dir / "index.html"),
        "episode_index": int(episode_index),
        "episode_id": payload["episode_id"],
        "dense_waypoints": payload["summary"]["dense_waypoints"],
        "video_frames_per_view": payload["summary"]["video_frames"],
    }


def generate_standalone_episode_report(
    dataset_root: str | Path, *, output_html: str | Path, episode_index: int = 0
) -> dict[str, Any]:
    output_html = Path(output_html).resolve()
    if output_html.suffix.lower() != ".html":
        raise ValueError(f"standalone report output must end in .html: {output_html}")
    if output_html.exists():
        raise FileExistsError(f"standalone report output already exists: {output_html}")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    staging = output_html.with_name(f".{output_html.name}.staging")
    if staging.exists():
        raise FileExistsError(f"standalone report staging output already exists: {staging}")

    payload, video_specs = load_episode_report(dataset_root, episode_index=episode_index)
    try:
        with tempfile.TemporaryDirectory(prefix="enhanced-vln-html-") as temporary:
            temporary_root = Path(temporary)
            video_sources: dict[str, str] = {}
            for view in VIEWS:
                output_video = temporary_root / f"{view}.mp4"
                subprocess.run(build_ffmpeg_extract_command(video_specs[view], output_path=output_video), check=True)
                video_sources[view] = media_file_data_uri(output_video, mime_type="video/mp4")
            staging.write_text(render_episode_html(payload, video_sources=video_sources), encoding="utf-8")
        staging.rename(output_html)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "html": str(output_html),
        "standalone": True,
        "episode_index": int(episode_index),
        "episode_id": payload["episode_id"],
        "dense_waypoints": payload["summary"]["dense_waypoints"],
        "video_frames_per_view": payload["summary"]["video_frames"],
        "html_bytes": output_html.stat().st_size,
    }


def _format_number(value: float) -> str:
    return f"{float(value):.4f}"


def _camera_rows(camera_parameters: Mapping[str, Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for view in VIEWS:
        camera = camera_parameters[view]
        pose = camera["pose"]
        pose_degrees = camera["pose_degrees"]
        rows.append(
            "<tr>"
            f'<td><span class="view-chip view-{view}">{html.escape(VIEW_LABELS[view])}</span></td>'
            f"<td>{_format_number(pose[0])}</td>"
            f"<td>{_format_number(pose[1])}</td>"
            f"<td>{_format_number(pose[2])}</td>"
            f"<td>{_format_number(pose[3])}<small>{_format_number(pose_degrees[3])}°</small></td>"
            f"<td>{_format_number(pose[4])}<small>{_format_number(pose_degrees[4])}°</small></td>"
            f"<td>{_format_number(pose[5])}<small>{_format_number(pose_degrees[5])}°</small></td>"
            f"<td>{_format_number(pose[6])}<small>{_format_number(pose_degrees[6])}°</small></td>"
            "</tr>"
        )
    return "".join(rows)


def _video_cards(video_sources: Mapping[str, str]) -> str:
    cards: list[str] = []
    for view in VIEWS:
        source = video_sources[view]
        source_label = "内嵌 MP4" if source.startswith("data:video/") else source
        cards.append(
            f'<article class="video-card view-{view}">'
            f'<div class="video-title"><span>{html.escape(VIEW_LABELS[view])}</span>'
            f"<code>{html.escape(source_label)}</code></div>"
            f'<video data-view="{view}" src="{html.escape(source)}" '
            'preload="metadata" controls playsinline muted></video>'
            "</article>"
        )
    return "".join(cards)


def render_episode_html(payload: Mapping[str, Any], *, video_sources: Mapping[str, str]) -> str:
    missing_views = [view for view in VIEWS if view not in video_sources]
    if missing_views:
        raise ValueError(f"missing video sources for views: {missing_views}")
    missing_cameras = [view for view in VIEWS if view not in payload["camera_parameters"]]
    if missing_cameras:
        raise ValueError(f"missing camera parameters for views: {missing_cameras}")

    dataset_label = html.escape(str(payload["dataset_label"]))
    episode_id = html.escape(str(payload["episode_id"]))
    scene_id = html.escape(str(payload["scene_id"]))
    instruction = html.escape(str(payload["instruction"]))
    transform_note = html.escape(str(payload["transform_note"]))
    summary = payload["summary"]
    trajectory_json = json.dumps(payload["dense_states"], ensure_ascii=False)
    sampled_json = json.dumps(payload["sampled_waypoint_indices"], ensure_ascii=False)
    camera_rows = _camera_rows(payload["camera_parameters"])
    video_cards = _video_cards(video_sources)

    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DATASET__ · __EPISODE__ 实例报告</title>
  <style>
    :root { color-scheme: dark; --bg:#071018; --panel:#0d1a24; --panel2:#112432; --line:#244252;
      --text:#eaf6fb; --muted:#9fb7c3; --cyan:#42d7e8; --orange:#ff9b54; --green:#56e39f;
      --pink:#f472b6; --yellow:#f7d154; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 15% -5%,#173446 0,transparent 34%),var(--bg);
      color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { width:min(1500px,calc(100% - 32px)); margin:24px auto 64px; }
    header { padding:28px; border:1px solid var(--line); border-radius:22px;
      background:linear-gradient(135deg,rgba(66,215,232,.12),rgba(13,26,36,.96) 42%,rgba(255,155,84,.08)); }
    h1 { margin:0 0 8px; font-size:clamp(26px,4vw,44px); letter-spacing:-.035em; }
    h2 { margin:0 0 18px; font-size:22px; }
    h3 { margin:0; font-size:16px; }
    p { margin:8px 0; }
    .muted, small { color:var(--muted); }
    .eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.15em; font-weight:750; font-size:12px; }
    .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:22px; }
    .metric { background:rgba(7,16,24,.6); border:1px solid var(--line); padding:14px 16px; border-radius:14px; }
    .metric b { display:block; font-size:22px; color:white; }
    section { margin-top:20px; padding:22px; border:1px solid var(--line); border-radius:18px; background:rgba(13,26,36,.94); }
    .two-col { display:grid; grid-template-columns:minmax(300px,.8fr) minmax(420px,1.2fr); gap:18px; }
    .note { border-left:3px solid var(--cyan); padding:9px 14px; background:#091722; border-radius:0 10px 10px 0; }
    .coord-grid { display:grid; grid-template-columns:260px 1fr; gap:20px; align-items:center; }
    .coord-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
    .coord-item { padding:10px 12px; background:var(--panel2); border-radius:10px; border:1px solid var(--line); }
    .coord-item b { color:var(--cyan); }
    .axis-diagram { width:100%; height:auto; background:#08131c; border-radius:14px; border:1px solid var(--line); }
    .plots { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .plot-card { min-width:0; background:#08131c; border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    .plot-card h3 { padding:12px 14px 0; }
    .plot-card svg { display:block; width:100%; height:330px; }
    .plot-caption { padding:0 14px 12px; color:var(--muted); font-size:12px; }
    .controls { position:sticky; top:8px; z-index:5; display:grid; grid-template-columns:auto auto 1fr auto; gap:10px;
      align-items:center; padding:12px; margin-bottom:14px; border:1px solid #315369; border-radius:13px; background:rgba(7,16,24,.94); backdrop-filter:blur(8px); }
    button { border:1px solid #3a6075; color:white; background:#143044; border-radius:9px; padding:9px 14px; cursor:pointer; }
    button:hover { background:#1b4058; }
    input[type=range] { width:100%; accent-color:var(--cyan); }
    .video-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .video-card { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#050b10; }
    .video-title { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:10px 12px; background:#0c1b26; }
    .video-title span { font-weight:750; }
    .video-title code { max-width:65%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted); font-size:11px; }
    video { width:100%; max-height:480px; display:block; background:#000; }
    table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
    th,td { padding:10px 9px; border-bottom:1px solid var(--line); text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    td small { display:block; font-size:10px; }
    .table-wrap { overflow:auto; }
    .view-chip { display:inline-block; border-radius:999px; padding:4px 9px; color:#071018; font-weight:800; }
    .view-front { --view:#42d7e8; } .view-back { --view:#ff9b54; }
    .view-left { --view:#56e39f; } .view-right { --view:#f472b6; }
    .view-chip { background:var(--view); }
    .instruction { font-size:17px; color:#ddecf2; }
    .legend { display:flex; flex-wrap:wrap; gap:15px; color:var(--muted); font-size:12px; margin:8px 0 0; }
    .legend i { width:12px; height:4px; display:inline-block; vertical-align:middle; margin-right:5px; border-radius:2px; }
    @media(max-width:980px) { .summary,.plots { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .two-col,.coord-grid { grid-template-columns:1fr; } }
    @media(max-width:680px) { main { width:min(100% - 18px,1500px); } section,header { padding:16px; }
      .summary,.plots,.video-grid,.coord-list { grid-template-columns:1fr; } .controls { grid-template-columns:auto auto 1fr; }
      #time-label { grid-column:1/-1; } }
  </style>
</head>
<body><main>
  <header>
    <div class="eyebrow">Enhanced VLN · LeRobot example</div>
    <h1>__DATASET__ 实例数据</h1>
    <p><b>Episode：</b>__EPISODE__　<b>Scene：</b>__SCENE__</p>
    <div class="summary">
      <div class="metric"><span>密集轨迹点（1 Hz）</span><b>__DENSE_COUNT__</b></div>
      <div class="metric"><span>图像/视频帧</span><b>__VIDEO_FRAMES__</b></div>
      <div class="metric"><span>轨迹时长</span><b>__DURATION__ s</b></div>
      <div class="metric"><span>图像采样间隔</span><b>__IMAGE_INTERVAL__ s</b></div>
    </div>
  </header>

  <section class="two-col">
    <div><h2>导航指令</h2><p class="instruction">__INSTRUCTION__</p></div>
    <div><h2>时间轴说明</h2>
      <p class="note">waypoint 轨迹为 1 Hz；图像每 __IMAGE_INTERVAL__ 秒采样一次；视频以 __VIDEO_FPS__ FPS 保存。
      因此视频播放 1 秒对应轨迹时间推进 __IMAGE_INTERVAL__ 秒。下方轨迹游标按该映射与视频同步。</p>
    </div>
  </section>

  <section>
    <h2>坐标系情况</h2>
    <div class="coord-grid">
      <svg class="axis-diagram" viewBox="0 0 260 230" role="img" aria-label="FRD coordinate diagram">
        <defs><marker id="arrow-c" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#42d7e8"/></marker>
        <marker id="arrow-o" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#ff9b54"/></marker>
        <marker id="arrow-g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#56e39f"/></marker></defs>
        <path d="M130 113 L130 31" stroke="#42d7e8" stroke-width="4" marker-end="url(#arrow-c)"/><text x="139" y="34" fill="#42d7e8">+X 前</text>
        <path d="M130 113 L214 145" stroke="#ff9b54" stroke-width="4" marker-end="url(#arrow-o)"/><text x="203" y="165" fill="#ff9b54">+Y 右</text>
        <path d="M130 113 L78 185" stroke="#56e39f" stroke-width="4" marker-end="url(#arrow-g)"/><text x="43" y="203" fill="#56e39f">+Z 下</text>
        <path d="M100 82 Q130 62 161 82" fill="none" stroke="#f7d154" stroke-width="3" marker-end="url(#arrow-o)"/>
        <text x="91" y="67" fill="#f7d154">+yaw 右转</text><circle cx="130" cy="113" r="8" fill="#eaf6fb"/>
      </svg>
      <div>
        <div class="coord-list">
          <div class="coord-item"><b>X：前为正</b><br>首帧机体朝向对齐</div>
          <div class="coord-item"><b>Y：右为正</b><br>水平面右侧为正</div>
          <div class="coord-item"><b>Z：下为正</b><br>下降为正、上升为负</div>
          <div class="coord-item"><b>Yaw：右转为正</b><br>绕 +Z 方向</div>
          <div class="coord-item"><b>Roll：右侧下沉为正</b><br>相机/机体角，radian</div>
          <div class="coord-item"><b>Pitch：抬头为正</b><br>相机/机体角，radian</div>
        </div>
        <p class="note"><b>转换：</b>__TRANSFORM__. XYZ 是首帧机体对齐后的 episode 局部世界轨迹，首帧为零；角度使用 radian。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>完整轨迹</h2>
    <div class="plots">
      <article class="plot-card"><h3>俯视图 X–Y</h3><svg id="trajectory-top" viewBox="0 0 520 330"></svg><div class="plot-caption">X 向前，Y 向右</div></article>
      <article class="plot-card"><h3>侧视图 X–Z</h3><svg id="trajectory-side" viewBox="0 0 520 330"></svg><div class="plot-caption">Z 向下；图中下方代表更大的 +Z</div></article>
      <article class="plot-card"><h3>等轴轨迹 X–Y–Z</h3><svg id="trajectory-iso" viewBox="0 0 520 330"></svg><div class="plot-caption">用于同时观察平面转向与高度变化</div></article>
    </div>
    <div class="legend"><span><i style="background:#42d7e8"></i>密集 1 Hz 轨迹</span><span><i style="background:#ff9b54"></i>5 秒图像采样点</span><span><i style="background:#f7d154"></i>视频当前帧对应位置</span></div>
  </section>

  <section>
    <h2>四视角完整视频</h2>
    <div class="controls"><button id="play-all">全部播放</button><button id="pause-all">全部暂停</button>
      <input id="timeline" type="range" min="0" max="1" step="0.01" value="0"><output id="time-label">video 0.0 s / trajectory 0 s</output></div>
    <div class="video-grid">__VIDEO_CARDS__</div>
  </section>

  <section>
    <h2>视角与相机 7 参数</h2>
    <p class="muted">每个 pose 为 [x, y, z, yaw, roll, pitch, fov]。位置单位为米，表中角度主值为 radian，下一行同时给出 degree 便于人工检查。</p>
    <div class="table-wrap"><table><thead><tr><th>视角</th><th>x</th><th>y</th><th>z</th><th>yaw</th><th>roll</th><th>pitch</th><th>FOV</th></tr></thead>
      <tbody>__CAMERA_ROWS__</tbody></table></div>
  </section>
</main>
<script>
const trajectory = __TRAJECTORY_JSON__;
const sampledWaypointIndices = __SAMPLED_JSON__;
const imageIntervalSeconds = __IMAGE_INTERVAL_NUMBER__;
const videoFps = __VIDEO_FPS_NUMBER__;
const NS = "http" + "://www.w3.org/2000/svg";
const videos = [...document.querySelectorAll("video[data-view]")];
const master = videos.find(v => v.dataset.view === "front") || videos[0];
const slider = document.getElementById("timeline");
const timeLabel = document.getElementById("time-label");
const markers = {};

function svgEl(name, attrs) { const el=document.createElementNS(NS,name); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); return el; }
function extent(values) { let lo=Math.min(...values), hi=Math.max(...values); if (hi-lo < 1e-8) { lo-=1; hi+=1; } const pad=(hi-lo)*.08; return [lo-pad,hi+pad]; }
function project(mode, p) {
  if (mode === "top") return [p[0], p[1]];
  if (mode === "side") return [p[0], p[2]];
  return [p[0] - .62*p[1], p[2] + .28*p[0] + .28*p[1]];
}
function drawPlot(svgId, mode) {
  const svg=document.getElementById(svgId), points=trajectory.map(p=>project(mode,p));
  const [xmin,xmax]=extent(points.map(p=>p[0])), [ymin,ymax]=extent(points.map(p=>p[1]));
  const px=x=>42+(x-xmin)/(xmax-xmin)*450, py=y=>292-(y-ymin)/(ymax-ymin)*245;
  for (let i=0;i<=5;i++) {
    const x=42+i*90, y=47+i*49;
    svg.append(svgEl("line",{x1:x,y1:47,x2:x,y2:292,stroke:"#16303f","stroke-width":1}));
    svg.append(svgEl("line",{x1:42,y1:y,x2:492,y2:y,stroke:"#16303f","stroke-width":1}));
  }
  svg.append(svgEl("polyline",{points:points.map(p=>`${px(p[0])},${py(p[1])}`).join(" "),fill:"none",stroke:"#42d7e8","stroke-width":3,"stroke-linejoin":"round"}));
  for (const idx of sampledWaypointIndices) { const p=points[Math.min(idx,points.length-1)]; svg.append(svgEl("circle",{cx:px(p[0]),cy:py(p[1]),r:2.3,fill:"#ff9b54",opacity:.8})); }
  const start=points[0], end=points[points.length-1];
  svg.append(svgEl("circle",{cx:px(start[0]),cy:py(start[1]),r:6,fill:"#56e39f",stroke:"#071018","stroke-width":2}));
  svg.append(svgEl("circle",{cx:px(end[0]),cy:py(end[1]),r:6,fill:"#f472b6",stroke:"#071018","stroke-width":2}));
  const marker=svgEl("circle",{cx:px(start[0]),cy:py(start[1]),r:7,fill:"#f7d154",stroke:"#071018","stroke-width":3}); svg.append(marker);
  markers[mode]={marker,points,px,py};
}
function setTrajectoryMarker(frameIndex) {
  const idx=sampledWaypointIndices[Math.max(0,Math.min(frameIndex,sampledWaypointIndices.length-1))] ?? 0;
  for (const item of Object.values(markers)) { const p=item.points[Math.min(idx,item.points.length-1)]; item.marker.setAttribute("cx",item.px(p[0])); item.marker.setAttribute("cy",item.py(p[1])); }
}
function syncFromMaster() {
  const duration=Number.isFinite(master.duration)?master.duration:sampledWaypointIndices.length/videoFps;
  slider.max=Math.max(duration,1); slider.value=master.currentTime;
  const frame=Math.min(sampledWaypointIndices.length-1,Math.floor(master.currentTime*videoFps+.001));
  const waypoint=sampledWaypointIndices[frame] ?? 0;
  timeLabel.value=`video ${master.currentTime.toFixed(1)} s / trajectory ${waypoint.toFixed(0)} s`;
  timeLabel.textContent=timeLabel.value; setTrajectoryMarker(frame);
  for (const v of videos) if (v!==master && Math.abs(v.currentTime-master.currentTime)>.25) v.currentTime=master.currentTime;
}
drawPlot("trajectory-top","top"); drawPlot("trajectory-side","side"); drawPlot("trajectory-iso","iso");
master.addEventListener("loadedmetadata",syncFromMaster); master.addEventListener("timeupdate",syncFromMaster); master.addEventListener("seeked",syncFromMaster);
slider.addEventListener("input",()=>{ for (const v of videos) v.currentTime=Number(slider.value); syncFromMaster(); });
document.getElementById("play-all").addEventListener("click",()=>videos.forEach(v=>v.play()));
document.getElementById("pause-all").addEventListener("click",()=>videos.forEach(v=>v.pause()));
</script></body></html>"""

    replacements = {
        "__DATASET__": dataset_label,
        "__EPISODE__": episode_id,
        "__SCENE__": scene_id,
        "__INSTRUCTION__": instruction,
        "__TRANSFORM__": transform_note,
        "__DENSE_COUNT__": str(summary["dense_waypoints"]),
        "__VIDEO_FRAMES__": str(summary["video_frames"]),
        "__DURATION__": _format_number(summary["trajectory_duration_seconds"]),
        "__IMAGE_INTERVAL__": _format_number(payload["image_interval_seconds"]),
        "__VIDEO_FPS__": _format_number(payload["video_fps"]),
        "__VIDEO_CARDS__": video_cards,
        "__CAMERA_ROWS__": camera_rows,
        "__TRAJECTORY_JSON__": trajectory_json,
        "__SAMPLED_JSON__": sampled_json,
        "__IMAGE_INTERVAL_NUMBER__": repr(float(payload["image_interval_seconds"])),
        "__VIDEO_FPS_NUMBER__": repr(float(payload["video_fps"])),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template
