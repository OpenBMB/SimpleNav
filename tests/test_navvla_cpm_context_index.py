from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tool.navvla.context_index import (
    ContextIndexConfig,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_CONTEXT_TOKEN_BUDGETS,
    _select_long_memory_candidate,
    _select_history,
    load_runtime_context_index,
    normalize_context_token_budgets,
    resolve_context_index_paths,
)
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec


def _history_selection_fixture(frame_count: int = 21):
    frames = [
        NavVLAFrame(
            frame_index=frame_index,
            timestamp=float(frame_index),
            media_paths={"front_image": Path(f"/nonexistent/{frame_index}.png")},
            state=[float(frame_index), 0.0, 0.0, 0.0],
            action=[[0.0, 0.0, 0.0, 0.0]],
            source_frame_index=frame_index,
            data_index=frame_index,
        )
        for frame_index in range(frame_count)
    ]
    camera = NavVLACameraSpec(name="front", video_key="front_image", viewpoint_type="front", azimuth_rad=0.0)
    task = NavVLATaskSpec(
        task_index=0,
        instruction="go forward",
        task_type="navigation",
        task_subtype="vln",
        platform_text="uav",
        dataset_source="pytest",
        scene_id="scene-a",
    )
    episode = NavVLAEpisode(
        episode_id="episode-a",
        task=task,
        frames=frames,
        cameras=[camera],
        split="vln_train",
        trajectory_id="traj-a",
    )
    spec = NavVLADatasetSpec(
        dataset_name="unit",
        fps=1.0,
        control_frequency_hz=1.0,
        context_policy_version="history-v2",
        cache_policy_version="none",
        action_horizon=1,
        action_dim=4,
        state_dim=4,
        split="vln_train",
    )
    return frames, episode, spec


def test_default_context_token_budgets_exclude_512() -> None:
    assert DEFAULT_CONTEXT_TOKEN_BUDGET == 1024
    assert DEFAULT_CONTEXT_TOKEN_BUDGETS == (1024, 2048)
    assert normalize_context_token_budgets(None) == (1024, 2048)
    assert normalize_context_token_budgets((512,)) == (512,)


def test_bats_overflow_uses_independent_priorities_instead_of_recent_tail() -> None:
    frames, episode, spec = _history_selection_fixture()
    config = ContextIndexConfig(
        epsilon=0.1,
        use_dynamic_bats_k=False,
        k=0.0,
        bats_token_budget=108,
        current_visual_tokens=64,
        history_visual_tokens=4,
        tvi_tokens=1,
        seed=42,
        budget_num_cameras=1,
        history_camera_names=("front",),
    )

    selected, probabilities, draws, ranked = _select_history(
        frames,
        20,
        episode=episode,
        spec=spec,
        config=config,
        bats_k=0.0,
        budget_num_cameras=1,
    )

    assert [frame.frame_index for frame in selected] == [0, 3, 7, 14, 17]
    assert [frame.frame_index for frame in selected] != [15, 16, 17, 18, 19]
    assert len(probabilities) == 5
    assert len(draws) == 5
    assert {frame.frame_index for frame in ranked} == {0, 3, 7, 14, 17}


def test_bats_history_uses_seeded_independent_probability_sampling() -> None:
    frames, episode, spec = _history_selection_fixture()
    config = ContextIndexConfig(
        epsilon=0.1,
        use_dynamic_bats_k=False,
        k=4.0,
        bats_token_budget=108,
        current_visual_tokens=64,
        history_visual_tokens=4,
        tvi_tokens=1,
        seed=42,
        budget_num_cameras=1,
        history_camera_names=("front",),
    )

    selected, probabilities, draws, _ranked = _select_history(
        frames,
        20,
        episode=episode,
        spec=spec,
        config=config,
        bats_k=4.0,
        budget_num_cameras=1,
    )

    assert [frame.frame_index for frame in selected] == [0, 14, 17]
    assert len(probabilities) == 3
    assert len(draws) == 3


def test_context_index_stores_only_compact_long_memory_frames(tiny_navvla_dataset_root: Path) -> None:
    context = load_runtime_context_index(resolve_context_index_paths(tiny_navvla_dataset_root, token_budget=1024))

    for data_index in context.meta["index"].astype(int).tolist():
        row = context.materialize_by_data_index(data_index)
        assert "long_memory_frames" in row
        assert "long_memory_steps" not in row
        assert "long_memory_blocks" not in row
        assert "long_memory_token_refs" not in row
        assert "long_memory_mask" not in row
        assert "long_memory_max_frame_index" not in row
        assert "long_memory_update_frame_index" not in row

    first_visible_memory = context.materialize_by_data_index(2)
    assert list(first_visible_memory["long_memory_frames"]) == [{"frame_index": 0, "camera_mask": 1}]


def test_select_long_memory_candidate_walks_bats_tail_until_valid() -> None:
    ranked = [
        {"frame_index": 10},
        {"frame_index": 11},
        {"frame_index": 13},
        {"frame_index": 14},
        {"frame_index": 12},
    ]

    candidate = _select_long_memory_candidate(
        ranked,
        memory_frame_indices={10, 12},
    )

    assert candidate == {"frame_index": 14}


def test_select_long_memory_candidate_rejects_not_newer_than_memory_max() -> None:
    ranked = [
        {"frame_index": 10},
        {"frame_index": 11},
        {"frame_index": 12},
    ]

    candidate = _select_long_memory_candidate(
        ranked,
        memory_frame_indices={13},
    )

    assert candidate is None


def test_streaming_context_builder_matches_in_memory_builder(tmp_path: Path) -> None:
    from tool.navvla.context_index import ContextIndexConfig, build_context_index, build_context_index_streaming
    from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frames = []
    for frame_index in range(6):
        image_path = image_dir / f"{frame_index}.png"
        Image.fromarray(np.full((8, 8, 3), frame_index * 30, dtype=np.uint8)).save(image_path)
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(frame_index),
                media_paths={"front_image": image_path},
                state=[float(frame_index), 0.0, 0.0, 0.0],
                action=[[0.0, 0.0, 0.0, 0.0]],
                source_frame_index=frame_index,
                data_index=frame_index,
            )
        )
    camera = NavVLACameraSpec(name="front", video_key="front_image", viewpoint_type="front", azimuth_rad=0.0)
    task = NavVLATaskSpec(
        task_index=0,
        instruction="go forward",
        task_type="navigation",
        task_subtype="vln",
        platform_text="uav",
        dataset_source="pytest",
        scene_id="scene-a",
    )
    episode = NavVLAEpisode(
        episode_id="episode-a",
        task=task,
        frames=frames,
        cameras=[camera],
        split="vln_train",
        trajectory_id="traj-a",
    )
    spec = NavVLADatasetSpec(
        dataset_name="streaming-test",
        fps=1.0,
        control_frequency_hz=1.0,
        context_policy_version="bats-v1",
        cache_policy_version="none",
        action_horizon=1,
        action_dim=4,
        state_dim=4,
        split="vln_train",
    )
    config = ContextIndexConfig(
        use_dynamic_bats_k=False,
        k=1.0,
        bats_token_budget=1024,
        budget_num_cameras=1,
        history_camera_names=("front",),
    )
    old_result = build_context_index(
        [episode],
        spec=spec,
        output_root=tmp_path / "old",
        config=config,
        cache_manifest=None,
        output_token_budget=1024,
    )
    new_result = build_context_index_streaming(
        [episode],
        spec=spec,
        output_root=tmp_path / "new",
        config=config,
        cache_manifest=None,
        output_token_budget=1024,
        batch_size=2,
    )

    old_runtime = load_runtime_context_index(old_result)
    new_runtime = load_runtime_context_index(new_result)
    pd.testing.assert_frame_equal(old_runtime.meta, new_runtime.meta)
    assert not old_result.refs_path.exists()
    assert not new_result.refs_path.exists()
    for name in old_runtime.arrays:
        np.testing.assert_array_equal(old_runtime.arrays[name], new_runtime.arrays[name])
    for index in range(len(frames)):
        assert old_runtime.materialize_by_data_index(index) == new_runtime.materialize_by_data_index(index)
    pd.testing.assert_frame_equal(pd.read_parquet(old_result.debug_path), pd.read_parquet(new_result.debug_path))
