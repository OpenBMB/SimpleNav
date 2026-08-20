import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.navvla_lerobot_datasets import NavVLALeRobotDataset
from tool.navvla.validation import _validate_runtime_context_alignment


def _dataset(required_cameras=None):
    dataset = object.__new__(NavVLALeRobotDataset)
    dataset.required_cameras = required_cameras
    return dataset


def test_truncate_context_allows_history_steps_and_blocks_to_have_different_lengths():
    dataset = _dataset(required_cameras=["front", "left"])
    context_row = {
        "context.index_key": "ctx-ok",
        "history_steps": [
            {"frame_index": 10},
            {"frame_index": 20},
        ],
        "history_blocks": [
            {"camera_name": "front", "step_index": 0},
            {"camera_name": "left", "step_index": 0},
            {"camera_name": "front", "step_index": 1},
            {"camera_name": "left", "step_index": 1},
        ],
        "history_token_refs": ["f0", "l0", "f1", "l1"],
        "history_mask": [True, True, False, True],
    }

    truncated = dataset._truncate_context_for_training(context_row)

    assert truncated["history_steps"] == context_row["history_steps"]
    assert truncated["history_token_refs"] == ["f0", "l0", "f1", "l1"]
    assert truncated["history_mask"] == [True, True, False, True]


def test_truncate_context_rejects_per_block_field_length_mismatch():
    dataset = _dataset(required_cameras=["front"])
    context_row = {
        "context.index_key": "ctx-bad-length",
        "history_steps": [{"frame_index": 10}],
        "history_blocks": [{"camera_name": "front", "step_index": 0}],
        "history_token_refs": [],
        "history_mask": [True],
    }

    with pytest.raises(ValueError, match="ctx-bad-length.*history_token_refs.*history_blocks"):
        dataset._truncate_context_for_training(context_row)


def test_truncate_context_rejects_block_step_index_out_of_range():
    dataset = _dataset(required_cameras=["front"])
    context_row = {
        "context.index_key": "ctx-bad-step",
        "history_steps": [{"frame_index": 10}],
        "history_blocks": [{"camera_name": "front", "step_index": 1}],
        "history_token_refs": ["f1"],
        "history_mask": [True],
    }

    with pytest.raises(IndexError, match="ctx-bad-step.*step_index.*history_steps"):
        dataset._truncate_context_for_training(context_row)


def test_history_tvi_reads_timestamp_from_history_steps():
    dataset = _dataset(required_cameras=["front"])
    dataset.cameras = {"front": {"azimuth_rad": 0.25}}
    context_row = {
        "history_steps": [{"timestamp": 3.5}],
        "history_blocks": [{"camera_name": "front", "step_index": 0}],
    }

    np.testing.assert_allclose(dataset._history_tvi(context_row), np.asarray([[3.5, 0.25]], dtype=np.float32))


def test_runtime_context_alignment_validates_storage_without_materializing_rows():
    class RuntimeContext:
        def __init__(self):
            self.meta = pd.DataFrame(
                [
                    {
                        "index": 0,
                        "history_offset": 0,
                        "history_count": 1,
                        "long_memory_offset": 0,
                        "long_memory_count": 0,
                    },
                    {
                        "index": 1,
                        "history_offset": 1,
                        "history_count": 2,
                        "long_memory_offset": 0,
                        "long_memory_count": 1,
                    },
                ]
            )
            self.arrays = {
                "history_frame_index": np.asarray([0, 0, 1], dtype=np.int64),
                "history_camera_mask": np.asarray([1, 1, 1], dtype=np.uint64),
                "long_memory_frame_index": np.asarray([0], dtype=np.int64),
                "long_memory_camera_mask": np.asarray([1], dtype=np.uint64),
            }
            self.camera_names = ["front"]

        def materialize_meta_row(self, row):
            raise AssertionError("alignment validation must not materialize nested context rows")

    report = _validate_runtime_context_alignment(RuntimeContext())

    assert report["rows"] == 2
    assert report["max_history_frames"] == 2
    assert report["max_long_memory_frames"] == 1


def test_frame_metadata_inventory_streams_count_and_small_sample(tmp_path):
    from tool.navvla.validation import _read_frame_metadata_inventory

    path = tmp_path / "frames.jsonl"
    path.write_text('{"index": 0}\n{"index": 1}\n{"index": 2}\n', encoding="utf-8")

    count, sample = _read_frame_metadata_inventory(path, sample_size=2)

    assert count == 3
    assert sample == [{"index": 0}, {"index": 1}]
