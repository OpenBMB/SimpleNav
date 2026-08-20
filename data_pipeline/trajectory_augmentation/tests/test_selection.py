import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch

from vln_aug.lerobot_io import (
    EpisodeMetadata,
    discover_train_splits,
    extract_episode_rows,
    iter_episode_tables,
    read_episode_metadata,
)
from vln_aug.selection import select_representative_episodes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path, episodes: list[dict], data_rows: list[dict]) -> Path:
    train = root / "vln_train"
    (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (train / "data" / "chunk-000").mkdir(parents=True)
    (train / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": len(episodes),
                "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(episodes),
        train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(data_rows),
        train / "data" / "chunk-000" / "part-000.parquet",
    )
    return train


def _episode(index: int, scene: str, length: int) -> dict:
    return {
        "episode_index": index,
        "episode_id": f"episode-{index}",
        "scene_id": scene,
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": 0,
    }


class DiscoveryTests(unittest.TestCase):
    def test_discovers_nested_train_splits_and_excludes_hidden_and_enhanced_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            expected = [
                _write_dataset(base / "plain", [_episode(0, "a", 1)], [{"episode_index": 0}]),
                _write_dataset(
                    base / "UAV-Flow" / "navvla_lerobot_uav_flow_sim",
                    [_episode(0, "b", 1)],
                    [{"episode_index": 0}],
                ),
                _write_dataset(
                    base / "VLNCE" / "navvla_lerobot_full_vlnce_r2r",
                    [_episode(0, "c", 1)],
                    [{"episode_index": 0}],
                ),
            ]
            _write_dataset(base / ".cache" / "ignored", [_episode(0, "x", 1)], [{"episode_index": 0}])
            _write_dataset(
                base / "old" / "vln_train_enhanced" / "ignored",
                [_episode(0, "y", 1)],
                [{"episode_index": 0}],
            )

            discovered = discover_train_splits(base)

            self.assertEqual(discovered, sorted(expected))

    def test_excludes_visible_backup_old_and_temp_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            expected = _write_dataset(base / "active", [_episode(0, "a", 1)], [{"episode_index": 0}])
            for name in ("backup", "old", "tmp", "dataset_backup_2026", "scratch_temp"):
                _write_dataset(base / name / "ignored", [_episode(0, "x", 1)], [{"episode_index": 0}])
            self.assertEqual(discover_train_splits(base), [expected])


class EpisodeReadTests(unittest.TestCase):
    def test_metadata_reader_requests_only_selection_and_location_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(
                Path(tmp),
                [_episode(1, "scene", 3)],
                [{"episode_index": 1, "frame_index": 0}],
            )
            real_read_table = pq.read_table
            calls = []

            def recording_read_table(*args, **kwargs):
                calls.append(kwargs.get("columns"))
                return real_read_table(*args, **kwargs)

            with patch("vln_aug.lerobot_io.pq.read_table", side_effect=recording_read_table):
                read_episode_metadata(train)

            self.assertEqual(
                calls,
                [[
                    "episode_index",
                    "episode_id",
                    "scene_id",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                ]],
            )

    def test_extracts_only_the_complete_metadata_indexed_episode_without_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = [_episode(7, "scene-a", 3), _episode(8, "scene-b", 2)]
            rows = [
                {"episode_index": 7, "frame_index": 0, "value": "a"},
                {"episode_index": 7, "frame_index": 1, "value": "b"},
                {"episode_index": 7, "frame_index": 2, "value": "c"},
                {"episode_index": 8, "frame_index": 0, "value": "other-a"},
                {"episode_index": 8, "frame_index": 1, "value": "other-b"},
            ]
            train = _write_dataset(Path(tmp) / "nested" / "dataset", episodes, rows)
            source = train / "data" / "chunk-000" / "part-000.parquet"
            before = _sha256(source)

            metadata = read_episode_metadata(train)
            table = extract_episode_rows(train, metadata[0])

            self.assertEqual([item.episode_index for item in metadata], [7, 8])
            self.assertEqual(table.column("episode_index").to_pylist(), [7, 7, 7])
            self.assertEqual(table.column("frame_index").to_pylist(), [0, 1, 2])
            self.assertEqual(table.column("value").to_pylist(), ["a", "b", "c"])
            self.assertEqual(_sha256(source), before)

    def test_metadata_reader_preserves_export_identity_and_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = _episode(7, "scene-a", 1)
            episode.update(
                {
                    "trajectory_id": "trajectory-7",
                    "task_index": 19,
                    "tasks": ["fly to the tower"],
                }
            )
            train = _write_dataset(
                Path(tmp), episode and [episode], [{"episode_index": 7, "frame_index": 0}]
            )

            metadata = read_episode_metadata(train)[0]

            self.assertEqual(metadata.trajectory_id, "trajectory-7")
            self.assertEqual(metadata.task_index, 19)
            self.assertEqual(metadata.tasks, ("fly to the tower",))

    def test_rejects_an_incomplete_episode_in_its_metadata_indexed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(
                Path(tmp),
                [_episode(4, "scene-a", 3)],
                [
                    {"episode_index": 4, "frame_index": 0},
                    {"episode_index": 4, "frame_index": 1},
                ],
            )
            record = read_episode_metadata(train)[0]

            with self.assertRaisesRegex(ValueError, "expected 3 rows, found 2"):
                extract_episode_rows(train, record)

    def test_rejects_missing_frame_even_when_row_count_matches_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(
                Path(tmp),
                [_episode(4, "scene-a", 3)],
                [
                    {"episode_index": 4, "frame_index": 0},
                    {"episode_index": 4, "frame_index": 2},
                    {"episode_index": 4, "frame_index": 2},
                ],
            )
            record = read_episode_metadata(train)[0]

            with self.assertRaisesRegex(ValueError, "non-contiguous frame_index"):
                extract_episode_rows(train, record)

    def test_iterates_episodes_while_reading_each_shared_data_part_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = [_episode(7, "scene-a", 2), _episode(8, "scene-b", 2)]
            rows = [
                {"episode_index": 7, "frame_index": 0, "observation.state": [0.0, 0.0, 0.0, 0.0]},
                {"episode_index": 7, "frame_index": 1, "observation.state": [1.0, 0.0, 0.0, 0.0]},
                {"episode_index": 8, "frame_index": 0, "observation.state": [0.0, 0.0, 0.0, 0.0]},
                {"episode_index": 8, "frame_index": 1, "observation.state": [1.0, 0.0, 0.0, 0.0]},
            ]
            train = _write_dataset(Path(tmp), episodes, rows)
            data_path = train / "data" / "chunk-000" / "part-000.parquet"
            real_read_table = pq.read_table
            data_reads = []

            def recording_read_table(path, *args, **kwargs):
                if Path(path) == data_path:
                    data_reads.append((str(path), kwargs.get("columns")))
                return real_read_table(path, *args, **kwargs)

            with patch("vln_aug.lerobot_io.pq.read_table", side_effect=recording_read_table):
                yielded = list(iter_episode_tables(train))

            self.assertEqual(
                data_reads,
                [
                    (
                        str(data_path),
                        ["episode_index", "frame_index", "observation.state"],
                    )
                ],
            )
            self.assertEqual([item.episode_index for item, _ in yielded], [7, 8])
            self.assertEqual([table.num_rows for _, table in yielded], [2, 2])


class SelectionTests(unittest.TestCase):
    def test_selects_different_scenes_from_lower_and_upper_length_quantiles(self):
        records = [
            _episode(0, "short-scene", 10),
            _episode(1, "middle-a", 20),
            _episode(2, "middle-b", 30),
            _episode(3, "long-scene", 100),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(Path(tmp), records, [{"episode_index": 0}])
            metadata = read_episode_metadata(train)

            result = select_representative_episodes(reversed(metadata))

            self.assertEqual([item.episode_index for item in result.selected], [0, 3])
            self.assertEqual(result.reason, "selected_scene_diverse_lower_upper_length_quantiles")

    def test_large_dataset_uses_quantiles_instead_of_degenerate_absolute_extremes(self):
        records = [_episode(0, "degenerate", 1)]
        records.extend(_episode(i, f"scene-{i % 3}", i + 5) for i in range(1, 101))
        records.append(_episode(102, "extreme", 10000))
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(Path(tmp), records, [{"episode_index": 0}])

            result = select_representative_episodes(read_episode_metadata(train))

            lengths = [item.length for item in result.selected]
            self.assertGreater(lengths[0], 1)
            self.assertLess(lengths[1], 10000)
            self.assertLessEqual(lengths[0], 20)
            self.assertGreaterEqual(lengths[1], 90)

    def test_selection_is_deterministic_when_multiple_pairs_are_equivalent(self):
        records = [
            _episode(9, "scene-b", 10),
            _episode(2, "scene-a", 10),
            _episode(8, "scene-d", 100),
            _episode(3, "scene-c", 100),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(Path(tmp), records, [{"episode_index": 2}])
            metadata = read_episode_metadata(train)

            forward = select_representative_episodes(metadata)
            backward = select_representative_episodes(reversed(metadata))

            self.assertEqual([item.episode_index for item in forward.selected], [2, 3])
            self.assertEqual(forward, backward)

    def test_records_fallback_reason_when_only_one_scene_is_available(self):
        records = [_episode(4, "same", 5), _episode(5, "same", 50)]
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(Path(tmp), records, [{"episode_index": 4}])

            result = select_representative_episodes(read_episode_metadata(train))

            self.assertEqual([item.episode_index for item in result.selected], [4, 5])
            self.assertEqual(result.reason, "fallback_no_scene_diversity")

    def test_records_fallback_reason_when_scenes_differ_but_lengths_do_not(self):
        records = [_episode(4, "scene-a", 12), _episode(5, "scene-b", 12)]
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(Path(tmp), records, [{"episode_index": 4}])

            result = select_representative_episodes(read_episode_metadata(train))

            self.assertEqual([item.episode_index for item in result.selected], [4, 5])
            self.assertEqual(result.reason, "fallback_no_length_diversity")

    def test_records_fallback_reason_for_a_single_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = _write_dataset(
                Path(tmp), [_episode(11, "only", 7)], [{"episode_index": 11}]
            )

            result = select_representative_episodes(read_episode_metadata(train))

            self.assertEqual([item.episode_index for item in result.selected], [11])
            self.assertEqual(result.reason, "fallback_only_one_episode_available")

    def test_records_fallback_reason_when_no_episode_is_available(self):
        result = select_representative_episodes([])

        self.assertEqual(result.selected, ())
        self.assertEqual(result.reason, "fallback_no_episodes_available")

    def test_large_split_selection_does_not_enumerate_all_episode_pairs(self):
        records = [
            EpisodeMetadata(
                episode_index=index,
                episode_id=f"episode-{index}",
                scene_id=f"scene-{index % 17}",
                length=index + 1,
                data_chunk_index=0,
                data_file_index=0,
            )
            for index in range(20_000)
        ]

        with patch(
            "vln_aug.selection.combinations",
            side_effect=AssertionError("must not enumerate episode pairs"),
            create=True,
        ):
            result = select_representative_episodes(records)

        self.assertEqual(
            [item.episode_index for item in result.selected], [2_000, 17_999]
        )


if __name__ == "__main__":
    unittest.main()
