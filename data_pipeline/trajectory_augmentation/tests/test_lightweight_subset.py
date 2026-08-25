import unittest

from vln_aug.lerobot_io import EpisodeMetadata
from vln_aug.lightweight_subset import plan_lightweight_subset


def episode(index: int, scene_id: str) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        episode_id=f"episode-{index}",
        scene_id=scene_id,
        length=10,
        data_chunk_index=0,
        data_file_index=0,
    )


class LightweightSubsetTests(unittest.TestCase):
    def test_selects_exact_half_after_forced_scene_exclusion(self):
        episodes = [
            episode(index, "1" if index < 4 else str(2 + index % 3))
            for index in range(16)
        ]

        plan = plan_lightweight_subset(
            episodes,
            retain_fraction=0.5,
            excluded_scene_ids={"1"},
            seed=20260716,
            stride_choices=(5, 6, 7, 8),
        )

        self.assertEqual(plan.source_episode_count, 16)
        self.assertEqual(plan.target_episode_count, 8)
        self.assertEqual(len(plan.selected_episode_indices), 8)
        self.assertTrue(
            all(index >= 4 for index in plan.selected_episode_indices)
        )
        self.assertEqual(plan.excluded_scene_episode_count, 4)
        self.assertEqual(plan.stride_episode_counts, {5: 2, 6: 2, 7: 2, 8: 2})

    def test_selection_and_stride_assignment_are_reproducible(self):
        episodes = [episode(index, str(2 + index % 4)) for index in range(40)]

        first = plan_lightweight_subset(
            episodes,
            retain_fraction=0.5,
            excluded_scene_ids={"1"},
            seed=99,
            stride_choices=(5, 6, 7, 8),
        )
        second = plan_lightweight_subset(
            reversed(episodes),
            retain_fraction=0.5,
            excluded_scene_ids={"1"},
            seed=99,
            stride_choices=(5, 6, 7, 8),
        )

        self.assertEqual(first.selected_episode_indices, second.selected_episode_indices)
        self.assertEqual(first.stride_by_episode_index, second.stride_by_episode_index)

    def test_rejects_exclusion_that_leaves_too_few_candidates(self):
        episodes = [episode(index, "1" if index < 8 else "2") for index in range(10)]

        with self.assertRaisesRegex(ValueError, "too few eligible episodes"):
            plan_lightweight_subset(
                episodes,
                retain_fraction=0.5,
                excluded_scene_ids={"1"},
                seed=1,
                stride_choices=(5, 6, 7, 8),
            )


if __name__ == "__main__":
    unittest.main()
