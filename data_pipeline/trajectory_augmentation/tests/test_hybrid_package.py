import unittest


class HybridPackageTests(unittest.TestCase):
    def test_selects_legacy_completed_episodes_and_fixed_stride_remaining(self):
        from vln_aug.hybrid_package import select_episode_payloads

        legacy = [
            {"episode_id": "a", "source": "legacy"},
            {"episode_id": "b", "source": "legacy"},
        ]
        fixed = [
            {"episode_id": "a", "source": "fixed"},
            {"episode_id": "b", "source": "fixed"},
        ]

        merged = select_episode_payloads(legacy, fixed, {"a"})

        self.assertEqual(
            merged,
            [
                {"episode_id": "a", "source": "legacy"},
                {"episode_id": "b", "source": "fixed"},
            ],
        )
