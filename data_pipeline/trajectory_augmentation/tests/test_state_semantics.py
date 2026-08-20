import unittest

from vln_aug.reporting import absolute_pose_support


class StateSemanticsTests(unittest.TestCase):
    def test_accepts_explicit_absolute_modes(self):
        for mode in (
            "source_world_absolute_pose_xyz_yaw",
            "indooruav_world_pose_xy_zdown_yaw_minus_pi_over_2",
            "nuscenes_global_ego_pose_xyz_yaw",
        ):
            supported, _ = absolute_pose_support({"navvla": {"state_mode": mode}})
            self.assertTrue(supported)

    def test_accepts_history_named_mode_only_with_explicit_absolute_storage_proof(self):
        supported, _ = absolute_pose_support(
            {
                "navvla": {
                    "state_mode": "history_relative_body_frame_actions",
                    "stored_observation_state": "absolute_pose_ned_xyz_yaw",
                    "state_dim": 4,
                }
            }
        )
        self.assertTrue(supported)

    def test_rejects_travel_history_mode_without_absolute_storage_proof(self):
        supported, reason = absolute_pose_support(
            {"navvla": {"state_mode": "history_relative_body_frame_actions"}}
        )
        self.assertFalse(supported)
        self.assertIn("not explicitly proven", reason)


if __name__ == "__main__":
    unittest.main()
