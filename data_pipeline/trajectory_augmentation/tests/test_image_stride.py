import unittest

from vln_aug.image_stride import assign_image_stride


class ImageStrideTests(unittest.TestCase):
    def test_assignment_is_stable_and_uses_only_configured_choices(self):
        choices = (1, 3, 5)
        first = assign_image_stride("OpenFly_lerobot", "episode-42", choices)
        second = assign_image_stride("OpenFly_lerobot", "episode-42", choices)
        self.assertEqual(first, second)
        self.assertIn(first, choices)
        openfly = [
            assign_image_stride("OpenFly_lerobot", f"episode-{index}", choices)
            for index in range(100)
        ]
        aerial = [
            assign_image_stride("AerialVLN_lerobot", f"episode-{index}", choices)
            for index in range(100)
        ]
        self.assertNotEqual(openfly, aerial)

    def test_large_dataset_is_approximately_split_into_thirds(self):
        counts = {1: 0, 3: 0, 5: 0}
        for episode_index in range(30000):
            stride = assign_image_stride(
                "Dataset", f"episode-{episode_index:06d}", (1, 3, 5)
            )
            counts[stride] += 1

        for count in counts.values():
            self.assertLess(abs(count - 10000), 350)

    def test_rejects_invalid_choices(self):
        with self.assertRaises(ValueError):
            assign_image_stride("Dataset", "episode", ())
        with self.assertRaises(ValueError):
            assign_image_stride("Dataset", "episode", (1, 0, 5))


if __name__ == "__main__":
    unittest.main()
