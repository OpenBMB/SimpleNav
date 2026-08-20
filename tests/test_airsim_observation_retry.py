from types import SimpleNamespace

from NavVLAeval.common.simulators.airsim import observation as observation_module


def test_sim_get_images_waits_for_renderer_and_retries_until_tenth_response(monkeypatch) -> None:
    invalid = SimpleNamespace(width=0, height=0, image_data_uint8=b"")
    valid = SimpleNamespace(width=2, height=2, image_data_uint8=b"x" * 12)

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def simGetImages(self, requests):
            del requests
            self.calls += 1
            return [valid if self.calls == 10 else invalid]

    sleeps = []
    monkeypatch.setattr(observation_module, "sleep", lambda delay: sleeps.append(delay), raising=False)
    client = _Client()

    responses = observation_module._sim_get_images_with_valid_scenes(
        client,
        requests=[object()],
        scene_response_indices=[0],
        label="openfly",
    )

    assert responses == [valid]
    assert client.calls == 10
    assert sleeps == [0.1] * 9
