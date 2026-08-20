from __future__ import annotations

from pathlib import Path

import numpy as np

from starVLA.dataloader import navvla_lerobot_datasets as datasets


def test_read_video_frame_retries_transient_open_failure(monkeypatch, tmp_path: Path) -> None:
    captures = []

    class FakeVideoCapture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.attempt = len(captures)
            self.released = False
            self.seek = None
            captures.append(self)

        def isOpened(self) -> bool:
            return self.attempt > 0

        def set(self, prop: int, value: int) -> bool:
            self.seek = (prop, value)
            return True

        def read(self):
            frame = np.asarray([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            self.released = True

    monkeypatch.setenv("NAVVLA_VIDEO_OPEN_RETRIES", "2")
    monkeypatch.setenv("NAVVLA_VIDEO_RETRY_SLEEP", "0")
    monkeypatch.setattr(datasets.cv2, "VideoCapture", FakeVideoCapture)

    image = datasets._read_video_frame(tmp_path / "video.mp4", frame_index=3)

    assert image.size == (2, 1)
    assert len(captures) == 2
    assert all(capture.released for capture in captures)
    assert captures[1].seek == (datasets.cv2.CAP_PROP_POS_FRAMES, 3)


def test_cpm_video_reader_reopens_after_transient_decode_failure(monkeypatch) -> None:
    from starVLA.dataloader.cpm_lerobot.video import VideoReaderCache

    captures = []

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.attempt = len(captures)
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, value: int) -> bool:
            self.position = int(value)
            return True

        def read(self):
            if self.attempt == 0:
                return False, None
            self.position += 1
            return True, np.asarray([[[0, 0, 255]]], dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    monkeypatch.setenv("NAVVLA_VIDEO_OPEN_RETRIES", "2")
    monkeypatch.setenv("NAVVLA_VIDEO_RETRY_SLEEP", "0")
    cache = VideoReaderCache(capture_factory=FakeCapture)

    image = cache.read("video.mp4", 3)

    assert image.size == (1, 1)
    assert len(captures) == 2
    assert captures[0].released
    assert cache.open_count == 2
