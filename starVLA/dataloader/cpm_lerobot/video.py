from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable

import cv2
from PIL import Image

from .parquet import LazyParquetRows


VIDEO_READER_CACHE_SIZE = 12
SEQUENTIAL_DECODE_LIMIT = 32


class LazyVideoIndex:
    def __init__(self, path: str | Path, *, video_keys: list[str], data_length: int) -> None:
        self.rows = LazyParquetRows(
            path,
            columns=("index", "video_key", "available", "video_frame_index", "chunk_index", "file_index"),
        )
        configured = list(dict.fromkeys(str(value) for value in video_keys))
        first_rows = [self.rows[index] for index in range(len(configured))]
        self.video_keys = [str(row["video_key"]) for row in first_rows]
        first_indices = {int(row["index"]) for row in first_rows}
        if len(first_indices) != 1 or set(self.video_keys) != set(configured):
            raise ValueError(
                f"video index first-row camera layout {self.video_keys} does not match configured keys {configured}"
            )
        self._ordinal = {value: index for index, value in enumerate(self.video_keys)}
        expected = int(data_length) * len(self.video_keys)
        if len(self.rows) != expected:
            raise ValueError(f"video index has {len(self.rows)} rows, expected {expected} for dense camera layout")

    def get(self, data_index: int, video_key: str) -> dict[str, object]:
        return self.get_by_row_position(
            int(data_index),
            video_key,
            expected_index=int(data_index),
        )

    def get_by_row_position(
        self,
        row_position: int,
        video_key: str,
        *,
        expected_index: int,
    ) -> dict[str, object]:
        try:
            ordinal = self._ordinal[str(video_key)]
        except KeyError as exc:
            raise KeyError(f"video key {video_key!r} is absent from the configured camera layout") from exc
        row = self.rows[int(row_position) * len(self.video_keys) + ordinal]
        if int(row["index"]) != int(expected_index) or str(row["video_key"]) != str(video_key):
            raise ValueError(
                f"video index is not order-stable at physical row={row_position} "
                f"logical index={expected_index} video_key={video_key!r}: {row}"
            )
        return row


@dataclass
class _Reader:
    capture: Any
    next_frame: int = 0


class VideoReaderCache:
    def __init__(
        self,
        *,
        max_readers: int = VIDEO_READER_CACHE_SIZE,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ) -> None:
        self.max_readers = int(max_readers)
        self.capture_factory = capture_factory
        self._readers: OrderedDict[str, _Reader] = OrderedDict()
        self.open_count = 0
        self.seek_count = 0

    def read(self, path: str | Path, frame_index: int) -> Image.Image:
        key = str(path)
        reader = self._readers.pop(key, None)
        if reader is None:
            attempts = max(1, int(os.environ.get("NAVVLA_VIDEO_OPEN_RETRIES", "3")))
            retry_sleep = max(0.0, float(os.environ.get("NAVVLA_VIDEO_RETRY_SLEEP", "0.2")))
            capture = None
            for attempt in range(attempts):
                candidate = self.capture_factory(key)
                if candidate.isOpened():
                    capture = candidate
                    break
                candidate.release()
                if attempt + 1 < attempts and retry_sleep:
                    time.sleep(retry_sleep)
            if capture is None:
                raise FileNotFoundError(f"video does not open: {path}")
            reader = _Reader(capture=capture)
            self.open_count += 1
        target = int(frame_index)
        attempts = max(1, int(os.environ.get("NAVVLA_VIDEO_OPEN_RETRIES", "3")))
        retry_sleep = max(0.0, float(os.environ.get("NAVVLA_VIDEO_RETRY_SLEEP", "0.2")))
        for attempt in range(attempts):
            self._readers[key] = reader
            while len(self._readers) > self.max_readers:
                _old_key, old_reader = self._readers.popitem(last=False)
                old_reader.capture.release()
            try:
                if target != reader.next_frame:
                    distance = target - reader.next_frame
                    if distance > 0 and distance <= SEQUENTIAL_DECODE_LIMIT:
                        for _ in range(distance):
                            ok, _discarded = reader.capture.read()
                            if not ok:
                                raise IndexError(f"failed to decode preceding frame before {target} from {path}")
                            reader.next_frame += 1
                    else:
                        reader.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                        reader.next_frame = target
                        self.seek_count += 1
                ok, frame = reader.capture.read()
                if not ok:
                    raise IndexError(f"failed to read frame {target} from {path}")
                reader.next_frame = target + 1
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except IndexError:
                self._readers.pop(key, None)
                reader.capture.release()
                if attempt + 1 == attempts:
                    raise
                if retry_sleep:
                    time.sleep(retry_sleep)
                reader = _Reader(capture=self.capture_factory(key))
                if not reader.capture.isOpened():
                    reader.capture.release()
                    raise FileNotFoundError(f"video does not open: {path}")
                self.open_count += 1

        raise AssertionError("unreachable")

    def close(self) -> None:
        while self._readers:
            _key, reader = self._readers.popitem(last=False)
            reader.capture.release()

    def __del__(self) -> None:
        self.close()
