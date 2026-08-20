from __future__ import annotations

from dataclasses import dataclass

import pytest

from NavVLAeval.common.simulators.habitat.vlnce031_runtime import VLNCE031HabitatRuntime


@dataclass(frozen=True)
class _Episode:
    episode_id: str


class _EpisodeIterator:
    def __init__(self, episodes: list[_Episode]) -> None:
        self.episodes = episodes
        self._index = 0

    def __next__(self) -> _Episode:
        episode = self.episodes[self._index]
        self._index += 1
        return episode

    def set_next_episode_by_id(self, episode_id: str | int) -> None:
        requested = str(episode_id)
        for index, episode in enumerate(self.episodes):
            if episode.episode_id == requested:
                self._index = index
                return
        raise ValueError(f"unknown episode_id={episode_id!r}")


class _FakeHabitatEnv:
    def __init__(self, *, honor_iterator_on_reset: bool = True) -> None:
        episodes = [_Episode("1"), _Episode("14")]
        self._episode_iterator = _EpisodeIterator(episodes)
        self._current_episode = episodes[0]
        self._episode_from_iter_on_reset = False
        self._honor_iterator_on_reset = honor_iterator_on_reset
        self.episode_iterator_set_count = 0

    @property
    def current_episode(self) -> _Episode:
        return self._current_episode

    @property
    def episode_iterator(self) -> _EpisodeIterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, iterator: _EpisodeIterator) -> None:
        self._episode_iterator = iterator
        self._episode_from_iter_on_reset = True
        self.episode_iterator_set_count += 1

    @property
    def episodes(self) -> list[_Episode]:
        return self._episode_iterator.episodes

    def reset(self) -> dict[str, str]:
        if self._episode_from_iter_on_reset and self._honor_iterator_on_reset:
            self._current_episode = next(self._episode_iterator)
        self._episode_from_iter_on_reset = True
        return {"episode_id": self._current_episode.episode_id}


def _runtime(env: _FakeHabitatEnv) -> VLNCE031HabitatRuntime:
    runtime = VLNCE031HabitatRuntime(data_root=".", split="val_unseen", gpu_id=0, load_on_init=False)
    runtime.env = env
    runtime._response = lambda observation: observation  # type: ignore[method-assign]
    return runtime


def test_reset_rearms_habitat_iterator_after_seeking_episode() -> None:
    env = _FakeHabitatEnv()

    response = _runtime(env).reset({"episode_id": "14"})

    assert env.episode_iterator_set_count == 1
    assert env.current_episode.episode_id == "14"
    assert response["episode_id"] == "14"


def test_reset_rejects_requested_and_actual_episode_mismatch() -> None:
    env = _FakeHabitatEnv(honor_iterator_on_reset=False)

    with pytest.raises(RuntimeError, match=r"requested_episode_id='14'.*actual_episode_id='1'"):
        _runtime(env).reset({"episode_id": "14"})
