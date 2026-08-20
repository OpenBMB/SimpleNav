from __future__ import annotations

import json
from pathlib import Path

from tool.navvla.adapters.traveluav import load_raw_episode_dir


def test_raw_episode_preserves_absolute_source_state_in_frame_metadata(tmp_path: Path) -> None:
    episode_dir = tmp_path / "NewYorkCity" / "ep_000001"
    (episode_dir / "frontcamera").mkdir(parents=True)
    (episode_dir / "frontcamera" / "000000.png").touch()
    source_state = [-181.5, 83.2, -5.4, 0.1, -0.2, -3.1]
    (episode_dir / "merged_data.json").write_text(
        json.dumps(
            {
                "index": [0],
                "trajectory_raw_detailed": [source_state],
                "conversations": [{"from": "human", "value": "Fly to the target."}],
            }
        ),
        encoding="utf-8",
    )

    episode = load_raw_episode_dir(episode_dir, split="val_seen", task_index=0)

    assert episode.frames[0].source_metadata["source_state"] == source_state
