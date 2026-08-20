from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time


@dataclass(frozen=True)
class EpisodeJob:
    episode_id: str
    scene_id: str
    source_episode_index: int
    byte_start: int
    byte_end: int
    request_count: int
    start_index: int
    shard_id: str = ""
    zero_position_camera_fallback_views: tuple = ()


class CollectorState:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                episode_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                source_episode_index INTEGER NOT NULL UNIQUE,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                request_count INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                shard_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                worker_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                zero_position_camera_fallback_views TEXT NOT NULL DEFAULT '',
                last_error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(jobs)")
        }
        if "shard_id" not in columns:
            self.connection.execute(
                "ALTER TABLE jobs ADD COLUMN shard_id TEXT NOT NULL DEFAULT ''"
            )
        if "zero_position_camera_fallback_views" not in columns:
            self.connection.execute(
                "ALTER TABLE jobs ADD COLUMN zero_position_camera_fallback_views TEXT NOT NULL DEFAULT ''"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS jobs_status_scene_shard ON jobs(status, scene_id, shard_id, source_episode_index)"
        )
        self.connection.commit()

    def close(self):
        self.connection.close()

    def clear_request_index(self):
        self.connection.execute("DELETE FROM jobs")
        self.connection.execute(
            "DELETE FROM run_metadata WHERE key NOT LIKE 'phase:%'"
        )
        self.connection.commit()

    def add_episode(self, episode_id, scene_id, source_episode_index,
                    byte_start, byte_end, request_count, start_index=0,
                    shard_id=""):
        self.connection.execute(
            """INSERT INTO jobs(
                episode_id, scene_id, source_episode_index, byte_start, byte_end,
                request_count, start_index, shard_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                scene_id=excluded.scene_id,
                source_episode_index=excluded.source_episode_index,
                byte_start=excluded.byte_start,
                byte_end=excluded.byte_end,
                request_count=excluded.request_count,
                start_index=excluded.start_index,
                shard_id=excluded.shard_id,
                updated_at=excluded.updated_at
            """,
            (str(episode_id), str(scene_id), int(source_episode_index),
             int(byte_start), int(byte_end), int(request_count), int(start_index),
             str(shard_id), time.time()),
        )
        self.connection.commit()

    def claim_next_job(self, worker_id, preferred_scene=None, skipped_scenes=()):
        skipped_scenes = tuple(str(item) for item in skipped_scenes)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            clauses = ["status='pending'"]
            parameters = []
            if skipped_scenes:
                clauses.append("scene_id NOT IN ({})".format(",".join("?" for _ in skipped_scenes)))
                parameters.extend(skipped_scenes)
            active_scene_order = (
                "CASE WHEN scene_id IN ("
                "SELECT DISTINCT scene_id FROM jobs "
                "WHERE status='running' AND worker_id<>?"
                ") THEN 1 ELSE 0 END"
            )
            ordering = active_scene_order + ", scene_id, shard_id, source_episode_index"
            parameters.append(str(worker_id))
            if preferred_scene is not None:
                ordering = (
                    "CASE WHEN scene_id=? THEN 0 ELSE 1 END, " + ordering
                )
                parameters.insert(len(skipped_scenes), str(preferred_scene))
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE {} ORDER BY {} LIMIT 1".format(
                    " AND ".join(clauses), ordering
                ), parameters,
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE jobs SET status='running', worker_id=?, attempts=attempts+1, updated_at=? WHERE episode_id=?",
                (str(worker_id), time.time(), row["episode_id"]),
            )
            self.connection.commit()
            return self._episode_job(row)
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _episode_job(row):
        return EpisodeJob(
            row["episode_id"], row["scene_id"], row["source_episode_index"],
            row["byte_start"], row["byte_end"], row["request_count"],
            row["start_index"], row["shard_id"],
            tuple(
                view for view in row["zero_position_camera_fallback_views"].split(",")
                if view
            ),
        )

    def claim_pending_in_scene(self, worker_id, scene_id):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """SELECT * FROM jobs
                   WHERE status='pending' AND scene_id=?
                   ORDER BY shard_id, source_episode_index LIMIT 1""",
                (str(scene_id),),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE jobs SET status='running', worker_id=?, attempts=attempts+1, updated_at=? WHERE episode_id=?",
                (str(worker_id), time.time(), row["episode_id"]),
            )
            self.connection.commit()
            return self._episode_job(row)
        except Exception:
            self.connection.rollback()
            raise

    def scene_primary_work_remaining(self, scene_id):
        row = self.connection.execute(
            """SELECT 1 FROM jobs
               WHERE scene_id=? AND (
                   (status='pending' AND attempts=0) OR
                   (status='running' AND attempts=1)
               ) LIMIT 1""",
            (str(scene_id),),
        ).fetchone()
        return row is not None

    def claim_failed_for_worker(self, worker_id, scene_id, retry_rounds):
        retry_rounds = int(retry_rounds)
        if retry_rounds < 0:
            raise ValueError("retry rounds must be non-negative")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """SELECT * FROM jobs
                   WHERE status='failed' AND worker_id=? AND scene_id=?
                     AND attempts<=?
                   ORDER BY source_episode_index LIMIT 1""",
                (str(worker_id), str(scene_id), retry_rounds),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, last_error=NULL, updated_at=? WHERE episode_id=?",
                (time.time(), row["episode_id"]),
            )
            self.connection.commit()
            return self._episode_job(row)
        except Exception:
            self.connection.rollback()
            raise

    def mark_complete(self, episode_id):
        self.connection.execute(
            "UPDATE jobs SET status='complete', worker_id=NULL, last_error=NULL, updated_at=? WHERE episode_id=?",
            (time.time(), str(episode_id)),
        )
        self.connection.commit()

    def mark_failed(self, episode_id, error):
        self.connection.execute(
            "UPDATE jobs SET status='failed', last_error=?, updated_at=? WHERE episode_id=?",
            (str(error), time.time(), str(episode_id)),
        )
        self.connection.commit()

    def enable_zero_position_camera_fallback_view(self, episode_id, view):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT zero_position_camera_fallback_views FROM jobs WHERE episode_id=?",
                (str(episode_id),),
            ).fetchone()
            if row is None:
                raise KeyError("unknown episode {}".format(episode_id))
            views = {
                item for item in row["zero_position_camera_fallback_views"].split(",")
                if item
            }
            views.add(str(view))
            self.connection.execute(
                "UPDATE jobs SET zero_position_camera_fallback_views=?, updated_at=? WHERE episode_id=?",
                (",".join(sorted(views)), time.time(), str(episode_id)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def mark_missing_scene(self, episode_id):
        self.connection.execute(
            "UPDATE jobs SET status='missing_scene', worker_id=NULL, last_error='scene archive unavailable', updated_at=? WHERE episode_id=?",
            (time.time(), str(episode_id)),
        )
        self.connection.commit()

    def reset_interrupted_jobs(self):
        self.connection.execute(
            "UPDATE jobs SET status='pending', worker_id=NULL, attempts=0, updated_at=? WHERE status='running'",
            (time.time(),),
        )
        self.connection.commit()

    def reset_failed_jobs(self):
        self.connection.execute(
            "UPDATE jobs SET status='pending', worker_id=NULL, attempts=0, last_error=NULL, updated_at=? WHERE status='failed'",
            (time.time(),),
        )
        self.connection.commit()

    def reset_job_pending(self, episode_id):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                """UPDATE jobs
                   SET status='pending', worker_id=NULL, attempts=0,
                       last_error=NULL, updated_at=?
                   WHERE episode_id=?""",
                (time.time(), str(episode_id)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def status_counts(self):
        return {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            )
        }

    def job_status(self, episode_id):
        row = self.connection.execute(
            "SELECT status FROM jobs WHERE episode_id=?", (str(episode_id),)
        ).fetchone()
        return None if row is None else row["status"]

    def iter_jobs(self, include_skipped=True, skipped_scenes=()):
        query = "SELECT * FROM jobs"
        parameters = []
        if not include_skipped and skipped_scenes:
            values = tuple(str(item) for item in skipped_scenes)
            query += " WHERE scene_id NOT IN ({})".format(",".join("?" for _ in values))
            parameters.extend(values)
        query += " ORDER BY source_episode_index"
        for row in self.connection.execute(query, parameters):
            yield EpisodeJob(
                row["episode_id"], row["scene_id"], row["source_episode_index"],
                row["byte_start"], row["byte_end"], row["request_count"],
                row["start_index"], row["shard_id"],
                tuple(
                    view for view in row["zero_position_camera_fallback_views"].split(",")
                    if view
                ),
            )

    def set_metadata(self, key, value):
        self.connection.execute(
            "INSERT INTO run_metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(key), str(value)),
        )
        self.connection.commit()

    def get_metadata(self, key, default=None):
        row = self.connection.execute(
            "SELECT value FROM run_metadata WHERE key=?", (str(key),)
        ).fetchone()
        return default if row is None else row["value"]
