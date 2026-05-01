import sqlite3
from datetime import datetime, timezone


class DatabaseLogger:
    def __init__(self, db_path: str = "logs.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    source TEXT NOT NULL,
                    from_cache INTEGER NOT NULL,
                    response_time_ms INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def log_interaction(
        self,
        query: str,
        response: str,
        source: str,
        user_id: str = "local",
        username: str = "local_user",
        from_cache: bool = False,
        response_time_ms: int = 0,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO logs (
                    timestamp, user_id, username, query, response, source, from_cache, response_time_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    user_id,
                    username,
                    query,
                    response,
                    source,
                    1 if from_cache else 0,
                    int(response_time_ms),
                ),
            )
            conn.commit()

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(from_cache), 0) AS cache_hits,
                    COALESCE(AVG(response_time_ms), 0) AS avg_response_time_ms
                FROM logs
                """
            ).fetchone()

        return {
            "total_requests": int(row["total_requests"]),
            "cache_hits": int(row["cache_hits"]),
            "avg_response_time_ms": float(row["avg_response_time_ms"]),
        }

    def close(self) -> None:
        return

    def __del__(self):
        return
