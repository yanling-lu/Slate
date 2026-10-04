import json, hashlib, sqlite3, threading
from datetime import datetime, timezone


class ImmutableAuditLog:
    """Append-only, hash-chained log. In prod, back with Postgres INSERT-only role."""
    def __init__(self, path: str = "audit.db"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL, agent TEXT, step TEXT,
                payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL
            )""")
        self._conn.commit()
        self._last_hash = self._fetch_last_hash()

    def _fetch_last_hash(self) -> str:
        row = self._conn.execute(
            "SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else "GENESIS"

    def log(self, agent: str, step: str, payload: dict) -> str:
        with self._lock:
            ts = datetime.now(timezone.utc).isoformat()
            body = json.dumps(
                {"ts": ts, "agent": agent, "step": step, "payload": payload},
                sort_keys=True, default=str)
            h = hashlib.sha256((self._last_hash + body).encode()).hexdigest()
            self._conn.execute(
                "INSERT INTO audit (ts, agent, step, payload, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?)",
                (ts, agent, step, body, self._last_hash, h))
            self._conn.commit()
            self._last_hash = h
            return h