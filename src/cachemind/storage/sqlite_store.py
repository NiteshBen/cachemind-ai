"""Persistent SQLite cache storage with WAL mode, indexing, and eviction management."""

import json
import sqlite3
import threading
import time
from typing import List, Literal, Optional

from cachemind.config import CacheEntry
from cachemind.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """File-backed SQLite storage engine for durable semantic caching."""

    def __init__(
        self,
        db_path: str = "cachemind_store.db",
        max_entries: int = 1000,
        eviction_policy: Literal["lru", "lfu", "fifo"] = "lru",
    ):
        self._db_path = db_path
        self._max_entries = max_entries
        self._eviction_policy = eviction_policy
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high performance concurrent reads
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        access_count INTEGER NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        ttl_seconds REAL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON cache_entries(namespace);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON cache_entries(last_accessed);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache_entries(created_at);")

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            key=row["key"],
            namespace=row["namespace"],
            prompt=row["prompt"],
            response=row["response"],
            embedding=json.loads(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            ttl_seconds=row["ttl_seconds"],
        )

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    return None

                entry = self._row_to_entry(row)
                if entry.is_expired():
                    cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return None

                now = time.time()
                cursor.execute(
                    "UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
                    (now, key),
                )
                conn.commit()
                entry.last_accessed = now
                entry.access_count += 1
                return entry

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check capacity
                cursor.execute("SELECT COUNT(*) as cnt FROM cache_entries")
                cnt = cursor.fetchone()["cnt"]

                cursor.execute("SELECT 1 FROM cache_entries WHERE key = ?", (entry.key,))
                exists = cursor.fetchone() is not None

                if cnt >= self._max_entries and not exists:
                    # Evict one
                    self._evict_one(conn)

                cursor.execute(
                    """
                    INSERT INTO cache_entries (
                        key, namespace, prompt, response, embedding, metadata,
                        created_at, last_accessed, access_count, prompt_tokens,
                        completion_tokens, ttl_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        namespace = excluded.namespace,
                        prompt = excluded.prompt,
                        response = excluded.response,
                        embedding = excluded.embedding,
                        metadata = excluded.metadata,
                        last_accessed = excluded.last_accessed,
                        access_count = excluded.access_count,
                        prompt_tokens = excluded.prompt_tokens,
                        completion_tokens = excluded.completion_tokens,
                        ttl_seconds = excluded.ttl_seconds;
                    """,
                    (
                        entry.key,
                        entry.namespace,
                        entry.prompt,
                        entry.response,
                        json.dumps(entry.embedding),
                        json.dumps(entry.metadata),
                        entry.created_at,
                        entry.last_accessed,
                        entry.access_count,
                        entry.prompt_tokens,
                        entry.completion_tokens,
                        entry.ttl_seconds,
                    ),
                )
                conn.commit()

    def _evict_one(self, conn: sqlite3.Connection) -> None:
        """Evicts a single row based on policy."""
        now = time.time()
        # Drop expired first if any
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key FROM cache_entries WHERE ttl_seconds IS NOT NULL AND (? - created_at) > ttl_seconds LIMIT 1",
            (now,),
        )
        expired_row = cursor.fetchone()
        if expired_row:
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (expired_row["key"],))
            return

        order_col = "last_accessed"
        if self._eviction_policy == "lfu":
            order_col = "access_count"
        elif self._eviction_policy == "fifo":
            order_col = "created_at"

        cursor.execute(f"SELECT key FROM cache_entries ORDER BY {order_col} ASC LIMIT 1")
        target = cursor.fetchone()
        if target:
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (target["key"],))

    def delete(self, key: str) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if namespace is None:
                    cursor.execute("DELETE FROM cache_entries")
                else:
                    cursor.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
                conn.commit()

    def all_entries(self, namespace: Optional[str] = None) -> List[CacheEntry]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                # Purge expired
                cursor.execute(
                    "DELETE FROM cache_entries WHERE ttl_seconds IS NOT NULL AND (? - created_at) > ttl_seconds",
                    (now,),
                )
                conn.commit()

                if namespace is None:
                    cursor.execute("SELECT * FROM cache_entries")
                else:
                    cursor.execute("SELECT * FROM cache_entries WHERE namespace = ?", (namespace,))
                rows = cursor.fetchall()
                return [self._row_to_entry(r) for r in rows]

    def count(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                cursor.execute(
                    "DELETE FROM cache_entries WHERE ttl_seconds IS NOT NULL AND (? - created_at) > ttl_seconds",
                    (now,),
                )
                conn.commit()
                if namespace is None:
                    cursor.execute("SELECT COUNT(*) as cnt FROM cache_entries")
                else:
                    cursor.execute("SELECT COUNT(*) as cnt FROM cache_entries WHERE namespace = ?", (namespace,))
                return cursor.fetchone()["cnt"]

    def prune_expired(self) -> int:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                cursor.execute(
                    "DELETE FROM cache_entries WHERE ttl_seconds IS NOT NULL AND (? - created_at) > ttl_seconds",
                    (now,),
                )
                conn.commit()
                return cursor.rowcount
