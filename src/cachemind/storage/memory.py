"""Thread-safe In-Memory cache storage with LRU, LFU, and FIFO eviction strategies."""

import threading
import time
from typing import Dict, List, Literal, Optional

from cachemind.config import CacheEntry
from cachemind.storage.base import BaseStorage


class InMemoryStorage(BaseStorage):
    """High-speed in-memory store supporting eviction algorithms and TTL management."""

    def __init__(
        self,
        max_entries: int = 1000,
        eviction_policy: Literal["lru", "lfu", "fifo"] = "lru",
    ):
        self._max_entries = max_entries
        self._eviction_policy = eviction_policy
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            entry.touch()
            return entry

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            # If at or exceeding capacity and inserting a new key, evict one
            if len(self._store) >= self._max_entries and entry.key not in self._store:
                self._evict_one()
            self._store[entry.key] = entry

    def _evict_one(self) -> None:
        """Evicts a single entry based on configured eviction policy."""
        # First check if there is an expired entry to drop
        now = time.time()
        for k, v in list(self._store.items()):
            if v.is_expired(now):
                del self._store[k]
                return

        if not self._store:
            return

        if self._eviction_policy == "lru":
            # Least recently accessed
            target_key = min(self._store.keys(), key=lambda k: self._store[k].last_accessed)
        elif self._eviction_policy == "lfu":
            # Least frequently accessed
            target_key = min(self._store.keys(), key=lambda k: self._store[k].access_count)
        else:  # fifo
            # Oldest created
            target_key = min(self._store.keys(), key=lambda k: self._store[k].created_at)

        del self._store[target_key]

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            if namespace is None:
                self._store.clear()
            else:
                keys_to_del = [k for k, v in self._store.items() if v.namespace == namespace]
                for k in keys_to_del:
                    del self._store[k]

    def all_entries(self, namespace: Optional[str] = None) -> List[CacheEntry]:
        with self._lock:
            now = time.time()
            valid = []
            expired_keys = []
            for k, v in self._store.items():
                if v.is_expired(now):
                    expired_keys.append(k)
                elif namespace is None or v.namespace == namespace:
                    valid.append(v)
            for k in expired_keys:
                del self._store[k]
            return valid

    def count(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            return len(self.all_entries(namespace=namespace))

    def prune_expired(self) -> int:
        with self._lock:
            now = time.time()
            expired = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in expired:
                del self._store[k]
            return len(expired)
