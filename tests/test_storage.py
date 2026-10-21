"""Tests for InMemoryStorage and SQLiteStorage backends."""

import os
import time
import pytest
from cachemind.config import CacheEntry
from cachemind.storage.memory import InMemoryStorage
from cachemind.storage.sqlite_store import SQLiteStorage


def _make_dummy_entry(key: str, ns: str = "default", ttl: int = None, p_toks: int = 10, c_toks: int = 20) -> CacheEntry:
    return CacheEntry(
        key=key,
        namespace=ns,
        prompt=f"Prompt for {key}",
        response=f"Response for {key}",
        embedding=[0.1, 0.2, 0.3],
        metadata={"tag": "test"},
        prompt_tokens=p_toks,
        completion_tokens=c_toks,
        ttl_seconds=ttl,
    )


# ---------------- Memory Storage Tests ----------------

def test_memory_storage_crud():
    store = InMemoryStorage(max_entries=10)
    entry = _make_dummy_entry("k1")
    store.set(entry)

    assert store.count() == 1
    fetched = store.get("k1")
    assert fetched is not None
    assert fetched.key == "k1"
    assert fetched.access_count == 2  # 1 initial + 1 from get touch

    assert store.delete("k1") is True
    assert store.get("k1") is None
    assert store.count() == 0


def test_memory_storage_lru_eviction():
    store = InMemoryStorage(max_entries=3, eviction_policy="lru")
    e1 = _make_dummy_entry("k1")
    e2 = _make_dummy_entry("k2")
    e3 = _make_dummy_entry("k3")
    store.set(e1)
    store.set(e2)
    store.set(e3)

    # Touch k1 so k2 becomes the least recently used
    time.sleep(0.01)
    store.get("k1")
    time.sleep(0.01)
    store.get("k3")

    # Add 4th entry -> k2 should be evicted
    e4 = _make_dummy_entry("k4")
    store.set(e4)

    assert store.get("k2") is None
    assert store.get("k1") is not None
    assert store.get("k3") is not None
    assert store.get("k4") is not None


def test_memory_storage_ttl_expiration():
    store = InMemoryStorage(max_entries=10)
    entry = _make_dummy_entry("k_ttl", ttl=1)
    # Fake created_at in the past
    entry.created_at = time.time() - 2.0
    store.set(entry)

    assert store.get("k_ttl") is None
    assert store.count() == 0


def test_memory_storage_namespaces():
    store = InMemoryStorage(max_entries=10)
    store.set(_make_dummy_entry("k1", ns="billing"))
    store.set(_make_dummy_entry("k2", ns="support"))

    assert store.count(namespace="billing") == 1
    assert store.count(namespace="support") == 1
    assert store.count() == 2

    store.clear(namespace="billing")
    assert store.count(namespace="billing") == 0
    assert store.count(namespace="support") == 1


# ---------------- SQLite Storage Tests ----------------

def test_sqlite_storage_crud(tmp_path):
    db_file = str(tmp_path / "test_store.db")
    store = SQLiteStorage(db_path=db_file, max_entries=10)

    e1 = _make_dummy_entry("k1")
    store.set(e1)

    assert store.count() == 1
    fetched = store.get("k1")
    assert fetched is not None
    assert fetched.key == "k1"
    assert fetched.prompt == "Prompt for k1"
    assert fetched.access_count == 2

    # Verify persistence by instantiating new storage instance on same db
    store2 = SQLiteStorage(db_path=db_file, max_entries=10)
    assert store2.count() == 1
    assert store2.get("k1") is not None

    assert store.delete("k1") is True
    assert store.count() == 0


def test_sqlite_storage_lru_eviction(tmp_path):
    db_file = str(tmp_path / "test_evict.db")
    store = SQLiteStorage(db_path=db_file, max_entries=2, eviction_policy="lru")

    store.set(_make_dummy_entry("k1"))
    time.sleep(0.01)
    store.set(_make_dummy_entry("k2"))

    # Touch k1
    time.sleep(0.01)
    store.get("k1")

    # Add k3 -> k2 should be evicted
    store.set(_make_dummy_entry("k3"))

    assert store.get("k2") is None
    assert store.get("k1") is not None
    assert store.get("k3") is not None


def test_sqlite_storage_ttl_expiration(tmp_path):
    db_file = str(tmp_path / "test_ttl.db")
    store = SQLiteStorage(db_path=db_file, max_entries=10)

    entry = _make_dummy_entry("k_ttl", ttl=1)
    entry.created_at = time.time() - 5.0
    store.set(entry)

    # get should detect expiration and return None
    assert store.get("k_ttl") is None
    assert store.count() == 0
