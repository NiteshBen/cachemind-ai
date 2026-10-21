"""Base storage interface for CacheMind-AI cache persistence."""

from abc import ABC, abstractmethod
from typing import List, Optional

from cachemind.config import CacheEntry


class BaseStorage(ABC):
    """Abstract storage provider managing CacheEntry records."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Fetch entry by key."""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Store or update entry."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry by key."""
        pass

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all entries or only entries within a specific namespace."""
        pass

    @abstractmethod
    def all_entries(self, namespace: Optional[str] = None) -> List[CacheEntry]:
        """Retrieve all active entries."""
        pass

    @abstractmethod
    def count(self, namespace: Optional[str] = None) -> int:
        """Get total number of cached entries."""
        pass

    @abstractmethod
    def prune_expired(self) -> int:
        """Removes all expired entries according to TTL. Returns count of purged items."""
        pass
