"""Abstract base interface for embedding providers in CacheMind-AI."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface defining the contract for text vectorization."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The output vector dimensionality."""
        pass
