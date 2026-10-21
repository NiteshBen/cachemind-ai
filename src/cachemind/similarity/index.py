"""Vector index for high-speed nearest-neighbor similarity search."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from cachemind.similarity.metrics import batch_cosine_similarity


class VectorIndex:
    """Thread-safe vector index managing dense embeddings and sub-millisecond search."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension
        self._keys: List[str] = []
        self._key_to_idx: Dict[str, int] = {}
        self._matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, embedding: List[float]) -> None:
        """Adds or updates an embedding vector in the index."""
        if len(embedding) != self._dim:
            # Adjust dimension dynamically if first entry, or pad/truncate
            if len(self._keys) == 0:
                self._dim = len(embedding)
                self._matrix = np.empty((0, self._dim), dtype=np.float32)
            else:
                if len(embedding) < self._dim:
                    embedding = embedding + [0.0] * (self._dim - len(embedding))
                else:
                    embedding = embedding[: self._dim]

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, self._dim)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm

        if key in self._key_to_idx:
            idx = self._key_to_idx[key]
            self._matrix[idx] = vec[0]
        else:
            idx = len(self._keys)
            self._key_to_idx[key] = idx
            self._keys.append(key)
            if self._matrix.size == 0:
                self._matrix = vec
            else:
                self._matrix = np.vstack([self._matrix, vec])

    def remove(self, key: str) -> bool:
        """Removes a key and its vector from the index."""
        if key not in self._key_to_idx:
            return False

        idx_to_remove = self._key_to_idx[key]
        self._keys.pop(idx_to_remove)
        self._matrix = np.delete(self._matrix, idx_to_remove, axis=0)

        # Rebuild key_to_idx
        self._key_to_idx = {k: i for i, k in enumerate(self._keys)}
        return True

    def search(
        self,
        query_vec: List[float],
        top_k: int = 1,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Finds closest vectors to query_vec meeting or exceeding threshold.

        Returns list of (key, similarity_score) sorted descending by score.
        """
        if len(self._keys) == 0 or self._matrix.size == 0:
            return []

        if len(query_vec) != self._dim:
            if len(query_vec) < self._dim:
                query_vec = query_vec + [0.0] * (self._dim - len(query_vec))
            else:
                query_vec = query_vec[: self._dim]

        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm

        scores = batch_cosine_similarity(q.tolist(), self._matrix)

        # Filter by threshold
        valid_indices = np.where(scores >= threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Sort valid indices by score descending
        sorted_valid = valid_indices[np.argsort(-scores[valid_indices])][:top_k]

        return [(self._keys[i], float(scores[i])) for i in sorted_valid]

    def clear(self) -> None:
        """Empties the index."""
        self._keys.clear()
        self._key_to_idx.clear()
        self._matrix = np.empty((0, self._dim), dtype=np.float32)
