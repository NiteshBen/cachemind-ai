"""High-performance, 100% offline local embedding provider using subword n-gram feature hashing."""

import hashlib
import re
from typing import List, Set
import numpy as np

from cachemind.embeddings.base import BaseEmbeddingProvider

DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "please", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


class LocalTFIDFEmbedder(BaseEmbeddingProvider):
    """Deterministic, zero-dependency subword & word n-gram feature hashing embedder.

    Generates L2-normalized dense vector representations in sub-millisecond execution time,
    capturing both lexical syntax and semantic intent without downloading external model weights.
    """

    def __init__(self, dim: int = 384, ngram_range: tuple = (3, 5)):
        self._dim = dim
        self._min_n, self._max_n = ngram_range

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str, seed: int = 0) -> int:
        """Deterministic hash of token string into vector index."""
        h = hashlib.md5(f"{seed}:{token}".encode("utf-8")).digest()
        val = int.from_bytes(h[:4], byteorder="little", signed=False)
        return val % self._dim

    def _tokenize_words(self, text: str) -> List[str]:
        """Lowercases and extracts clean alphanumeric word tokens."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if t.strip()]
        return tokens

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single text into an L2-normalized vector."""
        if not text or not text.strip():
            return [0.0] * self._dim

        vec = np.zeros(self._dim, dtype=np.float32)
        words = self._tokenize_words(text)

        # Word level hashing with stopword damping & word-level subwords
        for i, word in enumerate(words):
            is_stop = word in DEFAULT_STOPWORDS
            weight = 0.15 if is_stop else 1.2
            idx = self._hash_token(word, seed=42)
            vec[idx] += weight

            # Bigram with next word
            if i + 1 < len(words):
                next_word = words[i + 1]
                bigram = f"{word}_{next_word}"
                b_idx = self._hash_token(bigram, seed=77)
                vec[b_idx] += 0.6

            # Word-level character subwords
            if len(word) >= 3 and not is_stop:
                for n in range(self._min_n, min(self._max_n + 1, len(word) + 1)):
                    for s_i in range(len(word) - n + 1):
                        sub = word[s_i : s_i + n]
                        s_idx = self._hash_token(sub, seed=101)
                        vec[s_idx] += 0.5

        # Sublinear scaling: 1 + log(count)
        positive_mask = vec > 0
        vec[positive_mask] = 1.0 + np.log(vec[positive_mask])

        # L2 unit normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self._dim, dtype=np.float32)

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts sequentially."""
        return [self.embed_query(t) for t in texts]
