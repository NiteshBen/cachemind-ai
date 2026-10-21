"""Google Gemini embedding provider for CacheMind-AI with automatic local fallback."""

import os
from typing import List, Optional
import numpy as np

from cachemind.embeddings.base import BaseEmbeddingProvider
from cachemind.embeddings.local_tfidf import LocalTFIDFEmbedder


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Integrates Google GenAI Gemini text-embedding-004 for high-precision semantic representation.

    Gracefully falls back to LocalTFIDFEmbedder if GEMINI_API_KEY is not set or network is unreachable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-004",
        dimension: int = 768,
    ):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model_name = model_name
        self._dim = dimension
        self._fallback_embedder = LocalTFIDFEmbedder(dim=self._dim)
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self._api_key:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        except Exception:
            self._client = None

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query via Gemini, falling back to local embedder if offline."""
        if not self._client or not text or not text.strip():
            return self._fallback_embedder.embed_query(text)

        try:
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=text,
            )
            # Handle response structure
            if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
                vals = list(result.embedding.values)
                # Normalize L2
                arr = np.array(vals, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    arr = arr / norm
                return arr.tolist()
            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                vals = list(result.embeddings[0].values)
                arr = np.array(vals, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    arr = arr / norm
                return arr.tolist()
        except Exception:
            pass

        # Fallback to local offline embedder
        return self._fallback_embedder.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts."""
        return [self.embed_query(t) for t in texts]
