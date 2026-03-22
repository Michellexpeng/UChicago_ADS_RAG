"""
Google Gemini Embedding wrapper with SentenceTransformer-compatible interface.
"""

import os

import numpy as np
from google import genai


class GoogleEmbedderWrapper:
    """Wraps Google Gemini Embedding API to match SentenceTransformer.encode() interface."""

    def __init__(self, model_name: str = "gemini-embedding-001", batch_size: int = 100):
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        bs = batch_size or self.batch_size
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            result = self._client.models.embed_content(
                model=self.model_name, contents=batch
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        vecs = np.array(all_embeddings, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs /= norms
        return vecs
