"""
ONNX Runtime wrappers for MiniLM embedder and cross-encoder.

Drop-in replacements for SentenceTransformer.encode() and CrossEncoder.predict()
so that main.py and retrieval.py need minimal changes.
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class OnnxEmbedder:
    """Drop-in replacement for SentenceTransformer with .encode() interface."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODELS_DIR, "minilm_tokenizer")
        )
        self.session = ort.InferenceSession(
            os.path.join(MODELS_DIR, "minilm_embedder.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, sentences, normalize_embeddings=False, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        hidden = self.session.run(None, {k: v for k, v in inputs.items()})[0]
        # Mean pooling
        mask = inputs["attention_mask"][..., None]
        embeddings = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        return embeddings.astype("float32")


class OnnxCrossEncoder:
    """Drop-in replacement for CrossEncoder with .predict() interface."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODELS_DIR, "cross_encoder_tokenizer")
        )
        self.session = ort.InferenceSession(
            os.path.join(MODELS_DIR, "cross_encoder.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, pairs):
        queries = [p[0] for p in pairs]
        documents = [p[1] for p in pairs]
        inputs = self.tokenizer(
            queries, documents, padding=True, truncation=True, max_length=256, return_tensors="np"
        )
        logits = self.session.run(None, {k: v for k, v in inputs.items()})[0]
        return logits[:, 0]
