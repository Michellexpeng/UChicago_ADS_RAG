"""
Build FAISS index using Gemini-Embed-001 embeddings.

Reads data/chunked_documents.json and produces:
  - data/embeddings_gemini.npy
  - data/uchicago_ads_faiss_gemini.index

Does NOT modify any existing artifacts (embeddings.npy, uchicago_ads_faiss.index).

Usage:
  cd backend
  python scripts/build_gemini_index.py
"""

import json
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv

# Allow imports from backend/ (parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from embedder import GoogleEmbedderWrapper

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    # Load chunks
    chunks_path = os.path.join(DATA_DIR, "chunked_documents.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    # Encode using shared GoogleEmbedderWrapper
    embedder = GoogleEmbedderWrapper()
    print(f"Encoding with {embedder.model_name}...")
    vecs = embedder.encode(texts, normalize_embeddings=True)
    print(f"Encoded {vecs.shape[0]} texts, dim={vecs.shape[1]}")

    # Save embeddings
    emb_path = os.path.join(DATA_DIR, "embeddings_gemini.npy")
    np.save(emb_path, vecs)
    print(f"Saved {emb_path}  shape={vecs.shape}")

    # Build and save FAISS index
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    idx_path = os.path.join(DATA_DIR, "uchicago_ads_faiss_gemini.index")
    faiss.write_index(index, idx_path)
    print(f"Saved {idx_path}  n={index.ntotal}")

    print("Done.")


if __name__ == "__main__":
    main()
