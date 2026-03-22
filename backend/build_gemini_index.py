"""
Build FAISS index using Gemini-Embed-001 embeddings.

Reads chunked_documents.json (shared with MiniLM pipeline) and produces:
  - embeddings_gemini.npy
  - uchicago_ads_faiss_gemini.index

Does NOT modify any existing artifacts (embeddings.npy, uchicago_ads_faiss.index).

Usage:
  cd backend
  python build_gemini_index.py
"""

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-embedding-001"
BATCH_SIZE = 100  # API limit: 250 per request, 100 is safe


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts via Gemini Embedding API in batches."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = client.models.embed_content(model=MODEL_NAME, contents=batch)
        all_embeddings.extend([e.values for e in result.embeddings])
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} texts embedded")
    return np.array(all_embeddings, dtype="float32")


def main():
    # Load chunks
    with open("chunked_documents.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    # Encode
    print(f"Encoding with {MODEL_NAME}...")
    vecs = encode_texts(texts)

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs /= norms

    # Save embeddings
    np.save("embeddings_gemini.npy", vecs)
    print(f"Saved embeddings_gemini.npy  shape={vecs.shape}")

    # Build and save FAISS index
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, "uchicago_ads_faiss_gemini.index")
    print(f"Saved uchicago_ads_faiss_gemini.index  n={index.ntotal}")

    print("Done.")


if __name__ == "__main__":
    main()
