"""
Deduplicate chunks by content hash and rebuild embeddings + FAISS index.

Reads data/chunked_documents.json and produces:
  - data/chunked_documents_dedup.json
  - data/embeddings_dedup.npy          (MiniLM)
  - data/uchicago_ads_faiss_dedup.index (MiniLM)

Optionally (if GOOGLE_API_KEY is set and --gemini flag is used):
  - data/embeddings_gemini_dedup.npy
  - data/uchicago_ads_faiss_gemini_dedup.index

Does NOT modify any existing artifacts.

Usage:
  cd backend
  python scripts/deduplicate_chunks.py            # MiniLM only
  python scripts/deduplicate_chunks.py --gemini    # Also build Gemini index
"""

import argparse
import hashlib
import json
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MIN_TEXT_LENGTH = 50


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove exact-duplicate chunks (by text hash) and very short chunks.

    For duplicates, keep the first occurrence and merge source_urls from
    subsequent duplicates into a list on the canonical chunk.
    """
    seen: dict[str, int] = {}  # hash -> index in deduped list
    deduped: list[dict] = []
    stats = {"exact_dupes": 0, "too_short": 0}

    for chunk in chunks:
        text = chunk["text"]

        # Skip very short chunks
        if len(text.strip()) < MIN_TEXT_LENGTH:
            stats["too_short"] += 1
            continue

        h = hashlib.md5(text.encode("utf-8")).hexdigest()

        if h in seen:
            # Merge source_url into canonical chunk's source_urls
            canonical = deduped[seen[h]]
            dup_url = chunk["metadata"]["source_url"]
            if dup_url not in canonical["metadata"]["source_urls"]:
                canonical["metadata"]["source_urls"].append(dup_url)
            stats["exact_dupes"] += 1
        else:
            seen[h] = len(deduped)
            # Convert source_url to source_urls list
            meta = dict(chunk["metadata"])
            meta["source_urls"] = [meta.pop("source_url")]
            deduped.append({
                "text": text,
                "metadata": meta,
                "chunk_id": chunk["chunk_id"],
            })

    print(f"Deduplication stats:")
    print(f"  Original chunks:  {len(chunks)}")
    print(f"  Exact duplicates: {stats['exact_dupes']}")
    print(f"  Too short (<{MIN_TEXT_LENGTH} chars): {stats['too_short']}")
    print(f"  After dedup:      {len(deduped)}")
    return deduped


def build_minilm_index(texts: list[str]) -> tuple[np.ndarray, faiss.IndexFlatIP]:
    from sentence_transformers import SentenceTransformer

    print("Encoding with MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    vecs = vecs.astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return vecs, index


def build_gemini_index(texts: list[str]) -> tuple[np.ndarray, faiss.IndexFlatIP]:
    from embedder import GoogleEmbedderWrapper

    print("Encoding with Gemini-Embed-001...")
    embedder = GoogleEmbedderWrapper()
    vecs = embedder.encode(texts, normalize_embeddings=True)
    vecs = vecs.astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return vecs, index


def main():
    parser = argparse.ArgumentParser(description="Deduplicate chunks and rebuild index")
    parser.add_argument("--gemini", action="store_true", help="Also build Gemini embedding index")
    args = parser.parse_args()

    # Load original chunks
    src = os.path.join(DATA_DIR, "chunked_documents.json")
    with open(src, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Deduplicate
    deduped = deduplicate(chunks)
    texts = [c["text"] for c in deduped]

    # Save deduped chunks
    out_path = os.path.join(DATA_DIR, "chunked_documents_dedup.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")

    # Build MiniLM index
    vecs, index = build_minilm_index(texts)
    np.save(os.path.join(DATA_DIR, "embeddings_dedup.npy"), vecs)
    faiss.write_index(index, os.path.join(DATA_DIR, "uchicago_ads_faiss_dedup.index"))
    print(f"Saved MiniLM embeddings ({vecs.shape}) and FAISS index (n={index.ntotal})")

    # Optionally build Gemini index
    if args.gemini:
        vecs_g, index_g = build_gemini_index(texts)
        np.save(os.path.join(DATA_DIR, "embeddings_gemini_dedup.npy"), vecs_g)
        faiss.write_index(index_g, os.path.join(DATA_DIR, "uchicago_ads_faiss_gemini_dedup.index"))
        print(f"Saved Gemini embeddings ({vecs_g.shape}) and FAISS index (n={index_g.ntotal})")

    print("Done.")


if __name__ == "__main__":
    main()
