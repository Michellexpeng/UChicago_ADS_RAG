"""
UChicago Applied Data Science Q&A - FastAPI Backend

Usage:
  pip install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment Variables:
  GOOGLE_API_KEY=...
  EMBEDDING_MODEL=minilm|gemini  (default: minilm)
  USE_ONNX=true|false  (default: true — use ONNX Runtime for faster CPU inference)
"""

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from collections import Counter

from embedder import GoogleEmbedderWrapper
from retrieval import retrieve, rerank, is_list_query, _requested_count, tokenize_for_bm25
from prompt import translate_to_english, build_prompt, parse_citations

# ---------------------------------------------------------------------------
# Embedding model selection via env var
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "minilm").lower()
USE_ONNX = os.getenv("USE_ONNX", "true").lower() in ("true", "1", "yes")
USE_DEDUP = os.getenv("USE_DEDUP", "true").lower() in ("true", "1", "yes")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Global state (loaded once on startup)
# ---------------------------------------------------------------------------
embedder: Optional[SentenceTransformer] = None
cross_encoder: Optional[CrossEncoder] = None
chunk_records: list = []
faiss_index: Optional[faiss.IndexFlatIP] = None
bm25: Optional[BM25Okapi] = None
llm = None


# ---------------------------------------------------------------------------
# Lifespan: load models and data once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, cross_encoder, chunk_records, faiss_index, bm25, llm

    if EMBEDDING_MODEL == "gemini":
        print("Loading embedding model: Gemini-Embed-001 (API)...")
        embedder = GoogleEmbedderWrapper()
    elif USE_ONNX:
        from onnx_models import OnnxEmbedder, OnnxCrossEncoder
        print("Loading embedding model: MiniLM-L6-v2 (ONNX)...")
        embedder = OnnxEmbedder()
        print("Loading cross-encoder (ONNX)...")
        cross_encoder = OnnxCrossEncoder()
    else:
        print("Loading embedding model: MiniLM-L6-v2 (PyTorch)...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if cross_encoder is None:
        print("Loading cross-encoder (PyTorch)...")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    dedup_suffix = "_dedup" if USE_DEDUP else ""
    chunks_file = f"chunked_documents{dedup_suffix}.json"
    print(f"Loading chunk records from {chunks_file}...")
    with open(os.path.join(DATA_DIR, chunks_file), "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    gemini = EMBEDDING_MODEL == "gemini"
    emb_file = f"embeddings{'_gemini' if gemini else ''}{dedup_suffix}.npy"
    idx_file = f"uchicago_ads_faiss{'_gemini' if gemini else ''}{dedup_suffix}.index"
    emb_path = os.path.join(DATA_DIR, emb_file)
    idx_path = os.path.join(DATA_DIR, idx_file)

    if os.path.exists(idx_path):
        print(f"Loading pre-built FAISS index: {idx_file}")
        faiss_index = faiss.read_index(idx_path)
    else:
        print(f"Building FAISS index from {emb_file}...")
        vecs = np.load(emb_path).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        faiss_index.add(vecs)
        del vecs

    print("Building BM25 index...")
    tokenized = [tokenize_for_bm25(c["text"]) for c in chunk_records]
    bm25 = BM25Okapi(tokenized)

    print("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_output_tokens=512,
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    )

    print(f"✅ All resources loaded. Embedding={EMBEDDING_MODEL}. Server ready.")
    yield
    print("Shutting down...")


app = FastAPI(title="UChicago ADS Q&A API", lifespan=lifespan)

_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",")] if _origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


class SourceReference(BaseModel):
    url: str
    title: str = ""
    snippet: str


# ---------------------------------------------------------------------------
# Page chunk replacement
# ---------------------------------------------------------------------------
def _inject_page_chunks(hits: list, chunk_records: list, threshold: int = 3) -> list:
    """If 3+ hits come from the same page, replace them with the page-level chunk."""
    page_counts = Counter(h["page_title"] for h in hits if h.get("page_title"))

    page_chunks = {}
    for c in chunk_records:
        if c["metadata"].get("chunk_type") == "page":
            pt = c["metadata"].get("page_title", "")
            if pt:
                page_chunks[pt] = c

    replaced_pages = set()
    for page_title, count in page_counts.items():
        if count >= threshold and page_title in page_chunks:
            replaced_pages.add(page_title)

    if not replaced_pages:
        return hits

    result = []
    for pt in replaced_pages:
        pc = page_chunks[pt]
        meta = pc["metadata"]
        url = meta.get("source_urls", [meta.get("source_url", "")])[0] if "source_urls" in meta else meta.get("source_url", "")
        result.append({
            "chunk_id": pc["chunk_id"],
            "text": pc["text"],
            "url": url,
            "page_title": meta.get("page_title", ""),
            "labels": meta.get("labels", []),
            "score": 1.0,
        })

    for h in hits:
        if h.get("page_title") not in replaced_pages:
            result.append(h)

    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/ask")
async def ask(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    # Translate non-English queries for retrieval (BM25 + cross-encoder need English)
    en_query = await translate_to_english(body.question, llm)
    t1 = time.time()

    listing = is_list_query(en_query)
    retrieve_k = 15
    rerank_k = max(5, _requested_count(en_query)) if listing else 5
    candidates = retrieve(en_query, chunk_records, embedder, bm25, faiss_index, top_k=retrieve_k)
    t2 = time.time()

    hits = rerank(en_query, candidates, cross_encoder, top_k=rerank_k)
    t3 = time.time()

    hits = _inject_page_chunks(hits, chunk_records)
    # Use original query in prompt so LLM answers in the user's language
    prompt = build_prompt(body.question, hits)

    print(f"⏱ translate={t1-t0:.3f}s  retrieve={t2-t1:.3f}s  rerank={t3-t2:.3f}s  total_before_llm={t3-t0:.3f}s")

    async def stream():
        full_answer = ""
        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        clean_answer, sources, references = parse_citations(full_answer, hits)

        if clean_answer != full_answer.strip():
            yield f"data: {json.dumps({'type': 'replace', 'content': clean_answer})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'references': references})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_loaded": len(chunk_records)}
