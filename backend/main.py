"""
UChicago Applied Data Science Q&A - FastAPI Backend
Converts the Jupyter Notebook RAG pipeline into a production API server.

Usage:
  pip install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment Variables:
  GOOGLE_API_KEY=...
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # reads backend/.env automatically

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Embedding model selection via env var: "gemini" or "minilm" (default)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "minilm").lower()


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
    else:
        print("Loading embedding model: MiniLM-L6-v2 (local)...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading cross-encoder...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Loading chunk records...")
    with open("chunked_documents.json", "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    emb_file = "embeddings_gemini.npy" if EMBEDDING_MODEL == "gemini" else "embeddings.npy"
    idx_file = "uchicago_ads_faiss_gemini.index" if EMBEDDING_MODEL == "gemini" else "uchicago_ads_faiss.index"

    if os.path.exists(idx_file):
        print(f"Loading pre-built FAISS index: {idx_file}")
        faiss_index = faiss.read_index(idx_file)
    else:
        print(f"Building FAISS index from {emb_file}...")
        vecs = np.load(emb_file).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        faiss_index.add(vecs)
        del vecs

    print("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in chunk_records]
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

# Allow all origins in dev; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# RAG helpers (ported from notebook)
# ---------------------------------------------------------------------------
_DONT_KNOW_RE = re.compile(
    r"(不知道|无法回答|没有.*信息|don'?t know|no information|cannot answer|not sure)",
    re.IGNORECASE,
)

_LIST_RE = re.compile(
    r"\b(\d+)\s*(examples?|instances?|cases?|capstones?|projects?|topics?|courses?|ways?|tips?|ideas?)\b"
    r"|\b(list|give me|show me|what are|name)\b",
    re.IGNORECASE,
)

def _requested_count(query: str) -> int:
    """Return the explicit count requested (e.g. 'give me 5 examples' → 5), else 0."""
    m = re.search(r"\b(\d+)\s*(examples?|instances?|cases?|capstones?|projects?)", query, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def is_list_query(query: str) -> bool:
    return bool(_LIST_RE.search(query))


_SYNONYMS = {
    "tuition":              ["cost", "fee", "price", "expense", "how much"],
    "professor":            ["faculty", "instructor", "teacher"],
    "class":                ["course", "subject"],
    "job":                  ["career", "employment", "work", "hire", "company", "role"],
    "career outcomes":      ["hire", "company", "employer", "job placement", "work at"],
    "salary":               ["income", "pay", "compensation", "earning"],
    "apply":                ["application", "admission", "enroll"],
    "admission":            ["application", "how to apply", "eligibility", "requirement"],
    "scholarship":          ["financial aid", "funding", "grant", "award"],
    "online":               ["remote", "virtual", "distance"],
    "deadline":             ["due date", "cutoff"],
    "gpa":                  ["grade", "academic requirement"],
    "visa":                 ["immigration", "sponsorship", "opt", "stem"],
    "capstone":             ["project", "thesis", "final project"],
    "curriculum":           ["core courses", "required courses", "elective courses", "course list"],
    "specialization":       ["elective courses", "concentration", "focus area", "track"],
    "alumni":               ["graduate", "graduate outcome", "placement"],
    "core courses":         ["required courses", "mandatory classes"],
    "foundational courses": ["before join", "resources before joining", "preparation materials", "pre-course resources"],
}

# Build reverse index: synonym value → canonical key
_REVERSE_SYNONYMS: dict[str, str] = {}
for _key, _vals in _SYNONYMS.items():
    for _v in _vals:
        _REVERSE_SYNONYMS[_v] = _key

def _phrase_matches(phrase: str, q_lower: str) -> bool:
    """Match a phrase if all its words appear in the query (order-independent).
    Strips punctuation before tokenizing to handle trailing '?' etc.
    """
    q_tokens = set(re.sub(r"[^a-z0-9\s]", " ", q_lower).split())
    words = phrase.split()
    if len(words) == 1:
        return words[0] in q_tokens
    return all(w in q_tokens for w in words)


def expand_query(query: str) -> str:
    """Bidirectional synonym expansion for BM25.
    - Forward:  "foundational courses" → adds values (preparation materials, ...)
    - Backward: "resources before joining" → adds canonical key (foundational courses)
    Uses bag-of-words matching so inserted words don't break multi-word phrases.
    """
    q_lower = query.lower()
    extra = []
    all_phrases = sorted(
        list(_SYNONYMS.keys()) + list(_REVERSE_SYNONYMS.keys()),
        key=len, reverse=True
    )
    for phrase in all_phrases:
        if _phrase_matches(phrase, q_lower):
            if phrase in _SYNONYMS:
                extra.extend(_SYNONYMS[phrase])
            else:
                extra.append(_REVERSE_SYNONYMS[phrase])
    return query + " " + " ".join(extra) if extra else query


def is_dup(a: str, b: str, thresh: float = 0.95) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb)) > thresh


def retrieve(query: str, top_k: int = 10, dup_thresh: float = 0.95) -> list:
    """Hybrid retrieval: BM25 + semantic (FAISS) fused with RRF over all chunks.

    For list/enumeration queries the per-URL cap is tightened to 1 and
    top_k is expanded so the reranker has a diverse candidate pool.
    """
    listing = is_list_query(query)
    url_cap = 1 if listing else 3

    all_idxs = list(range(len(chunk_records)))
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    bm25_scores = bm25.get_scores(expand_query(query).lower().split())

    # Semantic scores over all chunks
    sub_vecs = np.stack([faiss_index.reconstruct(i) for i in all_idxs])
    sem_scores = sub_vecs @ q_emb.astype("float32")

    # RRF fusion (k=60)
    k = 60
    sem_ranks = np.argsort(-sem_scores)
    bm25_ranks = np.argsort(-bm25_scores)
    rrf = np.zeros(len(all_idxs))
    for rank, rel in enumerate(sem_ranks):
        rrf[rel] += 1.0 / (k + rank + 1)
    for rank, rel in enumerate(bm25_ranks):
        rrf[rel] += 1.0 / (k + rank + 1)

    # Heading boost: boost chunks whose heading overlaps with query keywords
    q_tokens = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
    q_tokens.update(expand_query(query).lower().split())
    for idx in all_idxs:
        heading = (chunk_records[idx]["metadata"].get("heading") or "").lower()
        if not heading:
            continue
        h_tokens = set(heading.split())
        overlap = len(q_tokens & h_tokens) / max(1, len(h_tokens))
        if overlap >= 0.5:
            ctype = chunk_records[idx]["metadata"].get("chunk_type", "")
            multiplier = 2.0 if ctype in ("accordion", "page") else 1.0
            rrf[idx] += overlap * 0.02 * multiplier

    order = np.argsort(-rrf)
    hits = []
    seen_texts = []
    url_count = {}

    for rel in order:
        if len(hits) >= top_k:
            break
        text = chunk_records[rel]["text"]
        if any(is_dup(text, s, dup_thresh) for s in seen_texts):
            continue
        url = chunk_records[rel]["metadata"]["source_url"]
        url_count.setdefault(url, 0)
        if url_count[url] >= url_cap:
            continue
        url_count[url] += 1
        seen_texts.append(text)
        meta = chunk_records[rel]["metadata"]
        hits.append({
            "chunk_id": chunk_records[rel]["chunk_id"],
            "text": text,
            "url": url,
            "page_title": meta.get("page_title", ""),
            "labels": meta.get("labels", []),
            "score": float(rrf[rel]),
        })

    return hits


def rerank(question: str, hits: list, top_k: int = 5) -> list:
    """Re-score (question, chunk) pairs with cross-encoder and return top_k.

    For list queries, return at least as many hits as the requested count.
    """
    n = max(top_k, _requested_count(question))
    pairs = [(question, h["text"]) for h in hits]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: -x[0])
    result = [h for _, h in ranked[:n]]
    # Preserve retrieval rank 1 (highest RRF) even if cross-encoder drops it
    if hits and hits[0] not in result:
        result.insert(0, hits[0])
    return result


def build_prompt(question: str, hits: list) -> str:
    listing = is_list_query(question)
    instruction = (
        "List each item clearly and separately."
        if listing
        else "Answer concisely and directly."
    )
    context = "\n\n".join(f"[Doc{i+1}] {h['text']}" for i, h in enumerate(hits))
    return (
        "You are a helpful assistant for University of Chicago's MS in Applied Data Science program.\n"
        "Use the provided context below to answer the user question. "
        f"{instruction} "
        "IMPORTANT: Do NOT start with disclaimers or statements about what the context lacks. "
        "If the context contains relevant information -- even under different terminology -- "
        "treat it as a direct answer and synthesize ALL relevant details. "
        "Only say you don't know if the context is completely unrelated. "
        "Format your answer in Markdown.\n"
        "At the very end of your answer, on a new line, list ONLY the document labels you "
        "actually used (e.g. [Doc1][Doc3]). Do not include documents you did not reference. "
        "If you cannot answer the question from the context, do NOT cite any documents.\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{question}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/ask")
async def ask(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    listing = is_list_query(body.question)
    retrieve_k = 20 if listing else 10
    rerank_k = max(5, _requested_count(body.question)) if listing else 5
    candidates = retrieve(body.question, top_k=retrieve_k)
    hits = rerank(body.question, candidates, top_k=rerank_k)
    prompt = build_prompt(body.question, hits)

    # Build doc-number → hit mapping for citation parsing
    doc_map: dict[int, dict] = {i + 1: h for i, h in enumerate(hits)}

    async def stream():
        full_answer = ""
        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Parse cited doc labels from the answer (e.g. [Doc1], [Doc3])
        cited_nums = set(int(m) for m in re.findall(r"\[Doc(\d+)\]", full_answer))

        # Build sources from only the cited docs, deduped by URL
        seen: set[str] = set()
        sources: list[str] = []
        references: list[dict] = []
        for num in sorted(cited_nums):
            h = doc_map.get(num)
            if not h or h["url"] in seen:
                continue
            seen.add(h["url"])
            sources.append(h["url"])
            references.append({"url": h["url"], "title": h.get("page_title", ""), "snippet": h["text"]})

        # Strip citation markers from the displayed answer
        clean_answer = re.sub(r"\s*\[Doc\d+\]", "", full_answer).strip()
        if clean_answer != full_answer:
            yield f"data: {json.dumps({'type': 'replace', 'content': clean_answer})}\n\n"

        # Suppress sources for "I don't know" style answers
        if _DONT_KNOW_RE.search(clean_answer) and len(clean_answer) < 200:
            sources.clear()
            references.clear()

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'references': references})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_loaded": len(chunk_records)}
