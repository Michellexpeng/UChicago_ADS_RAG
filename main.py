"""
UChicago Applied Data Science Q&A - FastAPI Backend
Converts the Jupyter Notebook RAG pipeline into a production API server.

Usage:
  pip install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment Variables:
  OPENAI_API_KEY=sk-...
"""

import os
import json
import re
import numpy as np
from collections import Counter
from typing import Optional
from contextlib import asynccontextmanager

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOpenAI
from nltk.stem import WordNetLemmatizer
import nltk

# ---------------------------------------------------------------------------
# Download NLTK data (run once)
# ---------------------------------------------------------------------------
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------------------------------------------------------------------------
# Global state (loaded once on startup)
# ---------------------------------------------------------------------------
embedder: Optional[SentenceTransformer] = None
chunk_records: list = []
embeddings: Optional[np.ndarray] = None
llm = None
llm_fallback = None
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Keyword groups (same as notebook)
# ---------------------------------------------------------------------------
KEYWORD_GROUPS = {
    "admission": ["admission", "apply", "application", "enrollment"],
    "career": ["career", "job", "employment", "profession"],
    "capstone": ["capstone", "final project"],
    "fee": ["tuition", "cost", "fee", "price"],
    "course": ["course", "class", "curriculum", "track"],
    "deadline": ["deadline", " due ", "submission"],
    "scholarship": ["scholarship", "financial aid"],
    "english": ["toefl", "ielts", " gre ", "language requirement", "english score"],
    "visa": ["visa", "sponsorship", "international student"],
    "faculty": ["faculty", "instructor", "professor", "teacher", "staff", "scholar", "fellow", "people"],
    "research": ["research"],
    "contact": ["contact", "outreach", "network", "workshop"],
    "summer": ["summer"],
    "news": ["news", "event"],
    "program": ["program"],
}


# ---------------------------------------------------------------------------
# Lifespan: load models and data once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, chunk_records, embeddings, llm, llm_fallback

    print("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading chunk records...")
    with open("chunked_documents.json", "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    print("Loading embeddings...")
    embeddings = np.load("embeddings.npy")
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings[:] = embeddings / norms

    api_key = os.getenv("OPENAI_API_KEY", "")
    print("Initializing LLMs...")
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.0, max_tokens=512, openai_api_key=api_key)
    llm_fallback = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=512, openai_api_key=api_key)

    print("✅ All resources loaded. Server ready.")
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


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# RAG helpers (ported from notebook)
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(tok, pos="n") for tok in tokens]
    return " ".join(lemmas)


def extract_query_labels(query: str, keyword_groups: dict) -> list:
    query_norm = normalize(query)
    labels = []
    for group, syns in keyword_groups.items():
        if any(s.lower() in query_norm for s in syns):
            labels.append(group)
    return labels


def is_dup(a: str, b: str, thresh: float = 0.95) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb)) > thresh


def retrieve(query: str, top_k: int = 5, dup_thresh: float = 0.95) -> list:
    """Label-first FAISS-style retrieval (using pre-loaded numpy embeddings)."""
    primary_labels = ["contact", "career", "course", "admission", "summer",
                      "capstone", "fee", "scholarship", "deadline"]

    special_idxs = [
        i for i, c in enumerate(chunk_records)
        if "datascience.uchicago.edu/education/masters-programs/" in c["metadata"].get("source_url", "")
    ]

    qlabels = extract_query_labels(query, KEYWORD_GROUPS)
    plabels = [l for l in qlabels if l in primary_labels]
    slabels = [l for l in qlabels if l not in primary_labels]

    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    hits = []
    seen_texts = []
    url_count = {}

    def collect_from(idxs):
        if not idxs:
            return
        sub_embs = embeddings[idxs]
        sims = cosine_similarity(q_emb.reshape(1, -1), sub_embs)[0]
        order = np.argsort(-sims)
        for rel in order:
            if sims[rel] < 0.25:
                return
            real_i = idxs[rel]
            text = chunk_records[real_i]["text"]
            if any(is_dup(text, s, dup_thresh) for s in seen_texts):
                continue
            url = chunk_records[real_i]["metadata"]["source_url"]
            url_count.setdefault(url, 0)
            if url_count[url] >= 3:
                continue
            url_count[url] += 1
            seen_texts.append(text)
            meta = chunk_records[real_i]["metadata"]
            hits.append({
                "chunk_id": chunk_records[real_i]["chunk_id"],
                "text": text,
                "url": url,
                "labels": meta.get("labels", []),
                "score": float(sims[rel]),
            })
            if len(hits) >= top_k:
                return

    if plabels:
        collect_from([i for i in special_idxs if any(l in chunk_records[i]["metadata"].get("labels", []) for l in plabels)])
    if len(hits) < top_k and slabels:
        collect_from([i for i in special_idxs if any(l in chunk_records[i]["metadata"].get("labels", []) for l in slabels)])
    if len(hits) < top_k and plabels:
        collect_from([i for i, c in enumerate(chunk_records) if any(l in c["metadata"].get("labels", []) for l in plabels)])
    if len(hits) < top_k and slabels:
        collect_from([i for i, c in enumerate(chunk_records) if any(l in c["metadata"].get("labels", []) for l in slabels)])
    if len(hits) < top_k:
        collect_from(list(range(len(chunk_records))))

    return hits[:top_k]


def generate_answer(question: str, hits: list) -> str:
    context = "\n\n".join(f"[Doc] {h['text']}" for h in hits)
    prompt = (
        "You are a helpful assistant for University of Chicago's MS in Applied Data Science program.\n"
        "Use the provided context below to answer the user question. "
        "If the answer is not in the context, say:\n"
        '"I\'m sorry, I don\'t have enough information to answer that question.".\n\n'
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{question}\n\n"
        "Answer:"
    )
    response = llm.predict(prompt).strip()
    if response.startswith("I'm sorry, I don't have enough information"):
        response = llm_fallback.predict(prompt).strip()
    return response


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=QuestionResponse)
async def ask(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    hits = retrieve(body.question, top_k=5)
    answer = generate_answer(body.question, hits)

    # Deduplicate source URLs
    seen = set()
    sources = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            sources.append(h["url"])

    return QuestionResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_loaded": len(chunk_records)}
