"""
Hybrid retrieval (BM25 + semantic FAISS) with RRF fusion, synonym expansion, and cross-encoder reranking.
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# List / enumeration detection
# ---------------------------------------------------------------------------
_LIST_RE = re.compile(
    r"\b(\d+)\s*(examples?|instances?|cases?|capstones?|projects?|topics?|courses?|ways?|tips?|ideas?)\b"
    r"|\b(list|give me|show me|what are|name)\b",
    re.IGNORECASE,
)


def _requested_count(query: str) -> int:
    """Return the explicit count requested (e.g. 'give me 5 examples' -> 5), else 0."""
    m = re.search(r"\b(\d+)\s*(examples?|instances?|cases?|capstones?|projects?)", query, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def is_list_query(query: str) -> bool:
    return bool(_LIST_RE.search(query))


# ---------------------------------------------------------------------------
# Synonym expansion for BM25
# ---------------------------------------------------------------------------
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
    "duration":             ["how long", "length", "time to complete", "quarters", "years"],
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

_REVERSE_SYNONYMS: dict[str, str] = {}
for _key, _vals in _SYNONYMS.items():
    for _v in _vals:
        _REVERSE_SYNONYMS[_v] = _key


def _phrase_matches(phrase: str, q_lower: str) -> bool:
    """Match a phrase if all its words appear in the query (order-independent)."""
    q_tokens = set(re.sub(r"[^a-z0-9\s]", " ", q_lower).split())
    words = phrase.split()
    if len(words) == 1:
        return words[0] in q_tokens
    return all(w in q_tokens for w in words)


def expand_query(query: str) -> str:
    """Bidirectional synonym expansion for BM25."""
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


def _is_dup(a: str, b: str, thresh: float = 0.95) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb)) > thresh


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, chunk_records: list, embedder, bm25, faiss_index,
             top_k: int = 10, dup_thresh: float = 0.95) -> list:
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
        if any(_is_dup(text, s, dup_thresh) for s in seen_texts):
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


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------
def rerank(question: str, hits: list, cross_encoder, top_k: int = 5) -> list:
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
