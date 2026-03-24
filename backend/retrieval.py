"""
Hybrid retrieval (BM25 + semantic FAISS) with RRF fusion, synonym expansion, and cross-encoder reranking.
"""

import re

import numpy as np


# ---------------------------------------------------------------------------
# BM25 tokenizer with simple stemming and stopword removal
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "of", "in", "to", "for", "with", "on",
    "at", "by", "from", "as", "into", "about", "between", "through",
    "and", "or", "but", "not", "no", "if", "so", "than",
})

def _simple_stem(word: str) -> str:
    """Conservative plural/suffix stripping for BM25. Only handles clear-cut cases."""
    if len(word) <= 4:
        return word
    # -ies -> -y (studies -> study, but not "series")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # -ses -> -se (courses -> course, analyses -> analyse)
    if word.endswith("ses") and len(word) > 4:
        return word[:-1]
    # -es -> -e (deadlines stays, but classes -> class handled by -es after consonant)
    # -s (simple plural, but not words ending in ss/us/is)
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "sis")):
        return word[:-1]
    return word


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text with lowercasing, stopword removal, and simple stemming."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [_simple_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1]

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
    "programming":          ["python", "R", "coding", "code"],
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


def _get_url(meta: dict) -> str:
    """Get primary URL from metadata, supporting both source_url and source_urls formats."""
    if "source_urls" in meta:
        return meta["source_urls"][0] if meta["source_urls"] else ""
    return meta.get("source_url", "")


def _is_dup(a: str, b: str, thresh: float = 0.95) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb)) > thresh


# ---------------------------------------------------------------------------
# Query intent → label detection (for label boost)
# ---------------------------------------------------------------------------
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "admission": ["admission", "admit", "apply", "application", "eligibility", "requirement",
                   "gre", "gmat", "toefl", "ielts", "transcript", "recommend"],
    "course": ["course", "class", "curriculum", "schedule", "quarter", "elective",
               "core", "foundational", "machine learning", "track", "progression"],
    "fee": ["tuition", "fee", "cost", "price", "scholarship", "financial aid",
            "funding", "expense", "payment"],
    "capstone": ["capstone", "project", "sponsor", "thesis", "research"],
    "career": ["career", "job", "employment", "hire", "salary", "outcome", "alumni",
               "graduate", "employer", "placement"],
    "application": ["deadline", "event", "date", "when to apply"],
    "contact": ["contact", "email", "phone", "advisor", "advising", "office"],
}


def _detect_query_labels(query: str) -> set[str]:
    """Detect which topic labels are relevant to a query."""
    q_lower = query.lower()
    matched = set()
    for label, keywords in _LABEL_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.add(label)
    return matched


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, chunk_records: list, embedder, bm25, faiss_index,
             top_k: int = 10, dup_thresh: float = 0.95) -> list:
    """Hybrid retrieval: BM25 + semantic (FAISS) fused with RRF over all chunks.

    Returns top_k candidates for the reranker. No URL cap is applied here
    so that relevant chunks from FAQ-heavy pages are not prematurely dropped.
    """
    n_chunks = len(chunk_records)
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    bm25_scores = bm25.get_scores(tokenize_for_bm25(expand_query(query)))

    # Semantic ranking via FAISS search (returns all chunks ranked by similarity)
    _, sem_ranked_idxs = faiss_index.search(q_emb.reshape(1, -1), n_chunks)
    sem_ranked_idxs = sem_ranked_idxs[0]

    # RRF fusion (k=60) — only needs rank positions, not raw scores
    k = 60
    bm25_ranks = np.argsort(-bm25_scores)
    rrf = np.zeros(n_chunks)
    for rank, idx in enumerate(sem_ranked_idxs):
        rrf[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        rrf[idx] += 1.0 / (k + rank + 1)

    # Heading boost: boost chunks whose heading overlaps with query keywords
    q_tokens = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
    q_tokens.update(expand_query(query).lower().split())
    for idx in range(n_chunks):
        heading = (chunk_records[idx]["metadata"].get("heading") or "").lower()
        if not heading:
            continue
        h_tokens = set(heading.split())
        overlap = len(q_tokens & h_tokens) / max(1, len(h_tokens))
        if overlap >= 0.5:
            ctype = chunk_records[idx]["metadata"].get("chunk_type", "")
            multiplier = 2.0 if ctype in ("accordion", "accordion_faq", "page") else 1.0
            rrf[idx] += overlap * 0.08 * multiplier

    # Label boost: boost chunks whose labels match query intent
    query_labels = _detect_query_labels(query)
    if query_labels:
        for idx in range(n_chunks):
            chunk_labels = set(chunk_records[idx]["metadata"].get("labels", []))
            if chunk_labels & query_labels:
                rrf[idx] += 0.03

    # Apply soft URL penalty: each additional chunk from the same URL
    # gets a diminishing score so diverse sources surface naturally.
    url_seen: dict[str, int] = {}
    for rel in np.argsort(-rrf):
        meta = chunk_records[rel]["metadata"]
        url = _get_url(meta)
        url_seen.setdefault(url, 0)
        if url_seen[url] > 0:
            rrf[rel] *= 0.9 ** url_seen[url]
        url_seen[url] += 1

    order = np.argsort(-rrf)
    hits = []
    seen_texts = []

    for rel in order:
        if len(hits) >= top_k:
            break
        text = chunk_records[rel]["text"]
        if any(_is_dup(text, s, dup_thresh) for s in seen_texts):
            continue
        seen_texts.append(text)
        meta = chunk_records[rel]["metadata"]
        hits.append({
            "chunk_id": chunk_records[rel]["chunk_id"],
            "text": text,
            "url": _get_url(meta),
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
    return [h for _, h in ranked[:n]]
