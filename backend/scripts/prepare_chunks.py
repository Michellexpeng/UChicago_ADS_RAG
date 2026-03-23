"""
End-to-end chunk preparation: label → chunk → deduplicate → embed → index.

Reads  data/uchicago_ads_pages_depth3.json  (147 pages with _raw_html)
Writes data/chunked_documents.json          (all chunks, overwrite)
       data/chunked_documents_dedup.json    (deduped)
       data/embeddings_dedup.npy            (MiniLM)
       data/uchicago_ads_faiss_dedup.index  (MiniLM FAISS)

Usage:
  cd backend
  python scripts/prepare_chunks.py              # MiniLM only
  python scripts/prepare_chunks.py --gemini     # Also build Gemini index
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

import faiss
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MIN_TEXT_LENGTH = 50

# ── NLTK setup ──────────────────────────────────────────────────
nltk_download("wordnet", quiet=True)
nltk_download("omw-1.4", quiet=True)
_lemmatizer = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════════
# 1. Page label annotation  (from notebook cells 10-15)
# ═══════════════════════════════════════════════════════════════

keyword_groups = {
    "admission": ["admission", "apply", "application", "enrollment"],
    "career": ["career", "job", "employment", "profession"],
    "capstone": ["capstone", "final project", "research"],
    "fee": ["tuition", "cost", "fee", "price"],
    "course": ["course", "class", "curriculum", "track"],
    "foundational": ["prequarter", "pre-quarter", "Quarter 1"],
    "resource": ["resource", "career", "support", "help"],
    "schedule": ["schedule", "timetable", "class time", "quarter"],
    "application": ["application", "deadline", " due ", "submission"],
    "scholarship": ["scholarship", "financial aid"],
    "english": ["toefl", "ielts", " gre ", "language requirement", "english score"],
    "visa": ["visa", "sponsorship", "international student"],
    "faculty": ["faculty", "instructor", "professor", "teacher", "staff", "scholar", "fellow", "people"],
    "contact": ["contact", "outreach", "network", "workshop"],
    "news": ["news", "event"],
}


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    tokens = text.split()
    return " ".join(_lemmatizer.lemmatize(tok, pos="n") for tok in tokens)


def extract_labels(title: str, url: str) -> list[str]:
    norm = normalize(title) + " " + normalize(url)
    labels = []
    for grp, synonyms in keyword_groups.items():
        for word in synonyms:
            if word.lower() in norm:
                labels.append(grp)
                break
    return labels if labels else ["other"]


def count_keyword_groups(text: str, title: str, url: str) -> dict[str, int]:
    norm = normalize(text) + " " + normalize(title) + " " + normalize(url) + " " + normalize(url) + " " + normalize(url)
    tokens = norm.split()
    counter = Counter(tokens)
    group_hits = {}
    for grp, syns in keyword_groups.items():
        hits = 0
        for s in syns:
            hits += counter[s.lower()]
        group_hits[grp] = hits
    return group_hits


def classify_based_on_hits(group_hits: dict, threshold_ratio: float = 0.6, min_total_hits: int = 5) -> str:
    total_hits = sum(group_hits.values())
    if total_hits < min_total_hits:
        return "Specific"
    top_hits = sorted(group_hits.values(), reverse=True)[:3]
    ratio = sum(top_hits) / total_hits
    return "Specific" if ratio >= threshold_ratio else "General"


def annotate_pages(pages_data: list[dict]) -> list[dict]:
    annotated = []
    for page in pages_data:
        url = page["url"]
        title = page["title"]
        text = page.get("text", "")
        group_hits = count_keyword_groups(text, title, url)
        level = classify_based_on_hits(group_hits)
        labels = extract_labels(title, url)
        if "people" in url:
            level = "Specific"
        annotated.append({
            "url": url,
            "title": title,
            "text": text,
            "_raw_html": page.get("_raw_html", ""),
            "labels": labels,
            "level": level,
        })
    return annotated


# ═══════════════════════════════════════════════════════════════
# 2. Chunking  (from notebook cell 20, with improved _extract_paragraphs)
# ═══════════════════════════════════════════════════════════════

_ALWAYS_REMOVE_TAGS = ["script", "style", "noscript", "iframe"]
_SKIP_CLASSES = re.compile(
    r"(site-header|site-footer|site-navigation|main-navigation|"
    r"breadcrumb|cookie|social-share|search-form|related-posts|"
    r"mega-menu|mobile-menu|skip-link)",
    re.IGNORECASE,
)


def _clean_soup(soup):
    for tag in _ALWAYS_REMOVE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    _ROOT_TAGS = {"html", "head", "body"}
    for el in list(soup.find_all(True)):
        if el.name in _ROOT_TAGS:
            continue
        if not el.attrs:
            continue
        cls = " ".join(el.get("class", []))
        if _SKIP_CLASSES.search(cls):
            el.decompose()
    return soup


def _main_content(soup):
    return soup.select_one(".main-content") or soup


def _make_chunk(text, url, chunk_type, **meta):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return None
    return {
        "text": text,
        "metadata": {
            "source_url": url,
            "chunk_type": chunk_type,
            "heading": meta.get("heading"),
            "sub_heading": meta.get("sub_heading"),
        },
    }


# ── Accordion helpers ────────────────────────────────────────

_ACCORDION_SUBSPLIT_CHARS = 1500
_FAQ_URL_RE = re.compile(r"faq|faqs|frequently|q-and-a", re.IGNORECASE)
_FAQ_BODY_RE = re.compile(r"frequently asked|<details|<dt>", re.IGNORECASE)
_QUESTION_RE = re.compile(r".{10,}\?$")


def _subsplit_schedule(content_el, url, heading):
    chunks = []
    for quarter_div in content_el.select("div.quarter"):
        title_el = quarter_div.find(["h3", "h4", "strong"])
        sub_heading = title_el.get_text(strip=True) if title_el else quarter_div.get_text(" ", strip=True)[:60]
        body = quarter_div.get_text(separator="\n", strip=True)
        c = _make_chunk(f"{heading} — {sub_heading}\n{body}", url, "accordion_schedule",
                        heading=heading, sub_heading=sub_heading)
        if c:
            chunks.append(c)
    return chunks


def _subsplit_nested_accordion(content_el, url, heading):
    chunks = []
    for li in content_el.select("ul.accordion li.accordion__item"):
        title_el = li.select_one(".accordion-title")
        body_el = li.select_one(".accordion__content")
        if not title_el:
            continue
        sub_heading = title_el.get_text(strip=True)
        body = body_el.get_text(separator="\n", strip=True) if body_el else li.get_text(separator="\n", strip=True)
        chunk_type = "accordion_faq" if _FAQ_URL_RE.search(url) else "accordion_course"
        c = _make_chunk(f"{heading} — {sub_heading}\n{body}", url, chunk_type,
                        heading=heading, sub_heading=sub_heading)
        if c:
            chunks.append(c)
    return chunks


def _subsplit_jobs(content_el, url, heading):
    chunks = []
    textblock = content_el.select_one("div.textblock")
    if not textblock:
        return chunks
    current_h3, current_parts = None, []

    def _flush(h3_text, parts):
        body = "\n".join(parts).strip()
        if not body:
            return None
        return _make_chunk(f"{heading} — {h3_text}\n{body}", url, "accordion_job",
                           heading=heading, sub_heading=h3_text)

    for el in textblock.children:
        if not hasattr(el, "name") or el.name is None:
            t = str(el).strip()
            if t:
                current_parts.append(t)
            continue
        if el.name == "h3":
            if current_h3:
                c = _flush(current_h3, current_parts)
                if c:
                    chunks.append(c)
            current_h3, current_parts = el.get_text(strip=True), []
        else:
            current_parts.append(el.get_text(separator="\n", strip=True))
    if current_h3:
        c = _flush(current_h3, current_parts)
        if c:
            chunks.append(c)
    return chunks


_ACCORDION_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def _subsplit_generic(content_el, url, heading):
    body = content_el.get_text(separator="\n", strip=True)
    chunks = []
    for i, t in enumerate(_ACCORDION_TEXT_SPLITTER.split_text(body)):
        sub_heading = f"{heading} (part {i + 1})"
        c = _make_chunk(f"{heading}\n{t}", url, "accordion_sub",
                        heading=heading, sub_heading=sub_heading)
        if c:
            chunks.append(c)
    return chunks


def _choose_subsplitter(content_el):
    if content_el.select("div.quarter"):
        return _subsplit_schedule
    if content_el.select("ul.accordion li.accordion__item"):
        return _subsplit_nested_accordion
    if content_el.select("div.textblock h3"):
        return _subsplit_jobs
    return None


def _has_accordion(soup):
    return bool(soup.select(".accordion-item, [data-accordion] > li"))


def _get_page_title(soup) -> str:
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True).split("|")[0].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


# ── Text splitter for paragraphs (sentence-aware, no overlap) ─

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
)


def _extract_accordion(html, url):
    soup = _clean_soup(BeautifulSoup(html, "html.parser"))
    main = _main_content(soup)
    page_title = _get_page_title(soup)
    chunks = []
    for item in main.select(".accordion-item"):
        title_el = item.select_one(".accordion-title")
        content_el = item.select_one(".accordion-content")
        if not title_el or not content_el:
            continue
        heading = title_el.get_text(strip=True)
        body = content_el.get_text(separator="\n", strip=True)
        parent = _make_chunk(f"{heading}\n{body}", url, "accordion",
                             heading=heading, sub_heading=None)
        if parent:
            chunks.append(parent)
        if len(body) <= _ACCORDION_SUBSPLIT_CHARS:
            continue
        subsplitter = _choose_subsplitter(content_el) or _subsplit_generic
        chunks.extend(subsplitter(content_el, url, heading))

    # Remaining non-accordion content — use paragraph splitting with context prefix
    for item in main.select(".accordion-item"):
        item.decompose()
    remaining = main.get_text(separator="\n", strip=True)
    if remaining and len(remaining) > 50:
        paragraphs = [p.strip() for p in remaining.split("\n\n") if p.strip()]
        buffer = ""
        for para in paragraphs:
            if buffer and len(buffer) + len(para) + 2 <= 800:
                buffer += "\n\n" + para
                continue
            if buffer:
                text = f"[{page_title}]\n{buffer}" if page_title else buffer
                c = _make_chunk(text, url, "section", heading=page_title)
                if c:
                    chunks.append(c)
                buffer = ""
            if len(para) <= 800:
                buffer = para
            else:
                for t in _TEXT_SPLITTER.split_text(para):
                    text = f"[{page_title}]\n{t}" if page_title else t
                    c = _make_chunk(text, url, "section", heading=page_title)
                    if c:
                        chunks.append(c)
        if buffer:
            text = f"[{page_title}]\n{buffer}" if page_title else buffer
            c = _make_chunk(text, url, "section", heading=page_title)
            if c:
                chunks.append(c)

    return chunks


def _extract_paragraphs(html, url):
    soup = _clean_soup(BeautifulSoup(html, "html.parser"))
    main = _main_content(soup)
    plain = main.get_text(separator="\n", strip=True)
    page_title = _get_page_title(soup)
    chunks = []

    # Page-level parent chunk (full text for holistic answers)
    parent = _make_chunk(plain, url, "page", heading=None)
    if parent:
        chunks.append(parent)

    # Step 1: split by double newline into paragraphs
    paragraphs = [p.strip() for p in plain.split("\n\n") if p.strip()]

    # Step 2: merge short paragraphs, split long ones at sentence boundaries
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) + 2 <= 800:
            buffer += "\n\n" + para
            continue
        if buffer:
            text = f"[{page_title}]\n{buffer}" if page_title else buffer
            c = _make_chunk(text, url, "section", heading=page_title)
            if c:
                chunks.append(c)
            buffer = ""
        if len(para) <= 800:
            buffer = para
        else:
            # Long paragraph: split at sentence boundaries
            sub_texts = _TEXT_SPLITTER.split_text(para)
            for t in sub_texts:
                text = f"[{page_title}]\n{t}" if page_title else t
                c = _make_chunk(text, url, "section", heading=page_title)
                if c:
                    chunks.append(c)
    # Flush remaining buffer
    if buffer:
        text = f"[{page_title}]\n{buffer}" if page_title else buffer
        c = _make_chunk(text, url, "section", heading=page_title)
        if c:
            chunks.append(c)

    return chunks


def _extract_faq(html, url):
    soup = _clean_soup(BeautifulSoup(html, "html.parser"))
    main = _main_content(soup)
    chunks = []
    for details in main.find_all("details"):
        summary = details.find("summary")
        if not summary:
            continue
        question = summary.get_text(strip=True)
        summary.decompose()
        answer = details.get_text(separator=" ", strip=True)
        c = _make_chunk(f"Q: {question}\nA: {answer}", url, "faq", heading=question)
        if c:
            chunks.append(c)
    if not chunks:
        for dt in main.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                q = dt.get_text(strip=True)
                a = dd.get_text(separator=" ", strip=True)
                c = _make_chunk(f"Q: {q}\nA: {a}", url, "faq", heading=q)
                if c:
                    chunks.append(c)
    if not chunks:
        for h in main.find_all(["h2", "h3", "h4", "strong"]):
            q = h.get_text(strip=True)
            if not _QUESTION_RE.match(q):
                continue
            parts = []
            for sib in h.find_next_siblings():
                if sib.name in ("h2", "h3", "h4"):
                    break
                parts.append(sib.get_text(separator=" ", strip=True))
            a = " ".join(parts).strip()
            if a:
                c = _make_chunk(f"Q: {q}\nA: {a}", url, "faq", heading=q)
                if c:
                    chunks.append(c)
    return chunks


_CAREER_INTERNSHIPS = (
    "Career Outcomes - Student Internships: "
    "The following companies have hired UChicago MS in Applied Data Science students as interns: "
    "Amazon, SiriusXM, Teragonia, T-Mobile, BMO, Motorola, Lenovo, CVS Health, Uber, "
    "Deloitte, TransUnion, JLL, Capital One, NASA, Apple, United Nations, UPS"
)
_CAREER_ALUMNI = (
    "Career Outcomes - Alumni Careers: "
    "UChicago MS in Applied Data Science graduates work at the following companies: "
    "Goldman Sachs, Microsoft, Niantic, Booz Allen, Thrivent, Glassdoor, Kuaishou, BLA, "
    "Peregrine Economics, Ant Group, Tempus, ADM, Chicago Bulls, OpenAI, Mastercard"
)


def extract_content(url, html):
    soup_probe = BeautifulSoup(html, "html.parser")
    if _has_accordion(soup_probe):
        chunks = _extract_accordion(html, url)
        if chunks:
            return chunks
    if _FAQ_URL_RE.search(url) or _FAQ_BODY_RE.search(html[:5000]):
        chunks = _extract_faq(html, url)
        if chunks:
            return chunks
    chunks = _extract_paragraphs(html, url)
    if "career-outcomes" in url:
        chunks.append(_make_chunk(_CAREER_INTERNSHIPS, url, "section", heading="Career Outcomes"))
        chunks.append(_make_chunk(_CAREER_ALUMNI, url, "section", heading="Career Outcomes"))
    return chunks


# ═══════════════════════════════════════════════════════════════
# 3. Deduplication  (from deduplicate_chunks.py)
# ═══════════════════════════════════════════════════════════════

def deduplicate(chunks: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    deduped: list[dict] = []
    stats = {"exact_dupes": 0, "too_short": 0}

    for chunk in chunks:
        text = chunk["text"]
        if len(text.strip()) < MIN_TEXT_LENGTH:
            stats["too_short"] += 1
            continue
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            canonical = deduped[seen[h]]
            dup_url = chunk["metadata"]["source_url"]
            if dup_url not in canonical["metadata"]["source_urls"]:
                canonical["metadata"]["source_urls"].append(dup_url)
            stats["exact_dupes"] += 1
        else:
            seen[h] = len(deduped)
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


# ═══════════════════════════════════════════════════════════════
# 4. Embedding + FAISS index
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prepare chunks: label → chunk → dedup → embed → index")
    parser.add_argument("--gemini", action="store_true", help="Also build Gemini embedding index")
    args = parser.parse_args()

    # 1. Load crawled pages
    pages_path = os.path.join(DATA_DIR, "uchicago_ads_pages_depth3.json")
    print(f"Loading pages from {pages_path}...")
    with open(pages_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)
    print(f"  {len(pages_data)} pages loaded")

    # 2. Annotate pages with labels
    print("Annotating pages...")
    annotated = annotate_pages(pages_data)

    # 3. Chunk all pages
    print("Chunking pages...")
    chunk_records = []
    uid = 0
    for page in annotated:
        url = page["url"]
        title = page["title"]
        html = page.get("_raw_html", "")
        if not html:
            continue
        chunks = extract_content(url, html)
        for c in chunks:
            c["metadata"]["page_title"] = title
            c["metadata"]["labels"] = page.get("labels", [])
            c["metadata"]["level"] = page.get("level", "General")
            c["chunk_id"] = f"chunk_{uid}_{title.replace(' ', '_')}"
            chunk_records.append(c)
            uid += 1

    print(f"  {len(chunk_records)} chunks generated")

    # Save raw chunks
    raw_path = os.path.join(DATA_DIR, "chunked_documents.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {raw_path}")

    # 4. Deduplicate
    print("Deduplicating...")
    deduped = deduplicate(chunk_records)
    texts = [c["text"] for c in deduped]

    dedup_path = os.path.join(DATA_DIR, "chunked_documents_dedup.json")
    with open(dedup_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"  Saved {dedup_path}")

    # 5. Build MiniLM index
    vecs, index = build_minilm_index(texts)
    np.save(os.path.join(DATA_DIR, "embeddings_dedup.npy"), vecs)
    faiss.write_index(index, os.path.join(DATA_DIR, "uchicago_ads_faiss_dedup.index"))
    print(f"  MiniLM embeddings {vecs.shape}, FAISS index n={index.ntotal}")

    # 6. Optionally build Gemini index
    if args.gemini:
        vecs_g, index_g = build_gemini_index(texts)
        np.save(os.path.join(DATA_DIR, "embeddings_gemini_dedup.npy"), vecs_g)
        faiss.write_index(index_g, os.path.join(DATA_DIR, "uchicago_ads_faiss_gemini_dedup.index"))
        print(f"  Gemini embeddings {vecs_g.shape}, FAISS index n={index_g.ntotal}")

    print("Done.")


if __name__ == "__main__":
    main()
