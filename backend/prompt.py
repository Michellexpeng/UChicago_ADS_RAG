"""
Prompt construction, query translation, and citation parsing for the RAG pipeline.
"""

import re

from retrieval import is_list_query

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def _is_pure_ascii(text: str) -> bool:
    """Return True if all non-space chars are ASCII."""
    return all(ord(c) < 128 for c in text if not c.isspace())


# ---------------------------------------------------------------------------
# Query translation
# ---------------------------------------------------------------------------
async def translate_to_english(query: str, llm) -> str:
    """Translate non-English query to English using the LLM.

    English queries pass through unchanged.
    """
    if _is_pure_ascii(query):
        return query
    response = await llm.ainvoke(
        "This is a question about UChicago's MS in Applied Data Science program. "
        f"Translate the following to English. Return ONLY the translation, nothing else.\n\n{query}"
    )
    return response.content.strip()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(question: str, hits: list) -> str:
    listing = is_list_query(question)
    instruction = (
        "List each item clearly and separately."
        if listing
        else "Answer concisely and directly."
    )
    context = "\n\n".join(f"[Doc{i+1}] {h['text']}" for i, h in enumerate(hits))

    lang_hint = ""
    if not _is_pure_ascii(question):
        lang_hint = "请用中文回答。专有名词（如课程名、项目名）保持英文。\n"

    return (
        "You are a helpful assistant for University of Chicago's MS in Applied Data Science program.\n"
        "Use the provided context below to answer the user question. "
        f"{instruction} "
        "IMPORTANT: Do NOT start with disclaimers or statements about what the context lacks. "
        "If the context contains relevant information -- even under different terminology -- "
        "synthesize all relevant details. "
        "If the context only partially addresses the question, answer the part you can "
        "and clearly state which part is not covered. "
        "Answer in the same language as the user's question. "
        "Keep proper nouns, program names, course titles, and academic terms in their original English form. "
        "Format your answer in Markdown.\n"
        "At the very end of your answer, on a new line, list ONLY the document labels you "
        "actually used (e.g. [Doc1][Doc3]). Do not include documents you did not reference. "
        "If you cannot answer the question from the context, do NOT cite any documents.\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"{lang_hint}"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------
_DONT_KNOW_RE = re.compile(
    r"(不知道|无法回答|没有.*信息|don'?t know|no information|cannot answer|not sure)",
    re.IGNORECASE,
)


def parse_citations(full_answer: str, hits: list) -> tuple[str, list[str], list[dict]]:
    """Parse [DocN] citations from the LLM answer and return (clean_text, sources, references).

    Only includes sources that were actually cited. Suppresses sources for
    "I don't know" style answers.
    """
    doc_map: dict[int, dict] = {i + 1: h for i, h in enumerate(hits)}
    cited_nums = set(int(m) for m in re.findall(r"\[Doc(\d+)\]", full_answer))

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

    clean_answer = re.sub(r"\s*\[Doc\d+\]", "", full_answer).strip()

    # Suppress sources for "I don't know" style answers
    if _DONT_KNOW_RE.search(clean_answer) and len(clean_answer) < 200:
        sources.clear()
        references.clear()

    return clean_answer, sources, references
