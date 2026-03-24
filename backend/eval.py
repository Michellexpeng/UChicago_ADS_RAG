"""
RAG retrieval evaluation script.

Supports two evaluation modes:
  1. Golden test set: hand-curated queries with labeled relevant chunk_ids
  2. Pseudo-query:    sample chunks, use first sentence as query, check if source chunk is retrieved

Metrics:
  - Recall@K:  fraction of relevant chunks found in top-K
  - MRR:       mean reciprocal rank of the first relevant result
  - NDCG@K:    normalized discounted cumulative gain (ranking quality)

Usage:
  cd backend
  python eval.py                          # golden test set (default)
  python eval.py --mode pseudo --samples 100  # pseudo-query evaluation
  python eval.py --dedup                  # evaluate with deduped chunks
"""

import argparse
import json
import math
import os
import sys

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, os.path.dirname(__file__))

from retrieval import retrieve, rerank, tokenize_for_bm25, RetrievalConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Simple query translation for evaluation (avoids LLM API costs)
# ---------------------------------------------------------------------------
def _translate_query(query: str) -> str:
    """Translate non-ASCII queries to English for retrieval.

    Uses a static map for known test queries. In production, main.py uses
    LLM-based translation via prompt.translate_to_english().
    """
    if all(ord(c) < 128 for c in query if not c.isspace()):
        return query  # already English

    _TRANSLATIONS = {
        "多少钱": "How much does the program cost?",
        "这个项目要读多久": "How long does the program take to complete?",
        "怎么申请": "How do I apply to the program?",
        "有奖学金吗": "Are there scholarships available?",
        "毕业后能找到什么工作": "What jobs can graduates find after the program?",
        "线上和线下有什么区别": "What is the difference between online and in-person programs?",
    }
    return _TRANSLATIONS.get(query, query)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items found in top-K retrieved."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: 1 if chunk_id is in relevant set, 0 otherwise.
    """
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant items ranked first
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_resources(dedup: bool = False):
    """Load chunks, embedder, BM25, FAISS index, and cross-encoder."""
    suffix = "_dedup" if dedup else ""

    chunks_path = os.path.join(DATA_DIR, f"chunked_documents{suffix}.json")
    emb_path = os.path.join(DATA_DIR, f"embeddings{suffix}.npy")
    idx_path = os.path.join(DATA_DIR, f"uchicago_ads_faiss{suffix}.index")

    print(f"Loading chunks from {os.path.basename(chunks_path)}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    if os.path.exists(idx_path):
        faiss_index = faiss.read_index(idx_path)
    else:
        vecs = np.load(emb_path).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        faiss_index.add(vecs)

    print("Building BM25 index...")
    tokenized = [tokenize_for_bm25(c["text"]) for c in chunk_records]
    bm25 = BM25Okapi(tokenized)

    print("Loading cross-encoder...")
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return chunk_records, embedder, bm25, faiss_index, cross_enc


# ---------------------------------------------------------------------------
# Golden test set evaluation
# ---------------------------------------------------------------------------
def eval_golden(chunk_records, embedder, bm25, faiss_index, cross_enc,
                k_values=(1, 3, 5, 10), retrieve_k=20, rerank_k=5):
    """Evaluate using the golden test set."""
    golden_path = os.path.join(DATA_DIR, "golden_test_set.json")
    if not os.path.exists(golden_path):
        print(f"ERROR: {golden_path} not found. Create it first.")
        sys.exit(1)

    with open(golden_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    print(f"\n{'='*60}")
    print(f"Golden Test Set Evaluation ({len(test_set)} queries)")
    print(f"retrieve_k={retrieve_k}, rerank_k={rerank_k}")
    print(f"{'='*60}\n")

    # Collect per-query results
    results = []
    for item in test_set:
        query = item["query"]
        retrieval_query = _translate_query(query)
        relevant = set(item["relevant_chunk_ids"])
        category = item.get("category", "unknown")

        candidates = retrieve(retrieval_query, chunk_records, embedder, bm25, faiss_index, top_k=retrieve_k)
        hits = rerank(retrieval_query, candidates, cross_enc, top_k=rerank_k)
        retrieved_ids = [h["chunk_id"] for h in hits]

        row = {"query": query, "category": category}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        row["mrr"] = reciprocal_rank(retrieved_ids, relevant)
        row[f"ndcg@{max(k_values)}"] = ndcg_at_k(retrieved_ids, relevant, max(k_values))
        results.append(row)

    # Print per-query results
    for r in results:
        status = "HIT" if r["recall@5"] > 0 else "MISS"
        print(f"  [{status}] {r['query'][:60]:<60}  R@5={r['recall@5']:.2f}  MRR={r['mrr']:.2f}")

    # Aggregate
    print(f"\n{'─'*60}")
    print("AGGREGATE RESULTS:")
    for k in k_values:
        key = f"recall@{k}"
        avg = sum(r[key] for r in results) / len(results)
        print(f"  Recall@{k}:  {avg:.3f}")
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    ndcg_key = f"ndcg@{max(k_values)}"
    avg_ndcg = sum(r[ndcg_key] for r in results) / len(results)
    print(f"  MRR:       {avg_mrr:.3f}")
    print(f"  NDCG@{max(k_values)}:   {avg_ndcg:.3f}")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    if len(categories) > 1:
        print(f"\n{'─'*60}")
        print("PER-CATEGORY BREAKDOWN:")
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            avg_r5 = sum(r["recall@5"] for r in cat_results) / len(cat_results)
            avg_m = sum(r["mrr"] for r in cat_results) / len(cat_results)
            print(f"  {cat:<20} (n={len(cat_results):>2})  R@5={avg_r5:.3f}  MRR={avg_m:.3f}")

    return results


# ---------------------------------------------------------------------------
# Pseudo-query evaluation
# ---------------------------------------------------------------------------
def eval_pseudo(chunk_records, embedder, bm25, faiss_index, cross_enc,
                n_samples=100, k_values=(1, 3, 5, 10), retrieve_k=20, rerank_k=10, seed=42):
    """Evaluate using pseudo-queries generated from chunk text."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(chunk_records), size=min(n_samples, len(chunk_records)), replace=False)

    print(f"\n{'='*60}")
    print(f"Pseudo-Query Evaluation ({len(indices)} samples)")
    print(f"retrieve_k={retrieve_k}, rerank_k={rerank_k}")
    print(f"{'='*60}\n")

    metrics = {f"recall@{k}": [] for k in k_values}
    metrics["mrr"] = []

    for idx in indices:
        chunk = chunk_records[idx]
        # Use first sentence as pseudo-query
        text = chunk["text"].strip()
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) < 10:
            continue

        true_id = chunk["chunk_id"]
        candidates = retrieve(first_sentence, chunk_records, embedder, bm25, faiss_index, top_k=retrieve_k)
        hits = rerank(first_sentence, candidates, cross_enc, top_k=rerank_k)
        retrieved_ids = [h["chunk_id"] for h in hits]

        relevant = {true_id}
        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
        metrics["mrr"].append(reciprocal_rank(retrieved_ids, relevant))

    print("PSEUDO-QUERY RESULTS:")
    for k in k_values:
        key = f"recall@{k}"
        avg = sum(metrics[key]) / len(metrics[key]) if metrics[key] else 0
        print(f"  Recall@{k}:  {avg:.3f}")
    avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0
    print(f"  MRR:       {avg_mrr:.3f}")

    return metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
def eval_ablation(retrieve_k=20, rerank_k=10, eval_mode="golden", n_samples=100):
    """Run ablation study: progressively enable features and measure impact.

    Each stage adds one feature on top of the previous stage.
    Both dedup and non-dedup chunks are tested at the final stage.
    """
    STAGES = [
        ("1. FAISS only",
         False,  # dedup
         RetrievalConfig(use_bm25=False, use_synonyms=False, use_heading_boost=False,
                         use_label_boost=False, use_url_penalty=False),
         False),  # use_reranking
        ("2. + BM25 hybrid (RRF)",
         False,
         RetrievalConfig(use_bm25=True, use_synonyms=False, use_heading_boost=False,
                         use_label_boost=False, use_url_penalty=False),
         False),
        ("3. + Boosting + reranking",
         False,
         RetrievalConfig(),  # all retrieval features on
         True),
        ("4. + Deduplication",
         True,
         RetrievalConfig(),
         True),
    ]

    print(f"\n{'='*70}")
    print(f"Ablation Study (eval={eval_mode}, retrieve_k={retrieve_k}, rerank_k={rerank_k})")
    print(f"{'='*70}")

    results_table = []

    for stage_name, dedup, config, use_reranking in STAGES:
        print(f"\n--- {stage_name} (dedup={dedup}) ---")
        chunk_records, embedder, bm25, faiss_index, cross_enc = load_resources(dedup=dedup)
        print(f"  Chunks: {len(chunk_records)}")

        if eval_mode == "golden":
            golden_path = os.path.join(DATA_DIR, "golden_test_set.json")
            with open(golden_path, "r", encoding="utf-8") as f:
                test_set = json.load(f)

            recall5_list, mrr_list = [], []
            for item in test_set:
                query = _translate_query(item["query"])
                relevant = set(item["relevant_chunk_ids"])
                candidates = retrieve(query, chunk_records, embedder, bm25, faiss_index,
                                      top_k=retrieve_k, config=config)
                if use_reranking:
                    hits = rerank(query, candidates, cross_enc, top_k=rerank_k)
                else:
                    hits = candidates[:rerank_k]
                retrieved_ids = [h["chunk_id"] for h in hits]
                recall5_list.append(recall_at_k(retrieved_ids, relevant, 5))
                mrr_list.append(reciprocal_rank(retrieved_ids, relevant))

            avg_r5 = sum(recall5_list) / len(recall5_list)
            avg_mrr = sum(mrr_list) / len(mrr_list)
            n_queries = len(test_set)

        else:  # pseudo
            rng = np.random.RandomState(42)
            indices = rng.choice(len(chunk_records), size=min(n_samples, len(chunk_records)), replace=False)
            recall5_list, mrr_list = [], []
            for idx in indices:
                chunk = chunk_records[idx]
                first_sentence = chunk["text"].strip().split(".")[0].strip()
                if len(first_sentence) < 10:
                    continue
                true_id = chunk["chunk_id"]
                candidates = retrieve(first_sentence, chunk_records, embedder, bm25, faiss_index,
                                      top_k=retrieve_k, config=config)
                if use_reranking:
                    hits = rerank(first_sentence, candidates, cross_enc, top_k=rerank_k)
                else:
                    hits = candidates[:rerank_k]
                retrieved_ids = [h["chunk_id"] for h in hits]
                recall5_list.append(recall_at_k(retrieved_ids, {true_id}, 5))
                mrr_list.append(reciprocal_rank(retrieved_ids, {true_id}))

            avg_r5 = sum(recall5_list) / len(recall5_list)
            avg_mrr = sum(mrr_list) / len(mrr_list)
            n_queries = len(recall5_list)

        results_table.append((stage_name, n_queries, avg_r5, avg_mrr))
        print(f"  Recall@5={avg_r5:.3f}  MRR={avg_mrr:.3f}")

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Stage':<42} {'N':>4}  {'R@5':>6}  {'MRR':>6}  {'ΔR@5':>6}")
    print(f"  {'─'*42} {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}")
    prev_r5 = None
    for stage_name, n, r5, mrr in results_table:
        delta = f"+{r5 - prev_r5:.3f}" if prev_r5 is not None else "  —"
        print(f"  {stage_name:<42} {n:>4}  {r5:.3f}  {mrr:.3f}  {delta:>6}")
        prev_r5 = r5

    return results_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument("--mode", choices=["golden", "pseudo", "both", "ablation"], default="both",
                        help="Evaluation mode (default: both)")
    parser.add_argument("--dedup", action="store_true",
                        help="Use deduplicated chunks")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples for pseudo-query evaluation")
    parser.add_argument("--retrieve-k", type=int, default=20,
                        help="Number of candidates to retrieve before reranking")
    parser.add_argument("--rerank-k", type=int, default=10,
                        help="Number of results after reranking")
    args = parser.parse_args()

    if args.mode == "ablation":
        eval_ablation(retrieve_k=args.retrieve_k, rerank_k=args.rerank_k,
                      eval_mode="golden", n_samples=args.samples)
        return

    chunk_records, embedder, bm25, faiss_index, cross_enc = load_resources(dedup=args.dedup)
    print(f"\nChunks loaded: {len(chunk_records)}")

    if args.mode in ("golden", "both"):
        golden_path = os.path.join(DATA_DIR, "golden_test_set.json")
        if os.path.exists(golden_path):
            eval_golden(chunk_records, embedder, bm25, faiss_index, cross_enc,
                        retrieve_k=args.retrieve_k, rerank_k=args.rerank_k)
        elif args.mode == "golden":
            print(f"ERROR: {golden_path} not found.")
            sys.exit(1)
        else:
            print(f"Skipping golden eval ({golden_path} not found)")

    if args.mode in ("pseudo", "both"):
        eval_pseudo(chunk_records, embedder, bm25, faiss_index, cross_enc,
                    n_samples=args.samples, retrieve_k=args.retrieve_k, rerank_k=args.rerank_k)


if __name__ == "__main__":
    main()
