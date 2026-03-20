import os
import json
import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

import re


# —————————————————————————————
# 1. loading
# —————————————————————————————
CHUNK_JSON     = "chunked_documents.json"
EMBEDDING_NPY  = "embeddings.npy"              
PRIMARY_LABELS =  ["contact", "career", "course", "admission", "summer", "capstone", "fee", "scholarship", "deadline"]
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
    "faculty": ["faculty", "instructor", "professor", "teacher","staff","scholar","fellow","people"],
    "research" :["research"],
    "contact" : ["contact","outreach","network","workshop"],
    "summer" : ['summer'],
    "news" : ["news","event"],
    "program": ["program"],
}

DEVICE         = "mps"   # or "cuda"／"cpu"

SIM_THRESHOLD  = 0.25    
DUP_THRESHOLD  = 0.95   
TOP_K          = 6      # return how many chunk

URL_PREFIX     = "datascience.uchicago.edu/education/masters-programs/"

SAMPLE_QUERIES = [
    "What are the core courses in the MS in Applied Data Science program?",
    "What are the admission requirements for the program?",
    "Tell me about the capstone project.",
    "What is the tuition cost for the program?",
    "What scholarships are available for the program?",
    "What are the minimum scores for the TOEFL and IELTS English Language Requirement?",
    "Is there an application fee waiver?",
    "What are the deadlines for the in-person program?",
    "How long will it take for me to receive a decision on my application?",
    "Can I set up an advising appointment with the enrollment management team?",
    "Where can I mail my official transcripts?",
    "Does the Master’s in Applied Data Science Online program provide visa sponsorship?",
    "How do I apply to the MBA/MS program?",
    "Is the MS in Applied Data Science program STEM/OPT eligible?",
    "How many courses must you complete to earn UChicago’s Master’s in Applied Data Science?",
    "What kind of careers can the student have after graduate from this program?",
    "How can I apply to the NorthWestern University machine learning and data science program",
    "How many students are there in the in-person program?",
    "Can you give me some examples of the capstone projects?",
    "Can you give me some professor names and their introduction related to generative AI?",
    "Does this program cares about DEI?",
    "What kind of scholarships does this program provide?",
    "What companies can students join after graduation? Give me some examples",
    "How many courses do the students take each quarter?",
    "How many courses the students should take to meet the graduation requirements?",
    "What resources does the program offer before students officially join?"
]

# —————————————————————————————
# 2. tool function
# —————————————————————————————

# （A）load all chunks、metadata
print("📂 Loading chunks from JSON …")
with open(CHUNK_JSON, "r", encoding="utf-8") as f:
    chunk_records: List[Dict[str, Any]] = json.load(f)

# （B）load numpy embedding
print("📂 Loading numpy embeddings …")
if os.path.exists(EMBEDDING_NPY):
    chunk_embeddings = np.load(EMBEDDING_NPY)
else:
    print("⚠️ embeddings.npy missing! reembedding everything…")
    embedder_tmp = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    chunk_texts = [rec["text"] for rec in chunk_records]
    chunk_embeddings = embedder_tmp.encode(chunk_texts, normalize_embeddings=True)
    np.save(EMBEDDING_NPY, chunk_embeddings)
    print("✅ embedding saved to", EMBEDDING_NPY)

print("🛠️ Loading Sentence-BERT embedder …")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# （D）retrive function
download('wordnet')
download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def normalize(text: str) -> str:
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(tok, pos='n') for tok in tokens]
    return ' '.join(lemmas)

def extract_query_labels(query, keyword_groups):
    query_norm = normalize(query)
    labels = []
    for group, syns in keyword_groups.items():
        if any(s.lower() in query_norm for s in syns):
            labels.append(group)
    return labels

def retrieve_two_phase(
    query: str,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    keyword_groups: dict,
    primary_labels: List[str],
    top_k: int = TOP_K,
    dup_thresh: float = DUP_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD,
):
    pool_idxs = [
        i for i, c in enumerate(chunks)
        if URL_PREFIX in c["metadata"].get("source_url", "")
    ]
    if pool_idxs:
        pass
    else:
        pool_idxs = list(range(len(chunks)))

    qlabels = extract_query_labels(query,keyword_groups)
    plabels = [l for l in qlabels if l in primary_labels]
    slabels = [l for l in qlabels if l not in primary_labels]

    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    def is_dup(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, len(sa | sb)) > dup_thresh
    
    hits = []
    seen_texts = []
    url_count = {}

    def phase_search(idxs: List[int]):
        """retrive in given idxs and add it to hits"""
        if not idxs:
            return
        sub_embs = chunk_embeddings[idxs]
        sims = cosine_similarity(q_emb.reshape(1, -1), sub_embs)[0]
        order = np.argsort(-sims)
        for rel in order:
            if sims[rel] < sim_threshold:
                return 
            real_i = idxs[rel]
            text = chunks[real_i]["text"]
            if any(is_dup(text, s) for s in seen_texts):
                continue
            if chunks[real_i]["metadata"]["source_url"] not in url_count:
                url_count[chunks[real_i]["metadata"]["source_url"]] = 1
            elif url_count[chunks[real_i]["metadata"]["source_url"]] >=3:
                continue
            else:
                url_count[chunks[real_i]["metadata"]["source_url"]] += 1
            seen_texts.append(text)
            meta = chunks[real_i]["metadata"]
            hits.append({
                "chunk_id": chunks[real_i]["chunk_id"],
                "text":      text,
                "url":       meta["source_url"],
                "labels":    meta.get("labels", []),
                "score":     float(sims[rel])
            })
            if len(hits) >= top_k:
                break

    if plabels:
        idxs_primary = [
            i for i in pool_idxs
            if any(l in chunks[i]["metadata"].get("labels", []) for l in plabels) 
        ]
        phase_search(idxs_primary)
        if len(hits) >= top_k:
            return hits[:top_k]

    if slabels:
        idxs_secondary = [
            i for i in pool_idxs
            if any(l in chunks[i]["metadata"].get("labels", []) for l in slabels)
        ]
        phase_search(idxs_secondary)
        if len(hits) >= top_k:
            return hits[:top_k]
        
    if plabels:
        idxs_primary = [
            i for i, c in enumerate(chunks)
            if any(l in c["metadata"].get("labels", []) for l in plabels) 
        ]
        phase_search(idxs_primary)
        if len(hits) >= top_k:
            return hits[:top_k]

    if slabels:
        idxs_secondary = [
            i for i, c in enumerate(chunks)
            if any(l in c["metadata"].get("labels", []) for l in slabels)
        ]
        phase_search(idxs_secondary)
        if len(hits) >= top_k:
            return hits[:top_k]
        
    phase_search(list(range(len(chunks))))
    return hits[:top_k]


# —————————————————————————————
# 3. Load LLM
# —————————————————————————————
llm = ChatOpenAI(
    model_name="gpt-4.1",
    temperature=0.0,
    max_tokens=512,
    openai_api_key="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"
)

llm_subtitude = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=512,
    openai_api_key="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"
)

def answer_with_rag(query: str):
    hits = retrieve_two_phase(
        query=query,
        chunks=chunk_records,
        chunk_embeddings=chunk_embeddings,
        embedder=embedder,
        keyword_groups=KEYWORD_GROUPS,
        primary_labels=PRIMARY_LABELS,
        top_k=TOP_K,
        dup_thresh=DUP_THRESHOLD,
        sim_threshold=SIM_THRESHOLD,
    )

    if not hits:
        return ("No relevant chunks found.", "I’m sorry, I don’t have enough information to answer that question.")

    # Prompt
    context = "\n\n".join(f"[Doc] {h['text']}" for h in hits)
    prompt  = (
        "You are a helpful assistant for University of Chicago’s MS in Applied Data Science program.\n"
        "Use the provided context below to answer the user question. "
        "If the answer is not in the context, say:\n"
        "\"I’m sorry, I don’t have enough information to answer that question.\".\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[QUESTION]\n{query}\n\n"
        "Answer:"
    )

    # LLM
    response = llm.predict(prompt)
    response = response.strip()  
    # if LLM failed to give a answer, try another LLM
    if response.startswith("I'm sorry, I don't have enough information to answer that question."):
        response = llm_subtitude.predict(prompt)
        response = response.strip()
    
    url = []
    for h in hits:
        if h['url'] not in url:
            url.append(h["url"])

    if url is not None:
        response = response + '\n' + "Please refer to the following urls:"
        for url_text in url:
            response = response + '\n' + url_text

    return response


# —————————————————————————————
# 4. Use Gradio to get Web interface
# —————————————————————————————
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 📚 UChicago ADS RAG QA Interface")
        gr.Markdown("Type you question below!")
        
        query_in = gr.Textbox(
            label="Ask a question.",
            placeholder="What are the admission requirements?",
            lines=2
        )
        sample_btns = gr.Dropdown(
            choices=SAMPLE_QUERIES,
            label="Sample Question"
        )
        answer_text = gr.Textbox(
            label="Answer",
            interactive=False,
            lines=8
        )
        """        
        source_text = gr.Textbox(
            label="URL & Score",
            interactive=False,
            lines=6
        )"""

        submit_btn = gr.Button("submit")

        sample_btns.change(fn=lambda x: x or "", inputs=sample_btns, outputs=query_in)

        submit_btn.click(
            fn=answer_with_rag,
            inputs=query_in,
            outputs=[answer_text]
        )

        gr.Markdown("----\n*Powered by gpt-4.1 + FAISS-based RAG*")
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        share=False, 
        server_name="0.0.0.0",
        server_port=7860
    )