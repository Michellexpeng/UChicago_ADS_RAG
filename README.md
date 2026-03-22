# UChicago Applied Data Science Q&A

A RAG-based Q&A system for the University of Chicago's MS in Applied Data Science program. Users can ask questions about admissions, curriculum, career outcomes, and more — the system retrieves relevant information from the program website and generates answers using an LLM.

**Live Demo:** https://uchicago-ads-rag.web.app
<img width="1512" height="860" alt="image" src="https://github.com/user-attachments/assets/176099db-dc1e-4562-9a4f-a6b28ffdfe36" />

## Architecture

- **Backend** — FastAPI server with hybrid retrieval (BM25 + FAISS semantic search), cross-encoder reranking, and Gemini LLM streaming
- **Frontend** — React + Vite + Tailwind chat interface

## Project Structure

```
backend/
  main.py              # FastAPI app, endpoints, startup
  embedder.py          # Google Gemini Embedding API wrapper
  retrieval.py         # Hybrid retrieval (BM25 + FAISS + RRF) and reranking
  prompt.py            # Prompt construction, query translation, citation parsing
  requirements.txt
  Dockerfile           # Backend container (Cloud Run)
  .env.example         # Environment variable template
  data/                # Chunked documents, embeddings, FAISS indices
  notebooks/           # Jupyter notebooks (scraping, chunking, evaluation)
  scripts/             # Index-building scripts

frontend/
  src/
    App.tsx            # Main app, API streaming logic
    components/
      ChatMessage.tsx  # Message rendering + source links
      ChatInput.tsx    # Text input + send button
      SampleQuestions.tsx  # Sidebar with sample questions
  Dockerfile           # Frontend container (Nginx, for docker-compose)
  firebase.json        # Firebase Hosting config

docker-compose.yml     # Local multi-container setup
deploy.sh              # One-click GCP deployment script
```

## Quick Start

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "GOOGLE_API_KEY=your-key-here" > .env

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Open http://localhost:5173

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Google AI API key for Gemini LLM and embeddings |
| `EMBEDDING_MODEL` | No | `minilm` | Embedding model: `minilm` (local) or `gemini` (API) |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS origins (e.g., `https://your-domain.web.app`) |
| `VITE_API_URL` | No | `http://localhost:8000` | Backend URL for the frontend |

## Key Features

- **Hybrid retrieval** — combines BM25 lexical search with FAISS semantic search using Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — re-scores candidates with `ms-marco-MiniLM-L-6-v2` for higher precision
- **Synonym expansion** — maps domain terms (e.g., "tuition" ↔ "cost", "fee", "price") for better BM25 recall
- **Chinese query support** — automatically translates non-English queries to English for retrieval, responds in the user's language
- **Citation-based sources** — only shows source links the LLM actually referenced in its answer
- **Streaming responses** — real-time token-by-token output via Server-Sent Events
- **Configurable embeddings** — switch between local MiniLM (384-dim) and Gemini API (768-dim) via environment variable

## Deployment

The app is deployed on Google Cloud Platform:
- **Backend** — Cloud Run (containerized FastAPI)
- **Frontend** — Firebase Hosting (static files + CDN)

### Deploy with Docker (local)

```bash
docker-compose up --build
# Frontend: http://localhost  |  Backend: http://localhost:8000
```

### Deploy to GCP

Prerequisites: `gcloud` CLI, Firebase CLI, a GCP project with billing enabled.

```bash
# One-click deploy
./deploy.sh
```

The script builds and deploys the backend to Cloud Run, then builds the frontend and deploys to Firebase Hosting. See `deploy.sh` for details.
