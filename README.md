# TalentMind AI — Phase 1: Policy RAG

A grounded HR-policy question-answering service: ask a question, get an answer
**with source citations**, backed by semantic search over policy PDFs.

## Pipeline

```
PDFs ──▶ extract ──▶ chunk ──▶ embed + store (Qdrant) ──▶ /search & /chat API
```

| Step | File | What it does |
|------|------|--------------|
| 2. Extract | `ingestion/extract.py` | PDFs → per-page text JSON (+ `manifest.json`) |
| 3. Chunk | `ingestion/chunk.py` | Page text → ~500-token chunks with metadata |
| 4-5. Index | `scripts/index.py` | Embed chunks (local BGE) → store in Qdrant |
| 6-7. API | `backend/main.py` | `/search` (semantic) + `/chat` (RAG with Claude) |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure env (then add your Anthropic key)
cp .env.example .env

# 3. Start Qdrant
docker compose up -d
```

## Run the pipeline

```bash
python -m ingestion.extract     # PDFs -> data/processed/policies_text/
python -m ingestion.chunk       # -> data/processed/chunks/chunks.json
python -m scripts.index         # embed + store in Qdrant
```

> After `extract`, you can edit `data/processed/manifest.json` to set the real
> `department` / `country` / `version` per document, then re-run `chunk` + `index`.

## Run the API

```bash
uvicorn backend.main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Search: `GET /search?q=How many annual leaves are provided?`
- Chat: `POST /chat` with `{"question": "Can I work remotely?"}`

## Stack

- **Embeddings:** `BAAI/bge-base-en-v1.5` (local, 768-dim)
- **Vector store:** Qdrant
- **LLM:** Claude (`claude-opus-4-8`)
- **API:** FastAPI
