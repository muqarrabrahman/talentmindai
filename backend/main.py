"""Steps 6 & 7 - Policy search API + grounded RAG chat API.

Run:  uvicorn backend.main:app --reload
Docs: http://localhost:8000/docs
"""
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

import config
from embeddings.embedder import Embedder
from retrievers import qdrant_store

app = FastAPI(title="TalentMind AI - Policy RAG")

# Loaded once at startup.
embedder = Embedder()
qdrant = qdrant_store.get_client()

SYSTEM_PROMPT = (
    "You are an HR policy assistant. Answer the employee's question using ONLY the "
    "policy excerpts in the context below. If the answer is not in the context, say you "
    "don't have that information rather than guessing. Keep answers short, and cite the "
    "source document name and page number for every fact you state."
)


def format_context(hits):
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        p = hit.payload
        source = f"{p['metadata']['document_name']}, page {p['page_start']}"
        blocks.append(f"[{i}] ({source})\n{p['text']}")
    return "\n\n".join(blocks)


def build_citations(hits):
    return [
        {
            "document_name": h.payload["metadata"]["document_name"],
            "page": h.payload["page_start"],
            "score": h.score,
        }
        for h in hits
    ]


@app.get("/search")
def search(q: str, top_k: int = config.TOP_K):
    """Semantic search over policy chunks (no LLM)."""
    vector = embedder.embed_query(q)
    hits = qdrant_store.search(qdrant, vector, top_k)
    return [{"score": h.score, **h.payload} for h in hits]


class ChatRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K


@app.post("/chat")
def chat(req: ChatRequest):
    """Retrieve relevant policy chunks, then ask Claude to answer from them."""
    vector = embedder.embed_query(req.question)
    hits = qdrant_store.search(qdrant, vector, req.top_k)
    context = format_context(hits)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {req.question}",
        }],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")
    return {"answer": answer, "citations": build_citations(hits)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
