"""Steps 4 & 5 - Embed chunks and store them in Qdrant.

Run:  python -m scripts.index
(Make sure Qdrant is running: docker compose up -d)
"""
import json

import config
from embeddings.embedder import Embedder
from retrievers import qdrant_store


def main():
    chunks_file = config.PROCESSED_CHUNKS / "chunks.json"
    if not chunks_file.exists():
        print("chunks.json not found. Run: python -m ingestion.extract && python -m ingestion.chunk")
        return

    chunks = json.loads(chunks_file.read_text())
    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    embedder = Embedder()
    vectors = embedder.embed_documents([c["text"] for c in chunks])

    client = qdrant_store.get_client()
    qdrant_store.recreate_collection(client)
    qdrant_store.upsert_chunks(client, chunks, vectors)
    print(f"Stored {len(chunks)} chunks in Qdrant collection '{config.QDRANT_COLLECTION}'")


if __name__ == "__main__":
    main()
