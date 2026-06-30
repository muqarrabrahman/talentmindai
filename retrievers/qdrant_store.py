"""Qdrant vector store helpers: connect, (re)create collection, upsert, search."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config

# Fixed namespace so a chunk_id always maps to the same point id.
# Re-ingesting a changed file overwrites the same points instead of duplicating.
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")


def get_client():
    return QdrantClient(url=config.QDRANT_URL)


def recreate_collection(client):
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )


def upsert_chunks(client, chunks, vectors):
    points = [
        PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, chunk["chunk_id"])),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def search(client, vector, top_k):
    # query_points replaces the removed .search() in recent qdrant-client.
    # .points is a list of ScoredPoint (each has .score and .payload).
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points
