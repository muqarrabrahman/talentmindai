"""Local embedding model wrapper (BAAI/bge-base-en-v1.5).

Loads the model once on construction. Vectors are L2-normalized so cosine
similarity == dot product, which is what the Qdrant collection uses.
"""
from sentence_transformers import SentenceTransformer

import config


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_documents(self, texts):
        """Embed a list of passages. Returns a list of float lists."""
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text):
        """Embed a single query (with the BGE query instruction prefix)."""
        vector = self.model.encode(config.QUERY_PREFIX + text, normalize_embeddings=True)
        return vector.tolist()
