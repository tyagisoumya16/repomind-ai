from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL


class EmbeddingService:
    """
    Handles text embeddings for both documents and queries.

    Documents and queries MUST use the same embedding model
    so that vector similarity works correctly.
    """

    def __init__(self) -> None:

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    # =====================================================
    # Embed Multiple Documents
    # =====================================================

    def embed_documents(
        self,
        documents: List[str],
    ) -> List[List[float]]:
        """
        Convert multiple documents into embeddings.
        """

        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    # =====================================================
    # Embed Single Query
    # =====================================================

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Convert a single user query into an embedding.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()