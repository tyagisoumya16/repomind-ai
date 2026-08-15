from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb

from backend.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
)
from backend.embeddings import EmbeddingService


class VectorStore:

    def __init__(self) -> None:

        self.embedding_service = EmbeddingService()

        self.client = None
        self.collection = None

    # =====================================================
    # Lazy Chroma Initialization
    # =====================================================

    def _initialize(self) -> None:

        if self.client is not None:
            return

        Path(CHROMA_PATH).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    # =====================================================
    # Add Chunks
    # =====================================================

    def add_chunks(
            self,
            chunks,
    ) -> int:

        if not chunks:
            return 0

        self._initialize()

        documents = []
        ids = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            # CodeChunk object se values read karo
            content = getattr(
                chunk,
                "content",
                "",
            )

            metadata = getattr(
                chunk,
                "metadata",
                {},
            )

            chunk_id = getattr(
                chunk,
                "id",
                None,
            )

            # Agar id available nahi hai
            if not chunk_id:
                chunk_id = f"chunk-{index}"

            if not content:
                continue

            content = str(content).strip()

            if not content:
                continue

            # Metadata ko normal dictionary banao
            if metadata is None:
                metadata = {}

            if not isinstance(metadata, dict):
                metadata = dict(metadata)

            documents.append(content)

            ids.append(
                str(chunk_id)
            )

            metadatas.append(metadata)

        if not documents:
            return 0

        embeddings = (
            self.embedding_service.embed_documents(
                documents
            )
        )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(documents)

    # =====================================================
    # Search Repository
    # =====================================================

    def search_repository(
        self,
        query: str,
        repository: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        self._initialize()

        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        result = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            where={
                "repository": repository
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = (
            result.get("documents", [[]])[0]
        )

        metadatas = (
            result.get("metadatas", [[]])[0]
        )

        distances = (
            result.get("distances", [[]])[0]
        )

        ids = (
            result.get("ids", [[]])[0]
        )

        results = []

        for index, document in enumerate(
            documents
        ):

            results.append(
                {
                    "id": ids[index],
                    "content": document,
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )

        return results

    # =====================================================
    # Delete Repository
    # =====================================================

    def delete_repository(
        self,
        repository: str,
    ) -> None:

        self._initialize()

        existing = (
            self.collection.get(
                where={
                    "repository": repository
                }
            )
        )

        ids = existing.get(
            "ids",
            [],
        )

        if ids:
            self.collection.delete(
                ids=ids
            )