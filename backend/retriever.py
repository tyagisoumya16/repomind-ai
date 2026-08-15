"""
RepoMind AI
Advanced Retrieval Layer

This module performs multi-stage retrieval:

1. Query expansion
2. Multiple semantic searches
3. Result merging
4. Duplicate removal
5. Repository filtering

The final ranking is handled separately by reranker.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.vector_store import VectorStore


# =========================================================
# Retriever
# =========================================================

class AdvancedRetriever:
    """
    Multi-query retrieval system.

    Instead of searching the vector database with only one
    query, this retriever can search using several related
    queries and merge the results.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
    ) -> None:

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )


    # =====================================================
    # Main Retrieval
    # =====================================================

    def retrieve(
        self,
        query: str,
        repository: str,
        expanded_queries: Optional[List[str]] = None,
        top_k_per_query: int = 6,
        max_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate chunks using multiple queries.

        Example:

            Original:
                "How does authentication work?"

            Expanded:
                "authentication"
                "login"
                "JWT"
                "token validation"

        Results from all searches are merged.
        """

        if not query or not query.strip():
            return []

        if not repository or not repository.strip():
            return []

        queries = self._build_query_list(
            original_query=query,
            expanded_queries=expanded_queries,
        )

        all_results: List[Dict[str, Any]] = []

        for current_query in queries:

            results = self.vector_store.search_repository(
                query=current_query,
                repository=repository,
                top_k=top_k_per_query,
            )

            for result in results:

                result["retrieval_query"] = (
                    current_query
                )

                all_results.append(result)

        merged_results = self._merge_results(
            all_results
        )

        return merged_results[
            :max_candidates
        ]


    # =====================================================
    # Query List
    # =====================================================

    @staticmethod
    def _build_query_list(
        original_query: str,
        expanded_queries: Optional[List[str]],
    ) -> List[str]:
        """
        Build a clean list of unique retrieval queries.
        """

        queries = [
            original_query.strip()
        ]

        if expanded_queries:

            for query in expanded_queries:

                if not query:
                    continue

                query = query.strip()

                if not query:
                    continue

                if query.lower() not in {
                    item.lower()
                    for item in queries
                }:
                    queries.append(query)

        return queries


    # =====================================================
    # Merge Results
    # =====================================================

    @staticmethod
    def _merge_results(
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge duplicate chunks returned by different
        queries.

        If the same chunk is found multiple times, the best
        distance is retained.
        """

        unique_results: Dict[
            str,
            Dict[str, Any]
        ] = {}

        for result in results:

            result_id = result.get(
                "id"
            )

            if not result_id:
                continue

            if result_id not in unique_results:

                unique_results[
                    result_id
                ] = result

                continue

            existing = unique_results[
                result_id
            ]

            existing_distance = existing.get(
                "distance"
            )

            current_distance = result.get(
                "distance"
            )

            if (
                current_distance is not None
                and (
                    existing_distance is None
                    or current_distance
                    < existing_distance
                )
            ):
                unique_results[
                    result_id
                ] = result

        merged = list(
            unique_results.values()
        )

        # Lower vector distance means greater similarity.
        merged.sort(
            key=lambda item: (
                item.get(
                    "distance"
                )
                if item.get("distance") is not None
                else float("inf")
            )
        )

        return merged


    # =====================================================
    # File-focused Retrieval
    # =====================================================

    def retrieve_by_file(
        self,
        query: str,
        repository: str,
        file_path: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve code only from a specific file.

        This is useful for agent tools when the agent already
        knows which file needs deeper investigation.
        """

        results = self.vector_store.search_repository(
            query=query,
            repository=repository,
            top_k=top_k * 3,
        )

        filtered = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            if metadata.get(
                "file_path"
            ) == file_path:

                filtered.append(
                    result
                )

            if len(filtered) >= top_k:
                break

        return filtered


    # =====================================================
    # Language-focused Retrieval
    # =====================================================

    def retrieve_by_language(
        self,
        query: str,
        repository: str,
        language: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results and prioritize a specific language.
        """

        results = self.vector_store.search_repository(
            query=query,
            repository=repository,
            top_k=top_k * 3,
        )

        filtered = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            if (
                metadata.get(
                    "language",
                    ""
                ).lower()
                == language.lower()
            ):

                filtered.append(
                    result
                )

            if len(filtered) >= top_k:
                break

        return filtered