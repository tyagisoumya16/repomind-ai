"""
RepoMind AI
Re-ranking Layer

Initial vector search is good at finding candidates, but
the first few results are not always the most useful ones.

This module performs a lightweight second-stage ranking.

The implementation uses lexical relevance combined with
the original vector similarity distance.

This keeps the system lightweight and avoids introducing
another large ML model.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List


# =========================================================
# Text Processing
# =========================================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "how",
    "what",
    "why",
    "where",
    "when",
    "does",
    "do",
    "this",
    "that",
    "from",
    "is",
    "it",
}


def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    tokens = re.findall(
        r"[A-Za-z0-9_]+",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


# =========================================================
# Re-ranker
# =========================================================

class CodeReranker:
    """
    Lightweight second-stage re-ranker.

    Ranking combines:

    - vector similarity
    - keyword overlap
    - file path relevance
    - exact phrase relevance
    """

    def __init__(
        self,
        vector_weight: float = 0.55,
        lexical_weight: float = 0.30,
        path_weight: float = 0.15,
    ) -> None:

        total = (
            vector_weight
            + lexical_weight
            + path_weight
        )

        if not math.isclose(
            total,
            1.0,
            abs_tol=0.001,
        ):
            raise ValueError(
                "Reranker weights must add up to 1.0."
            )

        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.path_weight = path_weight


    # =====================================================
    # Main Ranking
    # =====================================================

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidate chunks and return the best results.
        """

        if not candidates:
            return []

        query_tokens = set(
            tokenize(query)
        )

        ranked = []

        for candidate in candidates:

            content = candidate.get(
                "content",
                ""
            )

            metadata = candidate.get(
                "metadata",
                {}
            )

            file_path = metadata.get(
                "file_path",
                ""
            )

            distance = candidate.get(
                "distance"
            )

            vector_score = (
                self._distance_to_similarity(
                    distance
                )
            )

            lexical_score = (
                self._lexical_score(
                    query_tokens,
                    content
                )
            )

            path_score = (
                self._path_score(
                    query_tokens,
                    file_path
                )
            )

            final_score = (
                self.vector_weight
                * vector_score
                +
                self.lexical_weight
                * lexical_score
                +
                self.path_weight
                * path_score
            )

            updated_candidate = dict(
                candidate
            )

            updated_candidate[
                "vector_score"
            ] = vector_score

            updated_candidate[
                "lexical_score"
            ] = lexical_score

            updated_candidate[
                "path_score"
            ] = path_score

            updated_candidate[
                "rerank_score"
            ] = final_score

            ranked.append(
                updated_candidate
            )

        ranked.sort(
            key=lambda item: item[
                "rerank_score"
            ],
            reverse=True,
        )

        return ranked[:top_k]


    # =====================================================
    # Vector Similarity
    # =====================================================

    @staticmethod
    def _distance_to_similarity(
        distance: Any
    ) -> float:
        """
        Convert Chroma distance into a normalized similarity.

        Smaller distance = more similar.

        Formula:

            similarity = 1 / (1 + distance)
        """

        if distance is None:
            return 0.0

        try:
            distance = float(
                distance
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if distance < 0:
            distance = 0

        return 1.0 / (
            1.0 + distance
        )


    # =====================================================
    # Lexical Score
    # =====================================================

    @staticmethod
    def _lexical_score(
        query_tokens: set[str],
        content: str,
    ) -> float:
        """
        Calculate keyword overlap between query and content.
        """

        if not query_tokens:
            return 0.0

        content_tokens = set(
            tokenize(content)
        )

        if not content_tokens:
            return 0.0

        overlap = (
            query_tokens
            & content_tokens
        )

        return len(overlap) / len(
            query_tokens
        )


    # =====================================================
    # Path Score
    # =====================================================

    @staticmethod
    def _path_score(
        query_tokens: set[str],
        file_path: str,
    ) -> float:
        """
        Check whether query keywords appear in the file path.

        Example:

            Query:
                authentication

            File:
                backend/authentication.py

        gets a higher score.
        """

        if not query_tokens:
            return 0.0

        path_tokens = set(
            tokenize(file_path)
        )

        if not path_tokens:
            return 0.0

        overlap = (
            query_tokens
            & path_tokens
        )

        return min(
            1.0,
            len(overlap)
            / len(query_tokens)
        )