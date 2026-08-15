"""
RepoMind AI
Query Processing Layer

Responsible for turning a user's natural-language question
into a small set of retrieval-friendly queries.

Example:

User:
    "How does authentication work?"

Generated retrieval queries:

    authentication
    login implementation
    JWT token validation
    authorization middleware

This improves recall before re-ranking.
"""

from __future__ import annotations

import re
from typing import List


# =========================================================
# Query Processor
# =========================================================

class QueryProcessor:
    """
    Lightweight query expansion system.

    It does not call an LLM. This keeps retrieval fast and
    inexpensive.

    The LLM-based reasoning will happen later inside the
    Agentic RAG layer.
    """

    # Domain terms useful for repository exploration.
    CONCEPT_EXPANSIONS = {
        "authentication": [
            "authentication",
            "login",
            "JWT",
            "token validation",
            "authorization",
        ],
        "login": [
            "login",
            "authentication",
            "user credentials",
            "token",
            "session",
        ],
        "database": [
            "database",
            "DB",
            "repository",
            "query",
            "model",
            "ORM",
        ],
        "api": [
            "API",
            "endpoint",
            "route",
            "controller",
            "request",
            "response",
        ],
        "error": [
            "error",
            "exception",
            "try catch",
            "error handling",
            "logging",
        ],
        "testing": [
            "test",
            "unit test",
            "integration test",
            "pytest",
            "mock",
        ],
        "configuration": [
            "configuration",
            "config",
            "environment variables",
            ".env",
            "settings",
        ],
        "deployment": [
            "deployment",
            "Docker",
            "CI/CD",
            "production",
            "server",
        ],
    }


    # =====================================================
    # Process Query
    # =====================================================

    def process(
        self,
        query: str,
        max_queries: int = 5,
    ) -> List[str]:
        """
        Generate retrieval queries.

        The original query is always retained.
        """

        if not query or not query.strip():
            return []

        original = query.strip()

        queries = [
            original
        ]

        normalized = original.lower()

        # Add concept-specific expansions.
        for concept, expansions in (
            self.CONCEPT_EXPANSIONS.items()
        ):

            if concept in normalized:

                for expansion in expansions:

                    self._append_unique(
                        queries,
                        expansion,
                    )

        # Extract technical terms such as:
        #
        # JWT
        # FastAPI
        # authentication.py
        # getUser()
        # API
        #
        technical_terms = (
            self._extract_technical_terms(
                original
            )
        )

        for term in technical_terms:

            self._append_unique(
                queries,
                term,
            )

        # Add a focused semantic version.
        focused_query = (
            self._create_focused_query(
                original
            )
        )

        if focused_query:
            self._append_unique(
                queries,
                focused_query,
            )

        return queries[:max_queries]


    # =====================================================
    # Technical Terms
    # =====================================================

    @staticmethod
    def _extract_technical_terms(
        query: str
    ) -> List[str]:
        """
        Extract likely technical identifiers.
        """

        terms = re.findall(
            r"""
            [A-Za-z_][A-Za-z0-9_.-]*
            """,
            query,
            flags=re.VERBOSE,
        )

        result = []

        for term in terms:

            # Ignore normal English words.
            if len(term) < 3:
                continue

            if (
                term.lower()
                in {
                    "the",
                    "how",
                    "what",
                    "why",
                    "does",
                    "work",
                    "with",
                    "from",
                    "this",
                    "that",
                    "and",
                    "for",
                }
            ):
                continue

            result.append(term)

        return result


    # =====================================================
    # Focused Query
    # =====================================================

    @staticmethod
    def _create_focused_query(
        query: str
    ) -> str:
        """
        Remove conversational filler from the query.
        """

        filler_patterns = [
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bplease\b",
            r"\bexplain\b",
            r"\btell me\b",
            r"\bshow me\b",
            r"\bhow does\b",
            r"\bhow do\b",
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bwhere is\b",
        ]

        focused = query.lower()

        for pattern in filler_patterns:

            focused = re.sub(
                pattern,
                "",
                focused,
            )

        focused = re.sub(
            r"\s+",
            " ",
            focused,
        ).strip()

        return focused


    # =====================================================
    # Unique Append
    # =====================================================

    @staticmethod
    def _append_unique(
        queries: List[str],
        value: str,
    ) -> None:
        """
        Append a query only if it does not already exist.
        """

        value = value.strip()

        if not value:
            return

        existing = {
            query.lower()
            for query in queries
        }

        if value.lower() not in existing:
            queries.append(value)