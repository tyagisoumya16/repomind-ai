"""
RepoMind AI
LangChain Tools

These tools are exposed to the Agent.

The agent can decide which tool should be used
based on the user's question.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.tools import tool

from backend.retriever import AdvancedRetriever
from backend.reranker import CodeReranker
from backend.query_processor import QueryProcessor
from backend.context_compressor import ContextCompressor


# =========================================================
# Shared Services
# =========================================================

retriever = AdvancedRetriever()

reranker = CodeReranker()

query_processor = QueryProcessor()

context_compressor = ContextCompressor()


# =========================================================
# Active Repository
# =========================================================

_current_repository = ""


def set_current_repository(
    repository: str,
) -> None:
    """
    Set the repository currently being analyzed.
    """

    global _current_repository

    _current_repository = repository.strip()


def get_current_repository() -> str:
    """
    Return the currently selected repository.
    """

    return _current_repository


# =========================================================
# Result Formatter
# =========================================================

def _format_tool_results(
    results: List[Dict[str, Any]],
) -> str:
    """
    Convert retrieval results into compact JSON so the
    LangChain agent can reason over them.
    """

    if not results:
        return json.dumps(
            {
                "results": [],
                "message": (
                    "No relevant code was found."
                ),
            },
            indent=2,
        )

    formatted = []

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        formatted.append(
            {
                "file": metadata.get(
                    "file_path",
                    "unknown",
                ),
                "language": metadata.get(
                    "language",
                    "unknown",
                ),
                "start_line": metadata.get(
                    "start_line",
                    None,
                ),
                "end_line": metadata.get(
                    "end_line",
                    None,
                ),
                "score": result.get(
                    "rerank_score",
                    result.get("distance"),
                ),
                "code": result.get(
                    "content",
                    "",
                ),
            }
        )

    return json.dumps(
        {
            "results": formatted,
        },
        indent=2,
    )


# =========================================================
# Tool 1 — Search Code
# =========================================================

@tool
def search_code(
    query: str,
) -> str:
    """
    Search the repository for code relevant to a question.

    Use this tool when you need to understand how a feature,
    concept, class, function, API, database operation,
    authentication flow, or other functionality works.
    """

    repository = get_current_repository()

    if not repository:
        return (
            "No repository is currently selected."
        )

    expanded_queries = (
        query_processor.process(
            query
        )
    )

    candidates = retriever.retrieve(
        query=query,
        repository=repository,
        expanded_queries=expanded_queries,
        top_k_per_query=6,
        max_candidates=20,
    )

    ranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=8,
    )

    compressed = (
        context_compressor.compress(
            ranked
        )
    )

    return _format_tool_results(
        compressed
    )


# =========================================================
# Tool 2 — Search Specific File
# =========================================================

@tool
def search_file(
    file_path: str,
    query: str,
) -> str:
    """
    Search inside a specific repository file.

    Use this when the user asks about a particular file
    or when another tool has identified a file that needs
    deeper investigation.
    """

    repository = get_current_repository()

    if not repository:
        return (
            "No repository is currently selected."
        )

    results = retriever.retrieve_by_file(
        query=query,
        repository=repository,
        file_path=file_path,
        top_k=6,
    )

    ranked = reranker.rerank(
        query=query,
        candidates=results,
        top_k=6,
    )

    compressed = (
        context_compressor.compress(
            ranked
        )
    )

    return _format_tool_results(
        compressed
    )


# =========================================================
# Tool 3 — Search By Language
# =========================================================

@tool
def search_by_language(
    query: str,
    language: str,
) -> str:
    """
    Search repository code while focusing on a specific
    programming language.

    Example:

        query = "authentication"
        language = "python"
    """

    repository = get_current_repository()

    if not repository:
        return (
            "No repository is currently selected."
        )

    results = (
        retriever.retrieve_by_language(
            query=query,
            repository=repository,
            language=language,
            top_k=8,
        )
    )

    ranked = reranker.rerank(
        query=query,
        candidates=results,
        top_k=6,
    )

    compressed = (
        context_compressor.compress(
            ranked
        )
    )

    return _format_tool_results(
        compressed
    )


# =========================================================
# Tool 4 — Repository Structure
# =========================================================

@tool
def repository_structure() -> str:
    """
    Return the files currently indexed in the repository.

    Use this tool when the user asks:

    - What files are in the project?
    - Where is authentication implemented?
    - What is the project structure?
    - Which files are available?
    """

    repository = get_current_repository()

    if not repository:
        return (
            "No repository is currently selected."
        )

    # Search with an intentionally broad query so we can
    # recover a representative set of indexed files.
    results = retriever.vector_store.search_repository(
        query="source code project files implementation",
        repository=repository,
        top_k=100,
    )

    files = set()

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        file_path = metadata.get(
            "file_path"
        )

        if file_path:
            files.add(file_path)

    sorted_files = sorted(
        files
    )

    if not sorted_files:
        return (
            "No indexed files were found."
        )

    return json.dumps(
        {
            "repository": repository,
            "files": sorted_files,
        },
        indent=2,
    )


# =========================================================
# Tool Collection
# =========================================================

def get_tools() -> List[Any]:
    """
    Return all tools available to the RepoMind Agent.
    """

    return [
        search_code,
        search_file,
        search_by_language,
        repository_structure,
    ]