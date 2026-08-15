"""
RepoMind AI
Context Compression Layer

The retriever may return several code chunks.

Sending all of them directly to the LLM can:

- waste tokens
- introduce irrelevant information
- reduce answer quality

This module prepares a compact context while preserving
the most useful repository information.
"""

from __future__ import annotations

from typing import Any, Dict, List


# =========================================================
# Context Compressor
# =========================================================

class ContextCompressor:
    """
    Compress retrieved repository chunks into a clean
    context representation.
    """

    def __init__(
        self,
        max_context_characters: int = 18000,
        max_chunk_characters: int = 4500,
    ) -> None:

        self.max_context_characters = (
            max_context_characters
        )

        self.max_chunk_characters = (
            max_chunk_characters
        )


    # =====================================================
    # Compress
    # =====================================================

    def compress(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove redundant chunks and limit context size.
        """

        if not results:
            return []

        compressed = []

        seen_files = {}

        total_characters = 0

        for result in results:

            content = result.get(
                "content",
                ""
            )

            metadata = result.get(
                "metadata",
                {}
            )

            file_path = metadata.get(
                "file_path",
                "unknown"
            )

            if not content.strip():
                continue

            # Avoid repeatedly returning nearly identical
            # chunks from the same file.
            content = content.strip()

            if len(content) > self.max_chunk_characters:

                content = (
                    content[
                        :self.max_chunk_characters
                    ]
                    + "\n\n[Chunk truncated]"
                )

            # Very simple duplicate protection.
            normalized_content = (
                content.lower()
                .replace(" ", "")
                .replace("\n", "")
            )

            file_seen_contents = seen_files.setdefault(
                file_path,
                set(),
            )

            if normalized_content in file_seen_contents:
                continue

            file_seen_contents.add(
                normalized_content
            )

            remaining = (
                self.max_context_characters
                - total_characters
            )

            if remaining <= 0:
                break

            if len(content) > remaining:

                content = (
                    content[:remaining]
                    + "\n\n[Context limit reached]"
                )

            compressed_result = dict(
                result
            )

            compressed_result[
                "content"
            ] = content

            compressed.append(
                compressed_result
            )

            total_characters += len(
                content
            )

        return compressed


    # =====================================================
    # Build Prompt Context
    # =====================================================

    def build_context(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Convert compressed results into a structured text
        context for the LLM.
        """

        compressed_results = self.compress(
            results
        )

        if not compressed_results:
            return (
                "No relevant repository code was found."
            )

        sections = []

        for index, result in enumerate(
            compressed_results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            file_path = metadata.get(
                "file_path",
                "unknown"
            )

            language = metadata.get(
                "language",
                "unknown"
            )

            start_line = metadata.get(
                "start_line",
                "?"
            )

            end_line = metadata.get(
                "end_line",
                "?"
            )

            score = result.get(
                "rerank_score"
            )

            score_text = (
                f"{score:.4f}"
                if isinstance(
                    score,
                    (float, int)
                )
                else "N/A"
            )

            content = result.get(
                "content",
                ""
            )

            section = (
                f"===== SOURCE {index} =====\n"
                f"File: {file_path}\n"
                f"Language: {language}\n"
                f"Lines: {start_line}-{end_line}\n"
                f"Relevance Score: {score_text}\n"
                f"----- CODE -----\n"
                f"{content}\n"
                f"===== END SOURCE {index} ====="
            )

            sections.append(
                section
            )

        return "\n\n".join(
            sections
        )