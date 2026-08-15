"""
RepoMind AI
Code-aware document chunking.

This module converts repository files into smaller pieces
that can be embedded and retrieved efficiently.

Each chunk keeps useful metadata such as:

- file path
- programming language
- chunk number
- repository
- source URL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.github_service import RepositoryFile


# =========================================================
# Configuration
# =========================================================

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

MIN_CHUNK_SIZE = 50


# =========================================================
# Data Model
# =========================================================

@dataclass
class CodeChunk:
    """
    Represents one searchable piece of repository code.
    """

    chunk_id: str
    content: str
    metadata: dict


# =========================================================
# Language Detection
# =========================================================

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".toml": "toml",
    ".ini": "ini",
}


def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    """

    suffix = Path(file_path).suffix.lower()

    return EXTENSION_LANGUAGE_MAP.get(
        suffix,
        "text"
    )


# =========================================================
# Code Chunker
# =========================================================

class CodeChunker:
    """
    Splits source files into overlapping chunks.

    The first version intentionally uses a lightweight
    line-based strategy so that RepoMind can work with
    multiple programming languages without requiring a
    parser for every language.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    # =====================================================
    # Single File
    # =====================================================

    def chunk_file(
        self,
        repository_file: RepositoryFile,
        repository_name: str,
    ) -> List[CodeChunk]:
        """
        Convert one repository file into CodeChunk objects.
        """

        content = repository_file.content

        if not content.strip():
            return []

        lines = content.splitlines()

        if not lines:
            return []

        language = detect_language(
            repository_file.path
        )

        chunks: List[CodeChunk] = []

        start_line = 0
        chunk_number = 0

        total_lines = len(lines)

        while start_line < total_lines:

            current_lines: List[str] = []

            current_length = 0

            end_line = start_line

            while end_line < total_lines:

                line = lines[end_line]

                line_length = len(line) + 1

                # If adding this line would make the chunk
                # too large, stop and create the chunk.
                if (
                    current_lines
                    and current_length + line_length
                    > self.chunk_size
                ):
                    break

                current_lines.append(line)

                current_length += line_length

                end_line += 1

                # Very long individual lines should not
                # prevent a chunk from being created.
                if current_length >= self.chunk_size:
                    break

            chunk_content = "\n".join(
                current_lines
            ).strip()

            if len(chunk_content) >= MIN_CHUNK_SIZE:

                chunk_id = (
                    f"{repository_name}:"
                    f"{repository_file.path}:"
                    f"{chunk_number}"
                )

                metadata = {
                    "repository": repository_name,
                    "file_path": repository_file.path,
                    "language": language,
                    "chunk_number": chunk_number,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                    "source_url": repository_file.url,
                }

                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        metadata=metadata,
                    )
                )

            # End of file reached.
            if end_line >= total_lines:
                break

            # Convert overlap from characters into a
            # conservative number of lines.
            overlap_lines = self._calculate_overlap_lines(
                lines=lines,
                start_line=start_line,
                end_line=end_line,
            )

            next_start = end_line - overlap_lines

            # Safety check: always move forward.
            if next_start <= start_line:
                next_start = end_line

            start_line = next_start

            chunk_number += 1

        return chunks


    # =====================================================
    # Multiple Files
    # =====================================================

    def chunk_repository(
        self,
        repository_files: List[RepositoryFile],
        repository_name: str,
    ) -> List[CodeChunk]:
        """
        Chunk all repository files.
        """

        all_chunks: List[CodeChunk] = []

        for repository_file in repository_files:

            file_chunks = self.chunk_file(
                repository_file=repository_file,
                repository_name=repository_name,
            )

            all_chunks.extend(
                file_chunks
            )

        return all_chunks


    # =====================================================
    # Overlap Calculation
    # =====================================================

    def _calculate_overlap_lines(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
    ) -> int:
        """
        Estimate how many lines should overlap.

        We walk backward from the end of the chunk until
        approximately chunk_overlap characters are included.
        """

        overlap_length = 0
        overlap_lines = 0

        index = end_line - 1

        while index >= start_line:

            overlap_length += (
                len(lines[index]) + 1
            )

            overlap_lines += 1

            if overlap_length >= self.chunk_overlap:
                break

            index -= 1

        return overlap_lines


# =========================================================
# Convenience Function
# =========================================================

def chunk_repository_files(
    repository_files: List[RepositoryFile],
    repository_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[CodeChunk]:
    """
    Convenience wrapper around CodeChunker.
    """

    chunker = CodeChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk_repository(
        repository_files=repository_files,
        repository_name=repository_name,
    )