"""
RepoMind AI
FastAPI Backend

Endpoints:

POST /repository/index
    Index a GitHub repository.

POST /ask
    Ask RepoMind AI a question about the indexed repository.

GET /health
    Health check.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent import get_agent
from backend.chunker import chunk_repository_files
from backend.github_service import GitHubService
from backend.vector_store import VectorStore


# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="RepoMind AI",
    description=(
        "Agentic RAG system for understanding "
        "GitHub repositories."
    ),
    version="1.0.0",
)


# =========================================================
# Services
# =========================================================

github_service = GitHubService()

vector_store = VectorStore()


# =========================================================
# Request Models
# =========================================================

class IndexRequest(BaseModel):
    """
    Request for repository indexing.
    """

    repository_url: str = Field(
        ...,
        min_length=10,
        description=(
            "Public GitHub repository URL."
        ),
    )

    max_files: int = Field(
        default=300,
        ge=1,
        le=500,
    )


class AskRequest(BaseModel):
    """
    Request for repository question answering.
    """

    repository: str = Field(
        ...,
        min_length=1,
    )

    question: str = Field(
        ...,
        min_length=2,
    )


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health() -> dict:
    """
    Check whether the API is running.
    """

    return {
        "status": "ok",
        "service": "RepoMind AI",
    }


# =========================================================
# Index Repository
# =========================================================

@app.post("/repository/index")
def index_repository(
    request: IndexRequest,
) -> dict:
    """
    Download, chunk, embed, and index a GitHub repository.
    """

    try:

        repository_info = (
            github_service.get_repository_info(
                request.repository_url
            )
        )

        repository_name = (
            repository_info["full_name"]
        )

        # -------------------------------------------------
        # Load repository files
        # -------------------------------------------------

        repository_files = (
            github_service.load_repository(
                repository_url=request.repository_url,
                max_files=request.max_files,
            )
        )

        # -------------------------------------------------
        # Chunk files
        # -------------------------------------------------

        chunks = chunk_repository_files(
            repository_files=repository_files,
            repository_name=repository_name,
        )

        if not chunks:
            raise ValueError(
                "No usable code chunks were created."
            )

        # -------------------------------------------------
        # Replace previous index for this repository
        # -------------------------------------------------

        vector_store.delete_repository(
            repository_name
        )

        # -------------------------------------------------
        # Store embeddings
        # -------------------------------------------------

        added_chunks = (
            vector_store.add_chunks(
                chunks
            )
        )

        return {
            "success": True,
            "repository": repository_name,
            "files_indexed": len(
                repository_files
            ),
            "chunks_indexed": added_chunks,
            "default_branch": (
                repository_info[
                    "default_branch"
                ]
            ),
            "language": repository_info[
                "language"
            ],
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Repository indexing failed: "
                f"{str(exc)}"
            ),
        ) from exc


# =========================================================
# Ask Repository
# =========================================================

@app.post("/ask")
def ask_repository(
    request: AskRequest,
) -> dict:
    """
    Ask the Agent a question about an indexed repository.
    """

    try:

        agent = get_agent()

        answer = agent.ask(
            question=request.question,
            repository=request.repository,
        )

        return {
            "success": True,
            "repository": request.repository,
            "question": request.question,
            "answer": answer,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Agent request failed: "
                f"{str(exc)}"
            ),
        ) from exc