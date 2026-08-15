"""
RepoMind AI
Application configuration.

This file is responsible for loading environment variables
and exposing them to the rest of the application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# Project Paths
# =========================================================

# backend/config.py
#        ↓
# backend
#        ↓
# project root

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# Load Environment Variables
# =========================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# =========================================================
# Application Configuration
# =========================================================

APP_NAME = os.getenv(
    "APP_NAME",
    "RepoMind AI"
)

DEBUG = os.getenv(
    "DEBUG",
    "false"
).lower() == "true"


# =========================================================
# Groq Configuration
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# =========================================================
# GitHub Configuration
# =========================================================

GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    ""
)


# =========================================================
# Embedding Configuration
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# =========================================================
# ChromaDB Configuration
# =========================================================

CHROMA_PATH_VALUE = os.getenv(
    "CHROMA_PATH",
    "data/chroma"
)

CHROMA_PATH = BASE_DIR / CHROMA_PATH_VALUE


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "repomind_code"
)


# =========================================================
# Repository Storage
# =========================================================

REPOSITORY_PATH = BASE_DIR / "data" / "repositories"


# =========================================================
# Create Required Directories
# =========================================================

CHROMA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

REPOSITORY_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Validation
# =========================================================

def validate_configuration() -> None:
    """
    Validate required application configuration.

    Currently Groq API key is required because the final
    application uses Groq as the LLM provider.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add your Groq API key to the .env file."
        )