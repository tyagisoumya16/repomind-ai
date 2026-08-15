"""
RepoMind AI
GitHub Repository Service

Responsibilities:

1. Parse GitHub repository URLs.
2. Communicate with GitHub API.
3. Fetch repository files.
4. Filter files useful for code understanding.
5. Return clean repository data.

This module does NOT perform RAG.
It only handles GitHub repository ingestion.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import requests

from backend.config import GITHUB_TOKEN


# =========================================================
# Constants
# =========================================================

GITHUB_API = "https://api.github.com"

# Files that usually do not provide useful information
# for code understanding.
IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".next",
    ".venv",
    "venv",
    "env",
}

# Extensions that RepoMind can understand as source/code/text.
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".toml",
    ".ini",
    ".env.example",
}

# Files which are generally useful even when they do not
# have a normal source-code extension.
IMPORTANT_FILENAMES = {
    "README",
    "README.md",
    "Dockerfile",
    "Makefile",
    "requirements.txt",
    "package.json",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
}


# =========================================================
# Data Model
# =========================================================

@dataclass
class RepositoryFile:
    """
    Represents one file extracted from a GitHub repository.
    """

    path: str
    content: str
    sha: str
    url: str


# =========================================================
# GitHub Service
# =========================================================

class GitHubService:
    """
    Service responsible for communicating with GitHub.
    """

    def __init__(
        self,
        token: Optional[str] = None
    ) -> None:

        self.token = token or GITHUB_TOKEN

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "RepoMind-AI",
            }
        )

        if self.token:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.token}"
                }
            )


    # =====================================================
    # URL Parsing
    # =====================================================

    @staticmethod
    def parse_repository_url(
        repository_url: str
    ) -> tuple[str, str]:
        """
        Extract owner and repository name from a GitHub URL.

        Accepted examples:

        https://github.com/user/project
        https://github.com/user/project/
        http://github.com/user/project

        Returns:

            ("user", "project")
        """

        repository_url = repository_url.strip()

        parsed = urlparse(repository_url)

        if parsed.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError(
                "Please provide a valid GitHub repository URL."
            )

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if len(parts) < 2:
            raise ValueError(
                "Invalid GitHub repository URL. "
                "Expected format: "
                "https://github.com/owner/repository"
            )

        owner = parts[0]
        repository = parts[1]

        if repository.endswith(".git"):
            repository = repository[:-4]

        if not owner or not repository:
            raise ValueError(
                "Could not determine GitHub owner and repository."
            )

        return owner, repository


    # =====================================================
    # API Request
    # =====================================================

    def _get(
        self,
        endpoint: str
    ) -> requests.Response:
        """
        Perform a GET request against GitHub API.
        """

        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{GITHUB_API}{endpoint}"
        )

        response = self.session.get(
            url,
            timeout=30
        )

        if response.status_code == 404:
            raise ValueError(
                "GitHub repository or file was not found."
            )

        if response.status_code == 403:
            raise ValueError(
                "GitHub API rate limit exceeded. "
                "Add a GITHUB_TOKEN to the .env file."
            )

        if not response.ok:
            raise ValueError(
                f"GitHub API request failed: "
                f"{response.status_code} "
                f"{response.text[:300]}"
            )

        return response


    # =====================================================
    # Repository Information
    # =====================================================

    def get_repository_info(
        self,
        repository_url: str
    ) -> dict:
        """
        Get basic repository information.
        """

        owner, repository = self.parse_repository_url(
            repository_url
        )

        response = self._get(
            f"/repos/{owner}/{repository}"
        )

        data = response.json()

        return {
            "owner": owner,
            "name": data.get("name", repository),
            "full_name": data.get(
                "full_name",
                f"{owner}/{repository}"
            ),
            "description": data.get(
                "description"
            ),
            "default_branch": data.get(
                "default_branch",
                "main"
            ),
            "language": data.get(
                "language"
            ),
            "stars": data.get(
                "stargazers_count",
                0
            ),
            "forks": data.get(
                "forks_count",
                0
            ),
            "html_url": data.get(
                "html_url",
                repository_url
            ),
        }


    # =====================================================
    # File Filtering
    # =====================================================

    @staticmethod
    def is_supported_file(
        path: str
    ) -> bool:
        """
        Determine whether a repository file should be
        included in RepoMind's knowledge base.
        """

        normalized_path = path.replace(
            "\\",
            "/"
        )

        path_parts = normalized_path.split("/")

        # Ignore unwanted directories.
        for directory in path_parts:
            if directory.lower() in IGNORED_DIRECTORIES:
                return False

        filename = path_parts[-1]

        # Important project files.
        if filename in IMPORTANT_FILENAMES:
            return True

        # Normal extension based filtering.
        lower_filename = filename.lower()

        for extension in SUPPORTED_EXTENSIONS:
            if lower_filename.endswith(extension):
                return True

        return False


    # =====================================================
    # Repository Tree
    # =====================================================

    def get_repository_tree(
        self,
        repository_url: str
    ) -> List[dict]:
        """
        Fetch the complete repository file tree.
        """

        owner, repository = self.parse_repository_url(
            repository_url
        )

        repository_info = self.get_repository_info(
            repository_url
        )

        branch = repository_info["default_branch"]

        response = self._get(
            f"/repos/{owner}/{repository}/git/trees/"
            f"{branch}?recursive=1"
        )

        data = response.json()

        if data.get("truncated"):
            raise ValueError(
                "The GitHub repository is too large for "
                "the GitHub tree API response."
            )

        return data.get("tree", [])


    # =====================================================
    # File Content
    # =====================================================

    def get_file_content(
        self,
        repository_url: str,
        file_path: str
    ) -> RepositoryFile:
        """
        Download a single file from GitHub.
        """

        owner, repository = self.parse_repository_url(
            repository_url
        )

        response = self._get(
            f"/repos/{owner}/{repository}/contents/"
            f"{file_path}"
        )

        data = response.json()

        if isinstance(data, list):
            raise ValueError(
                f"'{file_path}' is a directory, not a file."
            )

        encoded_content = data.get(
            "content"
        )

        if not encoded_content:
            raise ValueError(
                f"GitHub did not return content for "
                f"'{file_path}'."
            )

        encoded_content = encoded_content.replace(
            "\n",
            ""
        )

        try:
            decoded_content = base64.b64decode(
                encoded_content
            ).decode(
                "utf-8",
                errors="replace"
            )
        except Exception as exc:
            raise ValueError(
                f"Could not decode '{file_path}': {exc}"
            ) from exc

        return RepositoryFile(
            path=file_path,
            content=decoded_content,
            sha=data.get("sha", ""),
            url=data.get("html_url", ""),
        )


    # =====================================================
    # Complete Repository
    # =====================================================

    def load_repository(
        self,
        repository_url: str,
        max_files: int = 300
    ) -> List[RepositoryFile]:
        """
        Load supported source files from a repository.

        max_files protects the application from attempting
        to ingest an unexpectedly huge repository.
        """

        tree = self.get_repository_tree(
            repository_url
        )

        files_to_load = []

        for item in tree:

            if item.get("type") != "blob":
                continue

            path = item.get("path", "")

            if not path:
                continue

            if not self.is_supported_file(path):
                continue

            files_to_load.append(path)

            if len(files_to_load) >= max_files:
                break

        repository_files: List[RepositoryFile] = []

        for file_path in files_to_load:

            try:
                repository_file = self.get_file_content(
                    repository_url,
                    file_path
                )

                # Skip completely empty files.
                if not repository_file.content.strip():
                    continue

                repository_files.append(
                    repository_file
                )

            except ValueError:
                # One problematic file should not stop the
                # complete repository ingestion.
                continue

        if not repository_files:
            raise ValueError(
                "No supported readable files were found "
                "in the repository."
            )

        return repository_files