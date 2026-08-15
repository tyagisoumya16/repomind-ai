"""
RepoMind AI
Streamlit Frontend

The frontend communicates with the FastAPI backend.

Features:

1. GitHub repository indexing
2. Repository statistics
3. Chat with repository
4. Conversation history
5. Clear chat
"""

from __future__ import annotations

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

INDEX_ENDPOINT = (
    f"{BACKEND_URL}/repository/index"
)

ASK_ENDPOINT = (
    f"{BACKEND_URL}/ask"
)

HEALTH_ENDPOINT = (
    f"{BACKEND_URL}/health"
)


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="RepoMind AI",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# Session State
# =========================================================

if "repository" not in st.session_state:
    st.session_state.repository = ""

if "repository_url" not in st.session_state:
    st.session_state.repository_url = ""

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "repository_info" not in st.session_state:
    st.session_state.repository_info = {}

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Helper Functions
# =========================================================

def check_backend() -> bool:
    """
    Check whether FastAPI backend is running.
    """

    try:

        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5,
        )

        return response.ok

    except requests.RequestException:

        return False


def index_repository(
    repository_url: str,
    max_files: int,
) -> tuple[bool, dict]:
    """
    Send repository indexing request to FastAPI.
    """

    try:

        response = requests.post(
            INDEX_ENDPOINT,
            json={
                "repository_url": repository_url,
                "max_files": max_files,
            },
            timeout=600,
        )

        data = response.json()

        if not response.ok:
            return False, data

        return True, data

    except requests.RequestException as exc:

        return False, {
            "detail": (
                "Could not connect to backend: "
                f"{exc}"
            )
        }

    except ValueError:

        return False, {
            "detail": (
                "Backend returned an invalid response."
            )
        }


def ask_question(
    repository: str,
    question: str,
) -> tuple[bool, dict]:
    """
    Ask the RepoMind Agent a question.
    """

    try:

        response = requests.post(
            ASK_ENDPOINT,
            json={
                "repository": repository,
                "question": question,
            },
            timeout=600,
        )

        data = response.json()

        if not response.ok:
            return False, data

        return True, data

    except requests.RequestException as exc:

        return False, {
            "detail": (
                "Could not connect to backend: "
                f"{exc}"
            )
        }

    except ValueError:

        return False, {
            "detail": (
                "Backend returned an invalid response."
            )
        }


# =========================================================
# Header
# =========================================================

st.title("🧠 RepoMind AI")

st.markdown(
    """
### Agentic RAG GitHub Repository Intelligence

Understand a GitHub repository using:

- 🔎 Advanced RAG
- 🧩 Code-aware chunking
- 🧠 Semantic retrieval
- 📊 Re-ranking
- 🔗 LangChain
- 🛠️ Tool Calling
- 🤖 Agentic RAG
- ⚡ Groq LLM
"""
)


# =========================================================
# Backend Status
# =========================================================

backend_running = check_backend()

if backend_running:

    st.success(
        "🟢 Backend connected"
    )

else:

    st.error(
        "🔴 Backend is not running. "
        "Start FastAPI before using RepoMind."
    )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("Repository")

    repository_url = st.text_input(
        "GitHub Repository URL",
        value=st.session_state.repository_url,
        placeholder=(
            "https://github.com/user/repository"
        ),
    )

    max_files = st.number_input(
        "Maximum files to index",
        min_value=1,
        max_value=500,
        value=300,
        step=10,
    )

    index_button = st.button(
        "🚀 Index Repository",
        use_container_width=True,
    )

    st.divider()

    if st.session_state.indexed:

        st.success(
            "Repository indexed"
        )

        repository_info = (
            st.session_state.repository_info
        )

        st.write(
            f"**Repository:** "
            f"{repository_info.get('repository', '')}"
        )

        st.write(
            f"**Files:** "
            f"{repository_info.get('files_indexed', 0)}"
        )

        st.write(
            f"**Chunks:** "
            f"{repository_info.get('chunks_indexed', 0)}"
        )

        st.write(
            f"**Language:** "
            f"{repository_info.get('language') or 'Unknown'}"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# Repository Indexing
# =========================================================

if index_button:

    if not backend_running:

        st.error(
            "Backend is not running."
        )

    elif not repository_url.strip():

        st.warning(
            "Please enter a GitHub repository URL."
        )

    else:

        with st.spinner(
            "Analyzing and indexing repository..."
        ):

            success, data = index_repository(
                repository_url=repository_url.strip(),
                max_files=int(max_files),
            )

        if success:

            st.session_state.repository_url = (
                repository_url.strip()
            )

            st.session_state.repository = (
                data["repository"]
            )

            st.session_state.repository_info = (
                data
            )

            st.session_state.indexed = True

            # Starting a new repository should also
            # start a clean conversation.
            st.session_state.messages = []

            st.success(
                "Repository indexed successfully!"
            )

            st.rerun()

        else:

            st.error(
                data.get(
                    "detail",
                    "Repository indexing failed.",
                )
            )


# =========================================================
# Main Chat Area
# =========================================================

if not st.session_state.indexed:

    st.info(
        "👈 Enter a public GitHub repository URL "
        "and click **Index Repository** to begin."
    )

    st.markdown(
        """
### Example questions you can ask

Once a repository is indexed, try:

> How does this project work?

> Explain the authentication flow.

> Where is the database connection implemented?

> Which files are responsible for API routes?

> Explain the main entry point.

> How does data flow from the API to the database?

> Find the code responsible for error handling.
"""
    )

else:

    st.subheader(
        f"💬 Chat with "
        f"{st.session_state.repository}"
    )

    # -----------------------------------------------------
    # Display previous messages
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------------------------------
    # User Input
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask anything about this repository..."
    )

    if question:

        question = question.strip()

        if not question:
            st.warning(
                "Please enter a question."
            )

        else:

            # Add user message.
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )

            # Ask Agent.
            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "RepoMind is investigating the repository..."
                ):

                    success, data = ask_question(
                        repository=(
                            st.session_state.repository
                        ),
                        question=question,
                    )

                if success:

                    answer = data.get(
                        "answer",
                        "No answer returned.",
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                else:

                    error_message = data.get(
                        "detail",
                        "Agent request failed.",
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Error: {error_message}"
                            ),
                        }
                    )