"""
RepoMind AI
Agentic RAG Engine

This is the main reasoning layer.

The LangChain agent can:

1. Understand the user's question.
2. Select appropriate repository tools.
3. Retrieve information.
4. Decide whether more information is required.
5. Call another tool when necessary.
6. Produce a final answer grounded in repository code.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
)
from backend.tools import (
    get_tools,
    set_current_repository,
)


# =========================================================
# System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are RepoMind AI, an expert software repository
analysis assistant.

Your job is to explain GitHub repositories using the
repository's actual indexed source code.

IMPORTANT RULES:

1. Do not invent repository behavior.
2. Base technical claims on retrieved repository code.
3. Use tools whenever repository-specific information
   is required.
4. If the first search does not provide enough information,
   perform another tool call.
5. For questions involving multiple files, investigate
   all relevant files before answering.
6. Mention file names when they are relevant.
7. Explain code in simple and structured language.
8. If the repository does not contain enough information,
   clearly say that the information could not be verified.
9. Never pretend that code exists if it was not retrieved.
10. Distinguish between what the code actually does and
    what could be improved.

TOOL STRATEGY:

- Use repository_structure when you need to understand
  project organization.
- Use search_code for general semantic code questions.
- Use search_file when a specific file requires deeper
  analysis.
- Use search_by_language when the question is related to
  a specific programming language.

AGENTIC BEHAVIOR:

Do not assume one retrieval step is always enough.

For example, if the user asks:

"Explain the authentication flow from API request
to database."

You should investigate multiple relevant parts such as:

API routes
authentication logic
services
database/model code

Then combine the evidence into one coherent explanation.

Your final response should be concise but useful.
"""


# =========================================================
# Agent Engine
# =========================================================

class RepoMindAgent:
    """
    LangChain-based Agentic RAG engine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:

        self.api_key = (
            api_key
            or GROQ_API_KEY
        )

        self.model_name = (
            model_name
            or GROQ_MODEL
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model_name,
            temperature=0.1,
            max_retries=2,
        )

        self.tools = get_tools()

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
        )


    # =====================================================
    # Analyze Repository
    # =====================================================

    def ask(
        self,
        question: str,
        repository: str,
    ) -> str:
        """
        Ask the Agent a repository-specific question.
        """

        question = question.strip()
        repository = repository.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not repository:
            raise ValueError(
                "Repository cannot be empty."
            )

        # Tell tools which repository is currently active.
        set_current_repository(
            repository
        )

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        return self._extract_final_answer(
            result
        )


    # =====================================================
    # Extract Answer
    # =====================================================

    @staticmethod
    def _extract_final_answer(
        result: Dict[str, Any],
    ) -> str:
        """
        Extract the final assistant response from LangChain's
        agent state.
        """

        messages = result.get(
            "messages",
            []
        )

        if not messages:
            return (
                "The agent did not return an answer."
            )

        # Walk backwards because the last assistant message
        # is normally the final response.
        for message in reversed(
            messages
        ):

            message_type = getattr(
                message,
                "type",
                "",
            )

            if message_type != "ai":
                continue

            content = getattr(
                message,
                "content",
                "",
            )

            if isinstance(
                content,
                str
            ):

                if content.strip():
                    return content.strip()

            # Some LangChain providers can return structured
            # content blocks.
            if isinstance(
                content,
                list
            ):

                text_parts = []

                for block in content:

                    if isinstance(
                        block,
                        dict,
                    ):

                        text = block.get(
                            "text"
                        )

                        if text:
                            text_parts.append(
                                str(text)
                            )

                if text_parts:
                    return "\n".join(
                        text_parts
                    ).strip()

        return (
            "The agent completed the request, "
            "but no final textual answer was returned."
        )


# =========================================================
# Factory
# =========================================================

_agent_instance: Optional[
    RepoMindAgent
] = None


def get_agent() -> RepoMindAgent:
    """
    Return a shared RepoMind Agent instance.
    """

    global _agent_instance

    if _agent_instance is None:

        _agent_instance = RepoMindAgent()

    return _agent_instance