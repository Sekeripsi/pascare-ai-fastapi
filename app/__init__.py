"""Application-wide observability defaults.

LangGraph emits standard LangChain callback events for each graph run. When a
LangSmith key is configured, enable tracing before any graph is imported so
the complete graph (including its nodes and nested LLM calls) is captured.
"""

import os
from collections.abc import Sequence
from typing import Any


def _configure_langsmith_tracing() -> None:
    """Enable remote graph tracing only when explicitly configured."""
    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        return


def graph_monitoring_config(
    *,
    callbacks: Sequence[Any] | None = None,
    graph_variant: str | None = None,
) -> dict[str, Any]:
    """Build consistent LangGraph run metadata for local and remote monitors."""
    variant = graph_variant or os.getenv("GRAPH_VARIANT", "baseline")
    config: dict[str, Any] = {
        "run_name": f"postvisit-{variant}",
        "tags": ["postvisit", "langgraph", f"graph:{variant}"],
        "metadata": {
            "service": "postvisit-fastapi",
            "graph_variant": variant,
        },
    }
    if callbacks:
        config["callbacks"] = list(callbacks)
    return config


_configure_langsmith_tracing()