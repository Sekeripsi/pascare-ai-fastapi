"""Selects which chat graph serves requests; GRAPH_VARIANT switches them.

baseline = original routed pipeline (app/agents.py)
multi    = tiered selective fan-out (app/multiagent.py)

Both graphs share the same invoke/stream contract, so routes stay unchanged.
"""
from langgraph.graph.state import CompiledStateGraph

from app.config import settings


def get_chat_graph() -> CompiledStateGraph:
    # Fail closed: a typo'd GRAPH_VARIANT must never silently serve the other
    # variant (a benchmark would then compare baseline against baseline).
    if settings.graph_variant == "multi":
        from app.multiagent import chat_graph_multi
        return chat_graph_multi
    if settings.graph_variant == "baseline":
        from app.agents import chat_graph
        return chat_graph
    raise ValueError(
        f"GRAPH_VARIANT '{settings.graph_variant}' tidak dikenal; "
        "gunakan 'baseline' atau 'multi'."
    )
