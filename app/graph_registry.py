"""Selects which chat graph serves requests; GRAPH_VARIANT switches them.

baseline = original routed pipeline (app/agents.py)
multi    = tiered selective fan-out (app/multiagent.py)

Both graphs share the same invoke/stream contract, so routes stay unchanged.
"""
from langgraph.graph.state import CompiledStateGraph

from app.config import settings


def get_chat_graph() -> CompiledStateGraph:
    if settings.graph_variant == "multi":
        from app.multiagent import chat_graph_multi
        return chat_graph_multi
    from app.agents import chat_graph
    return chat_graph
