import importlib

import app.config as cfg


def _reload_registry(monkeypatch, variant):
    monkeypatch.setenv("GRAPH_VARIANT", variant)
    importlib.reload(cfg)
    import app.graph_registry as reg
    return importlib.reload(reg)


def test_baseline_default(monkeypatch):
    reg = _reload_registry(monkeypatch, "baseline")
    from app.agents import chat_graph
    assert reg.get_chat_graph() is chat_graph
    assert cfg.settings.graph_variant == "baseline"


def test_multi_variant(monkeypatch):
    reg = _reload_registry(monkeypatch, "multi")
    from app.multiagent import chat_graph_multi
    assert reg.get_chat_graph() is chat_graph_multi
    assert cfg.settings.graph_variant == "multi"
