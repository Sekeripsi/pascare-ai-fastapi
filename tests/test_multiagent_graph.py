from app.guardrails import CRISIS_REPLY, NO_DATA_REPLY
from app.multiagent import DOMAINS, chat_graph_multi


class _FakeRouterLLM:
    """Stands in for the router so routing tests never touch the network."""

    def __init__(self, text: str):
        self._text = text

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._text)


def test_compiled_graph_has_expected_nodes():
    expected = {"safety_gate", "crisis", "router", "load_record", "synthesizer",
                "general", "out_of_scope", *DOMAINS}
    assert expected <= set(chat_graph_multi.get_graph().nodes)


def test_specialists_fan_out_from_load_record_and_fan_in():
    targets = {e.target for e in chat_graph_multi.get_graph().edges
               if e.source == "load_record"}
    assert set(DOMAINS) <= targets


def test_crisis_never_reaches_an_llm_node():
    result = chat_graph_multi.invoke(
        {"messages": [{"role": "user", "content": "saya mau bunuh diri"}], "token": None}
    )
    assert CRISIS_REPLY in result["messages"][-1].content


def test_invalid_token_short_circuits_no_data(monkeypatch):
    # Router is faked to force the rekam_medis branch; the invalid token then
    # short-circuits in load_record with zero further LLM calls.
    monkeypatch.setattr(
        "app.multiagent.router_llm_small",
        _FakeRouterLLM("route: rekam_medis\ndomains: pemeriksaan"),
    )
    result = chat_graph_multi.invoke(
        {"messages": [{"role": "user", "content": "tekanan darah saya berapa?"}],
         "token": "invalid-token"}
    )
    assert NO_DATA_REPLY in result["messages"][-1].content
