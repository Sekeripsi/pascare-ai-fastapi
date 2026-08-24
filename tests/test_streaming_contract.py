"""Streaming contract: intermediate specialist output must never reach the client."""
import importlib

import app.multiagent as ma


class _FakeLLM:
    def __init__(self, text: str):
        self._text = text

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._text)


ROUTER_TEXT = "route: rekam_medis\ndomains: diagnosa,pemeriksaan"


def _run_stream(monkeypatch, rows):
    """Fake all small/big LLMs, drive graph.stream like the API route does."""

    def fake_resolve(state):
        token = state.get("token")
        if not token or token == "invalid":
            return None, []
        return "Rizky", ["rekam-id-1"]

    monkeypatch.setattr(ma, "_resolve_patient_context", fake_resolve)
    monkeypatch.setattr(ma, "router_llm_small", _FakeLLM(ROUTER_TEXT))
    monkeypatch.setattr(
        ma, "small_llm",
        _FakeLLM("CATATAN-SPESIALIS-JANGAN-BOCOR: ini jawaban spesialis."),
    )
    monkeypatch.setattr(
        ma, "synthesizer_llm",
        _FakeLLM("Jawaban akhir untuk pasien, tersusun dari catatan."),
    )
    monkeypatch.setattr(ma, "SessionLocal", lambda: _FakeSession(rows))

    class _FakeResult:
        def mappings(self):
            return self

        def all(self):
            return rows

    class _FakeSession:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, stmt):
            return _FakeResult()

        def close(self):
            pass

    yielded = []
    state = {
        "messages": [{"role": "user", "content": "diagnosa dan tekanan darah saya?"}],
        "token": "valid-token",
    }
    for event in ma.chat_graph_multi.stream(state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if not node_output:
                continue
            for msg in node_output.get("messages", []):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    yielded.append((node_name, content))
    return yielded


def test_stream_never_leaks_specialist_output(monkeypatch):
    row = {k: None for k in (
        "pasien_nama", "tanggal", "poliklinik", "keluhan", "diagnosa", "kode_icd",
        "suhu", "nadi", "respirasi", "sistol", "diastol", "keadaan", "kesadaran",
        "tindakan", "pengobatan", "status_pulang", "kie", "plan",
    )}
    row.update({
        "pasien_nama": "Rizky", "tanggal": "2026-08-22 09:00:00", "keluhan": "Pusing",
        "diagnosa": "Cephalgia", "kode_icd": "R51", "suhu": 36.8,
    })
    yielded = _run_stream(monkeypatch, [row])

    # Only the synthesizer's final message may surface.
    assert [name for name, _ in yielded] == ["synthesizer"], yielded
    assert all("CATATAN-SPESIALIS" not in text for _, text in yielded), yielded
    assert any("Jawaban akhir" in text for _, text in yielded)


def test_no_data_path_streams_only_canned_reply(monkeypatch):
    yielded = _run_stream(monkeypatch, [])
    names = [name for name, _ in yielded]
    assert names == ["load_record"], yielded  # synthesizer bows out silently
    assert any("tidak ditemukan" in text for _, text in yielded)
