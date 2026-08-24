# Tiered Selective Fan-Out Multi-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nothing, add a parallel specialist-agent LangGraph variant (`app/multiagent.py`) selectable by env flag, plus a latency/token benchmark comparing it against the preserved baseline.

**Architecture:** Router (small LLM, temp 0) emits route + multi-label domain selection; touched specialists run in parallel on focused record slices (small LLM, skipped when slice empty); one synthesizer call (large LLM) composes the grounded final answer. Crisis/guardrail/scope logic reused unchanged from baseline.

**Tech Stack:** FastAPI, LangGraph, LangChain-Groq, SQLAlchemy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-multiagent-fanout-design.md`

## Global Constraints

- `app/agents.py`, `app/guardrails.py`, `app/database.py` stay byte-for-byte untouched.
- Python venv: `.venv/Scripts/python.exe` (Python 3.14). Run everything through it.
- Domain keys exactly: `diagnosa`, `pemeriksaan`, `pengobatan`, `rencana`.
- Fail-safe parsing: unparsable route -> `out_of_scope`; `rekam_medis` without parsable domains -> all four.
- Only synthesizer/crisis/general/out_of_scope append to `messages` (streaming must never leak specialist output).
- Env vars: `GROQ_SMALL_MODEL` (default `llama-3.1-8b-instant`), `GRAPH_VARIANT` (default `baseline`).
- No git repo yet: Task 0 initializes one so checkpoints are commits.

---

### Task 0: Git init + checkpoint

**Files:** none created.

- [ ] Step 1: `git init && git add -A && git commit -m "chore: baseline before multi-agent work"` (respect `.gitignore`; `.venv` must be ignored — extend `.gitignore` with `.venv/`, `__pycache__/`, `.env` first).

### Task 1: Config + env docs + pytest

**Files:**
- Modify: `app/config.py:28-34` (Settings class)
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `settings.groq_small_model: str`, `settings.graph_variant: str`

- [ ] **Step 1: Install pytest** into venv: `.venv/Scripts/python.exe -m pip install pytest`
- [ ] **Step 2: Write failing test**

```python
# tests/test_config.py
import os


def test_new_settings_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_SMALL_MODEL", raising=False)
    monkeypatch.delenv("GRAPH_VARIANT", raising=False)
    import importlib
    import app.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.groq_small_model == "llama-3.1-8b-instant"
    assert cfg.settings.graph_variant == "baseline"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GROQ_SMALL_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRAPH_VARIANT", "multi")
    import importlib
    import app.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.groq_small_model == "llama-3.3-70b-versatile"
    assert cfg.settings.graph_variant == "multi"
```

- [ ] **Step 3: Run** `.venv/Scripts/python.exe -m pytest tests/test_config.py -v` — expect FAIL (attribute missing).
- [ ] **Step 4: Implement** in Settings:

```python
    groq_small_model: str = os.getenv("GROQ_SMALL_MODEL", "llama-3.1-8b-instant")
    # baseline = original routed pipeline (app/agents.py); multi = tiered fan-out.
    graph_variant: str = os.getenv("GRAPH_VARIANT", "baseline")
```

Add to `.env.example`:

```
# Small/fast Groq model for router + specialist agents (free tier).
GROQ_SMALL_MODEL=llama-3.1-8b-instant
# Which chat graph serves requests: baseline | multi
GRAPH_VARIANT=baseline
```

- [ ] **Step 5: Run tests — PASS. Commit `feat: config for tiered multi-agent variant`.**

### Task 2: Router protocol parser (pure)

**Files:**
- Create: `app/multiagent.py` (starts as parser-only skeleton)
- Test: `tests/test_router_parsing.py`

**Interfaces:**
- Produces: `DOMAINS: tuple[str, ...]`, `parse_router_output(text: str) -> tuple[str, list[str]]`

- [ ] **Step 1: Failing tests**

```python
# tests/test_router_parsing.py
from app.multiagent import DOMAINS, parse_router_output


def test_clean_rekam_medis_with_domains():
    route, domains = parse_router_output("route: rekam_medis\ndomains: pemeriksaan, pengobatan")
    assert route == "rekam_medis"
    assert domains == ["pemeriksaan", "pengobatan"]


def test_general_has_no_domains():
    assert parse_router_output("route: general\ndomains: -")[0] == "general"


def test_unparsable_fails_safe_to_out_of_scope():
    route, domains = parse_router_output("saya tidak tahu")
    assert route == "out_of_scope"
    assert domains == []


def test_rekam_medis_without_domains_gets_all():
    route, domains = parse_router_output("route: rekam_medis")
    assert route == "rekam_medis"
    assert list(domains) == list(DOMAINS)


def test_unknown_domains_filtered_then_all():
    _, domains = parse_router_output("route: rekam_medis\ndomains: lab, jadwal")
    assert list(domains) == list(DOMAINS)


def test_duplicates_removed_order_kept():
    _, domains = parse_router_output("route: rekam_medis\ndomains: pengobatan, pengobatan, diagnosa")
    assert domains == ["pengobatan", "diagnosa"]
```

- [ ] **Step 2: Run — FAIL (module missing).**
- [ ] **Step 3: Implement** in `app/multiagent.py`:

```python
DOMAINS: tuple[str, ...] = ("diagnosa", "pemeriksaan", "pengobatan", "rencana")


def parse_router_output(text: str) -> tuple[str, list[str]]:
    """Parse the two-line router protocol; fail safe on anything unexpected."""
    route: str | None = None
    domains: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("route:") and route is None:
            value = stripped.split(":", 1)[1].strip()
            if "rekam_medis" in value:
                route = "rekam_medis"
            elif "general" in value:
                route = "general"
            elif "out" in value:
                route = "out_of_scope"
        elif stripped.startswith("domains"):
            raw = stripped.split(":", 1)[1] if ":" in stripped else ""
            for part in raw.replace(";", ",").split(","):
                token = part.strip()
                if token in DOMAINS and token not in domains:
                    domains.append(token)
    if route not in ("rekam_medis", "general", "out_of_scope"):
        return "out_of_scope", []
    if route == "rekam_medis" and not domains:
        domains = list(DOMAINS)  # safe superset: never drop a relevant specialist
    return route, domains
```

- [ ] **Step 4: Run — PASS. Commit `feat: fail-safe multi-label router parser`.**

### Task 3: Domain slice builder (pure)

**Files:**
- Modify: `app/multiagent.py`
- Test: `tests/test_slices.py`

**Interfaces:**
- Consumes: `DOMAINS`
- Produces: `build_domain_slices(rows: list[dict]) -> dict[str, str]` where `rows` items have the same keys as baseline `rekam_results` entries; returned dict maps every domain to formatted Indonesian text or `""` when the slice holds no data.

- [ ] **Step 1: Failing tests**

```python
# tests/test_slices.py
from app.multiagent import DOMAINS, build_domain_slices

ROW = {
    "pasien_nama": "Rizky Ramadhan",
    "tanggal": "2026-08-22 09:00:00",
    "poliklinik": "UMUM",
    "keluhan": "Sakit kepala sebelah kiri sejak 3 hari, mual",
    "diagnosa": "Cephalgia (sakit kepala tegang)",
    "kode_icd": "R51",
    "suhu": 36.8, "nadi": 78, "respirasi": 20, "sistol": 130, "diastol": 85,
    "keadaan": "Baik", "kesadaran": "KOMPOS_MENTIS",
    "tindakan": "Pemberian analgesik dan konseling",
    "pengobatan": '[{"nama": "Parasetamol 500 mg", "dosis": "3x1 tablet"}]',
    "status_pulang": "SEMBUH",
    "kie": "Istirahat cukup",
    "plan": "Kontrol ulang jika gejala berlanjut",
}


def test_all_domains_present_and_nonempty():
    slices = build_domain_slices([ROW])
    assert set(slices.keys()) == set(DOMAINS)
    assert all(slices[d] for d in DOMAINS)


def test_empty_row_yields_empty_slices():
    empty = {k: None for k in ROW}
    slices = build_domain_slices([empty])
    assert all(slices[d] == "" for d in DOMAINS)


def test_partial_row_marks_only_filled_domains():
    partial = {k: None for k in ROW}
    partial.update({"tanggal": "2026-08-22", "keluhan": "Batuk", "diagnosa": "ISPA"})
    slices = build_domain_slices([partial])
    assert slices["diagnosa"] != ""
    assert slices["pemeriksaan"] == ""
    assert slices["pengobatan"] == ""
    assert slices["rencana"] == ""
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** in `app/multiagent.py`:

```python
def _fmt(value) -> str:
    if value is None or str(value).strip() in ("", "-"):
        return "-"
    return str(value)


def build_domain_slices(rows: list[dict]) -> dict[str, str]:
    """Format each visit row into per-domain prompt text; '' marks no data."""
    parts: dict[str, list[str]] = {d: [] for d in DOMAINS}
    filled: dict[str, bool] = {d: False for d in DOMAINS}

    for row in rows:
        tanggal = _fmt(row.get("tanggal"))
        keluhan, diagnosa, icd = _fmt(row.get("keluhan")), _fmt(row.get("diagnosa")), _fmt(row.get("kode_icd"))
        parts["diagnosa"].append(
            f"Tanggal: {tanggal}\nKeluhan: {keluhan}\nDiagnosa: {diagnosa} (Kode ICD: {icd})"
        )
        if any(row.get(k) not in (None, "") for k in ("keluhan", "diagnosa", "kode_icd")):
            filled["diagnosa"] = True

        vital = (
            f"Suhu {_fmt(row.get('suhu'))}°C, Nadi {_fmt(row.get('nadi'))}x/menit, "
            f"Tekanan Darah {_fmt(row.get('sistol'))}/{_fmt(row.get('diastol'))} mmHg, "
            f"Respirasi {_fmt(row.get('respirasi'))}x/menit"
        )
        parts["pemeriksaan"].append(
            f"Tanggal: {tanggal} | Poliklinik: {_fmt(row.get('poliklinik'))}\n"
            f"{vital}\nKeadaan: {_fmt(row.get('keadaan'))}, Kesadaran: {_fmt(row.get('kesadaran'))}"
        )
        if any(row.get(k) not in (None, "") for k in ("suhu", "nadi", "respirasi", "sistol", "keadaan", "kesadaran")):
            filled["pemeriksaan"] = True

        parts["pengobatan"].append(
            f"Tanggal: {tanggal}\nTindakan: {_fmt(row.get('tindakan'))}\nPengobatan: {_fmt(row.get('pengobatan'))}"
        )
        if any(row.get(k) not in (None, "") for k in ("tindakan", "pengobatan")):
            filled["pengobatan"] = True

        parts["rencana"].append(
            f"Tanggal: {tanggal}\nStatus pulang: {_fmt(row.get('status_pulang'))}\n"
            f"KIE: {_fmt(row.get('kie'))}\nRencana: {_fmt(row.get('plan'))}"
        )
        if any(row.get(k) not in (None, "") for k in ("status_pulang", "kie", "plan")):
            filled["rencana"] = True

    return {
        d: "\n---\n".join(parts[d]).strip() if filled[d] else ""
        for d in DOMAINS
    }
```

- [ ] **Step 4: Run — PASS. Commit `feat: per-domain record slice builder`.**

### Task 4: Graph assembly (`app/multiagent.py` complete)

**Files:**
- Modify: `app/multiagent.py` (append nodes + graph)
- Test: `tests/test_multiagent_graph.py`

**Interfaces:**
- Consumes: `parse_router_output`, `build_domain_slices`, `DOMAINS`; from `app.agents`: `SYSTEM_PROMPT`, `_resolve_patient_context`, `safety_gate`, `crisis_node`, `out_of_scope_node`, table objects via its own DB session.
- Produces: `chat_graph_multi` (compiled LangGraph, same invoke/stream contract as `chat_graph`).

- [ ] **Step 1: Failing tests** (both paths below make ZERO LLM calls — crisis is gated deterministically, invalid token short-circuits):

```python
# tests/test_multiagent_graph.py
from app.guardrails import CRISIS_REPLY, NO_DATA_REPLY
from app.multiagent import DOMAINS, chat_graph_multi


def test_compiled_graph_has_expected_nodes():
    expected = {"safety_gate", "crisis", "router", "load_record", "synthesizer",
                "general", "out_of_scope", *DOMAINS}
    assert expected == set(chat_graph_multi.get_graph().nodes)


def test_crisis_never_reaches_an_llm_node():
    result = chat_graph_multi.invoke({"messages": [{"role": "user", "content": "saya mau bunuh diri"}], "token": None})
    assert CRISIS_REPLY in result["messages"][-1].content


def test_invalid_token_short_circuits_no_data():
    result = chat_graph_multi.invoke(
        {"messages": [{"role": "user", "content": "tekanan darah saya berapa?"}], "token": "invalid-token"}
    )
    assert NO_DATA_REPLY in result["messages"][-1].content
    assert result.get("rekam_medis", []) == []
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** (append to `app/multiagent.py`; DB query copied from baseline `rekam_medis_agent`, then sliced):

```python
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from app.config import settings
from app.guardrails import NO_DATA_REPLY
from app.database import (
    SessionLocal, Pasien, Pendaftaran, Anamnesis, Diagnosis, Pemeriksaan,
    Tindakan, Pengobatan, PulangRujuk, RekamMedis,
)
from app.agents import (
    SYSTEM_PROMPT, _last_content, _speaker, _resolve_patient_context,
    safety_gate, crisis_node, out_of_scope_node,
)

small_llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.groq_small_model,
    temperature=settings.groq_temperature,
    max_tokens=512,
)
router_llm_small = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.groq_small_model,
    temperature=0,
    max_tokens=256,
)
synthesizer_llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.groq_model,
    temperature=settings.groq_temperature,
    max_tokens=settings.groq_max_tokens,
)


def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class MultiAgentState(dict):
    messages: Annotated[list, add_messages]
    route: str
    domains: list[str]
    has_record: bool
    patient_name: str | None
    rekam_medis: list
    domain_slices: Annotated[dict, _merge_dicts]
    specialist_outputs: Annotated[dict, _merge_dicts]
    token: str | None


MULTI_ROUTER_PROMPT = (
    "Klasifikasikan pesan TERAKHIR pasien. Jawab PERSIS dalam format dua baris ini:\n"
    "route: <kategori>\n"
    "domains: <daftar domain>\n"
    "\n"
    "Kategori:\n"
    "- 'rekam_medis': menyangkut isi kunjungan medis — keluhan, diagnosa, kode ICD, hasil "
    "pemeriksaan (tekanan darah, suhu, nadi), tindakan, pengobatan/resep, jadwal kontrol, rujukan.\n"
    "- 'general': sapaan, ucapan terima kasih, atau pertanyaan cara pakai layanan.\n"
    "- 'out_of_scope': coding, berita, lelucon, tugas sekolah, atau penjelasan/nasihat medis umum.\n"
    "\n"
    "domains hanya diisi jika route=rekam_medis; pilih dari: diagnosa, pemeriksaan, pengobatan, rencana "
    "(pisah dengan koma, boleh lebih dari satu):\n"
    "- diagnosa: keluhan, diagnosa, kode ICD\n"
    "- pemeriksaan: tekanan darah, suhu, nadi, respirasi, keadaan umum\n"
    "- pengobatan: tindakan medis dan obat/resep\n"
    "- rencana: status pulang, anjuran KIE, jadwal kontrol, rujukan\n"
    "Jika bukan rekam_medis tulis: domains: -\n"
    "\n"
    "Contoh:\n"
    "\"Tekanan darah saya waktu periksa berapa ya?\" ->\nroute: rekam_medis\ndomains: pemeriksaan\n"
    "\"Obat saya apa saja dan diminum kapan?\" ->\nroute: rekam_medis\ndomains: pengobatan\n"
    "\"Diagnosa saya apa dan kontrol kapan lagi?\" ->\nroute: rekam_medis\ndomains: diagnosa,rencana\n"
    "\"Terima kasih banyak ya\" ->\nroute: general\ndomains: -\n"
    "\"Buatkan kode Python dong\" ->\nroute: out_of_scope\ndomains: -"
)


def router(state: MultiAgentState) -> dict:
    msgs = state["messages"]
    history = [
        f"{_speaker(m)}: {m.content if isinstance(m.content, str) else str(m.content)}"
        for m in msgs[:-1][-4:]
    ]
    human = ""
    if history:
        human += "Riwayat percakapan:\n" + "\n".join(history) + "\n\n"
    human += f"Pesan terakhir dari pasien: {_last_content(state)}"

    response = router_llm_small.invoke([
        SystemMessage(content=MULTI_ROUTER_PROMPT),
        HumanMessage(content=human),
    ])
    text = response.content if isinstance(response.content, str) else str(response.content)
    if not text.strip():
        text = response.additional_kwargs.get("reasoning_content") or ""
    route, domains = parse_router_output(text)
    return {"route": route, "domains": domains}


RECORD_QUERY_COLUMNS = lambda: select(  # noqa: E731 — mirrors baseline join shape
    Pasien.c.nama.label("pasien_nama"),
    Pendaftaran.c.tglKunjungan.label("tanggal"),
    Pendaftaran.c.poliklinik.label("poliklinik"),
    Anamnesis.c.keluhan.label("keluhan"),
    Diagnosis.c.diagnosis.label("diagnosa"),
    Diagnosis.c.kodeIcd.label("kode_icd"),
    Pemeriksaan.c.suhu.label("suhu"),
    Pemeriksaan.c.nadi.label("nadi"),
    Pemeriksaan.c.respirasi.label("respirasi"),
    Pemeriksaan.c.sistol.label("sistol"),
    Pemeriksaan.c.diastol.label("diastol"),
    Pemeriksaan.c.keadaan.label("keadaan"),
    Pemeriksaan.c.kesadaran.label("kesadaran"),
    Tindakan.c.tindakan.label("tindakan"),
    Pengobatan.c.pengobatan.label("pengobatan"),
    PulangRujuk.c.statusPulang.label("status_pulang"),
    PulangRujuk.c.kie.label("kie"),
    PulangRujuk.c.plan.label("plan"),
)


def load_record(state: MultiAgentState) -> dict:
    """One DB read: resolve token scope, fetch joined record, pre-slice domains."""
    patient_name, rekam_filter_ids = _resolve_patient_context(state)
    base = {
        "patient_name": patient_name,
        "rekam_medis": [],
        "domain_slices": {},
        "has_record": False,
    }
    if not rekam_filter_ids:
        return {"messages": [AIMessage(content=NO_DATA_REPLY)], **base}

    db = SessionLocal()
    try:
        stmt = RECORD_QUERY_COLUMNS().select_from(RekamMedis)
        for tbl, on in (
            (Pasien, RekamMedis.c.pasienId == Pasien.c.id),
            (Pendaftaran, RekamMedis.c.pendaftaranId == Pendaftaran.c.id),
            (Anamnesis, RekamMedis.c.anamnesisId == Anamnesis.c.id),
            (Diagnosis, RekamMedis.c.diagnosisId == Diagnosis.c.id),
            (Pemeriksaan, RekamMedis.c.pemeriksaanId == Pemeriksaan.c.id),
            (Tindakan, RekamMedis.c.tindakanId == Tindakan.c.id),
            (Pengobatan, RekamMedis.c.pengobatanId == Pengobatan.c.id),
            (PulangRujuk, RekamMedis.c.pulangRujukId == PulangRujuk.c.id),
        ):
            stmt = stmt.outerjoin(tbl, on)
        stmt = stmt.where(RekamMedis.c.id.in_(rekam_filter_ids)).order_by(
            Pendaftaran.c.tglKunjungan.desc()
        )
        rows = db.execute(stmt).mappings().all()

        record_rows, domain_rows = [], []
        for row in rows:
            record_rows.append({
                "pasien_nama": row["pasien_nama"], "tanggal": str(row["tanggal"]) if row["tanggal"] else None,
                "poliklinik": row["poliklinik"], "keluhan": row["keluhan"],
                "diagnosa": row["diagnosa"], "kode_icd": row["kode_icd"],
                "suhu": row["suhu"], "nadi": row["nadi"], "respirasi": row["respirasi"],
                "sistol": row["sistol"], "diastol": row["diastol"],
                "keadaan": row["keadaan"], "kesadaran": row["kesadaran"],
                "tindakan": row["tindakan"], "pengobatan": row["pengobatan"],
                "status_pulang": row["status_pulang"], "kie": row["kie"], "plan": row["plan"],
            })
            domain_rows.append({**record_rows[-1]})
        if not record_rows:
            return {"messages": [AIMessage(content=NO_DATA_REPLY)], **base}

        return {
            "patient_name": patient_name,
            "rekam_medis": record_rows,
            "domain_slices": build_domain_slices(domain_rows),
            "has_record": True,
        }
    finally:
        db.close()


SPECIALIST_RULES = (
    "\n\nAturan: jawab HANYA dari data kunjungan di bawah. Jika informasi yang ditanya "
    "tidak ada di data, katakan tidak tersedia. Jangan memberi diagnosis baru atau nasihat "
    "medis. Bahasa Indonesia singkat dan jelas (maksimal 4 kalimat)."
)
SPECIALIST_PROMPTS = {
    "diagnosa": "Anda agen spesialis DIAGNOSA pasca-kunjungan: menjelaskan keluhan, diagnosa, dan kode ICD dari kunjungan pasien dengan bahasa sederhana.",
    "pemeriksaan": "Anda agen spesialis PEMERIKSAAN pasca-kunjungan: menjelaskan hasil pengukuran vital (tekanan darah, suhu, nadi, respirasi) dan kondisi umum saat pemeriksaan.",
    "pengobatan": "Anda agen spesialis PENGOBATAN pasca-kunjungan: menjelaskan tindakan yang dilakukan serta obat/resep yang diberikan (nama, dosis, aturan pakai) persis seperti tercatat.",
    "rencana": "Anda agen spesialis RENCANA pasca-kunjungan: menjelaskan status pulang, anjuran KIE, rencana kontrol, dan rujukan sesuai catatan.",
}


def make_specialist(domain: str):
    def specialist_node(state: MultiAgentState) -> dict:
        if not state.get("has_record"):
            return {"specialist_outputs": {domain: ""}}
        slice_text = state.get("domain_slices", {}).get(domain, "")
        if not slice_text:
            # Empty slice: skip the LLM entirely; synthesizer ignores empty notes.
            return {"specialist_outputs": {domain: ""}}
        response = small_llm.invoke([
            SystemMessage(content=SPECIALIST_PROMPTS[domain] + SPECIALIST_RULES),
            HumanMessage(content=f"DATA KUNJUNGAN:\n{slice_text}\n\nPertanyaan pasien:\n{_last_content(state)}"),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"specialist_outputs": {domain: content}}

    return specialist_node


def general_agent(state: MultiAgentState) -> dict:
    content = _last_content(state)
    system_text = SYSTEM_PROMPT
    if state.get("patient_name"):
        system_text += f"\nPasien yang terhubung: {state['patient_name']}. Sapa dengan nama jika relevan."
    system_text += (
        "\n\nIni cabang sapaan/topik umum layanan: balas ramah dan singkat, perkenalkan diri "
        "sebagai asisten pasca-kunjungan, lalu ingatkan hal-hal yang bisa ditanyakan."
    )
    return {"messages": [small_llm.invoke([
        SystemMessage(content=system_text), HumanMessage(content=content),
    ])]}


def synthesizer(state: MultiAgentState) -> dict:
    outputs = state.get("specialist_outputs", {})
    notes = "\n\n".join(f"[Catatan {name}]\n{text}" for name, text in outputs.items() if text)
    system_text = SYSTEM_PROMPT
    if state.get("patient_name"):
        system_text += f"\nPasien yang terhubung: {state['patient_name']}. Bisa menyapanya dengan nama."
    system_text += (
        "\n\nJawaban akhir untuk pasien WAJIB disusun HANYA dari catatan spesialis berikut. "
        "Gabungkan bagian yang relevan dengan pertanyaan menjadi satu jawaban utuh yang mengalir, "
        "tanpa menyebut 'catatan spesialis'. Jika semua catatan kosong atau tidak memuat jawaban, "
        "katakan jujur bahwa informasi tidak tersedia di data kunjungan."
    )
    response = synthesizer_llm.invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=f"{notes}\n\nPertanyaan pasien:\n{_last_content(state)}"),
    ])
    return {"messages": [response]}


graph = StateGraph(MultiAgentState)
graph.add_node("safety_gate", safety_gate)
graph.add_node("crisis", crisis_node)
graph.add_node("router", router)
graph.add_node("load_record", load_record)
graph.add_node("synthesizer", synthesizer)
graph.add_node("general", general_agent)
graph.add_node("out_of_scope", out_of_scope_node)
for _d in DOMAINS:
    graph.add_node(_d, make_specialist(_d))
    graph.add_edge("load_record", _d)      # fan-out: parallel branches
    graph.add_edge(_d, "synthesizer")      # fan-in barrier

graph.add_edge(START, "safety_gate")
graph.add_conditional_edges("safety_gate", lambda s: s.get("route", "continue"), {"crisis": "crisis", "continue": "router"})
graph.add_edge("crisis", END)
graph.add_conditional_edges("router", lambda s: s.get("route", "out_of_scope"), {
    "rekam_medis": "load_record", "general": "general", "out_of_scope": "out_of_scope",
})
graph.add_conditional_edges("load_record", lambda s: "synthesize" if s.get("has_record") else "skip", {
    "synthesize": "synthesizer", "skip": END,
})
graph.add_edge("general", END)
graph.add_edge("out_of_scope", END)
graph.add_edge("synthesizer", END)

chat_graph_multi = graph.compile()
```

Note: `select` import comes from `sqlalchemy` at top of module (`from sqlalchemy import select`). `parse_router_output`/`build_domain_slices` already defined above in Tasks 2-3 in the same file.

- [ ] **Step 4: Run** `pytest tests/test_multiagent_graph.py -v` — PASS (crisis + no-data paths prove wiring without network).
- [ ] **Step 5: Commit `feat: tiered selective fan-out multi-agent graph`.**

### Task 5: Graph registry + API switch

**Files:**
- Create: `app/graph_registry.py`
- Modify: `app/routes/chat.py:7,44,75`
- Test: `tests/test_graph_registry.py`

**Interfaces:**
- Produces: `get_chat_graph() -> CompiledStateGraph` reading `settings.graph_variant`.

- [ ] **Step 1: Failing test**

```python
# tests/test_graph_registry.py
def test_baseline_default(monkeypatch):
    monkeypatch.setenv("GRAPH_VARIANT", "baseline")
    import importlib, app.config as cfg, app.graph_registry as reg
    importlib.reload(cfg)
    from app.agents import chat_graph
    importlib.reload(reg)
    assert reg.get_chat_graph() is chat_graph


def test_multi_variant(monkeypatch):
    monkeypatch.setenv("GRAPH_VARIANT", "multi")
    import importlib, app.config as cfg, app.graph_registry as reg
    importlib.reload(cfg)
    from app.multiagent import chat_graph_multi
    importlib.reload(reg)
    assert reg.get_chat_graph() is chat_graph_multi
```

- [ ] **Step 2: Run — FAIL. Implement:**

```python
# app/graph_registry.py
"""Selects which chat graph serves requests; GRAPH_VARIANT switches them."""
from langgraph.graph.state import CompiledStateGraph

from app.config import settings


def get_chat_graph() -> CompiledStateGraph:
    if settings.graph_variant == "multi":
        from app.multiagent import chat_graph_multi
        return chat_graph_multi
    from app.agents import chat_graph
    return chat_graph
```

In `chat.py`: replace `from app.agents import chat_graph` with `from app.graph_registry import get_chat_graph`; inside `_stream_reply` use `graph = get_chat_graph()` then `graph.stream(...)`; inside endpoint use `get_chat_graph().invoke(...)`. Nothing else changes — response contract identical.

- [ ] **Step 3: Run all tests — PASS. Boot smoke:**
  `GRAPH_VARIANT=multi .venv/Scripts/python.exe -m uvicorn main:app --port 8000` boots clean, Ctrl+C.
- [ ] **Step 4: Commit `feat: GRAPH_VARIANT graph registry for API`.**

### Task 6: Benchmark harness

**Files:**
- Create: `scripts/benchmark_compare.py`

**Interfaces:**
- Consumes: both compiled graphs; `UsageMetadataCallbackHandler` from `langchain_core.callbacks` (fallback manual counter included).
- Produces: console table + `benchmark_results.json`.

- [ ] **Step 1: Write script** (full code in file; structure):

```python
"""Latency/token comparison: baseline routed pipeline vs tiered fan-out multi-agent.

Runs the fixed question set through BOTH graphs against the live DB + Groq.
Token totals come from usage_metadata aggregated across every LLM call in the
run (router included). Results land in benchmark_results.json for the thesis.
"""
QUESTIONS = [...]  # 16 scenarios: 4 single-domain, 3 cross-domain, 2 general,
                   # 2 off-topic, 1 crisis, 1 invalid-token, 2 two-turn follow-ups
# Active PostVisit token fetched straight from DB (unexpired, isActive).
# Per scenario x variant x RUNS(default 2): perf_counter around graph.invoke,
# UsageMetadataCallbackHandler for input/output/total tokens + call count.
# Report: mean+median latency, mean tokens/call-count per variant per scenario,
# grand totals. json.dump with ISO timestamp.
```

- [ ] **Step 2: Smoke-run 2 scenarios** `--limit 2` — both variants answer, JSON written.
- [ ] **Step 3: Commit `test: benchmark harness comparing baseline vs multi-agent`.**

### Task 7: Full verification

- [ ] `pytest` whole suite green.
- [ ] `scripts/demo_guardrails.py` still passes (guardrails untouched).
- [ ] Live benchmark full run both variants; results JSON saved.
- [ ] Manual curl streaming check `stream:true` on multi variant — specialist output never leaks, disclaimer appended.

### Task 8: Adversarial review + fixes

- [ ] Workflow: parallel reviewers over new diff (correctness, LangGraph state/reducer misuse, security regression, silent-failure), verify findings, apply confirmed fixes, commit.
