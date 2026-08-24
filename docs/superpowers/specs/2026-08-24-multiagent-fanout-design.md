# Tiered Selective Fan-Out Multi-Agent Architecture — Design

Date: 2026-08-24
Status: Approved (approach C chosen over full fan-out A and supervisor B)

## Problem

Current system (`app/agents.py`) is a routed pipeline, not a multi-agent system:

- One monolithic `rekam_medis_agent` node joins 9 tables and dumps the entire
  record into a single prompt.
- Zero parallelism; pure sequential graph.
- One shared creative LLM instance; only the router is separated.
- Thesis goal is a measurable efficiency comparison between the current
  pipeline (baseline) and a multi-agent architecture.

## Goals

1. True multi-agent decomposition: specialist agents per medical data domain,
   running in parallel (LangGraph fan-out / fan-in).
2. Model tiering on Groq free tier: small fast model (`llama-3.1-8b-instant`)
   for router, specialists, and general chat; large model
   (`gpt-oss-120b` or configured `GROQ_MODEL`) reserved for final synthesis.
3. Selective fan-out: router emits a multi-label domain classification;
   only touched specialists run; specialists whose slice is empty skip the
   LLM call entirely.
4. Clean comparison study: baseline graph preserved byte-for-byte; an env
   flag switches graphs; identical API surface; benchmark script measures
   latency and token usage per variant on a fixed question set.

## Non-Goals

- Supervisor/tool-loop agentic patterns (slower than baseline; rejected).
- Vector DB, RAG infra, or new external services.
- Any change to guardrails semantics, token-scope security model, disclaimer,
  streaming API contract, or response schema.

## Architecture

```
START -> safety_gate -> crisis -> END            (unchanged, deterministic)
                    \-> router (small LLM, multi-label)
                          -> general (small LLM) -> END
                          -> out_of_scope (canned) -> END
                          -> rekam_medis -> load_record (one DB read, resolves
                                PostVisit token scope; empty -> canned reply -> END)
                             -> [diagnosa || pemeriksaan || pengobatan || rencana]
                                (parallel small-LLM specialists; skipped ones
                                 return "" without an LLM call)
                             -> synthesizer (large LLM, grounded ONLY on returned
                                specialist notes) -> END
```

### Domains

Fixed keys, derived from existing tables:

| Key          | Source columns                                              |
|--------------|-------------------------------------------------------------|
| `diagnosa`   | Anamnesis.keluhan, Diagnosis.diagnosis, Diagnosis.kodeIcd   |
| `pemeriksaan`| Pemeriksaan.* vitals, keadaan, kesadaran + visit meta       |
| `pengobatan` | Tindakan.tindakan, Pengobatan.pengobatan                     |
| `rencana`    | PulangRujuk.statusPulang/kie/plan, kunjungan date, kontrol   |

### Router protocol (small model, temperature 0)

Strict two-line answer:

```
route: rekam_medis | general | out_of_scope
domains: comma-separated subset of diagnosa,pemeriksaan,pengobatan,rencana
```

Parsing rules (deterministic, fail-safe):
- Unparsable route -> `out_of_scope` (same fallback as baseline).
- `rekam_medis` with missing/unparsable domains -> all four specialists
  (safe superset, never drops a relevant domain).
- Unknown domain tokens ignored; if all filtered out -> all four.

### State additions

```python
domains: list[str]                      # selected specialist keys
has_record: bool                        # False -> NO_DATA short-circuit taken
patient_name: str | None
record_rows: list                       # full joined rows, returned by API
domain_slices: dict[str, str]           # preformatted per-domain text
specialist_outputs: Annotated[dict, merge-reducer]  # concurrent writes from fan-out
```

Only synthesizer, crisis, general, out_of_scope append to `messages`, so the
existing streaming endpoint never leaks intermediate specialist output.

### Data access

One DB round trip in `load_record` (reuses `_resolve_patient_context` scope
resolution from `app/agents.py`). Specialists do zero DB access; they format
their slice into a focused prompt. Empty slice -> no LLM call.

## Files

- Create: `app/multiagent.py` — new graph, self-contained.
- Modify: `app/config.py` — `groq_small_model`, `graph_variant` settings.
- Modify: `app/routes/chat.py` — graph selection via `app/graph_registry.py`.
- Create: `app/graph_registry.py` — returns compiled graph per variant.
- Create: `tests/test_multiagent.py` — unit tests for pure logic + graph wiring.
- Create: `scripts/benchmark_compare.py` — runs fixed question set through
  both graphs, reports latency + token totals, saves JSON results.
- Modify: `.env.example` — document new variables.
- Untouched: `app/agents.py`, `app/guardrails.py`, `app/database.py`, seeds.

## Measurement plan (thesis)

- Variants: `baseline` (current graph), `multi` (new graph).
- Question set (~16): single-domain x4, cross-domain x3, general x2,
  off-topic x2, crisis x1, invalid/no-token x1, two-turn follow-up x2.
- Metrics: wall-clock latency per request, total prompt+completion tokens
  (aggregated across all LLM calls incl. router), per-call count.
- Output: console table + `benchmark_results.json`.

## Risks

- gpt-oss reasoning channel consumes budget on synthesis — mitigated: synthesis
  context smaller than baseline mega-context.
- Small-model router misparse — mitigated: fail-safe parsing rules above.
- Parallel fan-out adds barrier sync overhead — measured honestly by benchmark.
