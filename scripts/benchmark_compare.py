"""Latency/token comparison: baseline routed pipeline vs tiered fan-out multi-agent.

Runs a fixed question set through BOTH graphs against the live database and
Groq API. Token counts aggregate every LLM call in a run (router included)
via an on_llm_end callback, so variants are compared on total work, not just
the visible answer call.

Results land in benchmark_results.json for the thesis.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_compare.py [--limit N] [--runs N]
                                                          [--variants baseline,multi]
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings
from app.database import SessionLocal, PostVisit


class UsageCounter(BaseCallbackHandler):
    """Aggregates token usage across every LLM invocation in one graph run."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            llm_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            usage = {
                "input_tokens": llm_usage.get("prompt_tokens", 0),
                "output_tokens": llm_usage.get("completion_tokens", 0),
                "total_tokens": llm_usage.get("total_tokens", 0),
            }
        if not usage:
            try:
                msg = response.generations[0][0].message
                usage = getattr(msg, "usage_metadata", None) or {}
            except (IndexError, AttributeError):
                usage = {}
        self.calls += 1
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))


# Single-turn scenarios: name, kind, messages, token ("valid" | "invalid").
SCENARIOS = [
    {"name": "single_diagnosa", "turns": ["Diagnosa saya waktu periksa kemarin apa ya?"]},
    {"name": "single_pemeriksaan", "turns": ["Tekanan darah saya waktu diperiksa berapa?"]},
    {"name": "single_pengobatan", "turns": ["Obat yang saya dapat itu diminum kapan saja?"]},
    {"name": "single_rencana", "turns": ["Kapan jadwal kontrol berikutnya saya?"]},
    {"name": "cross_diag_pengobatan", "turns": ["Diagnosa saya apa dan obatnya apa saja?"]},
    {"name": "cross_vital_tindakan", "turns": ["Suhu badan saya berapa dan tindakan apa yang dilakukan?"]},
    {"name": "cross_tiga_domain", "turns": ["Keluhan saya apa, tekanan darah berapa, dan kapan kontrol lagi?"]},
    {"name": "lab_tidak_tercatat", "turns": ["Hasil laboratorium saya bagaimana?"]},
    {"name": "general_sapaan", "turns": ["Halo, kamu siapa?"]},
    {"name": "general_terimakasih", "turns": ["Terima kasih banyak ya"]},
    {"name": "offtopic_koding", "turns": ["Buatkan kode Python program kalkulator dong"]},
    {"name": "offtopic_berita", "turns": ["Ceritakan berita terbaru hari ini"]},
    {"name": "crisis", "turns": ["saya ingin mengakhiri hidup saya"]},
    {"name": "token_invalid", "turns": ["Tekanan darah saya berapa?"], "token": "invalid"},
    {
        "name": "followup_keluhan_obat",
        "turns": ["Keluhan saya waktu kemarin apa?", "Untuk keluhan itu obatnya diminum bagaimana?"],
    },
    {
        "name": "followup_vital",
        "turns": ["Tekanan darah saya berapa?", "Kalau suhu badan saya berapa?"],
    },
]


def fetch_active_token() -> str | None:
    db = SessionLocal()
    try:
        row = (
            db.query(PostVisit)
            .filter(
                PostVisit.c.isActive == True,  # noqa: E712 — SQLAlchemy syntax
                PostVisit.c.expiresAt > datetime.now(timezone.utc).replace(tzinfo=None),
            )
            .first()
        )
        return row.token if row else None
    finally:
        db.close()


def load_graphs(variants: list[str]) -> dict:
    graphs = {}
    if "baseline" in variants:
        from app.agents import chat_graph
        graphs["baseline"] = chat_graph
    if "multi" in variants:
        from app.multiagent import chat_graph_multi
        graphs["multi"] = chat_graph_multi
    return graphs


def run_turn(graph, messages: list[dict], token: str | None) -> dict:
    counter = UsageCounter()
    start = time.perf_counter()
    result = graph.invoke(
        {"messages": messages, "patient_id": None, "rekam_medis_id": None, "token": token},
        config={"callbacks": [counter]},
    )
    elapsed = time.perf_counter() - start
    last = result["messages"][-1]
    content = last.content if isinstance(last.content, str) else str(last.content)
    return {
        "latency_s": round(elapsed, 3),
        "llm_calls": counter.calls,
        "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens,
        "total_tokens": counter.total_tokens,
        "reply_excerpt": content[:120].replace("\n", " "),
    }


def run_scenario(graph, scenario: dict, token: str | None) -> dict:
    """Run all turns sequentially; later turns see earlier replies as history."""
    messages: list[dict] = []
    turns = []
    totals = {"latency_s": 0.0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    scen_token = scenario.get("token", "valid")
    for user_text in scenario["turns"]:
        messages.append({"role": "user", "content": user_text})
        used_token = token if scen_token == "valid" else scen_token
        metrics = run_turn(graph, list(messages), used_token)
        turns.append(metrics)
        for key in totals:
            totals[key] += metrics[key]
    return {"turns": turns, **{k: round(v, 3) if k == "latency_s" else v for k, v in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="only first N scenarios")
    parser.add_argument("--runs", type=int, default=2, help="runs per variant (median reported)")
    parser.add_argument("--variants", type=str, default="baseline,multi")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    token = fetch_active_token()
    if not token:
        raise SystemExit(
            "No active PostVisit token in DB. Run: "
            ".venv/Scripts/python.exe -m app.seed_with_token"
        )

    graphs = load_graphs(variants)
    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS

    print(f"Variants={variants} scenarios={len(scenarios)} runs={args.runs}")
    print(f"Small model: {settings.groq_small_model} | Big model: {settings.groq_model}\n")

    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "runs": args.runs, "variants": {}}

    for variant, graph in graphs.items():
        print(f"=== {variant} ===")
        variant_results = {}
        grand = {"latencies": [], "calls": 0, "in_tok": 0, "out_tok": 0}
        for scenario in scenarios:
            run_latencies, sample = [], None
            calls = in_tok = out_tok = 0
            for _ in range(args.runs):
                outcome = run_scenario(graph, scenario, token)
                run_latencies.append(outcome["latency_s"])
                calls += outcome["llm_calls"]
                in_tok += outcome["input_tokens"]
                out_tok += outcome["output_tokens"]
                sample = outcome
            med = statistics.median(run_latencies)
            grand["latencies"].append(med)
            grand["calls"] += calls // args.runs
            grand["in_tok"] += in_tok // args.runs
            grand["out_tok"] += out_tok // args.runs
            variant_results[scenario["name"]] = {
                "median_latency_s": round(med, 3),
                "mean_calls_per_req": round(calls / args.runs, 2),
                "mean_input_tokens": round(in_tok / args.runs),
                "mean_output_tokens": round(out_tok / args.runs),
                "sample_reply": sample["turns"][0]["reply_excerpt"],
            }
            print(f"{scenario['name']:24s} med={med:6.2f}s  calls={calls // args.runs:2d}  "
                  f"in={in_tok // args.runs:6d}  out={out_tok // args.runs:5d}")
        n = len(scenarios) or 1
        results["variants"][variant] = {
            "per_scenario": variant_results,
            "totals": {
                "mean_latency_s": round(statistics.mean(grand["latencies"]), 3),
                "median_latency_s": round(statistics.median(grand["latencies"]), 3),
                "total_calls": grand["calls"],
                "total_input_tokens": grand["in_tok"],
                "total_output_tokens": grand["out_tok"],
                "grand_total_tokens": grand["in_tok"] + grand["out_tok"],
            },
        }
        print(f"-- totals: mean={statistics.mean(grand['latencies']):.2f}s "
              f"in={grand['in_tok']} out={grand['out_tok']}\n")

    out_path = Path(__file__).parent.parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")

    if "baseline" in results["variants"] and "multi" in results["variants"]:
        b, m = (results["variants"][v]["totals"] for v in ("baseline", "multi"))
        print("\n=== DELTA multi vs baseline ===")
        print(f"latency mean : {b['mean_latency_s']}s -> {m['mean_latency_s']}s")
        print(f"tokens total : {b['grand_total_tokens']} -> {m['grand_total_tokens']}")


if __name__ == "__main__":
    main()
