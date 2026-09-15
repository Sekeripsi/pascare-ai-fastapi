"""Per-request latency + token logging for the chat graph.

Captures TTFT (time to first streamed token), total latency (synthesizer/
orchestrator finish), and prompt/completion token counts from every LLM
call in the chain. Records are appended to a CSV at metrics/llm_log.csv.

Columns: scenario_id, agent, ttft_s, total_latency_s, prompt_tokens,
completion_tokens, total_tokens, llm_calls, graph_variant, timestamp
"""
import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings


CSV_DIR = Path(os.getenv("LLM_LOG_DIR", "metrics"))
CSV_PATH = CSV_DIR / "llm_log.csv"

CSV_FIELDS = [
    "scenario_id",
    "agent",
    "ttft_s",
    "total_latency_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "llm_calls",
    "graph_variant",
    "timestamp",
]


def _ensure_csv():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


class RequestMetrics:
    """Tracks timing and token usage for a single chat request.

    Threaded through the graph via a callback handler and through the
    route via direct method calls.
    """

    def __init__(self, scenario_id: str | None = None):
        self.scenario_id = scenario_id or str(uuid.uuid4())
        self.graph_variant = settings.graph_variant
        self._t0: float | None = None
        self._ttft: float | None = None
        self._t_end: float | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.agents: list[str] = []

    def start(self):
        self._t0 = time.perf_counter()

    def mark_first_token(self):
        if self._ttft is None and self._t0 is not None:
            self._ttft = time.perf_counter() - self._t0

    def mark_done(self):
        if self._t0 is not None:
            self._t_end = time.perf_counter() - self._t0

    def add_usage(self, input_tokens: int, output_tokens: int, total: int, agent: str = ""):
        self.prompt_tokens += input_tokens
        self.completion_tokens += output_tokens
        self.total_tokens += total
        self.llm_calls += 1
        if agent:
            self.agents.append(agent)

    @property
    def ttft_s(self) -> float | None:
        return round(self._ttft, 4) if self._ttft is not None else None

    @property
    def total_latency_s(self) -> float | None:
        return round(self._t_end, 4) if self._t_end is not None else None

    def write(self):
        _ensure_csv()
        row = {
            "scenario_id": self.scenario_id,
            "agent": ";".join(self.agents) if self.agents else "-",
            "ttft_s": self.ttft_s if self.ttft_s is not None else "",
            "total_latency_s": self.total_latency_s if self.total_latency_s is not None else "",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "graph_variant": self.graph_variant,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


class LLMMetricsCallback(BaseCallbackHandler):
    """LangChain callback that feeds token usage into a RequestMetrics."""

    def __init__(self, metrics: RequestMetrics):
        self.metrics = metrics

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # Record the agent name from the invocation tags / metadata.
        tags = kwargs.get("tags") or []
        if tags:
            self.metrics.agents.append(tags[0])

    def on_llm_end(self, response, **kwargs) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            llm_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            if llm_usage:
                usage = {
                    "input_tokens": llm_usage.get("prompt_tokens", 0),
                    "output_tokens": llm_usage.get("completion_tokens", 0),
                    "total_tokens": llm_usage.get("total_tokens", 0),
                }
        if not usage:
            try:
                msg = response.generations[0][0].message
                usage = getattr(msg, "usage_metadata", None)
            except (IndexError, AttributeError):
                usage = None
        if usage:
            self.metrics.add_usage(
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
                int(usage.get("total_tokens", 0)),
            )

    def on_llm_new_token(self, token, **kwargs) -> None:
        self.metrics.mark_first_token()
