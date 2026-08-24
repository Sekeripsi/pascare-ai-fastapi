"""One-off probe: which free Groq model best serves the multi-label router?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.multiagent import MULTI_ROUTER_PROMPT, parse_router_output

QUESTIONS = [
    "Tekanan darah saya waktu diperiksa berapa?",
    "Diagnosa saya apa dan obatnya apa saja?",
    "Obat diminum kapan saja?",
    "Kapan jadwal kontrol berikutnya saya?",
    "Halo, kamu siapa?",
    "Terima kasih banyak ya",
    "Buatkan kode Python dong",
    "Jelaskan penyakit jantung secara umum",
]
# Expected route for sanity scoring.
EXPECTED = ["rekam_medis", "rekam_medis", "rekam_medis", "rekam_medis",
            "general", "general", "out_of_scope", "out_of_scope"]


def main():
    for model in ("openai/gpt-oss-20b", "qwen/qwen3.6-27b"):
        llm = ChatGroq(model=model, temperature=0, max_tokens=600)
        correct = 0
        latencies = []
        print(f"=== {model}")
        for question, expected in zip(QUESTIONS, EXPECTED):
            start = time.perf_counter()
            response = llm.invoke([
                SystemMessage(content=MULTI_ROUTER_PROMPT),
                HumanMessage(content=f"Pesan terakhir dari pasien: {question}"),
            ])
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            text = response.content if isinstance(response.content, str) else ""
            route, domains = parse_router_output(text)
            correct += route == expected
            usage = getattr(response, "usage_metadata", None) or {}
            print(f"{elapsed * 1000:5.0f}ms out={usage.get('output_tokens', '?'):>4} "
                  f"route={route:12s} domains={str(domains):40s} "
                  f"{'OK' if route == expected else 'MISS'} raw={text[:50]!r}")
        mean_ms = sum(latencies) / len(latencies) * 1000
        print(f"score={correct}/{len(QUESTIONS)} mean={mean_ms:.0f}ms\n")


if __name__ == "__main__":
    main()
