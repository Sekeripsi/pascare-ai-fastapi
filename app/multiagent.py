"""Tiered selective fan-out multi-agent graph (comparison variant).

The baseline routed pipeline in app/agents.py stays untouched; this module
adds a specialist-decomposed graph for the efficiency study:

    safety_gate -> router(small LLM, multi-label domains)
        -> general / out_of_scope / crisis   (same behavior as baseline)
        -> load_record (one DB read, token-scoped)
           -> parallel specialists per domain (small LLM, skipped when slice empty)
           -> synthesizer (large LLM) grounded only on returned specialist notes

Only synthesizer/crisis/general/out_of_scope append to `messages`, so the
streaming endpoint never leaks intermediate specialist output.
"""

DOMAINS: tuple[str, ...] = ("diagnosa", "pemeriksaan", "pengobatan", "rencana")


def parse_router_output(text: str) -> tuple[str, list[str]]:
    """Parse the two-line router protocol; fail safe on anything unexpected.

    Unparsable route -> out_of_scope (same fallback as baseline). A rekam_medis
    route without parsable domains fans out to ALL specialists: a safe superset
    that can never drop a relevant domain.
    """
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
        domains = list(DOMAINS)
    return route, domains


def _fmt(value) -> str:
    if value is None or str(value).strip() in ("", "-"):
        return "-"
    return str(value)


def _has(row: dict, *keys: str) -> bool:
    for k in keys:
        value = row.get(k)
        if value in (None, ""):
            continue
        if isinstance(value, str) and value.strip() in ("", "-"):
            continue
        return True
    return False


def build_domain_slices(rows: list[dict]) -> dict[str, str]:
    """Format visit rows into per-domain prompt text; '' marks an empty slice.

    `rows` items use the same keys as the baseline rekam_results entries.
    Empty slices let the specialist node skip its LLM call entirely.
    """
    parts: dict[str, list[str]] = {d: [] for d in DOMAINS}
    filled: dict[str, bool] = {d: False for d in DOMAINS}

    for row in rows:
        tanggal = _fmt(row.get("tanggal"))

        keluhan = _fmt(row.get("keluhan"))
        diagnosa = _fmt(row.get("diagnosa"))
        icd = _fmt(row.get("kode_icd"))
        parts["diagnosa"].append(
            f"Tanggal: {tanggal}\nKeluhan: {keluhan}\nDiagnosa: {diagnosa} (Kode ICD: {icd})"
        )
        if _has(row, "keluhan", "diagnosa", "kode_icd"):
            filled["diagnosa"] = True

        parts["pemeriksaan"].append(
            f"Tanggal: {tanggal} | Poliklinik: {_fmt(row.get('poliklinik'))}\n"
            f"Suhu {_fmt(row.get('suhu'))}°C, Nadi {_fmt(row.get('nadi'))}x/menit, "
            f"Tekanan Darah {_fmt(row.get('sistol'))}/{_fmt(row.get('diastol'))} mmHg, "
            f"Respirasi {_fmt(row.get('respirasi'))}x/menit\n"
            f"Keadaan: {_fmt(row.get('keadaan'))}, Kesadaran: {_fmt(row.get('kesadaran'))}"
        )
        if _has(row, "suhu", "nadi", "respirasi", "sistol", "keadaan", "kesadaran"):
            filled["pemeriksaan"] = True

        parts["pengobatan"].append(
            f"Tanggal: {tanggal}\nTindakan: {_fmt(row.get('tindakan'))}\n"
            f"Pengobatan: {_fmt(row.get('pengobatan'))}"
        )
        if _has(row, "tindakan", "pengobatan"):
            filled["pengobatan"] = True

        parts["rencana"].append(
            f"Tanggal: {tanggal}\nStatus pulang: {_fmt(row.get('status_pulang'))}\n"
            f"KIE: {_fmt(row.get('kie'))}\nRencana: {_fmt(row.get('plan'))}"
        )
        if _has(row, "status_pulang", "kie", "plan"):
            filled["rencana"] = True

    return {
        d: "\n---\n".join(parts[d]).strip() if filled[d] else ""
        for d in DOMAINS
    }
