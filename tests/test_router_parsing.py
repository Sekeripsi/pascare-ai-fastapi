from app.multiagent import DOMAINS, parse_router_output


def test_clean_rekam_medis_with_domains():
    route, domains = parse_router_output("route: rekam_medis\ndomains: pemeriksaan, pengobatan")
    assert route == "rekam_medis"
    assert domains == ["pemeriksaan", "pengobatan"]


def test_general_has_no_domains():
    route, domains = parse_router_output("route: general\ndomains: -")
    assert route == "general"
    assert domains == []


def test_out_of_scope_clean():
    route, domains = parse_router_output("route: out_of_scope\ndomains: -")
    assert route == "out_of_scope"
    assert domains == []


def test_unparsable_fails_safe_to_out_of_scope():
    route, domains = parse_router_output("saya tidak tahu harus jawab apa")
    assert route == "out_of_scope"
    assert domains == []


def test_empty_output_fails_safe():
    assert parse_router_output("") == ("out_of_scope", [])


def test_rekam_medis_without_domains_gets_all():
    route, domains = parse_router_output("route: rekam_medis")
    assert route == "rekam_medis"
    assert list(domains) == list(DOMAINS)


def test_unknown_domains_filtered_then_all():
    _, domains = parse_router_output("route: rekam_medis\ndomains: lab, jadwal")
    assert list(domains) == list(DOMAINS)


def test_duplicates_removed_order_kept():
    _, domains = parse_router_output(
        "route: rekam_medis\ndomains: pengobatan, pengobatan, diagnosa"
    )
    assert domains == ["pengobatan", "diagnosa"]


def test_case_insensitive_and_semicolons():
    _, domains = parse_router_output("Route: REKAM_MEDIS\nDomains: Diagnosa; Rencana")
    assert domains == ["diagnosa", "rencana"]
