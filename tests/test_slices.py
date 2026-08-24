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


def test_dash_only_values_count_as_empty():
    dash = {k: "-" for k in ROW}
    slices = build_domain_slices([dash])
    assert all(slices[d] == "" for d in DOMAINS)


def test_multiple_rows_joined():
    second = {k: None for k in ROW}
    second.update({"tanggal": "2026-08-01 10:00:00", "keluhan": "Batuk", "diagnosa": "ISPA"})
    slices = build_domain_slices([ROW, second])
    assert slices["diagnosa"].count("Tanggal:") == 2
    assert "ISPA" in slices["diagnosa"]
