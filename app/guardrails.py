"""Guardrail layer for the post-visit chatbot.

Everything in this module is deterministic (no LLM involved) so that the
safety-critical behavior of the chat cannot drift with prompt changes:

1. Crisis detection: keyword gate that routes emergency messages to a fixed
   reply BEFORE any LLM call happens.
2. Canned replies: fixed Indonesian messages for crisis, off-topic, and
   missing-record situations.
3. Disclaimer: appended server-side to every answer so both the invoke and
   streaming paths carry it.
"""

# Strong emergency signals only. Kept narrow on purpose to limit false
# positives; tune this list if the demo shows misses.
CRISIS_KEYWORDS: tuple[str, ...] = (
    "bunuh diri",
    "bunuhdiri",
    "mengakhiri hidup",
    "mau mati",
    "pengen mati",
    "ingin mati",
    "overdosis",
    "over dosis",
    "minum semua",
    "menelan semua",
    "nyeri dada berat",
    "sesak napas berat",
    "pendarahan hebat",
    "kejang",
    "pingsan",
    "tidak sadarkan diri",
)

CRISIS_REPLY = (
    "Mohon maaf mendengar kondisi Anda. Saya hanya asisten pasca-kunjungan dan "
    "tidak dapat menangani keadaan darurat.\n\n"
    "SEGERA hubungi 119 atau 112, atau datangi IGD terdekat sekarang juga. "
    "Jika memungkinkan, minta orang di sekitar Anda untuk menemani."
)

OFF_TOPIC_REPLY = (
    "Maaf, saya hanya bisa membantu menjelaskan kunjungan medis Anda yang baru "
    "selesai — misalnya diagnosa, hasil pemeriksaan, tindakan, pengobatan, atau "
    "rencana kontrol. Silakan ajukan pertanyaan seputar kunjungan Anda."
)

NO_DATA_REPLY = (
    "Maaf, data rekam medis kunjungan Anda tidak ditemukan atau tautan kunjungan "
    "sudah tidak aktif. Silakan gunakan tautan kunjungan yang masih berlaku, atau "
    "hubungi puskesmas untuk bantuan."
)

STREAM_FALLBACK_REPLY = (
    "Maaf, terjadi kendala saat memproses pertanyaan Anda. Silakan coba lagi."
)

DISCLAIMER = (
    "Catatan: informasi ini bersifat edukatif dan bukan pengganti diagnosis "
    "atau nasihat dari dokter yang merawat Anda."
)


def contains_crisis(text: str) -> bool:
    """Return True when the message matches an emergency keyword."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in CRISIS_KEYWORDS)
