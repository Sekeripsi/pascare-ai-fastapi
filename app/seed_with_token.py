import uuid
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta

from sqlalchemy import insert
from app.database import (
    SessionLocal,
    Pasien,
    Pendaftaran,
    KajianAwal,
    Anamnesis,
    Pemeriksaan,
    Diagnosis,
    Tindakan,
    Pengobatan,
    PulangRujuk,
    Asuhan,
    Lab,
    RekamMedis,
    PostVisit,
)

JWT_SECRET = "simpus-jwt-secret-key-2026"


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def generate_jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    message = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def seed():
    db = SessionLocal()
    try:
        pasien_id = str(uuid.uuid4())
        pendaftaran_id = str(uuid.uuid4())
        kajian_awal_id = str(uuid.uuid4())
        anamnesis_id = str(uuid.uuid4())
        pemeriksaan_id = str(uuid.uuid4())
        diagnosis_id = str(uuid.uuid4())
        tindakan_id = str(uuid.uuid4())
        pengobatan_id = str(uuid.uuid4())
        pulang_rujuk_id = str(uuid.uuid4())
        asuhan_id = str(uuid.uuid4())
        lab_id = str(uuid.uuid4())
        rekam_medis_id = str(uuid.uuid4())
        post_visit_id = str(uuid.uuid4())

        now = datetime.now(datetime.now().astimezone().tzinfo)
        expires_at = now + timedelta(days=7)

        token_payload = {
            "type": "post_visit",
            "rekamMedisId": rekam_medis_id,
            "pasienId": pasien_id,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        token = generate_jwt(token_payload, JWT_SECRET)

        db.execute(insert(Pasien).values(
            id=pasien_id,
            nik=str(int(now.timestamp())),
            noKk="320101010101000001",
            noJkn="0001234567890",
            catatanJkn="",
            noJamkesos="",
            catatanJamkesos="",
            nama="Rizky Ramadhan",
            jenisKelamin="L",
            tempatLahir="Jakarta",
            tanggalLahir=datetime(1990, 1, 1),
            umur=35,
            agama="ISLAM",
            statusKawin="KAWIN",
            alamatTinggal="Jl. Merdeka No. 10, Jakarta",
            alamatKtp="Jl. Merdeka No. 10, Jakarta",
            alamatDomisili="Jl. Merdeka No. 10, Jakarta",
            provinsi="DKI Jakarta",
            kabupaten="Jakarta Pusat",
            kecamatan="Gambir",
            kelurahanDesa="Gambir",
            rtRw="001/002",
            noTlp="081234567890",
            pemilikNoTlp="Rizky Ramadhan",
            email="rizky@example.com",
            layananDifabel="TIDAK",
            pendidikan="S1",
            pekerjaan="Karyawan Swasta",
            golDarah="O",
            rhesus="POSITIF",
            namaIbuKandung="Siti Aminah",
            createdAt=now,
            updatedAt=now,
        ))

        db.execute(insert(Pendaftaran).values(
            id=pendaftaran_id,
            tglKunjungan=datetime(2026, 8, 22, 9, 0, 0),
            noAntrian="A-001",
            unitLayanan="Puskesmas",
            jenisLayanan="Rawat Jalan",
            noRegis="REG-2026-001",
            poliklinik="UMUM",
            kehadiran="HADIR",
            targetStatus="TIDAK",
            pembayaran="BPJS",
            catatan="Kunjungan rutin",
        ))

        db.execute(insert(KajianAwal).values(
            id=kajian_awal_id,
            alergi="Tidak ada",
            riwayatPenyakitDahulu="Tidak ada",
            riwayatPenyakitKeluarga="Hipertensi",
        ))

        db.execute(insert(Anamnesis).values(
            id=anamnesis_id,
            keluhan="Sakit kepala sebelah kiri sejak 3 hari yang lalu, disertai mual",
        ))

        db.execute(insert(Pemeriksaan).values(
            id=pemeriksaan_id,
            keadaan="Baik",
            kesadaran="KOMPOS_MENTIS",
            respirasi=20,
            suhu=36.8,
            nadi=78,
            sistol=130,
            diastol=85,
        ))

        db.execute(insert(Diagnosis).values(
            id=diagnosis_id,
            diagnosis="Cephalgia (sakit kepala tegang)",
            kodeIcd="R51",
        ))

        db.execute(insert(Tindakan).values(
            id=tindakan_id,
            tindakan="Pemberian analgesik dan konseling",
        ))

        db.execute(insert(Pengobatan).values(
            id=pengobatan_id,
            pengobatan=json.dumps([
                {"nama": "Parasetamol 500 mg", "dosis": "3x1 tablet", "jumlah": 10, "aturan": "Sesudah makan"},
                {"nama": "Vitamin B Complex", "dosis": "1x1 tablet", "jumlah": 10, "aturan": "Pagi hari"},
            ]),
        ))

        db.execute(insert(PulangRujuk).values(
            id=pulang_rujuk_id,
            tglPulang=now,
            statusPulang="SEMBUH",
            kie="Istirahat yang cukup, minum air putih yang banyak",
            plan="Kontrol ulang jika gejala berlanjut",
            rencKunjBerikutnya=now + timedelta(days=7),
            rencPemeriksaan6Bln=now + timedelta(days=180),
            rujukInternal="",
            rujukEksternal="",
            remindedKunjunganD1="NOT_REMINDED",
            remindedKunjunganD3="NOT_REMINDED",
            remindedKunjunganD7="NOT_REMINDED",
            remindedPemeriksaanD1="NOT_REMINDED",
            remindedPemeriksaanD3="NOT_REMINDED",
            remindedPemeriksaanD7="NOT_REMINDED",
            remindedKunjunganLate="NOT_REMINDED",
            remindedPemeriksaanLate="NOT_REMINDED",
        ))

        db.execute(insert(Asuhan).values(
            id=asuhan_id,
            diagnosaData="Cephalgia",
            diagnosa="Sakit kepala tegang",
            intervensi="Pemberian obat dan konseling",
            implementasi="Pasien diberikan obat dan instruksi istirahat",
            evaluasi="Pasien membaik",
        ))

        db.execute(insert(Lab).values(
            id=lab_id,
            permintaanPemeriksaan="Tidak ada",
        ))

        db.execute(insert(RekamMedis).values(
            id=rekam_medis_id,
            pasienId=pasien_id,
            pendaftaranId=pendaftaran_id,
            kajianAwalId=kajian_awal_id,
            anamnesisId=anamnesis_id,
            pemeriksaanId=pemeriksaan_id,
            diagnosisId=diagnosis_id,
            tindakanId=tindakan_id,
            pengobatanId=pengobatan_id,
            pulangRujukId=pulang_rujuk_id,
            asuhanId=asuhan_id,
            labId=lab_id,
            status="SELESAI",
            completedAt=now,
        ))

        db.execute(insert(PostVisit).values(
            id=post_visit_id,
            rekamMedisId=rekam_medis_id,
            token=token,
            expiresAt=expires_at,
            isActive=True,
        ))

        db.commit()
        print("=== SEED BERHASIL ===")
        print(f"Pasien ID   : {pasien_id}")
        print(f"Rekam Medis : {rekam_medis_id}")
        print(f"Token       : {token}")
        print(f"Expires At  : {expires_at.isoformat()}")
        print("======================")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
