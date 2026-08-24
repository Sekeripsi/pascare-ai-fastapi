import uuid
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta, timezone

from sqlalchemy import insert, delete
from app.database import SessionLocal, RekamMedis, PostVisit

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
        existing = db.query(PostVisit).first()
        if existing:
            db.execute(delete(PostVisit))
            db.commit()
            print("Menghapus PostVisit lama...")

        rekam = db.query(RekamMedis).order_by(RekamMedis.c.createdAt.desc()).first()
        if not rekam:
            print("Tidak ada data RekamMedis di database!")
            return

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=7)

        token_payload = {
            "type": "post_visit",
            "rekamMedisId": str(rekam.id),
            "pasienId": str(rekam.pasienId),
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        token = generate_jwt(token_payload, JWT_SECRET)

        post_visit_id = str(uuid.uuid4())
        db.execute(insert(PostVisit).values(
            id=post_visit_id,
            rekamMedisId=str(rekam.id),
            token=token,
            expiresAt=expires_at,
            isActive=True,
            createdAt=now,
            updatedAt=now,
        ))
        db.commit()

        print("=== SEED BERHASIL ===")
        print(f"Rekam Medis : {rekam.id}")
        print(f"Pasien ID   : {rekam.pasienId}")
        print(f"Token       : {token}")
        print(f"Expires At  : {expires_at.isoformat()}")
        print("======================")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
