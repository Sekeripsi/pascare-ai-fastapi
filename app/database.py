from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://root:root1234@localhost:5432/simpus",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

metadata = MetaData()
metadata.reflect(bind=engine)

Pasien = metadata.tables["Pasien"]
RekamMedis = metadata.tables["RekamMedis"]
Pendaftaran = metadata.tables["Pendaftaran"]
Anamnesis = metadata.tables["Anamnesis"]
Diagnosis = metadata.tables["Diagnosis"]
Pemeriksaan = metadata.tables["Pemeriksaan"]
Tindakan = metadata.tables["Tindakan"]
Pengobatan = metadata.tables["Pengobatan"]
PulangRujuk = metadata.tables["PulangRujuk"]
PostVisit = metadata.tables["PostVisit"]
KajianAwal = metadata.tables["KajianAwal"]
Asuhan = metadata.tables["Asuhan"]
Lab = metadata.tables["Lab"]
CatatanDokter = metadata.tables["CatatanDokter"]
