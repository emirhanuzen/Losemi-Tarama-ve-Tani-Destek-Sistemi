"""Proje genelinde kullanılan dizin yolları ve ortak sabitler."""

from pathlib import Path

# Proje kök dizini (src/ klasörünün bir üstü)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

# Her pipeline aşamasının ara çıktılarının kaydedileceği alt klasörler
STAGE_OUTPUT_DIRS = {
    "dataset": OUTPUTS_DIR / "dataset",
    "preprocessing": OUTPUTS_DIR / "preprocessing",
    "segmentation": OUTPUTS_DIR / "segmentation",
    "features": OUTPUTS_DIR / "features",
    "classification": OUTPUTS_DIR / "classification",
    "evaluation": OUTPUTS_DIR / "evaluation",
}

# Sınıflandırma hedefi: 3 sınıf. Klasör adları data/raw/<sınıf>/ ile birebir aynıdır.
# ALL = akut lenfoblastik lösemi, AML = akut miyeloid lösemi, Healthy = normal hücre.
CLASS_NAMES = ["ALL", "AML", "Healthy"]
ABNORMAL_CLASSES = ["ALL", "AML"]  # Risk skorunda "anormal" sayılan sınıflar

# Veri seti indeksi (dosya yolu, etiket, grup_id, görüntü tipi ...) bu CSV'ye yazılır
DATASET_INDEX_CSV = PROCESSED_DATA_DIR / "dataset_index.csv"
DUPLICATE_PAIRS_CSV = PROCESSED_DATA_DIR / "kopya_ciftleri.csv"

RANDOM_STATE = 42  # Tekrarlanabilir sonuçlar için sabit rastgelelik tohumu
