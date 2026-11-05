"""
Veri seti kaynağı tespiti: Kaggle görüntüleri ile ALL-IDB (ALL-IDB1, ALL-IDB2) arasındaki örtüşme.

Amaç: Kaggle veri setinin (klinik kaynağı belirtilmemiş) ALL-IDB'den türetilip türetilmediğini ölçmek ve
dış doğrulamada ALL-IDB'nin hangi görüntülerinin GÜVENLE (eğitimde hiç görülmemiş) kullanılabileceğini
belirlemek. Hiçbir dosya silinmez veya değiştirilmez; ALL-IDB görüntüleri lisans gereği repoya girmez
(eşleşme listesi ve örnek figür data/processed/ altına yazılır, bu klasör .gitignore kapsamında).

ALL-IDB adlandırması: ImXXX_Y; Y=1 → ALL hastası, Y=0 → sağlıklı birey.
- ALL-IDB1: tam yaymalar (2592x1944 / 1712x1368) + .xyc blast koordinatları.
- ALL-IDB2: 257x257 tek hücre kırpımları (ALL-IDB1 yaymalarından kesilmiş).

Yöntem (Adım 1'deki kopya tespitinin genişletilmiş hâli; döndürme, aynalama, renk kayması, ölçek):
1. Tam görüntü eşleşmesi: döndürmeye/aynalamaya dayanıklı pHash adayları + ORB/RANSAC doğrulaması,
   Adım 1'in kabul kuralıyla (dataset.is_near_duplicate). ALL-IDB1 iki geometride karşılaştırılır:
   en-boy oranı korunmuş ve 256x256 kareye sıkıştırılmış (pilot: Kaggle'ın 256x256 yaymaları 4:3 ALL-IDB1
   görüntülerinin en-boy oranı bozularak küçültülmüş hâlleri; benzerlik dönüşümü bunu modelleyemiyor).
2. Tek hücre eşleşmesi: Kaggle görüntüsü ↔ ALL-IDB2 kırpımları ve ALL-IDB1'den çıkarılan çekirdek merkezli
   257x257 kırpımlar. pHash adayları + geniş bağlamlı şablon eşleştirmesi (NCC; 72 döndürme/ayna varyantı ×
   5 ölçek). Pürüzsüz yakın çekimlerde ORB/SIFT güvenilmez (az köşe, merkez görüntülerde sahte inlier).
3. Kırpım arama (1-2'de eşleşmeyen yaymalar): ORB oylaması (ALL-IDB1, en-boy korunmuş, aynalı dahil) ve
   küçültülmüş ALL-IDB1 üzerinde çok ölçekli şablon eşleştirme (öteleme + ölçek + ayna).
4. Bağlantılar: Kaggle kopya grupları (grup_id) üzerinden yayılım; ALL-IDB1'in kendi içindeki kopyaları;
   ALL-IDB2 kırpımlarının kaynak ALL-IDB1 yayması (şablon eşleştirme, doğal ölçek).

Negatif kontrol: ALL-IDB'de AML yok; AML görüntülerinin eşleşmesi yanlış pozitif demektir.

Çalıştırma (proje kökünden):
    python -m src.allidb_overlap [--allidb-dir <ALL_IDB1 ve ALL_IDB2 klasörlerini içeren dizin>]
    (varsayılan config.ALLIDB_DIR: LOSEMI_ALLIDB_DIR ortam değişkeni; yoksa Windows C:/Users/Emirhan/Downloads,
    WSL /mnt/c/Users/Emirhan/Downloads)
"""

import argparse
import re
import time
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import ALLIDB_DIR, DATASET_INDEX_CSV, N_JOBS, PROCESSED_DATA_DIR, SPLIT_CSV
from src.dataset import (
    BLACK_PIXEL_THRESHOLD,
    HASH_CANDIDATE_DISTANCE,
    MIN_NUCLEUS_FRACTION,
    ORB_MIN_INLIERS_FAR,
    ORB_RATIO,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    assign_groups,
    is_near_duplicate,
    nucleus_mask,
    orb_inliers,
    rotation_phashes,
)
from src.utils import load_image

ALLIDB_DEFAULT_DIR = ALLIDB_DIR   # config: LOSEMI_ALLIDB_DIR ortam değişkeni veya platform varsayılanı
MATCH_CSV = PROCESSED_DATA_DIR / "allidb_eslesme.csv"
STATUS_CSV = PROCESSED_DATA_DIR / "allidb_goruntu_durumu.csv"
FIGURE_PATH = PROCESSED_DATA_DIR / "allidb_eslesme_ornekleri.png"
CACHE_DIR = PROCESSED_DATA_DIR / "allidb_ara"   # Hash önbelleği (yeniden çalıştırmayı hızlandırır)
PROGRESS_EVERY = 100

# --- Parametreler ---
# Tam görüntü ORB'u Adım 1 ile aynı: uzun kenar 384 px, 500 nokta (dataset.orb_features ile aynı mantık).
ORB_SIDE = 384
ORB_FEATURES = 500
# Hash adayı: Adım 1 eşiği (≤ 14); her referans kümesinde Kaggle görüntüsü başına en yakın 5 aday doğrulanır
# (Healthy yaymaları birbirine çok benzediği için ≤ 14'te onlarca aday çıkabiliyor; gerçek kaynak pilotta hep ilk 1-2'deydi).
CANDIDATES_PER_SET = 5
# Hücre kırpımı: ALL-IDB2 ile aynı boy (257 px, doğal çözünürlük). Çekirdek merkezleri 1/4 ölçekte B−G maskesinden;
# 64 px'ten yakın merkezler tek hücre sayılır (bir lökositin parçalı çekirdeği birden fazla bileşen verebilir).
CELL_CROP = 257
CELL_DETECT_SCALE = 0.25
CELL_MERGE_DIST = 64
# Geniş bağlamlı NCC: Kaggle görüntüsünün 256 px karesinde merkez 180 px (döndürme köşelerinden bağımsız,
# 180 ≈ 256/√2) çekirdekle birlikte çevre eritrositlerini de içerir; eritrosit düzeni hücreye özgü bir "parmak izi".
# Merkezde yalnızca çekirdeği içeren küçük şablon farklı hücrelerde de 0.93-0.97 verdi (görsel incelemede yanlış).
# Ölçekler 0.6-1.0: yakınlaştırma augmentasyonu (Kaggle görüntüsü kaynağın daha dar bir alanını gösterebilir).
NCC_REGION = 180
NCC_SCALES = (0.6, 0.7, 0.8, 0.9, 1.0)
# Kabul eşiği 0.95: pilotta (25 ALL + 25 AML yakın çekim) gerçek eşleşmeler 0.990-0.999, AML (negatif kontrol)
# en fazla 0.85, farklı ALL hücreleri en fazla 0.89. 0.95 iki küme arasındaki boşluğun ortası.
NCC_MIN = 0.95
# Erken durdurma: bir aday >= 0.98 verdiyse (gerçek eşleşmeler 0.99+) kalan varyant/ölçek/adaylar denenmez.
NCC_EARLY_STOP = 0.98
# Hangi Kaggle görüntü tipi hangi referans türüyle karşılaştırılır: tam yayma <-> tam ALL-IDB1 görüntüsü,
# tek hücre yakın çekimi <-> hücre kırpımı (yayma ile tek hücre kırpımının pHash benzerliği tesadüftür).
TYPE_REFERENCES = {"yayma": ("tam", "sikistirilmis"), "yakin_cekim": ("idb2", "hucre")}
# Paralel işler: 100 görüntülük parçalar, en fazla N_JOBS işçi; her parça bitince ilerleme yazdırılır.
CHUNK = 100
# Kırpım arama: ALL-IDB1 uzun kenarı 512 px (ORB), 3000 nokta (tek Kaggle görüntüsü kaynağın küçük bir parçası olabilir;
# yoğun nokta gerekir). Kabul: Adım 1'in hash'siz ("uzak") kuralı, ≥ 30 inlier.
CROP_DB_SIDE = 512
CROP_DB_FEATURES = 3000
# Şablon taraması: ALL-IDB1 uzun kenarı 320 px'e küçültülür; Kaggle görüntüsünün merkez %70'i (döndürme köşeleri
# dışarıda) 32-160 px genişliklerde aranır (kaynak genişliğinin ~%15-75'i kadar bir kırpım). Döndürme taranmaz
# (döndürülmüş kırpımlar ORB oylamasıyla aranır).
# SONUÇ: Şablon taraması TEK BAŞINA kabul edilmez. 0.90'ı geçen 15 yaymanın (14 Healthy, 1 ALL) hiçbirinde ORB geometrik
# destek bulamadı (hizalama denemesinde 0 inlier); 8'i etiketle çelişiyordu (Kaggle Healthy ↔ ALL-IDB _1), aynı birkaç
# ALL-IDB görüntüsü tekrar tekrar çıkıyordu ve skorlar AML negatif kontrolünün en yükseğine (0.845) çok yakındı
# (0.90-0.94): tekdüze eritrosit dokusunda şablon korelasyonu ayırt edici değil. Bu satırlar CSV'de kabul=False.
# ORB kırpım eşleşmeleri ise hizalanınca görsel olarak hücre hücre örtüştü (düşük NCC'liler dahil; fark gürültü/bulanıklık).
TEMPLATE_SIDE = 320
TEMPLATE_WIDTHS = (32, 48, 64, 96, 128, 160)
TEMPLATE_MIN_NCC = 0.90
TEMPLATE_ACCEPTED = False
# ALL-IDB2 → ALL-IDB1 kaynak eşlemesi: doğal ölçekte kırpım, 1/4 küçültülmüş yaymada aranır; NCC ≥ 0.95
# (pilotta doğru kaynak 0.97-1.00; ALL-IDB1'in kendi içindeki kopyaları da 0.99 verdiği için tüm ≥ 0.95'ler alınır).
SOURCE_SCALE = 0.25
SOURCE_MIN_NCC = 0.95


# ---------------------------------------------------------------------------
# Dosyalar
# ---------------------------------------------------------------------------

def list_allidb(root: Path) -> pd.DataFrame:
    """
    ALL-IDB1 ve ALL-IDB2 görüntülerini listeler ve etiketlerini dosya adından çıkarır.

    Parametreler:
        root: ALL_IDB1 ve ALL_IDB2 klasörlerini içeren dizin (alt klasör derinliği önemsiz).

    Döndürür:
        idb_dosya, idb_yolu, idb_set (IDB1/IDB2), idb_etiket (ALL/Healthy) sütunlu tablo.
    """
    rows = []
    for subset, pattern in [("IDB1", "ALL_IDB1/**/*.jpg"), ("IDB2", "ALL_IDB2/**/*.tif")]:
        for p in sorted(root.glob(pattern)):
            m = re.match(r"Im\d+_([01])\.", p.name)
            if m:
                rows.append({"idb_dosya": p.name, "idb_yolu": str(p), "idb_set": subset,
                             "idb_etiket": "ALL" if m.group(1) == "1" else "Healthy"})
    table = pd.DataFrame(rows)
    if table.empty:
        raise FileNotFoundError(f"ALL-IDB görüntüsü bulunamadı: {root}")
    return table


def load_kaggle() -> pd.DataFrame:
    """
    Kaggle veri seti indeksini kilitli veri ayrımıyla (geliştirme/test) birleştirir.

    Döndürür:
        dosya_yolu, etiket, grup_id, tip, genislik, yukseklik, bolum sütunlu tablo.
    """
    index = pd.read_csv(DATASET_INDEX_CSV)[["dosya_yolu", "etiket", "grup_id", "tip", "genislik", "yukseklik"]]
    split = pd.read_csv(SPLIT_CSV)[["dosya_yolu", "bolum"]]
    return index.merge(split, on="dosya_yolu", how="left")


def gray_of(path: str) -> np.ndarray:
    """
    Görüntüyü gri seviye okur.

    Parametreler:
        path: Dosya yolu.

    Döndürür:
        uint8 gri görüntü.
    """
    return cv2.cvtColor(load_image(path), cv2.COLOR_RGB2GRAY)


def resize_long(gray: np.ndarray, side: int) -> np.ndarray:
    """
    Görüntüyü en-boy oranını koruyarak uzun kenarı side olacak şekilde ölçekler.

    Parametreler:
        gray: Gri görüntü.
        side: Hedef uzun kenar (px).

    Döndürür:
        Ölçeklenmiş görüntü.
    """
    s = side / max(gray.shape)
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Referanslar: ALL-IDB1 (iki geometri), ALL-IDB2, ALL-IDB1 hücre kırpımları
# ---------------------------------------------------------------------------

def idb1_cell_centers(gray_small: np.ndarray, color_small: np.ndarray) -> list[tuple[int, int]]:
    """
    Küçültülmüş ALL-IDB1 yaymasında lökosit çekirdeği merkezlerini bulur (Adım 1'in B−G çekirdek maskesi).

    Parametreler:
        gray_small: Küçültülmüş gri görüntü (yalnızca boyut için).
        color_small: Küçültülmüş RGB görüntü.

    Döndürür:
        Küçük ölçekte (x, y) merkezler listesi (yakın merkezler birleştirilmiş).
    """
    mask = nucleus_mask(color_small)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    min_area = MIN_NUCLEUS_FRACTION * gray_small.size
    merge = CELL_MERGE_DIST * CELL_DETECT_SCALE
    centers = []
    for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        c = cents[k]
        if all(np.hypot(c[0] - x, c[1] - y) >= merge for x, y in centers):
            centers.append((float(c[0]), float(c[1])))
    return centers


def build_references(idb: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Karşılaştırmada kullanılacak tüm referans görüntüleri ve pHash'lerini hazırlar.

    Referans türleri: "tam" (ALL-IDB1, en-boy korunmuş), "sikistirilmis" (ALL-IDB1, 256x256 kare),
    "idb2" (ALL-IDB2 kırpımı), "hucre" (ALL-IDB1'den çekirdek merkezli 257x257 kırpım).
    Görüntüler tek tek okunur; bellekte hash, küçük gri görüntüler (ORB/şablon için) ve kırpım koordinatları tutulur.

    Parametreler:
        idb: list_allidb çıktısı.

    Döndürür:
        (referans tablosu [ref_id, tur, idb_dosya, idb_set, idb_etiket, kirpim (x0, y0)], hash listesi (72,)).
    """
    rows, hashes, start = [], [], time.time()
    for i, r in enumerate(idb.itertuples(index=False)):
        color = load_image(r.idb_yolu)
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        base = {"idb_dosya": r.idb_dosya, "idb_set": r.idb_set, "idb_etiket": r.idb_etiket, "kirpim_x": -1, "kirpim_y": -1}
        if r.idb_set == "IDB2":
            rows.append({**base, "tur": "idb2"})
            hashes.append(rotation_phashes(gray))
        else:
            rows.append({**base, "tur": "tam"})
            hashes.append(rotation_phashes(gray))
            rows.append({**base, "tur": "sikistirilmis"})
            hashes.append(rotation_phashes(cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)))
            small = cv2.resize(color, None, fx=CELL_DETECT_SCALE, fy=CELL_DETECT_SCALE, interpolation=cv2.INTER_AREA)
            h, w = gray.shape
            for cx, cy in idb1_cell_centers(small[..., 0], small):
                x0 = int(np.clip(cx / CELL_DETECT_SCALE - CELL_CROP // 2, 0, w - CELL_CROP))
                y0 = int(np.clip(cy / CELL_DETECT_SCALE - CELL_CROP // 2, 0, h - CELL_CROP))
                rows.append({**base, "tur": "hucre", "kirpim_x": x0, "kirpim_y": y0})
                hashes.append(rotation_phashes(gray[y0:y0 + CELL_CROP, x0:x0 + CELL_CROP]))
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(idb):
            print(f"  [referans] {i + 1}/{len(idb)} ALL-IDB görüntüsü, {len(rows)} referans ({time.time() - start:.0f} sn)", flush=True)
    refs = pd.DataFrame(rows)
    refs.insert(0, "ref_id", range(len(refs)))
    return refs, hashes


@lru_cache(maxsize=16)
def cached_gray(path: str) -> np.ndarray:
    """
    ALL-IDB görüntüsünü gri okur; son 16 görüntü önbellekte tutulur (aynı referansa art arda gelen adaylar için).

    Parametreler:
        path: Dosya yolu.

    Döndürür:
        uint8 gri görüntü (önbellekteki nesne; değiştirilmemelidir).
    """
    return gray_of(path)


def reference_gray(ref: pd.Series, idb: pd.DataFrame) -> np.ndarray:
    """
    Bir referansın karşılaştırmada kullanılan gri görüntüsünü üretir.

    Parametreler:
        ref: Referans satırı.
        idb: list_allidb çıktısı (dosya yolu için).

    Döndürür:
        tam → orijinal; sikistirilmis → 256x256; idb2 → orijinal; hucre → 257x257 kırpım.
    """
    gray = cached_gray(idb.set_index("idb_dosya").at[ref["idb_dosya"], "idb_yolu"])
    if ref["tur"] == "sikistirilmis":
        return cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    if ref["tur"] == "hucre":
        x0, y0 = int(ref["kirpim_x"]), int(ref["kirpim_y"])
        return gray[y0:y0 + CELL_CROP, x0:x0 + CELL_CROP]
    return gray


# ---------------------------------------------------------------------------
# Hash adayları
# ---------------------------------------------------------------------------

def kaggle_hashes(kaggle: pd.DataFrame) -> np.ndarray:
    """
    Kaggle görüntülerinin döndürmeye dayanıklı pHash'lerini hesaplar (önbellekten okur/önbelleğe yazar).

    Parametreler:
        kaggle: load_kaggle çıktısı.

    Döndürür:
        (N, 72) uint64 hash matrisi.
    """
    cache = CACHE_DIR / "kaggle_hash.npy"
    if cache.exists():
        h = np.load(cache)
        if len(h) == len(kaggle):
            return h
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out, start = [], time.time()
    for i, p in enumerate(kaggle["dosya_yolu"]):
        out.append(rotation_phashes(gray_of(p)))
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(kaggle):
            print(f"  [Kaggle hash] {i + 1}/{len(kaggle)} ({time.time() - start:.0f} sn)", flush=True)
    h = np.stack(out)
    np.save(cache, h)
    return h


def cross_distances(hk: np.ndarray, hr: np.ndarray, block: int = 100) -> np.ndarray:
    """
    Kaggle × referans döndürmeye dayanıklı pHash mesafe matrisini hesaplar.

    d(k, r) = min(k'nin temel hash'i ↔ r'nin 72 varyantı, r'nin temel hash'i ↔ k'nin 72 varyantı).

    Parametreler:
        hk: (N, 72) Kaggle hash'leri.
        hr: (M, 72) referans hash'leri.
        block: Aynı anda işlenen Kaggle satırı (bellek sınırı).

    Döndürür:
        (N, M) uint8 mesafe matrisi.
    """
    dist = np.full((len(hk), len(hr)), 64, np.uint8)
    for lo in range(0, len(hk), block):
        k = hk[lo:lo + block]
        d1 = np.bitwise_count(k[:, 0][:, None, None] ^ hr[None, :, :]).min(axis=2)
        d2 = np.bitwise_count(hr[:, 0][None, :, None] ^ k[:, None, :]).min(axis=2)
        dist[lo:lo + block] = np.minimum(d1, d2)
    return dist


def hash_candidates(dist: np.ndarray, refs: pd.DataFrame, types: pd.Series) -> pd.DataFrame:
    """
    Her Kaggle görüntüsü ve referans türü için mesafesi ≤ HASH_CANDIDATE_DISTANCE olan en yakın adayları seçer.

    Yalnızca görüntü tipine uygun referans türleri (TYPE_REFERENCES) kullanılır.

    Parametreler:
        dist: cross_distances çıktısı.
        refs: Referans tablosu.
        types: Kaggle görüntü tipleri (dist satır sırasında).

    Döndürür:
        k (Kaggle satırı), ref_id, hash_mesafesi sütunlu aday tablosu.
    """
    rows = []
    for tur, sub in refs.groupby("tur"):
        cols = sub["ref_id"].to_numpy()
        d = dist[:, cols]
        top = np.argsort(d, axis=1)[:, :CANDIDATES_PER_SET]
        allowed = types.map(lambda t: tur in TYPE_REFERENCES[t]).to_numpy()
        for k in np.where(allowed)[0]:
            for j in top[k]:
                if d[k, j] <= HASH_CANDIDATE_DISTANCE:
                    rows.append((k, int(cols[j]), int(d[k, j])))
    return pd.DataFrame(rows, columns=["k", "ref_id", "hash_mesafesi"])


# ---------------------------------------------------------------------------
# Doğrulama
# ---------------------------------------------------------------------------

def orb_of(gray: np.ndarray, side: int = ORB_SIDE, n: int = ORB_FEATURES, mirror: bool = False) -> tuple:
    """
    dataset.orb_features ile aynı ORB çıkarımını bellekteki gri görüntüye uygular (dolgu maskeli).

    Parametreler:
        gray: Gri görüntü.
        side: Uzun kenar sınırı (büyükse küçültülür).
        n: Nokta sayısı.
        mirror: Yatay aynala.

    Döndürür:
        (noktalar (K,2) float32, tanımlayıcılar veya None).
    """
    if mirror:
        gray = np.ascontiguousarray(gray[:, ::-1])
    if max(gray.shape) > side:
        gray = resize_long(gray, side)
    mask = cv2.erode(((gray > BLACK_PIXEL_THRESHOLD) * 255).astype(np.uint8), np.ones((7, 7), np.uint8))
    kps, desc = cv2.ORB_create(nfeatures=n).detectAndCompute(gray, mask)
    return np.float32([kp.pt for kp in kps]), desc


def square256(gray: np.ndarray) -> np.ndarray:
    """
    Görüntüyü 256x256 kareye getirir (yakın çekimler zaten ~256 px kare).

    Parametreler:
        gray: Gri görüntü.

    Döndürür:
        256x256 gri görüntü.
    """
    return cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)


def rotation_variants(gray256: np.ndarray) -> list[np.ndarray]:
    """
    256x256 görüntünün 10° adımlı 36 döndürmesini, aynasız ve aynalı (72 varyant) üretir.

    Parametreler:
        gray256: 256x256 gri görüntü.

    Döndürür:
        72 varyant listesi.
    """
    out = []
    for base in (gray256, np.ascontiguousarray(gray256[:, ::-1])):
        for angle in range(0, 360, 10):
            out.append(cv2.warpAffine(base, cv2.getRotationMatrix2D((128, 128), angle, 1.0), (256, 256)))
    return out


def context_ncc(kaggle_gray: np.ndarray, ref_gray: np.ndarray) -> float:
    """
    Geniş bağlamlı şablon eşleştirmesi: Kaggle görüntüsünün merkez 180 px'i, referansın 72 varyantında aranır.

    Parametreler:
        kaggle_gray: Kaggle gri görüntüsü.
        ref_gray: Referans (hücre kırpımı) gri görüntüsü.

    Döndürür:
        En yüksek normalize çapraz korelasyon (TM_CCOEFF_NORMED; doğrusal parlaklık/kontrast farkına duyarsız).
    """
    s = square256(kaggle_gray)
    lo = (256 - NCC_REGION) // 2
    region = s[lo:lo + NCC_REGION, lo:lo + NCC_REGION]
    variants = rotation_variants(square256(ref_gray))
    best = -1.0
    for f in NCC_SCALES:
        tpl = cv2.resize(region, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        for v in variants:
            best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(v, tpl, cv2.TM_CCOEFF_NORMED))[1]))
            if best >= NCC_EARLY_STOP:
                return best
    return best


def verify_chunk(cands: pd.DataFrame, paths: dict, refs: pd.DataFrame, idb: pd.DataFrame) -> list[dict]:
    """
    Bir parça Kaggle görüntüsünün hash adaylarını doğrular (paralel işçide çalışır).

    Tam/sıkıştırılmış referanslarda ORB/RANSAC (Adım 1 kuralı: dataset.is_near_duplicate), hücre referanslarında
    geniş bağlamlı NCC. Bir görüntüde NCC >= NCC_EARLY_STOP olan hücre eşleşmesi bulunduysa kalan hücre adayları
    atlanır (aynı görüntü zaten bulundu). Adaylar hash mesafesine göre artan sırada denenir.

    Parametreler:
        cands: Bu parçanın adayları (k, ref_id, hash_mesafesi).
        paths: {k: Kaggle dosya yolu}.
        refs: Referans tablosu.
        idb: list_allidb çıktısı.

    Döndürür:
        Aday başına sonuç sözlükleri (orb_inlier, ncc, kabul, yontem eklenmiş).
    """
    out = []
    ref_by_id = refs.set_index("ref_id")
    for k, sub in cands.groupby("k"):
        kg = gray_of(paths[k])
        k_orb, done = None, False
        for _, c in sub.sort_values("hash_mesafesi").iterrows():
            ref = ref_by_id.loc[c["ref_id"]]
            inl, ncc = 0, np.nan
            if ref["tur"] in ("tam", "sikistirilmis"):
                rg = reference_gray(ref, idb)
                k_orb = k_orb or orb_of(kg)
                inl = orb_inliers(k_orb, orb_of(rg))
                if not is_near_duplicate(c["hash_mesafesi"], inl):
                    inl = max(inl, orb_inliers(k_orb, orb_of(rg, mirror=True)))
                ok, method = is_near_duplicate(c["hash_mesafesi"], inl), "phash+orb"
            elif done:
                continue
            else:
                ncc = context_ncc(kg, reference_gray(ref, idb))
                ok, method = ncc >= NCC_MIN, "phash+ncc"
                done = ncc >= NCC_EARLY_STOP
            out.append({**c.to_dict(), "orb_inlier": inl, "ncc": ncc, "kabul": bool(ok), "yontem": method})
    return out


def verify_candidates(cands: pd.DataFrame, kaggle: pd.DataFrame, refs: pd.DataFrame, idb: pd.DataFrame) -> pd.DataFrame:
    """
    Hash adaylarını 100 Kaggle görüntülük parçalar hâlinde, en fazla N_JOBS işçiyle paralel doğrular.

    Parametreler:
        cands: hash_candidates çıktısı.
        kaggle: load_kaggle çıktısı.
        refs: Referans tablosu.
        idb: list_allidb çıktısı.

    Döndürür:
        cands + orb_inlier, ncc, kabul, yontem sütunları.
    """
    ks = sorted(cands["k"].unique())
    parts = [cands[cands["k"].isin(ks[i:i + CHUNK])] for i in range(0, len(ks), CHUNK)]
    paths = kaggle["dosya_yolu"].to_dict()
    out, start, done = [], time.time(), 0
    parallel = Parallel(n_jobs=N_JOBS, return_as="generator")
    for part, res in zip(parts, parallel(delayed(verify_chunk)(p, paths, refs, idb) for p in parts)):
        out.extend(res)
        done += part["k"].nunique()
        print(f"  [doğrulama] {done}/{len(ks)} Kaggle görüntüsü ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Kırpım arama (ALL-IDB1)
# ---------------------------------------------------------------------------

def build_crop_db(idb1: pd.DataFrame) -> tuple:
    """
    ALL-IDB1 için ORB veritabanı (uzun kenar 512, aynalı dahil) ve şablon taraması için küçük gri görüntüler hazırlar.

    Parametreler:
        idb1: ALL-IDB1 satırları.

    Döndürür:
        (BFMatcher (eğitilmiş koleksiyon), nokta listesi, sahip listesi [(dosya, aynalı)], {dosya: 320 px gri}).
    """
    matcher, pts, owners, small = cv2.BFMatcher(cv2.NORM_HAMMING), [], [], {}
    for r in idb1.itertuples(index=False):
        gray = gray_of(r.idb_yolu)
        small[r.idb_dosya] = resize_long(gray, TEMPLATE_SIDE)
        g = resize_long(gray, CROP_DB_SIDE)
        for mirror in (False, True):
            p, d = orb_of(g, side=CROP_DB_SIDE, n=CROP_DB_FEATURES, mirror=mirror)
            matcher.add([d if d is not None else np.zeros((0, 32), np.uint8)])
            pts.append(p)
            owners.append((r.idb_dosya, mirror))
    return matcher, pts, owners, small


def orb_crop_search(kg: np.ndarray, matcher, pts: list, owners: list) -> tuple[str, bool, int]:
    """
    Kaggle görüntüsünü ALL-IDB1 ORB veritabanında arar (oylama + RANSAC benzerlik dönüşümü).

    Parametreler:
        kg: Kaggle gri görüntüsü.
        matcher, pts, owners: build_crop_db çıktısı.

    Döndürür:
        (en iyi ALL-IDB1 dosyası, aynalı mı, inlier sayısı); eşleşme yoksa ("", False, 0).
    """
    q_pts, q_desc = orb_of(kg)
    if q_desc is None or len(q_desc) < 12:
        return "", False, 0
    matches = matcher.knnMatch(q_desc, k=2)
    good = [m for m, n in (x for x in matches if len(x) == 2) if m.distance < ORB_RATIO * n.distance]
    if not good:
        return "", False, 0
    owner = np.array([m.imgIdx for m in good])
    best = ("", False, 0)
    for e in pd.Series(owner).value_counts().index[:3]:  # En çok oy alan 3 görüntü RANSAC'a girer
        sel = [m for m, o in zip(good, owner) if o == e]
        if len(sel) < 12:
            continue
        model, inl = cv2.estimateAffinePartial2D(q_pts[[m.queryIdx for m in sel]], pts[e][[m.trainIdx for m in sel]],
                                                 method=cv2.RANSAC, ransacReprojThreshold=4)
        n_inl = 0 if model is None else int(inl.sum())
        if n_inl > best[2]:
            best = (owners[e][0], owners[e][1], n_inl)
    return best


def template_crop_search(kg: np.ndarray, small: dict) -> tuple[str, bool, float]:
    """
    Kaggle görüntüsünün merkez %70'ini küçültülmüş ALL-IDB1 görüntülerinde çok ölçekli şablon olarak arar.

    Parametreler:
        kg: Kaggle gri görüntüsü.
        small: {ALL-IDB1 dosyası: 320 px gri görüntü}.

    Döndürür:
        (en iyi dosya, aynalı mı, en yüksek NCC).
    """
    h, w = kg.shape
    region = kg[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)]
    best = ("", False, -1.0)
    for mirror in (False, True):
        r = np.ascontiguousarray(region[:, ::-1]) if mirror else region
        for width in TEMPLATE_WIDTHS:
            tpl = cv2.resize(r, (width, max(8, round(width * r.shape[0] / r.shape[1]))), interpolation=cv2.INTER_AREA)
            for name, img in small.items():
                if tpl.shape[0] >= img.shape[0] or tpl.shape[1] >= img.shape[1]:
                    continue
                score = float(cv2.minMaxLoc(cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED))[1])
                if score > best[2]:
                    best = (name, mirror, score)
    return best


def crop_chunk(items: list[tuple[int, str]], idb1: pd.DataFrame) -> list[dict]:
    """
    Bir parça Kaggle görüntüsünde ALL-IDB1 kırpımı araması yapar (paralel işçide; veritabanı işçide kurulur,
    çünkü OpenCV eşleştiricisi işlemler arasında taşınamaz).

    Parametreler:
        items: [(k, Kaggle dosya yolu)] listesi.
        idb1: ALL-IDB1 satırları.

    Döndürür:
        Görüntü başına k, orb_dosya, orb_ayna, orb_inlier, sablon_dosya, sablon_ayna, sablon_ncc sözlükleri.
    """
    matcher, pts, owners, small = build_crop_db(idb1)
    out = []
    for k, path in items:
        kg = gray_of(path)
        of, om, oi = orb_crop_search(kg, matcher, pts, owners)
        tf, tm, ts = template_crop_search(kg, small)
        out.append({"k": k, "orb_dosya": of, "orb_ayna": om, "orb_inlier": oi,
                    "sablon_dosya": tf, "sablon_ayna": tm, "sablon_ncc": ts})
    return out


def crop_search(kaggle: pd.DataFrame, rows: list[int], idb: pd.DataFrame) -> pd.DataFrame:
    """
    Verilen Kaggle görüntülerinde ALL-IDB1 kırpımı araması yapar (ORB oylaması + şablon taraması), paralel.

    Parametreler:
        kaggle: load_kaggle çıktısı.
        rows: Aranacak Kaggle satır indeksleri.
        idb: list_allidb çıktısı.

    Döndürür:
        k, orb_dosya, orb_ayna, orb_inlier, sablon_dosya, sablon_ayna, sablon_ncc sütunlu tablo.
    """
    items = [(k, kaggle.at[k, "dosya_yolu"]) for k in rows]
    parts = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    idb1 = idb[idb["idb_set"] == "IDB1"]
    out, start, done = [], time.time(), 0
    parallel = Parallel(n_jobs=N_JOBS, return_as="generator")
    for part, res in zip(parts, parallel(delayed(crop_chunk)(p, idb1) for p in parts)):
        out.extend(res)
        done += len(part)
        print(f"  [kırpım arama] {done}/{len(items)} ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# ALL-IDB içi bağlantılar
# ---------------------------------------------------------------------------

def idb1_internal_groups(idb: pd.DataFrame, refs: pd.DataFrame, hashes: list) -> pd.Series:
    """
    ALL-IDB1 görüntülerinin kendi içindeki kopyalarını Adım 1 yöntemiyle (pHash adayı + ORB) gruplar.

    Parametreler:
        idb: list_allidb çıktısı.
        refs: Referans tablosu.
        hashes: Referans hash listesi.

    Döndürür:
        {ALL-IDB1 dosyası: grup no} serisi.
    """
    full = refs[refs["tur"] == "tam"].reset_index(drop=True)
    h = np.stack([hashes[i] for i in full["ref_id"]])
    dist = cross_distances(h, h)
    np.fill_diagonal(dist, 64)
    paths = idb.set_index("idb_dosya").loc[full["idb_dosya"], "idb_yolu"].tolist()
    pairs = []
    for i, j in zip(*np.where(np.triu(dist <= HASH_CANDIDATE_DISTANCE, k=1))):
        a, b = orb_of(gray_of(paths[i])), gray_of(paths[j])
        inl = max(orb_inliers(a, orb_of(b)), orb_inliers(a, orb_of(b, mirror=True)))
        if is_near_duplicate(int(dist[i, j]), inl):
            pairs.append((int(i), int(j)))
    return pd.Series(assign_groups(len(full), pairs), index=full["idb_dosya"])


def idb2_sources(idb: pd.DataFrame) -> pd.DataFrame:
    """
    Her ALL-IDB2 kırpımının kaynak ALL-IDB1 yaymasını (veya yaymalarını) şablon eşleştirmeyle bulur.

    Parametreler:
        idb: list_allidb çıktısı.

    Döndürür:
        idb2_dosya, kaynak_idb1 (";" ile ayrılmış, NCC ≥ SOURCE_MIN_NCC olanlar), en_iyi_ncc sütunlu tablo.
    """
    small = {r.idb_dosya: cv2.resize(gray_of(r.idb_yolu), None, fx=SOURCE_SCALE, fy=SOURCE_SCALE,
                                     interpolation=cv2.INTER_AREA)
             for r in idb[idb["idb_set"] == "IDB1"].itertuples(index=False)}
    rows, start = [], time.time()
    idb2 = idb[idb["idb_set"] == "IDB2"]
    for n, r in enumerate(idb2.itertuples(index=False)):
        tpl = cv2.resize(gray_of(r.idb_yolu), None, fx=SOURCE_SCALE, fy=SOURCE_SCALE, interpolation=cv2.INTER_AREA)
        scores = {name: float(cv2.minMaxLoc(cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED))[1])
                  for name, img in small.items()}
        hits = sorted((s, name) for name, s in scores.items() if s >= SOURCE_MIN_NCC)
        rows.append({"idb2_dosya": r.idb_dosya, "kaynak_idb1": ";".join(name for _, name in hits[::-1]),
                     "en_iyi_ncc": max(scores.values())})
        if (n + 1) % PROGRESS_EVERY == 0 or n + 1 == len(idb2):
            print(f"  [ALL-IDB2 kaynak] {n + 1}/{len(idb2)} ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------

def build_match_table(verified: pd.DataFrame, crops: pd.DataFrame, kaggle: pd.DataFrame, refs: pd.DataFrame,
                      idb: pd.DataFrame) -> pd.DataFrame:
    """
    Kabul edilen tüm eşleşmeleri (Kaggle görüntüsü ↔ ALL-IDB görüntüsü) tek tabloda toplar.

    Parametreler:
        verified: verify_candidates çıktısı.
        crops: crop_search çıktısı.
        kaggle: load_kaggle çıktısı.
        refs: Referans tablosu.
        idb: list_allidb çıktısı.

    Döndürür:
        Eşleşme tablosu (her satır bir Kaggle görüntüsü–ALL-IDB görüntüsü–yöntem üçlüsü).
    """
    acc = verified[verified["kabul"]].merge(refs[["ref_id", "tur", "idb_dosya", "kirpim_x", "kirpim_y"]], on="ref_id")
    acc = acc.rename(columns={"tur": "referans_turu"})
    parts = [acc[["k", "idb_dosya", "referans_turu", "yontem", "hash_mesafesi", "orb_inlier", "ncc"]]]
    if len(crops):
        o = crops[crops["orb_inlier"] >= ORB_MIN_INLIERS_FAR]
        parts.append(pd.DataFrame({"k": o["k"], "idb_dosya": o["orb_dosya"], "referans_turu": "kirpim",
                                   "yontem": "orb_kirpim", "hash_mesafesi": np.nan, "orb_inlier": o["orb_inlier"],
                                   "ncc": np.nan}))
        t = crops[crops["sablon_ncc"] >= TEMPLATE_MIN_NCC]
        parts.append(pd.DataFrame({"k": t["k"], "idb_dosya": t["sablon_dosya"], "referans_turu": "kirpim",
                                   "yontem": "sablon", "hash_mesafesi": np.nan, "orb_inlier": np.nan,
                                   "ncc": t["sablon_ncc"]}))
    m = pd.concat(parts, ignore_index=True)
    m["kabul"] = (m["yontem"] != "sablon") | TEMPLATE_ACCEPTED
    k = kaggle.loc[m["k"]].reset_index(drop=True)
    meta = idb.set_index("idb_dosya").loc[m["idb_dosya"]].reset_index()
    out = pd.DataFrame({
        "kaggle_dosya": k["dosya_yolu"].map(lambda p: Path(p).name), "kaggle_etiket": k["etiket"],
        "grup_id": k["grup_id"], "bolum": k["bolum"], "tip": k["tip"],
        "kaggle_boyut": k["genislik"].astype(str) + "x" + k["yukseklik"].astype(str),
        "idb_dosya": m["idb_dosya"], "idb_set": meta["idb_set"], "idb_etiket": meta["idb_etiket"],
        "referans_turu": m["referans_turu"], "yontem": m["yontem"], "hash_mesafesi": m["hash_mesafesi"],
        "orb_inlier": m["orb_inlier"], "ncc": m["ncc"].round(4), "kabul": m["kabul"].to_numpy(),
    })
    out["etiket_uyumu"] = out["kaggle_etiket"] == out["idb_etiket"]
    out["kaggle_yolu"] = k["dosya_yolu"]
    return out.sort_values(["kaggle_etiket", "grup_id", "kaggle_dosya"]).reset_index(drop=True)


def summarize_by_class(kaggle: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle'ın her sınıfında doğrudan ve grup yayılımıyla eşleşen görüntü/grup sayılarını özetler.

    Parametreler:
        kaggle: load_kaggle çıktısı.
        matches: build_match_table çıktısı.

    Döndürür:
        Sınıf başına özet tablo.
    """
    direct = set(matches["kaggle_yolu"])
    groups = set(matches["grup_id"])
    k = kaggle.assign(dogrudan=kaggle["dosya_yolu"].isin(direct), grupla=kaggle["grup_id"].isin(groups))
    rows = []
    for cls, s in k.groupby("etiket"):
        g = s.groupby("grup_id").agg(eslesen=("grupla", "first"), bolum=("bolum", "first"))
        rows.append({"sinif": cls, "goruntu": len(s), "dogrudan_eslesen_goruntu": int(s["dogrudan"].sum()),
                     "grupla_eslesen_goruntu": int(s["grupla"].sum()), "grup": len(g),
                     "eslesen_grup": int(g["eslesen"].sum()),
                     "eslesen_grup_gelistirme": int((g["eslesen"] & (g["bolum"] == "gelistirme")).sum()),
                     "eslesen_grup_test": int((g["eslesen"] & (g["bolum"] == "test")).sum())})
    out = pd.DataFrame(rows)
    out["grup_orani"] = (out["eslesen_grup"] / out["grup"]).round(3)
    out["goruntu_orani_grupla"] = (out["grupla_eslesen_goruntu"] / out["goruntu"]).round(3)
    return out


def idb_status(idb: pd.DataFrame, matches: pd.DataFrame, internal: pd.Series, sources: pd.DataFrame) -> pd.DataFrame:
    """
    Her ALL-IDB görüntüsünün Kaggle'da görülüp görülmediğini ve dış doğrulamada kullanılabilirliğini belirler.

    - ALL-IDB1: kendisi, iç kopyası veya ondan kesilmiş bir ALL-IDB2/hücre kırpımı bir Kaggle görüntüsüyle eşleştiyse
      "görüldü"; hangi bölümde (geliştirme/test) görüldüğü yazılır.
    - ALL-IDB2: kendisi eşleştiyse "doğrudan görüldü"; kaynak ALL-IDB1 yayması görüldüyse "kaynak yayma görüldü"
      (hücre eğitimde düşük çözünürlükte, yaymanın içinde görülmüş olabilir). İkisi de yoksa güvenli.

    Parametreler:
        idb: list_allidb çıktısı.
        matches: build_match_table çıktısı.
        internal: idb1_internal_groups çıktısı.
        sources: idb2_sources çıktısı.

    Döndürür:
        idb_dosya, idb_set, idb_etiket, durum, gorulen_bolumler, eslesen_kaggle_grup sayısı, kaynak_idb1, dis_dogrulama_guvenli.
    """
    src = sources.set_index("idb2_dosya")["kaynak_idb1"].str.split(";").to_dict()
    by_idb = matches.groupby("idb_dosya").agg(bolumler=("bolum", lambda s: set(s)), gruplar=("grup_id", lambda s: set(s)))
    seen_sets = {f: (r["bolumler"], r["gruplar"]) for f, r in by_idb.iterrows()}
    # ALL-IDB1 görülme: doğrudan + IDB2 kırpımları üzerinden + iç kopyalar üzerinden
    idb1_seen: dict[str, tuple[set, set]] = {}
    for f in internal.index:
        idb1_seen[f] = tuple(set(x) for x in seen_sets.get(f, (set(), set())))
    for f2, parents in src.items():
        if f2 in seen_sets:
            for p in filter(None, parents):
                idb1_seen[p][0].update(seen_sets[f2][0])
                idb1_seen[p][1].update(seen_sets[f2][1])
    for g, members in internal.groupby(internal).groups.items():
        b = set().union(*(idb1_seen[m][0] for m in members))
        gr = set().union(*(idb1_seen[m][1] for m in members))
        for m in members:
            idb1_seen[m] = (set(b), set(gr))
    rows = []
    for r in idb.itertuples(index=False):
        if r.idb_set == "IDB1":
            b, gr = idb1_seen[r.idb_dosya]
            durum = "goruldu" if b else "gorulmedi"
            parents = ""
        else:
            b, gr = (set(x) for x in seen_sets.get(r.idb_dosya, (set(), set())))
            parents_list = [p for p in src.get(r.idb_dosya, []) if p]
            parents = ";".join(parents_list)
            parent_b = set().union(*(idb1_seen[p][0] for p in parents_list)) if parents_list else set()
            durum = "dogrudan_goruldu" if b else ("kaynak_yayma_goruldu" if parent_b else "gorulmedi")
            b = b | parent_b
        rows.append({"idb_dosya": r.idb_dosya, "idb_set": r.idb_set, "idb_etiket": r.idb_etiket, "durum": durum,
                     "gorulen_bolumler": ";".join(sorted(b)), "eslesen_kaggle_grup": len(gr),
                     "kaynak_idb1": parents, "idb1_ic_grup": int(internal.get(r.idb_dosya, -1)),
                     "dis_dogrulama_guvenli": durum == "gorulmedi"})
    return pd.DataFrame(rows)


def plot_examples(matches: pd.DataFrame, idb: pd.DataFrame, n_per: int = 2) -> Path:
    """
    Yöntem ve referans türü başına birkaç eşleşen Kaggle ↔ ALL-IDB çiftini yan yana çizip data/processed/ altına kaydeder.

    (outputs/ yerine data/processed/: ALL-IDB görüntüleri lisans gereği repoya girmemeli.)

    Parametreler:
        matches: Eşleşme tablosu (kaggle_yolu sütunuyla).
        idb: list_allidb çıktısı.
        n_per: Yöntem ve referans türü başına örnek sayısı (farklı Kaggle gruplarından).

    Döndürür:
        Kaydedilen figürün yolu.
    """
    paths = idb.set_index("idb_dosya")["idb_yolu"]
    unique = matches.drop_duplicates("grup_id")
    picks = pd.concat([sub.sample(min(n_per, len(sub)), random_state=0)
                       for _, sub in unique.groupby(["yontem", "referans_turu"])])
    if picks.empty:
        return FIGURE_PATH
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.8), facecolor=SURFACE, squeeze=False)

    def thumb(p: str) -> np.ndarray:
        img = load_image(p)
        s = 220 / max(img.shape[:2])
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    for c, (_, r) in enumerate(picks.iterrows()):
        cells = [(r["kaggle_yolu"], f"Kaggle {r['kaggle_etiket']}\n{r['kaggle_dosya']}"),
                 (paths[r["idb_dosya"]], f"{r['idb_set']} {r['idb_dosya']}\n{r['yontem']} / {r['referans_turu']}")]
        for row, (img_path, cap) in enumerate(cells):
            ax = axes[row, c]
            ax.imshow(thumb(img_path))
            ax.set_title(cap, fontsize=7, color=TEXT_SECONDARY if row else TEXT_PRIMARY)
            ax.axis("off")
    fig.suptitle("Kaggle ↔ ALL-IDB eşleşme örnekleri (üst: Kaggle, alt: ALL-IDB)", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return FIGURE_PATH


def print_report(kaggle: pd.DataFrame, matches: pd.DataFrame, status: pd.DataFrame) -> None:
    """
    Sınıf başına örtüşme, yöntem dağılımı, etiket uyumu ve ALL-IDB durum özetlerini yazdırır.

    Parametreler:
        kaggle: load_kaggle çıktısı.
        matches: Eşleşme tablosu (kaggle_yolu sütunuyla).
        status: idb_status çıktısı.
    """
    summary = summarize_by_class(kaggle, matches)
    print("\n=== Kaggle sınıf başına ALL-IDB örtüşmesi ===\n" + summary.to_string(index=False))
    print("\nYöntem × referans türü (eşleşme satırı / farklı Kaggle görüntüsü):\n"
          + matches.groupby(["yontem", "referans_turu"]).agg(satir=("kaggle_dosya", "size"),
                                                            goruntu=("kaggle_dosya", "nunique")).to_string())
    img_level = matches.groupby(["kaggle_yolu", "kaggle_etiket"])["idb_etiket"].agg(
        lambda s: "/".join(sorted(set(s)))).reset_index()
    print("\nEtiket uyumu (Kaggle etiketi × eşleşen ALL-IDB etiketi, görüntü düzeyi):\n"
          + pd.crosstab(img_level["kaggle_etiket"], img_level["idb_etiket"]).to_string())
    print("\nEşleşen Kaggle görüntüsü (Kaggle sınıf, tip × ALL-IDB seti):\n"
          + pd.crosstab([matches["kaggle_etiket"], matches["tip"]], matches["idb_set"], values=matches["kaggle_yolu"],
                        aggfunc="nunique").fillna(0).astype(int).to_string())
    print("\n=== ALL-IDB görüntü durumu ===\n"
          + pd.crosstab([status["idb_set"], status["idb_etiket"]], status["durum"]).to_string())
    seen = status[status["durum"] != "gorulmedi"]
    print("Görülen ALL-IDB görüntülerinin Kaggle bölümü:\n"
          + seen.groupby(["idb_set", "gorulen_bolumler"]).size().to_string())


def cached_stage(name: str, func, *args):
    """
    Uzun süren bir aşamanın sonucunu önbellekten okur; yoksa hesaplayıp önbelleğe yazar.

    Çalışma yarıda kalırsa (ör. bilgisayar uykuya geçtiğinde) tamamlanan aşamalar yeniden hesaplanmaz.
    Girdiler (ALL-IDB veya Kaggle indeksi) değişirse CACHE_DIR silinmelidir.

    Parametreler:
        name: Önbellek dosyasının adı (uzantısız).
        func: Aşamayı hesaplayan fonksiyon.
        *args: func argümanları.

    Döndürür:
        Aşamanın sonucu.
    """
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        print(f"   (önbellekten: {path.name})")
        return pd.read_pickle(path)
    result = func(*args)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, path)
    return result


def load_saved_matches(kaggle: pd.DataFrame) -> pd.DataFrame:
    """
    Kaydedilmiş eşleşme CSV'sini okuyup Kaggle dosya yollarını (kaggle_yolu) geri ekler.

    Parametreler:
        kaggle: load_kaggle çıktısı.

    Döndürür:
        Eşleşme tablosu (kaggle_yolu sütunuyla).
    """
    matches = pd.read_csv(MATCH_CSV)
    key = kaggle.assign(kaggle_dosya=kaggle["dosya_yolu"].map(lambda p: Path(p).name))
    key = key[["kaggle_dosya", "etiket", "dosya_yolu"]].rename(columns={"etiket": "kaggle_etiket", "dosya_yolu": "kaggle_yolu"})
    return matches.merge(key, on=["kaggle_dosya", "kaggle_etiket"], how="left")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    """Örtüşmeyi ölçer, eşleşme listesini ve ALL-IDB durum tablosunu kaydeder, özetleri yazdırır."""
    parser = argparse.ArgumentParser(description="Kaggle ↔ ALL-IDB örtüşme analizi")
    parser.add_argument("--allidb-dir", default=str(ALLIDB_DEFAULT_DIR),
                        help="ALL_IDB1 ve ALL_IDB2 klasörlerini içeren dizin (varsayılan config.ALLIDB_DIR; LOSEMI_ALLIDB_DIR ile değiştirilebilir)")
    parser.add_argument("--rapor", action="store_true",
                        help="Yeniden hesaplamadan kayıtlı CSV'lerden özet ve figür üret")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    start = time.time()
    idb = list_allidb(Path(args.allidb_dir))
    kaggle = load_kaggle()
    print(f"ALL-IDB: {idb.groupby(['idb_set', 'idb_etiket']).size().to_dict()}; Kaggle: {len(kaggle)} görüntü")
    if args.rapor:
        matches = load_saved_matches(kaggle)
        if "kabul" not in matches:  # Eski CSV: kabul sütunu yoksa kuralı uygula
            matches["kabul"] = (matches["yontem"] != "sablon") | TEMPLATE_ACCEPTED
            matches.drop(columns="kaggle_yolu").to_csv(MATCH_CSV, index=False, encoding="utf-8")
        accepted = matches[matches["kabul"]]
        refs, ref_hashes = cached_stage("referanslar", build_references, idb)
        internal = cached_stage("idb1_ic_gruplar", idb1_internal_groups, idb, refs, ref_hashes)
        sources = cached_stage("idb2_kaynaklar", idb2_sources, idb)
        status = idb_status(idb, accepted, internal, sources)
        status.to_csv(STATUS_CSV, index=False, encoding="utf-8")
        plot_examples(accepted, idb)
        print_report(kaggle, accepted, status)
        print(f"\nReddedilen (yalnızca şablon) eşleşme: {(~matches['kabul']).sum()}. Figür: {FIGURE_PATH}")
        return

    print("1) Referanslar ve hash'ler...")
    refs, ref_hashes = cached_stage("referanslar", build_references, idb)
    print("   " + refs["tur"].value_counts().to_string().replace("\n", ", "))
    hk = kaggle_hashes(kaggle)
    dist = cross_distances(hk, np.stack(ref_hashes))
    cands = hash_candidates(dist, refs, kaggle["tip"])
    print(f"   hash adayı: {len(cands)} çift, {cands['k'].nunique()} Kaggle görüntüsü")

    print("2) Aday doğrulama (ORB / NCC)...")
    verified = cached_stage("dogrulama", verify_candidates, cands, kaggle, refs, idb)
    v = verified.merge(refs[["ref_id", "tur"]], on="ref_id").merge(
        kaggle[["etiket"]], left_on="k", right_index=True)
    print("   NCC (hücre referansları) dağılımı, sınıf bazında:\n"
          + v[v["tur"].isin(["idb2", "hucre"])].groupby("etiket")["ncc"].describe(percentiles=[.5, .9, .99]).round(3).to_string())

    print("3) Kırpım arama (eşleşmeyen yaymalar)...")
    matched_k = set(verified.loc[verified["kabul"], "k"])
    rows = [k for k in kaggle.index if k not in matched_k and kaggle.at[k, "tip"] == "yayma"]
    crops = cached_stage("kirpim_arama", crop_search, kaggle, rows, idb)
    c = crops.merge(kaggle[["etiket"]], left_on="k", right_index=True)
    print("   Kırpım arama skorları (AML = negatif kontrol):\n"
          + c.groupby("etiket")[["orb_inlier", "sablon_ncc"]].describe(percentiles=[.5, .9, .99]).round(3).T.to_string())

    print("4) ALL-IDB içi bağlantılar...")
    internal = cached_stage("idb1_ic_gruplar", idb1_internal_groups, idb, refs, ref_hashes)
    sources = cached_stage("idb2_kaynaklar", idb2_sources, idb)
    print(f"   ALL-IDB1 iç kopya grubu: {internal.nunique()} grup / {len(internal)} görüntü; "
          f"kaynağı bulunamayan ALL-IDB2: {(sources['kaynak_idb1'] == '').sum()}")

    matches = build_match_table(verified, crops, kaggle, refs, idb)
    matches.drop(columns="kaggle_yolu").to_csv(MATCH_CSV, index=False, encoding="utf-8")
    accepted = matches[matches["kabul"]]
    status = idb_status(idb, accepted, internal, sources)
    status.to_csv(STATUS_CSV, index=False, encoding="utf-8")
    plot_examples(accepted, idb)
    print_report(kaggle, accepted, status)
    print(f"\nKaydedildi: {MATCH_CSV}, {STATUS_CSV}, {FIGURE_PATH}  ({time.time() - start:.0f} sn)")


if __name__ == "__main__":
    # Paralel işçiler fonksiyonları modül adıyla bulur; "python -m" ile __main__ altında tanımlı kalmasınlar
    from src import allidb_overlap
    allidb_overlap.main()
