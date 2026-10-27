"""
Veri seti indeksleme ve analizi.

- Görüntü yollarını ve etiketlerini DataFrame olarak toplar.
- Temel özellikleri (format, çözünürlük, kanal, siyah dolgu oranı) çıkarır.
- Birebir (MD5) ve neredeyse aynı (döndürmeye/aynalamaya dayanıklı pHash adayları +
  ORB/RANSAC geometrik doğrulama) kopyaları bulur, benzer görüntüleri gruplayıp her
  görüntüye bir `grup_id` verir. İleride train/test ayrımı bu gruplara göre yapılacaktır
  (veri sızıntısını önlemek için).
- Görüntü tipini (çok hücreli yayma / tek hücreye yakın çekim) tahmin eder.

Bellek dostu tasarım: görüntüler TEK TEK okunur; bellekte yalnızca hash'ler ve sayısal
özellikler tutulur (1800 görüntü için ~1 MB). ORB tanımlayıcıları yalnızca aday çiftler için
hesaplanır ve küçük bir önbellekte (son 64 görüntü) tutulur.

Bu modül HİÇBİR görüntüyü silmez veya değiştirmez; yalnızca listeler.

Çalıştırma (proje kökünden):
    python -m src.dataset
"""

import hashlib
import io
import random
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    CLASS_NAMES,
    DATASET_INDEX_CSV,
    DUPLICATE_PAIRS_CSV,
    RANDOM_STATE,
    RAW_DATA_DIR,
)
from src.utils import load_image, save_figure

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Uzun döngülerde kaç görüntüde bir ilerleme yazdırılacağı
PROGRESS_EVERY = 100

# Siyah dolgu eşiği: augmentasyonla döndürülen görüntülerin köşeleri saf siyah (0,0,0)
# ile doldurulmuş. JPEG sıkıştırması bu pikselleri 0-10 aralığına dağıttığı için 15
# kullanılır; boyanmış kan yaymasında bu kadar koyu (her kanal < 15) gerçek doku yoktur.
BLACK_PIXEL_THRESHOLD = 15

# --- Döndürmeye dayanıklı pHash ---
# Görüntünün merkez karesi 256x256'ya ölçeklenir, 0°-350° arası 10° adımla döndürülür
# (aynalı hâli dahil 72 varyant) ve her varyantın MERKEZ 180x180 bölgesinin pHash'i alınır.
# 180 ≈ 256/√2: merkez etrafında hangi açıyla döndürülürse döndürülsün bu bölge daima
# görüntünün içinde kalır, yani siyah dolgu köşeleri hash'e girmez.
# 10° adım: gerçek döndürme açısıyla en yakın varyant arasında en fazla 5° fark kalır;
# pHash'in düşük frekanslı DCT katsayıları bu kadar farka dayanıklıdır.
HASH_SIZE = 256
HASH_CROP = 180
HASH_ANGLE_STEP = 10
# Aday eşiği: 64 bitlik hash'te Hamming mesafesi <= 14 olan çiftler ORB doğrulamasına gider.
# Farklı görüntü çiftlerinin dağılımı 20-24 civarında tepe yapar; 14 bu kütlenin dışında
# kalırken döndürülmüş/kırpılmış augmentasyonların büyük kısmını aday listesine alır.
HASH_CANDIDATE_DISTANCE = 14
# Not: pHash TEK BAŞINA güvenilir değildir. Healthy yaymaları (tekdüze eritrosit dokusu)
# birbirine çok benzediği için mesafe <= 4 eşiğinde bile zincirleme eşleşmeyle yüzlerce
# farklı görüntü tek gruba düşüyordu. Bu yüzden her aday ORB ile doğrulanır.

# --- ORB + RANSAC doğrulama ---
ORB_MAX_SIDE = 384      # Hız için uzun kenar 384 px'e küçültülür; hücre detayı korunur
ORB_N_FEATURES = 500    # Görüntü başına 500 nokta: eşleşme kararı için yeterli, hızlı
ORB_RATIO = 0.75        # Lowe oran testi: belirsiz eşleşmeleri eler (literatür standardı)
ORB_MIN_GOOD = 12       # RANSAC'a girmek için gereken minimum iyi eşleşme sayısı
ORB_CACHE_SIZE = 64     # Önbellekte tutulan (görüntü, aynalı mı) özellik sayısı (~1.3 MB)
# Kabul kuralı (aday çiftlerin görsel incelemesiyle belirlendi):
# - hash mesafesi <= 10 ve inlier >= 15: incelenen tüm örnekler aynı alanın döndürülmüş /
#   renk kaydırılmış / yakınlaştırılmış hâliydi.
# - hash mesafesi 12-14 aralığında kırmızı kan hücrelerinin tekrarlayan dokusu 25 inlier'e
#   kadar yanlış eşleşme üretti (ör. ALL yayması ile Healthy yayması). Bu bölgede >= 30 istenir.
ORB_MIN_INLIERS_CLOSE = 15
ORB_CLOSE_HASH_DISTANCE = 10
ORB_MIN_INLIERS_FAR = 30

# Görüntü tipi tespiti (çekirdek segmentasyonu tabanlı) eşikleri:
# Lökosit çekirdekleri (mor) B kanalında yüksek, G kanalında düşüktür; eritrositler ve
# arka planda B≈G olur. Bu yüzden B-G farkı çekirdekleri ayırmak için basit bir indekstir.
NUCLEUS_MIN_INDEX = 40       # Otsu eşiği çok düşük çıkarsa (çekirdek yoksa) taban değer
TYPE_MAX_SIDE = 512          # Tip tespiti için yeterli çözünürlük, hızlı işlem
MIN_NUCLEUS_FRACTION = 0.0005  # Görüntü alanının %0.05'inden küçük lekeler (trombosit, gürültü) sayılmaz
# Yakın çekim kararı: en büyük çekirdek alanın >= %3'ü (yaymada tek çekirdek ~%0.1-0.5),
# toplam çekirdek alanının >= %60'ı tek nesnede (tek hücre baskın) ve solidity >= 0.85
# (birbirine yapışmış çok hücreli kümelerin girintili, düşük solidity'li sınırını eler).
CLOSEUP_MIN_LARGEST = 0.03
CLOSEUP_MIN_DOMINANCE = 0.6
CLOSEUP_MIN_SOLIDITY = 0.85

# Grafik renkleri (dataviz referans paletinin ilk üç kategorik slotu; renk körlüğü için doğrulanmış)
CLASS_COLORS = {"ALL": "#2a78d6", "AML": "#eb6834", "Healthy": "#1baf7a"}
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _progress(done: int, total: int, start: float, what: str) -> None:
    """
    Her PROGRESS_EVERY adımda bir (ve sonda) ilerlemeyi ekrana yazdırır.

    Parametreler:
        done: Tamamlanan adım sayısı.
        total: Toplam adım sayısı.
        start: Başlangıç zamanı (time.time()).
        what: Yapılan işin kısa adı.
    """
    if done % PROGRESS_EVERY == 0 or done == total:
        print(f"  [{what}] {done}/{total} ({time.time() - start:.0f} sn)", flush=True)


# ---------------------------------------------------------------------------
# İndeks
# ---------------------------------------------------------------------------

def list_images(raw_dir: Path = RAW_DATA_DIR, class_names: list[str] = CLASS_NAMES) -> pd.DataFrame:
    """
    data/raw/<sınıf>/ klasörlerindeki görüntüleri listeler.

    Parametreler:
        raw_dir: Sınıf klasörlerini içeren ham veri dizini.
        class_names: Okunacak sınıf klasörlerinin adları (aynı zamanda etiketler).

    Döndürür:
        'dosya_yolu' ve 'etiket' sütunlarına sahip DataFrame (sınıf ve dosya adına göre sıralı).
    """
    rows = []
    for label in class_names:
        for path in sorted((raw_dir / label).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"dosya_yolu": str(path), "etiket": label})
    return pd.DataFrame(rows, columns=["dosya_yolu", "etiket"])


# ---------------------------------------------------------------------------
# Tek görüntü ölçümleri (görüntü yalnızca bu fonksiyonların içinde bellekte durur)
# ---------------------------------------------------------------------------

def black_padding_ratio(image: np.ndarray) -> float:
    """
    Görüntüdeki saf siyah (dolgu) piksellerin oranını hesaplar.

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        0-1 arası oran; döndürme augmentasyonu yapılmış görüntülerde > 0 olur.
    """
    return float((image.max(axis=2) < BLACK_PIXEL_THRESHOLD).mean())


def file_header_info(data: bytes) -> dict:
    """
    Dosya baytlarından format, çözünürlük ve kanal bilgisini okur (pikselleri çözmeden).

    Parametreler:
        data: Görüntü dosyasının ham baytları.

    Döndürür:
        format, genislik, yukseklik, kanal, mod anahtarlı sözlük.
    """
    with Image.open(io.BytesIO(data)) as im:
        return {"format": im.format, "genislik": im.size[0], "yukseklik": im.size[1],
                "kanal": len(im.getbands()), "mod": im.mode}


def rotation_phashes(gray: np.ndarray) -> np.ndarray:
    """
    Görüntünün döndürülmüş ve aynalanmış varyantlarının pHash'lerini hesaplar.

    Merkez kare 256x256'ya ölçeklenir; aynasız ve aynalı hâli 10° adımlarla döndürülür ve
    her varyantın merkez 180x180 bölgesinin 64 bitlik pHash'i alınır.

    Parametreler:
        gray: Gri seviye görüntü.

    Döndürür:
        (72,) uint64 dizisi; ilk eleman döndürülmemiş, aynalanmamış görüntünün hash'idir.
    """
    h, w = gray.shape
    s = min(h, w)
    square = gray[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    square = cv2.resize(square, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    c, lo = HASH_SIZE / 2, (HASH_SIZE - HASH_CROP) // 2
    hashes = []
    for base in (square, np.ascontiguousarray(square[:, ::-1])):
        for angle in range(0, 360, HASH_ANGLE_STEP):
            rot = cv2.warpAffine(base, cv2.getRotationMatrix2D((c, c), angle, 1.0), (HASH_SIZE, HASH_SIZE))
            bits = imagehash.phash(Image.fromarray(rot[lo:lo + HASH_CROP, lo:lo + HASH_CROP])).hash
            hashes.append(int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big"))
    return np.array(hashes, dtype=np.uint64)


def nucleus_mask(image: np.ndarray) -> np.ndarray:
    """
    Mor boyanmış lökosit çekirdeklerinin kaba ikili maskesini çıkarır (yalnızca tip tespiti için).

    B-G fark indeksine Otsu eşiği uygulanır; siyah dolgu bölgesi hariç tutulur.

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        uint8 ikili maske (0/1).
    """
    r, g, b = [c.astype(np.int16) for c in cv2.split(image)]
    valid = image.max(axis=2) >= BLACK_PIXEL_THRESHOLD
    index = np.clip(b - g, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(index[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((index > max(otsu, NUCLEUS_MIN_INDEX)) & valid).astype(np.uint8)
    # 3x3 açma: tek piksellik JPEG gürültüsünü siler, çekirdek şeklini bozmaz
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def estimate_image_type(image: np.ndarray) -> dict:
    """
    Görüntünün çok hücreli yayma mı yoksa tek hücreye yakın çekim mi olduğunu tahmin eder.

    Çekirdek bileşenlerinin sayısı ve görüntü alanına göre boyutu (büyütme göstergesi) kullanılır.

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        tip ('yayma' / 'yakin_cekim'), tahmini_cekirdek_sayisi, en_buyuk_cekirdek_orani,
        baskinlik, solidity anahtarlı sözlük.
    """
    scale = TYPE_MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = nucleus_mask(image)
    valid_area = max(int((image.max(axis=2) >= BLACK_PIXEL_THRESHOLD).sum()), 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    fractions = stats[1:, cv2.CC_STAT_AREA] / valid_area
    keep = fractions >= MIN_NUCLEUS_FRACTION
    if not keep.any():
        return {"tip": "yayma", "tahmini_cekirdek_sayisi": 0, "en_buyuk_cekirdek_orani": 0.0,
                "baskinlik": 0.0, "solidity": 0.0}
    j = int(np.argmax(fractions))
    largest = float(fractions[j])
    dominance = largest / float(fractions[keep].sum())
    contours, _ = cv2.findContours((labels == j + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    solidity = cv2.contourArea(contour) / max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    closeup = (largest >= CLOSEUP_MIN_LARGEST and dominance >= CLOSEUP_MIN_DOMINANCE
               and solidity >= CLOSEUP_MIN_SOLIDITY)
    return {"tip": "yakin_cekim" if closeup else "yayma",
            "tahmini_cekirdek_sayisi": int(keep.sum()),
            "en_buyuk_cekirdek_orani": round(largest, 4),
            "baskinlik": round(dominance, 3),
            "solidity": round(float(solidity), 3)}


def scan_image(path: str | Path) -> tuple[dict, np.ndarray]:
    """
    Tek bir görüntüyü bir kez okuyup tüm ölçümleri çıkarır; görüntünün kendisini döndürmez.

    Parametreler:
        path: Görüntü dosyasının yolu.

    Döndürür:
        (özellik sözlüğü [format, çözünürlük, kanal, md5, siyah dolgu, görüntü tipi ...],
         döndürmeye dayanıklı pHash dizisi (72,) uint64).
    """
    data = Path(path).read_bytes()
    info = file_header_info(data)
    info["dosya_boyutu_kb"] = round(len(data) / 1024, 1)
    info["md5"] = hashlib.md5(data).hexdigest()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    info["siyah_dolgu_orani"] = round(black_padding_ratio(rgb), 4)
    info.update(estimate_image_type(rgb))
    hashes = rotation_phashes(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    info["phash"] = f"{int(hashes[0]):016x}"
    return info, hashes


def scan_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Tüm görüntüleri sırayla (tek tek) tarar; yalnızca özellikleri ve hash'leri saklar.

    Parametreler:
        df: 'dosya_yolu' sütunu olan DataFrame.

    Döndürür:
        (özellik sütunları eklenmiş DataFrame, (N, 72) uint64 hash matrisi).
    """
    rows, hashes = [], np.zeros((len(df), 2 * 360 // HASH_ANGLE_STEP), dtype=np.uint64)
    start = time.time()
    for k, path in enumerate(df["dosya_yolu"]):
        info, hashes[k] = scan_image(path)
        rows.append(info)
        _progress(k + 1, len(df), start, "tarama")
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1), hashes


def summarize_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sınıf bazında görüntü sayısı, format, kanal ve çözünürlük özetini çıkarır.

    Parametreler:
        df: scan_dataset çıktısı.

    Döndürür:
        Her satırı bir sınıf olan özet tablo.
    """
    rows = []
    for label, g in df.groupby("etiket", sort=False):
        rows.append({
            "etiket": label,
            "goruntu_sayisi": len(g),
            "formatlar": ", ".join(f"{k}:{v}" for k, v in g["format"].value_counts().items()),
            "kanal": ", ".join(str(k) for k in sorted(g["kanal"].unique())),
            "farkli_cozunurluk": g[["genislik", "yukseklik"]].drop_duplicates().shape[0],
            "en_yaygin_cozunurluk": "{}x{}".format(*g[["genislik", "yukseklik"]].value_counts().index[0]),
            "min_cozunurluk": f"{g['genislik'].min()}x{g['yukseklik'].min()}",
            "max_cozunurluk": f"{g['genislik'].max()}x{g['yukseklik'].max()}",
            "ort_cozunurluk": f"{g['genislik'].mean():.0f}x{g['yukseklik'].mean():.0f}",
            "siyah_dolgulu": int((g["siyah_dolgu_orani"] > 0.01).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kopya tespiti
# ---------------------------------------------------------------------------

def find_exact_duplicate_pairs(md5s: pd.Series) -> list[tuple[int, int]]:
    """
    Aynı MD5 özetine sahip dosya çiftlerini bulur.

    Parametreler:
        md5s: İndeksi DataFrame satırı olan MD5 serisi.

    Döndürür:
        (i, j) indeks çiftleri listesi (i < j).
    """
    pairs = []
    for _, idx in md5s.groupby(md5s).groups.items():
        pairs.extend(combinations(sorted(idx), 2))
    return pairs


def hash_distance_matrix(hashes: np.ndarray, block: int = 200) -> np.ndarray:
    """
    Döndürmeye dayanıklı pHash mesafe matrisini blok blok hesaplar.

    d(i, j) = i'nin döndürülmemiş hash'i ile j'nin 72 varyantı arasındaki en küçük Hamming
    mesafesi; matris simetrik hâle getirilir.

    Parametreler:
        hashes: (N, 72) uint64 hash matrisi.
        block: Aynı anda işlenen satır sayısı (bellek kullanımını sınırlar).

    Döndürür:
        (N, N) uint8 mesafe matrisi (köşegen 64).
    """
    n = len(hashes)
    dist = np.full((n, n), 64, dtype=np.uint8)
    for lo in range(0, n, block):
        base = hashes[lo:lo + block, 0][:, None]
        for k in range(hashes.shape[1]):
            d = np.bitwise_count(base ^ hashes[None, :, k]).astype(np.uint8)
            np.minimum(dist[lo:lo + block], d, out=dist[lo:lo + block])
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 64)
    return dist


def find_hash_candidates(hashes: np.ndarray, max_distance: int = HASH_CANDIDATE_DISTANCE) -> list[tuple[int, int, int]]:
    """
    Hash mesafesi eşiğin altında kalan aday çiftleri bulur.

    Parametreler:
        hashes: (N, 72) uint64 hash matrisi.
        max_distance: Aday sayılacak maksimum Hamming mesafesi.

    Döndürür:
        (i, j, mesafe) listesi (i < j, i'ye göre sıralı).
    """
    dist = hash_distance_matrix(hashes)
    ii, jj = np.where(np.triu(dist <= max_distance, k=1))
    return [(int(i), int(j), int(dist[i, j])) for i, j in zip(ii, jj)]


def orb_features(path: str | Path, mirror: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Görüntüden ORB anahtar nokta konumlarını ve tanımlayıcılarını çıkarır.

    Siyah dolgu bölgesi maskelenir; aksi hâlde dolgu kenarları sahte köşe noktası üretir.

    Parametreler:
        path: Görüntü dosyasının yolu.
        mirror: True ise görüntü yatay aynalanır (ORB aynalamaya karşı değişmez değildir).

    Döndürür:
        (noktalar (K,2) float32, tanımlayıcılar (K,32) uint8 veya None).
    """
    gray = cv2.cvtColor(load_image(path), cv2.COLOR_RGB2GRAY)
    if mirror:
        gray = np.ascontiguousarray(gray[:, ::-1])
    scale = ORB_MAX_SIDE / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = (gray > BLACK_PIXEL_THRESHOLD).astype(np.uint8) * 255
    # 7x7 aşındırma: dolgu sınırındaki keskin geçişte nokta oluşmasını engeller
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    keypoints, desc = cv2.ORB_create(nfeatures=ORB_N_FEATURES).detectAndCompute(gray, mask)
    return np.float32([kp.pt for kp in keypoints]), desc


def orb_inliers(feat_a: tuple, feat_b: tuple) -> int:
    """
    İki görüntü arasındaki geometrik olarak tutarlı ORB eşleşmesi sayısını hesaplar.

    Lowe oran testinden geçen eşleşmelere RANSAC ile benzerlik dönüşümü
    (döndürme + ölçek + öteleme) uydurulur; inlier sayısı döndürülür.

    Parametreler:
        feat_a, feat_b: orb_features çıktıları.

    Döndürür:
        RANSAC inlier sayısı (eşleşme yoksa 0).
    """
    (pts_a, desc_a), (pts_b, desc_b) = feat_a, feat_b
    if desc_a is None or desc_b is None or len(desc_a) < ORB_MIN_GOOD or len(desc_b) < ORB_MIN_GOOD:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < ORB_RATIO * n.distance]
    if len(good) < ORB_MIN_GOOD:
        return 0
    src = pts_a[[m.queryIdx for m in good]]
    dst = pts_b[[m.trainIdx for m in good]]
    # 4 px yeniden izdüşüm hatası: 384 px'lik görüntüde JPEG/yeniden örnekleme kaymasına tolerans
    model, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
    return 0 if model is None else int(inliers.sum())


def is_near_duplicate(hash_distance: int, inliers: int) -> bool:
    """
    Hash mesafesi ve ORB inlier sayısına göre çiftin aynı kaynaktan gelip gelmediğine karar verir.

    Parametreler:
        hash_distance: Döndürmeye dayanıklı pHash mesafesi.
        inliers: ORB/RANSAC inlier sayısı.

    Döndürür:
        True ise çift "neredeyse aynı" kabul edilir.
    """
    if hash_distance <= ORB_CLOSE_HASH_DISTANCE and inliers >= ORB_MIN_INLIERS_CLOSE:
        return True
    return inliers >= ORB_MIN_INLIERS_FAR


def verify_candidates(paths: list[str], candidates: list[tuple[int, int, int]]) -> pd.DataFrame:
    """
    Aday çiftleri ORB + RANSAC ile doğrular (j'nin aynalı hâli de denenir).

    ORB özellikleri yalnızca küçük bir LRU önbellekte tutulur; adaylar i'ye göre sıralı
    olduğu için aynı görüntü art arda kullanılır ve tekrar okuma azdır.

    Parametreler:
        paths: Görüntü yolları (sıra = DataFrame satır sırası).
        candidates: find_hash_candidates çıktısı.

    Döndürür:
        i, j, hash_mesafesi, orb_inlier, kabul sütunlu DataFrame.
    """
    feats = lru_cache(maxsize=ORB_CACHE_SIZE)(lambda k, mirror: orb_features(paths[k], mirror))
    rows, start, last_i = [], time.time(), -1
    for i, j, d in candidates:
        n = orb_inliers(feats(i, False), feats(j, False))
        if not is_near_duplicate(d, n):
            n = max(n, orb_inliers(feats(i, False), feats(j, True)))
        rows.append((i, j, d, n, is_near_duplicate(d, n)))
        # İlerleme: aday listesi i'ye göre sıralı; her 100 görüntü satırında bir yazdırılır
        if i // PROGRESS_EVERY != last_i // PROGRESS_EVERY:
            print(f"  [ORB doğrulama] görüntü {i}/{len(paths)}, aday {len(rows)}/{len(candidates)} "
                  f"({time.time() - start:.0f} sn)", flush=True)
            last_i = i
    feats.cache_clear()
    return pd.DataFrame(rows, columns=["i", "j", "hash_mesafesi", "orb_inlier", "kabul"])


def assign_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Kopya çiftlerinden bağlantılı bileşenleri (union-find) bularak grup kimliği atar.

    A~B ve B~C ise A, B, C aynı gruba düşer (aynı kaynağın farklı augmentasyonları).

    Parametreler:
        n: Toplam görüntü sayısı.
        pairs: (i, j) çiftleri.

    Döndürür:
        (n,) int dizisi; kopyası olmayan her görüntü kendi tekil grubundadır.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    # Grup kimliklerini 0'dan başlayan ardışık sayılara çevir
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def build_duplicate_table(df: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """
    MD5 ve doğrulanmış hash+ORB çiftlerini tek kopya tablosunda toplar.

    Parametreler:
        df: 'dosya_yolu', 'etiket', 'md5' sütunlu DataFrame (indeks 0..N-1).
        verified: verify_candidates çıktısı.

    Döndürür:
        i, j, yontem ('birebir' / 'phash+orb'), hash_mesafesi, orb_inlier, etiket_i, etiket_j,
        kapsam ('sinif_ici' / 'siniflar_arasi'), dosya_i, dosya_j sütunlu DataFrame.
    """
    accepted = verified[verified["kabul"]].drop(columns="kabul").assign(yontem="phash+orb")
    exact = pd.DataFrame(find_exact_duplicate_pairs(df["md5"]), columns=["i", "j"]).assign(yontem="birebir")
    pairs = pd.concat([exact, accepted]).drop_duplicates(subset=["i", "j"], keep="first")
    pairs = pairs.sort_values(["i", "j"]).reset_index(drop=True)
    pairs["etiket_i"] = df.loc[pairs["i"], "etiket"].to_numpy()
    pairs["etiket_j"] = df.loc[pairs["j"], "etiket"].to_numpy()
    pairs["kapsam"] = np.where(pairs["etiket_i"] == pairs["etiket_j"], "sinif_ici", "siniflar_arasi")
    pairs["dosya_i"] = df.loc[pairs["i"], "dosya_yolu"].map(lambda p: Path(p).name).to_numpy()
    pairs["dosya_j"] = df.loc[pairs["j"], "dosya_yolu"].map(lambda p: Path(p).name).to_numpy()
    return pairs


# ---------------------------------------------------------------------------
# Görselleştirme (outputs/dataset/)
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes) -> None:
    """
    Grafik eksenlerine sade (yalnızca y ızgarası, üst/sağ çerçevesiz) stil uygular.

    Parametreler:
        ax: matplotlib ekseni.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)


def plot_class_distribution(df: pd.DataFrame) -> Path:
    """
    Sınıf başına görüntü sayısını ve benzersiz grup (kopyalardan arındırılmış) sayısını çizer.

    Parametreler:
        df: 'etiket' ve 'grup_id' sütunlu DataFrame.

    Döndürür:
        Kaydedilen figürün yolu.
    """
    counts = df["etiket"].value_counts().reindex(CLASS_NAMES)
    groups = df.groupby("etiket")["grup_id"].nunique().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    x = np.arange(len(CLASS_NAMES))
    w = 0.38
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars1 = ax.bar(x - w / 2 - 0.01, counts, w, color=colors, label="Görüntü sayısı")
    bars2 = ax.bar(x + w / 2 + 0.01, groups, w, color=colors, alpha=0.45, hatch="///",
                   edgecolor=SURFACE, label="Benzersiz grup sayısı")
    for bars in (bars1, bars2):
        ax.bar_label(bars, padding=3, color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, counts.max() * 1.12)  # Çubuk etiketleri için üstte boşluk
    ax.set_ylabel("Adet", color=TEXT_SECONDARY)
    ax.set_title("Sınıf dağılımı: ham görüntü vs. kopyalardan arındırılmış grup",
                 color=TEXT_PRIMARY, loc="left", fontsize=11, pad=28)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    return save_figure(fig, "dataset", "sinif_dagilimi")


def plot_image_types(df: pd.DataFrame) -> Path:
    """
    Her sınıfta yayma ve yakın çekim görüntü sayılarını yatay yığılmış çubukla çizer.

    Parametreler:
        df: 'etiket' ve 'tip' sütunlu DataFrame.

    Döndürür:
        Kaydedilen figürün yolu.
    """
    table = pd.crosstab(df["etiket"], df["tip"]).reindex(index=CLASS_NAMES, columns=["yayma", "yakin_cekim"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=SURFACE)
    y = np.arange(len(CLASS_NAMES))[::-1]
    left = np.zeros(len(CLASS_NAMES))
    styles = {"yayma": dict(color="#2a78d6", label="Çok hücreli yayma"),
              "yakin_cekim": dict(color="#eb6834", label="Tek hücre yakın çekim")}
    for col in ["yayma", "yakin_cekim"]:
        vals = table[col].to_numpy()
        bars = ax.barh(y, vals, left=left, height=0.55, edgecolor=SURFACE, linewidth=2, **styles[col])
        ax.bar_label(bars, labels=[str(v) if v else "" for v in vals], label_type="center",
                     color="white", fontsize=9, fontweight="bold")
        left += vals
    ax.set_yticks(y, CLASS_NAMES)
    ax.set_xlabel("Görüntü sayısı", color=TEXT_SECONDARY)
    ax.set_title("Görüntü tipi dağılımı (tahmini)", color=TEXT_PRIMARY, loc="left", fontsize=11)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    return save_figure(fig, "dataset", "goruntu_tipleri")


def _thumb(path: str, size: int = 200) -> np.ndarray:
    """
    Görüntüyü en-boy oranını koruyarak kare küçük resme (beyaz kenarlı) dönüştürür.

    Parametreler:
        path: Görüntü yolu.
        size: Kare kenar uzunluğu (px).

    Döndürür:
        (size, size, 3) RGB dizi.
    """
    img = load_image(path)
    s = size / max(img.shape[:2])
    img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def plot_image_grid(rows: list[tuple[str, list[str]]], name: str, title: str,
                    captions: list[list[str]] | None = None) -> Path:
    """
    Satır başlıklı bir görüntü ızgarası çizer ve outputs/dataset/ altına kaydeder.

    Görüntüler küçük resim olarak tek tek okunur (tam çözünürlükte bellekte tutulmaz).

    Parametreler:
        rows: [(satır_başlığı, [görüntü_yolları])] listesi.
        name: Dosya adı (uzantısız).
        title: Figür başlığı.
        captions: Her görüntünün altına yazılacak kısa açıklamalar (isteğe bağlı).

    Döndürür:
        Kaydedilen figürün yolu.
    """
    n_cols = max(len(paths) for _, paths in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.9 * n_cols, 2.05 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for r, (label, paths) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(paths):
                ax.imshow(_thumb(paths[c]))
                if captions:
                    ax.set_title(captions[r][c], fontsize=7, color=TEXT_SECONDARY, pad=2)
        axes[r, 0].text(-0.06, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9, color=TEXT_PRIMARY)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return save_figure(fig, "dataset", name)


def save_dataset_figures(df: pd.DataFrame, pairs: pd.DataFrame) -> list[Path]:
    """
    Veri seti analizinin tüm sunum görsellerini outputs/dataset/ altına kaydeder.

    Parametreler:
        df: Tam veri seti indeksi (etiket, tip, grup_id, grup_boyutu dahil).
        pairs: build_duplicate_table çıktısı.

    Döndürür:
        Kaydedilen dosya yolları.
    """
    rng = random.Random(RANDOM_STATE)
    out = [plot_class_distribution(df), plot_image_types(df)]

    # Her sınıftan rastgele örnekler
    rows = [(c, rng.sample(df.loc[df["etiket"] == c, "dosya_yolu"].tolist(), 6)) for c in CLASS_NAMES]
    out.append(plot_image_grid(rows, "ornek_goruntuler", "Her sınıftan rastgele örnekler"))

    # Kopya grupları: her sınıfın en büyük grubu + (varsa) sınıflar arası kopya içeren çiftler
    group_rows, group_caps = [], []
    for c in CLASS_NAMES:
        sizes = df[df["etiket"] == c].groupby("grup_id").size()
        gid = sizes.idxmax()
        members = df[df["grup_id"] == gid]
        sample = members.sample(min(6, len(members)), random_state=RANDOM_STATE)
        group_rows.append((f"{c}\ngrup {gid} ({len(members)} görüntü)", sample["dosya_yolu"].tolist()))
        group_caps.append([f"{e} · {Path(p).name[-12:]}" for e, p in zip(sample["etiket"], sample["dosya_yolu"])])
    cross = pairs[pairs["kapsam"] == "siniflar_arasi"]
    for _, pr in cross.drop_duplicates(subset=["etiket_i", "etiket_j"]).head(3).iterrows():
        sub = df.loc[[pr["i"], pr["j"]]]
        group_rows.append((f"Sınıflar arası\n({pr['yontem']})", sub["dosya_yolu"].tolist()))
        group_caps.append([f"{e} · {Path(p).name[-12:]}" for e, p in zip(sub["etiket"], sub["dosya_yolu"])])
    out.append(plot_image_grid(group_rows, "kopya_grup_ornekleri",
                               "Kopya grubu örnekleri (aynı kaynağın augmentasyonları)", group_caps))

    # Yayma ve yakın çekim örnekleri (her sınıf × tip)
    type_rows = []
    for t, t_name in [("yayma", "Yayma"), ("yakin_cekim", "Yakın çekim")]:
        for c in CLASS_NAMES:
            pool = df.loc[(df["etiket"] == c) & (df["tip"] == t), "dosya_yolu"].tolist()
            if pool:
                type_rows.append((f"{c}\n{t_name}", rng.sample(pool, min(6, len(pool)))))
    out.append(plot_image_grid(type_rows, "yayma_ve_yakin_cekim", "Görüntü tipleri: yayma vs. yakın çekim"))
    return out


# ---------------------------------------------------------------------------
# Uçtan uca analiz
# ---------------------------------------------------------------------------

def build_dataset_index() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ham veri setini tarayıp tam indeksi ve kopya tablolarını üretir.

    Adımlar: listeleme → tek geçişte tarama (özellik, MD5, hash, tip) → hash adayları →
    ORB doğrulama → grup_id ataması.

    Döndürür:
        (indeks DataFrame'i, kabul edilen kopya çiftleri, doğrulanan tüm adaylar).
    """
    df = list_images()
    print(f"{len(df)} görüntü bulundu. Tek tek taranıyor...")
    df, hashes = scan_dataset(df)
    candidates = find_hash_candidates(hashes)
    del hashes
    print(f"{len(candidates)} hash adayı ORB ile doğrulanıyor...")
    verified = verify_candidates(df["dosya_yolu"].tolist(), candidates)
    pairs = build_duplicate_table(df, verified)
    df["grup_id"] = assign_groups(len(df), list(zip(pairs["i"], pairs["j"])))
    df["grup_boyutu"] = df.groupby("grup_id")["grup_id"].transform("size")
    df["grup_sinif_sayisi"] = df.groupby("grup_id")["etiket"].transform("nunique")
    return df, pairs, verified


def print_report(df: pd.DataFrame, pairs: pd.DataFrame, verified: pd.DataFrame) -> None:
    """
    Analiz bulgularını konsola özetler.

    Parametreler:
        df: Veri seti indeksi.
        pairs: Kabul edilen kopya çiftleri.
        verified: Doğrulanan tüm hash adayları (reddedilenler dahil).
    """
    pd.set_option("display.width", 200)
    print("\n=== Temel özellikler ===")
    print(summarize_properties(df).to_string(index=False))
    print("\n=== Kopya çiftleri (yöntem × kapsam) ===")
    print(pd.crosstab(pairs["yontem"], pairs["kapsam"], margins=True))
    labels = df["etiket"].to_numpy()
    cross_cand = verified[labels[verified["i"]] != labels[verified["j"]]]
    print(f"\nSınıflar arası hash adayı: {len(cross_cand)}, ORB ile kabul edilen: {int(cross_cand['kabul'].sum())}")
    exact_files = df["md5"].duplicated(keep=False)
    print("\n=== Birebir kopyalar ===")
    print(f"Birebir kopyası olan dosya: {exact_files.sum()}, fazlalık dosya: {df['md5'].duplicated().sum()}")
    print(df.assign(fazlalik=df["md5"].duplicated()).groupby("etiket")["fazlalik"].sum())
    print("\n=== Gruplar ===")
    g = df.groupby("etiket").agg(goruntu=("grup_id", "size"), benzersiz_grup=("grup_id", "nunique"),
                                 kopyali_goruntu=("grup_boyutu", lambda s: int((s > 1).sum())),
                                 en_buyuk_grup=("grup_boyutu", "max"))
    print(g)
    print(f"Toplam grup: {df['grup_id'].nunique()}, çok sınıflı grup: "
          f"{df.loc[df['grup_sinif_sayisi'] > 1, 'grup_id'].nunique()}")
    print("\n=== Görüntü tipi ===")
    print(pd.crosstab(df["etiket"], df["tip"], margins=True))
    print("\n=== Benzersiz grup × tip (grubun ilk görüntüsünün tipi) ===")
    first = df.drop_duplicates("grup_id")
    print(pd.crosstab(first["etiket"], first["tip"], margins=True))


if __name__ == "__main__":
    index_df, pair_df, verified_df = build_dataset_index()
    DATASET_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(DATASET_INDEX_CSV, index=False, encoding="utf-8")
    pair_df.to_csv(DUPLICATE_PAIRS_CSV, index=False, encoding="utf-8")
    print_report(index_df, pair_df, verified_df)
    for p in save_dataset_figures(index_df, pair_df):
        print("Kaydedildi:", p)
