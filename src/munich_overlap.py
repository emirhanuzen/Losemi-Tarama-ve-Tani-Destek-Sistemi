"""
Adım 8.0 — Ön kontrol: Kaggle veri seti ile Münih AML sitomorfoloji veri seti arasında kaynak örtüşmesi.

Amaç: Münih görüntülerini dış doğrulamada kullanmadan önce, hiçbirinin (veya aynı hücrenin) Kaggle eğitim/test
verisinde bulunmadığını göstermek. Eşleşen Münih görüntüleri dış doğrulamadan çıkarılır. Hiçbir dosya silinmez.

Münih görüntüleri 400x400 tek hücre kırpımlarıdır (100x objektif, görüş alanı ~4 eritrosit çapı). Kaggle'da
aynı görüş alanına sahip olanlar yakın çekimlerdir; yaymalar ~25 eritrosit çapı genişliğindedir. Bu yüzden iki
düzeyde aranır (ALL-IDB kaynak tespitinin karşılıkları, src/allidb_overlap.py):
A) Görüntü düzeyi: tüm Kaggle görüntüleri ↔ tüm Münih görüntüleri. Döndürmeye/aynalamaya dayanıklı pHash adayları
   (her Kaggle görüntüsü için en yakın 3 Münih görüntüsü, eşikten bağımsız) + geniş bağlamlı NCC (72 döndürme/ayna
   varyantı × 5 ölçek; TM_CCOEFF_NORMED doğrusal parlaklık/kontrast farkına duyarsız) ve ORB/RANSAC.
B) Hücre düzeyi: Kaggle hücre kırpımları (K-means, data/processed/cells) ↔ Münih merkez hücresi. İkisi de aynı
   pipeline'la (aynı kırpım payı) çıkarılır ve eritrosit çapı oranıyla aynı ölçeğe getirilir; böylece bir Münih
   hücresinin bir Kaggle yaymasından kesilmiş olması da yakalanır. Her Münih hücresi için en yakın 3 Kaggle hücresi
   (pHash) NCC ile doğrulanır.
Her iki düzey için pozitif kontrol: Münih görüntüleri döndürülüp aynalanır, kırpılır, rengi kaydırılır, JPEG ile
sıkıştırılır (B'de ayrıca Kaggle yayma ölçeğine küçültülür); yöntemin bunları kaynağına bağlayıp bağlamadığı ölçülür.

C) Kapsamlı yakın çekim taraması: pozitif kontrolde pHash adayları gerçek kaynağı çoğu zaman kaçırdığı için kaynağı
   bilinmeyen Kaggle yakın çekimleri tüm Münih görüntüleriyle (ölçek eritrosit çapıyla eşitlenmiş kaba NCC + tam
   çözünürlükte geniş bağlamlı NCC) karşılaştırılır; pozitif ve negatif (ALL-IDB kaynaklı) kontrollerle.
Dışlama kuralı ve hücre düzeyinin neden yalnızca raporlandığı: write_exclusions.

Çalıştırma (proje kökünden):
    python -m src.munich_overlap             # A + B (+ dışlama listesi)
    python -m src.munich_overlap kapsamli    # C (+ dışlama listesi)
    python -m src.munich_overlap dislanan    # yalnızca dışlama listesi ve özet (kayıtlı sonuçlardan)
Çıktılar: outputs/external/munich_kaynak_*.csv, munich_dislanan.csv (dış doğrulamadan çıkarılacak görüntüler).
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.allidb_overlap import context_ncc, cross_distances, gray_of, kaggle_hashes, load_kaggle, orb_of
from src.config import (
    CELLS_INDEX_CSV,
    EXTERNAL_PROCESSED_DIR,
    MUNICH_CLASSES,
    MUNICH_DIR,
    MUNICH_EXTRA_CLASSES,
    N_JOBS,
    RANDOM_STATE,
)
from src.dataset import HASH_CANDIDATE_DISTANCE, is_near_duplicate, orb_inliers, rotation_phashes
from src.utils import load_image, resolve_path, stage_dir

CACHE_DIR = EXTERNAL_PROCESSED_DIR / "munich_ara"
PROGRESS_EVERY = 100
CHUNK = 100
# Aday sayısı: eşikten bağımsız en yakın 3. Münih'te aynı sınıfın hücreleri birbirine çok benzer (ör. tüm LYT'ler
# yuvarlak, koyu çekirdekli); gerçek kaynak hash'te ilk sıralarda olur (pozitif kontrolde ölçülür).
TOP_K = 3
# Görüntü düzeyi kabul: ALL-IDB tek hücre eşleşmesiyle aynı NCC eşiği (0.95; allidb_overlap.NCC_MIN).
WHOLE_NCC_MIN = 0.95
# Hücre düzeyi karşılaştırma ölçeği: eritrosit çapı 20 px. Kaggle yayma hücreleri ~10 px çapta; 20 px'e büyütmek
# bilgi eklemez ama döndürmede örnekleme kaybını azaltır. Münih hücreleri ~58 px'ten küçültülür.
CELL_REF_DIAMETER = 20
# Eritrosit çapı tahminindeki hataya tolerans: şablon 0.85-1.15 ölçeklerinde aranır.
CELL_SCALES = (0.85, 1.0, 1.15)
# Hücre düzeyi kabul eşiği: küçük, yuvarlak çekirdeklerde ilgisiz hücreler de yüksek NCC verebildiği için eşik
# negatif dağılımın üstünde seçilir (bkz. cell_threshold; pozitif kontrolle birlikte raporlanır).
CELL_NCC_FLOOR = 0.95
N_POSITIVE = 40


# ---------------------------------------------------------------------------
# Münih görüntüleri
# ---------------------------------------------------------------------------

def list_munich() -> pd.DataFrame:
    """
    Münih sınıf klasörlerindeki görüntüleri listeler (eksik klasör uyarıyla atlanır).

    Döndürür:
        sinif (MYO/LYT/NGS/MYB), goreli_yol (MUNICH_DIR'e göre, ileri eğik çizgili), beklenen (proje sınıfı veya
        boş) sütunlu tablo; sınıf ve dosya adına göre sıralı.
    """
    rows = []
    for cls in list(MUNICH_CLASSES) + MUNICH_EXTRA_CLASSES:
        folder = MUNICH_DIR / cls
        if not folder.is_dir():
            print(f"  Uyarı: {folder} yok, {cls} atlandı.")
            continue
        # Yalnızca tamamlanmış TIFF'ler (indirme sırasında oluşan .partial vb. dosyalar atlanır)
        for p in sorted(q for q in folder.iterdir() if q.suffix.lower() in (".tif", ".tiff")):
            rows.append({"sinif": cls, "goreli_yol": f"{cls}/{p.name}", "beklenen": MUNICH_CLASSES.get(cls, "")})
    return pd.DataFrame(rows, columns=["sinif", "goreli_yol", "beklenen"])


def munich_image(rel: str) -> np.ndarray:
    """
    Bir Münih görüntüsünü RGB okur (TIFF'teki alfa kanalı atılır).

    Parametreler:
        rel: MUNICH_DIR'e göre göreli yol.

    Döndürür:
        RGB uint8 görüntü.
    """
    return load_image(MUNICH_DIR / rel)


def center_cell(rgb: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Uçtan uca pipeline'ın segmentasyonuyla görüntü merkezine en yakın hücrenin kırpımını çıkarır.

    Parametreler:
        rgb: RGB görüntü.

    Döndürür:
        (renk normalize kırpım veya hücre yoksa None, eritrosit çapı (256 px uzayında)).
    """
    from src.pipeline import segment_with_fallback
    from src.preprocessing import preprocess
    from src.segmentation import extract_cells

    _, steps = preprocess(rgb)
    seg = segment_with_fallback(steps)
    cells = extract_cells(steps["renk_normalize"], seg["etiketler"], steps["gecerli_maske"], seg["eritrosit_capi"])
    if not cells:
        return None, seg["eritrosit_capi"]
    centers = [np.hypot(c["kutu"][0] + c["kutu"][2] / 2 - 128, c["kutu"][1] + c["kutu"][3] / 2 - 128) for c in cells]
    return cells[int(np.argmin(centers))]["kirpim"], seg["eritrosit_capi"]


def munich_chunk(rels: list[str]) -> list[dict]:
    """
    Bir parça Münih görüntüsü için görüntü ve merkez hücre hash'lerini hesaplar (paralel işçide çalışır).

    Parametreler:
        rels: Göreli yollar.

    Döndürür:
        Görüntü başına {goreli_yol, h_goruntu (72), h_hucre (72 veya sıfır), eritrosit_capi, hucre_var}.
    """
    out = []
    for rel in rels:
        rgb = munich_image(rel)
        crop, diameter = center_cell(rgb)
        out.append({
            "goreli_yol": rel, "h_goruntu": rotation_phashes(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
            "h_hucre": rotation_phashes(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)) if crop is not None
            else np.zeros(72, np.uint64),
            "eritrosit_capi": diameter, "hucre_var": crop is not None,
        })
    return out


def munich_hashes(images: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm Münih görüntülerinin hash'lerini sınıf başına önbellekli hesaplar (MYO sonradan eklenirse yalnızca o hesaplanır).

    Parametreler:
        images: list_munich çıktısı.

    Döndürür:
        images + h_goruntu, h_hucre (nesne sütunu, (72,) dizi), eritrosit_capi, hucre_var.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for cls, sub in images.groupby("sinif", sort=False):
        cache = CACHE_DIR / f"hash_{cls}.pkl"
        if cache.exists():
            done = pd.read_pickle(cache)
            if set(done["goreli_yol"]) == set(sub["goreli_yol"]):
                parts.append(done)
                print(f"  [Münih hash] {cls}: önbellekten ({len(done)})")
                continue
        rels = sub["goreli_yol"].tolist()
        chunks = [rels[i:i + CHUNK] for i in range(0, len(rels), CHUNK)]
        rows, start = [], time.time()
        with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
            for i, res in enumerate(par(delayed(munich_chunk)(c) for c in chunks)):
                rows += res
                if (i + 1) % max(1, PROGRESS_EVERY // CHUNK) == 0 or i + 1 == len(chunks):
                    print(f"  [Münih hash] {cls}: {len(rows)}/{len(rels)} ({time.time() - start:.0f} sn)", flush=True)
        done = sub.merge(pd.DataFrame(rows), on="goreli_yol")
        done.to_pickle(cache)
        parts.append(done)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Kaggle hücre kırpımları
# ---------------------------------------------------------------------------

def kaggle_cell_chunk(paths: list[str]) -> list[np.ndarray]:
    """
    Kaggle hücre kırpımlarının döndürmeye dayanıklı pHash'lerini hesaplar (paralel işçide).

    Parametreler:
        paths: Kırpım PNG yolları.

    Döndürür:
        (72,) hash dizileri listesi.
    """
    return [rotation_phashes(gray_of(p)) for p in paths]


def kaggle_cell_hashes() -> tuple[pd.DataFrame, np.ndarray]:
    """
    K-means hücre indeksindeki tüm Kaggle kırpımlarının hash'lerini önbellekli hesaplar.

    Döndürür:
        (hücre tablosu: hucre_id, kirpim_yolu, kaynak_dosya, etiket, tip, eritrosit_capi; (N, 72) hash matrisi).
    """
    cells = pd.read_csv(CELLS_INDEX_CSV)[["hucre_id", "kirpim_yolu", "kaynak_dosya", "etiket", "tip", "eritrosit_capi"]]
    cache = CACHE_DIR / "kaggle_hucre_hash.npy"
    if cache.exists():
        h = np.load(cache)
        if len(h) == len(cells):
            print(f"  [Kaggle hücre hash] önbellekten ({len(h)})")
            return cells, h
    paths = cells["kirpim_yolu"].tolist()
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]
    out, start = [], time.time()
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for res in par(delayed(kaggle_cell_chunk)(c) for c in chunks):
            out += res
            print(f"  [Kaggle hücre hash] {len(out)}/{len(paths)} ({time.time() - start:.0f} sn)", flush=True)
    h = np.stack(out)
    np.save(cache, h)
    return cells, h


# ---------------------------------------------------------------------------
# Aday seçimi ve doğrulama
# ---------------------------------------------------------------------------

def top_candidates(hq: np.ndarray, hr: np.ndarray, k: int = TOP_K, block: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Her sorgu için en yakın k referansı (döndürmeye dayanıklı pHash mesafesiyle) bulur; tam matris tutulmaz.

    Parametreler:
        hq: (N, 72) sorgu hash'leri.
        hr: (M, 72) referans hash'leri.
        k: Aday sayısı.
        block: Aynı anda işlenen sorgu sayısı (bellek: block × M × 72 × 8 bayt).

    Döndürür:
        ((N, k) referans indeksleri, (N, k) mesafeler).
    """
    idx = np.zeros((len(hq), k), np.int64)
    dist = np.zeros((len(hq), k), np.uint8)
    start = time.time()
    for lo in range(0, len(hq), block):
        d = cross_distances(hq[lo:lo + block], hr, block=block)
        top = np.argsort(d, axis=1, kind="stable")[:, :k]
        idx[lo:lo + block] = top
        dist[lo:lo + block] = np.take_along_axis(d, top, axis=1)
        if (lo // block + 1) % 200 == 0:
            print(f"    aday: {lo + block}/{len(hq)} ({time.time() - start:.0f} sn)", flush=True)
    return idx, dist


def to_ref_scale(gray: np.ndarray, diameter: float, ref: float = CELL_REF_DIAMETER) -> np.ndarray:
    """
    Gri kırpımı eritrosit çapı ref px olacak şekilde yeniden ölçekler.

    Parametreler:
        gray: Gri kırpım.
        diameter: Kırpımın eritrosit çapı (px).
        ref: Hedef çap (px).

    Döndürür:
        Ölçeklenmiş gri görüntü (en az 8x8).
    """
    s = ref / max(diameter, 1e-6)
    size = (max(8, round(gray.shape[1] * s)), max(8, round(gray.shape[0] * s)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def rotations(gray: np.ndarray, step: int = 10) -> list[np.ndarray]:
    """
    Görüntünün aynasız ve aynalı hâlinin step derecelik döndürmelerini (boyut korunur, köşeler yansıtılır) üretir.

    Parametreler:
        gray: Gri görüntü.
        step: Açı adımı (derece).

    Döndürür:
        Varyant listesi (2 × 360/step).
    """
    h, w = gray.shape
    out = []
    for base in (gray, np.ascontiguousarray(gray[:, ::-1])):
        for angle in range(0, 360, step):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            out.append(cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out


def cell_ncc(kaggle_gray: np.ndarray, d_kaggle: float, munich_gray: np.ndarray, d_munich: float) -> float:
    """
    İki hücre kırpımını ortak ölçekte karşılaştırır: Kaggle kırpımının merkez %70'i, Münih kırpımının 72
    döndürme/ayna varyantında ve 3 ölçekte aranır.

    Parametreler:
        kaggle_gray, d_kaggle: Kaggle gri kırpımı ve eritrosit çapı.
        munich_gray, d_munich: Münih gri kırpımı ve eritrosit çapı.

    Döndürür:
        En yüksek normalize çapraz korelasyon (şablon sığmazsa -1).
    """
    k = to_ref_scale(kaggle_gray, d_kaggle)
    m = to_ref_scale(munich_gray, d_munich)
    h, w = k.shape
    core = k[round(0.15 * h):round(0.85 * h), round(0.15 * w):round(0.85 * w)]
    variants = rotations(m)
    best = -1.0
    for f in CELL_SCALES:
        tpl = cv2.resize(core, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        if tpl.shape[0] > m.shape[0] or tpl.shape[1] > m.shape[1] or tpl.std() < 1:
            continue
        for v in variants:
            best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(v, tpl, cv2.TM_CCOEFF_NORMED))[1]))
    return best


def whole_chunk(pairs: list[tuple]) -> list[dict]:
    """
    Görüntü düzeyi adayları doğrular: geniş bağlamlı NCC ve ORB/RANSAC (Adım 1 kuralı).

    Geniş bağlamlı NCC (çift başına 360 şablon eşleştirmesi) yalnızca Kaggle yakın çekimlerinde ve hash
    mesafesi ≤ HASH_CANDIDATE_DISTANCE olan çiftlerde hesaplanır: Kaggle yaymalarının görüş alanı (~25 eritrosit
    çapı) bir Münih görüntüsününkinin (~4) birkaç katı olduğundan yayma bütünüyle bir Münih görüntüsü olamaz;
    yaymaların içindeki hücreler B düzeyinde aranır. ORB/RANSAC her çifte uygulanır.

    Parametreler:
        pairs: (kaggle_yolu, kaggle_tipi, munich_goreli_yol, hash_mesafesi) listesi.

    Döndürür:
        Aday başına {kaggle, munich, hash_mesafesi, ncc (hesaplanmadıysa NaN), orb_inlier, kabul}.
    """
    out, cache = [], {}
    for kp, tip, mp, hd in pairs:
        if kp not in cache:
            cache.clear()
            cache[kp] = gray_of(kp)
        kg = cache[kp]
        mg = cv2.cvtColor(munich_image(mp), cv2.COLOR_RGB2GRAY)
        ncc = context_ncc(kg, mg) if tip == "yakin_cekim" or hd <= HASH_CANDIDATE_DISTANCE else np.nan
        inl = max(orb_inliers(orb_of(kg), orb_of(mg)), orb_inliers(orb_of(kg), orb_of(mg, mirror=True)))
        out.append({"kaggle": kp, "munich": mp, "hash_mesafesi": int(hd), "ncc": ncc, "orb_inlier": inl,
                    "kabul": bool((ncc >= WHOLE_NCC_MIN) or is_near_duplicate(int(hd), inl))})
    return out


def cell_chunk(items: list[tuple]) -> list[dict]:
    """
    Hücre düzeyi adayları NCC ile doğrular (paralel işçide).

    Parametreler:
        items: (munich_goreli_yol, d_munich, [(kaggle_kirpim_yolu, d_kaggle, hash_mesafesi), ...]) listesi.

    Döndürür:
        Aday başına {munich, kaggle_kirpim, hash_mesafesi, ncc}.
    """
    out = []
    for mp, dm, cands in items:
        crop, d_seg = center_cell(munich_image(mp))
        if crop is None:
            continue
        mg = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        for kp, dk, hd in cands:
            out.append({"munich": mp, "kaggle_kirpim": kp, "hash_mesafesi": int(hd),
                        "ncc": cell_ncc(gray_of(kp), dk, mg, d_seg)})
    return out


def run_parallel(func, items: list, what: str, chunk: int = 50) -> pd.DataFrame:
    """
    Bir doğrulama fonksiyonunu parçalar hâlinde paralel çalıştırır ve ilerleme yazdırır.

    Parametreler:
        func: Parça fonksiyonu (liste alır, sözlük listesi döndürür).
        items: İş öğeleri.
        what: İlerleme etiketi.
        chunk: Parça boyu.

    Döndürür:
        Sonuç tablosu.
    """
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    rows, start, done = [], time.time(), 0
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for c, res in zip(chunks, par(delayed(func)(c) for c in chunks)):
            rows += res
            done += len(c)
            if done % PROGRESS_EVERY < chunk or done == len(items):
                print(f"  [{what}] {done}/{len(items)} ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pozitif kontrol
# ---------------------------------------------------------------------------

def disguise(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Görüntüyü bir veri seti oluşturucunun yapabileceği dönüşümlerle değiştirir: döndürme, aynalama, merkez
    kırpma, Lab renk kayması, JPEG sıkıştırma ve 256x256'ya küçültme.

    Parametreler:
        rgb: RGB görüntü.
        rng: Rastgele sayı üreteci.

    Döndürür:
        Dönüştürülmüş RGB görüntü.
    """
    h, w = rgb.shape[:2]
    img = rgb[:, ::-1] if rng.random() < 0.5 else rgb
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), borderMode=cv2.BORDER_REFLECT)
    f = rng.uniform(0.8, 0.95)
    ch, cw = round(h * f), round(w * f)
    img = img[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[..., 0] *= rng.uniform(0.85, 1.15)
    lab[..., 1:] += rng.uniform(-12, 12, 2)
    img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def positive_control(munich: pd.DataFrame) -> pd.DataFrame:
    """
    Yöntemin, değiştirilmiş Münih görüntülerini kaynağına bağlayıp bağlamadığını ölçer.

    A) Görüntü: disguise(Münih) sorgusu tüm Münih hash'lerinde aranır; kaynak ilk 3 aday içinde mi, NCC ve kabul?
    B) Hücre: Münih merkez hücresi disguise edilip Kaggle yayma ölçeğine (eritrosit çapı ~10 px) küçültülür; tüm
       Münih hücre hash'lerinde aranır; kaynak ilk 3'te mi, NCC kaç?

    Parametreler:
        munich: munich_hashes çıktısı.

    Döndürür:
        Kontrol başına {duzey, munich, kaynak_sirasi, ncc, kabul} tablosu.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    pool = munich[munich["hucre_var"]].reset_index(drop=True)
    sample = pool.iloc[rng.choice(len(pool), min(N_POSITIVE, len(pool)), replace=False)]
    hw = np.stack(munich["h_goruntu"].to_numpy())
    hc = np.stack(pool["h_hucre"].to_numpy())
    rows = []
    for _, r in sample.iterrows():
        rgb = munich_image(r["goreli_yol"])
        fake = disguise(rgb, rng)
        fg = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
        idx, _ = top_candidates(rotation_phashes(fg)[None], hw)
        src = int(np.where(munich["goreli_yol"].to_numpy() == r["goreli_yol"])[0][0])
        rank = int(np.where(idx[0] == src)[0][0]) + 1 if src in idx[0] else 0
        ncc = context_ncc(fg, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        rows.append({"duzey": "goruntu", "munich": r["goreli_yol"], "kaynak_sirasi": rank, "ncc": ncc,
                     "kabul": bool(rank > 0 and ncc >= WHOLE_NCC_MIN)})
        # B) Hücre: merkez hücre → Kaggle yayma ölçeği (d ≈ 10 px), renk/döndürme/JPEG
        crop, d_m = center_cell(rgb)
        small = cv2.resize(crop, None, fx=10 / d_m, fy=10 / d_m, interpolation=cv2.INTER_AREA)
        fake_c = disguise(cv2.resize(small, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), rng)
        fake_c = cv2.resize(fake_c, small.shape[1::-1], interpolation=cv2.INTER_AREA)   # 256'dan geri, d ≈ 10
        fc = cv2.cvtColor(fake_c, cv2.COLOR_BGR2GRAY)
        idx, _ = top_candidates(rotation_phashes(fc)[None], hc)
        src = int(np.where(pool["goreli_yol"].to_numpy() == r["goreli_yol"])[0][0])
        rank = int(np.where(idx[0] == src)[0][0]) + 1 if src in idx[0] else 0
        ncc = cell_ncc(fc, 10.0 * small.shape[1] / fake_c.shape[1], cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), d_m)
        rows.append({"duzey": "hucre", "munich": r["goreli_yol"], "kaynak_sirasi": rank, "ncc": ncc,
                     "kabul": None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C) Kapsamlı yakın çekim taraması (pHash adaylarından bağımsız)
# ---------------------------------------------------------------------------
# Pozitif kontrolde pHash adayları gerçek kaynağı ilk 3'e çoğu zaman koyamadı (Münih sınıfları kendi içinde çok
# benzer). Münih kökenli olabilecek tek Kaggle görüntüleri, kaynağı bilinmeyen yakın çekimlerdir (Münih tek hücre
# kırpımı; yaymanın görüş alanı ~6 kat büyük). Bunlar TÜM Münih görüntüleriyle iki aşamada karşılaştırılır:
# 1) Kaba tarama: iki görüntü de eritrosit çapı EX_DIAMETER px olacak şekilde küçültülür (ölçek eşitlenir); sorgunun
#    merkez %70'i 36 döndürme/ayna varyantında Münih küçük görüntüsünde aranır (NCC). Görüntü başına en iyi EX_TOP.
# 2) Doğrulama: EX_TOP aday tam çözünürlükte geniş bağlamlı NCC ile (allidb_overlap.context_ncc, eşik 0.95).
# Pozitif kontrol: disguise() ile değiştirilmiş Münih görüntüleri; negatif kontrol: kaynağı ALL-IDB olan yakın çekimler.
EX_DIAMETER = 10.0     # Kaba tarama ölçeği: eritrosit 10 px (Münih 400 px görüntü → ~43 px); hücre yapısı seçilebilir
EX_ROT_STEP = 20       # 20° adım × ayna = 36 varyant; merkez kırpım küçük olduğundan 10° kaba aşamada gereksiz
EX_TOP = 10            # Doğrulanan aday sayısı
EX_POSITIVE = 20
EX_NEGATIVE = 40


def ex_thumb(gray: np.ndarray, diameter256: float) -> np.ndarray:
    """
    Gri görüntüyü önce uzun kenarı 256 px'e (pipeline uzayı), sonra eritrosit çapı EX_DIAMETER olacak şekilde küçültür.

    Parametreler:
        gray: Gri görüntü.
        diameter256: 256 px uzayında eritrosit çapı.

    Döndürür:
        Küçük gri görüntü (en az 16 px).
    """
    s = 256 / max(gray.shape) * EX_DIAMETER / max(diameter256, 1e-6)
    size = (max(16, round(gray.shape[1] * s)), max(16, round(gray.shape[0] * s)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def munich_thumbs(munich: pd.DataFrame) -> list[np.ndarray]:
    """
    Tüm Münih görüntülerinin kaba tarama küçük görüntülerini önbellekli hesaplar (munich_hashes sırasıyla).

    Parametreler:
        munich: munich_hashes çıktısı (goreli_yol, eritrosit_capi).

    Döndürür:
        Küçük gri görüntü listesi.
    """
    cache = CACHE_DIR / "kucuk_goruntuler.pkl"
    if cache.exists():
        done = pd.read_pickle(cache)
        if list(done["goreli_yol"]) == list(munich["goreli_yol"]):
            return list(done["kucuk"])
    items = list(zip(munich["goreli_yol"], munich["eritrosit_capi"]))
    chunks = [items[i:i + 500] for i in range(0, len(items), 500)]
    thumbs, start = [], time.time()
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for res in par(delayed(_thumb_chunk)(c) for c in chunks):
            thumbs += res
            print(f"  [Münih küçük] {len(thumbs)}/{len(items)} ({time.time() - start:.0f} sn)", flush=True)
    pd.DataFrame({"goreli_yol": munich["goreli_yol"], "kucuk": thumbs}).to_pickle(cache)
    return thumbs


def _thumb_chunk(items: list[tuple[str, float]]) -> list[np.ndarray]:
    """
    Bir parça Münih görüntüsünün küçük görüntülerini üretir (paralel işçide).

    Parametreler:
        items: (göreli yol, eritrosit çapı) listesi.

    Döndürür:
        Küçük gri görüntüler.
    """
    return [ex_thumb(cv2.cvtColor(munich_image(p), cv2.COLOR_RGB2GRAY), d) for p, d in items]


def query_variants(thumb: np.ndarray) -> list[np.ndarray]:
    """
    Sorgu küçük görüntüsünün döndürme/ayna varyantlarının merkez %70 kırpımları (döndürme köşeleri dışarıda).

    Parametreler:
        thumb: Sorgu küçük gri görüntüsü.

    Döndürür:
        36 şablon.
    """
    h, w = thumb.shape
    return [v[round(0.15 * h):round(0.85 * h), round(0.15 * w):round(0.85 * w)] for v in rotations(thumb, EX_ROT_STEP)]


def coarse_scan(queries: list[tuple[str, np.ndarray]], thumbs: list[np.ndarray]) -> list[dict]:
    """
    Kaba tarama: her sorgu için tüm Münih küçük görüntülerinde en yüksek NCC'yi hesaplar, en iyi EX_TOP'u döndürür.

    Parametreler:
        queries: (sorgu kimliği, sorgu küçük görüntüsü) listesi.
        thumbs: Münih küçük görüntüleri.

    Döndürür:
        Sorgu başına {sorgu, adaylar (indeks listesi), kaba_ncc (aynı sırada)}.
    """
    out = []
    for qid, q in queries:
        tpls = [t for t in query_variants(q) if t.std() >= 1]
        scores = np.full(len(thumbs), -1.0)
        for j, m in enumerate(thumbs):
            best = -1.0
            for t in tpls:
                if t.shape[0] > m.shape[0] or t.shape[1] > m.shape[1]:
                    t = cv2.resize(t, (min(t.shape[1], m.shape[1]), min(t.shape[0], m.shape[0])))
                best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED))[1]))
            scores[j] = best
        top = np.argsort(-scores)[:EX_TOP]
        out.append({"sorgu": qid, "adaylar": top.tolist(), "kaba_ncc": scores[top].tolist()})
    return out


def _coarse_chunk(queries: list[tuple[str, np.ndarray]]) -> list[dict]:
    """
    coarse_scan'i işçide çalıştırır; Münih küçük görüntüleri önbellek dosyasından bir kez okunur.

    Parametreler:
        queries: (sorgu kimliği, sorgu küçük görüntüsü) listesi.

    Döndürür:
        coarse_scan çıktısı.
    """
    thumbs = list(pd.read_pickle(CACHE_DIR / "kucuk_goruntuler.pkl")["kucuk"])
    return coarse_scan(queries, thumbs)


def _verify_chunk(items: list[tuple]) -> list[dict]:
    """
    Kaba tarama adaylarını tam çözünürlükte geniş bağlamlı NCC ile doğrular (işçide).

    Parametreler:
        items: (sorgu kimliği, sorgu gri görüntüsü, [Münih göreli yolları]) listesi.

    Döndürür:
        Aday başına {sorgu, munich, ncc}.
    """
    out = []
    for qid, gray, cands in items:
        for mp in cands:
            out.append({"sorgu": qid, "munich": mp,
                        "ncc": context_ncc(gray, cv2.cvtColor(munich_image(mp), cv2.COLOR_RGB2GRAY))})
    return out


def exhaustive_closeups(munich: pd.DataFrame) -> pd.DataFrame:
    """
    Kaynağı bilinmeyen Kaggle yakın çekimlerini (ve kontrolleri) tüm Münih görüntüleriyle karşılaştırır.

    Sorgular: (1) kaynak "bilinmiyor" yakın çekimler; (2) pozitif kontrol: EX_POSITIVE disguise(Münih) görüntüsü
    (gerçek kaynağı bilinir); (3) negatif kontrol: kaynağı ALL-IDB olan EX_NEGATIVE yakın çekim (Münih'te olamaz).

    Parametreler:
        munich: munich_hashes çıktısı.

    Döndürür:
        Sorgu başına {sorgu, grup, etiket, en_iyi_munich, kaba_ncc, ncc, kaynak_sirasi (pozitifte), kabul}.
    """
    from src.preprocessing import preprocess
    from src.segmentation import image_rbc_diameter

    from src.classifier import load_split

    thumbs = munich_thumbs(munich)
    split = load_split()   # v2 ayrımı: tip ve kaynak (ALL-IDB / bilinmiyor) sütunlarıyla
    cells = pd.read_csv(CELLS_INDEX_CSV).groupby("kaynak_dosya")["eritrosit_capi"].first()
    close = split[split["tip"] == "yakin_cekim"]
    rng = np.random.default_rng(RANDOM_STATE)
    unknown = close[close["kaynak"] == "bilinmiyor"]
    known = close[close["kaynak"] != "bilinmiyor"]
    known = known.iloc[rng.choice(len(known), min(EX_NEGATIVE, len(known)), replace=False)]
    rows, grays = [], {}
    for grp, sub in (("kaggle_bilinmiyor", unknown), ("negatif_ALL-IDB", known)):
        for _, r in sub.iterrows():
            g = gray_of(r["dosya_yolu"])
            grays[r["dosya_yolu"]] = g
            rows.append({"sorgu": r["dosya_yolu"], "grup": grp, "etiket": r["etiket"], "kaynak": "",
                         "d": float(cells.get(r["dosya_yolu"], np.nan))})
    pool = munich[munich["hucre_var"]].reset_index(drop=True)
    for _, r in pool.iloc[rng.choice(len(pool), EX_POSITIVE, replace=False)].iterrows():
        fake = disguise(munich_image(r["goreli_yol"]), rng)          # BGR (imdecode)
        rgb = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
        d, _ = image_rbc_diameter(preprocess(rgb)[1])
        qid = "pozitif/" + r["goreli_yol"]
        grays[qid] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rows.append({"sorgu": qid, "grup": "pozitif_kontrol", "etiket": r["sinif"], "kaynak": r["goreli_yol"], "d": d})
    q = pd.DataFrame(rows)
    # Eritrosit çapı ölçülemeyen yakın çekimde Münih medyanı kullanılır (iki veri seti de ~100x)
    q["d"] = q["d"].fillna(float(munich["eritrosit_capi"].median()))
    print(f"  sorgu: {q['grup'].value_counts().to_dict()}", flush=True)

    queries = [(r.sorgu, ex_thumb(grays[r.sorgu], r.d)) for r in q.itertuples()]
    chunks = [queries[i:i + 5] for i in range(0, len(queries), 5)]
    coarse, start = [], time.time()
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for res in par(delayed(_coarse_chunk)(c) for c in chunks):
            coarse += res
            if len(coarse) % 20 == 0 or len(coarse) == len(queries):
                print(f"  [kaba tarama] {len(coarse)}/{len(queries)} ({time.time() - start:.0f} sn)", flush=True)
    paths = munich["goreli_yol"].to_numpy()
    items = [(c["sorgu"], grays[c["sorgu"]], list(paths[c["adaylar"]])) for c in coarse]
    fine = run_parallel(_verify_chunk, items, "doğrulama", chunk=5)
    coarse_df = pd.DataFrame([{"sorgu": c["sorgu"], "munich": paths[j], "kaba_ncc": s}
                              for c in coarse for j, s in zip(c["adaylar"], c["kaba_ncc"])])
    fine = fine.merge(coarse_df, on=["sorgu", "munich"])
    best = fine.sort_values("ncc", ascending=False).drop_duplicates("sorgu")
    out = q.merge(best.rename(columns={"munich": "en_iyi_munich"}), on="sorgu", how="left")
    rank = {}
    for c in coarse:
        src = q.loc[q["sorgu"] == c["sorgu"], "kaynak"].iat[0]
        lst = list(paths[c["adaylar"]])
        rank[c["sorgu"]] = lst.index(src) + 1 if src in lst else 0
    out["kaynak_sirasi"] = out["sorgu"].map(rank).where(out["grup"] == "pozitif_kontrol")
    out["kabul"] = out["ncc"] >= WHOLE_NCC_MIN
    return out


def run_exhaustive() -> None:
    """Kapsamlı yakın çekim taramasını çalıştırır ve outputs/external/munich_kaynak_kapsamli.csv'ye yazar."""
    out_dir = stage_dir("external")
    munich = munich_hashes(list_munich())
    res = exhaustive_closeups(munich)
    res.to_csv(out_dir / "munich_kaynak_kapsamli.csv", index=False, encoding="utf-8")
    write_exclusions(munich)
    print(res.groupby("grup").agg(n=("ncc", "size"), kabul=("kabul", "sum"), ncc_medyan=("ncc", "median"),
                                  ncc_max=("ncc", "max"), kaba_medyan=("kaba_ncc", "median"),
                                  kaynak_ilk10=("kaynak_sirasi", lambda s: (s > 0).mean())).round(3).to_string())


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    """Münih ↔ Kaggle kaynak tespitini çalıştırır, sonuçları ve dışlanacak görüntü listesini yazar."""
    out_dir = stage_dir("external")
    start = time.time()
    print("1) Münih görüntüleri")
    munich = list_munich()
    print(munich.groupby("sinif").size().to_string())
    munich = munich_hashes(munich)

    print("2) Kaggle görüntü ve hücre hash'leri")
    kaggle = load_kaggle()
    hk = kaggle_hashes(kaggle)
    kcells, hkc = kaggle_cell_hashes()

    print("3) Pozitif kontrol")
    pos_path = out_dir / "munich_kaynak_pozitif_kontrol.csv"
    pos = pd.read_csv(pos_path) if pos_path.exists() else positive_control(munich)
    pos.to_csv(pos_path, index=False, encoding="utf-8")
    pc = pos[pos["duzey"] == "hucre"]
    # Hücre eşiği sabit (CELL_NCC_FLOOR); pozitiflerin kaçının bunu geçtiği ve gerçek adayların NCC dağılımı
    # birlikte raporlanır (eşiği pozitiflere göre düşürmek ilgisiz benzer hücreleri de kabul ettirebilir).
    print(pos.groupby("duzey").agg(n=("ncc", "size"), ilk3=("kaynak_sirasi", lambda s: (s > 0).mean()),
                                   ncc_medyan=("ncc", "median"), ncc_min=("ncc", "min")).round(3).to_string())
    print(f"  hücre pozitiflerinden NCC ≥ {CELL_NCC_FLOOR}: {(pc['ncc'] >= CELL_NCC_FLOOR).mean():.0%}")

    print("4) A) Görüntü düzeyi: her Kaggle görüntüsü için en yakın 3 Münih görüntüsü")
    hm = np.stack(munich["h_goruntu"].to_numpy())
    whole_path = out_dir / "munich_kaynak_goruntu_adaylari.csv"
    # Aday listesi aranan Münih sınıflarına bağlı (en yakın 3, tüm Münih içinde): sınıf kümesi değişince yeniden
    signature = CACHE_DIR / f"goruntu_adaylari_{'+'.join(sorted(munich['sinif'].unique()))}.csv"
    if signature.exists():
        whole = pd.read_csv(signature)
        print("  önbellekten")
    else:
        idx, dist = top_candidates(hk, hm)
        pairs = [(kaggle["dosya_yolu"].iat[i], kaggle["tip"].iat[i], munich["goreli_yol"].iat[j], dist[i, n])
                 for i in range(len(kaggle)) for n, j in enumerate(idx[i])]
        whole = run_parallel(whole_chunk, pairs, "görüntü doğrulama")
        whole = whole.merge(kaggle[["dosya_yolu", "etiket", "tip"]].rename(columns={"dosya_yolu": "kaggle"}), on="kaggle")
        whole.to_csv(signature, index=False, encoding="utf-8")
    whole.to_csv(whole_path, index=False, encoding="utf-8")
    print(f"  hash ≤ {HASH_CANDIDATE_DISTANCE}: {(whole['hash_mesafesi'] <= HASH_CANDIDATE_DISTANCE).sum()} aday; "
          f"NCC en yüksek {whole['ncc'].max():.3f}; ORB inlier en yüksek {whole['orb_inlier'].max()}; "
          f"kabul: {whole['kabul'].sum()}")

    print("5) B) Hücre düzeyi: her Münih merkez hücresi için en yakın 3 Kaggle hücresi")
    cell_path = out_dir / "munich_kaynak_hucre_adaylari.csv"
    with_cell = munich[munich["hucre_var"]].reset_index(drop=True)
    # Hücre adayları her Münih görüntüsü için bağımsız: önbellekte olmayan görüntüler eklenir (ör. sonradan MYO)
    cell_cache = CACHE_DIR / "hucre_adaylari.csv"
    cell = pd.read_csv(cell_cache) if cell_cache.exists() else pd.DataFrame(columns=["munich"])
    todo = with_cell[~with_cell["goreli_yol"].isin(cell["munich"])].reset_index(drop=True)
    if len(todo):
        idx, dist = top_candidates(np.stack(todo["h_hucre"].to_numpy()), hkc)
        items = [(todo["goreli_yol"].iat[i], todo["eritrosit_capi"].iat[i],
                  [(kcells["kirpim_yolu"].iat[j], kcells["eritrosit_capi"].iat[j], dist[i, n])
                   for n, j in enumerate(idx[i])]) for i in range(len(todo))]
        new = run_parallel(cell_chunk, items, "hücre doğrulama", chunk=25)
        new = new.merge(kcells.rename(columns={"kirpim_yolu": "kaggle_kirpim"}), on="kaggle_kirpim")
        cell = pd.concat([cell, new], ignore_index=True) if len(cell) else new
        cell.to_csv(cell_cache, index=False, encoding="utf-8")
    else:
        print("  önbellekten")
    cell = cell[cell["munich"].isin(with_cell["goreli_yol"])]
    cell.to_csv(cell_path, index=False, encoding="utf-8")
    cell["kabul"] = cell["ncc"] >= CELL_NCC_FLOOR
    best = cell.sort_values("ncc", ascending=False).drop_duplicates("munich")
    print("  Münih hücresi başına en yüksek NCC dağılımı:")
    print(best["ncc"].describe(percentiles=[0.5, 0.9, 0.99]).round(3).to_string())

    write_exclusions(munich)
    print(f"(süre {time.time() - start:.0f} sn)")


def cell_negative_control() -> pd.DataFrame:
    """
    Hücre düzeyi kuralın negatif kontrolü: en iyi Kaggle hücre eşleşmesi kaynağı ALL-IDB olan bir Kaggle görüntüsünden
    gelen Münih hücreleri kesin yanlış pozitiftir (ALL-IDB, Münih'ten bağımsız bir İtalyan veri setidir).

    Döndürür:
        Kaggle kaynağı (ALL-IDB / bilinmiyor) başına {n, ncc ≥ eşik oranı, %99 NCC, en yüksek NCC}.
    """
    from src.classifier import load_split

    cell = pd.read_csv(stage_dir("external") / "munich_kaynak_hucre_adaylari.csv")
    split = load_split()[["dosya_yolu", "kaynak"]].rename(columns={"dosya_yolu": "kaynak_dosya"})
    best = cell.sort_values("ncc", ascending=False).drop_duplicates("munich").merge(split, on="kaynak_dosya", how="left")
    return best.groupby("kaynak").agg(n=("ncc", "size"), esik_ustu_orani=("ncc", lambda s: (s >= CELL_NCC_FLOOR).mean()),
                                      ncc_p99=("ncc", lambda s: s.quantile(0.99)), ncc_max=("ncc", "max")).reset_index()


def write_exclusions(munich: pd.DataFrame) -> pd.DataFrame:
    """
    Dışlanacak Münih görüntülerini ve özet tabloyu yazar.

    Kural (ihtiyatlı): görüntü düzeyinde (A: pHash adayları, C: kapsamlı yakın çekim taraması) kabul edilen her
    eşleşmenin Münih görüntüsü dışlanır; görsel incelemede hiçbiri aynı hücre değildi, ama dışlamak zararsızdır
    (sınıf başına binlerce görüntü). Hücre düzeyi (B) eşleşmeler dışlanmaz, yalnızca raporlanır: Kaggle yayma
    hücreleri ~10 px eritrosit çapında; bu çözünürlükte farklı yuvarlak çekirdekler de NCC ≥ 0.95 veriyor (negatif
    kontrol: kaynağı ALL-IDB olan, Münih'te olamayacak Kaggle hücrelerinde oran daha da yüksek). B'yi dışlamak ayrıca
    Kaggle hücrelerine en çok benzeyen Münih hücrelerini çıkarıp dış doğrulamayı kolaylaştırırdı (seçim yanlılığı).

    Parametreler:
        munich: munich_hashes çıktısı.

    Döndürür:
        Dışlanan görüntüler tablosu.
    """
    out_dir = stage_dir("external")
    whole = pd.read_csv(out_dir / "munich_kaynak_goruntu_adaylari.csv")
    cell = pd.read_csv(out_dir / "munich_kaynak_hucre_adaylari.csv")
    parts = [whole[whole["kabul"]].assign(duzey="goruntu_A", kaynak=lambda d: d["kaggle"])[["munich", "duzey", "kaynak", "ncc"]]]
    ex_path = out_dir / "munich_kaynak_kapsamli.csv"
    ex = pd.read_csv(ex_path) if ex_path.exists() else pd.DataFrame(columns=["grup", "kabul"])
    real = ex[(ex["grup"] != "pozitif_kontrol") & ex["kabul"].astype(bool)]
    parts.append(real.assign(duzey="goruntu_C_" + real["grup"], kaynak=real["sorgu"])
                 .rename(columns={"en_iyi_munich": "munich"})[["munich", "duzey", "kaynak", "ncc"]])
    excluded = pd.concat(parts).drop_duplicates("munich")
    excluded["gorsel_inceleme"] = "farklı hücre"   # Tümü elle incelendi (Adım 8.0)
    excluded.to_csv(out_dir / "munich_dislanan.csv", index=False, encoding="utf-8")

    cls_of = lambda s: s.str.split("/").str[0]  # noqa: E731
    best = cell.sort_values("ncc", ascending=False).drop_duplicates("munich")
    summary = munich.groupby("sinif").agg(goruntu=("goreli_yol", "size"), hucre_bulunan=("hucre_var", "sum"),
                                          eritrosit_capi_medyan=("eritrosit_capi", "median"))
    summary["eslesen_A_goruntu"] = whole[whole["kabul"]].groupby(cls_of(whole["munich"]))["munich"].nunique() \
        .reindex(summary.index, fill_value=0)
    summary["eslesen_C_kapsamli"] = real.groupby(cls_of(real["en_iyi_munich"]))["en_iyi_munich"].nunique() \
        .reindex(summary.index, fill_value=0)
    summary["hucre_B_esik_ustu_raporlanan"] = best[best["ncc"] >= CELL_NCC_FLOOR].groupby(cls_of(best["munich"])).size() \
        .reindex(summary.index, fill_value=0)
    summary["en_yuksek_ncc_A"] = whole.groupby(cls_of(whole["munich"]))["ncc"].max().reindex(summary.index)
    summary["en_yuksek_ncc_B"] = best.groupby(cls_of(best["munich"]))["ncc"].max().reindex(summary.index)
    summary["dislanan"] = excluded.groupby(cls_of(excluded["munich"])).size().reindex(summary.index, fill_value=0)
    summary.reset_index().to_csv(out_dir / "munich_kaynak_ozet.csv", index=False, encoding="utf-8")
    neg = cell_negative_control()
    neg.to_csv(out_dir / "munich_kaynak_hucre_negatif_kontrol.csv", index=False, encoding="utf-8")
    print("\n=== Özet ===\n" + summary.round(3).to_string())
    print("Hücre düzeyi negatif kontrol:\n" + neg.round(3).to_string(index=False))
    print(f"Dışlanan Münih görüntüsü: {len(excluded)}")
    return excluded


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["kapsamli"]:
        run_exhaustive()
    elif sys.argv[1:] == ["dislanan"]:
        write_exclusions(munich_hashes(list_munich()))
    else:
        main()
