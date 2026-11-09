"""
Adım 10 — LeukemiaAttri ile laboratuvardan bağımsız ALL/AML ayrımı.

LeukemiaAttri (Rehman ve ark., MICCAI 2024; arXiv 2504.02602) H_100X_C1 alt kümesi: 47 hasta, 2368 görüntü (640×640,
1000×), COCO kutuları ve 7 morfolojik öznitelik etiketi. Adım 9'da ALL ile AML'nin laboratuvarlar arasında
ayrılamadığı görüldü (Kaggle'da ALL ve AML farklı kaynaklardan geldiği için model laboratuvar imzasını öğreniyordu).
LeukemiaAttri'de lenfoblast, miyeloblast ve normal lökositler AYNI laboratuvardan, aynı hastaların yaymalarından
geliyor: sınıf ile laboratuvar birbirinden ayrılıyor.

Hücreler segmentasyonla değil, uzman etiket kutularından alınır (mevcut segmentasyon bu sette mavi eritrositleri
çekirdek sanıyor); çekirdek kutunun İÇİNDE K-means (Otsu geri dönüşlü) ile ayrılır. Öznitelikler mevcut 58 + 15
biyolojik öznitelik (Adım 4 ve 9 ile aynı kod).

Bölümler (python -m src.leukemiaattri <bölüm>):
    hucre        Görüntü indeksi, eritrosit çapı, hücre öznitelikleri → data/processed/leukemiaattri/ (git dışı)
    kontrol      Sınıf başına kırpım + çekirdek maskesi figürü → data/processed/leukemiaattri/ (görüntü içerir, git dışı)
    baslangic    Mevcut model (v2_kalibre) ve v3 → LA TEST (dis_test_baslangic.csv; bir kez yazılır, değişmez)
    kural        Ön kayıtlı seçim kuralı (deneylerden ÖNCE; üzerine yazılmaz)
    deney        Yapılandırmalar × (LOSO A, LOSO B, LA-CV, Kaggle) → yapilandirma_sonuclari.csv
    laboratuvar  Laboratuvar kısayolu testi (aynı sınıf içinde kaynak AUC'si) yapılandırma başına
    son          Seçilen yapılandırma → LA TEST (BİR KEZ) + bootstrap + v4 (varsayılan DEĞİŞMEZ)
    gorseller    outputs/leukemiaattri/ figürleri (görüntü içermez)

Lisans belirsiz: LeukemiaAttri görüntüsü içeren hiçbir dosya outputs/ altına yazılmaz.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import (
    CELL_MARGIN_RATIO,
    CLASS_NAMES,
    KMEANS_K,
    KMEANS_N_INIT,
    LEUKEMIAATTRI_ASPECT,
    LEUKEMIAATTRI_ATTRIBUTES,
    LEUKEMIAATTRI_DIR,
    LEUKEMIAATTRI_EXTRA_AML,
    LEUKEMIAATTRI_HEALTHY,
    LEUKEMIAATTRI_MAIN_LABELS,
    LEUKEMIAATTRI_PARAMS,
    LEUKEMIAATTRI_PROCESSED_DIR,
    LEUKEMIAATTRI_RBC_TOLERANCE,
    N_JOBS,
    RANDOM_STATE,
)
from src.utils import stage_dir

LA_SOURCE = "LeukemiaAttri"
OUT_STAGE = "leukemiaattri"
PROGRESS_EVERY = 100
CHUNK = 20
IMAGES_CSV = "la_goruntu.csv"
CELLS_CSV = "la_hucre.csv"


def out_dir() -> Path:
    """outputs/leukemiaattri/ klasörünü döndürür (yoksa oluşturur)."""
    return stage_dir(OUT_STAGE)


def proc_dir() -> Path:
    """data/processed/leukemiaattri/ klasörünü döndürür (yoksa oluşturur; git dışı)."""
    LEUKEMIAATTRI_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return LEUKEMIAATTRI_PROCESSED_DIR


# ---------------------------------------------------------------------------
# 1) Görüntü ve etiket indeksi
# ---------------------------------------------------------------------------

def load_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    COCO JSON etiketlerini okur. Görüntü listesi Images/ klasöründen alınır; txt_labels altındaki "(1)" kopyaları
    hiç okunmaz (yalnızca JSON kullanılır; txt dosyaları JSON ile birebir aynı).

    train.json her hücre için ayrı bir "image" kaydı tutuyor (7894 kayıt, 1676 benzersiz ad): kutular dosya adına
    göre birleştirilir.

    Döndürür:
        (görüntüler: id, bolum, dosya, hasta, tani; hücreler: id, kutu_id, hucre_tipi, x, y, w, h + 7 öznitelik).
    """
    images, cells = [], []
    for part in ("train", "test"):
        files = sorted(p.name for p in (LEUKEMIAATTRI_DIR / "Images" / part).glob("*.png"))
        for f in files:
            patient, _, _, diag = f[:-4].split("_")
            images.append({"id": f"{part}/{f}", "bolum": part, "dosya": f, "hasta": int(patient), "tani": diag})
        data = json.loads((LEUKEMIAATTRI_DIR / "json_labels" / f"{part}.json").read_text(encoding="utf-8"))
        names = {i["id"]: i["file_name"] for i in data["images"]}
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            cells.append({"id": f"{part}/{names[a['image_id']]}", "kutu_id": f"{part}:{a['id']}",
                          "hucre_tipi": a["category_name"], "x": x, "y": y, "w": w, "h": h,
                          **{k: a[k] for k in LEUKEMIAATTRI_ATTRIBUTES}})
    images, cells = pd.DataFrame(images), pd.DataFrame(cells)
    missing = set(cells["id"]) - set(images["id"])
    if missing:
        raise ValueError(f"Klasörde olmayan görüntüye ait etiket: {sorted(missing)[:5]}")
    cells = cells.merge(images[["id", "bolum", "hasta", "tani"]], on="id", how="left", validate="many_to_one")
    return images, cells.assign(**label_columns(cells))


def label_columns(cells: pd.DataFrame) -> dict:
    """
    Hücre tipi + hasta tanısından proje etiketleri (gerekçe: config.LEUKEMIAATTRI_MAIN_LABELS).

    Parametreler:
        cells: hucre_tipi, tani sütunlu tablo.

    Döndürür:
        {"etiket": ana tanım ("" = dışarıda), "etiket_ek": ek AML varyantı dahil, "alt_tip": Healthy alt tipi}.
    """
    main, extra, sub = [], [], []
    for t, d in zip(cells["hucre_tipi"], cells["tani"]):
        lab = LEUKEMIAATTRI_MAIN_LABELS.get((d, t), "Healthy" if t in LEUKEMIAATTRI_HEALTHY else "")
        main.append(lab)
        extra.append("AML" if lab == "" and t in LEUKEMIAATTRI_EXTRA_AML and d in ("AML", "APML") else lab)
        sub.append(t if lab == "Healthy" else "")
    return {"etiket": main, "etiket_ek": extra, "alt_tip": sub}


# ---------------------------------------------------------------------------
# 2) Geometri, eritrosit çapı, kutu içinde çekirdek
# ---------------------------------------------------------------------------

def load_corrected(image_id: str) -> np.ndarray:
    """
    Görüntüyü okur ve en-boy düzeltmesini uygular (yükseklik / LEUKEMIAATTRI_ASPECT; gerekçe config'de).

    Parametreler:
        image_id: "train/<dosya>" biçiminde kimlik.

    Döndürür:
        Düzeltilmiş RGB görüntü.
    """
    from src.utils import load_image

    rgb = load_image(LEUKEMIAATTRI_DIR / "Images" / image_id)
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w, round(h / LEUKEMIAATTRI_ASPECT)), interpolation=cv2.INTER_AREA)


def to_processed_box(box: tuple, original_shape: tuple, steps: dict) -> tuple[float, float, float, float]:
    """
    Orijinal (640×640, gerilmiş) koordinatlardaki kutuyu ön işlenmiş (256 px, dolgulu) görüntüye taşır.

    Parametreler:
        box: (x, y, w, h) orijinal piksel.
        original_shape: Orijinal görüntü (H, W).
        steps: preprocess ara çıktıları (düzeltilmiş görüntünün).

    Döndürür:
        (x0, y0, x1, y1) 256 px uzayında (ondalıklı).
    """
    corrected = steps["orijinal"].shape[:2]
    size = steps["boyutlandirilmis"].shape[0]
    s = size / max(corrected)
    new_h, new_w = round(corrected[0] * s), round(corrected[1] * s)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    sy = corrected[0] / original_shape[0] * s
    sx = corrected[1] / original_shape[1] * s
    x, y, w, h = box
    return left + x * sx, top + y * sy, left + (x + w) * sx, top + (y + h) * sy


def image_rbc(image_id: str) -> dict:
    """
    Bir görüntünün eritrosit çapı tahmini (ön işlenmiş 256 px uzayında; segmentation.estimate_rbc_diameter).

    Parametreler:
        image_id: Görüntü kimliği.

    Döndürür:
        {id, cap_tahmin, guvenilir, bilesen_sayisi, kapsama}.
    """
    from src.preprocessing import preprocess
    from src.segmentation import estimate_rbc_diameter

    steps = preprocess(load_corrected(image_id))[1]
    est = estimate_rbc_diameter(steps["renk_normalize"], steps["gecerli_maske"])
    return {"id": image_id, "cap_tahmin": est["cap"], "guvenilir": bool(est["guvenilir"]),
            "bilesen_sayisi": est["bilesen_sayisi"], "kapsama": est["kapsama"]}


def resolve_diameters(images: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Kullanılacak eritrosit çapı: görüntü tahmini güvenilir ve TRAIN medyanına ±LEUKEMIAATTRI_RBC_TOLERANCE yakınsa
    kendisi, değilse train medyanı (test görüntüleri medyana katılmaz; gerekçe config'de).

    Parametreler:
        images: image_rbc sütunlarıyla görüntü tablosu.

    Döndürür:
        (images + cap, cap_kaynagi; train medyanı).
    """
    ok = images["guvenilir"].astype(bool) & images["cap_tahmin"].notna()
    median = float(images.loc[ok & (images["bolum"] == "train"), "cap_tahmin"].median())
    near = ok & (images["cap_tahmin"] - median).abs().le(LEUKEMIAATTRI_RBC_TOLERANCE * median)
    return images.assign(cap=np.where(near, images["cap_tahmin"], median).round(2),
                         cap_kaynagi=np.where(near, "goruntu", "medyan")), median


def box_kmeans(image: np.ndarray, valid: np.ndarray, inv_b: np.ndarray):
    """
    Kutu piksellerinde K-means (k = KMEANS_K, Lab): çekirdek kümesi b* merkezi en düşük (en mavi/mor) küme; Adım 2'deki
    segment_kmeans ile aynı kural ve kontrast tabanı (segmentation.contrast_floor), yalnızca kutunun içine uygulanır.

    Parametreler:
        image: Kutu bölgesi (renk normalize RGB).
        valid: Geçerli piksel maskesi.
        inv_b: Ters b* (preprocess "filtreli") kutu bölgesi.

    Döndürür:
        (ham çekirdek maskesi, eğitilmiş model ve çekirdek küme indeksi) veya (boş maske, None).
    """
    from sklearn.cluster import KMeans

    from src.segmentation import contrast_floor, lab_features

    pixels = lab_features(image)[valid]
    if len(pixels) < 3 * KMEANS_K:
        return np.zeros(valid.shape, bool), None
    model = KMeans(n_clusters=KMEANS_K, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE).fit(pixels)
    nuc = int(np.argmin(model.cluster_centers_[:, 2]))
    raw = np.zeros(valid.shape, bool)
    raw[valid] = model.labels_ == nuc
    return raw & contrast_floor(inv_b), (model, nuc)


def box_otsu(valid: np.ndarray, inv_b: np.ndarray) -> np.ndarray:
    """
    Geri dönüş: kutu içinde ters b* üzerinde Otsu eşiği (+ kontrast tabanı).

    Parametreler:
        valid: Geçerli piksel maskesi.
        inv_b: Ters b* kutu bölgesi.

    Döndürür:
        Ham çekirdek maskesi.
    """
    from src.segmentation import contrast_floor

    if valid.sum() < 10:
        return np.zeros(valid.shape, bool)
    t, _ = cv2.threshold(inv_b[valid].reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (inv_b > t) & valid & contrast_floor(inv_b)


def pick_nucleus(mask: np.ndarray) -> np.ndarray:
    """
    Kutudaki çekirdek bileşenlerinden hücrenin kendi çekirdeğini seçer: en büyük bileşen + ağırlık merkezi kutunun
    orta %60'ında kalan diğer bileşenler (nötrofil lobları bağlı olmayabilir). Kutunun kenarına sızan komşu
    hücre çekirdekleri (kutular örtüşebiliyor) bu yüzden dışarıda kalır.

    Parametreler:
        mask: Son işlenmiş çekirdek maskesi (kutu bölgesi).

    Döndürür:
        Hücrenin çekirdek maskesi (boş olabilir).
    """
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = [largest] + [k for k in range(1, n) if k != largest
                        and 0.2 * w <= cents[k, 0] <= 0.8 * w and 0.2 * h <= cents[k, 1] <= 0.8 * h]
    return np.isin(labels, keep)


def cell_crop(steps: dict, box256: tuple, diameter: float) -> dict | None:
    """
    Bir etiket kutusundan hücre kırpımı ve kodlu maske (segmentation.extract_cells biçiminde).

    Çekirdek kutu içinde ayrılır (K-means, boşsa Otsu); kırpım penceresi çekirdeğin sınır kutusu + CELL_MARGIN_RATIO ×
    eritrosit çapı (extract_cells ile aynı). Penceredeki diğer çekirdek pikselleri (aynı K-means modeliyle)
    MASK_OTHER_NUCLEUS, dolgu MASK_PADDING olarak kodlanır.

    Parametreler:
        steps: preprocess ara çıktıları.
        box256: (x0, y0, x1, y1) 256 px uzayında.
        diameter: Eritrosit çapı (px, 256 uzayı).

    Döndürür:
        {kirpim, maske, kutu, alan, kutu_alani, yontem, kenara_degiyor} veya çekirdek bulunamazsa None.
    """
    from src.segmentation import MASK_NUCLEUS, MASK_OTHER_NUCLEUS, MASK_PADDING, contrast_floor, lab_features, postprocess

    image, valid, inv_b = steps["renk_normalize"], steps["gecerli_maske"], steps["filtreli"]
    H, W = valid.shape
    x0, y0 = max(0, int(np.floor(box256[0]))), max(0, int(np.floor(box256[1])))
    x1, y1 = min(W, int(np.ceil(box256[2]))), min(H, int(np.ceil(box256[3])))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    sl = (slice(y0, y1), slice(x0, x1))
    raw, fitted = box_kmeans(image[sl], valid[sl], inv_b[sl])
    method = "kutu_kmeans"
    nucleus = pick_nucleus(postprocess(raw, valid[sl], diameter))
    if not nucleus.any():
        nucleus = pick_nucleus(postprocess(box_otsu(valid[sl], inv_b[sl]), valid[sl], diameter))
        method, fitted = "kutu_otsu", None
    if not nucleus.any():
        return None
    full = np.zeros((H, W), bool)
    full[sl] = nucleus
    ys, xs = np.nonzero(full)
    margin = max(2, round(CELL_MARGIN_RATIO * diameter))
    wy0, wy1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    wx0, wx1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    win = (slice(wy0, wy1), slice(wx0, wx1))
    if fitted is not None:
        model, nuc = fitted
        other = model.predict(lab_features(image[win]).reshape(-1, 3)).reshape(full[win].shape) == nuc
    else:
        t, _ = cv2.threshold(inv_b[win][valid[win]].reshape(-1, 1).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        other = inv_b[win] > t
    other &= contrast_floor(inv_b[win]) & valid[win] & ~full[win]
    coded = np.zeros(full[win].shape, np.uint8)
    coded[~valid[win]] = MASK_PADDING
    coded[other] = MASK_OTHER_NUCLEUS
    coded[full[win]] = MASK_NUCLEUS
    edge = np.zeros((H, W), bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    touching = cv2.dilate((edge | ~valid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return {"kirpim": image[win].copy(), "maske": coded, "kutu": (wx0, wy0, wx1 - wx0, wy1 - wy0),
            "alan": int(full.sum()), "kutu_alani": float((box256[2] - box256[0]) * (box256[3] - box256[1])),
            "yontem": method, "kenara_degiyor": bool(full[touching].any())}


def image_cells(image_id: str, boxes: pd.DataFrame, diameter: float, keep_crops: bool = False) -> list[dict]:
    """
    Bir görüntünün tüm etiket kutularından hücre öznitelikleri (58 + 15 biyolojik).

    Parametreler:
        image_id: Görüntü kimliği.
        boxes: Bu görüntünün kutuları (kutu_id, x, y, w, h).
        diameter: Eritrosit çapı (256 px uzayı).
        keep_crops: True ise kırpım ve maske de döndürülür (kontrol figürü için).

    Döndürür:
        Hücre satırları (çekirdek bulunamayan kutu için yalnızca kutu_id + meta_cekirdek_yok=True).
    """
    from src.bio_features import bio_features
    from src.features import crop_features
    from src.preprocessing import preprocess
    from src.utils import load_image

    original = load_image(LEUKEMIAATTRI_DIR / "Images" / image_id).shape[:2]
    steps = preprocess(load_corrected(image_id))[1]
    rows = []
    for b in boxes.itertuples(index=False):
        box256 = to_processed_box((b.x, b.y, b.w, b.h), original, steps)
        c = cell_crop(steps, box256, diameter)
        if c is None:
            rows.append({"kutu_id": b.kutu_id, "meta_cekirdek_yok": True})
            continue
        cut = b.x <= 0 or b.y <= 0 or b.x + b.w >= original[1] - 1 or b.y + b.h >= original[0] - 1
        x, y, w, h = c["kutu"]
        row = {"kutu_id": b.kutu_id, "meta_cekirdek_yok": False,
               **crop_features(c["kirpim"], c["maske"], diameter), **bio_features(c["kirpim"], c["maske"], diameter),
               "meta_kenara_degiyor": bool(c["kenara_degiyor"] or cut), "meta_kutu_kesik": bool(cut),
               "meta_alan": c["alan"], "meta_cekirdek_kutu_orani": c["alan"] / max(c["kutu_alani"], 1.0),
               "meta_merkez_uzaklik": float(np.hypot(x + w / 2 - 128, y + h / 2 - 128)),
               "meta_eritrosit_capi": diameter, "meta_yontem": c["yontem"], "meta_geri_donus": c["yontem"] != "kutu_kmeans"}
        if keep_crops:
            row.update({"kirpim": c["kirpim"], "maske": c["maske"]})
        rows.append(row)
    return rows


def cells_chunk(items: list[tuple[str, pd.DataFrame, float]]) -> list[dict]:
    """
    Bir parça görüntüyü işler (paralel işçide; görüntüler tek tek okunur).

    Parametreler:
        items: (görüntü kimliği, kutular, eritrosit çapı) listesi.

    Döndürür:
        Hücre satırları.
    """
    rows = []
    for image_id, boxes, d in items:
        rows += image_cells(image_id, boxes, d)
    return rows


def run_parallel(func, items: list, tag: str) -> list:
    """
    Görüntü listesini CHUNK'lık parçalar hâlinde paralel işler (n_jobs = N_JOBS) ve ilerleme yazdırır.

    Parametreler:
        func: Parça işleyen fonksiyon (liste döndürür).
        items: İşlenecek öğeler.
        tag: İlerleme etiketi.

    Döndürür:
        Birleştirilmiş sonuç listesi.
    """
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    out, done, start = [], 0, time.time()
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for c, res in zip(chunks, par(delayed(func)(c) for c in chunks)):
            out += res
            done += len(c)
            if done % PROGRESS_EVERY < CHUNK or done == len(items):
                print(f"  [{tag}] {done}/{len(items)} görüntü ({time.time() - start:.0f} sn)", flush=True)
    return out


def rbc_chunk(ids: list[str]) -> list[dict]:
    """Bir parça görüntünün eritrosit çapı tahmini (paralel işçide)."""
    return [image_rbc(i) for i in ids]


def run_diameters() -> pd.DataFrame:
    """
    Görüntü indeksi + eritrosit çapı → data/processed/leukemiaattri/la_goruntu.csv (varsa okunur).

    Döndürür:
        Görüntü tablosu (cap = kullanılacak çap, cap_kaynagi = goruntu / medyan).
    """
    path = proc_dir() / IMAGES_CSV
    if path.exists():
        return pd.read_csv(path)
    images, boxes = load_annotations()
    print(f"Eritrosit çapı: {len(images)} görüntü", flush=True)
    rbc = pd.DataFrame(run_parallel(rbc_chunk, list(images["id"]), "eritrosit"))
    images, median = resolve_diameters(images.merge(rbc, on="id"))
    images["kutu_sayisi"] = images["id"].map(boxes.groupby("id").size()).fillna(0).astype(int)
    images.to_csv(path, index=False, encoding="utf-8")
    print(f"Train medyanı {median:.2f} px; görüntü tahmini kullanılan: {(images['cap_kaynagi'] == 'goruntu').mean():.1%}; "
          f"güvenilir: {images['guvenilir'].mean():.1%}", flush=True)
    return images


def run_cells() -> pd.DataFrame:
    """
    Tüm etiket kutularının öznitelikleri → data/processed/leukemiaattri/la_hucre.csv (git dışı; varsa okunur).
    Önce run_diameters (eritrosit çapı) çalışır.

    Döndürür:
        Hücre tablosu.
    """
    images = run_diameters()
    _, boxes = load_annotations()
    cell_path = proc_dir() / CELLS_CSV
    if cell_path.exists():
        return pd.read_csv(cell_path, keep_default_na=False, na_values=[""])
    diam = images.set_index("id")["cap"]
    items = [(i, g[["kutu_id", "x", "y", "w", "h"]], float(diam[i])) for i, g in boxes.groupby("id", sort=True)]
    print(f"Hücre öznitelikleri: {len(boxes)} kutu, {len(items)} görüntü", flush=True)
    feats = pd.DataFrame(run_parallel(cells_chunk, items, "hücre"))
    cells = boxes.merge(feats, on="kutu_id", how="left", validate="one_to_one")
    cells.to_csv(cell_path, index=False, encoding="utf-8")
    print(f"Çekirdek bulunamayan kutu: {int(cells['meta_cekirdek_yok'].astype(bool).sum())}; Otsu geri dönüşü: "
          f"{int((cells['meta_yontem'] == 'kutu_otsu').sum())}", flush=True)
    summary = cells.groupby(["bolum", "hucre_tipi"]).agg(
        kutu=("kutu_id", "size"), cekirdek_yok=("meta_cekirdek_yok", lambda s: int(s.astype(bool).sum())),
        otsu=("meta_yontem", lambda s: int((s == "kutu_otsu").sum())),
        cekirdek_kutu_orani=("meta_cekirdek_kutu_orani", "median")).reset_index()
    summary.to_csv(out_dir() / "hucre_cikarma_ozeti.csv", index=False, encoding="utf-8")
    return cells


# ---------------------------------------------------------------------------
# 3) Maske kontrol figürü (görüntü içerir: data/processed/leukemiaattri/, git dışı)
# ---------------------------------------------------------------------------

def run_mask_check(per_class: int = 6) -> None:
    """
    Her hücre tipinden rastgele kırpımlar ve çekirdek maskesi konturları → data/processed/leukemiaattri/
    kontrol_maskeler.png (git dışı). Maske doğruluğu gözle kontrol edilir.

    Parametreler:
        per_class: Hücre tipi başına örnek.
    """
    from src.segmentation import MASK_NUCLEUS, MASK_OTHER_NUCLEUS

    images, boxes = load_annotations()
    diam = pd.read_csv(proc_dir() / IMAGES_CSV).set_index("id")["cap"]
    boxes = boxes[boxes["bolum"] == "train"]   # Test görüntüleri kontrol figürüne de girmez
    rng = np.random.default_rng(RANDOM_STATE)
    types = sorted(boxes["hucre_tipi"].unique())
    fig, axes = plt.subplots(len(types), per_class, figsize=(1.7 * per_class, 1.8 * len(types)))
    for i, t in enumerate(types):
        sub = boxes[boxes["hucre_tipi"] == t]
        pick = sub.iloc[rng.choice(len(sub), min(per_class, len(sub)), replace=False)]
        for j in range(per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(pick):
                continue
            b = pick.iloc[[j]]
            row = image_cells(b["id"].iat[0], b, float(diam[b["id"].iat[0]]), keep_crops=True)[0]
            if row["meta_cekirdek_yok"]:
                ax.set_title("çekirdek yok", fontsize=6)
                continue
            ax.imshow(row["kirpim"])
            ax.contour(row["maske"] == MASK_NUCLEUS, levels=[0.5], colors="lime", linewidths=0.8)
            if (row["maske"] == MASK_OTHER_NUCLEUS).any():
                ax.contour(row["maske"] == MASK_OTHER_NUCLEUS, levels=[0.5], colors="red", linewidths=0.5)
            ax.set_title((f"{t}\n" if j == 0 else "\n") + f"{row['meta_yontem'][5:]} {row['meta_cekirdek_kutu_orani']:.2f}",
                         fontsize=6, loc="left")
    fig.suptitle("LeukemiaAttri: kutu içi çekirdek maskesi (yeşil: hücre çekirdeği, kırmızı: komşu çekirdek)", fontsize=9)
    fig.tight_layout()
    path = proc_dir() / "kontrol_maskeler.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Kaydedildi: {path}")


# ---------------------------------------------------------------------------
# 4) Birleşik tablo (Kaggle + ALL-IDB + Münih + LeukemiaAttri)
# ---------------------------------------------------------------------------

def la_cells() -> pd.DataFrame:
    """
    LeukemiaAttri hücre tablosu, Adım 9 birleşik tablosuyla uyumlu sütunlarla (çekirdeği bulunamayan kutular hariç).

    Döndürür:
        id ("LA/<bölüm>/<dosya>"), kaynak, grup ("la:<hasta>"), la_bolum, hasta, tani, hucre_tipi, etiket, etiket_ek,
        alt_tip, etiketli, 7 öznitelik etiketi, öznitelikler ve meta sütunları.
    """
    cells = pd.read_csv(proc_dir() / CELLS_CSV, keep_default_na=False, na_values=[""])
    cells = cells[~cells["meta_cekirdek_yok"].astype(str).str.lower().eq("true")].copy()
    for c in ("etiket", "etiket_ek", "alt_tip"):
        cells[c] = cells[c].fillna("").astype(str)
    cells["meta_kenara_degiyor"] = cells["meta_kenara_degiyor"].astype(str).str.lower().eq("true")
    cells["meta_geri_donus"] = cells["meta_geri_donus"].astype(str).str.lower().eq("true")
    return cells.rename(columns={"bolum": "la_bolum"}).assign(
        id="LA/" + cells["id"], kaynak=LA_SOURCE, grup="la:" + cells["hasta"].astype(str),
        etiketli=cells["etiket"] != "", merkez=False)


def la_images(cells: pd.DataFrame) -> pd.DataFrame:
    """
    LeukemiaAttri görüntü tablosu (risk eşiği seçimi için): görüntü etiketi hastanın tanısı (ALL; AML ve APML → AML;
    CLL / CML → "" , projenin sınıfı değil).

    Parametreler:
        cells: la_cells çıktısı.

    Döndürür:
        id, kaynak, etiket, hasta, tani, la_bolum, grup.
    """
    img = cells.groupby("id").agg(hasta=("hasta", "first"), tani=("tani", "first"), la_bolum=("la_bolum", "first"),
                                  grup=("grup", "first")).reset_index()
    return img.assign(kaynak=LA_SOURCE, etiket=img["tani"].map({"ALL": "ALL", "AML": "AML", "APML": "AML"}).fillna(""))


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dört kaynağın birleşik hücre ve görüntü tabloları (Adım 9 önbellekleri + LeukemiaAttri).

    Döndürür:
        (hücreler, görüntüler). LA dışı hücrelerde etiket_ek = etiket, alt_tip = "".
    """
    from src.multicenter import build_tables

    cells3, images3 = build_tables()
    la = la_cells()
    cells3 = cells3.assign(etiket_ek=cells3["etiket"], alt_tip="", la_bolum="", hasta=-1, tani="")
    cells = pd.concat([cells3, la], ignore_index=True, sort=False)
    for c in ("etiket", "etiket_ek", "alt_tip", "la_bolum", "tani"):
        cells[c] = cells[c].fillna("").astype(str)
    cells["etiketli"] = cells["etiketli"].astype(bool)
    cells["meta_kenara_degiyor"] = cells["meta_kenara_degiyor"].astype(bool)
    images = pd.concat([images3, la_images(la)], ignore_index=True, sort=False)
    return cells, images


def feature_lists(cells: pd.DataFrame) -> dict[str, list[str]]:
    """
    Öznitelik kümeleri: "bio" (58 + 15), "renksiz" (şekil + doku + biyolojik; hiçbir renk_ özniteliği yok),
    "renksiz_goreli" (renksiz + çekirdek − halka renk farkları renk_fark_*: mutlak boya / zemin rengini değil,
    aynı hücre içindeki kontrastı ölçer).

    Parametreler:
        cells: Birleşik hücre tablosu.

    Döndürür:
        {küme adı: sütunlar}.
    """
    from src.multicenter import feature_sets

    bio = feature_sets(cells)["bio"]
    colorless = [c for c in bio if not c.startswith("renk_")]
    return {"bio": bio, "renksiz": colorless,
            "renksiz_goreli": colorless + [c for c in bio if c.startswith("renk_fark_")]}


# ---------------------------------------------------------------------------
# 5) Yapılandırmalar, ağırlıklar, hizalama, öznitelik tahmini
# ---------------------------------------------------------------------------

CONFIGS = {
    "a_cok_merkezli": {"oznitelik": "bio", "agirlik": "sinif"},
    "b_dengeli": {"oznitelik": "bio", "agirlik": "kaynak"},
    "b_ek_aml": {"oznitelik": "bio", "agirlik": "kaynak", "ek": True},
    "c_renksiz": {"oznitelik": "renksiz", "agirlik": "kaynak"},
    "c_goreli_renk": {"oznitelik": "renksiz_goreli", "agirlik": "kaynak"},
    "d_zskor": {"oznitelik": "bio", "agirlik": "kaynak", "hizalama": "z"},
    "d_coral": {"oznitelik": "bio", "agirlik": "kaynak", "hizalama": "coral"},
    "e_oznitelik_tahmini": {"oznitelik": "bio", "agirlik": "kaynak", "attr": True},
}
CONFIG_LABELS = {
    "a_cok_merkezli": "(a) Çok merkezli, sınıf dengeli",
    "b_dengeli": "(b) + kaynak × sınıf ve Healthy alt tip dengesi",
    "b_ek_aml": "(b) + ek AML varyantı (monoblast, promonosit, anormal promiyelosit)",
    "c_renksiz": "(c) Renksiz: şekil + doku + biyolojik",
    "c_goreli_renk": "(c) Renksiz + göreli renk (çekirdek − halka)",
    "d_zskor": "(d) Kaynak başına z-skor (etiketsiz)",
    "d_coral": "(d) Kaynak başına CORAL (etiketsiz)",
    "e_oznitelik_tahmini": "(e) + 7 morfolojik öznitelik tahmini",
}
REFERENCES = {"v2_kalibre": "Mevcut model (v2_kalibre, varsayılan)", "v3": "v3 (Adım 9; LeukemiaAttri görmedi)"}
EVAL_FOLDS = ["A", "B", "C"]
ATTR_COLUMNS = [f"tahmin_{a}" for a in LEUKEMIAATTRI_ATTRIBUTES]


def target_column(cells: pd.DataFrame, config: dict) -> pd.Series:
    """
    Eğitim hedefi: ek AML varyantında LeukemiaAttri hücrelerinin etiket_ek'i, diğerlerinde etiket.

    Parametreler:
        cells: Hücre tablosu.
        config: CONFIGS değeri.

    Döndürür:
        Hedef serisi ("" = eğitime girmez).
    """
    if config.get("ek"):
        return pd.Series(np.where(cells["kaynak"] == LA_SOURCE, cells["etiket_ek"], cells["etiket"]), index=cells.index)
    return cells["etiket"]


def train_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    """
    Örnek ağırlıkları.

    - "sinif": yalnızca sınıf dengesi (her sınıfın toplam ağırlığı eşit); kaynaklar sayıları oranında katkı verir.
    - "kaynak": (1) LeukemiaAttri Healthy alt tip dengesi: nötrofillerin toplam ağırlığı lenfosit sayısına indirilir
      (config.LEUKEMIAATTRI_NEUTROPHIL_CAP_TO); (2) kaynak × sınıf tavanı (MULTICENTER_SOURCE_CAP, ağırlık toplamı
      üzerinden; Adım 9 ile aynı gerekçe); (3) sınıf toplamları eşitlenir.
    Ortalama ağırlık 1'e ölçeklenir.

    Parametreler:
        df: Eğitim hücreleri (kaynak, hedef, alt_tip).
        scheme: "sinif" veya "kaynak".

    Döndürür:
        (n,) ağırlıklar.
    """
    from src.config import LEUKEMIAATTRI_NEUTROPHIL_CAP_TO, MULTICENTER_SOURCE_CAP

    y = df["hedef"].to_numpy()
    w = np.ones(len(df))
    if scheme == "kaynak":
        la = (df["kaynak"] == LA_SOURCE).to_numpy()
        neut = la & (df["alt_tip"] == "neutrophil").to_numpy()
        ref = int((la & (df["alt_tip"] == LEUKEMIAATTRI_NEUTROPHIL_CAP_TO).to_numpy()).sum())
        if neut.sum() > ref > 0:
            w[neut] = ref / neut.sum()
        key = (df["kaynak"].astype(str) + "|" + df["hedef"].astype(str)).to_numpy()
        sums = pd.Series(w).groupby(key).sum()
        w = w * np.minimum(1.0, MULTICENTER_SOURCE_CAP / pd.Series(key).map(sums).to_numpy())
    totals = pd.Series(w).groupby(y).sum()
    w = w * (totals.mean() / pd.Series(y).map(totals).to_numpy())
    return w / w.mean()


def source_stats(pool: pd.DataFrame, cols: list[str]) -> dict[str, dict]:
    """
    Kaynak başına etiketsiz öznitelik istatistikleri (ortalama, std; CORAL için z-skor sonrası korelasyon).
    Etiket KULLANILMAZ: havuz, o kaynağın ilgili bölümdeki TÜM hücreleridir (sınıfı bilinmeyenler dahil).

    Parametreler:
        pool: Hücreler (kaynak + öznitelikler).
        cols: Öznitelikler.

    Döndürür:
        {kaynak: {"ort", "std", "beyazlatma", "n"}}.
    """
    from src.config import LEUKEMIAATTRI_CORAL_LAMBDA

    stats = {}
    for src, sub in pool.groupby("kaynak"):
        X = sub[cols].to_numpy(np.float64)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Z = np.nan_to_num((X - mu) / sd)
        C = np.cov(Z, rowvar=False) + LEUKEMIAATTRI_CORAL_LAMBDA * np.eye(len(cols))
        vals, vecs = np.linalg.eigh(C)
        stats[src] = {"ort": mu, "std": sd, "beyazlatma": vecs @ np.diag(vals ** -0.5) @ vecs.T, "n": len(sub)}
    return stats


def apply_alignment(df: pd.DataFrame, cols: list[str], stats: dict, mode: str, source_map: dict | None = None) -> pd.DataFrame:
    """
    Kaynak başına hizalama: "z" → (x − ort) / std (eksikler korunur); "coral" → z-skor, eksikler 0, sonra kaynağın
    beyazlatma matrisi (Σ + λI)^(−1/2): her kaynağın kovaryansı ortak birim kovaryansa taşınır (CORAL'ın çok kaynaklı,
    ortak hedefli biçimi).

    Parametreler:
        df: Hücreler (kaynak sütunu).
        cols: Öznitelikler.
        stats: source_stats çıktısı.
        mode: "z" veya "coral".
        source_map: {hücrenin kaynağı: kullanılacak istatistik anahtarı} (None: kendisi).

    Döndürür:
        Öznitelikleri hizalanmış kopya.
    """
    out = df.copy()
    X = df[cols].to_numpy(np.float64)
    Y = np.full_like(X, np.nan)
    for src in df["kaynak"].unique():
        sel = (df["kaynak"] == src).to_numpy()
        st = stats[(source_map or {}).get(src, src)]
        Z = (X[sel] - st["ort"]) / st["std"]
        Y[sel] = Z if mode == "z" else np.nan_to_num(Z) @ st["beyazlatma"]
    out[cols] = Y
    return out


def attr_model():
    """Öznitelik tahmin modeli: küçük HGB (lr 0.1, 150 iterasyon, 15 yaprak); 7 ayrı model, hız için küçük tutuldu."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, max_leaf_nodes=15, random_state=RANDOM_STATE)


def attr_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Öznitelik etiketi olan LeukemiaAttri hücreleri ("none" ve 4 = etiketsiz hariç).

    Parametreler:
        df: Hücreler.

    Döndürür:
        Alt tablo.
    """
    la = df[(df["kaynak"] == LA_SOURCE) & (df["hucre_tipi"] != "none")]
    return la[(la[LEUKEMIAATTRI_ATTRIBUTES] != 4).all(axis=1)]


def fit_attr_models(train: pd.DataFrame, cols: list[str]) -> tuple[dict, pd.Series]:
    """
    7 morfolojik öznitelik için ara modeller (yalnızca eğitim bölümündeki LeukemiaAttri hücreleri). Eğitim
    hücrelerinin kendi tahminleri hasta gruplu 3 kat katlar dışıdır (ana model, ezberlenmiş tahmine güvenmesin).

    Parametreler:
        train: Eğitim bölümü hücreleri (LA hücreleri dahil).
        cols: Girdi öznitelikleri.

    Döndürür:
        (modeller, {kutu_id: katlar dışı tahmin satırı} için DataFrame indeksli kutu_id).
    """
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    t = attr_targets(train)
    X = t[cols].to_numpy(np.float64)
    oof = np.zeros((len(t), len(LEUKEMIAATTRI_ATTRIBUTES)))
    models = {}
    with threadpool_limits(limits=N_JOBS):
        folds = list(GroupKFold(n_splits=3).split(X, groups=t["hasta"].to_numpy()))
        for a_i, a in enumerate(LEUKEMIAATTRI_ATTRIBUTES):
            y = t[a].to_numpy()
            for tr, va in folds:
                m = attr_model().fit(X[tr], y[tr])
                oof[va, a_i] = attr_single(m, X[va])
            models[a] = attr_model().fit(X, y)
    return models, pd.DataFrame(oof, columns=ATTR_COLUMNS, index=t["kutu_id"].to_numpy())


def attr_single(model, X: np.ndarray) -> np.ndarray:
    """
    Tek öznitelik modelinin tahmini: ikili özniteliklerde P(=1), hücre boyutunda (0/1/2) beklenen değer.

    Parametreler:
        model: Eğitilmiş model.
        X: Öznitelik matrisi.

    Döndürür:
        (n,) tahmin.
    """
    p = model.predict_proba(X)
    if len(model.classes_) > 2:
        return p @ np.asarray(model.classes_, np.float64)
    return p[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))


def add_attr_features(df: pd.DataFrame, cols: list[str], models: dict, oof: pd.DataFrame) -> pd.DataFrame:
    """
    Hücrelere tahmin edilen öznitelikleri ekler: ara modellerin eğitimindeki hücrelere katlar dışı tahmin, diğerlerine
    tam modelin tahmini.

    Parametreler:
        df: Hücreler.
        cols: Girdi öznitelikleri.
        models: fit_attr_models modelleri.
        oof: fit_attr_models katlar dışı tahminleri.

    Döndürür:
        ATTR_COLUMNS eklenmiş kopya.
    """
    from threadpoolctl import threadpool_limits

    out = df.copy()
    with threadpool_limits(limits=N_JOBS):
        pred = np.stack([attr_single(models[a], df[cols].to_numpy(np.float64)) for a in LEUKEMIAATTRI_ATTRIBUTES], axis=1)
    out[ATTR_COLUMNS] = pred
    if "kutu_id" in df:
        hit = df["kutu_id"].isin(oof.index).to_numpy()
        if hit.any():
            out.loc[hit, ATTR_COLUMNS] = oof.loc[df.loc[hit, "kutu_id"]].to_numpy()
    return out


def prepare(train: pd.DataFrame, others: list[pd.DataFrame], config: dict, lists: dict,
            pools: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str], dict]:
    """
    Yapılandırmanın dönüşümlerini uygular (hizalama ve / veya öznitelik tahmini).

    Hizalama istatistikleri: eğitimde görülen her kaynak için eğitim bölümündeki TÜM hücreleri (pools["egitim"]);
    eğitimde görülmeyen test kaynağı için kendi etiketsiz test hücreleri (pools["test"]). Etiket kullanılmaz.

    Parametreler:
        train: Eğitim hücreleri (hedef sütunlu).
        others: Tahmin edilecek tablolar (test / doğrulama).
        config: CONFIGS değeri.
        lists: feature_lists çıktısı.
        pools: {"egitim": eğitim bölümünün tüm hücreleri, "test": test hücreleri}.

    Döndürür:
        (eğitim, diğerleri, kullanılan öznitelikler, bilgi sözlüğü — modeli kaydetmek için dönüşüm parametreleri).
    """
    cols = list(lists[config["oznitelik"]])
    info = {}
    if config.get("hizalama"):
        stats = source_stats(pools["egitim"], cols)
        seen = set(stats)
        test_pool = pools["test"][~pools["test"]["kaynak"].isin(seen)]
        if len(test_pool):
            stats.update(source_stats(test_pool, cols))
        train = apply_alignment(train, cols, stats, config["hizalama"])
        others = [apply_alignment(o, cols, stats, config["hizalama"]) for o in others]
        info["istatistik"] = stats
    if config.get("attr"):
        models, oof = fit_attr_models(train, cols)
        train = add_attr_features(train, cols, models, oof)
        others = [add_attr_features(o, cols, models, oof.iloc[:0]) for o in others]
        info["oznitelik_modelleri"] = models
        cols = cols + ATTR_COLUMNS
    return train, others, cols, info


# ---------------------------------------------------------------------------
# 6) Eğitim (iç CV ile hiperparametre) ve katlar
# ---------------------------------------------------------------------------

def inner_folds4(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    İç CV katları: StratifiedGroupKFold (grup = birleşik kaynak grubu / LA hastası, katman = kaynak × hedef).

    Parametreler:
        df: Eğitim hücreleri.

    Döndürür:
        (eğitim, doğrulama) indeks çiftleri.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from src.config import MULTICENTER_INNER_FOLDS

    strata = (df["kaynak"] + "|" + df["hedef"]).to_numpy()
    cv = StratifiedGroupKFold(n_splits=MULTICENTER_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(df)), strata, df["grup"].to_numpy()))


def fit4(df: pd.DataFrame, cols: list[str], params: dict, scheme: str):
    """
    Ağırlıklı HGB eğitimi (multicenter.make_model yapısı: medyan doldurma → standartlaştırma → HGB).

    Parametreler:
        df: Eğitim hücreleri (hedef).
        cols: Öznitelikler.
        params: Hiperparametreler.
        scheme: Ağırlık şeması.

    Döndürür:
        Eğitilmiş model.
    """
    from threadpoolctl import threadpool_limits

    from src.multicenter import make_model

    model = make_model(params)
    with threadpool_limits(limits=N_JOBS):
        model.fit(df[cols].to_numpy(np.float64), df["hedef"].to_numpy(), model__sample_weight=train_weights(df, scheme))
    return model


def tune4(df: pd.DataFrame, cols: list[str], scheme: str, tag: str) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """
    Küçük ızgarada (MULTICENTER_GRID) iç CV ile hiperparametre seçimi; ölçüt ağırlıklı makro F1 (eğitim ağırlıklarıyla).
    Test verisi bu tabloda YOKTUR.

    Parametreler:
        df: Eğitim hücreleri.
        cols: Öznitelikler.
        scheme: Ağırlık şeması.
        tag: İlerleme etiketi.

    Döndürür:
        (en iyi parametreler, ızgara tablosu, en iyinin katlar dışı olasılıkları).
    """
    from sklearn.metrics import f1_score

    from src.config import MULTICENTER_GRID
    from src.multicenter import proba_of

    folds = inner_folds4(df)
    w, y = train_weights(df, scheme), df["hedef"].to_numpy()
    rows, best = [], (-1.0, None, None)
    for params in MULTICENTER_GRID:
        t0 = time.time()
        oof = np.zeros((len(df), len(CLASS_NAMES)))
        for tr, va in folds:
            oof[va] = proba_of(fit4(df.iloc[tr], cols, params, scheme), df.iloc[va], cols)
        score = f1_score(y, np.asarray(CLASS_NAMES)[oof.argmax(axis=1)], labels=CLASS_NAMES, average="macro", sample_weight=w)
        rows.append({**params, "ic_cv_agirlikli_makro_f1": score})
        print(f"    [{tag}] lr={params['learning_rate']} yaprak={params['max_leaf_nodes']}: F1 {score:.3f} "
              f"({time.time() - t0:.0f} sn)", flush=True)
        if score > best[0]:
            best = (score, params, oof)
    return best[1], pd.DataFrame(rows), best[2]


def la_cv_folds(cells: pd.DataFrame) -> pd.DataFrame:
    """
    LeukemiaAttri TRAIN hastaları için 5 kat (StratifiedGroupKFold, katman = tanı, grup = hasta). Bir kez
    outputs/leukemiaattri/la_cv_katlari.csv'ye yazılır, sonra oradan okunur.

    Parametreler:
        cells: Birleşik hücre tablosu.

    Döndürür:
        hasta, tani, kat.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from src.config import LEUKEMIAATTRI_CV_FOLDS

    path = out_dir() / "la_cv_katlari.csv"
    if path.exists():
        return pd.read_csv(path)
    la = cells[(cells["kaynak"] == LA_SOURCE) & (cells["la_bolum"] == "train")]
    cv = StratifiedGroupKFold(n_splits=LEUKEMIAATTRI_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k, (_, va) in enumerate(cv.split(np.zeros(len(la)), la["tani"].to_numpy(), la["hasta"].to_numpy())):
        for h in sorted(la["hasta"].iloc[va].unique()):
            rows.append({"hasta": int(h), "tani": la.loc[la["hasta"] == h, "tani"].iat[0], "kat": k})
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding="utf-8")
    return table


def fold_data(cells: pd.DataFrame, images: pd.DataFrame, fold: str, config: dict) -> dict:
    """
    Bir değerlendirme katının eğitim / test tabloları ve hizalama havuzları.

    - A / B / C: Adım 9 katları (multicenter.fold_split) + LeukemiaAttri TRAIN'in etiketli hücreleri eğitime eklenir.
    - LA<k>: Kaggle geliştirme + ALL-IDB + Münih + LA TRAIN (k. kat hastaları hariç) → k. kat hastaları.
    LeukemiaAttri TEST hiçbir katta yoktur (kodla doğrulanır).

    Parametreler:
        cells, images: load_all çıktıları.
        fold: "A", "B", "C" veya "LA0".."LA4".
        config: CONFIGS değeri.

    Döndürür:
        {"egitim", "test_hucre", "test_goruntu", "havuz_egitim", "havuz_test", "dogrulama"}.
    """
    from src.multicenter import fold_split, verify_fold

    cells = cells.assign(hedef=target_column(cells, config))
    la = cells["kaynak"] == LA_SOURCE
    la_train = la & (cells["la_bolum"] == "train")
    if fold in EVAL_FOLDS:
        base, test_cells, test_imgs = fold_split(cells[~la], images[images["kaynak"] != LA_SOURCE], fold)
        train = pd.concat([base, cells[la_train & (cells["hedef"] != "")]], ignore_index=True)
        check = verify_fold(fold, train, test_imgs)
        pool_train = cells[cells["id"].isin(set(train["id"]))]
    else:
        k = int(fold[2:])
        folds = la_cv_folds(cells)
        val_patients = set(folds.loc[folds["kat"] == k, "hasta"])
        kag_dev = (cells["kaynak"] == "Kaggle") & (cells["kaggle_bolum"] == "gelistirme")
        other = ~la & cells["etiketli"] & (kag_dev | cells["kaynak"].isin(["ALL-IDB", "Münih"]))
        in_val = la_train & cells["hasta"].isin(val_patients)
        train = cells[other | (la_train & ~in_val & (cells["hedef"] != ""))].reset_index(drop=True)
        test_cells = cells[in_val].reset_index(drop=True)
        test_imgs = images[images["id"].isin(set(test_cells["id"]))]
        pool_train = cells[cells["id"].isin(set(train["id"]))]
        check = {"kat": fold, "ortak_hasta": len(set(train.loc[train["kaynak"] == LA_SOURCE, "hasta"]) & val_patients)}
        if check["ortak_hasta"]:
            raise AssertionError(f"{fold}: doğrulama hastası eğitimde")
    if ((train["kaynak"] == LA_SOURCE) & (train["la_bolum"] == "test")).any():
        raise AssertionError(f"{fold}: LeukemiaAttri TEST eğitime girmiş")
    return {"egitim": train.reset_index(drop=True), "test_hucre": test_cells.reset_index(drop=True),
            "test_goruntu": test_imgs.reset_index(drop=True), "havuz_egitim": pool_train,
            "havuz_test": test_cells, "dogrulama": check}


# ---------------------------------------------------------------------------
# 7) LeukemiaAttri ölçüleri
# ---------------------------------------------------------------------------

def blast_type(proba: np.ndarray) -> np.ndarray:
    """
    Blast tiplemesi: yalnızca ALL ile AML arasında (P(ALL) ≥ P(AML) → ALL). "Blast olduğu bilinen hücre hangi tip?"

    Parametreler:
        proba: (n, 3) olasılık (CLASS_NAMES sırası).

    Döndürür:
        "ALL" / "AML" dizisi.
    """
    return np.where(proba[:, 0] >= proba[:, 1], "ALL", "AML")


def la_metrics(df: pd.DataFrame, proba: np.ndarray, prefix: str = "") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    LeukemiaAttri hücre ve hasta düzeyi ölçüleri (ana tanım hücreleri; tüm hücreler, kenara değenler dahil).

    Hücre: ALL→ALL, AML→AML, Healthy→Healthy duyarlılıkları (3 sınıf, en olası sınıf), makro F1, blastlar içinde
    ALL/AML dengeli doğruluğu (blast_type), ek AML varyantı hücrelerinde AML oranı, Healthy alt tip duyarlılıkları.
    Hasta (ALL ve AML hastaları; APML ayrıca): (1) blast çoğunluğu: hastanın etiketli blastlarının blast_type
    çoğunluğu tanıyla aynı mı; (2) baskın tip: tüm etiketli hücrelerinden 3 sınıfta ALL / AML tahmin edilenlerin
    çoğunluğu (risk raporundaki "baskın lösemi tipi" tanımı).

    Parametreler:
        df: LA hücreleri (etiket, etiket_ek, alt_tip, hasta, tani, hucre_tipi).
        proba: Olasılıklar.
        prefix: Ölçü adı öneki.

    Döndürür:
        (ölçüler, karışıklık tablosu (satır: gerçek, sütun: tahmin), hasta tablosu).
    """
    from sklearn.metrics import f1_score

    pred = np.asarray(CLASS_NAMES)[proba.argmax(axis=1)]
    btype = blast_type(proba)
    y, yx = df["etiket"].to_numpy(), df["etiket_ek"].to_numpy()
    lab = y != ""
    m = {f"{prefix}n_{c}": int((y == c).sum()) for c in CLASS_NAMES}
    for c in CLASS_NAMES:
        m[f"{prefix}duyarlilik_{c}"] = float((pred[y == c] == c).mean()) if (y == c).any() else np.nan
    m[f"{prefix}makro_f1"] = float(f1_score(y[lab], pred[lab], labels=CLASS_NAMES, average="macro"))
    blast = np.isin(y, ["ALL", "AML"])
    rec = [float((btype[y == c] == c).mean()) for c in ("ALL", "AML")]
    m.update({f"{prefix}blast_ALL_dogru": rec[0], f"{prefix}blast_AML_dogru": rec[1],
              f"{prefix}blast_dengeli_dogruluk": float(np.mean(rec)),
              f"{prefix}blast_anormal_orani": float(np.isin(pred[blast], ["ALL", "AML"]).mean())})
    extra = (y == "") & (yx == "AML")
    m[f"{prefix}ek_AML_n"] = int(extra.sum())
    m[f"{prefix}ek_AML_AML_orani"] = float((pred[extra] == "AML").mean()) if extra.any() else np.nan
    m[f"{prefix}ek_AML_blast_AML_orani"] = float((btype[extra] == "AML").mean()) if extra.any() else np.nan
    sub = df["alt_tip"].to_numpy()
    for t in LEUKEMIAATTRI_HEALTHY:
        if (sub == t).any():
            m[f"{prefix}healthy_{t}"] = float((pred[sub == t] == "Healthy").mean())
    rows_def = [("ALL", y == "ALL"), ("AML", y == "AML"), ("Healthy", y == "Healthy"), ("AML_ek", extra)]
    conf = pd.DataFrame([{"gercek": name, **{c: int((pred[sel] == c).sum()) for c in CLASS_NAMES}} for name, sel in rows_def])
    patients = []
    for h, g in df.assign(_p=pred, _b=btype).groupby("hasta"):
        d = g["tani"].iat[0]
        if d not in ("ALL", "AML", "APML"):
            continue
        truth = "ALL" if d == "ALL" else "AML"
        bl = g[g["etiket"].isin(["ALL", "AML"])]
        n_all, n_aml = int((bl["_b"] == "ALL").sum()), int((bl["_b"] == "AML").sum())
        dom_all, dom_aml = int((g["_p"] == "ALL").sum()), int((g["_p"] == "AML").sum())
        # Ek tanım: tüm blastlar (ek AML varyantı dahil); ana tanımda blastı olmayan AML-M5 hastası "eşit" sayılmasın
        bx = g[g["etiket_ek"].isin(["ALL", "AML"])]
        x_all, x_aml = int((bx["_b"] == "ALL").sum()), int((bx["_b"] == "AML").sum())
        patients.append({"hasta": int(h), "tani": d, "blast": len(bl), "blast_ALL": n_all, "blast_AML": n_aml,
                         "blast_ek_cogunluk": "ALL" if x_all > x_aml else ("AML" if x_aml > x_all else "eşit"),
                         "blast_cogunluk": "ALL" if n_all > n_aml else ("AML" if n_aml > n_all else "eşit"),
                         "hucre": len(g), "anormal_ALL": dom_all, "anormal_AML": dom_aml,
                         "baskin_tip": "ALL" if dom_all > dom_aml else ("AML" if dom_aml > dom_all else "eşit"),
                         "dogru_tip": truth})
    pt = pd.DataFrame(patients)
    main = pt[pt["tani"].isin(["ALL", "AML"])] if len(pt) else pt
    if len(main):
        m[f"{prefix}hasta_n"] = len(main)
        m[f"{prefix}hasta_blast_cogunluk_dogru"] = float((main["blast_cogunluk"] == main["dogru_tip"]).mean())
        m[f"{prefix}hasta_baskin_tip_dogru"] = float((main["baskin_tip"] == main["dogru_tip"]).mean())
        m[f"{prefix}hasta_blast_ek_cogunluk_dogru"] = float((main["blast_ek_cogunluk"] == main["dogru_tip"]).mean())
        for d in ("ALL", "AML"):
            s = main[main["tani"] == d]
            m[f"{prefix}hasta_blast_cogunluk_dogru_{d}"] = float((s["blast_cogunluk"] == s["dogru_tip"]).mean()) if len(s) else np.nan
    return m, conf, pt


def la_bootstrap(df: pd.DataFrame, proba: np.ndarray, n: int | None = None) -> pd.DataFrame:
    """
    Hasta düzeyi bootstrap güven aralığı: hastalar tanı içinde yerine koyarak çekilir (ALL, AML, APML, CLL, CML ayrı
    ayrı; hücreler hastalarıyla birlikte gelir), ölçüler her örneklemde yeniden hesaplanır. %95 yüzdelik aralık.

    Parametreler:
        df: LA hücreleri.
        proba: Olasılıklar.
        n: Tekrar sayısı (None: config.LEUKEMIAATTRI_BOOTSTRAP).

    Döndürür:
        olcu, deger, alt, ust, n_tekrar.
    """
    from src.config import LEUKEMIAATTRI_BOOTSTRAP

    n = n or LEUKEMIAATTRI_BOOTSTRAP
    keys = ["blast_dengeli_dogruluk", "duyarlilik_ALL", "duyarlilik_AML", "duyarlilik_Healthy", "makro_f1",
            "hasta_blast_cogunluk_dogru", "hasta_baskin_tip_dogru"]
    point, _, _ = la_metrics(df, proba)
    df = df.reset_index(drop=True)
    idx_by_patient = df.groupby("hasta").indices
    by_diag = df.groupby("tani")["hasta"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    samples = {k: [] for k in keys}
    for _ in range(n):
        parts, new_ids = [], []
        for d, pats in by_diag.items():
            for j, h in enumerate(rng.choice(pats, size=len(pats), replace=True)):
                idx = idx_by_patient[h]
                parts.append(idx)
                new_ids.append(np.full(len(idx), f"{h}_{len(new_ids)}"))
        idx = np.concatenate(parts)
        sub = df.iloc[idx].assign(hasta=np.concatenate(new_ids))
        try:
            m, _, _ = la_metrics(sub, proba[idx])
        except (ValueError, ZeroDivisionError):
            continue
        for k in keys:
            samples[k].append(m.get(k, np.nan))
    rows = []
    for k in keys:
        s = np.asarray(samples[k], np.float64)
        s = s[np.isfinite(s)]
        rows.append({"olcu": k, "deger": point.get(k, np.nan), "alt": float(np.percentile(s, 2.5)) if len(s) else np.nan,
                     "ust": float(np.percentile(s, 97.5)) if len(s) else np.nan, "n_tekrar": len(s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8) Başlangıç: mevcut modeller LA TEST'te (bir kez, kilitli)
# ---------------------------------------------------------------------------

def reference_bundles() -> dict:
    """
    Referans modeller: varsayılan (ACTIVE_MODEL_FILE = v2_kalibre) ve v3.

    Döndürür:
        {ad: (model, meta)}.
    """
    from src.config import ACTIVE_MODEL_FILE, MULTICENTER_MODEL_FILE

    return {"v2_kalibre": load_any_model(ACTIVE_MODEL_FILE), "v3": load_any_model(MULTICENTER_MODEL_FILE)}


def load_any_model(name: str) -> tuple:
    """
    classifier.load_model + pickle uyumluluğu: v3, "python -m src.multicenter" ile kaydedildiği için sarmalayıcı sınıfı
    pickle'da "__main__.TemperatureScaledModel" adıyla duruyor; başka bir modülden yüklenirken bu ad
    src.multicenter'daki sınıfa yönlendirilir (model dosyası değiştirilmez).

    Parametreler:
        name: Model dosya adı (uzantısız).

    Döndürür:
        (model, meta).
    """
    import __main__

    from src.classifier import load_model
    from src.multicenter import TemperatureScaledModel

    if not hasattr(__main__, "TemperatureScaledModel"):
        __main__.TemperatureScaledModel = TemperatureScaledModel
    return load_model(name)


def run_baseline() -> None:
    """
    Mevcut model (v2_kalibre) ve v3 — ikisi de LeukemiaAttri'yi hiç görmedi — LA TEST'te: hücre düzeyi karışıklık,
    duyarlılıklar, blastlar içinde ALL/AML dengeli doğruluk; hasta düzeyi blast çoğunluğu ve baskın tip; hasta
    bootstrap aralıkları. Yazılan: outputs/leukemiaattri/dis_test_baslangic.csv (+ _karisiklik, _hasta, _ga).
    Dosya varsa hiçbir şey yeniden hesaplanmaz ve üzerine yazılmaz.
    """
    from src.external_validation import predict_cells

    path = out_dir() / "dis_test_baslangic.csv"
    if path.exists():
        print(f"Başlangıç sonucu kilitli, değiştirilmedi: {path}")
        print(pd.read_csv(path).to_string(index=False))
        return
    test = la_cells()
    test = test[test["la_bolum"] == "test"].reset_index(drop=True)
    rows, confs, pats, cis = [], [], [], []
    for name, bundle in reference_bundles().items():
        proba = predict_cells(bundle, test)
        m, conf, pt = la_metrics(test, proba)
        rows += [{"model": name, "olcu": k, "deger": v} for k, v in m.items()]
        confs.append(conf.assign(model=name))
        pats.append(pt.assign(model=name))
        cis.append(la_bootstrap(test, proba).assign(model=name))
        print(f"{name}: blast ALL/AML dengeli {m['blast_dengeli_dogruluk']:.3f} | ALL {m['duyarlilik_ALL']:.3f} "
              f"AML {m['duyarlilik_AML']:.3f} Healthy {m['duyarlilik_Healthy']:.3f} | hasta blast çoğunluğu "
              f"{m['hasta_blast_cogunluk_dogru']:.3f}", flush=True)
    od = out_dir()
    pd.concat(confs).to_csv(od / "dis_test_baslangic_karisiklik.csv", index=False, encoding="utf-8")
    pd.concat(pats).to_csv(od / "dis_test_baslangic_hasta.csv", index=False, encoding="utf-8")
    pd.concat(cis).to_csv(od / "dis_test_baslangic_ga.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).assign(kayit=time.strftime("%Y-%m-%d %H:%M:%S")).to_csv(path, index=False, encoding="utf-8")
    print(f"Kilitlendi: {path}")


# ---------------------------------------------------------------------------
# 9) Ön kayıtlı seçim kuralı ve deneyler
# ---------------------------------------------------------------------------

RULE_CSV = "secim_kurali.csv"
FOLD_CSV = "yapilandirma_katlari.csv"
SUMMARY_CSV = "yapilandirma_sonuclari.csv"
LA_FOLDS = [f"LA{k}" for k in range(5)]


def kaggle_reference() -> float:
    """
    Kaggle kısıtının tabanı: mevcut modelin (v2_kalibre) Kaggle v2 kilitli test görüntü makro F1'i (Adım 9 Kat C,
    outputs/multicenter/loso_ozet.csv, i_mevcut).

    Döndürür:
        Görüntü makro F1.
    """
    t = pd.read_csv(stage_dir("multicenter") / "loso_ozet.csv")
    return float(t.loc[t["yapilandirma"] == "i_mevcut", "C_goruntu_makro_f1"].iat[0])


def write_rule() -> None:
    """
    Ön kayıtlı seçim kuralını outputs/leukemiaattri/secim_kurali.csv'ye yazar. Kural zaten varsa veya deney sonucu
    dosyası varsa DOKUNMAZ (kural sonuçlardan önce sabitlenir).
    """
    from src.config import LEUKEMIAATTRI_MAX_KAGGLE_DROP

    path = out_dir() / RULE_CSV
    if path.exists():
        print(f"Kural zaten kayıtlı, değiştirilmedi: {path}")
        return
    if (out_dir() / FOLD_CSV).exists():
        raise RuntimeError("Deney sonuçları kuraldan önce üretilmiş; kural artık ön kayıtlı sayılamaz.")
    base = kaggle_reference()
    rows = [
        {"madde": "skor", "tanim": "(LOSO A ALL→ALL + LOSO B dengeli doğruluk + LA-CV ALL/AML dengeli doğruluk) / 3; "
         "en yüksek seçilir", "deger": ""},
        {"madde": "LOSO A ALL→ALL", "tanim": "ALL-IDB güvenli ALL görüntülerinde görüntü tahmini = ALL oranı (Adım 9 "
         "Kat A tanımı); eğitimde ALL-IDB ve ALL-IDB kaynaklı Kaggle yok", "deger": ""},
        {"madde": "LOSO B dengeli doğruluk", "tanim": "Münih TEST merkez hücre: MYO→AML, LYT→Healthy, NGS→Healthy "
         "duyarlılıklarının ortalaması; eğitimde Münih yok", "deger": ""},
        {"madde": "LA-CV ALL/AML dengeli doğruluk", "tanim": "LeukemiaAttri TRAIN, hasta gruplu 5 kat, katlar dışı: "
         "etiketli blastlarda (lenfoblast→ALL, miyeloblast→AML) P(ALL) ≥ P(AML) tiplemesinin iki sınıf duyarlılığı "
         "ortalaması", "deger": ""},
        {"madde": "kısıt", "tanim": "Kaggle v2 kilitli test görüntü makro F1 ≥ mevcut model − eşik", "deger": LEUKEMIAATTRI_MAX_KAGGLE_DROP},
        {"madde": "kısıt tabanı", "tanim": "mevcut model (v2_kalibre) Kaggle görüntü makro F1 (Adım 9 Kat C)", "deger": round(base, 4)},
        {"madde": "adaylar", "tanim": "; ".join(CONFIGS), "deger": ""},
        {"madde": "eşitlik", "tanim": "skor eşitse Kaggle görüntü makro F1 yüksek olan", "deger": ""},
        {"madde": "hiçbiri sağlamazsa", "tanim": "skoru en yüksek olan seçilir, kısıt ihlali raporlanır; varsayılan model "
         "zaten değişmez", "deger": ""},
        {"madde": "LA TEST", "tanim": "seçimde kullanılmaz; seçilen yapılandırma yalnızca bir kez değerlendirilir", "deger": ""},
        {"madde": "kayıt zamanı", "tanim": time.strftime("%Y-%m-%d %H:%M:%S"), "deger": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"Ön kayıtlı seçim kuralı yazıldı: {path}")


def la_oof_path(config: str) -> Path:
    """LA-CV katlar dışı tahmin dosyası (data/processed/leukemiaattri/, git dışı)."""
    return proc_dir() / f"la_cv_tahmin_{config}.csv"


def run_config_fold(cells: pd.DataFrame, images: pd.DataFrame, lists: dict, name: str, fold: str) -> dict:
    """
    Bir yapılandırma × kat: dönüşümler, iç CV ile hiperparametre, eğitim, test / doğrulama tahmini ve ölçüler.
    LA katlarında katlar dışı tahminler la_cv_tahmin_<yapılandırma>.csv'ye eklenir.

    Parametreler:
        cells, images: load_all çıktıları.
        lists: feature_lists çıktısı.
        name: Yapılandırma adı.
        fold: Kat.

    Döndürür:
        Sonuç satırı.
    """
    from src.multicenter import evaluate_fold, proba_of

    config = CONFIGS[name]
    t0 = time.time()
    d = fold_data(cells, images, fold, config)
    train, (test,), cols, _ = prepare(d["egitim"], [d["test_hucre"]], config, lists,
                                      {"egitim": d["havuz_egitim"], "test": d["havuz_test"]})
    params = LEUKEMIAATTRI_PARAMS   # Sabit (gerekçe: config); ızgara yalnızca son modelde
    model = fit4(train, cols, params, config["agirlik"])
    proba = proba_of(model, test, cols)
    row = {"yapilandirma": name, "kat": fold, "oznitelik_sayisi": len(cols), "egitim_hucre": len(train),
           "egitim_LA_hucre": int((train["kaynak"] == LA_SOURCE).sum()), "test_hucre": len(test),
           "learning_rate": params["learning_rate"], "max_leaf_nodes": params["max_leaf_nodes"]}
    if fold in EVAL_FOLDS:
        metrics, _ = evaluate_fold(fold, d["test_hucre"], d["test_goruntu"], proba)
        row.update(metrics)
    else:
        oof = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASS_NAMES]).assign(
            kutu_id=d["test_hucre"]["kutu_id"].to_numpy(), kat=fold)
        path = la_oof_path(name)
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=oof.columns)
        pd.concat([old[old["kat"] != fold], oof], ignore_index=True).to_csv(path, index=False, encoding="utf-8")
    row["sure_sn"] = round(time.time() - t0)
    return row


def run_experiments(names: list[str] | None = None, folds: list[str] | None = None) -> None:
    """
    Yapılandırmalar × katlar (A, B, C, LA0-LA4). Daha önce tamamlanan yapılandırma × kat atlanır (yeniden başlatılabilir);
    her kattan sonra outputs/leukemiaattri/yapilandirma_katlari.csv güncellenir.

    Parametreler:
        names: Yapılandırmalar (None: hepsi).
        folds: Katlar (None: hepsi).
    """
    if not (out_dir() / RULE_CSV).exists():
        raise RuntimeError("Önce ön kayıtlı kuralı yazın: python -m src.leukemiaattri kural")
    cells, images = load_all()
    lists = feature_lists(cells)
    path = out_dir() / FOLD_CSV
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["yapilandirma", "kat"])
    for name in names or list(CONFIGS):
        for fold in folds or EVAL_FOLDS + LA_FOLDS:
            if ((done["yapilandirma"] == name) & (done["kat"] == fold)).any():
                continue
            print(f"\n=== {name} / {fold} ===", flush=True)
            row = run_config_fold(cells, images, lists, name, fold)
            done = pd.concat([done[(done["yapilandirma"] != name) | (done["kat"] != fold)], pd.DataFrame([row])],
                             ignore_index=True)
            done.to_csv(path, index=False, encoding="utf-8")
            show = {k: v for k, v in row.items() if k in ("A_ALL_tipleme", "A_healthy_ozgulluk", "B_dengeli_dogruluk",
                                                         "C_goruntu_makro_f1", "ic_cv_f1")}
            print(f"  {name}/{fold}: " + ", ".join(f"{k}={v:.3f}" for k, v in show.items()) + f" ({row['sure_sn']} sn)",
                  flush=True)


def reference_la_cv(cells: pd.DataFrame) -> dict[str, dict]:
    """
    Referans modellerin (LeukemiaAttri'yi hiç görmedi) LA TRAIN ölçüleri: eğitimde olmadıkları için LA-CV'nin
    karşılığıdır.

    Parametreler:
        cells: Birleşik hücre tablosu.

    Döndürür:
        {ad: la_metrics ölçüleri}.
    """
    from src.external_validation import predict_cells

    la = cells[(cells["kaynak"] == LA_SOURCE) & (cells["la_bolum"] == "train")].reset_index(drop=True)
    return {name: la_metrics(la, predict_cells(b, la))[0] for name, b in reference_bundles().items()}


def summary_table() -> pd.DataFrame:
    """
    Yapılandırma başına tek satır: LOSO A / B, Kaggle (C) ölçüleri, LA-CV (katlar dışı tahminlerin tamamında)
    ölçüleri, ön kayıtlı skor, kısıt ve seçim; referans satırları (mevcut model, v3) aday değildir.
    LOSO A / B / C referans değerleri Adım 9'dan: mevcut model = i_mevcut, v3 = ii_cok_merkezli_58 (v3'ün
    yapılandırması, LOSO katlarında eğitilmiş hâli; kayıtlı v3 tüm kaynakları gördüğü için LOSO'da kullanılamaz).

    Döndürür:
        Özet tablo (outputs/leukemiaattri/yapilandirma_sonuclari.csv).
    """
    from src.config import LEUKEMIAATTRI_MAX_KAGGLE_DROP

    folds = pd.read_csv(out_dir() / FOLD_CSV)
    cells, _ = load_all()
    la = cells[(cells["kaynak"] == LA_SOURCE) & (cells["la_bolum"] == "train")].reset_index(drop=True)
    keep = ("A_ALL_tipleme", "A_healthy_ozgulluk", "A_duyarlilik", "A_ALL_AML_orani", "B_dengeli_dogruluk",
            "B_duyarlilik_MYO", "B_duyarlilik_LYT", "B_duyarlilik_NGS", "B_MYO_ALL_orani", "C_goruntu_makro_f1",
            "C_hucre_makro_f1")
    rows = []
    step9 = pd.read_csv(stage_dir("multicenter") / "loso_ozet.csv").set_index("yapilandirma")
    ref_la = reference_la_cv(cells)
    for ref, s9 in (("v2_kalibre", "i_mevcut"), ("v3", "ii_cok_merkezli_58")):
        row = {"yapilandirma": ref, "ad": REFERENCES[ref], "aday": False,
               **{k: step9.loc[s9, k] for k in keep if k in step9.columns}}
        row.update({f"LA_{k}": v for k, v in ref_la[ref].items()})
        rows.append(row)
    for name in CONFIGS:
        sub = folds[folds["yapilandirma"] == name]
        if sub.empty:
            continue
        row = {"yapilandirma": name, "ad": CONFIG_LABELS[name], "aday": True}
        for _, r in sub[sub["kat"].isin(EVAL_FOLDS)].iterrows():
            row.update({k: r[k] for k in keep if k in r and pd.notna(r[k])})
        if la_oof_path(name).exists() and set(LA_FOLDS) <= set(sub["kat"]):
            oof = pd.read_csv(la_oof_path(name)).set_index("kutu_id")
            m = la.merge(oof, left_on="kutu_id", right_index=True, how="inner")
            row.update({f"LA_{k}": v for k, v in la_metrics(m, m[[f"p_{c}" for c in CLASS_NAMES]].to_numpy())[0].items()})
        row["toplam_sure_dk"] = round(sub["sure_sn"].sum() / 60, 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    parts = ["A_ALL_tipleme", "B_dengeli_dogruluk", "LA_blast_dengeli_dogruluk"]
    table["skor"] = table[parts].mean(axis=1, skipna=False)
    base = kaggle_reference()
    table["kisit_saglandi"] = table["C_goruntu_makro_f1"] >= base - LEUKEMIAATTRI_MAX_KAGGLE_DROP
    cand = table[table["aday"] & table["skor"].notna()]
    ok = cand[cand["kisit_saglandi"]]
    pool = ok if len(ok) else cand
    chosen = pool.sort_values(["skor", "C_goruntu_makro_f1"], ascending=False)["yapilandirma"].iat[0] if len(pool) else ""
    table["secilen"] = table["yapilandirma"] == chosen
    return table


def run_summary() -> pd.DataFrame:
    """Özet tabloyu kaydeder ve yazdırır."""
    table = summary_table()
    table.to_csv(out_dir() / SUMMARY_CSV, index=False, encoding="utf-8")
    cols = ["yapilandirma", "A_ALL_tipleme", "A_healthy_ozgulluk", "B_dengeli_dogruluk", "LA_blast_dengeli_dogruluk",
            "LA_duyarlilik_Healthy", "LA_hasta_blast_cogunluk_dogru", "C_goruntu_makro_f1", "skor", "kisit_saglandi", "secilen"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# 10) Laboratuvar kısayolu testi
# ---------------------------------------------------------------------------

def run_lab_shortcut() -> pd.DataFrame:
    """
    Her yapılandırmanın temsilinde (öznitelik kümesi + hizalama / öznitelik tahmini) hücrenin KAYNAĞINI tahmin eden
    HGB (hasta / kaynak gruplu 3 kat, katlar dışı). Havuz: Kaggle geliştirme, ALL-IDB, Münih (TEST hariç),
    LeukemiaAttri TRAIN etiketli hücreleri (hiçbir test seti yok). Ağırlıklar kaynak × sınıf grupları eşit olacak
    şekilde (sınıf karışımı kaynağı ele vermesin). Ölçüler: kaynaklar arası makro AUC ve AYNI SINIF içinde kaynak
    AUC'si (Adım 9 tanımı); 0.5 = laboratuvar ayırt edilemiyor.
    Not: (d)'de kaynak başına standartlaştırma ilk iki momenti tanım gereği eşitler; düşüş beklenen bir sonuçtur,
    kalan AUC daha yüksek momentlerdeki / öznitelikler arası ilişkilerdeki laboratuvar imzasıdır.

    Döndürür:
        yapilandirma, kaynak_makro_auc, sinif_ici_kaynak_auc_<sınıf>, sinif_ici_LA_auc_<sınıf> tablosu.
    """
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    from src.multicenter import make_model, proba_of

    cells, _ = load_all()
    lists = feature_lists(cells)
    path = out_dir() / "laboratuvar_kisayolu.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["yapilandirma"])
    kag_dev = (cells["kaynak"] == "Kaggle") & (cells["kaggle_bolum"] == "gelistirme")
    mun = (cells["kaynak"] == "Münih") & (cells["munih_bolum"].fillna("") != "test")
    la_tr = (cells["kaynak"] == LA_SOURCE) & (cells["la_bolum"] == "train")
    sel = (kag_dev | (cells["kaynak"] == "ALL-IDB") | mun | la_tr)
    params = {"learning_rate": 0.1, "max_iter": 200, "early_stopping": False}
    names = ["v2_ozellikleri"] + list(CONFIGS)
    for name in names:
        if (old["yapilandirma"] == name).any():
            continue
        t0 = time.time()
        config = CONFIGS.get(name, {"oznitelik": "bio", "agirlik": "kaynak"})
        if name == "v2_ozellikleri":
            config = {"oznitelik": "temel"}
            lists = {**lists, "temel": [c for c in lists["bio"] if "_bio_" not in c]}
        df = cells.assign(hedef=target_column(cells, config))
        pool = df[sel]
        train = pool[pool["hedef"] != ""].reset_index(drop=True)
        train, _, cols, _ = prepare(train, [], config, lists, {"egitim": pool, "test": pool.iloc[:0]})
        srcs = sorted(train["kaynak"].unique())
        key = train["kaynak"] + "|" + train["hedef"]
        w = (1.0 / key.map(key.value_counts())).to_numpy()
        w = w / w.mean()
        oof = np.zeros((len(train), len(srcs)))
        for tr, va in inner_folds4(train):
            model = make_model(params)
            with threadpool_limits(limits=N_JOBS):
                model.fit(train.iloc[tr][cols].to_numpy(np.float64), train["kaynak"].iloc[tr].to_numpy(),
                          model__sample_weight=w[tr])
            oof[va] = proba_of(model, train.iloc[va], cols, srcs)
        y = train["kaynak"].to_numpy()
        row = {"yapilandirma": name, "n_oznitelik": len(cols), "kaynaklar": "+".join(srcs),
               "kaynak_makro_auc": float(roc_auc_score(y, oof, multi_class="ovr", labels=srcs, average="macro"))}
        for cls in CLASS_NAMES:
            s = (train["hedef"] == cls).to_numpy()
            present = [k for k in srcs if (y[s] == k).sum() >= 20]
            if len(present) >= 2:
                row[f"sinif_ici_kaynak_auc_{cls}"] = float(np.mean(
                    [roc_auc_score(y[s] == k, oof[s, srcs.index(k)]) for k in present]))
                if LA_SOURCE in present:
                    row[f"sinif_ici_LA_auc_{cls}"] = float(roc_auc_score(y[s] == LA_SOURCE, oof[s, srcs.index(LA_SOURCE)]))
        row["sure_sn"] = round(time.time() - t0)
        old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        old.to_csv(path, index=False, encoding="utf-8")
        print(f"  {name}: kaynak AUC {row['kaynak_makro_auc']:.3f} | " + ", ".join(
            f"{k.replace('sinif_ici_', '')}={v:.3f}" for k, v in row.items() if k.startswith("sinif_ici")) +
              f" ({row['sure_sn']} sn)", flush=True)
    return old


# ---------------------------------------------------------------------------
# 10b) Tanı: blast tipi bilgisi özniteliklerde var mı? (yalnızca LA TRAIN, aynı hasta katları)
# ---------------------------------------------------------------------------

def run_diagnostics() -> None:
    """
    LA-CV'de ALL/AML ayrımının neden zayıf kaldığını ayrıştırır (yalnızca LA TRAIN, la_cv_katlari.csv hasta katları;
    LA TEST'e dokunulmaz). Lenfoblast (ALL) ve miyeloblast (AML) hücrelerinde, hasta gruplu katlar dışı:
    1) Yalnızca LeukemiaAttri ile eğitilen ikili HGB: 73 öznitelik ve renksiz küme (başka laboratuvar yok).
    2) Uzman öznitelik etiketleriyle (7 morfolojik etiket, gerçek değerler) aynı model: bilginin uzman düzeyinde
       var olup olmadığının üst sınırı.
    3) Ara modellerin bu 7 etiketi bizim özniteliklerimizden tahmin başarısı (AUC; blastlarda).
    4) Kaynak başına z-skorun ayrık öznitelikte kaynağa özgü değer üretip üretmediği (parmak izi kontrolü).
    Yazılan: tani_blast_ayrimi.csv, tani_oznitelik_tahmini.csv, tani_zskor_parmak_izi.csv.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from threadpoolctl import threadpool_limits

    from src.multicenter import make_model

    cells, _ = load_all()
    lists = feature_lists(cells)
    folds = la_cv_folds(cells).set_index("hasta")["kat"]
    la = cells[(cells["kaynak"] == LA_SOURCE) & (cells["la_bolum"] == "train")].reset_index(drop=True)
    blasts = la[la["etiket"].isin(["ALL", "AML"])].reset_index(drop=True)
    fold_of = blasts["hasta"].map(folds).to_numpy()
    y = blasts["etiket"].to_numpy()

    def oof_binary(X: np.ndarray) -> np.ndarray:
        p = np.zeros(len(X))
        for k in sorted(set(fold_of)):
            tr, va = fold_of != k, fold_of == k
            w = pd.Series(y[tr]).map(1.0 / pd.Series(y[tr]).value_counts()).to_numpy()
            model = make_model(LEUKEMIAATTRI_PARAMS)
            with threadpool_limits(limits=N_JOBS):
                model.fit(X[tr], y[tr], model__sample_weight=w / w.mean())
                p[va] = model.predict_proba(X[va])[:, list(model.classes_).index("ALL")]
        return p

    rows = []
    for name, X in [("LA_ici_73_oznitelik", blasts[lists["bio"]].to_numpy(np.float64)),
                    ("LA_ici_renksiz", blasts[lists["renksiz"]].to_numpy(np.float64)),
                    ("uzman_7_oznitelik_etiketi", blasts[LEUKEMIAATTRI_ATTRIBUTES].to_numpy(np.float64))]:
        p = oof_binary(X)
        pred = np.where(p >= 0.5, "ALL", "AML")
        rows.append({"girdi": name, "n_ALL": int((y == "ALL").sum()), "n_AML": int((y == "AML").sum()),
                     "dengeli_dogruluk": float(balanced_accuracy_score(y, pred)),
                     "auc": float(roc_auc_score(y == "ALL", p))})
        print(f"  {name}: dengeli doğruluk {rows[-1]['dengeli_dogruluk']:.3f}, AUC {rows[-1]['auc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir() / "tani_blast_ayrimi.csv", index=False, encoding="utf-8")

    t = attr_targets(la).reset_index(drop=True)
    t_fold = t["hasta"].map(folds).to_numpy()
    X = t[lists["bio"]].to_numpy(np.float64)
    is_blast = t["hucre_tipi"].isin(["lymphoblast", "myeloblast"]).to_numpy()
    rows = []
    for a in LEUKEMIAATTRI_ATTRIBUTES:
        target = (t[a].to_numpy() >= 1)   # cell_size: 1-2 = büyük; diğerleri 0/1
        p = np.zeros(len(t))
        for k in sorted(set(t_fold)):
            tr, va = t_fold != k, t_fold == k
            with threadpool_limits(limits=N_JOBS):
                p[va] = attr_model().fit(X[tr], target[tr]).predict_proba(X[va])[:, 1]
        auc_all = roc_auc_score(target, p) if 0 < target.mean() < 1 else np.nan
        tb = target[is_blast]
        auc_b = roc_auc_score(tb, p[is_blast]) if 0 < tb.mean() < 1 else np.nan
        lab = t.loc[is_blast, "hucre_tipi"]
        rows.append({"oznitelik": a, "pozitif_orani": float(target.mean()), "auc_tum_hucreler": auc_all,
                     "auc_blastlarda": auc_b,
                     "lenfoblastta_pozitif": float(target[is_blast][lab.to_numpy() == "lymphoblast"].mean()),
                     "miyeloblastta_pozitif": float(target[is_blast][lab.to_numpy() == "myeloblast"].mean())})
        print(f"  {a}: AUC tüm {auc_all:.3f}, blastlarda {auc_b:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir() / "tani_oznitelik_tahmini.csv", index=False, encoding="utf-8")

    lab_cells = cells[cells["etiketli"] | (cells["kaynak"] == LA_SOURCE)]
    stats = source_stats(lab_cells, lists["bio"])
    z = apply_alignment(lab_cells, lists["bio"], stats, "z")
    rows = []
    for col in ("sekil_lob_sayisi", "doku_bio_nukleol_sayi"):
        vals = {s: set(np.round(z.loc[z["kaynak"] == s, col].dropna(), 6)) for s in z["kaynak"].unique()}
        shared = set.intersection(*vals.values()) if vals else set()
        union = set.union(*vals.values())
        rows.append({"oznitelik": col, "ham_farkli_deger": int(lab_cells[col].nunique()),
                     "z_sonrasi_farkli_deger": len(union), "tum_kaynaklarda_ortak_deger": len(shared),
                     **{f"deger_sayisi_{s}": len(v) for s, v in vals.items()}})
    pd.DataFrame(rows).to_csv(out_dir() / "tani_zskor_parmak_izi.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# 11) Son değerlendirme (LA TEST, bir kez) ve v4
# ---------------------------------------------------------------------------

class LabAdaptedModel:
    """
    v4 sarmalayıcısı: girdi öznitelikleri → (isteğe bağlı) kaynak hizalaması → (isteğe bağlı) morfolojik öznitelik
    tahminleri → HGB → temperature kalibrasyonu. predict_proba, meta["oznitelikler"] sütunlarını (dönüşüm ÖNCESİ)
    bekler; external_validation.predict_cells ve pipeline ile aynı arayüz.

    Hizalamalı yapılandırmada yeni bir laboratuvarın istatistikleri etiketsiz hücrelerinden set_source_stats ile
    verilmelidir (birkaç görüntü); verilmezse predict_proba'ya gelen hücrelerin kendi istatistikleri (≥ 30 hücre)
    ya da yoksa LeukemiaAttri istatistikleri kullanılır.
    """

    def __init__(self, base, temperature: float, mode: str | None = None, stats: dict | None = None,
                 attr_models: dict | None = None, default_source: str = LA_SOURCE):
        """
        Parametreler:
            base: Eğitilmiş HGB Pipeline.
            temperature: Sıcaklık.
            mode: None, "z" veya "coral".
            stats: source_stats çıktısı (eğitim kaynakları).
            attr_models: Öznitelik tahmin modelleri (None: yok).
            default_source: Hizalamada varsayılan istatistik kaynağı.
        """
        self.base, self.temperature, self.mode = base, float(temperature), mode
        self.stats, self.attr_models, self.default_source = stats or {}, attr_models, default_source
        self.active = None
        self.classes_ = base.classes_

    def set_source_stats(self, X_unlabeled: np.ndarray) -> None:
        """
        Yeni laboratuvarın etiketsiz hücrelerinden hizalama istatistikleri.

        Parametreler:
            X_unlabeled: (n, öznitelik) matris (girdi öznitelikleri).
        """
        cols = [f"x{i}" for i in range(X_unlabeled.shape[1])]
        df = pd.DataFrame(X_unlabeled, columns=cols).assign(kaynak="yeni")
        self.active = source_stats(df, cols)["yeni"]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Dönüşümler (hizalama, öznitelik tahmini).

        Parametreler:
            X: Girdi öznitelikleri.

        Döndürür:
            Modelin beklediği matris.
        """
        X = np.asarray(X, np.float64)
        if self.mode:
            st = self.active
            if st is None:
                st = (source_stats(pd.DataFrame(X).assign(kaynak="g").rename(columns=str), [str(i) for i in range(X.shape[1])])["g"]
                      if len(X) >= 30 else self.stats[self.default_source])
            Z = (X - st["ort"]) / st["std"]
            X = Z if self.mode == "z" else np.nan_to_num(Z) @ st["beyazlatma"]
        if self.attr_models:
            X = np.hstack([X, np.stack([attr_single(self.attr_models[a], X) for a in LEUKEMIAATTRI_ATTRIBUTES], axis=1)])
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Kalibre olasılıklar.

        Parametreler:
            X: Girdi öznitelikleri.

        Döndürür:
            (n, sınıf) olasılık.
        """
        from src.multicenter import temperature_scale

        return temperature_scale(self.base.predict_proba(self.transform(X)), self.temperature)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        En olası sınıf.

        Parametreler:
            X: Girdi öznitelikleri.

        Döndürür:
            Etiketler.
        """
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(axis=1)]


def chosen_config() -> str:
    """Ön kayıtlı kurala göre seçilen yapılandırma (yapilandirma_sonuclari.csv'den; yoksa yeniden hesaplanır)."""
    path = out_dir() / SUMMARY_CSV
    table = pd.read_csv(path) if path.exists() else run_summary()
    return str(table.loc[table["secilen"].astype(bool), "yapilandirma"].iat[0])


def oof_image_proba(train: pd.DataFrame, pool: pd.DataFrame, cols: list[str], params: dict, scheme: str) -> np.ndarray:
    """
    Risk eşiği seçimi için görüntü havuzundaki TÜM hücrelerin (etiketsizler dahil) katlar dışı olasılıkları: her
    hücre, grubunun iç CV katında dışarıda kaldığı modelden tahmin edilir (multicenter.run_final_model ile aynı).

    Parametreler:
        train: Etiketli eğitim hücreleri (dönüştürülmüş).
        pool: Eşik havuzunun tüm hücreleri (dönüştürülmüş).
        cols: Öznitelikler.
        params: Hiperparametreler.
        scheme: Ağırlık şeması.

    Döndürür:
        (len(pool), 3) olasılık.
    """
    from src.multicenter import proba_of

    folds = inner_folds4(train)
    group_fold = {}
    for k, (_, va) in enumerate(folds):
        group_fold.update(dict.fromkeys(train["grup"].iloc[va], k))
    rng = np.random.default_rng(RANDOM_STATE)
    for g in sorted(set(pool["grup"]) - set(group_fold)):
        group_fold[g] = int(rng.integers(len(folds)))
    fold_of = pool["grup"].map(group_fold).to_numpy()
    proba = np.zeros((len(pool), len(CLASS_NAMES)))
    for k, (tr, _) in enumerate(folds):
        sel = fold_of == k
        if sel.any():
            proba[sel] = proba_of(fit4(train.iloc[tr], cols, params, scheme), pool[sel], cols)
    return proba


def run_final() -> None:
    """
    Seçilen yapılandırma: Kaggle (tümü) + ALL-IDB + Münih + LA TRAIN ile iç CV'de hiperparametre, eğitim, temperature
    kalibrasyonu → LA TEST'te BİR KEZ değerlendirme (hücre + hasta + bootstrap; dosya varsa tekrarlanmaz) → v4 olarak
    kaydedilir (risk eşikleri kaynak dengeli, katlar dışı görüntü skorlarından). Varsayılan model DEĞİŞMEZ.
    LA TEST'teki laboratuvar eğitimde görüldü (hastalar görülmedi); görülmemiş laboratuvar başarısı LOSO A / B'dir.
    """
    import importlib

    from src.classifier import save_model
    from src.config import LEUKEMIAATTRI_MODEL_FILE, MULTICENTER_GRID, MULTICENTER_SOURCE_CAP, RISK_THRESHOLDS
    from src.multicenter import fit_temperature, proba_of, temperature_scale

    od = out_dir()
    result_path = od / "la_test_son.csv"
    if result_path.exists():
        print(f"LA TEST son değerlendirmesi zaten yapıldı, tekrarlanmadı: {result_path}")
        return
    name = chosen_config()
    config = CONFIGS[name]
    print(f"Seçilen yapılandırma: {name} — {CONFIG_LABELS[name]}", flush=True)
    cells, images = load_all()
    lists = feature_lists(cells)
    cells = cells.assign(hedef=target_column(cells, config))
    la = cells["kaynak"] == LA_SOURCE
    la_test = la & (cells["la_bolum"] == "test")
    train = cells[~la_test & (cells["hedef"] != "") & (cells["etiketli"] | la)].reset_index(drop=True)
    if ((train["kaynak"] == LA_SOURCE) & (train["la_bolum"] == "test")).any():
        raise AssertionError("LA TEST eğitime girmiş")
    test = cells[la_test].reset_index(drop=True)
    pool = cells[cells["id"].isin(set(train["id"]))].reset_index(drop=True)   # eşik havuzu: eğitim görüntülerinin tüm hücreleri
    train_t, (test_t, pool_t), cols, info = prepare(train, [test, pool], config, lists, {"egitim": pool, "test": test.iloc[:0]})
    params, grid, oof = tune4(train_t, cols, config["agirlik"], f"v4/{name}")
    base = fit4(train_t, cols, params, config["agirlik"])
    temp = fit_temperature(oof, train_t["hedef"].to_numpy(), train_weights(train_t, config["agirlik"]))

    # --- LA TEST: tek değerlendirme
    proba = temperature_scale(proba_of(base, test_t, cols), temp)
    m, conf, pt = la_metrics(test, proba)
    ci = la_bootstrap(test, proba)
    pd.DataFrame([{"model": "v4", "olcu": k, "deger": v} for k, v in m.items()]).assign(
        yapilandirma=name, kayit=time.strftime("%Y-%m-%d %H:%M:%S")).to_csv(result_path, index=False, encoding="utf-8")
    conf.assign(model="v4").to_csv(od / "la_test_son_karisiklik.csv", index=False, encoding="utf-8")
    pt.assign(model="v4").to_csv(od / "la_test_son_hasta.csv", index=False, encoding="utf-8")
    ci.assign(model="v4").to_csv(od / "la_test_son_ga.csv", index=False, encoding="utf-8")
    pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASS_NAMES]).assign(kutu_id=test["kutu_id"].to_numpy()).to_csv(
        proc_dir() / "la_test_son_tahmin.csv", index=False, encoding="utf-8")
    print(f"LA TEST: blast ALL/AML dengeli {m['blast_dengeli_dogruluk']:.3f} | ALL {m['duyarlilik_ALL']:.3f} AML "
          f"{m['duyarlilik_AML']:.3f} Healthy {m['duyarlilik_Healthy']:.3f} | hasta blast çoğunluğu "
          f"{m['hasta_blast_cogunluk_dogru']:.3f}", flush=True)
    print(ci.round(3).to_string(index=False), flush=True)

    # --- Risk eşikleri
    low, high, per_source = v4_thresholds(train_t, pool, pool_t, images, cols, params, config["agirlik"], temp)

    # --- Kaydet (sınıf, modül yolu src.leukemiaattri olsun diye içe aktarılan modülden alınır)
    mod = importlib.import_module("src.leukemiaattri")
    model = mod.LabAdaptedModel(base, temp, config.get("hizalama"), info.get("istatistik"), info.get("oznitelik_modelleri"))
    in_cols = list(lists[config["oznitelik"]])
    meta = {"model": "hgb", "parametreler": params, "oznitelikler": in_cols, "model_oznitelikleri": cols,
            "siniflar": CLASS_NAMES, "segmentasyon": "kmeans (LeukemiaAttri: etiket kutusu içinde)", "yapilandirma": name,
            "yapilandirma_adi": CONFIG_LABELS[name], "kalibrasyon": "temperature (ağırlıklı iç CV katlar dışı)",
            "sicaklik": temp, "agirlik": config["agirlik"], "kaynak_agirligi_tavani": MULTICENTER_SOURCE_CAP,
            "hizalama": config.get("hizalama"), "oznitelik_tahmini": bool(config.get("attr")),
            "ic_cv_f1": float(grid["ic_cv_agirlikli_makro_f1"].max()), "izgara": MULTICENTER_GRID,
            "risk_esikleri": [low, high], "risk_esikleri_varsayilan_model": list(RISK_THRESHOLDS),
            "egitim_hucre_sayisi": int(len(train_t)),
            "egitim_kaynak_sinif": {f"{s}|{c}": int(n) for (s, c), n in train_t.groupby(["kaynak", "hedef"]).size().items()},
            "dis_test": "LeukemiaAttri TEST (laboratuvar eğitimde görüldü, hastalar görülmedi): la_test_son.csv; "
                        "görülmemiş laboratuvar: LOSO A / B (yapilandirma_sonuclari.csv)",
            "varsayilan_model_degil": True}
    save_model(model, LEUKEMIAATTRI_MODEL_FILE, meta)
    pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in meta.items()}]
                 ).to_csv(od / "v4_model_ozeti.csv", index=False, encoding="utf-8")
    grid.to_csv(od / "v4_ic_cv_izgara.csv", index=False, encoding="utf-8")
    print(f"v4 kaydedildi: models/{LEUKEMIAATTRI_MODEL_FILE}.joblib | T={temp:.3f} | risk eşikleri {low} / {high} | "
          f"{len(cols)} öznitelik, {params}", flush=True)
    print(per_source.round(3).to_string(index=False))


def v4_thresholds(train_t: pd.DataFrame, pool: pd.DataFrame, pool_t: pd.DataFrame, images: pd.DataFrame, cols: list[str],
                  params: dict, scheme: str, temp: float) -> tuple[float, float, pd.DataFrame]:
    """
    v4 risk eşikleri: eğitim görüntülerinin katlar dışı görüntü skorlarından, multicenter.balanced_thresholds ile
    (duyarlılık / özgüllük kaynak ortalaması). Kaggle / ALL-IDB / Münih'te görüntü etiketi kullanılır.

    LeukemiaAttri'de görüntü etiketi hastanın tanısıdır; oysa lösemi hastasının birçok görüntüsünde yalnızca normal
    hücre (nötrofil, lenfosit) var ve bu görüntülerin risk skoru doğru olarak 0'dır. Tanıyı görüntü etiketi saymak
    duyarlılık hedefini (%98) ancak alt eşiği 0'ın altına indirerek sağlar ("Düşük" hiç kullanılamaz; ilk denemede
    −0.01 çıktı). Bu yüzden LA görüntüsü en az bir etiketli blast (ana tanım) içeriyorsa "anormal" sayılır; blastsız
    LA görüntüleri belirsizdir (ne pozitif ne negatif) ve eşik seçimine girmez. LA'da sağlıklı hasta olmadığı için
    LA yalnızca duyarlılığa katkı verir. Yalnızca eğitim verisi kullanılır (LA TEST yok).

    Parametreler:
        train_t: Dönüştürülmüş etiketli eğitim hücreleri.
        pool: Eşik havuzunun tüm hücreleri (dönüştürülmemiş; görüntü birleştirme için).
        pool_t: Aynı havuz, dönüştürülmüş.
        images: Görüntü tablosu.
        cols: Model öznitelikleri.
        params: Hiperparametreler.
        scheme: Ağırlık şeması.
        temp: Sıcaklık.

    Döndürür:
        (düşük/orta, orta/yüksek, kaynak başına özet). Aday tablosu ve özet outputs/leukemiaattri/'ye yazılır.
    """
    from src.external_validation import image_results
    from src.multicenter import balanced_thresholds, temperature_scale
    from src.config import RISK_TARGET_SENSITIVITY

    od = out_dir()
    pool_proba = temperature_scale(oof_image_proba(train_t, pool_t, cols, params, scheme), temp)
    img = image_results(pool, pool_proba, images[images["id"].isin(set(pool["id"]))][["id"]])
    img = img.merge(images[["id", "kaynak", "etiket"]], on="id")
    has_blast = set(pool.loc[(pool["kaynak"] == LA_SOURCE) & pool["etiket"].isin(["ALL", "AML"]), "id"])
    is_la = img["kaynak"] == LA_SOURCE
    img.loc[is_la, "etiket"] = np.where(img.loc[is_la, "id"].isin(has_blast), img.loc[is_la, "etiket"], "")
    img = img[(img[[f"n_{c}" for c in CLASS_NAMES]].sum(axis=1) > 0) & img["etiket"].isin(CLASS_NAMES)]
    img["anormal"] = img["etiket"].isin(["ALL", "AML"])
    low, high, cand = balanced_thresholds(img)
    # Alt uçtaki önemsiz aday (en küçük skor − 0.01): o eşikte hiçbir görüntü "Düşük" olamaz, duyarlılık kendiliğinden 1.
    # balanced_thresholds üst uçtaki karşılığını zaten çıkarıyor; burada da çıkarılır. Kalan adaylarda hedef (%98)
    # sağlanamıyorsa aynı fonksiyonun geri dönüşü uygulanır: duyarlılığı en yüksek (eşitse en büyük) eşik.
    cand["onemsiz_alt_uc"] = cand["esik"] <= 0
    real = cand[~cand["onemsiz_alt_uc"]]
    if low <= 0 and len(real):
        ok = real[real["duyarlilik"] >= RISK_TARGET_SENSITIVITY]
        best = ok if len(ok) else real[real["duyarlilik"] == real["duyarlilik"].max()]
        low = round(float(best["esik"].max()), 4)
        cand["hedef_duyarlilik_saglandi"] = len(ok) > 0
        if high < low:
            low, high = high, low
    cand.to_csv(od / "v4_risk_esik_adaylari.csv", index=False, encoding="utf-8")
    per_source = pd.DataFrame([{
        "kaynak": s, "goruntu": len(g), "losemi_goruntu": int(g["anormal"].sum()),
        "losemi_dusuk_orani": float((g.loc[g["anormal"], "anormal_oran"] < low).mean()),
        "healthy_yuksek_orani": float((g.loc[~g["anormal"], "anormal_oran"] >= high).mean()) if (~g["anormal"]).any() else np.nan}
        for s, g in img.groupby("kaynak")])
    per_source.to_csv(od / "v4_risk_esikleri_kaynak.csv", index=False, encoding="utf-8")
    return low, high, per_source


def run_v4_thresholds() -> None:
    """
    Kayıtlı v4'ün risk eşiklerini v4_thresholds ile yeniden seçer ve model meta verisini günceller (model ağırlıkları
    ve LA TEST sonucu değişmez; yalnızca eğitim verisi kullanılır). İlk kayıtta LA görüntü etiketi tanıdan alınmış ve
    alt eşik −0.01 çıkmıştı (gerekçe: v4_thresholds).
    """
    import joblib

    from src.classifier import save_model
    from src.config import LEUKEMIAATTRI_MODEL_FILE, MODELS_DIR

    bundle = joblib.load(MODELS_DIR / f"{LEUKEMIAATTRI_MODEL_FILE}.joblib")
    model, meta = bundle["model"], bundle["meta"]
    name = meta["yapilandirma"]
    config = CONFIGS[name]
    cells, images = load_all()
    lists = feature_lists(cells)
    cells = cells.assign(hedef=target_column(cells, config))
    la = cells["kaynak"] == LA_SOURCE
    la_test = la & (cells["la_bolum"] == "test")
    train = cells[~la_test & (cells["hedef"] != "") & (cells["etiketli"] | la)].reset_index(drop=True)
    pool = cells[cells["id"].isin(set(train["id"]))].reset_index(drop=True)
    train_t, (pool_t,), cols, _ = prepare(train, [pool], config, lists, {"egitim": pool, "test": pool.iloc[:0]})
    if cols != meta["model_oznitelikleri"]:
        raise ValueError("Öznitelik listesi kayıtlı v4 ile uyuşmuyor")
    old = meta["risk_esikleri"]
    low, high, per_source = v4_thresholds(train_t, pool, pool_t, images, cols, meta["parametreler"], config["agirlik"],
                                          meta["sicaklik"])
    meta.update({"risk_esikleri": [low, high], "risk_esikleri_ilk_kayit": old,
                 "risk_esigi_LA_etiketi": "en az bir etiketli blast içeren LA görüntüsü = anormal; blastsız LA görüntüsü dışarıda"})
    save_model(model, LEUKEMIAATTRI_MODEL_FILE, meta)
    pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in meta.items()}]
                 ).to_csv(out_dir() / "v4_model_ozeti.csv", index=False, encoding="utf-8")
    print(f"v4 risk eşikleri: {old} → {[low, high]}")
    print(per_source.round(3).to_string(index=False))


def comparison_table() -> pd.DataFrame:
    """
    LA TEST karşılaştırması: başlangıç (v2_kalibre, v3) ve v4, nokta değeri + hasta bootstrap %95 aralığı.

    Döndürür:
        model, olcu, deger, alt, ust (outputs/leukemiaattri/la_test_karsilastirma.csv).
    """
    od = out_dir()
    parts = [pd.read_csv(od / "dis_test_baslangic_ga.csv")]
    if (od / "la_test_son_ga.csv").exists():
        parts.append(pd.read_csv(od / "la_test_son_ga.csv"))
    table = pd.concat(parts, ignore_index=True)[["model", "olcu", "deger", "alt", "ust"]]
    table.to_csv(od / "la_test_karsilastirma.csv", index=False, encoding="utf-8")
    return table


# ---------------------------------------------------------------------------
# 12) Görseller ve sayım tabloları (outputs/leukemiaattri/; görüntü İÇERMEZ)
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [("A_ALL_tipleme", "LOSO A\nALL-IDB ALL→ALL"), ("A_healthy_ozgulluk", "LOSO A\nHealthy özgüllüğü"),
                   ("B_dengeli_dogruluk", "LOSO B\nMünih dengeli"), ("LA_blast_dengeli_dogruluk", "LA-CV\nALL/AML dengeli"),
                   ("LA_duyarlilik_Healthy", "LA-CV\nHealthy duyarlılığı"), ("C_goruntu_makro_f1", "Kaggle test\ngörüntü F1"),
                   ("skor", "Ön kayıtlı\nskor")]
TEST_METRICS = [("blast_dengeli_dogruluk", "Blast ALL/AML\ndengeli doğruluk"), ("duyarlilik_ALL", "ALL→ALL"),
                ("duyarlilik_AML", "AML→AML"), ("duyarlilik_Healthy", "Healthy→Healthy"),
                ("hasta_blast_cogunluk_dogru", "Hasta düzeyi\n(blast çoğunluğu)")]
MODEL_LABELS = {"v2_kalibre": "Mevcut (v2_kalibre)", "v3": "v3", "v4": "v4"}


def count_tables() -> None:
    """
    Hücre sayıları (tip × bölüm × proje etiketi) ve hasta sayıları → hucre_sayilari.csv, hasta_sayilari.csv.
    """
    cells = la_cells()
    t = cells.groupby(["hucre_tipi", "la_bolum"]).size().unstack(fill_value=0)
    lab = cells.assign(etiket=cells["etiket"].replace("", "dışarıda")).groupby(["hucre_tipi", "etiket"]).size().unstack(fill_value=0)
    t.join(lab, how="left").reset_index().to_csv(out_dir() / "hucre_sayilari.csv", index=False, encoding="utf-8")
    p = cells.groupby(["tani", "la_bolum"])["hasta"].nunique().unstack(fill_value=0)
    p.reset_index().to_csv(out_dir() / "hasta_sayilari.csv", index=False, encoding="utf-8")


def plot_summary(table: pd.DataFrame) -> None:
    """
    Yapılandırma × ölçü ısı tablosu (değerler yazılı; seçilen yapılandırma işaretli) → yapilandirma_sonuclari.png.

    Parametreler:
        table: summary_table çıktısı.
    """
    from src.dataset import SURFACE, TEXT_PRIMARY

    metrics = [(k, lab) for k, lab in SUMMARY_METRICS if k in table]
    vals = table[[k for k, _ in metrics]].to_numpy(np.float64)
    names = [("► " if s else "") + (REFERENCES.get(n) or CONFIG_LABELS.get(n, n)) for n, s in zip(table["yapilandirma"], table["secilen"])]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(table) + 1.6), facecolor=SURFACE)
    ax.imshow(np.nan_to_num(vals, nan=0), cmap="Blues", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(v) and v > 0.75 else TEXT_PRIMARY)
    ax.set_xticks(range(len(metrics)), [lab for _, lab in metrics], fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.axhline(1.5, color=TEXT_PRIMARY, lw=0.8)
    ax.xaxis.tick_top()
    ax.set_title("Adım 10 yapılandırmaları (ilk iki satır referans; ► ön kayıtlı kurala göre seçilen)",
                 fontsize=9, color=TEXT_PRIMARY, loc="left", pad=34)
    fig.tight_layout()
    from src.utils import save_figure

    save_figure(fig, OUT_STAGE, "yapilandirma_sonuclari")


def plot_test_comparison(table: pd.DataFrame) -> None:
    """
    LA TEST: başlangıç modelleri ve v4, hasta bootstrap %95 aralıklı çubuklar → la_test_karsilastirma.png.

    Parametreler:
        table: comparison_table çıktısı.
    """
    from src.dataset import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
    from src.multicenter import style_axis
    from src.utils import save_figure

    models = [m for m in MODEL_LABELS if m in set(table["model"])]
    colors = {"v2_kalibre": "#9a9890", "v3": "#7fb2e5", "v4": "#0d3f7a"}
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    x = np.arange(len(TEST_METRICS))
    width = 0.8 / len(models)
    for k, mdl in enumerate(models):
        sub = table[table["model"] == mdl].set_index("olcu")
        v = np.array([sub.loc[m, "deger"] if m in sub.index else np.nan for m, _ in TEST_METRICS])
        lo = np.array([sub.loc[m, "alt"] if m in sub.index else np.nan for m, _ in TEST_METRICS])
        hi = np.array([sub.loc[m, "ust"] if m in sub.index else np.nan for m, _ in TEST_METRICS])
        pos = x + (k - (len(models) - 1) / 2) * width
        ax.bar(pos, v, width * 0.9, color=colors[mdl], label=MODEL_LABELS[mdl])
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=TEXT_PRIMARY, elinewidth=0.7, capsize=2)
        for p, val in zip(pos, v):
            ax.text(p, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT_PRIMARY)
    ax.axhline(0.5, color="#c0392b", lw=0.8, ls=":")
    ax.set_xticks(x, [lab for _, lab in TEST_METRICS], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Oran (hasta bootstrap %95 GA)", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("LeukemiaAttri TEST (4 ALL + 5 AML hastası; laboratuvar v4 eğitiminde görüldü, hastalar görülmedi)",
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.17))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "la_test_karsilastirma")


def plot_test_confusions() -> None:
    """LA TEST karışıklık matrisleri (yalnızca sayılar) → la_test_karisiklik.png."""
    from src.dataset import SURFACE, TEXT_PRIMARY
    from src.utils import save_figure

    od = out_dir()
    parts = [pd.read_csv(od / "dis_test_baslangic_karisiklik.csv")]
    if (od / "la_test_son_karisiklik.csv").exists():
        parts.append(pd.read_csv(od / "la_test_son_karisiklik.csv"))
    conf = pd.concat(parts)
    models = [m for m in MODEL_LABELS if m in set(conf["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.0), facecolor=SURFACE)
    rows_lab = {"ALL": "ALL (lenfoblast)", "AML": "AML (miyeloblast)", "Healthy": "Healthy (5 tip)", "AML_ek": "AML ek varyant"}
    for ax, mdl in zip(np.atleast_1d(axes), models):
        t = conf[conf["model"] == mdl].set_index("gercek")[CLASS_NAMES]
        norm = t.div(t.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(t)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{t.iat[i, j]}\n%{100 * norm.iat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if norm.iat[i, j] > 0.55 else TEXT_PRIMARY)
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=7)
        ax.set_yticks(range(len(t)), [rows_lab.get(r, r) for r in t.index], fontsize=7)
        ax.set_title(MODEL_LABELS[mdl], fontsize=8, color=TEXT_PRIMARY, loc="left")
    fig.suptitle("LeukemiaAttri TEST, hücre düzeyi (satır: gerçek, sütun: tahmin)", fontsize=9, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "la_test_karisiklik")


def plot_lab_shortcut(lab: pd.DataFrame) -> None:
    """
    Yapılandırma başına sınıf içi kaynak AUC'si (0.5 = laboratuvar ayırt edilemiyor) → laboratuvar_kisayolu.png.

    Parametreler:
        lab: run_lab_shortcut çıktısı.
    """
    from src.dataset import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
    from src.multicenter import style_axis
    from src.utils import save_figure

    metrics = [c for c in [f"sinif_ici_kaynak_auc_{k}" for k in CLASS_NAMES] if c in lab]
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    x = np.arange(len(lab))
    width = 0.8 / len(metrics)
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, m in enumerate(metrics):
        ax.bar(x + (k - (len(metrics) - 1) / 2) * width, lab[m], width * 0.9, color=palette[k], label=m.rsplit("_", 1)[1])
    ax.axhline(0.5, color=TEXT_SECONDARY, lw=0.7, ls=":")
    labels = ["v2 öznitelikleri (58)" if n == "v2_ozellikleri" else n for n in lab["yapilandirma"]]
    ax.set_xticks(x, labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Aynı sınıf içinde kaynak AUC'si", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("Laboratuvar kısayolu: aynı sınıftaki hücrelerin laboratuvarı ne kadar ele veriliyor? (4 kaynak)",
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, title="sınıf", title_fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.3))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "laboratuvar_kisayolu")


def run_figures() -> None:
    """Adım 10 tabloları ve figürleri (outputs/leukemiaattri/; görüntü içermez)."""
    count_tables()
    od = out_dir()
    if (od / SUMMARY_CSV).exists():
        plot_summary(pd.read_csv(od / SUMMARY_CSV))
    plot_test_comparison(comparison_table())
    plot_test_confusions()
    if (od / "laboratuvar_kisayolu.csv").exists():
        plot_lab_shortcut(pd.read_csv(od / "laboratuvar_kisayolu.csv"))
    print(f"Figürler: {od}")


PARTS = {"cap": run_diameters, "hucre": run_cells, "kontrol": run_mask_check, "baslangic": run_baseline,
         "kural": write_rule, "ozet": run_summary, "laboratuvar": run_lab_shortcut, "son": run_final, "esik": run_v4_thresholds, "tani": run_diagnostics,
         "gorseller": run_figures}


def main() -> None:
    """Komut satırı: python -m src.leukemiaattri <bölüm>."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    start = time.time()
    part = sys.argv[1]
    if part == "deney":
        # python -m src.leukemiaattri deney [yapılandırma,...] [kat,...]
        names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "hepsi" else None
        folds = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        run_experiments(names, folds)
    else:
        PARTS[part]()
    print(f"\nToplam süre: {time.time() - start:.0f} sn")


if __name__ == "__main__":
    main()
