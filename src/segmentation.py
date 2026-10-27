"""
Aşama 2 - Segmentasyon.

Hedef: beyaz kan hücresi ÇEKİRDEKLERİNİ arka plandan (eritrosit + plazma) ayırmak.
Otsu, K-means ve Watershed ayrı ayrı uygulanır; üçü de etiketli maske döndürür ve aynı
morfolojik son işlemden geçer. Yöntemlerin karşılaştırması: src/segmentation_analysis.py.

Ölçek: Görüntüler farklı büyütmelerde (yayma / tek hücre yakın çekim) olduğu için tüm boyut
parametreleri (minimum nesne alanı, açma çekirdeği, watershed marker mesafesi, kırpım payı)
sabit piksel yerine görüntüdeki tahmini ERİTROSİT ÇAPINA oranla belirlenir.

Girdiler preprocessing.preprocess() çıktısındaki ara görüntülerdir:
'kontrast' (ters Lab b*, CLAHE), 'filtreli', 'renk_normalize' (RGB) ve 'gecerli_maske'.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from sklearn.cluster import KMeans

from src.config import (
    CELL_MARGIN_RATIO,
    CELLS_DIRS,
    CELLS_INDEX_CSVS,
    DATASET_INDEX_CSV,
    KMEANS_K,
    KMEANS_MAX_PIXELS,
    KMEANS_N_INIT,
    MIN_NUCLEUS_AREA_RATIO,
    NUCLEUS_MIN_CONTRAST,
    OPENING_RADIUS_RATIO,
    RANDOM_STATE,
    RBC_FALLBACK_DIAMETER,
    RBC_MIN_COMPONENTS,
    RBC_MIN_COVERAGE,
    RBC_MIN_INSCRIBED_RADIUS,
    SEGMENTATION_METHOD,
    SEGMENTATION_METHODS,
    TARGET_LAB_BACKGROUND,
    WATERSHED_H_RATIO,
)
from src.dataset import estimate_image_type, nucleus_mask
from src.preprocessing import preprocess
from src.utils import load_image

# Normalize edilmiş görüntüde zeminin ters b* değeri (normalizasyon zemini bu değere taşır)
BACKGROUND_INV_B = 255 - TARGET_LAB_BACKGROUND["ortalama"][2]


# ---------------------------------------------------------------------------
# Eritrosit boyutu (ölçek) tahmini
# ---------------------------------------------------------------------------

def rbc_mask(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Eritrositlerin kaba ikili maskesini çıkarır.

    Eritrositler plazmadan koyudur: lökosit çekirdekleri (B−G maskesi, 5x5 genişletilmiş)
    dışlandıktan sonra L kanalında Otsu eşiği uygulanır. Eritrositin soluk merkezi
    (bikonkav şekil) delik doldurmayla kapatılır.

    Parametreler:
        image: Renk normalize RGB görüntü.
        valid: Geçerli piksel maskesi.

    Döndürür:
        Eritrosit maskesi (bool).
    """
    lightness = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]
    nuclei = cv2.dilate(nucleus_mask(image), np.ones((5, 5), np.uint8)).astype(bool)
    region = valid & ~nuclei
    if region.sum() < 100:
        return np.zeros_like(valid)
    t, _ = cv2.threshold(lightness[region].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ndi.binary_fill_holes((lightness <= t) & region)
    # 3x3 açma: eritrositler arasındaki 1 px'lik köprüleri ve JPEG beneklerini siler
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def estimate_rbc_diameter(image: np.ndarray, valid: np.ndarray) -> dict:
    """
    Görüntüdeki ortanca eritrosit çapını (piksel) tahmin eder.

    Her eritrosit bileşeninin en büyük iç teğet çemberi (mesafe dönüşümünün maksimumu × 2)
    hesaplanır; bileşen alanıyla ağırlıklı medyan alınır.
    - İç teğet çember, birbirine yapışmış eritrositlerde bile TEK hücrenin çapını verir ve
      görüntü kenarında kesilmiş hücrede merkez görünüyorsa çapı doğru ölçer (görüntü sınırı
      ve dolgu "arka plan" sayılmaz). Yakın çekimlerde eritrositlerin çoğu kenara değdiği için önemli.
    - Alan ağırlığı, kenarda kalan küçük parçaların medyanı aşağı çekmesini engeller.

    Parametreler:
        image: Renk normalize RGB görüntü.
        valid: Geçerli piksel maskesi.

    Döndürür:
        cap (float, güvenilmezse NaN), bilesen_sayisi, kapsama (eritrosit alan oranı),
        guvenilir (bool), maske (eritrosit maskesi) sözlüğü.
    """
    mask = rbc_mask(image, valid)
    coverage = float(mask[valid].mean()) if valid.any() else 0.0
    src = mask.astype(np.uint8)
    src[~valid] = 1  # Dolgu ve görüntü sınırı mesafe dönüşümünde "sıfır" (arka plan) sayılmasın
    dist = cv2.distanceTransform(np.pad(src, 1, constant_values=1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    result = {"cap": np.nan, "bilesen_sayisi": 0, "kapsama": coverage, "guvenilir": False, "maske": mask}
    if n <= 1:
        return result
    radii = ndi.maximum(dist, labels, index=np.arange(1, n))
    keep = radii >= RBC_MIN_INSCRIBED_RADIUS
    result["bilesen_sayisi"] = int(keep.sum())
    if keep.sum() < RBC_MIN_COMPONENTS or coverage < RBC_MIN_COVERAGE:
        return result
    diameters, weights = 2 * radii[keep], stats[1:, cv2.CC_STAT_AREA][keep].astype(np.float64)
    order = np.argsort(diameters)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    result["cap"] = float(diameters[order][np.searchsorted(cumulative, 0.5)])
    result["guvenilir"] = True
    return result


def resolve_rbc_diameter(estimate: dict, image_type: str) -> float:
    """
    Tahmin güvenilmezse görüntü tipinin tipik eritrosit çapını kullanır.

    Parametreler:
        estimate: estimate_rbc_diameter çıktısı.
        image_type: "yayma" veya "yakin_cekim".

    Döndürür:
        Kullanılacak eritrosit çapı (px).
    """
    return estimate["cap"] if estimate["guvenilir"] else RBC_FALLBACK_DIAMETER[image_type]


def rbc_area(diameter: float) -> float:
    """
    Çaptan dairesel eritrosit alanını hesaplar.

    Parametreler:
        diameter: Eritrosit çapı (px).

    Döndürür:
        π (d/2)² (px²).
    """
    return float(np.pi * (diameter / 2) ** 2)


# ---------------------------------------------------------------------------
# Ortak son işlem
# ---------------------------------------------------------------------------

def postprocess(mask: np.ndarray, valid: np.ndarray, rbc_diameter: float) -> np.ndarray:
    """
    Üç yöntemde de aynı morfolojik son işlem: açma → delik doldurma → küçük nesne temizliği.

    Parametreler:
        mask: Ham ikili çekirdek maskesi.
        valid: Geçerli piksel maskesi.
        rbc_diameter: Eritrosit çapı (px); çekirdek boyutu ve minimum alan buna oranlanır.

    Döndürür:
        Temizlenmiş ikili maske (bool).
    """
    radius = max(1, round(OPENING_RADIUS_RATIO * rbc_diameter))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx((mask & valid).astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    filled = ndi.binary_fill_holes(opened)
    return remove_small(filled, rbc_diameter)


def remove_small(mask: np.ndarray, rbc_diameter: float) -> np.ndarray:
    """
    Alanı MIN_NUCLEUS_AREA_RATIO × eritrosit alanından küçük bileşenleri siler.

    Parametreler:
        mask: İkili maske.
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        Temizlenmiş ikili maske (bool).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_NUCLEUS_AREA_RATIO * rbc_area(rbc_diameter)
    return keep[labels]


def contrast_floor(channel: np.ndarray) -> np.ndarray:
    """
    Çekirdek olabilecek pikselleri işaretler: renk normalize ters b* değeri zeminden en az
    NUCLEUS_MIN_CONTRAST gri seviye yüksek olmalı.

    Renk normalizasyonu zemini her görüntüde aynı değere taşıdığı için bu mutlak eşik
    anlamlıdır. Çekirdeği çok az olan görüntülerde (Healthy yayma, alanın ~%0.2'si) Otsu ve
    K-means histogramı zemin içinde (eritrosit/plazma) bölmeye zorlanır; bu taban sahte
    "çekirdek"leri engeller.

    Parametreler:
        channel: Filtrelenmiş, renk normalize ters b* (preprocess()["filtreli"]).

    Döndürür:
        bool maske.
    """
    return channel.astype(np.float64) >= BACKGROUND_INV_B + NUCLEUS_MIN_CONTRAST


def to_labels(mask: np.ndarray) -> np.ndarray:
    """
    İkili maskeyi 8-bağlantılı etiketli maskeye çevirir.

    Parametreler:
        mask: İkili maske.

    Döndürür:
        int32 etiket görüntüsü (0 = arka plan).
    """
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1].astype(np.int32)


# ---------------------------------------------------------------------------
# Üç yöntem
# ---------------------------------------------------------------------------

def otsu_threshold(channel: np.ndarray, valid: np.ndarray) -> float:
    """
    Yalnızca geçerli piksellerin histogramından Otsu eşiğini hesaplar.

    Parametreler:
        channel: Tek kanallı uint8 görüntü.
        valid: Geçerli piksel maskesi.

    Döndürür:
        Eşik değeri.
    """
    t, _ = cv2.threshold(channel[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def segment_otsu(steps: dict, rbc_diameter: float) -> np.ndarray:
    """
    Otsu eşikleme ile çekirdekleri ayırır.

    Otsu, histogramı iki sınıfa bölen ve sınıflar arası varyansı en büyük yapan eşiği seçer.
    Eşik yalnızca geçerli piksellerden (dolgu hariç) hesaplanır.

    Parametreler:
        steps: preprocess() ara çıktıları ("kontrast", "filtreli", "gecerli_maske").
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        Etiketli çekirdek maskesi (int32).
    """
    channel, valid = steps["kontrast"], steps["gecerli_maske"]
    raw = (channel > otsu_threshold(channel, valid)) & contrast_floor(steps["filtreli"])
    return to_labels(postprocess(raw, valid, rbc_diameter))


def lab_features(image: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü gerçek CIE Lab ölçeğine (L* 0-100, a*/b* −128..127) çevirir.

    OpenCV'nin 8 bit Lab'ında L*, 0-255'e gerildiği için Öklid mesafesinde a*/b*'ye göre
    2.55 kat ağırlık kazanır. K-means'in mesafesi algısal renk farkına (ΔE) karşılık gelsin
    diye gerçek ölçek kullanılır.

    Parametreler:
        image: RGB uint8 görüntü.

    Döndürür:
        (H, W, 3) float32 Lab.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def kmeans_clusters(image: np.ndarray, valid: np.ndarray, k: int = KMEANS_K) -> tuple[np.ndarray, np.ndarray]:
    """
    Geçerli pikselleri Lab uzayında k kümeye ayırır.

    Hız ve bellek için model en fazla KMEANS_MAX_PIXELS rastgele pikselle eğitilir, sonra tüm
    geçerli piksellere uygulanır.

    Parametreler:
        image: Renk normalize RGB görüntü.
        valid: Geçerli piksel maskesi.
        k: Küme sayısı.

    Döndürür:
        (küme haritası (H, W) int, dolguda −1; küme merkezleri (k, 3) gerçek Lab).
    """
    lab = lab_features(image)
    pixels = lab[valid]
    rng = np.random.default_rng(RANDOM_STATE)
    sample = pixels[rng.choice(len(pixels), min(len(pixels), KMEANS_MAX_PIXELS), replace=False)]
    model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE).fit(sample)
    cluster_map = np.full(valid.shape, -1, np.int32)
    cluster_map[valid] = model.predict(pixels)
    return cluster_map, model.cluster_centers_


def segment_kmeans(steps: dict, rbc_diameter: float, k: int = KMEANS_K) -> np.ndarray:
    """
    Lab renk uzayında K-means kümeleme ile çekirdekleri ayırır.

    Çekirdek kümesi otomatik seçilir: b* merkezi en düşük (= ters b* en yüksek, en mavi/mor) küme.

    Parametreler:
        steps: preprocess() ara çıktıları ("renk_normalize", "filtreli", "gecerli_maske").
        rbc_diameter: Eritrosit çapı (px).
        k: Küme sayısı.

    Döndürür:
        Etiketli çekirdek maskesi (int32).
    """
    valid = steps["gecerli_maske"]
    cluster_map, centers = kmeans_clusters(steps["renk_normalize"], valid, k)
    raw = (cluster_map == int(np.argmin(centers[:, 2]))) & contrast_floor(steps["filtreli"])
    return to_labels(postprocess(raw, valid, rbc_diameter))


def watershed_stages(steps: dict, rbc_diameter: float) -> dict[str, np.ndarray]:
    """
    Marker kontrollü watershed'in tüm aşamalarını hesaplar (görselleştirme için de kullanılır).

    1. İkili maske: Otsu (ortak son işlemle temizlenmiş).
    2. Mesafe dönüşümü: her ön plan pikselinin en yakın arka plana uzaklığı; çekirdek
       merkezlerinde tepe yapar.
    3. Markerlar: mesafe dönüşümünün h-maksimumları, yani çevresinden en az
       h = WATERSHED_H_RATIO × eritrosit çapı kadar yüksek tepeler. İki çekirdek ancak aralarındaki
       boyun (eyer noktası) her iki tepeden de h kadar daha dar ise ayrılır; tek bir uzun
       çekirdeğin hafif dalgalı mesafe yüzeyi bölünmez. Markersız kalan bileşene en yüksek
       noktasından bir marker eklenir (hiçbir çekirdek kaybolmasın diye).
    4. Watershed: −mesafe yüzeyi markerlardan "su basılarak" doldurulur; farklı markerlardan
       gelen sular buluştuğu yerde 1 px'lik sınır (set) çizilir.

    Parametreler:
        steps: preprocess() ara çıktıları.
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        {"ikili", "mesafe", "markerlar", "etiketler"} sözlüğü.
    """
    valid = steps["gecerli_maske"]
    binary = segment_otsu(steps, rbc_diameter) > 0
    dist = ndi.distance_transform_edt(binary)
    components = to_labels(binary)
    # h en az 0.5 px: mesafe dönüşümünün piksel ölçeğindeki gürültü tepeleri marker olmasın
    markers = ndi.label(h_maxima(dist, max(0.5, WATERSHED_H_RATIO * rbc_diameter)) & binary)[0].astype(np.int32)
    for comp in np.setdiff1d(np.unique(components[binary]), np.unique(components[markers > 0])):
        y, x = np.unravel_index(np.argmax(np.where(components == comp, dist, -1)), dist.shape)
        markers[y, x] = markers.max() + 1
    labels = watershed(-dist, markers, mask=binary, watershed_line=True)
    # Ayrılan parçalar arasında minimum alan kuralı yeniden uygulanır (aşırı bölünme kırıntıları)
    labels = to_labels(remove_small(labels > 0, rbc_diameter) & valid)
    return {"ikili": binary, "mesafe": dist, "markerlar": markers, "etiketler": labels}


def segment_watershed(steps: dict, rbc_diameter: float) -> np.ndarray:
    """
    Marker kontrollü watershed ile birbirine değen çekirdekleri ayırır.

    Parametreler:
        steps: preprocess() ara çıktıları.
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        Etiketli çekirdek maskesi (int32).
    """
    return watershed_stages(steps, rbc_diameter)["etiketler"]


METHODS = {"otsu": segment_otsu, "kmeans": segment_kmeans, "watershed": segment_watershed}


def image_rbc_diameter(steps: dict, image_type: str | None = None) -> tuple[float, dict]:
    """
    Görüntünün eritrosit çapını tahmin eder; güvenilmezse görüntü tipine göre yedek değer kullanır.

    Parametreler:
        steps: preprocess() ara çıktıları.
        image_type: Biliniyorsa "yayma"/"yakin_cekim"; None ise görüntüden tahmin edilir.

    Döndürür:
        (kullanılacak çap, estimate_rbc_diameter çıktısı + "tip").
    """
    estimate = estimate_rbc_diameter(steps["renk_normalize"], steps["gecerli_maske"])
    if image_type is None:
        image_type = estimate_image_type(steps["boyutlandirilmis"])["tip"]
    estimate["tip"] = image_type
    return resolve_rbc_diameter(estimate, image_type), estimate


def segment(steps: dict, method: str = SEGMENTATION_METHOD, image_type: str | None = None) -> tuple[np.ndarray, float, dict]:
    """
    Ön işlenmiş görüntüde seçilen yöntemle çekirdek segmentasyonu yapar.

    Parametreler:
        steps: preprocess() ara çıktıları.
        method: "otsu", "kmeans" veya "watershed".
        image_type: Biliniyorsa görüntü tipi (eritrosit çapı yedeği için).

    Döndürür:
        (etiketli çekirdek maskesi, eritrosit çapı, eritrosit tahmini sözlüğü).
    """
    diameter, estimate = image_rbc_diameter(steps, image_type)
    return METHODS[method](steps, diameter), diameter, estimate


# ---------------------------------------------------------------------------
# Hücre çıkarma
# ---------------------------------------------------------------------------

# Kaydedilen maske dosyasındaki piksel kodları (öznitelik çıkarmada sitoplazma halkası komşu
# çekirdekleri ve dolguyu dışlayabilsin diye tek bir 8 bit PNG'de saklanır)
MASK_NUCLEUS, MASK_OTHER_NUCLEUS, MASK_PADDING = 255, 128, 64

def extract_cells(image: np.ndarray, labels: np.ndarray, valid: np.ndarray, rbc_diameter: float) -> list[dict]:
    """
    Her çekirdeği, etrafında sitoplazma payı bırakarak kırpar.

    Pay = CELL_MARGIN_RATIO × eritrosit çapı (her yönde). Kırpım görüntü sınırında kesilir.

    Parametreler:
        image: Renk normalize RGB görüntü.
        labels: Etiketli çekirdek maskesi.
        valid: Geçerli piksel maskesi.
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        Her hücre için {"kirpim" (RGB), "cekirdek_maskesi" (bool, yalnızca bu çekirdek),
        "maske" (uint8 kodlu: MASK_NUCLEUS bu çekirdek, MASK_OTHER_NUCLEUS kırpıma giren komşu
        çekirdekler, MASK_PADDING dolgu, 0 diğer), "kutu" (x, y, w, h, kırpım koordinatları),
        "alan", "kenara_degiyor"} listesi.
        kenara_degiyor: çekirdek görüntü sınırına veya dolguya değiyor (hücre eksik olabilir).
    """
    margin = max(2, round(CELL_MARGIN_RATIO * rbc_diameter))
    h, w = labels.shape
    edge = np.zeros_like(valid)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    # Dolguya veya görüntü sınırına komşu pikseller (3x3 genişletme = 1 px temas)
    touching = cv2.dilate((edge | ~valid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    cells = []
    for region, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - margin), min(h, sl[0].stop + margin)
        x0, x1 = max(0, sl[1].start - margin), min(w, sl[1].stop + margin)
        window = labels[y0:y1, x0:x1]
        nucleus = window == region
        coded = np.zeros(window.shape, np.uint8)
        coded[~valid[y0:y1, x0:x1]] = MASK_PADDING
        coded[(window > 0) & ~nucleus] = MASK_OTHER_NUCLEUS
        coded[nucleus] = MASK_NUCLEUS
        cells.append({"kirpim": image[y0:y1, x0:x1].copy(), "cekirdek_maskesi": nucleus, "maske": coded,
                      "kutu": (x0, y0, x1 - x0, y1 - y0), "alan": int(nucleus.sum()),
                      "kenara_degiyor": bool((labels == region)[touching].any())})
    return cells


def save_png(image: np.ndarray, path: Path) -> None:
    """
    Görüntüyü PNG olarak kaydeder (Windows'ta Unicode yollar için imencode + tofile).

    Parametreler:
        image: RGB, gri veya bool görüntü.
        path: Hedef dosya yolu.
    """
    img = image.astype(np.uint8) * 255 if image.dtype == bool else image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Kodlanamadı: {path}")
    buf.tofile(str(path))


def build_cell_dataset(method: str = SEGMENTATION_METHOD, out_dir: Path | None = None,
                       index_csv: Path | None = None) -> pd.DataFrame:
    """
    Tüm veri setinde çekirdekleri segmente edip hücre kırpımlarını ve hücre indeksini kaydeder.

    Görüntüler tek tek işlenir; her hücrenin kırpımı (renk normalize RGB) ve çekirdek maskesi
    hemen diske yazılır, bellekte yalnızca indeks satırları tutulur.
    Dosya adı: <etiket>/<etiket>_<görüntü sırası>_<hücre sırası>.png (+ _maske.png).

    Parametreler:
        method: Segmentasyon yöntemi.
        out_dir: Kırpımların klasörü (None: config.CELLS_DIRS[method]).
        index_csv: Hücre indeksinin yolu (None: config.CELLS_INDEX_CSVS[method]).

    Döndürür:
        Hücre indeksi DataFrame'i.
    """
    out_dir = out_dir or CELLS_DIRS[method]
    index_csv = index_csv or CELLS_INDEX_CSVS[method]
    index = pd.read_csv(DATASET_INDEX_CSV)
    rows, start = [], time.time()
    for i, image_row in index.iterrows():
        _, steps = preprocess(load_image(image_row["dosya_yolu"]))
        labels, diameter, estimate = segment(steps, method, image_row["tip"])
        cells = extract_cells(steps["renk_normalize"], labels, steps["gecerli_maske"], diameter)
        class_dir = out_dir / image_row["etiket"]
        class_dir.mkdir(parents=True, exist_ok=True)
        for k, cell in enumerate(cells):
            name = f"{image_row['etiket']}_{i:04d}_{k:02d}"
            save_png(cell["kirpim"], class_dir / f"{name}.png")
            save_png(cell["maske"], class_dir / f"{name}_maske.png")
            x, y, w, h = cell["kutu"]
            rows.append({
                "hucre_id": name, "kirpim_yolu": str(class_dir / f"{name}.png"),
                "maske_yolu": str(class_dir / f"{name}_maske.png"), "kaynak_dosya": image_row["dosya_yolu"],
                "goruntu_sirasi": i, "grup_id": image_row["grup_id"], "etiket": image_row["etiket"],
                "tip": image_row["tip"], "eritrosit_capi": round(diameter, 2),
                "eritrosit_tahmini_guvenilir": estimate["guvenilir"], "kenara_degiyor": cell["kenara_degiyor"],
                "cekirdek_alani": cell["alan"], "kutu_x": x, "kutu_y": y, "kutu_w": w, "kutu_h": h,
                "segmentasyon_yontemi": method})
        if (i + 1) % 100 == 0 or i + 1 == len(index):
            print(f"  [hücre çıkarma: {method}] {i + 1}/{len(index)} görüntü, {len(rows)} hücre ({time.time() - start:.0f} sn)",
                  flush=True)
    cells_df = pd.DataFrame(rows)
    cells_df.to_csv(index_csv, index=False, encoding="utf-8")
    return cells_df


def report_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Sınıf ve görüntü tipi bazında hücre sayılarını ve hücresiz görüntüleri özetler.

    Parametreler:
        cells: build_cell_dataset çıktısı.

    Döndürür:
        (etiket, tip) indeksli özet tablo.
    """
    index = pd.read_csv(DATASET_INDEX_CSV)
    per_image = cells.groupby("goruntu_sirasi").size().reindex(range(len(index)), fill_value=0)
    index = index.assign(hucre=per_image.to_numpy())
    summary = index.groupby(["etiket", "tip"]).agg(
        goruntu=("hucre", "size"), hucresiz_goruntu=("hucre", lambda s: int((s == 0).sum())),
        hucre=("hucre", "sum"), goruntu_basina_medyan=("hucre", "median"))
    edge = cells.groupby(["etiket", "tip"])["kenara_degiyor"].mean().rename("kenara_degen_orani")
    groups = cells.groupby(["etiket", "tip"])["grup_id"].nunique().rename("hucreli_grup")
    return summary.join(edge).join(groups).round(3)


if __name__ == "__main__":
    import sys

    # Varsayılan: seçilen yöntem. "--hepsi": üç yöntem (Adım 5'teki karşılaştırma için).
    methods = SEGMENTATION_METHODS if "--hepsi" in sys.argv else [SEGMENTATION_METHOD]
    for m in methods:
        print(f"Hücre çıkarma ({m}) başlıyor; görüntüler tek tek işleniyor...")
        print(report_cells(build_cell_dataset(m)).to_string())
