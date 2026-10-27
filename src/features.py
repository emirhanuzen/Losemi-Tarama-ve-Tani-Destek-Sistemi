"""
Aşama 3 - Öznitelik çıkarma.

Her hücre kırpımından (Adım 3, renk normalize RGB) ve kodlu çekirdek maskesinden üç grup
öznitelik çıkarılır:
- Şekil (sekil_*): çekirdek maskesinden; boyut ölçüleri eritrosit çapına oranlı.
- Renk (renk_*): çekirdek ve çekirdek çevresi halkası için Lab/HSV ortalama ve std, ve farkları.
- Doku (doku_*): yalnızca çekirdek içinde GLCM (4 açının ortalaması) ve entropi.

ÖLÇEK: Görüntüler farklı büyütmelerde (yaymada eritrosit ~10 px, yakın çekimde ~62 px). Ham
piksel ölçüleri ve piksel mesafeli doku ölçüleri büyütmeyi, dolayısıyla görüntü tipini kodlar
(kısayol). Bu yüzden her hücre önce eritrosit çapı FEATURE_REF_RBC_DIAMETER px olacak şekilde
yeniden örneklenir; tüm öznitelikler bu ortak ölçekte hesaplanır.

Sütun adlandırması: öznitelikler sekil_/renk_/doku_ önekli; sınıflandırmada KULLANILMAYACAK
her şey (görüntü tipi, boyutu, konum, yöntem, ham piksel ölçüleri, N/C) meta_ önekli veya
etiket / grup_id / hucre_id. feature_columns() yalnızca öznitelik sütunlarını döndürür.

Çalıştırma (proje kökünden):
    python -m src.features            # üç yöntemin öznitelik tabloları
"""

import time

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

from src.config import (
    CELLS_INDEX_CSVS,
    CYTOPLASM_MIN_CONTRAST,
    DATASET_INDEX_CSV,
    FEATURE_PREFIXES,
    FEATURE_REF_RBC_DIAMETER,
    FEATURES_CSVS,
    GLCM_ANGLES_DEG,
    GLCM_DISTANCES,
    GLCM_GRAY_RANGE,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
    LOBE_DEPTH_RATIO,
    NC_MAX_DISTANCE_RATIO,
    NC_MIN_NATIVE_RBC_DIAMETER,
    RING_GAP_PX,
    RING_MIN_PIXELS,
    RING_WIDTH_RATIO,
    SEGMENTATION_METHODS,
)
from src.segmentation import BACKGROUND_INV_B, MASK_NUCLEUS, MASK_OTHER_NUCLEUS, MASK_PADDING

COLOR_CHANNELS = ["L", "a", "b", "H", "S", "V"]


# ---------------------------------------------------------------------------
# Okuma ve ölçek
# ---------------------------------------------------------------------------

def read_png(path: str, color: bool) -> np.ndarray:
    """
    PNG dosyasını okur (Windows'ta Unicode yollar için fromfile + imdecode).

    Parametreler:
        path: Dosya yolu.
        color: True ise RGB, False ise tek kanal.

    Döndürür:
        uint8 görüntü.
    """
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color else img


def to_reference_scale(crop: np.ndarray, coded: np.ndarray, rbc_diameter: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Hücre kırpımını ve kodlu maskesini eritrosit çapı FEATURE_REF_RBC_DIAMETER px olacak şekilde ölçekler.

    Görüntü küçültmede INTER_AREA (piksel ortalaması: ince kromatin detayı ortalanır, örtüşme
    yapmaz), büyütmede INTER_CUBIC ile; maske ise her kod için ayrı ayrı alan ortalaması alınıp
    0.5 eşiğiyle (kenarların yarısından fazlası kaplanmışsa) yeniden örneklenir.

    Parametreler:
        crop: RGB kırpım.
        coded: Kodlu maske (MASK_* değerleri).
        rbc_diameter: Kaynak görüntüdeki eritrosit çapı (px).

    Döndürür:
        (ölçeklenmiş kırpım, ölçeklenmiş kodlu maske, ölçek çarpanı).
    """
    scale = FEATURE_REF_RBC_DIAMETER / rbc_diameter
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    if abs(scale - 1) < 1e-6:
        return crop, coded, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    out = cv2.resize(crop, size, interpolation=interp)
    resized = np.zeros((size[1], size[0]), np.uint8)
    for code in (MASK_PADDING, MASK_OTHER_NUCLEUS, MASK_NUCLEUS):  # Sonraki kod öncekinin üstüne yazar
        frac = cv2.resize((coded == code).astype(np.float32), size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        resized[frac >= 0.5] = code
    return out, resized, scale


def largest_component(mask: np.ndarray) -> np.ndarray:
    """
    İkili maskenin en büyük bağlı bileşenini döndürür (yeniden örnekleme kopuk parça bırakabilir).

    Parametreler:
        mask: İkili maske.

    Döndürür:
        bool maske.
    """
    labels, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


# ---------------------------------------------------------------------------
# Şekil
# ---------------------------------------------------------------------------

def lobe_count(nucleus: np.ndarray, rbc_diameter: float) -> tuple[int, float]:
    """
    Çekirdek lob sayısını dışbükeylik kusurlarından tahmin eder.

    Derinliği ≥ LOBE_DEPTH_RATIO × eritrosit çapı olan her girinti iki lob arasındaki bir
    "bel"dir. İki lob en az iki girinti üretir (iki yandan); n ≥ 2 derin girinti → n lob,
    aksi hâlde tek lob (tek girinti böbrek/at nalı şeklidir).

    Parametreler:
        nucleus: Çekirdek maskesi (referans ölçekte).
        rbc_diameter: Referans ölçekteki eritrosit çapı.

    Döndürür:
        (lob sayısı, en derin girinti / eritrosit çapı).
    """
    contours, _ = cv2.findContours(nucleus.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return 1, 0.0
    hull = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:  # Kendini kesen çok küçük konturlarda OpenCV hata verebilir
        defects = None
    if defects is None:
        return 1, 0.0
    # OpenCV derinliği 8 bit kesirli sabit noktalı verir; sürüme göre (N,1,4) veya (N,4) döner
    depths = defects.reshape(-1, 4)[:, 3] / 256.0
    deep = int((depths >= LOBE_DEPTH_RATIO * rbc_diameter).sum())
    return (deep if deep >= 2 else 1), float(depths.max() / rbc_diameter)


def shape_features(nucleus: np.ndarray, rbc_diameter: float) -> dict:
    """
    Çekirdek maskesinden ölçekten bağımsız şekil özniteliklerini çıkarır.

    Boyutlar eritrosit çapına oranlıdır: alan/d², çevre/d, eksenler/d. Dairesellik 4πA/P²
    (daire = 1), eksantrisite (eşdeğer elipsin odak uzaklığı / büyük eksen; daire = 0),
    solidity (alan / dışbükey örtü alanı), extent (alan / sınır kutusu alanı).

    Parametreler:
        nucleus: Çekirdek maskesi (referans ölçekte).
        rbc_diameter: Referans ölçekteki eritrosit çapı.

    Döndürür:
        sekil_* anahtarlı sözlük.
    """
    props = regionprops(nucleus.astype(np.uint8))[0]
    area, perimeter = float(props.area), float(props.perimeter)
    major, minor = float(props.axis_major_length), float(props.axis_minor_length)
    lobes, depth = lobe_count(nucleus, rbc_diameter)
    return {
        "sekil_alan": area / rbc_diameter ** 2,
        "sekil_cevre": perimeter / rbc_diameter,
        "sekil_dairesellik": min(1.0, 4 * np.pi * area / perimeter ** 2) if perimeter > 0 else np.nan,
        "sekil_eksantrisite": float(props.eccentricity),
        "sekil_solidity": float(props.solidity),
        "sekil_extent": float(props.extent),
        "sekil_buyuk_eksen": major / rbc_diameter,
        "sekil_kucuk_eksen": minor / rbc_diameter,
        "sekil_eksen_orani": minor / major if major > 0 else np.nan,
        "sekil_lob_sayisi": lobes,
        "sekil_girinti_derinligi": depth,
    }


# ---------------------------------------------------------------------------
# Renk
# ---------------------------------------------------------------------------

def perinuclear_ring(coded: np.ndarray, rbc_diameter: float) -> np.ndarray:
    """
    Çekirdeği çevreleyen halkayı (sitoplazma bölgesi yaklaşımı) döndürür.

    Halka: çekirdeğe uzaklığı (RING_GAP_PX, RING_GAP_PX + RING_WIDTH_RATIO × d] olan pikseller;
    komşu çekirdekler ve dolgu hariç. İlk RING_GAP_PX piksellik bant atlanır, çünkü çekirdek
    sınırı bulanıktır (JPEG + yeniden örnekleme + medyan filtre) ve çekirdek rengini taşır.

    Parametreler:
        coded: Kodlu maske (referans ölçekte).
        rbc_diameter: Referans ölçekteki eritrosit çapı.

    Döndürür:
        bool halka maskesi.
    """
    dist = cv2.distanceTransform((coded != MASK_NUCLEUS).astype(np.uint8), cv2.DIST_L2, 5)
    ring = (dist > RING_GAP_PX) & (dist <= RING_GAP_PX + RING_WIDTH_RATIO * rbc_diameter)
    return ring & (coded != MASK_OTHER_NUCLEUS) & (coded != MASK_PADDING)


def color_channels(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Gerçek ölçekte Lab (L* 0-100, a*/b* −128..127) ve HSV (H derece 0-360, S/V 0-1) kanalları.

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        {"L", "a", "b", "H", "S", "V"} → float32 kanal.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    return {"L": lab[..., 0] * 100 / 255, "a": lab[..., 1] - 128, "b": lab[..., 2] - 128,
            "H": hsv[..., 0] * 2, "S": hsv[..., 1] / 255, "V": hsv[..., 2] / 255}


def circular_stats(degrees: np.ndarray) -> tuple[float, float]:
    """
    Ton (H) açısının dairesel ortalamasını ve dairesel std'sini hesaplar.

    H dairesel olduğu için (0° = 360° kırmızı) aritmetik ortalama yanlıştır: 350° ve 10°'nin
    ortalaması 0° olmalıdır, 180° değil.

    Parametreler:
        degrees: Açı değerleri (derece).

    Döndürür:
        (ortalama 0-360°, dairesel std derece).
    """
    rad = np.deg2rad(degrees)
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    r = min(1.0, np.hypot(s, c))
    return float(np.rad2deg(np.arctan2(s, c)) % 360), float(np.rad2deg(np.sqrt(-2 * np.log(max(r, 1e-12)))))


def region_color_stats(channels: dict, region: np.ndarray, prefix: str) -> dict:
    """
    Bir bölgedeki Lab ve HSV kanallarının ortalama ve std'sini hesaplar (H dairesel).

    Parametreler:
        channels: color_channels çıktısı.
        region: Bölge maskesi.
        prefix: Sütun öneki (ör. "renk_cekirdek").

    Döndürür:
        {prefix_<kanal>_ort, prefix_<kanal>_std} sözlüğü (bölge boşsa NaN).
    """
    out = {}
    for name in COLOR_CHANNELS:
        values = channels[name][region]
        if len(values) == 0:
            mean, std = np.nan, np.nan
        elif name == "H":
            mean, std = circular_stats(values)
        else:
            mean, std = float(values.mean()), float(values.std())
        out[f"{prefix}_{name}_ort"], out[f"{prefix}_{name}_std"] = mean, std
    return out


def color_features(image: np.ndarray, nucleus: np.ndarray, ring: np.ndarray) -> dict:
    """
    Çekirdek ve çekirdek çevresi halkasının renk istatistiklerini ve aralarındaki farkı çıkarır.

    Fark: Lab ortalamalarının farkı (çekirdek − halka) ve ΔE (Lab'da Öklid renk farkı).

    Parametreler:
        image: RGB görüntü (referans ölçekte).
        nucleus: Çekirdek maskesi.
        ring: perinuclear_ring çıktısı.

    Döndürür:
        renk_* anahtarlı sözlük.
    """
    channels = color_channels(image)
    out = region_color_stats(channels, nucleus, "renk_cekirdek")
    halo = ring if ring.sum() >= RING_MIN_PIXELS else np.zeros_like(ring)
    out.update(region_color_stats(channels, halo, "renk_halka"))
    diffs = [out[f"renk_cekirdek_{c}_ort"] - out[f"renk_halka_{c}_ort"] for c in "Lab"]
    out.update({"renk_fark_L": diffs[0], "renk_fark_a": diffs[1], "renk_fark_b": diffs[2],
                "renk_fark_deltaE": float(np.sqrt(np.sum(np.square(diffs))))})
    return out


# ---------------------------------------------------------------------------
# Doku (GLCM)
# ---------------------------------------------------------------------------

def quantize_gray(image: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """
    L* (parlaklık) kanalını GLCM_GRAY_RANGE aralığında `levels` gri seviyeye niceler.

    Sabit (veri seti genelinde) aralık kullanılır: renk normalizasyonu zemini hizaladığı için
    aynı gri seviye her görüntüde aynı kromatin koyuluğunu ifade eder. Hücre başına min-max
    germe, soluk (düşük kontrastlı) çekirdeklerdeki gürültüyü yapay doku gibi büyütürdü.

    Parametreler:
        image: RGB görüntü.
        levels: Gri seviye sayısı.

    Döndürür:
        0..levels−1 uint8 görüntü.
    """
    lightness = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    lo, hi = GLCM_GRAY_RANGE
    q = np.floor((lightness - lo) / (hi - lo) * levels)
    return np.clip(q, 0, levels - 1).astype(np.uint8)


def masked_glcm(gray: np.ndarray, mask: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """
    Yalnızca maske içindeki piksel çiftlerinden GLCM hesaplar.

    skimage.graycomatrix maske desteklemediği için maske dışı pikseller ayrılmış bir gri seviyeye
    (levels) atanır, levels+1 seviyeli matris hesaplanır ve bu seviyenin satır/sütunu atılır;
    böylece en az bir ucu maske dışında olan çiftler sayılmaz.

    Parametreler:
        gray: 0..levels−1 nicelenmiş görüntü.
        mask: Çekirdek maskesi.
        levels: Gri seviye sayısı.

    Döndürür:
        (levels, levels, mesafe, açı) sayım matrisi (simetrik).
    """
    coded = np.where(mask, gray, levels).astype(np.uint8)
    glcm = graycomatrix(coded, distances=list(GLCM_DISTANCES), angles=np.deg2rad(GLCM_ANGLES_DEG).tolist(),
                        levels=levels + 1, symmetric=True, normed=False)
    return glcm[:levels, :levels].astype(np.float64)


def glcm_properties(glcm: np.ndarray) -> dict[str, np.ndarray]:
    """
    GLCM'den 6 doku özelliğini (mesafe × açı) hesaplar; çifti olmayan mesafe/açı NaN olur.

    Parametreler:
        glcm: masked_glcm çıktısı (sayımlar).

    Döndürür:
        {özellik: (mesafe, açı) dizisi}.
    """
    totals = glcm.sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        props = {p: graycoprops(glcm, p).astype(np.float64) for p in GLCM_PROPERTIES}
    for p in props:
        props[p][totals == 0] = np.nan
    # Düz (tek gri seviyeli) çekirdekte korelasyon tanımsızdır (0/0); doğrusal ilişki yok sayılır
    props["correlation"] = np.where(np.isnan(props["correlation"]) & (totals > 0), 0.0, props["correlation"])
    return props


def texture_features(image: np.ndarray, nucleus: np.ndarray) -> dict:
    """
    Çekirdek içi GLCM özelliklerini (her mesafe için 4 açının ortalaması) ve entropiyi çıkarır.

    Açı ortalaması: veri setinde döndürülmüş kopyalar var; tek bir yönün özelliği hücrenin
    dönme açısına bağlı olurdu. 0°/45°/90°/135° ortalaması dönmeye (yaklaşık) duyarsızdır.

    Parametreler:
        image: RGB görüntü (referans ölçekte).
        nucleus: Çekirdek maskesi.

    Döndürür:
        doku_<özellik>_d<mesafe> ve doku_entropi anahtarlı sözlük.
    """
    gray = quantize_gray(image)
    props = glcm_properties(masked_glcm(gray, nucleus))
    out = {}
    for p, values in props.items():
        with np.errstate(invalid="ignore"):
            per_distance = np.nanmean(values, axis=1) if np.isfinite(values).any() else np.full(len(GLCM_DISTANCES), np.nan)
        for d, v in zip(GLCM_DISTANCES, per_distance):
            out[f"doku_{p}_d{d}"] = float(v)
    hist = np.bincount(gray[nucleus], minlength=GLCM_LEVELS).astype(np.float64)
    prob = hist[hist > 0] / hist.sum()
    out["doku_entropi"] = float(-(prob * np.log2(prob)).sum())
    return out


# ---------------------------------------------------------------------------
# N/C oranı (meta veri)
# ---------------------------------------------------------------------------

def nc_ratio(crop: np.ndarray, coded: np.ndarray, rbc_diameter: float) -> tuple[float, str]:
    """
    Çekirdek / hücre (N/C) alan oranını YEREL çözünürlükte tahmin eder.

    Sitoplazma: çekirdeğe ≤ NC_MAX_DISTANCE_RATIO × d uzaklıkta, normalize ters b* değeri zeminden
    en az CYTOPLASM_MIN_CONTRAST yüksek (mavimsi), komşu çekirdek/dolgu olmayan ve çekirdeğe
    bağlı pikseller. N/C = A_çekirdek / (A_çekirdek + A_sitoplazma).

    Güvenilirlik: eritrosit çapı < NC_MIN_NATIVE_RBC_DIAMETER px ise (yaymalar) sitoplazma
    1-2 px'lik bir banttır ve sınır bulanıklığından ayrılamaz → NaN. Sitoplazma kırpım sınırına
    ulaşıyorsa (hücre kırpımdan taşıyor) → NaN.

    Parametreler:
        crop: Yerel çözünürlükte RGB kırpım.
        coded: Yerel çözünürlükte kodlu maske.
        rbc_diameter: Yerel eritrosit çapı (px).

    Döndürür:
        (N/C oranı veya NaN, durum: "guvenilir" / "dusuk_cozunurluk" / "sitoplazma_kesik" / "sitoplazma_yok").
    """
    if rbc_diameter < NC_MIN_NATIVE_RBC_DIAMETER:
        return np.nan, "dusuk_cozunurluk"
    nucleus = coded == MASK_NUCLEUS
    inv_b = 255 - cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 2].astype(np.float32)
    dist = cv2.distanceTransform((~nucleus).astype(np.uint8), cv2.DIST_L2, 5)
    candidate = ((inv_b >= BACKGROUND_INV_B + CYTOPLASM_MIN_CONTRAST) & (dist <= NC_MAX_DISTANCE_RATIO * rbc_diameter)
                 & (coded == 0))
    labels, _ = ndi.label(candidate | nucleus)
    cell = labels == labels[nucleus][0]
    cytoplasm = cell & ~nucleus
    if cytoplasm.sum() == 0:
        return np.nan, "sitoplazma_yok"
    border = np.zeros_like(cytoplasm)
    border[[0, -1], :] = border[:, [0, -1]] = True
    if (cytoplasm & border).any():
        return np.nan, "sitoplazma_kesik"
    return float(nucleus.sum() / cell.sum()), "guvenilir"


# ---------------------------------------------------------------------------
# Hücre ve tablo
# ---------------------------------------------------------------------------

def cell_features(cell: pd.Series) -> dict:
    """
    Tek bir hücrenin tüm özniteliklerini ve meta verisini çıkarır.

    Parametreler:
        cell: Hücre indeksi satırı (kirpim_yolu, maske_yolu, eritrosit_capi, ...).

    Döndürür:
        Öznitelik + meta sözlüğü.
    """
    crop = read_png(cell["kirpim_yolu"], color=True)
    coded = read_png(cell["maske_yolu"], color=False)
    native_nucleus = coded == MASK_NUCLEUS
    nc, nc_status = nc_ratio(crop, coded, cell["eritrosit_capi"])
    image, coded_ref, scale = to_reference_scale(crop, coded, cell["eritrosit_capi"])
    nucleus = coded_ref == MASK_NUCLEUS
    if nucleus.sum() < 3:  # Çok küçük çekirdek yeniden örneklemede kaybolursa en yakın piksel kullanılır
        coded_ref = cv2.resize(coded, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        nucleus = coded_ref == MASK_NUCLEUS
    nucleus = largest_component(nucleus)
    ring_coded = coded_ref.copy()
    ring_coded[(coded_ref == MASK_NUCLEUS) & ~nucleus] = 0  # Atılan küçük parçalar halkaya dahil olabilir
    ring = perinuclear_ring(ring_coded, FEATURE_REF_RBC_DIAMETER)
    props = regionprops(native_nucleus.astype(np.uint8))[0]
    return {
        **shape_features(nucleus, FEATURE_REF_RBC_DIAMETER),
        **color_features(image, nucleus, ring),
        **texture_features(image, nucleus),
        "meta_ham_alan_px": int(props.area), "meta_ham_cevre_px": float(props.perimeter),
        "meta_olcek_carpani": scale, "meta_halka_piksel": int(ring.sum()),
        "meta_nc_orani": nc, "meta_nc_durum": nc_status,
    }


def build_feature_table(method: str) -> pd.DataFrame:
    """
    Bir segmentasyon yönteminin tüm hücreleri için öznitelik tablosunu üretir ve kaydeder.

    Hücreler tek tek okunur; bellekte yalnızca öznitelik satırları tutulur.

    Parametreler:
        method: "kmeans", "otsu" veya "watershed".

    Döndürür:
        Öznitelik + meta veri DataFrame'i (data/processed/features_<yöntem>.csv).
    """
    cells = pd.read_csv(CELLS_INDEX_CSVS[method])
    images = pd.read_csv(DATASET_INDEX_CSV)[["dosya_yolu", "genislik", "yukseklik"]]
    rows, start = [], time.time()
    for i, (_, cell) in enumerate(cells.iterrows()):
        rows.append(cell_features(cell))
        if (i + 1) % 1000 == 0 or i + 1 == len(cells):  # 1000 hücrede bir (≈ 100 görüntü)
            print(f"  [öznitelik: {method}] {i + 1}/{len(cells)} hücre ({time.time() - start:.0f} sn)", flush=True)
    feats = pd.DataFrame(rows)
    meta = pd.DataFrame({
        "hucre_id": cells["hucre_id"], "etiket": cells["etiket"], "grup_id": cells["grup_id"],
        "meta_kaynak_dosya": cells["kaynak_dosya"], "meta_tip": cells["tip"], "meta_yontem": method,
        "meta_kenara_degiyor": cells["kenara_degiyor"], "meta_eritrosit_capi": cells["eritrosit_capi"],
        "meta_eritrosit_guvenilir": cells["eritrosit_tahmini_guvenilir"],
        "meta_kutu_x": cells["kutu_x"], "meta_kutu_y": cells["kutu_y"],
        "meta_kutu_w": cells["kutu_w"], "meta_kutu_h": cells["kutu_h"],
        "meta_kirpim_yolu": cells["kirpim_yolu"], "meta_maske_yolu": cells["maske_yolu"],
    })
    sizes = images.set_index("dosya_yolu").loc[cells["kaynak_dosya"]]
    meta["meta_goruntu_genislik"] = sizes["genislik"].to_numpy()
    meta["meta_goruntu_yukseklik"] = sizes["yukseklik"].to_numpy()
    table = pd.concat([meta, feats], axis=1)
    table.to_csv(FEATURES_CSVS[method], index=False, encoding="utf-8")
    return table


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Sınıflandırmada kullanılabilecek öznitelik sütunlarını döndürür (sekil_/renk_/doku_ önekli).

    Meta veri (meta_*), etiket, grup_id ve hucre_id bilinçli olarak hariçtir.

    Parametreler:
        df: Öznitelik tablosu.

    Döndürür:
        Sütun adları listesi.
    """
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


if __name__ == "__main__":
    for m in SEGMENTATION_METHODS:
        print(f"Öznitelik çıkarma ({m}) başlıyor; hücreler tek tek işleniyor...")
        t = build_feature_table(m)
        print(f"  {len(t)} hücre × {len(feature_columns(t))} öznitelik → {FEATURES_CSVS[m]}")
