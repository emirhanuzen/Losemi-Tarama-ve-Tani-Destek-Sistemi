"""
Aşama 1 - Ön İşleme.

Sırası: boyut standartlaştırma (kenar doldurmalı) → dolgu maskesi → renk normalizasyonu
(arka plan tabanlı Reinhard: zemin hizalama) → renk uzayı dönüşümü ve kanal seçimi
(Lab b*, ters) → gürültü azaltma (medyan) → kontrast iyileştirme (CLAHE).

Gürültü azaltma kontrast iyileştirmeden ÖNCE uygulanır: CLAHE yerel histogramı gerdiği için
önceden var olan JPEG/sensör gürültüsünü de büyütür.

Tüm parametreler ve seçim gerekçeleri src/config.py içindedir. Siyah dolgu bölgeleri (döndürme
augmentasyonu ve kenar doldurma) her adımda "geçerli piksel" maskesiyle dışlanır.
"""

import cv2
import numpy as np

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_PX,
    COLOR_CHANNEL,
    COLOR_NORM_METHOD,
    COLOR_SPACE,
    DARK_PADDING_MAX,
    DARK_PADDING_MAX_CHROMA,
    DARK_PADDING_MAX_STD,
    DARK_PADDING_MIN_AREA,
    DENOISE_METHOD,
    IMAGE_SIZE,
    INVERT_CHANNEL,
    MEDIAN_KSIZE,
    PADDING_THRESHOLD,
    PADDING_VALUE,
    REINHARD_BG_MARGIN,
    TARGET_LAB_ALL,
    TARGET_LAB_BACKGROUND,
)
from src.dataset import nucleus_mask
from src.utils import save_stage_output

# RGB'den dönüşüm kodları; "GRAY" tek kanallı döner
_COLOR_CODES = {"HSV": cv2.COLOR_RGB2HSV, "LAB": cv2.COLOR_RGB2LAB, "GRAY": cv2.COLOR_RGB2GRAY}


# ---------------------------------------------------------------------------
# Boyut ve maske
# ---------------------------------------------------------------------------

def resize_with_padding(image: np.ndarray, size: int = IMAGE_SIZE, pad_value: int = PADDING_VALUE) -> np.ndarray:
    """
    Görüntüyü en-boy oranını koruyarak size x size kareye sığdırır, boş kalan kenarları doldurur.

    Küçültmede INTER_AREA (piksel ortalaması, örtüşme/aliasing yapmaz), büyütmede INTER_CUBIC
    (yumuşak, basamaksız) kullanılır.

    Parametreler:
        image: RGB (veya tek kanallı) giriş görüntüsü.
        size: Hedef kare kenar uzunluğu (px).
        pad_value: Kenar doldurma değeri.

    Döndürür:
        (size, size[, 3]) uint8 görüntü; içerik ortalanmıştır.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    return cv2.copyMakeBorder(resized, top, size - new_h - top, left, size - new_w - left,
                              cv2.BORDER_CONSTANT, value=[pad_value] * (3 if image.ndim == 3 else 1))


def valid_pixel_mask(image: np.ndarray, threshold: int = PADDING_THRESHOLD) -> np.ndarray:
    """
    Dolgu olmayan (gerçek doku içeren) pikselleri işaretler.

    İki tür dolgu dışlanır:
    - Siyah dolgu: her kanalı threshold'un altında. Bazı augmentasyonlar siyah köşelere gürültü
      eklemiş; bu tek tük parlak pikseller morfolojik açma ile maskeden atılır.
    - Koyu renkli dolgu (görüntülerin ~%3'ü): döndürme köşeleri siyah yerine koyu gri/mor
      sabit renkle doldurulmuş. Koyu, dokusuz ve görüntü kenarına bağlı büyük bölgeler dolgu sayılır.

    Parametreler:
        image: RGB görüntü.
        threshold: Her kanalı bu değerin altındaki pikseller dolgu sayılır.

    Döndürür:
        (H, W) bool maske; True = geçerli piksel.
    """
    mask = (image.max(axis=2) >= threshold) & ~dark_padding(image)
    # 5x5 açma: dolgudaki 1-2 piksellik gürültü beneklerini siler; doku bölgesi geniş ve
    # içinde dolgu eşiğinin altında piksel olmadığı için sınırı en fazla köşelerde 1-2 px kırpar
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


def dark_padding(image: np.ndarray) -> np.ndarray:
    """
    Siyah olmayan sabit renkli (koyu) dolgu bölgelerini bulur.

    Koşullar: her kanal < DARK_PADDING_MAX (koyu), 5x5 yerel std < DARK_PADDING_MAX_STD (dokusuz;
    çekirdeklerde kromatin dokusu vardır), renk doygunluğu (maks − min kanal) <
    DARK_PADDING_MAX_CHROMA (nötr gri; koyu çekirdek doygun mordur), görüntü kenarına değen ve alanı görüntünün
    ≥ DARK_PADDING_MIN_AREA kadarı olan bağlı bileşen (döndürme köşeleri büyük üçgenlerdir).

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        bool maske; True = koyu dolgu.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - mean * mean, 0))
    chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2)
    candidate = ((image.max(axis=2) < DARK_PADDING_MAX) & (local_std < DARK_PADDING_MAX_STD)
                 & (chroma < DARK_PADDING_MAX_CHROMA))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(n, bool)
    for comp in border[border > 0]:
        keep[comp] = stats[comp, cv2.CC_STAT_AREA] >= DARK_PADDING_MIN_AREA * gray.size
    return keep[labels]


def fill_padding(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Dolgu piksellerini geçerli piksellerin medyanıyla doldurur.

    Filtre ve CLAHE dolgu sınırında sahte, çok güçlü bir kenar görmesin diye kullanılır.

    Parametreler:
        channel: Tek kanallı uint8 görüntü.
        mask: valid_pixel_mask çıktısı.

    Döndürür:
        Dolgusu doldurulmuş kopya.
    """
    out = channel.copy()
    if mask.any() and not mask.all():
        out[~mask] = np.uint8(np.median(channel[mask]))
    return out


def apply_mask(channel: np.ndarray, mask: np.ndarray, value: int = 0) -> np.ndarray:
    """
    Dolgu piksellerini sabit bir değere çeker (ara çıktıların görselleştirilmesi ve sonraki aşamalar için).

    Parametreler:
        channel: Tek kanallı görüntü.
        mask: valid_pixel_mask çıktısı.
        value: Dolgu piksellerine yazılacak değer (0: ters b* kanalında arka planla aynı yön).

    Döndürür:
        Maskelenmiş kopya.
    """
    out = channel.copy()
    out[~mask] = value
    return out


def background_of(nucleus: np.ndarray, valid: np.ndarray, margin: int = REINHARD_BG_MARGIN) -> np.ndarray:
    """
    Çekirdeğin geçiş halkası hariç geçerli arka plan (eritrosit + plazma) piksellerini döndürür.

    Parametreler:
        nucleus: Kaba çekirdek maskesi.
        valid: Geçerli piksel maskesi.
        margin: Çekirdek maskesinin genişletileceği kare çekirdek boyutu (px).

    Döndürür:
        Arka plan maskesi (bool).
    """
    grown = cv2.dilate(nucleus.astype(np.uint8), np.ones((margin, margin), np.uint8))
    return valid & ~grown.astype(bool)


# ---------------------------------------------------------------------------
# Renk normalizasyonu
# ---------------------------------------------------------------------------

def lab_statistics(image: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bir bölgedeki Lab kanallarının ortalama ve standart sapmasını hesaplar.

    Parametreler:
        image: RGB görüntü.
        region: İstatistiğin hesaplanacağı piksel maskesi.

    Döndürür:
        (ortalama (3,), std (3,)); sıra L, a, b (OpenCV 8 bit ölçeği).
    """
    px = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[region].astype(np.float64)
    return px.mean(axis=0), px.std(axis=0)


def normalization_region(image: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """
    Renk istatistiklerinin hesaplanacağı bölgeyi seçer.

    Klasik Reinhard tüm geçerli pikselleri kullanır. Arka plan tabanlı yöntemler kaba çekirdek
    maskesini (B−G + Otsu) dışlar: eritrosit ve plazma her sınıfta bulunur, çekirdek oranı ise
    sınıfa göre çok değişir (AML yaymaları çekirdekle dolu, Healthy'de neredeyse yok).

    Parametreler:
        image: RGB görüntü.
        mask: Geçerli piksel maskesi.
        method: Normalizasyon yöntemi.

    Döndürür:
        Bölge maskesi (bool); arka plan 100 pikselden azsa tüm geçerli pikseller.
    """
    if method == "reinhard":
        return mask
    region = background_of(nucleus_mask(image).astype(bool) & mask, mask)
    return region if region.sum() >= 100 else mask  # 100 px: kararlı ortalama/std için asgari örnek


def normalize_color(image: np.ndarray, mask: np.ndarray, method: str = COLOR_NORM_METHOD) -> np.ndarray:
    """
    Renk dağılımını hedef Lab istatistiklerine taşır (Reinhard ve varyantları).

    Reinhard: her Lab kanalı için x' = (x − μ_görüntü) / σ_görüntü · σ_hedef + μ_hedef.
    - "reinhard": klasik; istatistikler tüm geçerli piksellerden.
    - "reinhard_arka_plan": aynı formül, istatistikler yalnızca arka plandan.
    - "arka_plan_kaydirma": yalnızca ortalama hizalanır, x' = x − μ_arka_plan + μ_hedef
      (std ölçeklemesi yok).
    - "arka_plan_hizalama": zemin ortalaması hizalanır; L* çarpımsal (L' = L · μ_hedef / μ_arka_plan,
      pozlama farkı ölçek etkisidir, siyah 0'da kalır), a*/b* toplamsal (renk kayması).
      Seçim gerekçesi src/config.py içinde.
    - "yok": görüntü değiştirilmez.

    Parametreler:
        image: RGB görüntü.
        mask: Geçerli piksel maskesi (dolgu dönüşümden etkilenmez, 0 kalır).
        method: Yöntem adı.

    Döndürür:
        Normalize edilmiş RGB uint8 görüntü.
    """
    if method == "yok":
        return image.copy()
    region = normalization_region(image, mask, method)
    mean, std = lab_statistics(image, region)
    target = TARGET_LAB_ALL if method == "reinhard" else TARGET_LAB_BACKGROUND
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
    if method == "arka_plan_kaydirma":
        lab = lab - mean + np.asarray(target["ortalama"])
    elif method == "arka_plan_hizalama":
        lab[..., 0] *= target["ortalama"][0] / max(mean[0], 1e-6)
        lab[..., 1:] += np.asarray(target["ortalama"][1:]) - mean[1:]
    elif method in ("reinhard", "reinhard_arka_plan"):
        lab = (lab - mean) / (std + 1e-6) * np.asarray(target["std"]) + np.asarray(target["ortalama"])
    else:
        raise ValueError(f"Bilinmeyen renk normalizasyonu: {method}")
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out[~mask] = 0
    return out


# ---------------------------------------------------------------------------
# Renk uzayı
# ---------------------------------------------------------------------------

def convert_color_space(image: np.ndarray, target: str = COLOR_SPACE) -> np.ndarray:
    """
    RGB görüntüyü hedef renk uzayına dönüştürür.

    Parametreler:
        image: RGB giriş görüntüsü.
        target: "RGB", "HSV", "LAB" veya "GRAY".

    Döndürür:
        Dönüştürülmüş görüntü (GRAY için tek kanallı).
    """
    target = target.upper()
    if target == "RGB":
        return image.copy()
    return cv2.cvtColor(image, _COLOR_CODES[target])


def all_color_channels(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    RGB, HSV, Lab ve gri seviyedeki 10 kanalın hepsini ayrı ayrı döndürür (karşılaştırma için).

    Not: OpenCV'de H 0-179 aralığındadır ve dairesel bir açıdır (kırmızı 0 ve 179'da);
    Lab'da a* ve b* 128 merkezlidir (128 = nötr).

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        {"RGB-R": ..., "HSV-S": ..., "Lab-b": ..., "Gri": ...} sözlüğü.
    """
    channels = {}
    for space, names in (("RGB", "RGB"), ("HSV", "HSV"), ("LAB", ("L", "a", "b"))):
        converted = convert_color_space(image, space)
        prefix = "Lab" if space == "LAB" else space
        for k, name in enumerate(names):
            channels[f"{prefix}-{name}"] = converted[..., k]
    channels["Gri"] = convert_color_space(image, "GRAY")
    return channels


def extract_channel(image: np.ndarray, space: str = COLOR_SPACE, channel: int = COLOR_CHANNEL,
                    invert: bool = INVERT_CHANNEL) -> np.ndarray:
    """
    Seçilen renk uzayından tek bir kanalı çıkarır (gerekirse ters çevirir).

    Parametreler:
        image: RGB görüntü.
        space: Renk uzayı adı.
        channel: Kanal indeksi.
        invert: True ise 255 - kanal döndürülür (çekirdek parlak olsun diye).

    Döndürür:
        Tek kanallı uint8 görüntü.
    """
    converted = convert_color_space(image, space)
    out = converted if converted.ndim == 2 else converted[..., channel]
    return 255 - out if invert else out.copy()


# ---------------------------------------------------------------------------
# Gürültü azaltma
# ---------------------------------------------------------------------------

def reduce_noise(channel: np.ndarray, mask: np.ndarray | None = None, method: str = DENOISE_METHOD,
                 ksize: int = MEDIAN_KSIZE, sigma_color: float = 25) -> np.ndarray:
    """
    Tek kanallı görüntüdeki gürültüyü azaltır.

    - gaussian: komşuları uzaklığa göre ağırlıklı ortalar; gürültüyle birlikte kenarı da yumuşatır.
    - median: komşuluğun medyanını alır; tuz-biber/JPEG gürültüsünü siler, basamak kenarları korur.
    - bilateral: uzaklık + yoğunluk farkıyla ağırlıklandırır; yoğunluk farkı sigma_color'dan
      büyük kenarları korur.

    Parametreler:
        channel: Tek kanallı uint8 görüntü.
        mask: Geçerli piksel maskesi; verilirse dolgu önce medyanla doldurulur.
        method: "median", "gaussian" veya "bilateral".
        ksize: Çekirdek boyutu (tek sayı; bilateral için komşuluk çapı).
        sigma_color: Yalnızca bilateral için yoğunluk sigması.

    Döndürür:
        Filtrelenmiş uint8 görüntü.
    """
    src = fill_padding(channel, mask) if mask is not None else channel
    if method == "median":
        return cv2.medianBlur(src, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(src, (ksize, ksize), 0)  # sigma=0: OpenCV çekirdek boyutundan hesaplar
    if method == "bilateral":
        # sigmaSpace = ksize: uzamsal ağırlık komşuluk çapıyla orantılı kalsın
        return cv2.bilateralFilter(src, ksize, sigma_color, ksize)
    raise ValueError(f"Bilinmeyen filtre: {method}")


# ---------------------------------------------------------------------------
# Kontrast iyileştirme
# ---------------------------------------------------------------------------

def equalize_histogram(channel: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Global histogram eşitleme uygular; dönüşüm yalnızca geçerli piksellerin histogramından hesaplanır.

    Parametreler:
        channel: Tek kanallı uint8 görüntü.
        mask: Geçerli piksel maskesi (None ise tüm pikseller).

    Döndürür:
        Eşitlenmiş uint8 görüntü.
    """
    values = channel[mask] if mask is not None else channel.ravel()
    cdf = np.bincount(values, minlength=256).cumsum().astype(np.float64)
    cdf_min = cdf[cdf > 0][0]
    lut = np.clip((cdf - cdf_min) * 255 / max(cdf[-1] - cdf_min, 1), 0, 255).astype(np.uint8)
    return lut[channel]


def clahe_grid(size: int, tile_px: int = CLAHE_TILE_PX) -> tuple[int, int]:
    """
    İstenen karo piksel boyutundan OpenCV'nin tileGridSize (karo SAYISI) değerini hesaplar.

    Parametreler:
        size: Görüntü kenar uzunluğu (px).
        tile_px: Hedef karo kenar uzunluğu (px).

    Döndürür:
        (karo_sayısı, karo_sayısı).
    """
    n = max(1, round(size / tile_px))
    return n, n


def enhance_contrast(channel: np.ndarray, mask: np.ndarray | None = None, method: str = "clahe",
                     clip_limit: float = CLAHE_CLIP_LIMIT, tile_px: int = CLAHE_TILE_PX) -> np.ndarray:
    """
    Tek kanallı görüntünün kontrastını iyileştirir.

    Parametreler:
        channel: Tek kanallı uint8 görüntü.
        mask: Geçerli piksel maskesi; verilirse dolgu önce medyanla doldurulur.
        method: "clahe" veya "he" (global histogram eşitleme).
        clip_limit: CLAHE kırpma sınırı (ortalama bin yüksekliğinin katı).
        tile_px: CLAHE karo kenar uzunluğu (px).

    Döndürür:
        Kontrastı iyileştirilmiş uint8 görüntü.
    """
    if method == "he":
        return equalize_histogram(channel, mask)
    src = fill_padding(channel, mask) if mask is not None else channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=clahe_grid(channel.shape[0], tile_px))
    return clahe.apply(src)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, save_name: str | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Ön işleme adımlarını sırayla uygular; son görüntüyü ve tüm ara çıktıları döndürür.

    Adımlar: boyutlandırma (kenar doldurmalı) → geçerli piksel maskesi → renk normalizasyonu →
    Lab b* (ters) → medyan filtre → CLAHE → dolgunun 0'a çekilmesi.

    Maske normalizasyondan ÖNCE hesaplanır; sonraki aşamalar maskeyi yeniden hesaplamak yerine
    sözlükteki "gecerli_maske"yi kullanmalıdır (normalizasyon koyu pikselleri kaydırabilir).

    Parametreler:
        image: RGB giriş görüntüsü (herhangi bir boyutta).
        save_name: Verilirse ara çıktılar outputs/preprocessing/<save_name>_<adım>.png olarak kaydedilir.

    Döndürür:
        (son görüntü (IMAGE_SIZE, IMAGE_SIZE) uint8,
         {"orijinal", "boyutlandirilmis", "gecerli_maske", "renk_normalize", "kanal", "filtreli",
          "kontrast"} sözlüğü).
        "renk_normalize" RGB'dir; renk öznitelikleri için sonraki aşamalarda kullanılır.
    """
    resized = resize_with_padding(image)
    mask = valid_pixel_mask(resized)
    normalized = normalize_color(resized, mask)
    channel = extract_channel(normalized)
    filtered = reduce_noise(channel, mask)
    enhanced = enhance_contrast(filtered, mask)
    steps = {
        "orijinal": image,
        "boyutlandirilmis": resized,
        "gecerli_maske": mask,
        "renk_normalize": normalized,
        "kanal": apply_mask(channel, mask),
        "filtreli": apply_mask(filtered, mask),
        "kontrast": apply_mask(enhanced, mask),
    }
    if save_name:
        for name, img in steps.items():
            if name != "orijinal":
                save_stage_output(img, "preprocessing", f"{save_name}_{name}")
    return steps["kontrast"], steps
