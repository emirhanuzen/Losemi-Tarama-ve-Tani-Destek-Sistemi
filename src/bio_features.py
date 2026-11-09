"""
Adım 9 — Biyolojik öznitelikler (mevcut 58 özniteliğe ek, 15 öznitelik).

Hematologların blastı olgun hücreden ve lenfoblastı miyeloblasttan ayırırken baktığı dört morfolojik ipucu:
1. Nükleol (doku_bio_nukleol_*): blastlarda çekirdek içinde 1-5 belirgin, soluk/parlak nükleol bulunur; olgun
   lenfosit ve nötrofilde görünmez. Miyeloblastta nükleoller genellikle daha çok ve belirgin, lenfoblastta (L1)
   az ve silik.
2. Sitoplazma granülerliği (doku_bio_sitoplazma_*): miyeloid seri (miyeloblast, promiyelosit, nötrofil) azurofilik /
   nötrofilik granül taşır, Auer çubuğu AML'ye özgüdür; lenfoblast ve lenfosit sitoplazması agranülerdir.
3. Kromatin dokusu (doku_bio_lbp_*, doku_bio_glcm_d5_*): blast kromatini ince, dağınık ("ince retiküler");
   olgun lenfositte kaba, topaklı. Mevcut GLCM d=1-3 px'e ek olarak daha büyük ölçek (d=5) ve parlaklığa
   duyarsız LBP.
4. Çekirdek kenar düzensizliği (sekil_bio_*): miyeloblast/monoblast çekirdeğinde çentik ve kıvrım, segmentli
   nötrofilde lob; lenfoblast çekirdeği çoğunlukla yuvarlak / düzgün.

Tümü mevcut öznitelikler gibi referans ölçekte (eritrosit çapı 10 px; config.FEATURE_REF_RBC_DIAMETER) hesaplanır:
kaynaklar arasında büyütme farkı (Kaggle yayma ~10 px, Münih / ALL-IDB2 ~60 px) öznitelik değerine taşınmasın.
Kısıt: bu ölçekte bir nükleol (1-3 µm) ~1.5-4 px'tir; ölçüm çözünürlük sınırındadır.

Mutlak renk / zemin rengi KULLANILMAZ: tüm ölçüler yerel kontrasta (top-hat, Laplace, LBP) veya şekle dayanır. N/C
oranı eklenmedi: yalnızca yakın çekimlerde hesaplanabiliyor (config.NC_MIN_NATIVE_RBC_DIAMETER), varlığı görüntü
tipini ele verir.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.config import FEATURE_REF_RBC_DIAMETER, GLCM_ANGLES_DEG, GLCM_LEVELS
from src.features import largest_component, perinuclear_ring, quantize_gray, to_reference_scale
from src.segmentation import MASK_NUCLEUS

# Nükleol: beyaz top-hat yapı elemanı 5x5 elips (yarıçap 2 px = 0.2 d). Referans ölçekte nükleol ~1.5-4 px;
# 5 px'ten küçük parlak yapılar top-hat'te kalır, çekirdeğin genel parlaklık eğimi (≥ 6 px) çıkarılır.
NUCLEOLUS_SE = 5
# Top-hat eşiği 4 L* birimi (0-100 ölçeği): renk normalizasyonu sonrası zemin gürültüsü std ~3-5 (8 bit) ≈
# 1.2-2 L*; eşik gürültünün ~2-3 katı. Daha düşük eşik (2) kromatin dokusundaki her açık pikseli nükleol saydı.
NUCLEOLUS_MIN_TOPHAT = 4.0
# En az 2 piksel: tek piksel tepeler JPEG / yeniden örnekleme gürültüsüdür.
NUCLEOLUS_MIN_PIXELS = 2
# Granül: siyah top-hat 3x3 (yarıçap 1 px). Granüller (0.2-1 µm) referans ölçekte alt piksel; yalnızca topak hâlinde
# 1-2 px'lik koyu noktalar olarak görünür. Eşik nükleolle aynı gerekçeyle 4 L*.
GRANULE_SE = 3
GRANULE_MIN_TOPHAT = 4.0
# LBP: P=8, R=1 (en ince kromatin) ve P=16, R=2 (topak ölçeği); "uniform" kodlama dönmeye duyarsızdır (veri setinde
# döndürülmüş kopyalar var) ve monoton gri dönüşümlere (boya yoğunluğu, pozlama) değişmezdir.
LBP_SCALES = ((8, 1), (16, 2))
# GLCM d=5: referans ölçekte çekirdek çapı ~12-20 px; d=5 çekirdeğin çeyreği-üçte biri, topaklar arası ölçek
# (mevcut d=1-3 ince doku). Daha büyük mesafede çekirdek içinde kalan piksel çifti azalır.
GLCM_COARSE_DISTANCE = 5
# Kenar: maske 4 kat büyütülüp (doğrusal) 0.5'te eşiklenir; referans ölçekte ~40-60 px'lik konturun piksel
# basamaklanması (1 px'lik "çentikler") düzgünleşir. Radyal profil 64 noktada örneklenir (en fazla 32 harmonik).
CONTOUR_UPSAMPLE = 4
CONTOUR_SAMPLES = 64
# Fourier: 1. harmonik konum, 2. harmonik elipslik; ≥ 3. harmonikler lob / çentik / kıvrım (16'ya kadar: daha
# yüksek harmonikler 64 noktalı örneklemede piksel gürültüsüdür).
FOURIER_HIGH_FROM = 3
FOURIER_MAX = 16

BIO_FEATURES = [
    "doku_bio_nukleol_sayi", "doku_bio_nukleol_alan_orani", "doku_bio_nukleol_kontrast",
    "doku_bio_sitoplazma_laplace_std", "doku_bio_sitoplazma_koyu_nokta",
    "doku_bio_lbp_r1_entropi", "doku_bio_lbp_r1_duzensiz", "doku_bio_lbp_r2_entropi", "doku_bio_lbp_r2_duzensiz",
    "doku_bio_glcm_d5_contrast", "doku_bio_glcm_d5_homogeneity",
    "sekil_bio_konveksite", "sekil_bio_cukur_derinligi", "sekil_bio_yaricap_cv", "sekil_bio_fourier_yuksek",
]
BIO_GROUPS = {
    "nukleol": [c for c in BIO_FEATURES if "nukleol" in c],
    "sitoplazma_granul": [c for c in BIO_FEATURES if "sitoplazma" in c],
    "kromatin": [c for c in BIO_FEATURES if "lbp" in c or "glcm_d5" in c],
    "kenar_duzensizligi": [c for c in BIO_FEATURES if c.startswith("sekil_bio_")],
}


def reference_regions(crop: np.ndarray, coded: np.ndarray, rbc_diameter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kırpımı referans ölçeğe getirip çekirdek ve çekirdek çevresi halkasını features.crop_features ile AYNI
    adımlarla üretir (yeniden örnekleme, en büyük bileşen, halka).

    Parametreler:
        crop: Renk normalize RGB kırpım.
        coded: Kodlu maske.
        rbc_diameter: Kaynak görüntüdeki eritrosit çapı (px).

    Döndürür:
        (referans ölçekte RGB, çekirdek maskesi, halka maskesi).
    """
    image, coded_ref, _ = to_reference_scale(crop, coded, rbc_diameter)
    nucleus = coded_ref == MASK_NUCLEUS
    if nucleus.sum() < 3:
        coded_ref = cv2.resize(coded, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        nucleus = coded_ref == MASK_NUCLEUS
    nucleus = largest_component(nucleus)
    ring_coded = coded_ref.copy()
    ring_coded[(coded_ref == MASK_NUCLEUS) & ~nucleus] = 0
    return image, nucleus, perinuclear_ring(ring_coded, FEATURE_REF_RBC_DIAMETER)


def lightness(image: np.ndarray) -> np.ndarray:
    """
    L* kanalı (0-100, float32).

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        L* dizisi.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255


def nucleolus_features(L: np.ndarray, nucleus: np.ndarray) -> dict:
    """
    Çekirdek içi küçük parlak bölgeler (nükleol adayları): beyaz top-hat + mutlak eşik.

    Çekirdek maskesi doldurulur (soluk nükleol b* kanalında çekirdekten dışlanıp delik bırakabilir) ve 1 px
    aşındırılır (sınırdaki çekirdek → zemin geçişi top-hat'te parlak görünür).

    Parametreler:
        L: L* kanalı (referans ölçek).
        nucleus: Çekirdek maskesi.

    Döndürür:
        {sayı, alan oranı (çekirdek alanına), kontrast (adaylarda ortalama top-hat; aday yoksa 0)}.
    """
    filled = ndi.binary_fill_holes(nucleus)
    inner = cv2.erode(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    keys = BIO_GROUPS["nukleol"]
    if inner.sum() < 9:
        return dict.fromkeys(keys, np.nan)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (NUCLEOLUS_SE, NUCLEOLUS_SE))
    # Çekirdek dışı pikseller çekirdek medyanıyla doldurulur: zemin (çok parlak) top-hat'e sızmasın
    Lf = np.where(filled, L, np.median(L[filled])).astype(np.float32)
    tophat = cv2.morphologyEx(Lf, cv2.MORPH_TOPHAT, se)
    cand = inner & (tophat >= NUCLEOLUS_MIN_TOPHAT)
    labels, n = ndi.label(cand)
    if n:
        sizes = ndi.sum(cand, labels, range(1, n + 1))
        keep = np.isin(labels, np.flatnonzero(sizes >= NUCLEOLUS_MIN_PIXELS) + 1)
    else:
        keep = cand
    n_keep = int(ndi.label(keep)[1])
    return {keys[0]: float(n_keep), keys[1]: float(keep.sum() / filled.sum()),
            keys[2]: float(tophat[keep].mean()) if keep.any() else 0.0}


def granularity_features(L: np.ndarray, ring: np.ndarray) -> dict:
    """
    Çekirdek çevresi halkasında ince doku (Laplace std) ve küçük koyu nokta (siyah top-hat) yoğunluğu.

    Parametreler:
        L: L* kanalı (referans ölçek).
        ring: Halka maskesi (features.perinuclear_ring).

    Döndürür:
        {Laplace std, koyu nokta oranı}; halka < 5 pikselse NaN.
    """
    keys = BIO_GROUPS["sitoplazma_granul"]
    if ring.sum() < 5:
        return dict.fromkeys(keys, np.nan)
    lap = cv2.Laplacian(L, cv2.CV_32F, ksize=3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GRANULE_SE, GRANULE_SE))
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se)
    return {keys[0]: float(lap[ring].std()), keys[1]: float((blackhat[ring] >= GRANULE_MIN_TOPHAT).mean())}


def chromatin_features(image: np.ndarray, L: np.ndarray, nucleus: np.ndarray) -> dict:
    """
    Çok ölçekli kromatin dokusu: iki ölçekte uniform LBP (normalize entropi, uniform olmayan kod oranı) ve GLCM d=5
    contrast / homogeneity (4 açı ortalaması, mevcut nicemleme: features.quantize_gray).

    LBP çekirdeğin 1 px aşındırılmış içinde sayılır (sınır pikselleri komşularında zemini görür).

    Parametreler:
        image: RGB (referans ölçek).
        L: L* kanalı.
        nucleus: Çekirdek maskesi.

    Döndürür:
        doku_bio_lbp_* ve doku_bio_glcm_d5_* sözlüğü.
    """
    out = {}
    inner = cv2.erode(nucleus.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    for (p, r), tag in zip(LBP_SCALES, ("r1", "r2")):
        if inner.sum() < 5:
            out[f"doku_bio_lbp_{tag}_entropi"] = out[f"doku_bio_lbp_{tag}_duzensiz"] = np.nan
            continue
        codes = local_binary_pattern(L8, p, r, method="uniform")[inner].astype(int)
        hist = np.bincount(codes, minlength=p + 2).astype(np.float64)
        prob = hist[hist > 0] / hist.sum()
        out[f"doku_bio_lbp_{tag}_entropi"] = float(-(prob * np.log2(prob)).sum() / np.log2(p + 2))
        out[f"doku_bio_lbp_{tag}_duzensiz"] = float(hist[p + 1] / hist.sum())   # p+1: uniform olmayan kod
    gray = quantize_gray(image)
    coded = np.where(nucleus, gray, GLCM_LEVELS).astype(np.uint8)
    glcm = graycomatrix(coded, distances=[GLCM_COARSE_DISTANCE], angles=np.deg2rad(GLCM_ANGLES_DEG).tolist(),
                        levels=GLCM_LEVELS + 1, symmetric=True, normed=False)[:GLCM_LEVELS, :GLCM_LEVELS].astype(np.float64)
    totals = glcm.sum(axis=(0, 1))[0]
    for prop in ("contrast", "homogeneity"):
        if totals.sum() == 0:
            out[f"doku_bio_glcm_d5_{prop}"] = np.nan
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = graycoprops(glcm, prop)[0].astype(np.float64)
        out[f"doku_bio_glcm_d5_{prop}"] = float(np.nanmean(np.where(totals > 0, vals, np.nan)))
    return out


def boundary_features(nucleus: np.ndarray) -> dict:
    """
    Çekirdek kenar düzensizliği: konveksite (dışbükey örtü çevresi / çevre), en derin dışbükeylik kusuru
    (eşdeğer çapa oranlı), merkezden kontura radyal uzaklığın değişim katsayısı ve radyal profilin yüksek
    harmonik (≥ 3) enerji oranı.

    Parametreler:
        nucleus: Çekirdek maskesi (referans ölçek).

    Döndürür:
        sekil_bio_* sözlüğü; kontur çok küçükse NaN.
    """
    keys = BIO_GROUPS["kenar_duzensizligi"]
    filled = ndi.binary_fill_holes(nucleus).astype(np.float32)
    padded = np.pad(filled, 2)
    big = cv2.resize(padded, None, fx=CONTOUR_UPSAMPLE, fy=CONTOUR_UPSAMPLE, interpolation=cv2.INTER_LINEAR) >= 0.5
    contours, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return dict.fromkeys(keys, np.nan)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if len(cnt) < 12 or area <= 0:
        return dict.fromkeys(keys, np.nan)
    perim = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    convexity = float(cv2.arcLength(hull, True) / perim) if perim > 0 else np.nan
    eq_diam = np.sqrt(4 * area / np.pi)
    depth = 0.0
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:   # Kendini kesen kontur (nadir): kusur hesaplanamaz
            defects = None
        if defects is not None:
            depth = float(np.asarray(defects).reshape(-1, 4)[:, 3].max() / 256.0)   # 8.8 sabit noktalı derinlik
    pts = cnt[:, 0, :].astype(np.float64)
    # Kontur yay uzunluğuna göre eşit aralıklı yeniden örneklenir (köşe yoğunluğu profili bozmasın)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T))]
    t = np.linspace(0, seg[-1], CONTOUR_SAMPLES, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    xs, ys = np.interp(t, seg, closed[:, 0]), np.interp(t, seg, closed[:, 1])
    m = cv2.moments(cnt)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    radius = np.hypot(xs - cx, ys - cy)
    spec = np.abs(np.fft.rfft(radius - radius.mean())) ** 2
    total = spec[1:FOURIER_MAX + 1].sum()
    return {keys[0]: convexity, keys[1]: depth / eq_diam,
            keys[2]: float(radius.std() / radius.mean()),
            keys[3]: float(spec[FOURIER_HIGH_FROM:FOURIER_MAX + 1].sum() / total) if total > 0 else 0.0}


def bio_features(crop: np.ndarray, coded: np.ndarray, rbc_diameter: float) -> dict:
    """
    Bir hücre kırpımından 15 biyolojik özniteliği çıkarır (referans ölçekte).

    Parametreler:
        crop: Renk normalize RGB kırpım (segmentation.extract_cells "kirpim").
        coded: Kodlu maske ("maske").
        rbc_diameter: Kaynak görüntüdeki eritrosit çapı (px).

    Döndürür:
        BIO_FEATURES anahtarlı sözlük.
    """
    image, nucleus, ring = reference_regions(crop, coded, rbc_diameter)
    L = lightness(image)
    return {**nucleolus_features(L, nucleus), **granularity_features(L, ring),
            **chromatin_features(image, L, nucleus), **boundary_features(nucleus)}
