"""
Ön işleme parametrelerinin karşılaştırmalı analizi ve sunum figürleri.

Karşılaştırılanlar: dolgu maskesi, renk normalizasyonu (Reinhard varyantları), renk kanalı,
kontrast iyileştirme (HE / CLAHE) ve gürültü filtresi. src/config.py içindeki ön işleme
parametreleri bu analizin sonuçlarına göre seçildi.

Ölçümler her sınıftan farklı kopya gruplarından seçilen örneklerde yapılır (aynı kaynağın
augmentasyonları sonuçları şişirmesin diye). Görüntüler tek tek okunur; bellekte yalnızca
skorlar ve sınıf başına toplam histogramlar tutulur.

Ayrışma ölçütleri için "referans çekirdek maskesi" gerekir; elle işaretlenmiş maske olmadığı
için iki bağımsız kaba maske kullanılır:
- B−G: Adım 1'deki B−G fark indeksi + Otsu (src.dataset.nucleus_mask)
- S: HSV doygunluk kanalı + Otsu (B−G'ye yakın kanalların lehine yanlılığı kontrol etmek için)

Çalıştırma (proje kökünden):
    python -m src.preprocessing_analysis
"""

import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_PX,
    CLASS_NAMES,
    COLOR_NORM_METHOD,
    DATASET_INDEX_CSV,
    IMAGE_SIZE,
    MEDIAN_KSIZE,
    PADDING_THRESHOLD,
    RANDOM_STATE,
    TARGET_LAB_ALL,
    TARGET_LAB_BACKGROUND,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes, nucleus_mask
from src.preprocessing import (
    all_color_channels,
    apply_mask,
    background_of,
    enhance_contrast,
    extract_channel,
    fill_padding,
    lab_statistics,
    normalization_region,
    normalize_color,
    preprocess,
    reduce_noise,
    resize_with_padding,
    valid_pixel_mask,
)
from src.utils import load_image, save_figure, stage_dir

# Sınıf başına 30 farklı grup: 90 görüntü medyan skorları kararlı kılmaya yeter, analiz ~15 sn sürer
SAMPLES_PER_CLASS = 30

# Karşılaştırılan renk normalizasyonları (ad → preprocessing.normalize_color yöntemi)
NORM_METHODS = {"Normalizasyonsuz": "yok", "Klasik Reinhard": "reinhard",
                "Arka plan Reinhard": "reinhard_arka_plan", "Arka plan kaydırma": "arka_plan_kaydirma",
                "Zemin hizalama": "arka_plan_hizalama"}
# Çekirdek kanalı figüründe gösterilen kanallar (renk_uzaylari.png'de 10 kanalın hepsi var)
NUCLEUS_CHANNELS = ["Gri", "HSV-S", "Lab-a", "Lab-b", "B−G"]
# Denenen kontrast ayarları: (clipLimit, karo_px). 256 px'de 64/32/16 px karo = 4/8/16 ızgara.
CLAHE_GRID = [(c, t) for c in (1.0, 1.5, 2.0, 3.0, 4.0) for t in (64, 32, 16)]
# Denenen filtreler: (ad, yöntem, çekirdek, sigma_color)
FILTERS = [
    ("Filtresiz", None, 0, 0),
    ("Gaussian 3x3", "gaussian", 3, 0), ("Gaussian 5x5", "gaussian", 5, 0),
    ("Medyan 3x3", "median", 3, 0), ("Medyan 5x5", "median", 5, 0), ("Medyan 7x7", "median", 7, 0),
    ("Bilateral d5 σc10", "bilateral", 5, 10), ("Bilateral d5 σc25", "bilateral", 5, 25),
    ("Bilateral d7 σc15", "bilateral", 7, 15), ("Bilateral d7 σc50", "bilateral", 7, 50),
]
NUCLEUS_COLOR, BACKGROUND_COLOR, PADDING_COLOR = "#7b3fb8", "#b9b7ae", "#eb6834"
CHOSEN_NORM = next(k for k, v in NORM_METHODS.items() if v == COLOR_NORM_METHOD)
CHOSEN_FILTER = f"Medyan {MEDIAN_KSIZE}x{MEDIAN_KSIZE}"


# ---------------------------------------------------------------------------
# Örnek seçimi ve ölçütler
# ---------------------------------------------------------------------------

def distinct_groups() -> pd.DataFrame:
    """
    Her kopya grubunun ilk görüntüsünü döndürür (aynı kaynağın augmentasyonları tekrar sayılmasın).

    Döndürür:
        dataset_index.csv satırlarından oluşan DataFrame.
    """
    return pd.read_csv(DATASET_INDEX_CSV).drop_duplicates("grup_id")


def sample_distinct_groups(n_per_class: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Her sınıftan farklı kopya gruplarına ait rastgele örnekler seçer.

    Parametreler:
        n_per_class: Sınıf başına örnek sayısı.

    Döndürür:
        dataset_index.csv satırlarından oluşan DataFrame (indeks 0..N-1).
    """
    first = distinct_groups()
    return pd.concat([first[first["etiket"] == c].sample(n_per_class, random_state=RANDOM_STATE)
                      for c in CLASS_NAMES], ignore_index=True)


def nucleus_channels(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Çekirdek kanalı karşılaştırmasındaki kanalları döndürür (10 standart kanal + B−G indeksi).

    Parametreler:
        image: RGB görüntü.

    Döndürür:
        {kanal adı: tek kanallı uint8 görüntü}.
    """
    channels = all_color_channels(image)
    channels["B−G"] = np.clip(image[..., 2].astype(np.int16) - image[..., 1], 0, 255).astype(np.uint8)
    return channels


def reference_masks(rgb: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """
    Ayrışma ölçümü için iki bağımsız kaba çekirdek maskesi üretir.

    Parametreler:
        rgb: Boyutlandırılmış RGB görüntü.
        valid: Geçerli piksel maskesi.

    Döndürür:
        {"B-G": maske, "S": maske} (bool).
    """
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    t, _ = cv2.threshold(sat[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3x3 açma: tek piksellik JPEG lekelerini siler
    s_mask = cv2.morphologyEx(((sat > t) & valid).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return {"B-G": nucleus_mask(rgb).astype(bool) & valid, "S": s_mask.astype(bool)}


def fisher_ratio(fg: np.ndarray, bg: np.ndarray) -> float:
    """
    İki piksel kümesinin Fisher oranını hesaplar: (μ1 − μ2)² / (σ1² + σ2²).

    Büyük değer = sınıf ortalamaları uzak, sınıf içi dağılımlar dar → eşiklemeyle iyi ayrılır.
    Doğrusal (afin) yoğunluk dönüşümlerinden etkilenmez.

    Parametreler:
        fg: Ön plan (çekirdek) piksel değerleri.
        bg: Arka plan piksel değerleri.

    Döndürür:
        Fisher oranı.
    """
    fg, bg = fg.astype(np.float64), bg.astype(np.float64)
    return float((fg.mean() - bg.mean()) ** 2 / (fg.var() + bg.var() + 1e-9))


def otsu_separability(values: np.ndarray) -> float:
    """
    Otsu ayrışabilirlik ölçütü η = σ²_sınıflar_arası / σ²_toplam (0-1) hesaplar.

    Etiket gerektirmez; yalnızca histogramın iki moda ne kadar iyi bölündüğünü ölçer
    (ayrılan iki modun çekirdek/arka plan olduğunu garanti etmez).

    Parametreler:
        values: Geçerli piksel değerleri (uint8).

    Döndürür:
        η değeri.
    """
    t, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = values <= t
    w0 = low.mean()
    if w0 in (0.0, 1.0):
        return 0.0
    v = values.astype(np.float64)
    return float(w0 * (1 - w0) * (v[low].mean() - v[~low].mean()) ** 2 / max(v.var(), 1e-9))


def laplacian_noise(image: np.ndarray, region: np.ndarray) -> float:
    """
    Bölgedeki ortalama mutlak Laplace yanıtını (yüksek frekans / gürültü düzeyi) hesaplar.

    Parametreler:
        image: Tek kanallı görüntü.
        region: Ölçüm yapılacak bölge maskesi (düz arka plan).

    Döndürür:
        Ortalama |Laplace| değeri.
    """
    return float(np.abs(cv2.Laplacian(image.astype(np.float64), cv2.CV_64F, ksize=1))[region].mean())


def edge_metrics(filtered: np.ndarray, ring: np.ndarray, near: np.ndarray, valid: np.ndarray,
                 canny_low: float, canny_high: float) -> dict:
    """
    Filtre sonrası kenar ölçütlerini hesaplar.

    Parametreler:
        filtered: Filtrelenmiş tek kanallı görüntü.
        ring: Referans çekirdek sınırı (1-2 px halka).
        near: Sınırın ±2 px komşuluğu.
        valid: Geçerli piksel maskesi.
        canny_low, canny_high: Canny histerezis eşikleri (tüm filtrelerde aynı).

    Döndürür:
        sobel_sinir (sınırdaki ortalama gradyan), canny_hassasiyet (Canny kenarlarının sınıra
        yakın olan oranı), canny_duyarlilik (sınırın Canny ile yakalanan oranı) sözlüğü.
    """
    f = filtered.astype(np.float64)
    sobel = np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0), cv2.Sobel(f, cv2.CV_64F, 0, 1))
    edges = cv2.Canny(filtered, canny_low, canny_high).astype(bool) & valid
    edges_near = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return {"sobel_sinir": float(sobel[ring].mean()),
            "canny_hassasiyet": float((edges & near).sum() / max(edges.sum(), 1)),
            "canny_duyarlilik": float((ring & edges_near).sum() / max(ring.sum(), 1))}


def canny_thresholds(channel: np.ndarray) -> tuple[float, float]:
    """
    Görüntüye uyarlanmış Canny eşikleri: yüksek eşik = Otsu eşiğinin yarısı, alçak = yüksek / 2.

    Otsu eşiği kanalın dinamik aralığını yansıttığı için farklı kontrastlı görüntülerde
    karşılaştırılabilir kenar haritaları verir; 2:1 oran Canny'nin önerdiği aralıktadır.

    Parametreler:
        channel: Filtresiz tek kanallı görüntü.

    Döndürür:
        (alçak, yüksek) eşikler.
    """
    otsu, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(otsu * 0.5, 10.0)
    return high / 2, high


def lab_histograms(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Geçerli piksellerin L, a, b histogramlarını hesaplar (sınıf bazında toplanmak üzere).

    Parametreler:
        image: RGB görüntü.
        valid: Geçerli piksel maskesi.

    Döndürür:
        (3, 256) piksel sayıları.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[valid]
    return np.stack([np.bincount(lab[:, k], minlength=256) for k in range(3)])


def score_image(path: str) -> tuple[dict[str, list[dict]], dict[str, np.ndarray]] | None:
    """
    Tek bir görüntü için normalizasyon, kanal, kontrast ve filtre skorlarını hesaplar.

    Referans maskeler ham görüntüden çıkarılır; kanal/kontrast/filtre ölçümleri pipeline'daki
    gibi seçilen renk normalizasyonundan sonra yapılır.

    Parametreler:
        path: Görüntü yolu.

    Döndürür:
        ({"normalizasyon", "kanal", "kontrast", "gurultu"} satır listeleri,
         {normalizasyon adı: (3, 256) Lab histogramı}); çekirdek bulunamadıysa None.
    """
    raw = resize_with_padding(load_image(path))
    valid = valid_pixel_mask(raw)
    refs = reference_masks(raw, valid)
    nucleus = refs["B-G"]
    if nucleus.sum() < 20:  # 20 px: güvenilir ortalama/varyans için asgari çekirdek alanı
        return None
    bgs = {k: background_of(m, valid) for k, m in refs.items()}
    bg = bgs["B-G"]
    rows = {"normalizasyon": [], "kanal": [], "kontrast": [], "gurultu": []}
    hists = {}

    _, std_raw = lab_statistics(raw, normalization_region(raw, valid, "reinhard_arka_plan"))
    for name, method in NORM_METHODS.items():
        img = normalize_color(raw, valid, method)
        hists[name] = lab_histograms(img, valid)
        b = extract_channel(img)
        stretch = {"reinhard": TARGET_LAB_ALL["std"][2] / max(lab_statistics(raw, valid)[1][2], 1e-6),
                   "reinhard_arka_plan": TARGET_LAB_BACKGROUND["std"][2] / max(std_raw[2], 1e-6)}.get(method, 1.0)
        rows["normalizasyon"].append({
            "yontem": name, "fisher_b": fisher_ratio(b[nucleus], b[bg]),
            "zemin_b": float(np.median(b[bg])), "cekirdek_b": float(np.median(b[nucleus])),
            "fark": float(np.median(b[nucleus]) - np.median(b[bg])), "b_germe": stretch,
            # RGB'de 0/255'e kırpılan çekirdek pikseli oranı: Lab→RGB gamut kırpmasının göstergesi
            "rgb_kirpma_cekirdek": float(((img == 0) | (img == 255)).any(axis=2)[nucleus].mean())})

    rgb = normalize_color(raw, valid)
    for name, ch in nucleus_channels(rgb).items():
        row = {"kanal": name, "otsu_eta": otsu_separability(ch[valid])}
        for k, m in refs.items():
            if m.sum() >= 20:
                row[f"fisher_{k}"] = fisher_ratio(ch[m], ch[bgs[k]])
        rows["kanal"].append(row)

    channel = fill_padding(extract_channel(rgb), valid)
    filtered = reduce_noise(channel, valid)
    variants = {"Orijinal": filtered, "Histogram eşitleme": enhance_contrast(filtered, valid, method="he")}
    for clip, tile in CLAHE_GRID:
        variants[f"CLAHE c{clip:g} {tile}px"] = enhance_contrast(filtered, valid, clip_limit=clip, tile_px=tile)
    for name, out in variants.items():
        rows["kontrast"].append({"yontem": name, "fisher": fisher_ratio(out[nucleus], out[bg]),
                                 "fark": float(out[nucleus].mean() - out[bg].mean()),
                                 "bg_gurultu": laplacian_noise(out, bg)})

    k3 = np.ones((3, 3), np.uint8)
    n8 = nucleus.astype(np.uint8)
    ring = (cv2.dilate(n8, k3) - cv2.erode(n8, k3)).astype(bool)
    near = cv2.dilate(ring.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    low, high = canny_thresholds(channel)
    for name, method, ksize, sc in FILTERS:
        out = channel if method is None else reduce_noise(channel, valid, method, ksize, sc)
        rows["gurultu"].append({"filtre": name, "bg_gurultu": laplacian_noise(out, bg),
                                **edge_metrics(out, ring, near, valid, low, high)})
    return rows, hists


def run_scores(samples: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Örnekleri tek tek işleyip skor tablolarını ve sınıf bazında toplam Lab histogramlarını üretir.

    Parametreler:
        samples: sample_distinct_groups çıktısı.

    Döndürür:
        ({"normalizasyon", "kanal", "kontrast", "gurultu"} → DataFrame,
         {normalizasyon adı: {sınıf: (3, 256) histogram}}).
    """
    tables = {"normalizasyon": [], "kanal": [], "kontrast": [], "gurultu": []}
    hists = {n: {c: np.zeros((3, 256)) for c in CLASS_NAMES} for n in NORM_METHODS}
    start = time.time()
    for k, (path, label, kind) in enumerate(zip(samples["dosya_yolu"], samples["etiket"], samples["tip"])):
        result = score_image(path)
        if result is not None:
            rows, h = result
            for t, rs in rows.items():
                tables[t].extend({"etiket": label, "tip": kind, "goruntu": k, **r} for r in rs)
            for n, arr in h.items():
                hists[n][label] += arr
        if (k + 1) % 100 == 0 or k + 1 == len(samples):
            print(f"  [skor] {k + 1}/{len(samples)} ({time.time() - start:.0f} sn)", flush=True)
    return {t: pd.DataFrame(r) for t, r in tables.items()}, hists


def compute_lab_targets(groups: pd.DataFrame) -> dict[str, dict]:
    """
    Reinhard hedef istatistiklerini tüm farklı gruplardan sınıf dengeli olarak hesaplar.

    Her görüntünün Lab ortalama/std'si alınır; önce sınıf içinde, sonra sınıflar arasında
    ortalanır (600'er görüntülük ama farklı grup sayısı farklı sınıflar eşit ağırlık alsın diye).

    Parametreler:
        groups: distinct_groups çıktısı.

    Döndürür:
        {"tum": {"ortalama", "std"}, "arka_plan": {"ortalama", "std"}} (config ile karşılaştırma için).
    """
    rows, start = [], time.time()
    for k, (path, label) in enumerate(zip(groups["dosya_yolu"], groups["etiket"])):
        img = resize_with_padding(load_image(path))
        valid = valid_pixel_mask(img)
        for region_name, method in (("tum", "reinhard"), ("arka_plan", "reinhard_arka_plan")):
            mean, std = lab_statistics(img, normalization_region(img, valid, method))
            rows.append({"bolge": region_name, "etiket": label, **{f"m{i}": v for i, v in enumerate(mean)},
                         **{f"s{i}": v for i, v in enumerate(std)}})
        if (k + 1) % 100 == 0 or k + 1 == len(groups):
            print(f"  [hedef] {k + 1}/{len(groups)} ({time.time() - start:.0f} sn)", flush=True)
    df = pd.DataFrame(rows)
    out = {}
    for region_name, g in df.groupby("bolge"):
        balanced = g.groupby("etiket").mean(numeric_only=True).mean()
        out[region_name] = {"ortalama": tuple(round(balanced[f"m{i}"], 2) for i in range(3)),
                            "std": tuple(round(balanced[f"s{i}"], 2) for i in range(3))}
    return out


def summarize_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renk normalizasyonu yöntemlerinin özetini çıkarır.

    Görüntüler arası std (zemin_b_std, cekirdek_b_std, fark_std) düşükse yöntem renkleri tutarlı
    hâle getiriyor demektir; sınıf bazında fark ve germe katsayısı sınıfa bağlı bozulmayı gösterir.

    Parametreler:
        df: run_scores()[0]["normalizasyon"].

    Döndürür:
        Yöntem başına özet tablo.
    """
    g = df.groupby("yontem")
    out = g[["fisher_b"]].median()
    out = out.join(g[["zemin_b", "cekirdek_b", "fark"]].std().add_suffix("_std"))
    out = out.join(df.pivot_table(index="yontem", columns="etiket", values="fark", aggfunc="median").add_prefix("fark_"))
    out = out.join(df.pivot_table(index="yontem", columns="etiket", values="b_germe", aggfunc="median").add_prefix("b_germe_"))
    return out.reindex(list(NORM_METHODS)).round(2)


def normalization_fisher_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ters b* kanalında çekirdek/arka plan Fisher oranını normalizasyon durumuna göre
    sınıf ve görüntü tipi bazında özetler (aynı görüntüler üzerinde).

    Parametreler:
        df: run_scores()[0]["normalizasyon"].

    Döndürür:
        Durum başına tümü / sınıf / tip medyanları, normalizasyonsuza göre ±%5'ten fazla artan ve
        azalan görüntü sayısı ve çekirdekte RGB kırpma oranı.
    """
    shown = ["Normalizasyonsuz", "Klasik Reinhard", CHOSEN_NORM]
    df = df[df["yontem"].isin(shown)]
    out = df.groupby("yontem")["fisher_b"].median().rename("tumu").to_frame()
    out = out.join(df.pivot_table(index="yontem", columns="etiket", values="fisher_b", aggfunc="median"))
    out = out.join(df.pivot_table(index="yontem", columns="tip", values="fisher_b", aggfunc="median"))
    wide = df.pivot_table(index="goruntu", columns="yontem", values="fisher_b")
    base = wide["Normalizasyonsuz"]
    out["artan"] = [int((wide[m] > base * 1.05).sum()) for m in out.index]
    out["azalan"] = [int((wide[m] < base * 0.95).sum()) for m in out.index]
    out = out.join(df.pivot_table(index="yontem", columns="etiket", values="rgb_kirpma_cekirdek", aggfunc="mean")
                   .add_prefix("rgb_kirpma_"))
    out["goruntu"] = len(wide)
    return out.reindex(shown).round(3)


def summarize_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kanal skorlarının medyan özetini çıkarır (B−G referanslı Fisher'a göre sıralı).

    Parametreler:
        df: run_scores()[0]["kanal"].

    Döndürür:
        Kanal başına özet tablo.
    """
    out = df.groupby("kanal")[["fisher_B-G", "fisher_S", "otsu_eta"]].median()
    per_class = df.pivot_table(index="kanal", columns="etiket", values="fisher_B-G", aggfunc="median")
    out = out.join(per_class.add_prefix("fisher_B-G_"))
    return out.sort_values("fisher_B-G", ascending=False).round(3)


def summarize_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kontrast yöntemlerinin medyan özetini çıkarır.

    Parametreler:
        df: run_scores()[0]["kontrast"].

    Döndürür:
        Yöntem başına fisher, fark, fark_p10 (düşük kontrastlı görüntüler), bg_gurultu ve tip bazlı Fisher.
    """
    g = df.groupby("yontem")
    out = g[["fisher", "fark", "bg_gurultu"]].median()
    out["fark_p10"] = g["fark"].quantile(0.1)
    out = out.join(df.pivot_table(index="yontem", columns="tip", values="fisher", aggfunc="median").add_prefix("fisher_"))
    return out.sort_values("fisher", ascending=False).round(3)


def summarize_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre skorlarının medyan özetini çıkarır; gradyan ve gürültü filtresiz görüntüye oranlanır.

    Parametreler:
        df: run_scores()[0]["gurultu"].

    Döndürür:
        Filtre başına gurultu_orani (düşük iyi), kenar_koruma (1'e yakın iyi), Canny ölçütleri.
    """
    base = df[df["filtre"] == "Filtresiz"].set_index("goruntu")
    df = df.assign(gurultu_orani=df["bg_gurultu"] / df["goruntu"].map(base["bg_gurultu"]),
                   kenar_koruma=df["sobel_sinir"] / df["goruntu"].map(base["sobel_sinir"]))
    cols = ["gurultu_orani", "kenar_koruma", "canny_hassasiyet", "canny_duyarlilik"]
    out = df.groupby("filtre")[cols].median()
    return out.reindex([f[0] for f in FILTERS]).round(3)


# ---------------------------------------------------------------------------
# Figür yardımcıları
# ---------------------------------------------------------------------------

def _pick_examples(samples: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """
    Figürler için her sınıftan örnek seçer: varsa bir büyük çözünürlüklü ve bir yakın çekim görüntü dahil.

    Parametreler:
        samples: Aday satırlar.
        per_class: Sınıf başına örnek sayısı.

    Döndürür:
        Seçilen satırlar.
    """
    rng = random.Random(RANDOM_STATE)
    picks = []
    for c in CLASS_NAMES:
        pool = samples[samples["etiket"] == c]
        chosen = []
        for cond in (pool["genislik"] > 260, pool["tip"] == "yakin_cekim"):
            cand = [i for i in pool.index[cond] if i not in chosen]
            if cand and len(chosen) < per_class:
                chosen.append(rng.choice(cand))
        rest = [i for i in pool.index[pool["tip"] == "yayma"] if i not in chosen]
        chosen += rng.sample(rest, per_class - len(chosen))
        picks += chosen
    return samples.loc[picks]


def _show(ax: plt.Axes, img: np.ndarray, title: str = "", fontsize: float = 8) -> None:
    """
    Görüntüyü eksende başlıklı ve çerçevesiz gösterir (tek kanal → gri tonlama 0-255).

    Parametreler:
        ax: matplotlib ekseni.
        img: Görüntü.
        title: Başlık.
        fontsize: Başlık yazı boyutu.
    """
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=fontsize, color=TEXT_PRIMARY, pad=3)
    ax.axis("off")


def _row_label(ax: plt.Axes, text: str, fontsize: float = 8) -> None:
    """
    Satırın ilk eksenine dikey satır etiketi yazar.

    Parametreler:
        ax: Satırın ilk ekseni.
        text: Etiket metni.
        fontsize: Yazı boyutu.
    """
    ax.text(-0.08, 0.5, text, transform=ax.transAxes, rotation=90, ha="right", va="center",
            fontsize=fontsize, color=TEXT_PRIMARY)


def _highlight(ax: plt.Axes) -> None:
    """
    Seçilen yöntemin eksenini turuncu çerçeveyle vurgular.

    Parametreler:
        ax: matplotlib ekseni.
    """
    ax.axis("on")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(PADDING_COLOR), s.set_linewidth(2.5)


def _label(row: pd.Series) -> str:
    """
    Örnek satırından "sınıf · tip" etiketi üretir.

    Parametreler:
        row: dataset_index satırı.

    Döndürür:
        Etiket metni.
    """
    return f"{row['etiket']}\n{row['tip'].replace('_', ' ')}"


def _histogram(ax: plt.Axes, img: np.ndarray, nucleus: np.ndarray, bg: np.ndarray) -> None:
    """
    Arka plan ve referans çekirdek piksellerinin histogramlarını üst üste çizer (log ölçek).

    Parametreler:
        ax: matplotlib ekseni.
        img: Tek kanallı görüntü.
        nucleus: Referans çekirdek maskesi.
        bg: Arka plan maskesi.
    """
    bins = np.arange(0, 257, 4)  # 4 gri seviyelik bin: 64 sütun, küçük figürde okunur
    ax.hist(img[bg], bins=bins, color=BACKGROUND_COLOR, label="arka plan")
    ax.hist(img[nucleus], bins=bins, color=NUCLEUS_COLOR, alpha=0.85, label="çekirdek")
    ax.set_yscale("log")
    ax.set_xlim(0, 255)
    _style_axes(ax)
    ax.tick_params(labelsize=6)


def _prepared(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Figürler için bir örneği pipeline'daki gibi hazırlar.

    Parametreler:
        row: dataset_index satırı.

    Döndürür:
        (normalize RGB, geçerli maske, referans çekirdek, arka plan, dolgusu doldurulmuş b* (ters)).
    """
    raw = resize_with_padding(load_image(row["dosya_yolu"]))
    valid = valid_pixel_mask(raw)
    nucleus = nucleus_mask(raw).astype(bool) & valid
    rgb = normalize_color(raw, valid)
    return rgb, valid, nucleus, background_of(nucleus, valid), fill_padding(extract_channel(rgb), valid)


# ---------------------------------------------------------------------------
# Figürler
# ---------------------------------------------------------------------------

def plot_padding_mask(examples: pd.DataFrame) -> None:
    """
    Dolgu maskesinin oluşturulmasını gösterir: görüntü, ham eşik, açma sonrası maske, maskelenmiş görüntü.

    Parametreler:
        examples: Örnek satırlar (döndürülmüş ve kenar doldurulmuş görüntüler).
    """
    titles = ["Boyutlandırılmış", f"Ham eşik (her kanal ≥ {PADDING_THRESHOLD})", "Maske (5x5 açma sonrası)",
              "Maskelenmiş (turuncu = dışlanan)"]
    fig, axes = plt.subplots(len(examples), 4, figsize=(9.2, 2.45 * len(examples)), facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        img = resize_with_padding(load_image(row["dosya_yolu"]))
        raw = img.max(axis=2) >= PADDING_THRESHOLD
        mask = valid_pixel_mask(img)
        shown = img.copy()
        shown[~mask] = tuple(int(PADDING_COLOR[i:i + 2], 16) for i in (1, 3, 5))
        h, w = load_image(row["dosya_yolu"]).shape[:2]
        speckles = int((raw & ~mask).sum())
        for c, (im, t) in enumerate(zip([img, raw.astype(np.uint8) * 255, mask.astype(np.uint8) * 255, shown], titles)):
            _show(axes[r, c], im, t if r == 0 else "", 8)
        axes[r, 2].text(0.5, -0.06, f"silinen benek: {speckles} px", transform=axes[r, 2].transAxes,
                        ha="center", va="top", fontsize=7, color=TEXT_SECONDARY)
        _row_label(axes[r, 0], f"{row['etiket']} · {w}x{h}\ndolgu %{100 * (~mask).mean():.0f}")
    fig.suptitle("Dolgu maskesi: döndürme köşeleri ve kenar doldurma tüm hesaplardan dışlanır",
                 fontsize=10.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "dolgu_maskesi")


def plot_color_normalization(examples: pd.DataFrame, norm_summary: pd.DataFrame) -> None:
    """
    Renk normalizasyonu yöntemlerini aynı örnekler üzerinde önce/sonra olarak karşılaştırır.

    Parametreler:
        examples: Farklı renk tonlarına sahip örnek satırlar.
        norm_summary: summarize_normalization çıktısı.
    """
    fig, axes = plt.subplots(len(NORM_METHODS), len(examples), figsize=(1.85 * len(examples), 2.15 * len(NORM_METHODS)),
                             facecolor=SURFACE)
    for c, (_, row) in enumerate(examples.iterrows()):
        raw = resize_with_padding(load_image(row["dosya_yolu"]))
        valid = valid_pixel_mask(raw)
        for r, (name, method) in enumerate(NORM_METHODS.items()):
            _show(axes[r, c], normalize_color(raw, valid, method), f"{row['etiket']} · {row['tip'].replace('_', ' ')}" if r == 0 else "", 7)
            if name == CHOSEN_NORM:
                _highlight(axes[r, c])
    for r, name in enumerate(NORM_METHODS):
        s = norm_summary.loc[name]
        _row_label(axes[r, 0], f"{name}{' (seçilen)' if name == CHOSEN_NORM else ''}\n"
                               f"zemin b* std {s['zemin_b_std']:.1f} · çekirdek b* std {s['cekirdek_b_std']:.1f}", 7)
    fig.suptitle("Renk normalizasyonu (Reinhard varyantları) — std: 90 görüntü arasında b* medyanlarının "
                 "yayılımı (düşük = tutarlı)", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "renk_normalizasyonu")


def plot_color_distribution(hists: dict) -> None:
    """
    Sınıf bazında L, a, b histogramlarını normalizasyon öncesi ve sonrası için çizer.

    Parametreler:
        hists: run_scores()[1] (normalizasyon adı → sınıf → (3, 256) histogram).
    """
    shown = ["Normalizasyonsuz", "Klasik Reinhard", CHOSEN_NORM]
    # a* ve b* 128 (nötr) çevresinde dar bir aralıkta; okunabilirlik için yalnızca dolu aralık gösterilir
    ranges = [(0, 255), (100, 180), (90, 170)]
    fig, axes = plt.subplots(3, len(shown), figsize=(3.4 * len(shown), 6.6), facecolor=SURFACE, sharex="row")
    x = np.arange(256)
    for c, name in enumerate(shown):
        for k, ch in enumerate(["L", "a*", "b*"]):
            ax = axes[k, c]
            for label in CLASS_NAMES:
                h = hists[name][label][k]
                ax.plot(x, h / max(h.sum(), 1), color=CLASS_COLORS[label], lw=1.6, label=label)
            ax.set_xlim(*ranges[k])
            _style_axes(ax)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(f"{ch}\n(yoğunluk)", fontsize=8, color=TEXT_SECONDARY)
            if k == 0:
                ax.set_title(f"{name}{' (seçilen)' if name == CHOSEN_NORM else ''}", fontsize=9.5, color=TEXT_PRIMARY)
    axes[0, 0].legend(frameon=False, fontsize=7.5, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Sınıf bazında Lab dağılımları (her sınıftan 30 farklı grup, geçerli pikseller)",
                 fontsize=10.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "renk_dagilimi")


def plot_color_spaces(examples: pd.DataFrame, channel_summary: pd.DataFrame) -> None:
    """
    Örnek görüntülerin 10 renk kanalını yan yana çizer; sütun başlığında medyan Fisher oranı yazar.

    Parametreler:
        examples: Her sınıftan birer örnek (satırlar).
        channel_summary: summarize_channels çıktısı.
    """
    order = list(all_color_channels(np.zeros((2, 2, 3), np.uint8)).keys())
    best = channel_summary.drop(index="B−G").index[0]
    fig, axes = plt.subplots(len(examples), len(order) + 1, figsize=(1.75 * (len(order) + 1), 1.95 * len(examples)),
                             facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        rgb = _prepared(row)[0]
        _show(axes[r, 0], rgb, "RGB (normalize)" if r == 0 else "")
        _row_label(axes[r, 0], _label(row))
        for c, (name, ch) in enumerate(all_color_channels(rgb).items(), start=1):
            _show(axes[r, c], ch, f"{name}\nF={channel_summary.at[name, 'fisher_B-G']:.1f}" if r == 0 else "")
            if name == best:
                _highlight(axes[r, c])
    fig.suptitle(f"Renk uzayları ve kanallar — F: çekirdek/arka plan Fisher oranı (90 görüntü medyanı); "
                 f"seçilen: {best}", fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "renk_uzaylari")


def plot_nucleus_channels(examples: pd.DataFrame, channel_summary: pd.DataFrame) -> None:
    """
    Çekirdek için aday kanalları (gri, HSV-S, Lab-a, Lab-b, B−G) yan yana çizer ve skor tablosunu ekler.

    Parametreler:
        examples: Her sınıftan birer örnek.
        channel_summary: summarize_channels çıktısı.
    """
    n = len(examples)
    fig = plt.figure(figsize=(1.95 * (len(NUCLEUS_CHANNELS) + 1), 2.05 * n + 1.9), facecolor=SURFACE)
    grid = fig.add_gridspec(n + 1, len(NUCLEUS_CHANNELS) + 1, height_ratios=[1] * n + [0.95])
    for r, (_, row) in enumerate(examples.iterrows()):
        rgb = _prepared(row)[0]
        ax = fig.add_subplot(grid[r, 0])
        _show(ax, rgb, "RGB (normalize)" if r == 0 else "")
        _row_label(ax, _label(row))
        channels = nucleus_channels(rgb)
        for c, name in enumerate(NUCLEUS_CHANNELS, start=1):
            ax = fig.add_subplot(grid[r, c])
            _show(ax, channels[name], name if r == 0 else "")
            if name == "Lab-b":
                _highlight(ax)
    table = channel_summary.loc[NUCLEUS_CHANNELS]
    ax = fig.add_subplot(grid[n, :])
    ax.axis("off")
    cells = [[f"{table.at[k, 'fisher_B-G']:.1f}" + (" *" if k == "B−G" else ""), f"{table.at[k, 'fisher_S']:.1f}",
              f"{table.at[k, 'otsu_eta']:.2f}"] + [f"{table.at[k, f'fisher_B-G_{c}']:.1f}" for c in CLASS_NAMES]
             for k in NUCLEUS_CHANNELS]
    tbl = ax.table(cellText=cells, rowLabels=NUCLEUS_CHANNELS, loc="center", cellLoc="center",
                   colLabels=["Fisher (B−G ref.)", "Fisher (S ref.)", "Otsu η"] + [f"Fisher {c}" for c in CLASS_NAMES])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e6e5e0")
        if r > 0 and NUCLEUS_CHANNELS[r - 1] == "Lab-b":
            cell.set_text_props(fontweight="bold")
    ax.text(0, -0.08, "* B−G referans maskesi B−G kanalından üretildiği için bu değer dairesel (kendini ölçüyor); "
            "bağımsız S referansına bakın.", transform=ax.transAxes, fontsize=7, color=TEXT_SECONDARY)
    fig.suptitle("Çekirdek kanalı seçimi — 90 görüntü (sınıf başına 30 farklı grup) medyanı",
                 fontsize=10.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "renk_kanallari")


def plot_contrast(examples: pd.DataFrame, contrast_summary: pd.DataFrame) -> None:
    """
    Seçilen kanalda orijinal, histogram eşitleme ve CLAHE clipLimit 1/2/4'ü görüntü + histogram olarak karşılaştırır.

    Parametreler:
        examples: Karşılaştırılacak örnek satırlar (2 adet).
        contrast_summary: summarize_contrast çıktısı.
    """
    chosen = f"CLAHE c{CLAHE_CLIP_LIMIT:g} {CLAHE_TILE_PX}px"
    methods = ["Orijinal", "Histogram eşitleme"] + [f"CLAHE c{c} {CLAHE_TILE_PX}px" for c in (1, 2, 4)]
    fig, axes = plt.subplots(2 * len(examples), len(methods), facecolor=SURFACE,
                             figsize=(2.3 * len(methods), 3.4 * len(examples)),
                             gridspec_kw={"height_ratios": [2.3, 1] * len(examples)})
    for e, (_, row) in enumerate(examples.iterrows()):
        _, valid, nucleus, bg, channel = _prepared(row)
        filtered = reduce_noise(channel, valid)
        for c, m in enumerate(methods):
            if m == "Orijinal":
                out = filtered
            elif m == "Histogram eşitleme":
                out = enhance_contrast(filtered, valid, method="he")
            else:
                clip, tile = m.split()[1:]
                out = enhance_contrast(filtered, valid, clip_limit=float(clip[1:]), tile_px=int(tile[:-2]))
            s = contrast_summary.loc[m]
            title = (f"{m.replace(f' {CLAHE_TILE_PX}px', '')}{' (seçilen)' if m == chosen else ''}\n"
                     f"F={s['fisher']:.1f}  gürültü={s['bg_gurultu']:.1f}") if e == 0 else ""
            _show(axes[2 * e, c], apply_mask(out, valid), title, 7.5)
            if m == chosen:
                _highlight(axes[2 * e, c])
            _histogram(axes[2 * e + 1, c], out, nucleus, bg)
        _row_label(axes[2 * e, 0], _label(row))
    axes[1, 0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(f"Kontrast iyileştirme (Lab b*, ters, medyan sonrası; CLAHE karo {CLAHE_TILE_PX} px) — "
                 "başlıklar 90 görüntü medyanı; histogramlar log ölçekli", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "kontrast")


def plot_clahe_grid(contrast_summary: pd.DataFrame) -> None:
    """
    CLAHE clipLimit × karo boyutu ızgarasındaki Fisher oranı ve gürültü düzeyini ısı haritası olarak çizer.

    Parametreler:
        contrast_summary: summarize_contrast çıktısı.
    """
    clips = sorted({c for c, _ in CLAHE_GRID})
    tiles = sorted({t for _, t in CLAHE_GRID}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, col, title, cmap in ((axes[0], "fisher", "Fisher oranı (yüksek iyi)", "Blues"),
                                 (axes[1], "bg_gurultu", "Arka plan gürültüsü (düşük iyi)", "Oranges")):
        grid = np.array([[contrast_summary.at[f"CLAHE c{c:g} {t}px", col] for t in tiles] for c in clips])
        ax.imshow(grid, cmap=cmap, aspect="auto")
        for i in range(len(clips)):
            for j in range(len(tiles)):
                chosen = clips[i] == CLAHE_CLIP_LIMIT and tiles[j] == CLAHE_TILE_PX
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if chosen else "normal",
                        color="white" if grid[i, j] > grid.mean() else TEXT_PRIMARY)
        ax.set_xticks(range(len(tiles)), [f"{t} px\n({IMAGE_SIZE // t}x{IMAGE_SIZE // t})" for t in tiles], fontsize=8)
        ax.set_yticks(range(len(clips)), [f"{c:g}" for c in clips], fontsize=8)
        ax.set_xlabel("Karo boyutu (tileGridSize)", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylabel("clipLimit", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(title, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
    o = contrast_summary.loc["Orijinal"]
    fig.suptitle(f"CLAHE parametre ızgarası (kalın: seçilen) — referans: orijinal F={o['fisher']:.1f}, "
                 f"gürültü={o['bg_gurultu']:.1f}", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", "clahe_parametre_izgarasi")


def _plot_filters(examples: pd.DataFrame, noise_summary: pd.DataFrame, shown: list[str], name: str, title: str) -> None:
    """
    Verilen filtrelerin sonuçlarını ve Canny kenar haritalarını yan yana çizer (merkez 128x128 bölge).

    Parametreler:
        examples: Örnek satırlar.
        noise_summary: summarize_noise çıktısı.
        shown: Gösterilecek filtre adları (FILTERS içinden).
        name: Kaydedilecek dosya adı.
        title: Figür başlığı.
    """
    spec = {f[0]: f for f in FILTERS}
    lo, hi = IMAGE_SIZE // 4, IMAGE_SIZE * 3 // 4
    fig, axes = plt.subplots(2 * len(examples), len(shown), figsize=(2.2 * len(shown), 4.6 * len(examples)),
                             facecolor=SURFACE)
    for e, (_, row) in enumerate(examples.iterrows()):
        _, valid, _, _, channel = _prepared(row)
        low, high = canny_thresholds(channel)
        for c, f in enumerate(shown):
            _, method, ksize, sc = spec[f]
            out = channel if method is None else reduce_noise(channel, valid, method, ksize, sc)
            edges = cv2.Canny(out, low, high) * valid
            s = noise_summary.loc[f]
            t = (f"{f}{' (seçilen)' if f == CHOSEN_FILTER else ''}\n"
                 f"gürültü×{s['gurultu_orani']:.2f}  kenar×{s['kenar_koruma']:.2f}") if e == 0 else ""
            _show(axes[2 * e, c], out[lo:hi, lo:hi], t, 7.5)
            _show(axes[2 * e + 1, c], 255 - edges[lo:hi, lo:hi], "Canny kenarları" if e == 0 else "", 7)
            if f == CHOSEN_FILTER:
                _highlight(axes[2 * e, c]), _highlight(axes[2 * e + 1, c])
        _row_label(axes[2 * e, 0], _label(row))
    fig.suptitle(title, fontsize=10.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "preprocessing", name)


def plot_noise_filters(examples: pd.DataFrame, noise_summary: pd.DataFrame) -> None:
    """
    Filtre TÜRLERİNİ (Gaussian, medyan 3x3/5x5, bilateral) ve Canny kenar haritalarını karşılaştırır.

    Parametreler:
        examples: Örnek satırlar.
        noise_summary: summarize_noise çıktısı.
    """
    _plot_filters(examples, noise_summary,
                  ["Filtresiz", "Gaussian 5x5", "Medyan 3x3", "Medyan 5x5", "Bilateral d5 σc10"], "gurultu_filtreleri",
                  "Gürültü filtreleri ve Canny kenar haritaları (Lab b*, ters; merkez 128x128) — "
                  "gürültü/kenar: filtresize oran, 90 görüntü medyanı")


def plot_noise_kernel_sizes(examples: pd.DataFrame, noise_summary: pd.DataFrame) -> None:
    """
    Gaussian ve medyan filtrede ÇEKİRDEK BOYUTUNUN (3/5/7) etkisini karşılaştırır.

    Parametreler:
        examples: Örnek satırlar.
        noise_summary: summarize_noise çıktısı.
    """
    _plot_filters(examples, noise_summary,
                  ["Filtresiz", "Gaussian 3x3", "Gaussian 5x5", "Medyan 3x3", "Medyan 5x5", "Medyan 7x7"], "gurultu",
                  "Çekirdek boyutu karşılaştırması (Lab b*, ters; merkez 128x128) — "
                  "gürültü/kenar: filtresize oran, 90 görüntü medyanı")


def plot_pipeline_examples(examples: pd.DataFrame) -> None:
    """
    Örnek görüntülerin tüm ön işleme aşamalarını (preprocess ara çıktıları) satır satır çizer.

    Parametreler:
        examples: Her sınıftan 3 örnek satır.
    """
    titles = {"orijinal": "Orijinal", "boyutlandirilmis": f"Boyutlandırılmış {IMAGE_SIZE}x{IMAGE_SIZE}",
              "gecerli_maske": "Geçerli piksel maskesi", "renk_normalize": "Renk normalizasyonu",
              "kanal": "Lab b* (ters)", "filtreli": f"Medyan {MEDIAN_KSIZE}x{MEDIAN_KSIZE}",
              "kontrast": f"CLAHE (clip {CLAHE_CLIP_LIMIT:g}, {CLAHE_TILE_PX} px)"}
    fig, axes = plt.subplots(len(examples), len(titles), figsize=(1.95 * len(titles), 2.0 * len(examples)),
                             facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        _, steps = preprocess(load_image(row["dosya_yolu"]))
        for c, (key, title) in enumerate(titles.items()):
            img = steps[key]
            if key == "orijinal":
                img = resize_with_padding(img, 512, 255)  # yalnızca gösterim: en-boy oranı görünsün
            elif key == "gecerli_maske":
                img = img.astype(np.uint8) * 255
            _show(axes[r, c], img, title if r == 0 else "", 7.5)
            if key == "gecerli_maske":  # Tamamen dolu (beyaz) maske figür zemininde kaybolmasın
                axes[r, c].add_patch(plt.Rectangle((-0.5, -0.5), img.shape[1], img.shape[0], fill=False,
                                                   edgecolor="#c3c2b7", linewidth=1))
        h, w = steps["orijinal"].shape[:2]
        _row_label(axes[r, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}\n{w}x{h}", 7.5)
        del steps
    fig.suptitle("Ön işleme pipeline'ı — her sınıftan 3 örnek", fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # Uzun figürde başlık sütun başlıklarına binmesin
    save_figure(fig, "preprocessing", "pipeline_ornekleri")


def _padding_examples(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Dolgu maskesi figürü için örnek seçer: en çok gürültü beneği olan döndürülmüş görüntü,
    başka sınıftan bir döndürülmüş görüntü ve kenar doldurulan büyük çözünürlüklü bir görüntü.

    Parametreler:
        groups: distinct_groups çıktısı.

    Döndürür:
        Seçilen satırlar.
    """
    rotated = groups[(groups["siyah_dolgu_orani"] > 0.05) & (groups["genislik"] <= 260)].sample(40, random_state=RANDOM_STATE)

    def speckles(path: str) -> int:
        img = resize_with_padding(load_image(path))
        return int(((img.max(axis=2) >= PADDING_THRESHOLD) & ~valid_pixel_mask(img)).sum())

    noisy = rotated.loc[[max(rotated.index, key=lambda i: speckles(rotated.at[i, "dosya_yolu"]))]]
    # Büyük görüntülerin döndürülmüş kopyaları grubun ilk görüntüsü olmadığı için ayrı seçilir
    other = rotated[rotated["etiket"] != noisy["etiket"].iloc[0]].sample(1, random_state=RANDOM_STATE)
    large = groups[(groups["genislik"] > 260) & (groups["siyah_dolgu_orani"] < 0.01)]
    return pd.concat([noisy, other, large.sample(1, random_state=RANDOM_STATE)])


def _normalization_examples(samples: pd.DataFrame, norm_scores: pd.DataFrame, per_class: int = 2) -> pd.DataFrame:
    """
    Renk normalizasyonu figürü için her sınıftan zemin rengi medyandan en uzak örnekleri seçer.

    Parametreler:
        samples: Skorlanan örnekler.
        norm_scores: run_scores()[0]["normalizasyon"].
        per_class: Sınıf başına örnek sayısı.

    Döndürür:
        Seçilen satırlar.
    """
    raw = norm_scores[norm_scores["yontem"] == "Normalizasyonsuz"].set_index("goruntu")
    dev = (raw["zemin_b"] - raw["zemin_b"].median()).abs()
    picks = []
    for c in CLASS_NAMES:
        idx = [i for i in dev.sort_values(ascending=False).index if samples.at[i, "etiket"] == c]
        picks += idx[:per_class]
    return samples.loc[picks]


# ---------------------------------------------------------------------------
# Uçtan uca
# ---------------------------------------------------------------------------

def main() -> None:
    """Skorları hesaplar, tabloları (CSV) ve karşılaştırma figürlerini outputs/preprocessing/ altına kaydeder."""
    pd.set_option("display.width", 220)
    groups = distinct_groups()
    print(f"Reinhard hedef istatistikleri {len(groups)} farklı gruptan hesaplanıyor...")
    targets = compute_lab_targets(groups)
    for region, cfg in (("arka_plan", TARGET_LAB_BACKGROUND), ("tum", TARGET_LAB_ALL)):
        drift = max(abs(a - b) for key in ("ortalama", "std") for a, b in zip(targets[region][key], cfg[key]))
        print(f"  {region}: {targets[region]}" + ("" if drift < 0.5 else "  <-- config.py'deki değerden farklı!"))

    samples = sample_distinct_groups()
    print(f"{len(samples)} örnek (sınıf başına {SAMPLES_PER_CLASS} farklı grup) tek tek işleniyor...")
    scores, hists = run_scores(samples)
    summaries = {"renk_normalizasyonu_skorlari": summarize_normalization(scores["normalizasyon"]),
                 "normalizasyon_fisher": normalization_fisher_table(scores["normalizasyon"]),
                 "kanal_skorlari": summarize_channels(scores["kanal"]),
                 "kontrast_skorlari": summarize_contrast(scores["kontrast"]),
                 "gurultu_skorlari": summarize_noise(scores["gurultu"])}
    out_dir = stage_dir("preprocessing")
    for name, table in summaries.items():
        table.to_csv(out_dir / f"{name}.csv", encoding="utf-8")
        print(f"\n=== {name} ===\n{table.to_string()}")

    channels = summaries["kanal_skorlari"]
    smear = groups[(groups["etiket"] == "AML") & (groups["tip"] == "yayma")].sample(1, random_state=RANDOM_STATE)
    closeup = groups[(groups["etiket"] == "ALL") & (groups["tip"] == "yakin_cekim")].sample(1, random_state=RANDOM_STATE)
    two = pd.concat([smear, closeup])
    plot_padding_mask(_padding_examples(groups))
    plot_color_normalization(_normalization_examples(samples, scores["normalizasyon"]),
                             summaries["renk_normalizasyonu_skorlari"])
    plot_color_distribution(hists)
    plot_color_spaces(_pick_examples(groups, 1), channels)
    plot_nucleus_channels(_pick_examples(groups, 1), channels)
    plot_contrast(two, summaries["kontrast_skorlari"])
    plot_clahe_grid(summaries["kontrast_skorlari"])
    plot_noise_filters(two, summaries["gurultu_skorlari"])
    plot_noise_kernel_sizes(two, summaries["gurultu_skorlari"])
    plot_pipeline_examples(_pick_examples(groups, 3))
    print("\nFigürler:", ", ".join(sorted(p.name for p in out_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
