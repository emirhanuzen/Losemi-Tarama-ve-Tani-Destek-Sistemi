"""
Segmentasyon yöntemlerinin (Otsu, K-means, Watershed) etiket gerektirmeyen karşılaştırması ve
sunum figürleri.

Elle işaretlenmiş çekirdek maskesi olmadığı için ölçütler etiketsizdir:
- Sınır uyumu: bulunan çekirdek dış sınırı boyunca ortalama Lab renk gradyanı (gerçek kenara
  oturan sınır yüksek gradyan üzerindedir).
- Bölge ayrışması: çekirdek ile arka plan arasında Fisher oranı (ters b*). Tek başına yanlıdır:
  maske çekirdeğin yalnızca en koyu merkezine daraldıkça artar; bu yüzden sınır uyumuyla birlikte okunur.
- Aşırı segmentasyon: küçük parça oranı (alanı < 0.5 eritrosit alanı olan nesneler).
- Eksik ayırma: birleşik kalma oranı (solidity < 0.85 olan, en az 0.5 eritrosit alanında nesneler;
  birbirine değen çekirdekler girintili, düşük solidity'li tek bölge oluşturur).
- Eksik segmentasyon: hiç çekirdek bulunamayan görüntü oranı (her görüntüde en az bir lökosit var).
- Sınıf tutarlılığı: sınıf medyanlarının değişim katsayısı (std / ortalama).
- Çalışma süresi.

Ölçümler her sınıftan farklı kopya gruplarından 30'ar görüntüde, yayma ve yakın çekim ayrı
raporlanır. Görüntüler tek tek işlenir.

Çalıştırma (proje kökünden):
    python -m src.segmentation_analysis
"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from src.config import (
    CELLS_INDEX_CSV,
    CLASS_NAMES,
    KMEANS_K,
    NUCLEUS_MIN_CONTRAST,
    RANDOM_STATE,
    RBC_FALLBACK_DIAMETER,
    SEGMENTATION_METHOD,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes
from src.preprocessing import preprocess
from src.preprocessing_analysis import distinct_groups, fisher_ratio, sample_distinct_groups
from src.segmentation import (
    BACKGROUND_INV_B,
    METHODS,
    estimate_rbc_diameter,
    image_rbc_diameter,
    kmeans_clusters,
    lab_features,
    otsu_threshold,
    rbc_area,
    segment_kmeans,
    watershed_stages,
)
from src.utils import load_image, save_figure, stage_dir

METHOD_NAMES = {"otsu": "Otsu", "kmeans": "K-means", "watershed": "Watershed"}
K_VALUES = (2, 3, 4, 5)
SMALL_FRAGMENT_RATIO = 0.5   # Alanı < 0.5 eritrosit alanı: çekirdek için çok küçük parça
MERGED_SOLIDITY = 0.85       # Tek çekirdekler genelde > 0.9; birleşik çekirdeklerin girintili sınırı düşürür
# Seçim kuralındaki ölçütler ve yönleri (True = yüksek iyi)
RANK_METRICS = {"sinir_gradyani": True, "fisher": True, "kucuk_parca_orani": False,
                "birlesik_orani": False, "bos_goruntu_orani": False, "sinif_tutarsizligi": False}
CONTOUR_COLORS = [(255, 214, 0), (0, 229, 255), (255, 64, 160), (120, 255, 80), (255, 140, 0)]


# ---------------------------------------------------------------------------
# Ölçütler
# ---------------------------------------------------------------------------

def lab_gradient(image: np.ndarray) -> np.ndarray:
    """
    Gerçek Lab ölçeğinde renk gradyanı büyüklüğünü (kanalların Sobel gradyanlarının Öklid normu) hesaplar.

    Renk gradyanı kullanılır ki yalnızca b* üzerinde çalışan yöntemler (Otsu, Watershed) kendi
    kanalında ölçülerek kayırılmasın.

    Parametreler:
        image: Renk normalize RGB görüntü.

    Döndürür:
        (H, W) float32 gradyan büyüklüğü.
    """
    lab = lab_features(image)
    return np.sqrt(sum(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
                       for c in range(3)))


def outer_boundary(labels: np.ndarray) -> np.ndarray:
    """
    Çekirdeklerin arka planla temas eden dış sınırını döndürür.

    Watershed'in çekirdekler arasına çizdiği 1 px'lik ayırma çizgileri 3x3 kapama ile
    kapatılır; böylece yalnızca çekirdek−arka plan sınırı ölçülür.

    Parametreler:
        labels: Etiketli maske.

    Döndürür:
        bool sınır maskesi.
    """
    closed = cv2.morphologyEx((labels > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return closed.astype(bool) & ~cv2.erode(closed, np.ones((3, 3), np.uint8)).astype(bool)


def object_solidity(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Her nesnenin alanını ve solidity'sini (alan / dışbükey örtü alanı) hesaplar.

    Parametreler:
        labels: Etiketli maske.

    Döndürür:
        (alanlar, solidity değerleri) dizileri.
    """
    areas, solidity = [], []
    for region, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        obj = (labels[sl] == region).astype(np.uint8)
        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
        areas.append(obj.sum())
        solidity.append(min(1.0, obj.sum() / hull) if hull > 0 else 1.0)
    return np.array(areas), np.array(solidity)


def segmentation_metrics(labels: np.ndarray, steps: dict, gradient: np.ndarray, rbc_diameter: float) -> dict:
    """
    Bir segmentasyon sonucunun etiketsiz ölçütlerini hesaplar.

    Parametreler:
        labels: Etiketli çekirdek maskesi.
        steps: preprocess() ara çıktıları.
        gradient: lab_gradient çıktısı.
        rbc_diameter: Eritrosit çapı (px).

    Döndürür:
        nesne_sayisi, bos, sinir_gradyani, fisher, kucuk_parca_orani, birlesik_orani sözlüğü
        (çekirdek bulunamadıysa oran ölçütleri NaN).
    """
    out = {"nesne_sayisi": int(labels.max()), "bos": int(labels.max() == 0)}
    if labels.max() == 0:
        return out
    mask = labels > 0
    valid = steps["gecerli_maske"]
    background = valid & ~cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    channel = steps["filtreli"].astype(np.float64)
    areas, solidity = object_solidity(labels)
    area_rbc = rbc_area(rbc_diameter)
    big = areas >= SMALL_FRAGMENT_RATIO * area_rbc
    out.update({
        "sinir_gradyani": float(gradient[outer_boundary(labels)].mean()),
        "fisher": fisher_ratio(channel[mask], channel[background]) if background.any() else np.nan,
        "kucuk_parca_orani": float((~big).mean()),
        "birlesik_orani": float((solidity[big] < MERGED_SOLIDITY).mean()) if big.any() else 0.0,
    })
    out.update(split_line_contrast(labels, channel))
    return out


def split_line_contrast(labels: np.ndarray, channel: np.ndarray) -> dict:
    """
    Watershed ayırma çizgilerinin kontrastını ölçer (aşırı bölünme tanısı).

    Doğru ayırma çizgisi iki çekirdek arasındaki daha soluk boşluktan (sitoplazma/sınır) geçer;
    tek bir çekirdeği bölen çizgi çekirdek içiyle aynı parlaklıktadır.
    kontrast = (medyan_çekirdek_içi − medyan_çizgi) / std_çekirdek_içi. Yüksek = doğru bölme.

    Parametreler:
        labels: Etiketli maske.
        channel: Filtrelenmiş ters b*.

    Döndürür:
        {"ayirma_cizgisi_kontrasti", "ayirma_cizgisi_px"} (çizgi yoksa yalnızca piksel sayısı 0).
    """
    mask = labels > 0
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    line = closed & ~mask
    if line.sum() < 3:
        return {"ayirma_cizgisi_px": int(line.sum())}
    inside = channel[mask]
    return {"ayirma_cizgisi_px": int(line.sum()),
            "ayirma_cizgisi_kontrasti": float((np.median(inside) - np.median(channel[line])) / max(inside.std(), 1.0))}


def run_comparison(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Örnekleri tek tek işleyip üç yöntemi ve K-means'in k değerlerini ölçer.

    Parametreler:
        samples: sample_distinct_groups çıktısı.

    Döndürür:
        Görüntü × yöntem başına bir satır.
    """
    rows, start = [], time.time()
    variants = {**{m: (fn, None) for m, fn in METHODS.items()},
                **{f"kmeans_k{k}": (segment_kmeans, k) for k in K_VALUES if k != KMEANS_K}}
    for i, (path, label, kind) in enumerate(zip(samples["dosya_yolu"], samples["etiket"], samples["tip"])):
        _, steps = preprocess(load_image(path))
        diameter, estimate = image_rbc_diameter(steps, kind)
        gradient = lab_gradient(steps["renk_normalize"])
        for name, (fn, k) in variants.items():
            t0 = time.perf_counter()
            labels = fn(steps, diameter) if k is None else fn(steps, diameter, k)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"goruntu": i, "etiket": label, "tip": kind, "yontem": name, "sure_ms": ms,
                         "eritrosit_capi": diameter, "eritrosit_guvenilir": estimate["guvenilir"],
                         **segmentation_metrics(labels, steps, gradient, diameter)})
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"  [karşılaştırma] {i + 1}/{len(samples)} ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    Yöntem × görüntü tipi (ve "tümü") özet tablosunu çıkarır.

    Sınıf tutarsızlığı: sınıf bazında sınır gradyanı ve Fisher medyanlarının değişim
    katsayılarının ortalaması (0 = her sınıfta aynı başarı).

    Parametreler:
        df: run_comparison çıktısı.
        methods: Özetlenecek yöntemler.

    Döndürür:
        (yontem, tip) indeksli özet tablo.
    """
    rows = []
    df = df[df["yontem"].isin(methods)]
    for kind, part in [("tümü", df)] + list(df.groupby("tip")):
        for method, g in part.groupby("yontem"):
            cls = g.groupby("etiket")[["sinir_gradyani", "fisher"]].median()
            cv = (cls.std(ddof=0) / cls.mean()).mean() if len(cls) > 1 else np.nan
            rows.append({"yontem": method, "tip": kind, "goruntu": len(g),
                         "sinir_gradyani": g["sinir_gradyani"].median(), "fisher": g["fisher"].median(),
                         "kucuk_parca_orani": g["kucuk_parca_orani"].mean(),
                         "birlesik_orani": g["birlesik_orani"].mean(),
                         "bos_goruntu_orani": g["bos"].mean(), "nesne_sayisi": g["nesne_sayisi"].median(),
                         "sinif_tutarsizligi": cv, "sure_ms": g["sure_ms"].median(),
                         "ayirma_cizgisi_kontrasti": g["ayirma_cizgisi_kontrasti"].median()
                         if "ayirma_cizgisi_kontrasti" in g else np.nan})
    return pd.DataFrame(rows).set_index(["yontem", "tip"]).round(3)


def class_table(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    Sınıf tutarlılığı için yöntem × sınıf bazında ana ölçütleri çıkarır.

    Parametreler:
        df: run_comparison çıktısı.
        methods: Yöntemler.

    Döndürür:
        (yontem, etiket) indeksli tablo.
    """
    g = df[df["yontem"].isin(methods)].groupby(["yontem", "etiket"])
    return g.agg(sinir_gradyani=("sinir_gradyani", "median"), fisher=("fisher", "median"),
                 kucuk_parca_orani=("kucuk_parca_orani", "mean"), birlesik_orani=("birlesik_orani", "mean"),
                 bos_goruntu_orani=("bos", "mean"), nesne_sayisi=("nesne_sayisi", "median")).round(3)


def select_method(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Seçim kuralı: her görüntü tipinde (yayma, yakın çekim) yöntemler RANK_METRICS'teki her
    ölçütte sıralanır (1 = en iyi, eşitlikte ortalama sıra); tipin ortalama sırası alınır.
    Nihai skor iki tipin ortalama sırasının ortalamasıdır (iki tip eşit ağırlıklı; yakın çekimler
    az olsa da ALL'ın yarısıdır). En düşük skor kazanır; eşitlikte daha hızlı yöntem seçilir.

    Parametreler:
        summary: summarize çıktısı (yalnızca üç ana yöntem).

    Döndürür:
        (seçilen yöntem, sıralama tablosu).
    """
    ranks = []
    for kind in ("yayma", "yakin_cekim"):
        part = summary.xs(kind, level="tip")
        r = pd.DataFrame({m: part[m].rank(ascending=not high) for m, high in RANK_METRICS.items()})
        r["ortalama_sira"] = r.mean(axis=1)
        ranks.append(r.assign(tip=kind))
    table = pd.concat(ranks).reset_index().set_index(["yontem", "tip"])
    score = table.groupby("yontem")["ortalama_sira"].mean().rename("nihai_skor").to_frame()
    score["sure_ms"] = summary.xs("tümü", level="tip")["sure_ms"]
    best = score.sort_values(["nihai_skor", "sure_ms"]).index[0]
    return best, table.join(score["nihai_skor"], on="yontem")


# ---------------------------------------------------------------------------
# Figür yardımcıları
# ---------------------------------------------------------------------------

def draw_contours(image: np.ndarray, labels: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Her çekirdeğin sınırını farklı renkle görüntünün üzerine çizer.

    Parametreler:
        image: RGB görüntü.
        labels: Etiketli maske.
        thickness: Çizgi kalınlığı (px).

    Döndürür:
        Kontur bindirmeli RGB kopya.
    """
    out = image.copy()
    for region in range(1, labels.max() + 1):
        contours, _ = cv2.findContours((labels == region).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, CONTOUR_COLORS[region % len(CONTOUR_COLORS)], thickness)
    return out


def _show(ax: plt.Axes, img: np.ndarray, title: str = "", fontsize: float = 8, cmap: str | None = None) -> None:
    """
    Görüntüyü eksende başlıklı ve çerçevesiz gösterir.

    Parametreler:
        ax: matplotlib ekseni.
        img: Görüntü.
        title: Başlık.
        fontsize: Başlık yazı boyutu.
        cmap: Tek kanallı görüntü için renk haritası (varsayılan gri).
    """
    ax.imshow(img, cmap=cmap or ("gray" if img.ndim == 2 else None))
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


def _prepare(row: pd.Series) -> tuple[dict, float, dict]:
    """
    Figür örneğini ön işler ve eritrosit çapını hesaplar.

    Parametreler:
        row: dataset_index satırı.

    Döndürür:
        (preprocess ara çıktıları, eritrosit çapı, eritrosit tahmini).
    """
    _, steps = preprocess(load_image(row["dosya_yolu"]))
    diameter, estimate = image_rbc_diameter(steps, row["tip"])
    return steps, diameter, estimate


def _examples(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Her sınıftan bir yayma ve (varsa) bir yakın çekim örneği seçer.

    Parametreler:
        groups: distinct_groups çıktısı.

    Döndürür:
        Seçilen satırlar.
    """
    picks = []
    for c in CLASS_NAMES:
        for kind in ("yayma", "yakin_cekim"):
            pool = groups[(groups["etiket"] == c) & (groups["tip"] == kind) & (groups["genislik"] <= 260)]
            if len(pool):
                picks.append(pool.sample(1, random_state=RANDOM_STATE))
    return pd.concat(picks)


# ---------------------------------------------------------------------------
# Figürler
# ---------------------------------------------------------------------------

def plot_method_comparison(examples: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Örnekler × (orijinal, Otsu, K-means, Watershed) kontur bindirmeli karşılaştırma figürü.

    Parametreler:
        examples: _examples çıktısı.
        summary: summarize çıktısı.
    """
    fig, axes = plt.subplots(len(examples), 4, figsize=(9.2, 2.35 * len(examples)), facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        steps, diameter, _ = _prepare(row)
        rgb = steps["renk_normalize"]
        _show(axes[r, 0], rgb, "Orijinal (renk normalize)" if r == 0 else "")
        _row_label(axes[r, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}\neritrosit ≈ {diameter:.0f} px")
        for c, method in enumerate(METHODS, start=1):
            labels = METHODS[method](steps, diameter)
            s = summary.loc[(method, "tümü")]
            title = (f"{METHOD_NAMES[method]}{' (seçilen)' if method == SEGMENTATION_METHOD else ''}\n"
                     f"gradyan {s['sinir_gradyani']:.0f} · Fisher {s['fisher']:.1f}") if r == 0 else ""
            _show(axes[r, c], draw_contours(rgb, labels), title, 7.5)
            axes[r, c].text(0.02, 0.02, f"{labels.max()} çekirdek", transform=axes[r, c].transAxes, fontsize=6.5,
                            color="white", va="bottom", bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
    fig.suptitle("Segmentasyon yöntemleri — kontur bindirmeli (her renk ayrı çekirdek); başlıklar 90 görüntü medyanı",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "yontem_karsilastirma")


def plot_otsu_histogram(examples: pd.DataFrame) -> None:
    """
    Ters b* (CLAHE) histogramı üzerinde Otsu eşiğini ve kontrast tabanını gösterir.

    Parametreler:
        examples: Örnek satırlar.
    """
    fig, axes = plt.subplots(len(examples), 2, figsize=(8.4, 2.3 * len(examples)), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 2.3]})
    for r, (_, row) in enumerate(examples.iterrows()):
        steps, diameter, _ = _prepare(row)
        valid, channel = steps["gecerli_maske"], steps["kontrast"]
        t = otsu_threshold(channel, valid)
        labels = METHODS["otsu"](steps, diameter)
        _show(axes[r, 0], draw_contours(steps["renk_normalize"], labels), "Otsu sonucu" if r == 0 else "")
        _row_label(axes[r, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}")
        ax = axes[r, 1]
        values = channel[valid]
        ax.hist(values[values <= t], bins=np.arange(0, 257, 2), color="#b9b7ae", label="eşik altı (arka plan)")
        ax.hist(values[values > t], bins=np.arange(0, 257, 2), color="#7b3fb8", label="eşik üstü (çekirdek adayı)")
        ax.axvline(t, color=CLASS_COLORS["AML"], lw=1.6)
        ax.text(t + 2, 0.92, f"Otsu = {t:.0f}", transform=ax.get_xaxis_transform(), fontsize=8, color=CLASS_COLORS["AML"])
        ax.set_yscale("log")
        ax.set_xlim(0, 255)
        _style_axes(ax)
        ax.tick_params(labelsize=7)
        frac = (values > t).mean()
        ax.set_title(f"eşik üstü piksel: %{100 * frac:.1f}", fontsize=8, color=TEXT_SECONDARY, loc="right")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1, 1].set_xlabel("Ters b* (CLAHE sonrası)", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle(f"Otsu eşiği — yalnızca geçerli pikseller; ayrıca normalize ters b* ≥ zemin + {NUCLEUS_MIN_CONTRAST} tabanı uygulanır",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "otsu_histogram")


def plot_kmeans_k(examples: pd.DataFrame, summary_k: pd.DataFrame) -> None:
    """
    Farklı k değerlerinde K-means küme haritalarını (küme merkez renkleriyle) ve seçilen çekirdek kümesini çizer.

    Parametreler:
        examples: Örnek satırlar.
        summary_k: summarize çıktısı (k varyantları dahil).
    """
    fig, axes = plt.subplots(2 * len(examples), len(K_VALUES) + 1, figsize=(2.1 * (len(K_VALUES) + 1), 4.3 * len(examples)),
                             facecolor=SURFACE)
    for e, (_, row) in enumerate(examples.iterrows()):
        steps, diameter, _ = _prepare(row)
        rgb, valid = steps["renk_normalize"], steps["gecerli_maske"]
        _show(axes[2 * e, 0], rgb, "Orijinal" if e == 0 else "")
        axes[2 * e + 1, 0].axis("off")
        _row_label(axes[2 * e, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}")
        for c, k in enumerate(K_VALUES, start=1):
            cmap, centers = kmeans_clusters(rgb, valid, k)
            lab8 = np.clip(np.column_stack([centers[:, 0] * 255 / 100, centers[:, 1:] + 128]), 0, 255).astype(np.uint8)
            colors = cv2.cvtColor(lab8[None], cv2.COLOR_LAB2RGB)[0]
            painted = np.zeros((*cmap.shape, 3), np.uint8)
            painted[cmap >= 0] = colors[cmap[cmap >= 0]]
            nucleus = int(np.argmin(centers[:, 2]))
            name = f"kmeans_k{k}" if k != KMEANS_K else "kmeans"
            s = summary_k.loc[(name, "tümü")]
            title = (f"k = {k}{' (seçilen)' if k == KMEANS_K else ''}\ngradyan {s['sinir_gradyani']:.0f} · "
                     f"Fisher {s['fisher']:.1f} · boş %{100 * s['bos_goruntu_orani']:.0f}") if e == 0 else ""
            _show(axes[2 * e, c], painted, title, 7)
            labels = segment_kmeans(steps, diameter, k)
            _show(axes[2 * e + 1, c], draw_contours(rgb, labels), "çekirdek kümesi (en düşük b*)" if e == 0 else "", 7)
            for j, col in enumerate(colors):  # Küme renk lejandı
                axes[2 * e, c].add_patch(plt.Rectangle((4 + 14 * j, 4), 12, 12, color=col / 255,
                                                       ec="white" if j == nucleus else "black", lw=2 if j == nucleus else 0.5))
    fig.suptitle("K-means k seçimi — üst: kümeler (merkez rengiyle boyalı, beyaz çerçeve = çekirdek kümesi); alt: çekirdek konturları",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "kmeans_k_secimi")


def plot_watershed_stages(examples: pd.DataFrame) -> None:
    """
    Watershed aşamalarını çizer: ikili maske → mesafe dönüşümü → markerlar → sonuç.

    Parametreler:
        examples: Örnek satırlar.
    """
    titles = ["Orijinal", "1. İkili maske (Otsu)", "2. Mesafe dönüşümü", "3. Markerlar (yerel maks.)", "4. Watershed sonucu"]
    fig, axes = plt.subplots(len(examples), 5, figsize=(11, 2.4 * len(examples)), facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        steps, diameter, _ = _prepare(row)
        st = watershed_stages(steps, diameter)
        rgb = steps["renk_normalize"]
        markers = rgb.copy()
        markers[st["ikili"]] = (markers[st["ikili"]] * 0.5 + 64).astype(np.uint8)
        for cy, cx in ndi.center_of_mass(st["markerlar"] > 0, st["markerlar"], range(1, st["markerlar"].max() + 1)):
            cv2.circle(markers, (int(round(cx)), int(round(cy))), max(2, round(diameter / 8)), (255, 40, 40), -1)
        panels = [rgb, st["ikili"].astype(np.uint8) * 255, st["mesafe"], markers, draw_contours(rgb, st["etiketler"])]
        for c, (img, t) in enumerate(zip(panels, titles)):
            _show(axes[r, c], img, t if r == 0 else "", 8, cmap="magma" if c == 2 else None)
        axes[r, 3].text(0.02, 0.02, f"{int(st['markerlar'].max())} marker", transform=axes[r, 3].transAxes,
                        fontsize=6.5, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        _row_label(axes[r, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}")
    fig.suptitle("Marker kontrollü watershed — markerlar mesafe dönüşümünün h-maksimumları "
                 "(h = 0.15 × eritrosit çapı)", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "watershed_asamalari")


def find_touching_examples(groups: pd.DataFrame, n: int = 4, pool_size: int = 60) -> list[tuple[pd.Series, tuple]]:
    """
    Otsu'nun tek nesne olarak birleştirdiği, Watershed'in birden fazla çekirdeğe ayırdığı bölgeleri bulur.

    Parametreler:
        groups: distinct_groups çıktısı.
        n: Döndürülecek örnek sayısı.
        pool_size: Taranacak yayma görüntüsü sayısı.

    Döndürür:
        [(satır, (y0, y1, x0, x1) kırpım kutusu)] listesi; en çok parçaya ayrılan bölgeler önce.
    """
    pool = groups[groups["tip"] == "yayma"].sample(pool_size, random_state=RANDOM_STATE)
    found = []
    for _, row in pool.iterrows():
        steps, diameter, _ = _prepare(row)
        otsu = METHODS["otsu"](steps, diameter)
        ws = watershed_stages(steps, diameter)["etiketler"]
        for region, sl in enumerate(ndi.find_objects(otsu), start=1):
            if sl is None:
                continue
            parts = len(np.setdiff1d(np.unique(ws[sl][otsu[sl] == region]), [0]))
            if parts >= 2:
                pad = round(diameter)
                box = (max(0, sl[0].start - pad), sl[0].stop + pad, max(0, sl[1].start - pad), sl[1].stop + pad)
                found.append((parts, row, box))
    found.sort(key=lambda f: -f[0])
    picked, used = [], set()
    for parts, row, box in found:  # Her görüntüden en fazla bir örnek
        if row["dosya_yolu"] not in used:
            picked.append((row, box))
            used.add(row["dosya_yolu"])
        if len(picked) == n:
            break
    return picked


def plot_touching(examples: list[tuple[pd.Series, tuple]]) -> None:
    """
    Dokunan çekirdekler: Otsu'nun birleştirdiği, Watershed'in ayırdığı bölgeleri yakınlaştırarak gösterir.

    Parametreler:
        examples: find_touching_examples çıktısı.
    """
    fig, axes = plt.subplots(len(examples), 3, figsize=(7.2, 2.5 * len(examples)), facecolor=SURFACE, squeeze=False)
    for r, (row, (y0, y1, x0, x1)) in enumerate(examples):
        steps, diameter, _ = _prepare(row)
        rgb = steps["renk_normalize"]
        otsu = METHODS["otsu"](steps, diameter)
        ws = watershed_stages(steps, diameter)["etiketler"]
        crop = (slice(y0, y1), slice(x0, x1))
        n_otsu = len(np.setdiff1d(np.unique(otsu[crop]), [0]))
        n_ws = len(np.setdiff1d(np.unique(ws[crop]), [0]))
        for c, (img, t) in enumerate([(rgb, "Orijinal"), (draw_contours(rgb, otsu), f"Otsu: {n_otsu} nesne"),
                                      (draw_contours(rgb, ws), f"Watershed: {n_ws} nesne")]):
            big = cv2.resize(img[crop], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            _show(axes[r, c], big, t, 8)
        _row_label(axes[r, 0], row["etiket"])
    fig.suptitle("Birbirine değen çekirdekler — Otsu tek bölge, Watershed ayrı çekirdekler (4x büyütülmüş)",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "dokunan_cekirdekler")


def plot_rbc_size(examples: pd.DataFrame) -> None:
    """
    Eritrosit çapı tahminini gösterir: görüntü, eritrosit maskesi, iç teğet çemberler ve tahmin.

    Parametreler:
        examples: Örnek satırlar (güvenilir ve güvenilmez tahminler dahil).
    """
    fig, axes = plt.subplots(len(examples), 3, figsize=(7.4, 2.45 * len(examples)), facecolor=SURFACE)
    for r, (_, row) in enumerate(examples.iterrows()):
        steps, diameter, est = _prepare(row)
        rgb, valid, mask = steps["renk_normalize"], steps["gecerli_maske"], est["maske"]
        overlay = rgb.copy()
        overlay[mask] = (overlay[mask] * 0.45 + np.array([40, 200, 90]) * 0.55).astype(np.uint8)
        circles = rgb.copy()
        src = mask.astype(np.uint8)
        src[~valid] = 1
        dist = cv2.distanceTransform(np.pad(src, 1, constant_values=1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        n, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for region in range(1, n):
            comp = np.where(labels == region, dist, 0)
            y, x = np.unravel_index(np.argmax(comp), comp.shape)
            if comp[y, x] >= 2:
                cv2.circle(circles, (int(x), int(y)), int(round(comp[y, x])), (255, 214, 0), 1)
        status = (f"tahmin {est['cap']:.1f} px" if est["guvenilir"]
                  else f"güvenilmez ({est['bilesen_sayisi']} bileşen, kapsama %{100 * est['kapsama']:.0f})\n"
                       f"→ yedek ({row['tip'].replace('_', ' ')}) {diameter:.0f} px")
        for c, (img, t) in enumerate([(rgb, "Orijinal"), (overlay, "Eritrosit maskesi (L, Otsu)"),
                                      (circles, "İç teğet çemberler")]):
            _show(axes[r, c], img, t if r == 0 else "", 8)
        axes[r, 2].text(1.03, 0.5, status, transform=axes[r, 2].transAxes, fontsize=7.5, va="center", color=TEXT_PRIMARY)
        _row_label(axes[r, 0], f"{row['etiket']} · {row['tip'].replace('_', ' ')}")
    fig.suptitle("Eritrosit çapı tahmini — iç teğet çember çaplarının alan ağırlıklı medyanı",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "eritrosit_boyutu")


def _rbc_examples(groups: pd.DataFrame, comparison: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """
    Eritrosit figürü için örnekler: her sınıftan bir yayma, bir yakın çekim ve bir güvenilmez tahmin.

    Parametreler:
        groups: distinct_groups çıktısı.
        comparison: run_comparison çıktısı.
        samples: Karşılaştırma örnekleri.

    Döndürür:
        Seçilen satırlar.
    """
    picks = [groups[(groups["etiket"] == c) & (groups["tip"] == "yayma") & (groups["genislik"] <= 260)]
             .sample(1, random_state=RANDOM_STATE) for c in CLASS_NAMES]
    picks.append(groups[(groups["etiket"] == "ALL") & (groups["tip"] == "yakin_cekim")].sample(1, random_state=RANDOM_STATE))
    bad = comparison[(comparison["yontem"] == SEGMENTATION_METHOD) & ~comparison["eritrosit_guvenilir"]]
    if len(bad):
        picks.append(samples.loc[[bad["goruntu"].iloc[0]]])
    return pd.concat(picks)


def compute_fallback_diameters(groups: pd.DataFrame) -> dict[str, float]:
    """
    Görüntü tipi başına güvenilir eritrosit çapı tahminlerinin medyanını hesaplar (yedek değer).

    Parametreler:
        groups: distinct_groups çıktısı (tüm farklı gruplar).

    Döndürür:
        {"yayma": çap, "yakin_cekim": çap} ve güvenilir oranları.
    """
    rows, start = [], time.time()
    for i, (path, kind) in enumerate(zip(groups["dosya_yolu"], groups["tip"])):
        _, steps = preprocess(load_image(path))
        est = estimate_rbc_diameter(steps["renk_normalize"], steps["gecerli_maske"])
        rows.append({"tip": kind, "cap": est["cap"], "guvenilir": est["guvenilir"]})
        if (i + 1) % 100 == 0 or i + 1 == len(groups):
            print(f"  [eritrosit] {i + 1}/{len(groups)} ({time.time() - start:.0f} sn)", flush=True)
    df = pd.DataFrame(rows)
    return df.groupby("tip").agg(cap=("cap", "median"), guvenilir_orani=("guvenilir", "mean"),
                                 goruntu=("cap", "size")).round(2)


def plot_cell_examples(cells: pd.DataFrame, per_class: int = 12) -> None:
    """
    Hücre indeksinden her sınıf ve görüntü tipinden rastgele hücre kırpımlarını (çekirdek konturuyla) çizer.

    Parametreler:
        cells: data/processed/cells_index.csv içeriği.
        per_class: Satır başına kırpım sayısı.
    """
    rows = [(c, t) for c in CLASS_NAMES for t in ("yayma", "yakin_cekim")
            if ((cells["etiket"] == c) & (cells["tip"] == t)).any()]
    fig, axes = plt.subplots(len(rows), per_class, figsize=(1.15 * per_class, 1.3 * len(rows)), facecolor=SURFACE)
    for r, (c, t) in enumerate(rows):
        pool = cells[(cells["etiket"] == c) & (cells["tip"] == t)]
        sample = pool.sample(min(per_class, len(pool)), random_state=RANDOM_STATE)
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k >= len(sample):
                continue
            cell = sample.iloc[k]
            crop = cv2.cvtColor(cv2.imdecode(np.fromfile(cell["kirpim_yolu"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = cv2.imdecode(np.fromfile(cell["maske_yolu"], np.uint8), cv2.IMREAD_GRAYSCALE) == 255
            scale = 96 / max(crop.shape[:2])  # Kırpımlar farklı boyutta; gösterim için eşitlenir
            size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, contours, -1, (255, 214, 0), 1)
            ax.imshow(crop)
            if cell["kenara_degiyor"]:
                ax.set_title("kenar", fontsize=6, color=CLASS_COLORS["AML"], pad=1)
        n = len(pool)
        _row_label(axes[r, 0], f"{c}\n{t.replace('_', ' ')} ({n})", 7)
    fig.suptitle(f"Çıkarılan hücre örnekleri ({SEGMENTATION_METHOD}; sarı = çekirdek sınırı, 'kenar' = görüntü/dolgu sınırına değiyor)",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "segmentation", "hucre_ornekleri")


def to_markdown(df: pd.DataFrame) -> str:
    """
    DataFrame'i (çok seviyeli indeks dahil) markdown tablosuna çevirir (ek bağımlılık gerektirmeden).

    Parametreler:
        df: Tablo.

    Döndürür:
        Markdown metni.
    """
    flat = df.reset_index()
    fmt = lambda v: f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, flat.columns)) + " |", "|" + "---|" * len(flat.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in flat.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Uçtan uca
# ---------------------------------------------------------------------------

def main() -> None:
    """Karşılaştırmayı çalıştırır, tabloları ve figürleri outputs/segmentation/ altına kaydeder."""
    pd.set_option("display.width", 220)
    out_dir = stage_dir("segmentation")
    groups = distinct_groups()
    print(f"Yedek eritrosit çapları {len(groups)} farklı gruptan hesaplanıyor...")
    fallback = compute_fallback_diameters(groups)
    print(fallback.to_string())
    for kind, cap in fallback["cap"].items():
        if abs(cap - RBC_FALLBACK_DIAMETER[kind]) > 0.5:
            print(f"  <-- {kind}: config.py'deki {RBC_FALLBACK_DIAMETER[kind]} değerinden farklı!")
    fallback.to_csv(out_dir / "eritrosit_capi_yedek.csv", encoding="utf-8")

    samples = sample_distinct_groups()
    print(f"{len(samples)} örnek (sınıf başına 30 farklı grup) tek tek işleniyor...")
    comparison = run_comparison(samples)
    comparison.to_csv(out_dir / "karsilastirma_goruntu.csv", index=False, encoding="utf-8")
    main_methods = list(METHODS)
    summary = summarize(comparison, main_methods)
    best, ranks = select_method(summary)
    summary.to_csv(out_dir / "karsilastirma.csv", encoding="utf-8")
    ranks.round(2).to_csv(out_dir / "secim_siralamasi.csv", encoding="utf-8")
    classes = class_table(comparison, main_methods)
    classes.to_csv(out_dir / "karsilastirma_sinif.csv", encoding="utf-8")
    k_names = ["kmeans" if k == KMEANS_K else f"kmeans_k{k}" for k in K_VALUES]
    summary_k = summarize(comparison, k_names)
    summary_k.to_csv(out_dir / "kmeans_k_karsilastirma.csv", encoding="utf-8")
    with open(out_dir / "karsilastirma.md", "w", encoding="utf-8") as f:
        f.write("# Segmentasyon karşılaştırması\n\n## Yöntem × görüntü tipi\n\n" + to_markdown(summary) +
                "\n\n## Sınıf bazında\n\n" + to_markdown(classes) + "\n\n## Seçim sıralaması\n\n" +
                to_markdown(ranks.round(2)) + f"\n\nSeçilen yöntem: **{best}**\n\n## K-means k\n\n" + to_markdown(summary_k) + "\n")
    print("\n=== Yöntem × tip ===\n" + summary.to_string())
    print("\n=== Sınıf bazında ===\n" + classes.to_string())
    print("\n=== Sıralama ===\n" + ranks.round(2).to_string())
    print(f"\nKurala göre seçilen yöntem: {best} (config: {SEGMENTATION_METHOD})")
    print("\n=== K-means k ===\n" + summary_k.to_string())

    examples = _examples(groups)
    plot_method_comparison(examples, summary)
    plot_otsu_histogram(examples[examples["tip"] == "yayma"].head(2).pipe(lambda d: pd.concat([d, examples[examples["tip"] == "yakin_cekim"].head(1)])))
    plot_kmeans_k(examples.iloc[[0, 1, 4]], summary_k)
    plot_watershed_stages(examples[examples["tip"] == "yayma"].head(2).pipe(lambda d: pd.concat([d, examples[examples["tip"] == "yakin_cekim"].tail(1)])))
    plot_touching(find_touching_examples(groups))
    plot_rbc_size(_rbc_examples(groups, comparison, samples))
    if CELLS_INDEX_CSV.exists():  # Hücre indeksi `python -m src.segmentation` ile üretilir
        plot_cell_examples(pd.read_csv(CELLS_INDEX_CSV))
    print("\nFigürler:", ", ".join(sorted(p.name for p in out_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
