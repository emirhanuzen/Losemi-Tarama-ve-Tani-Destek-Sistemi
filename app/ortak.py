"""
Arayüz sayfalarının ortak yardımcıları: önbellekli yükleme ve analiz, görsel hazırlama, Türkçe öznitelik
adları, uyarı ve rozet bileşenleri.

Bu modül yeni bir görüntü işleme veya sınıflandırma algoritması içermez; analiz src/pipeline.py ve
src/risk_report.py üzerinden yapılır. Buradaki görsel işlemler (büyütme, sınır çizimi) yalnızca gösterim içindir.
"""

import hashlib
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.segmentation import find_boundaries
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    OUTPUTS_DIR,
    RISK_NO_CELL_LEVEL,
    RISK_RESULT_SUFFIX,
    STAGE_OUTPUT_DIRS,
)
from src.dataset import CLASS_COLORS
from src.utils import resolve_path

DISCLAIMER = ("**Araştırma ve eğitim amaçlı bir prototiptir; klinik teşhis aracı değildir.** "
              "Sonuçlar tıbbi karar için kullanılamaz; kesin tanı uzman hekim tarafından konur.")
CLASS_LABELS = {"ALL": "ALL (akut lenfoblastik lösemi)", "AML": "AML (akut miyeloid lösemi)",
                "Healthy": "Healthy (normal hücre)"}
# Risk rozeti renkleri: Düşük yeşil (Healthy sınıf rengiyle aynı), Orta turuncu, Yüksek kırmızı.
# Orta için AML'nin turuncusundan (#eb6834) ayırt edilebilen sarıya yakın bir turuncu seçildi.
RISK_COLORS = {"Düşük": "#1baf7a", "Orta": "#e08a00", "Yüksek": "#c0392b", RISK_NO_CELL_LEVEL: "#52514e"}
EXAMPLE_TYPE_NAMES = {"ALL": "ALL", "AML": "AML", "Healthy": "Healthy", "dusuk_guven": "Düşük güven"}
# Karşılaştırma görselinde kontur rengi: pembe/mor zeminde ve sınıf renklerinden ayrı görünen sarı
COMPARE_CONTOUR_COLOR = (255, 214, 0)
# Yüklenebilecek en küçük görüntü kenarı (px): daha küçük görüntüde eritrosit çapı tahmini anlamsızlaşır
MIN_IMAGE_SIDE = 64

# ---------------------------------------------------------------------------
# Türkçe öznitelik adları
# ---------------------------------------------------------------------------

SHAPE_NAMES = {
    "alan": "Alan / d²", "cevre": "Çevre / d", "dairesellik": "Dairesellik", "eksantrisite": "Eksantrisite",
    "solidity": "Doluluk (solidity)", "extent": "Kutu doluluğu (extent)", "buyuk_eksen": "Büyük eksen / d",
    "kucuk_eksen": "Küçük eksen / d", "eksen_orani": "Eksen oranı", "lob_sayisi": "Lob sayısı",
    "girinti_derinligi": "Girinti derinliği",
}
REGION_NAMES = {"cekirdek": "Çekirdek", "halka": "Çekirdek çevresi"}
CHANNEL_NAMES = {"L": "L* (parlaklık)", "a": "a* (kırmızı-yeşil)", "b": "b* (sarı-mavi)", "H": "ton (H)",
                 "S": "doygunluk (S)", "V": "değer (V)"}
GLCM_NAMES = {"contrast": "kontrast", "dissimilarity": "farklılık", "homogeneity": "homojenlik",
              "energy": "enerji", "correlation": "korelasyon", "ASM": "açısal ikinci moment (ASM)"}
GROUP_NAMES = {"sekil": "Şekil", "renk": "Renk", "doku": "Doku"}


def feature_name_tr(column: str) -> str:
    """
    Öznitelik sütun adını arayüzde gösterilecek Türkçe ada çevirir.

    Parametreler:
        column: features.py sütun adı (ör. "renk_cekirdek_V_ort", "doku_contrast_d2").

    Döndürür:
        Türkçe ad (ör. "Çekirdek değer (V) ortalaması", "GLCM kontrast (d=2)"); tanınmazsa sütun adı.
    """
    group, _, rest = column.partition("_")
    if group == "sekil":
        return SHAPE_NAMES.get(rest, rest)
    if group == "renk":
        parts = rest.split("_")
        if parts[0] == "fark":
            return "Çekirdek − çevre renk farkı (ΔE)" if parts[1] == "deltaE" else \
                f"Çekirdek − çevre farkı: {CHANNEL_NAMES.get(parts[1], parts[1])}"
        if len(parts) == 3:
            stat = "ortalaması" if parts[2] == "ort" else "std sapması"
            return f"{REGION_NAMES.get(parts[0], parts[0])} {CHANNEL_NAMES.get(parts[1], parts[1])} {stat}"
    if group == "doku":
        if rest == "entropi":
            return "Entropi"
        prop, _, dist = rest.rpartition("_")
        return f"GLCM {GLCM_NAMES.get(prop, prop)} ({dist.replace('d', 'd=')})"
    return column


# ---------------------------------------------------------------------------
# Ortak bileşenler
# ---------------------------------------------------------------------------

def disclaimer() -> None:
    """Klinik kullanım uyarısını sabit metinle gösterir (her sayfanın üstünde)."""
    st.warning(DISCLAIMER, icon=":material/warning:")


def badge_html(text: str, color: str) -> str:
    """
    Renkli, yuvarlak köşeli bir rozetin HTML'ini üretir.

    Parametreler:
        text: Rozet metni.
        color: Arka plan rengi (hex).

    Döndürür:
        HTML metni (st.markdown(..., unsafe_allow_html=True) ile gösterilir).
    """
    return (f"<span style='background:{color};color:#fff;padding:0.25rem 0.8rem;border-radius:999px;"
            f"font-weight:600;font-size:1.05rem;display:inline-block'>{text}</span>")


def risk_badge(level: str) -> str:
    """
    Risk seviyesi rozetinin HTML'ini üretir (Düşük yeşil, Orta turuncu, Yüksek kırmızı).

    Parametreler:
        level: Risk seviyesi metni.

    Döndürür:
        HTML metni.
    """
    return badge_html(level, RISK_COLORS.get(level, "#52514e"))


def legend_html() -> str:
    """
    Sınıf renkleri lejantının HTML'ini üretir (PDF rapordaki renklerle aynı).

    Döndürür:
        HTML metni.
    """
    items = "".join(
        f"<span style='display:inline-flex;align-items:center;margin-right:1.2rem'>"
        f"<span style='width:0.9rem;height:0.9rem;background:{CLASS_COLORS[c]};border-radius:3px;"
        f"display:inline-block;margin-right:0.4rem'></span>{CLASS_LABELS[c]}</span>"
        for c in CLASS_NAMES)
    return f"<div style='font-size:0.9rem;margin:0.3rem 0 0.8rem'>{items}</div>"


def show_figure(path: Path, caption: str | None = None) -> None:
    """
    outputs/ altındaki bir figürü gösterir; dosya yoksa bilgi mesajı verir (uygulama çökmez).

    Parametreler:
        path: Figür yolu.
        caption: Figür altı açıklama.
    """
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figür bulunamadı: `{path.relative_to(OUTPUTS_DIR.parent).as_posix()}`. İlgili analiz komutunu "
                "çalıştırın (README → Kurulum ve kullanım).")


# ---------------------------------------------------------------------------
# Önbellekli dosya okuma
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    """
    Bir CSV'yi okur ve önbelleğe alır.

    Parametreler:
        path: Dosya yolu.

    Döndürür:
        DataFrame; dosya yoksa None.
    """
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def output_csv(stage: str, name: str) -> pd.DataFrame | None:
    """
    outputs/<aşama>/<ad> CSV'sini önbellekli okur.

    Parametreler:
        stage: Aşama adı (config.STAGE_OUTPUT_DIRS anahtarı).
        name: Dosya adı.

    Döndürür:
        DataFrame veya None.
    """
    return load_csv(str(STAGE_OUTPUT_DIRS[stage] / name))


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_model_cached():
    """
    Kalibre sınıflandırıcıyı bir kez yükler (tüm oturumlar paylaşır).

    Döndürür:
        (model, meta veri sözlüğü) — src.pipeline.get_model ile aynı nesne.
    """
    from src.pipeline import get_model
    return get_model()


@st.cache_data(show_spinner=False)
def top_features(per_group: int = 5) -> pd.DataFrame:
    """
    Her öznitelik grubundan (şekil, renk, doku) ANOVA F sırasına göre en ayırt edici öznitelikleri seçer.

    Kaynak outputs/features/oznitelik_onemi.csv (Adım 4). Sınıf medyanları hücre değeriyle karşılaştırma için
    birlikte döndürülür.

    Parametreler:
        per_group: Grup başına öznitelik sayısı (5 → toplam 15).

    Döndürür:
        oznitelik, grup, ad, medyan_ALL, medyan_AML, medyan_Healthy sütunlu tablo (grup içinde önem sırasıyla).
    """
    table = output_csv("features", "oznitelik_onemi.csv")
    if table is None:
        return pd.DataFrame(columns=["oznitelik", "grup", "ad", "medyan_ALL", "medyan_AML", "medyan_Healthy"])
    table = table.assign(grup=table["oznitelik"].str.split("_").str[0]).sort_values("f_sirasi")
    out = table.groupby("grup", sort=False).head(per_group).copy()
    out["ad"] = out["oznitelik"].map(feature_name_tr)
    order = {g: i for i, g in enumerate(GROUP_NAMES)}
    out = out.sort_values(["grup", "f_sirasi"], key=lambda s: s.map(order) if s.name == "grup" else s)
    return out[["oznitelik", "grup", "ad", "medyan_ALL", "medyan_AML", "medyan_Healthy"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def example_images() -> pd.DataFrame:
    """
    Hazır örnek görüntüleri (reports/ornekler/ PDF'lerinin görüntüleri) listeler.

    Kaynak outputs/risk/ornek_secimi<sonek>.csv. Yalnızca kaynağı "bilinmiyor" olan görüntüler alınır
    (ALL-IDB türevleri lisans gereği gösterilmez).

    Döndürür:
        dosya_yolu, etiket, ornek_turu, metin sütunlu tablo; dosya yoksa boş tablo.
    """
    table = output_csv("risk", f"ornek_secimi{RISK_RESULT_SUFFIX}.csv")
    if table is None:
        return pd.DataFrame(columns=["dosya_yolu", "etiket", "ornek_turu", "metin"])
    table = table[table["kaynak"] == "bilinmiyor"].reset_index(drop=True)
    stems = table["dosya_yolu"].map(lambda p: " ".join(Path(p).stem.split()))
    table["metin"] = [f"{EXAMPLE_TYPE_NAMES.get(t, t)} — {s}" + (f" (gerçek sınıf: {e})" if t == "dusuk_guven" else "")
                      for t, s, e in zip(table["ornek_turu"], stems, table["etiket"])]
    return table[["dosya_yolu", "etiket", "ornek_turu", "metin"]]


# ---------------------------------------------------------------------------
# Görüntü hazırlama (yalnızca gösterim)
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray:
    """
    Yüklenen dosyanın baytlarını RGB görüntüye çevirir.

    Parametreler:
        data: Dosya içeriği (jpg, png, tif).

    Döndürür:
        RGB uint8 görüntü.

    Hata:
        ValueError: Dosya görüntü olarak çözülemezse veya görüntü çok küçükse (anlaşılır Türkçe mesajla).
    """
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Dosya görüntü olarak okunamadı. Dosya bozuk olabilir ya da biçimi desteklenmiyor "
                         "(desteklenen: JPG, PNG, TIF).")
    if min(bgr.shape[:2]) < MIN_IMAGE_SIDE:
        raise ValueError(f"Görüntü çok küçük ({bgr.shape[1]}x{bgr.shape[0]} px). En az {MIN_IMAGE_SIDE} px "
                         "kenarlı bir kan yayması görüntüsü yükleyin.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_display(image: np.ndarray, side: int = 512) -> np.ndarray:
    """
    Küçük görüntüyü (256 px ara çıktı, hücre kırpımı) gösterim için büyütür.

    Parametreler:
        image: Gri veya RGB görüntü; bool maske de olabilir.
        side: Hedef uzun kenar (px).

    Döndürür:
        uint8 görüntü.
    """
    img = image.astype(np.uint8) * 255 if image.dtype == bool else image
    scale = side / max(img.shape[:2])
    size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if image.dtype == bool else cv2.INTER_CUBIC)


def boundary_overlay(rgb: np.ndarray, labels: np.ndarray, color: tuple[int, int, int],
                     scale: int = 2) -> np.ndarray:
    """
    Etiketli çekirdek maskesinin sınırlarını görüntünün üstüne çizer (gösterim için büyütülmüş).

    Etiketler en yakın komşuyla büyütülür, böylece birbirine değen çekirdeklerin ayrım çizgisi de görünür.

    Parametreler:
        rgb: 256x256 RGB görüntü.
        labels: Aynı boyutta etiketli maske (0 = zemin).
        color: Sınır rengi (RGB).
        scale: Büyütme çarpanı.

    Döndürür:
        Sınırları çizilmiş RGB görüntü.
    """
    big = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big_labels = np.repeat(np.repeat(labels, scale, axis=0), scale, axis=1)
    big[find_boundaries(big_labels, mode="inner")] = color
    return big


def label_mask_image(labels: np.ndarray) -> np.ndarray:
    """
    Etiketli maskeyi siyah zemin üstünde beyaz çekirdekler olarak görselleştirir.

    Parametreler:
        labels: Etiketli maske.

    Döndürür:
        uint8 gri görüntü.
    """
    return (labels > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Önbellekli analiz
# ---------------------------------------------------------------------------

def image_key(data: bytes) -> str:
    """
    Görüntü baytlarından kısa bir özet anahtarı üretir (önbellek anahtarı ve dosya adları için).

    Parametreler:
        data: Dosya içeriği.

    Döndürür:
        SHA-1 özetinin onaltılık metni.
    """
    return hashlib.sha1(data).hexdigest()


@st.cache_data(show_spinner=False, max_entries=16)
def analyze_cached(data: bytes, name: str, _progress=None) -> dict:
    """
    Görüntüyü uçtan uca pipeline'dan geçirir; aynı görüntü (aynı baytlar) için sonuç önbellekten gelir.

    Parametreler:
        data: Görüntü dosyasının baytları (önbellek anahtarı bu içeriğin özetidir).
        name: Raporda görünecek dosya adı.
        _progress: İsteğe bağlı ilerleme geri çağrısı (alt çizgili: önbellek anahtarına girmez).

    Döndürür:
        src.pipeline.run_pipeline çıktısı.
    """
    from src.pipeline import run_pipeline
    return run_pipeline(decode_image(data), image_name=name, progress=_progress)


def analyze_with_progress(data: bytes, name: str) -> dict:
    """
    Analizi ayrı bir iş parçacığında çalıştırır ve ana iş parçacığında aşama aşama ilerleme çubuğu gösterir.

    İlerleme çubuğu önbellekli fonksiyonun içinde güncellenmez: Streamlit, önbellekli fonksiyon içindeki öğe
    çağrılarını kaydedip önbellekten okurken yeniden oynatır, dışarıda oluşturulmuş bir öğe için bu hata verir.
    Bu yüzden geri çağrı yalnızca paylaşılan bir sözlüğü günceller, çubuk ana iş parçacığında çizilir.

    Parametreler:
        data: Görüntü baytları.
        name: Görüntü adı.

    Döndürür:
        run_pipeline çıktısı.

    Hata:
        Pipeline'ın fırlattığı hata aynen yeniden fırlatılır.
    """
    state = {"asama": "Başlıyor", "oran": 0.0, "sonuc": None, "hata": None}

    def report(stage: str, done: float) -> None:
        state["asama"], state["oran"] = stage, done

    def work() -> None:
        try:
            state["sonuc"] = analyze_cached(data, name, _progress=report)
        except Exception as exc:  # noqa: BLE001 — hata ana iş parçacığında kullanıcıya gösterilir
            state["hata"] = exc

    thread = threading.Thread(target=work, daemon=True)
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()
    bar = st.progress(0.0, text="Analiz başlıyor...")
    while thread.is_alive():
        bar.progress(min(state["oran"], 1.0), text=f"{state['asama']}...")
        time.sleep(0.1)
    thread.join()
    bar.empty()
    if state["hata"] is not None:
        raise state["hata"]
    return state["sonuc"]


@st.cache_data(show_spinner="Üç segmentasyon yöntemi çalıştırılıyor...", max_entries=16)
def compare_methods(key: str, _steps: dict, diameter: float) -> dict:
    """
    Aynı ön işlenmiş görüntüde Otsu, K-means ve Watershed segmentasyonlarını çalıştırır.

    src.segmentation.METHODS kullanılır (pipeline ile aynı fonksiyonlar ve aynı eritrosit çapı).

    Parametreler:
        key: Görüntü özeti (önbellek anahtarı).
        _steps: preprocess ara çıktıları (anahtara girmez).
        diameter: Eritrosit çapı (px).

    Döndürür:
        {yöntem: etiketli maske} sözlüğü (sıra: otsu, kmeans, watershed).
    """
    from src.segmentation import METHODS
    return {m: METHODS[m](_steps, diameter) for m in ("otsu", "kmeans", "watershed")}


@st.cache_data(show_spinner="PDF rapor hazırlanıyor...", max_entries=16)
def pdf_bytes(key: str, _result: dict) -> bytes:
    """
    src.risk_report.generate_pdf_report ile PDF raporu üretir ve baytlarını döndürür.

    PDF geçici bir klasöre yazılır, okunur ve silinir (reports/ altına dosya bırakılmaz).

    Parametreler:
        key: Görüntü özeti (önbellek anahtarı).
        _result: run_pipeline çıktısı (anahtara girmez).

    Döndürür:
        PDF dosyasının baytları.
    """
    from src.risk_report import generate_pdf_report
    with tempfile.TemporaryDirectory() as tmp:
        path = generate_pdf_report(_result, Path(tmp) / "rapor.pdf")
        return path.read_bytes()


def example_bytes(path: str) -> bytes | None:
    """
    Hazır örnek görüntünün baytlarını okur.

    Parametreler:
        path: CSV'deki göreli yol (data/raw/...).

    Döndürür:
        Baytlar; dosya yoksa None.
    """
    p = resolve_path(path)
    return p.read_bytes() if p.exists() else None


def preprocessing_panels(steps: dict) -> list[tuple[str, np.ndarray, str]]:
    """
    Ön işleme ara çıktılarını başlık ve bir cümlelik açıklamayla gösterim sırasına koyar.

    Parametreler:
        steps: preprocess ara çıktıları.

    Döndürür:
        (başlık, görüntü, açıklama) listesi.
    """
    return [
        ("Orijinal", steps["orijinal"],
         f"Yüklenen görüntü ({steps['orijinal'].shape[1]}x{steps['orijinal'].shape[0]} px)."),
        ("Boyutlandırılmış", steps["boyutlandirilmis"],
         f"En-boy oranı korunarak {IMAGE_SIZE}x{IMAGE_SIZE}'ya getirildi; boş kalan kenarlar dolduruldu."),
        ("Dolgu maskesi", steps["gecerli_maske"],
         "Beyaz: geçerli pikseller. Dolgu ve döndürmeden kalan koyu köşeler sonraki hesaplardan dışlanır."),
        ("Renk normalizasyonu", steps["renk_normalize"],
         "Zemin (eritrosit + plazma) rengi ortak bir hedefe hizalandı; çekirdek−zemin farkı korunur."),
        ("Lab b* kanalı (ters)", steps["kanal"],
         "Mor çekirdek bu kanalda parlak görünür; pembe eritrosit ve plazma nötr kalır."),
        ("Medyan filtre (5x5)", steps["filtreli"],
         "Tuz-biber gürültüsü azaltılır; çekirdek sınırları korunur."),
        ("CLAHE", steps["kontrast"],
         "Yerel kontrast artırıldı (clip 1.0, 64 px karo); segmentasyonun girdisi bu görüntüdür."),
    ]
