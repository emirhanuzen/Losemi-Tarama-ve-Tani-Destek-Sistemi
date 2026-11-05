"""
Tüm aşamaları uçtan uca birleştiren ana pipeline.

Tek bir görüntü için: ön işleme → segmentasyon (K-means; hücre yoksa Otsu geri dönüşü) →
hücre çıkarma → öznitelik çıkarma → kayıtlı modelle hücre tahmini → risk skoru.
run_pipeline tüm ara çıktıları sözlük olarak döndürür (Streamlit arayüzü de bunu kullanır).

Çalıştırma (proje kökünden):
    python -m src.pipeline --goruntu <yol>          # sonucu konsola yazar
    python -m src.pipeline --goruntu <yol> --pdf    # ayrıca reports/ altına PDF rapor
"""

import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    REPORT_IMAGE_MAX_SIDE,
    SEGMENTATION_FALLBACK_METHOD,
    SEGMENTATION_METHOD,
)
from src.dataset import CLASS_COLORS
from src.features import crop_features
from src.preprocessing import preprocess
from src.risk_report import compute_risk, dominant_label, generate_pdf_report
from src.segmentation import METHODS, extract_cells, image_rbc_diameter
from src.utils import load_image

PROBA_COLUMNS = [f"p_{c}" for c in CLASS_NAMES]


@lru_cache(maxsize=1)
def get_model():
    """
    Kayıtlı sınıflandırıcıyı bir kez yükler ve önbellekte tutar (arayüzde her görüntüde yeniden yüklenmesin).

    Döndürür:
        (model, meta veri sözlüğü).
    """
    from src.classifier import load_model  # sklearn yalnızca tahmin gerektiğinde yüklenir
    return load_model()


# ---------------------------------------------------------------------------
# Segmentasyon (geri dönüşlü)
# ---------------------------------------------------------------------------

def segment_with_fallback(steps: dict, image_type: str | None = None) -> dict:
    """
    Seçilen yöntemle (K-means) segmentasyon yapar; hiç çekirdek bulunamazsa Otsu'ya geri döner.

    Parametreler:
        steps: preprocess() ara çıktıları.
        image_type: Biliniyorsa "yayma"/"yakin_cekim"; None ise görüntüden tahmin edilir.

    Döndürür:
        {"etiketler" (etiketli maske), "maske" (bool), "yontem" (kullanılan yöntem), "geri_donus" (bool),
         "birincil_bolge_sayisi", "eritrosit_capi", "eritrosit_tahmini_guvenilir", "tip"} sözlüğü.
    """
    diameter, estimate = image_rbc_diameter(steps, image_type)
    labels = METHODS[SEGMENTATION_METHOD](steps, diameter)
    n_primary = int(len(np.unique(labels)) - 1)
    method, fallback = SEGMENTATION_METHOD, False
    if n_primary == 0:
        labels = METHODS[SEGMENTATION_FALLBACK_METHOD](steps, diameter)
        method, fallback = SEGMENTATION_FALLBACK_METHOD, True
    return {"etiketler": labels, "maske": labels > 0, "yontem": method, "geri_donus": fallback,
            "birincil_bolge_sayisi": n_primary, "eritrosit_capi": diameter,
            "eritrosit_tahmini_guvenilir": bool(estimate["guvenilir"]), "tip": estimate["tip"]}


def cell_contours(labels: np.ndarray, cells: list[dict]) -> list[np.ndarray]:
    """
    Her hücrenin çekirdek konturunu 256x256 ön işleme koordinatlarında çıkarır.

    Parametreler:
        labels: Etiketli çekirdek maskesi.
        cells: extract_cells çıktısı (kutu ve cekirdek_maskesi ile).

    Döndürür:
        Her hücre için (N, 2) float kontur noktaları (x, y) listesi (en büyük dış kontur).
    """
    out = []
    for cell in cells:
        x0, y0, _, _ = cell["kutu"]
        found, _ = cv2.findContours(cell["cekirdek_maskesi"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest = max(found, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        out.append(largest + [x0, y0])
    return out


# ---------------------------------------------------------------------------
# Görselleştirme (rapor ve arayüz için)
# ---------------------------------------------------------------------------

def display_geometry(original_shape: tuple) -> tuple[float, float, float, tuple[int, int]]:
    """
    256x256 ön işleme koordinatlarını gösterim boyutundaki orijinal görüntüye çeviren dönüşümü hesaplar.

    resize_with_padding ile aynı ölçek ve ortalama payını kullanır; gösterim boyutu uzun kenarı en fazla
    REPORT_IMAGE_MAX_SIDE olan orijinal görüntüdür (büyükler küçültülür, 256 px'lik görüntüler
    büyütülür; konturlar her iki durumda da aynı çözünürlükte net görünür).

    Parametreler:
        original_shape: Orijinal görüntünün (H, W[, 3]) şekli.

    Döndürür:
        (çarpan, sol pay, üst pay, (gösterim genişliği, yüksekliği)): x_göst = (x − sol + 0.5) × çarpan − 0.5.
    """
    h, w = original_shape[:2]
    s = IMAGE_SIZE / max(h, w)
    new_w, new_h = max(1, round(w * s)), max(1, round(h * s))
    left, top = (IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2
    d = REPORT_IMAGE_MAX_SIDE / max(h, w)
    size = (round(w * d), round(h * d))
    return d / s, left, top, size


def draw_cells(original: np.ndarray, contours: list[np.ndarray], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Orijinal görüntüyü gösterim boyutuna getirir ve hücre konturlarını sınıf renkleriyle, numaralı çizer.

    Parametreler:
        original: Orijinal RGB görüntü.
        contours: cell_contours çıktısı (256 px koordinatları).
        classes: Her hücrenin tahmin edilen sınıfı.

    Döndürür:
        (gösterim boyutunda orijinal görüntü, konturlu görüntü).
    """
    factor, left, top, size = display_geometry(original.shape)
    interp = cv2.INTER_AREA if size[0] < original.shape[1] else cv2.INTER_CUBIC
    base = cv2.resize(original, size, interpolation=interp)
    overlay = base.copy()
    long_side = max(size)
    thickness = max(2, round(long_side / 400))  # 800 px'de 2 px: ince çekirdeklerde de görünür, kapatmaz
    # Numara etiketi: PDF'te görsel ~8.6 cm genişliğe küçülür; 800 px'de 0.8 ölçekli, 2 px kalın yazı ~2.5 mm
    # yüksekliğe denk gelir ve basılı raporda okunur (önceki 0.45 / 1 px ~1.3 mm idi). Beyaz kutu içinde koyu
    # yazı her zemin renginde okunur; kutunun çerçevesi hücrenin sınıf rengindedir (numara ↔ kontur eşleşmesi).
    font_scale = 0.8 * long_side / 800
    font_thick = max(1, round(2 * long_side / 800))
    pad = max(2, round(4 * long_side / 800))
    text_color = (31, 30, 28)   # koyu gri-siyah (rapor metin rengi)
    for pts, cls in zip(contours, classes):
        color = tuple(int(CLASS_COLORS[cls][i:i + 2], 16) for i in (1, 3, 5))
        disp = ((pts - [left, top] + 0.5) * factor - 0.5).round().astype(np.int32)
        cv2.polylines(overlay, [disp.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)
    # Etiketler konturlardan sonra çizilir: komşu hücrenin konturu numaranın üstüne binmesin
    placed: list[tuple[int, int, int, int]] = []
    for k, (pts, cls) in enumerate(zip(contours, classes), start=1):
        color = tuple(int(CLASS_COLORS[cls][i:i + 2], 16) for i in (1, 3, 5))
        disp = ((pts - [left, top] + 0.5) * factor - 0.5).round().astype(np.int32)
        label = str(k)
        (tw, th), base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        tx, ty = label_position(disp, (tw, th, base_line), pad, thickness, size, placed)
        box = (tx - pad, ty - th - pad, tx + tw + pad, ty + base_line + pad - 1)
        placed.append(box)
        cv2.rectangle(overlay, box[:2], box[2:], (255, 255, 255), -1)
        cv2.rectangle(overlay, box[:2], box[2:], color, max(1, thickness - 1), cv2.LINE_AA)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)
    return base, overlay


def label_position(disp: np.ndarray, text: tuple[int, int, int], pad: int, thickness: int, size: tuple[int, int],
                   placed: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """
    Bir hücre numarası kutusu için, daha önce yerleştirilmiş kutularla çakışmayan bir konum seçer.

    Adaylar sırayla: konturun sol üstünün üstü, sol altının altı, sağ üstünün sağı. Hepsi çakışıyorsa ilk aday
    kutu genişliği kadar sağa (gerekirse aşağı) kaydırılır. Her aday görüntü sınırına kırpılır. Yoğun AML
    yaymalarında da numaralar üst üste binip okunmaz hâle gelmesin diye.

    Parametreler:
        disp: Konturun gösterim koordinatlarındaki noktaları (N, 2).
        text: (yazı genişliği, yüksekliği, taban çizgisi) — cv2.getTextSize.
        pad: Kutu iç boşluğu (px).
        thickness: Kontur kalınlığı (px).
        size: Gösterim boyutu (genişlik, yükseklik).
        placed: Önceki kutular (x0, y0, x1, y1).

    Döndürür:
        Yazının sol alt başlangıç noktası (tx, ty).
    """
    tw, th, base_line = text
    x0, y0 = disp[:, 0].min(), disp[:, 1].min()
    x1, y1 = disp[:, 0].max(), disp[:, 1].max()

    def clip(tx: int, ty: int) -> tuple[int, int]:
        return (int(np.clip(tx, pad, size[0] - tw - pad - 1)),
                int(np.clip(ty, th + 2 * pad, size[1] - base_line - pad - 1)))

    def free(tx: int, ty: int) -> bool:
        a = (tx - pad, ty - th - pad, tx + tw + pad, ty + base_line + pad)
        return all(a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1] for b in placed)

    candidates = [clip(x0, y0 - pad - thickness), clip(x0, y1 + th + 2 * pad + thickness),
                  clip(x1 + pad + thickness, y0 + th)]
    for c in candidates:
        if free(*c):
            return c
    tx, ty = candidates[0]
    step_x, step_y = tw + 2 * pad + 2, th + base_line + 2 * pad + 2
    for i in range(1, 12):
        c = clip(tx + i * step_x, ty) if tx + i * step_x + tw + pad < size[0] else clip(tx, ty + i * step_y)
        if free(*c):
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Uçtan uca
# ---------------------------------------------------------------------------

def predict_cells(features: pd.DataFrame) -> np.ndarray:
    """
    Kayıtlı modelle hücre olasılıklarını hesaplar.

    Parametreler:
        features: Öznitelik tablosu (modelin öznitelik sütunlarını içeren).

    Döndürür:
        (hücre sayısı, 3) olasılık dizisi; sütun sırası CLASS_NAMES.
    """
    from src.classifier import reorder_proba
    model, meta = get_model()
    if features.empty:
        return np.zeros((0, len(CLASS_NAMES)))
    X = features[meta["oznitelikler"]].to_numpy(np.float64)
    return reorder_proba(model, model.predict_proba(X))


def run_pipeline(image: str | Path | np.ndarray, image_name: str | None = None,
                 image_type: str | None = None) -> dict:
    """
    Tek bir kan yayması görüntüsünde tüm aşamaları çalıştırır ve ara çıktıları döndürür.

    Parametreler:
        image: Görüntü yolu veya RGB numpy dizisi.
        image_name: Raporda görünecek ad (yol verildiyse dosya adı kullanılır).
        image_type: Biliniyorsa "yayma"/"yakin_cekim"; None ise görüntüden tahmin edilir.

    Döndürür:
        {
          "goruntu_adi": str,
          "on_isleme": preprocess ara çıktıları (orijinal, boyutlandirilmis, gecerli_maske, renk_normalize,
                       kanal, filtreli, kontrast),
          "segmentasyon": segment_with_fallback çıktısı (etiketler, maske, yontem, geri_donus, ...),
          "konturlar": hücre konturları (256 px koordinatları),
          "kirpimlar": hücre kırpımları (RGB) listesi,
          "oznitelikler": hücre öznitelik tablosu (DataFrame),
          "hucreler": hücre tablosu (hucre_no, sinif, p_ALL, p_AML, p_Healthy, kenara_degiyor, alan, kutu_*),
          "risk": compute_risk çıktısı,
          "gorsel_orijinal", "gorsel_kontur": rapor/arayüz için gösterim görüntüleri,
        }
    """
    if isinstance(image, (str, Path)):
        image_name = image_name or Path(image).name
        image = load_image(image)
    image_name = image_name or "goruntu"
    _, steps = preprocess(image)
    seg = segment_with_fallback(steps, image_type)
    cells = extract_cells(steps["renk_normalize"], seg["etiketler"], steps["gecerli_maske"], seg["eritrosit_capi"])
    # Eğitim tablosunda çap 2 ondalığa yuvarlanarak saklandı (cells_index.csv); aynı değerle hesapla
    diameter = round(seg["eritrosit_capi"], 2)
    features = pd.DataFrame([crop_features(c["kirpim"], c["maske"], diameter) for c in cells])
    proba = predict_cells(features)
    classes = list(np.asarray(CLASS_NAMES)[proba.argmax(axis=1)]) if len(proba) else []
    table = pd.DataFrame({
        "hucre_no": np.arange(1, len(cells) + 1),
        "sinif": classes,
        **{col: proba[:, i] for i, col in enumerate(PROBA_COLUMNS)},
        "kenara_degiyor": [c["kenara_degiyor"] for c in cells],
        "alan": [c["alan"] for c in cells],
        **{f"kutu_{k}": [c["kutu"][i] for c in cells] for i, k in enumerate("xywh")},
    })
    contours = cell_contours(seg["etiketler"], cells)
    risk = compute_risk(table, seg["yontem"], seg["geri_donus"])
    original, overlay = draw_cells(image, contours, classes)
    return {
        "goruntu_adi": image_name, "on_isleme": steps, "segmentasyon": seg, "konturlar": contours,
        "kirpimlar": [c["kirpim"] for c in cells], "oznitelikler": features, "hucreler": table, "risk": risk,
        "gorsel_orijinal": original, "gorsel_kontur": overlay,
    }


def print_result(result: dict) -> None:
    """
    Pipeline sonucunu konsola özetler.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    r, seg = result["risk"], result["segmentasyon"]
    c = r["hucre_sayilari"]
    print(f"Görüntü: {result['goruntu_adi']}")
    print(f"Segmentasyon: {seg['yontem']}" + (" (geri dönüş: K-means hücre bulamadı)" if seg["geri_donus"] else "")
          + f", eritrosit çapı {seg['eritrosit_capi']:.1f} px, tip {seg['tip']}")
    print(f"Hücre: {r['toplam_hucre']} (ALL {c['ALL']}, AML {c['AML']}, Healthy {c['Healthy']})")
    print(f"Anormal hücre oranı: {r['anormal_oran']:.3f} (yumuşak {r['anormal_oran_yumusak']:.3f})")
    print(f"Risk seviyesi: {r['risk_seviyesi']}, baskın tip: {dominant_label(r)}")
    print(f"Güven: {r['guven']}" + "".join(f"\n  - {x}" for x in r["guven_nedenleri"]))


def main() -> None:
    """Komut satırı: python -m src.pipeline --goruntu <yol> [--pdf] [--cikti <pdf yolu>]."""
    parser = argparse.ArgumentParser(description="Uçtan uca lösemi tarama pipeline'ı")
    parser.add_argument("--goruntu", required=True, help="Kan yayması görüntüsünün yolu")
    parser.add_argument("--pdf", action="store_true", help="reports/ altına PDF rapor üret")
    parser.add_argument("--cikti", default=None, help="PDF yolu (varsayılan reports/<ad>_rapor.pdf)")
    args = parser.parse_args()
    result = run_pipeline(args.goruntu)
    print_result(result)
    if args.pdf:
        print(f"PDF rapor: {generate_pdf_report(result, args.cikti)}")


if __name__ == "__main__":
    main()
