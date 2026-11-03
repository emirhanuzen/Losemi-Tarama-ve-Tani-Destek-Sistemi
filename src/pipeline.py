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
from src.risk_report import compute_risk, generate_pdf_report
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
    thickness = max(2, round(size[0] / 400))  # 800 px'de 2 px: ince çekirdeklerde de görünür, kapatmaz
    for k, (pts, cls) in enumerate(zip(contours, classes), start=1):
        color = tuple(int(CLASS_COLORS[cls][i:i + 2], 16) for i in (1, 3, 5))
        disp = ((pts - [left, top] + 0.5) * factor - 0.5).round().astype(np.int32)
        cv2.polylines(overlay, [disp.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)
        x, y = disp[:, 0].min(), disp[:, 1].min()
        label = str(k)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx, ty = int(np.clip(x, 0, size[0] - tw - 2)), int(np.clip(y - 3, th + 2, size[1] - 2))
        cv2.rectangle(overlay, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (255, 255, 255), -1)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return base, overlay


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
    print(f"Risk seviyesi: {r['risk_seviyesi']}, baskın tip: {r['baskin_tip']}"
          + (f" ({r['baskin_oran']:.0%})" if r["baskin_tip"] != "-" else ""))
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
