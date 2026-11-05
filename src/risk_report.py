"""
Aşama 5 - Risk Skoru ve Otomatik PDF Rapor.

- Risk skoru: görüntüdeki anormal (ALL + AML) hücre oranı. Hücre olasılıklarının ortalamasıyla
  hesaplanan yumuşak bir versiyonu da verilir.
- Baskın lösemi tipi: anormal hücreler içinde ALL mi AML mi çoğunlukta (oranıyla).
- Risk seviyesi (Düşük / Orta / Yüksek): eşikler geliştirme setinden seçildi (config.RISK_THRESHOLDS,
  seçim: src/risk_analysis.py).
- Güven bayrağı: az hücre, Otsu geri dönüşü veya çoğunlukla kesik hücre → "düşük güven".
- PDF rapor ReportLab ile, Türkçe karakter destekli DejaVuSans fontuyla üretilir.

Bu modül model veya segmentasyon çalıştırmaz; uçtan uca akış src/pipeline.py içindedir.
"""

import io
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import (
    ABNORMAL_CLASSES,
    CLASS_NAMES,
    LOW_CONFIDENCE_EDGE_FRACTION,
    LOW_CONFIDENCE_MIN_CELLS,
    ACTIVE_EVAL_SUFFIX,
    ACTIVE_MODEL_FILE,
    MODELS_DIR,
    REPORT_DISCLAIMER,
    REPORT_MAX_CELL_ROWS,
    REPORTS_DIR,
    RISK_INCLUDE_EDGE_CELLS,
    RISK_LEVELS,
    RISK_NO_CELL_LEVEL,
    RISK_TARGET_SENSITIVITY,
    RISK_TARGET_SPECIFICITY,
    RISK_THRESHOLDS,
    RISK_RESULT_SUFFIX,
    SEGMENTATION_METHOD,
    STAGE_OUTPUT_DIRS,
)
from src.dataset import CLASS_COLORS

PROBA_COLUMNS = [f"p_{c}" for c in CLASS_NAMES]
# Baskın tip gösterilmediğinde yazılan değer: anormal hücre yok ya da risk seviyesi Düşük / Değerlendirilemedi.
NO_DOMINANT = "—"
# Baskın tip yalnızca bu seviyelerde gösterilir. Düşük riskte anormal hücreler çoğunlukla tek tük yanlış
# sınıflamalardır (ör. Healthy görüntüsünde 1 hücre); onlara bir lösemi tipi atamak yanıltıcı olur.
DOMINANT_LEVELS = ("Orta", "Yüksek")
METHOD_NAMES = {"kmeans": "K-means", "otsu": "Otsu", "watershed": "Watershed"}
MODEL_TITLES = {"hgb": "Gradyan artırma (HistGradientBoosting)", "svm": "SVM (RBF)", "rf": "Random Forest",
                "knn": "k-NN", "lojistik": "Lojistik regresyon"}


# ---------------------------------------------------------------------------
# Risk skoru
# ---------------------------------------------------------------------------

def risk_level(score: float, thresholds: tuple = RISK_THRESHOLDS) -> str:
    """
    Risk skorunu Düşük / Orta / Yüksek seviyesine çevirir.

    Parametreler:
        score: Anormal hücre oranı (0-1); NaN ise hücre bulunamamıştır.
        thresholds: (Düşük/Orta, Orta/Yüksek) eşikleri; skor eşiğe eşitse üst seviye.

    Döndürür:
        RISK_LEVELS elemanı veya RISK_NO_CELL_LEVEL; eşikler henüz belirlenmemişse (None) None.
    """
    if score is None or np.isnan(score):
        return RISK_NO_CELL_LEVEL
    low, high = thresholds
    if low is None or high is None:  # Eşik seçimi (src.risk_analysis) öncesi: yalnızca skor hesaplanır
        return None
    return RISK_LEVELS[int(score >= low) + int(score >= high)]


def dominant_type(cells: pd.DataFrame) -> tuple[str, float]:
    """
    Anormal hücreler içinde baskın lösemi tipini ve payını bulur.

    Sayıca çoğunluk esas alınır; ALL ve AML hücre sayısı eşitse anormal hücrelerdeki P(ALL) ve
    P(AML) toplamları karar verir.

    Parametreler:
        cells: "sinif" ve p_ALL, p_AML sütunlu hücre tablosu.

    Döndürür:
        (tip, pay): tip "ALL"/"AML" (anormal hücre yoksa NO_DOMINANT), pay = o tipteki hücre / anormal hücre.
    """
    abnormal = cells[cells["sinif"].isin(ABNORMAL_CLASSES)]
    if abnormal.empty:
        return NO_DOMINANT, float("nan")
    counts = abnormal["sinif"].value_counts().reindex(ABNORMAL_CLASSES, fill_value=0)
    if counts["ALL"] != counts["AML"]:
        winner = counts.idxmax()
    else:
        winner = "ALL" if abnormal["p_ALL"].sum() >= abnormal["p_AML"].sum() else "AML"
    return winner, float(counts[winner] / len(abnormal))


def confidence_flags(n_cells: int, fallback: bool, edge_fraction: float) -> list[str]:
    """
    Düşük güven nedenlerini listeler (boş liste = güven normal).

    Parametreler:
        n_cells: Hücre sayısı.
        fallback: Otsu geri dönüşü kullanıldı mı.
        edge_fraction: Kenara değen hücrelerin oranı.

    Döndürür:
        Türkçe neden cümleleri listesi.
    """
    reasons = []
    if n_cells == 0:
        return ["Görüntüde hiç hücre çekirdeği bulunamadı."]
    if n_cells < LOW_CONFIDENCE_MIN_CELLS:
        reasons.append(f"Yalnızca {n_cells} hücre bulundu (en az {LOW_CONFIDENCE_MIN_CELLS} gerekir); "
                       "tek bir hücre hatası skoru büyük ölçüde değiştirir.")
    if fallback:
        reasons.append(f"{METHOD_NAMES[SEGMENTATION_METHOD]} hücre bulamadı; Otsu geri dönüşü kullanıldı "
                       "(model bu yöntemin kırpımlarıyla eğitilmedi).")
    if edge_fraction > LOW_CONFIDENCE_EDGE_FRACTION:
        reasons.append(f"Hücrelerin %{100 * edge_fraction:.0f}'i görüntü kenarına değiyor (kesik hücreler "
                       "daha az güvenilir sınıflanır).")
    return reasons


def compute_risk(cells: pd.DataFrame, method: str = SEGMENTATION_METHOD, fallback: bool = False,
                 thresholds: tuple = RISK_THRESHOLDS) -> dict:
    """
    Bir görüntünün hücre tahminlerinden risk skorunu, seviyesini, baskın tipi ve güven durumunu hesaplar.

    Parametreler:
        cells: Görüntünün hücre tablosu: "sinif", p_ALL, p_AML, p_Healthy, "kenara_degiyor".
        method: Kullanılan segmentasyon yöntemi.
        fallback: Geri dönüş (Otsu) kullanıldı mı.
        thresholds: Risk seviyesi eşikleri.

    Döndürür:
        {toplam_hucre, hucre_sayilari {sınıf: sayı}, anormal_hucre, anormal_oran (risk skoru),
         anormal_oran_yumusak, baskin_tip, baskin_oran, baskin_tip_yumusak, baskin_oran_yumusak,
         baskin_tip_ham, baskin_oran_ham, risk_seviyesi, guven ("Normal"/"Düşük"), guven_nedenleri, kenar_orani,
         segmentasyon_yontemi, geri_donus, esikler} sözlüğü.
        baskin_tip / baskin_oran (ve yumuşak hâlleri) yalnızca risk seviyesi Orta veya Yüksekse doludur; aksi hâlde
        NO_DOMINANT / NaN. Seviyeden bağımsız ham değer baskin_tip_ham / baskin_oran_ham'dadır (analiz için).
    """
    used = cells if RISK_INCLUDE_EDGE_CELLS else cells[~cells["kenara_degiyor"].astype(bool)]
    n = len(used)
    counts = {c: int((used["sinif"] == c).sum()) for c in CLASS_NAMES}
    n_abnormal = sum(counts[c] for c in ABNORMAL_CLASSES)
    hard = n_abnormal / n if n else float("nan")
    soft = float(used[[f"p_{c}" for c in ABNORMAL_CLASSES]].sum(axis=1).mean()) if n else float("nan")
    dom, dom_share = dominant_type(used)
    # Yumuşak baskın tip: tüm hücrelerin ortalama P(ALL) ve P(AML)'sinin birbirine oranı
    p_all, p_aml = (float(used["p_ALL"].mean()), float(used["p_AML"].mean())) if n else (0.0, 0.0)
    soft_dom = NO_DOMINANT if p_all + p_aml == 0 else ("ALL" if p_all >= p_aml else "AML")
    soft_share = max(p_all, p_aml) / (p_all + p_aml) if p_all + p_aml > 0 else float("nan")
    edge_fraction = float(cells["kenara_degiyor"].astype(bool).mean()) if len(cells) else 0.0
    reasons = confidence_flags(n, fallback, edge_fraction)
    level = risk_level(hard, thresholds)
    show = level in DOMINANT_LEVELS   # Düşük / Değerlendirilemedi / eşiksiz: baskın tip gösterilmez
    return {
        "toplam_hucre": n, "hucre_sayilari": counts, "anormal_hucre": n_abnormal,
        "anormal_oran": hard, "anormal_oran_yumusak": soft,
        "baskin_tip": dom if show else NO_DOMINANT, "baskin_oran": dom_share if show else float("nan"),
        "baskin_tip_yumusak": soft_dom if show else NO_DOMINANT,
        "baskin_oran_yumusak": soft_share if show else float("nan"),
        "baskin_tip_ham": dom, "baskin_oran_ham": dom_share,
        "risk_seviyesi": level,
        "guven": "Düşük" if reasons else "Normal", "guven_nedenleri": reasons, "kenar_orani": edge_fraction,
        "segmentasyon_yontemi": method, "geri_donus": fallback, "esikler": tuple(thresholds),
    }


# ---------------------------------------------------------------------------
# Model bilgisi (raporda gösterilir)
# ---------------------------------------------------------------------------

def model_summary() -> dict:
    """
    Kayıtlı modelin adını ve kilitli test seti metriklerini okur (PDF'teki model bilgisi için).

    Kaynaklar etkin modele göre seçilir (config.ACTIVE_MODEL_FILE: ayrım sürümü + kalibrasyon):
    models/<ACTIVE_MODEL_FILE>.json, outputs/evaluation/test_metrikleri<ACTIVE_EVAL_SUFFIX>.csv ve
    outputs/risk/risk_esikleri<RISK_RESULT_SUFFIX>.json (Adım 6 eşik seçimi). Dosya yoksa ilgili alan boş kalır.

    Döndürür:
        {model, oznitelik_sayisi, segmentasyon, test: {düzey: metrikler}, esik: {...}} sözlüğü.
    """
    out = {"model": "-", "oznitelik_sayisi": 0, "segmentasyon": SEGMENTATION_METHOD, "test": {}, "esik": {}}
    meta_path = MODELS_DIR / f"{ACTIVE_MODEL_FILE}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out["model"] = MODEL_TITLES.get(meta.get("model"), meta.get("model", "-"))
        out["oznitelik_sayisi"] = len(meta.get("oznitelikler", []))
        out["segmentasyon"] = meta.get("segmentasyon", SEGMENTATION_METHOD)
    test_path = STAGE_OUTPUT_DIRS["evaluation"] / f"test_metrikleri{ACTIVE_EVAL_SUFFIX}.csv"
    if test_path.exists():
        table = pd.read_csv(test_path).set_index("duzey")
        for key, row in [("hucre", "hucre (kenara degmeyen)"), ("goruntu", "goruntu (hucre bulunan)")]:
            if row in table.index:
                out["test"][key] = table.loc[row].to_dict()
    thr_path = STAGE_OUTPUT_DIRS["risk"] / f"risk_esikleri{RISK_RESULT_SUFFIX}.json"
    if thr_path.exists():
        out["esik"] = json.loads(thr_path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# PDF rapor
# ---------------------------------------------------------------------------

def register_fonts() -> tuple[str, str]:
    """
    Türkçe karakterleri (ç, ğ, ı, İ, ö, ş, ü) içeren DejaVuSans fontunu ReportLab'e kaydeder.

    DejaVuSans matplotlib ile birlikte gelir (ek kurulum gerekmez). ReportLab'in yerleşik
    Helvetica fontu ğ, ı, ş gibi karakterleri içermez.

    Döndürür:
        (normal font adı, kalın font adı).
    """
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in [("DejaVuSans", "DejaVuSans.ttf"), ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    # Paragraflardaki <b> etiketinin kalın fonta eşlenmesi için aile kaydı
    pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
                                  italic="DejaVuSans", boldItalic="DejaVuSans-Bold")
    return "DejaVuSans", "DejaVuSans-Bold"


def rgb_to_flowable(image: np.ndarray, width_cm: float) -> Image:
    """
    RGB numpy görüntüsünü PNG olarak belleğe kodlayıp verilen genişlikte ReportLab Image'ına çevirir.

    Parametreler:
        image: RGB uint8 görüntü.
        width_cm: Sayfadaki genişlik (cm); yükseklik en-boy oranından.

    Döndürür:
        platypus Image nesnesi.
    """
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Görüntü PNG olarak kodlanamadı")
    h, w = image.shape[:2]
    return Image(io.BytesIO(buf.tobytes()), width=width_cm * cm, height=width_cm * cm * h / w)


def fmt_pct(x: float) -> str:
    """
    0-1 aralığındaki oranı Türkçe yüzde biçimine çevirir (ör. 0.25 → "%25,0").

    Parametreler:
        x: Oran (NaN olabilir).

    Döndürür:
        Biçimlenmiş metin; NaN için "-".
    """
    return "-" if x is None or np.isnan(x) else f"%{100 * x:.1f}".replace(".", ",")


def dominant_label(risk: dict) -> str:
    """
    Baskın lösemi tipinin rapor ve arayüzde gösterilecek metnini üretir (tek biçim, her yerde aynı).

    Parametreler:
        risk: compute_risk çıktısı.

    Döndürür:
        Orta/Yüksek riskte "anormal hücreler içinde ALL %87,5"; diğer durumlarda NO_DOMINANT ("—").
    """
    if risk["baskin_tip"] == NO_DOMINANT:
        return NO_DOMINANT
    return f"anormal hücreler içinde {risk['baskin_tip']} {fmt_pct(risk['baskin_oran'])}"


def fmt_num(x: float, digits: int = 3) -> str:
    """
    Sayıyı ondalık virgülle biçimler.

    Parametreler:
        x: Sayı (NaN olabilir).
        digits: Ondalık basamak.

    Döndürür:
        Biçimlenmiş metin; NaN için "-".
    """
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}".replace(".", ",")


def report_styles() -> dict[str, ParagraphStyle]:
    """
    Raporda kullanılan paragraf stillerini oluşturur.

    Döndürür:
        {stil adı: ParagraphStyle} sözlüğü.
    """
    font, bold = register_fonts()
    base = ParagraphStyle("metin", fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#0b0b0b"))
    return {
        "metin": base,
        "kucuk": ParagraphStyle("kucuk", parent=base, fontSize=8, leading=10, textColor=colors.HexColor("#52514e")),
        "baslik": ParagraphStyle("baslik", parent=base, fontName=bold, fontSize=16, leading=20, spaceAfter=2),
        "bolum": ParagraphStyle("bolum", parent=base, fontName=bold, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4),
        "uyari": ParagraphStyle("uyari", parent=base, fontName=bold, fontSize=9, leading=12, alignment=TA_CENTER,
                                textColor=colors.HexColor("#8a1c1c")),
        "hucre": ParagraphStyle("hucre", parent=base, fontSize=8, leading=10),
    }


def legend_table(styles: dict) -> Table:
    """
    Kontur renklerinin lejantını (ALL / AML / Healthy renk kutuları) tablo olarak oluşturur.

    Parametreler:
        styles: report_styles çıktısı.

    Döndürür:
        platypus Table.
    """
    names = {"ALL": "ALL (akut lenfoblastik)", "AML": "AML (akut miyeloid)", "Healthy": "Healthy (normal)"}
    row = []
    for c in CLASS_NAMES:
        row += ["", Paragraph(names[c], styles["kucuk"])]
    table = Table([row], colWidths=[0.45 * cm, 4.2 * cm] * len(CLASS_NAMES), rowHeights=[0.45 * cm])
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, c in enumerate(CLASS_NAMES):
        style.append(("BACKGROUND", (2 * i, 0), (2 * i, 0), colors.HexColor(CLASS_COLORS[c])))
    table.setStyle(TableStyle(style))
    return table


def summary_table(risk: dict, styles: dict) -> Table:
    """
    Özet kutusunu oluşturur: hücre sayıları, anormal oran, risk seviyesi, baskın tip, güven.

    Parametreler:
        risk: compute_risk çıktısı.
        styles: report_styles çıktısı.

    Döndürür:
        platypus Table.
    """
    counts = risk["hucre_sayilari"]
    dom = dominant_label(risk)
    reasons = "<br/>".join(f"• {r}" for r in risk["guven_nedenleri"]) or "Güven bayrağı yok."
    level_colors = {"Düşük": "#1baf7a", "Orta": "#d9a400", "Yüksek": "#c0392b", RISK_NO_CELL_LEVEL: "#52514e"}
    level = f"<font color='{level_colors[risk['risk_seviyesi']]}'><b>{risk['risk_seviyesi']}</b></font>"
    low, high = risk["esikler"]
    p = lambda text: Paragraph(text, styles["metin"])  # noqa: E731 (kısa yardımcı)
    rows = [
        [p("<b>Toplam hücre</b>"), p(str(risk["toplam_hucre"])),
         p("<b>ALL / AML / Healthy</b>"), p(f"{counts['ALL']} / {counts['AML']} / {counts['Healthy']}")],
        [p("<b>Anormal hücre oranı</b>"), p(f"<b>{fmt_pct(risk['anormal_oran'])}</b> ({risk['anormal_hucre']} hücre)"),
         p("<b>Yumuşak oran</b>"), p(f"{fmt_pct(risk['anormal_oran_yumusak'])} (ort. P(ALL)+P(AML))")],
        [p("<b>Risk seviyesi</b>"), p(level),
         p("<b>Eşikler</b>"), p(f"Düşük &lt; {fmt_num(low, 2)} ≤ Orta &lt; {fmt_num(high, 2)} ≤ Yüksek")],
        [p("<b>Baskın lösemi tipi</b>"), p(dom),
         p("<b>Segmentasyon</b>"), p(METHOD_NAMES[risk["segmentasyon_yontemi"]]
                                     + (" (geri dönüş)" if risk["geri_donus"] else ""))],
        [p("<b>Güven durumu</b>"), p(f"<b>{risk['guven']}</b>"), p(reasons), ""],
    ]
    table = Table(rows, colWidths=[3.6 * cm, 4.6 * cm, 3.4 * cm, 6.4 * cm])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#52514e")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6d5d0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f3ef")),
        ("SPAN", (2, 4), (3, 4)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def cell_table(cells: pd.DataFrame, styles: dict) -> list:
    """
    Hücre tablosunu oluşturur: her hücre için numara, sınıf, üç olasılık ve kenar durumu.

    En fazla REPORT_MAX_CELL_ROWS satır gösterilir; kalan hücreler sınıf sayılarıyla tek satırda özetlenir.

    Parametreler:
        cells: Hücre tablosu (hucre_no, sinif, p_*, kenara_degiyor).
        styles: report_styles çıktısı.

    Döndürür:
        platypus akış elemanları listesi.
    """
    if cells.empty:
        return [Paragraph("Görüntüde hücre bulunamadı.", styles["metin"])]
    font, bold = register_fonts()
    header = ["No", "Sınıf", "P(ALL)", "P(AML)", "P(Healthy)", "Tahmin olasılığı", "Kenara değiyor"]
    shown = cells.head(REPORT_MAX_CELL_ROWS)
    rows = [header]
    for _, c in shown.iterrows():
        rows.append([str(c["hucre_no"]), c["sinif"], fmt_num(c["p_ALL"]), fmt_num(c["p_AML"]),
                     fmt_num(c["p_Healthy"]), fmt_num(c[f"p_{c['sinif']}"]), "Evet" if c["kenara_degiyor"] else "Hayır"])
    table = Table(rows, colWidths=[1.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 3.0 * cm, 2.8 * cm], repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#52514e")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f3ef")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for i, (_, c) in enumerate(shown.iterrows(), start=1):
        style.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(CLASS_COLORS[c["sinif"]])))
    table.setStyle(TableStyle(style))
    out = [table]
    rest = cells.iloc[REPORT_MAX_CELL_ROWS:]
    if not rest.empty:
        rc = rest["sinif"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        out.append(Spacer(1, 3))
        out.append(Paragraph(f"… ve {len(rest)} hücre daha: ALL {rc['ALL']}, AML {rc['AML']}, Healthy {rc['Healthy']} "
                             "(özet sayılara dahildir).", styles["kucuk"]))
    return out


def pipeline_summary(result: dict, model: dict, styles: dict) -> list:
    """
    Pipeline ve model bilgisi bölümünü oluşturur.

    Parametreler:
        result: run_pipeline çıktısı (segmentasyon bilgisi için).
        model: model_summary çıktısı.
        styles: report_styles çıktısı.

    Döndürür:
        platypus akış elemanları listesi.
    """
    seg = result["segmentasyon"]
    used = METHOD_NAMES[seg["yontem"]] + (" (K-means hücre bulamadı, geri dönüş)" if seg["geri_donus"] else "")
    steps = [
        "Ön işleme: 256×256 kenar doldurmalı boyutlandırma → dolgu maskesi → arka plan tabanlı renk "
        "normalizasyonu (Reinhard) → Lab b* kanalı (ters) → 5×5 medyan → CLAHE (clip 1,0; 64 px karo).",
        f"Segmentasyon: {used}. Tahmini eritrosit çapı {fmt_num(seg['eritrosit_capi'], 1)} px "
        f"({'görüntüden' if seg['eritrosit_tahmini_guvenilir'] else 'görüntü tipi yedeği'}, tip: "
        f"{'yayma' if seg['tip'] == 'yayma' else 'yakın çekim'}); boyut parametreleri bu çapa oranlı.",
        f"Öznitelik çıkarma: {model['oznitelik_sayisi'] or 58} öznitelik (şekil, Lab/HSV renk, GLCM doku), "
        "eritrosit çapı 10 px referans ölçeğinde.",
        f"Sınıflandırma: {model['model']}, hücre başına ALL / AML / Healthy olasılıkları.",
        "Risk skoru: anormal (ALL + AML) hücre sayısı / toplam hücre; seviye eşikleri geliştirme setinden seçildi.",
    ]
    out = [Paragraph(f"• {s}", styles["metin"]) for s in steps]
    test = model["test"]
    if test:
        cell, img = test.get("hucre", {}), test.get("goruntu", {})
        out.append(Spacer(1, 4))
        out.append(Paragraph(
            f"<b>Model başarısı (kilitli test seti):</b> hücre düzeyi makro F1 {fmt_num(cell.get('makro_f1', np.nan))}, "
            f"görüntü düzeyi makro F1 {fmt_num(img.get('makro_f1', np.nan))}. Duyarlılık (hücre): ALL "
            f"{fmt_num(cell.get('duyarlilik_ALL', np.nan))}, AML {fmt_num(cell.get('duyarlilik_AML', np.nan))}, "
            f"Healthy {fmt_num(cell.get('duyarlilik_Healthy', np.nan))}; anormal hücre duyarlılığı "
            f"{fmt_num(cell.get('anormal_duyarlilik', np.nan))}.", styles["metin"]))
    thr = model["esik"]
    if thr:
        out.append(Paragraph(
            f"<b>Risk eşikleri (geliştirme seti):</b> Düşük/Orta {fmt_num(thr['dusuk_orta'], 2)} — lösemi görüntülerinin "
            f"{fmt_pct(thr['gelistirme_duyarlilik_dusuk_orta'])}'i bu eşiğin üstünde (hedef ≥ {fmt_pct(RISK_TARGET_SENSITIVITY)}); "
            f"Orta/Yüksek {fmt_num(thr['orta_yuksek'], 2)} — Healthy görüntülerinin "
            f"{fmt_pct(thr['gelistirme_ozgulluk_orta_yuksek'])}'i bu eşiğin altında (hedef ≥ {fmt_pct(RISK_TARGET_SPECIFICITY)}).",
            styles["metin"]))
    return out


def page_footer(canvas, doc) -> None:
    """
    Her sayfanın altına sayfa numarasını yazar (ReportLab onPage geri çağrısı).

    Parametreler:
        canvas: ReportLab canvas.
        doc: Belge şablonu.
    """
    font, _ = register_fonts()
    canvas.saveState()
    canvas.setFont(font, 7)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Sayfa {doc.page}")
    canvas.drawString(1.6 * cm, 1.0 * cm, "Araştırma ve eğitim amaçlı prototip — klinik teşhis aracı değildir.")
    canvas.restoreState()


def generate_pdf_report(result: dict, output_path: str | Path | None = None) -> Path:
    """
    Uçtan uca pipeline sonucundan PDF rapor üretir.

    İçerik: başlık, tarih, görüntü adı; orijinal görüntü ve sınıf renkli hücre konturları yan yana
    (lejantla); özet kutusu; hücre tablosu (en fazla 30 satır); pipeline ve model bilgisi; sabit uyarı.

    Parametreler:
        result: src.pipeline.run_pipeline çıktısı ("goruntu_adi", "gorsel_orijinal", "gorsel_kontur",
            "hucreler", "risk", "segmentasyon").
        output_path: PDF yolu; None ise reports/<görüntü adı>_rapor.pdf.

    Döndürür:
        Oluşturulan PDF dosyasının yolu.
    """
    styles = report_styles()
    name = result["goruntu_adi"]
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"{Path(name).stem.replace(' ', '_')}_rapor.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    risk = result["risk"]
    model = model_summary()

    img_w = 8.6  # cm; iki görsel + aralık A4'ün 18 cm'lik yazı alanına sığar
    images = Table([[rgb_to_flowable(result["gorsel_orijinal"], img_w), rgb_to_flowable(result["gorsel_kontur"], img_w)],
                    [Paragraph("Orijinal görüntü", styles["kucuk"]),
                     Paragraph("Hücre çekirdeği konturları (numaralar hücre tablosundaki No)", styles["kucuk"])]],
                   colWidths=[(img_w + 0.4) * cm] * 2)
    images.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))

    story = [
        Paragraph("Lösemi Tarama ve Tanı Destek Raporu", styles["baslik"]),
        Paragraph(f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; Görüntü: {name}",
                  styles["kucuk"]),
        Spacer(1, 6),
        Paragraph(REPORT_DISCLAIMER, styles["uyari"]),
        Spacer(1, 8),
        images, Spacer(1, 2), legend_table(styles),
        Paragraph("Özet", styles["bolum"]), summary_table(risk, styles),
        Paragraph("Hücre tablosu", styles["bolum"]), *cell_table(result["hucreler"], styles),
        KeepTogether([Paragraph("Pipeline özeti ve model bilgisi", styles["bolum"]),
                      *pipeline_summary(result, model, styles)]),
        Spacer(1, 10),
        Paragraph(REPORT_DISCLAIMER, styles["uyari"]),
    ]
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.6 * cm, title="Lösemi Tarama Raporu",
                            author="Lösemi Tarama ve Tanı Destek Sistemi", subject=name)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output_path
