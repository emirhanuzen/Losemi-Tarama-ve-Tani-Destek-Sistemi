"""
Adım 6 analizleri: Otsu geri dönüşünün etkisi, risk eşiklerinin seçimi, test seti risk sonuçları,
figürler ve örnek PDF raporlar.

- Hücre tabloları Adım 3-4'te üretildi (K-means ve Otsu öznitelik tabloları); geri dönüş, K-means'in hücre
  bulamadığı görüntülerde Otsu hücrelerini kullanmaktır. Model yeniden eğitilmez.
- Eşik seçimi yalnızca GELİŞTİRME setinde, katlar dışı (out-of-fold) tahminlerle yapılır: her katta model
  diğer katların K-means hücreleriyle (son modelle aynı yapılandırma) eğitilir, o katın hücrelerini
  (geri dönüş hücreleri dahil) tahmin eder.
- Test seti: kayıtlı son model (models/) ile; yalnızca seçilmiş eşiklerin sonucunu raporlamak için.
- Örnek PDF'ler gerçek uçtan uca pipeline ile (src.pipeline.run_pipeline) üretilir; aynı görüntülerde
  tablo tabanlı sonuçla tutarlılık da kontrol edilir. Örnekler yalnızca kaynağı "bilinmiyor" olan test
  görüntülerinden seçilir: ALL-IDB türevleri lisans gereği paylaşılamaz, reports/ornekler/ ise repoya girer.

İki bölüm (argümansız: ikisi sırayla):
- analiz:   geri dönüş etkisi, eşik seçimi, test risk sonuçları, figürler (yalnızca tablolar; skimage gerekmez).
- ornekler: analiz çıktısından (risk_goruntu_sonuclari.csv) örnek seçimi, PDF'ler ve pipeline tutarlılık
            kontrolü (görüntü işleme kütüphaneleri gerekir).

Çalıştırma (proje kökünden):
    python -m src.risk_analysis [analiz] [ornekler]
"""

import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.classifier import (
    GROUP_COLUMN,
    NO_CELL,
    aggregate_images,
    fit_model,
    label_metrics,
    load_cells,
    load_model,
    load_split,
    reorder_proba,
)
from src.config import (
    ABNORMAL_CLASSES,
    CLASS_NAMES,
    CLASSIFIER_CALIBRATION,
    CLASSIFIER_MODEL,
    CLASSIFIER_PARAMS,
    CV_FOLDS,
    EXAMPLE_REPORTS_DIR,
    N_JOBS,
    RANDOM_STATE,
    RISK_RESULT_SUFFIX,
    RISK_LEVELS,
    RISK_NO_CELL_LEVEL,
    RISK_TARGET_SENSITIVITY,
    RISK_TARGET_SPECIFICITY,
    RISK_THRESHOLDS,
    SEGMENTATION_FALLBACK_METHOD,
    SEGMENTATION_METHOD,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.risk_report import NO_DOMINANT, compute_risk, generate_pdf_report
from src.utils import save_figure, stage_dir

PROBA_COLUMNS = [f"p_{c}" for c in CLASS_NAMES]
# Çıktı dosyası soneki: kalibre modelin sonuçları ayrı dosyalara yazılır (kalibrasyonsuz sonuçlar silinmez)
RS = RISK_RESULT_SUFFIX


# ---------------------------------------------------------------------------
# Hücre tabloları ve tahminler
# ---------------------------------------------------------------------------

def load_cells_with_fallback() -> pd.DataFrame:
    """
    K-means hücre tablosuna, K-means'in hiç hücre bulamadığı görüntüler için Otsu hücrelerini ekler.

    Döndürür:
        Öznitelik tablosu + bolum, kat, "yontem" (kmeans/otsu) ve "geri_donus" sütunları.
    """
    primary = load_cells(SEGMENTATION_METHOD)
    fallback = load_cells(SEGMENTATION_FALLBACK_METHOD)
    split = load_split()
    empty = set(split["dosya_yolu"]) - set(primary["meta_kaynak_dosya"])
    fallback = fallback[fallback["meta_kaynak_dosya"].isin(empty)]
    return pd.concat([primary.assign(yontem=SEGMENTATION_METHOD, geri_donus=False),
                      fallback.assign(yontem=SEGMENTATION_FALLBACK_METHOD, geri_donus=True)], ignore_index=True)


def dev_oof_proba(dev: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """
    Geliştirme setinde katlar dışı olasılıkları hesaplar; eğitim yalnızca K-means hücreleriyle.

    Son modelle aynı yapılandırma (CLASSIFIER_MODEL, CLASSIFIER_PARAMS, CLASSIFIER_CALIBRATION, kenar hücreleri
    dahil) kullanılır; kalibrasyonun iç katları her eğitim katının içinde kaynak gruplarıyla kurulur (fit_model);
    geri dönüş (Otsu) hücreleri eğitime girmez, yalnızca kendi katlarında tahmin edilir.

    Parametreler:
        dev: Geliştirme hücreleri (kat, yontem sütunlarıyla).
        columns: Öznitelik sütunları.

    Döndürür:
        (hücre, 3) olasılık dizisi, CLASS_NAMES sırasında.
    """
    proba = np.full((len(dev), len(CLASS_NAMES)), np.nan)
    X = dev[columns].to_numpy(np.float64)
    y = dev["etiket"].to_numpy()
    folds = dev["kat"].to_numpy()
    trainable = (dev["yontem"] == SEGMENTATION_METHOD).to_numpy()
    groups = dev[GROUP_COLUMN].to_numpy()
    with threadpool_limits(limits=N_JOBS):
        for fold in range(CV_FOLDS):
            train = (folds != fold) & trainable
            model = fit_model(X[train], y[train], groups[train], CLASSIFIER_MODEL, CLASSIFIER_PARAMS,
                              CLASSIFIER_CALIBRATION)
            proba[folds == fold] = reorder_proba(model, model.predict_proba(X[folds == fold]))
            print(f"  [katlar dışı] kat {fold + 1}/{CV_FOLDS}", flush=True)
    return proba


def test_proba(test: pd.DataFrame) -> np.ndarray:
    """
    Kayıtlı son modelle test hücrelerinin olasılıklarını hesaplar.

    Parametreler:
        test: Test hücreleri.

    Döndürür:
        (hücre, 3) olasılık dizisi, CLASS_NAMES sırasında.
    """
    model, meta = load_model()
    with threadpool_limits(limits=N_JOBS):
        return reorder_proba(model, model.predict_proba(test[meta["oznitelikler"]].to_numpy(np.float64)))


# ---------------------------------------------------------------------------
# Görüntü düzeyi risk
# ---------------------------------------------------------------------------

def image_risk_table(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame,
                     thresholds: tuple = RISK_THRESHOLDS) -> pd.DataFrame:
    """
    Her görüntü için compute_risk'i (pipeline ile aynı fonksiyon) çalıştırıp düz bir tablo üretir.

    Parametreler:
        cells: Hücre tablosu (meta_kaynak_dosya, meta_kenara_degiyor, yontem, geri_donus), proba ile aynı sırada.
        proba: Hücre olasılıkları.
        images: Görüntüler (dosya_yolu, etiket, tip, grup_id).
        thresholds: Risk seviyesi eşikleri (None ise seviye hesaplanmaz).

    Döndürür:
        Görüntü başına: dosya_yolu, etiket, tip, grup_id, yontem, geri_donus, toplam_hucre, ALL/AML/Healthy sayıları,
        anormal_oran, anormal_oran_yumusak, baskin_tip, baskin_oran, risk_seviyesi, guven, guven_nedenleri, kenar_orani.
    """
    table = pd.DataFrame(proba, columns=PROBA_COLUMNS)
    table["sinif"] = np.asarray(CLASS_NAMES)[proba.argmax(axis=1)]
    table["kenara_degiyor"] = cells["meta_kenara_degiyor"].to_numpy(bool)
    table["dosya_yolu"] = cells["meta_kaynak_dosya"].to_numpy()
    table["yontem"] = cells["yontem"].to_numpy()
    groups = dict(tuple(table.groupby("dosya_yolu")))
    rows = []
    for _, img in images.iterrows():
        g = groups.get(img["dosya_yolu"], table.iloc[0:0])
        fallback = bool(len(g)) and g["yontem"].iloc[0] != SEGMENTATION_METHOD
        # Hücresiz görüntüde de K-means boş kaldığı için Otsu denenmiştir (o da boşsa yine geri dönüş)
        fallback = fallback or g.empty
        r = compute_risk(g, SEGMENTATION_FALLBACK_METHOD if fallback else SEGMENTATION_METHOD, fallback, thresholds)
        rows.append({
            "dosya_yolu": img["dosya_yolu"], "etiket": img["etiket"], "tip": img["tip"], "grup_id": img["grup_id"],
            "yontem": r["segmentasyon_yontemi"], "geri_donus": r["geri_donus"], "toplam_hucre": r["toplam_hucre"],
            **{f"n_{c}": r["hucre_sayilari"][c] for c in CLASS_NAMES},
            "anormal_oran": r["anormal_oran"], "anormal_oran_yumusak": r["anormal_oran_yumusak"],
            "baskin_tip": r["baskin_tip"], "baskin_oran": r["baskin_oran"],
            "baskin_tip_ham": r["baskin_tip_ham"], "baskin_oran_ham": r["baskin_oran_ham"],
            "risk_seviyesi": r["risk_seviyesi"], "guven": r["guven"],
            "guven_nedenleri": " | ".join(r["guven_nedenleri"]), "kenar_orani": r["kenar_orani"],
            "kaynak": img.get("kaynak", "bilinmiyor"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Eşik seçimi
# ---------------------------------------------------------------------------

def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """
    Aday eşikleri üretir: sıralı benzersiz skorların ardışık orta noktaları (+ uçlar).

    Orta nokta, eşiğin tam bir skor değerine (ör. 1 hücreli görüntüde 0 veya 1, 2 hücrelide 0.5)
    denk gelmesini ve sonucun "≥" / ">" seçimine bağlı kalmasını önler.

    Parametreler:
        scores: Görüntü skorları (NaN'sız).

    Döndürür:
        Artan sıralı aday eşikler.
    """
    u = np.unique(scores)
    return np.concatenate([[u[0] - 0.01], (u[:-1] + u[1:]) / 2, [u[-1] + 0.01]])


def sens_spec(scores: np.ndarray, abnormal: np.ndarray, t: float) -> tuple[float, float]:
    """
    Bir eşikte duyarlılık ve özgüllüğü hesaplar (skor ≥ t → pozitif).

    Parametreler:
        scores: Görüntü skorları.
        abnormal: Gerçek etiket lösemi (ALL/AML) mi.
        t: Eşik.

    Döndürür:
        (duyarlılık = lösemi görüntülerinde skor ≥ t oranı, özgüllük = Healthy görüntülerde skor < t oranı).
    """
    return float(np.mean(scores[abnormal] >= t)), float(np.mean(scores[~abnormal] < t))


def select_thresholds(scores: np.ndarray, abnormal: np.ndarray,
                      target_sens: float = RISK_TARGET_SENSITIVITY,
                      target_spec: float = RISK_TARGET_SPECIFICITY) -> tuple[float, float, pd.DataFrame]:
    """
    İki hedefle risk eşiklerini seçer (gerekçe: config.RISK_THRESHOLDS yorumu).

    - Düşük/Orta: duyarlılık ≥ target_sens olan EN BÜYÜK aday (lösemi görüntülerinin en fazla %2'si "Düşük").
    - Orta/Yüksek: özgüllük ≥ target_spec olan EN KÜÇÜK aday (Healthy görüntülerinin en fazla %5'i "Yüksek").
    - İkinci eşik birinciden küçük çıkarsa (iki dağılım arasında boşluk varsa) yerleri değiştirilir: iki koşul
      da sağlanmaya devam eder, aradaki aralık "Orta" olur.

    Parametreler:
        scores: Geliştirme görüntü skorları (hücresiz görüntüler hariç).
        abnormal: Gerçek etiket lösemi mi.
        target_sens: Düşük/Orta eşiğinde hedef duyarlılık.
        target_spec: Orta/Yüksek eşiğinde hedef özgüllük.

    Döndürür:
        (düşük/orta eşiği, orta/yüksek eşiği, aday tablosu [esik, duyarlilik, ozgulluk, youden]).
    """
    cands = threshold_candidates(scores)
    rows = [(t, *sens_spec(scores, abnormal, t)) for t in cands]
    table = pd.DataFrame(rows, columns=["esik", "duyarlilik", "ozgulluk"])
    table["youden"] = table["duyarlilik"] + table["ozgulluk"] - 1
    ok_sens = table[table["duyarlilik"] >= target_sens]
    ok_spec = table[table["ozgulluk"] >= target_spec]
    if ok_sens.empty or ok_spec.empty:
        raise ValueError("Hedef duyarlılık/özgüllük hiçbir eşikte sağlanamıyor")
    low, high = float(ok_sens["esik"].max()), float(ok_spec["esik"].min())
    if high < low:
        low, high = high, low
    return round(low, 4), round(high, 4), table


def level_metrics(img: pd.DataFrame, thresholds: tuple) -> dict:
    """
    Seçilen eşiklerde görüntü düzeyi duyarlılık/özgüllük ve seviye dağılımını özetler.

    Parametreler:
        img: image_risk_table çıktısı.
        thresholds: (düşük/orta, orta/yüksek).

    Döndürür:
        Düz metrik sözlüğü (hücresiz görüntüler ayrı sayılır).
    """
    scored = img[img["toplam_hucre"] > 0]
    s = scored["anormal_oran"].to_numpy()
    ab = scored["etiket"].isin(ABNORMAL_CLASSES).to_numpy()
    out = {"n_goruntu": len(img), "n_hucresiz": int((img["toplam_hucre"] == 0).sum())}
    for name, t in zip(["dusuk_orta", "orta_yuksek"], thresholds):
        sens, spec = sens_spec(s, ab, t)
        out[f"duyarlilik_{name}"], out[f"ozgulluk_{name}"] = sens, spec
    level = img["risk_seviyesi"]
    out["lösemi_dusuk_orani"] = float((level[img["etiket"] != "Healthy"] == "Düşük").mean())
    out["healthy_yuksek_orani"] = float((level[img["etiket"] == "Healthy"] == "Yüksek").mean())
    out["dusuk_guven_orani"] = float((img["guven"] == "Düşük").mean())
    # Gösterilen baskın tip (yalnızca Orta/Yüksek riskte) gerçek lösemi tipiyle aynı mı
    abn = img[img["etiket"].isin(ABNORMAL_CLASSES) & (img["baskin_tip"] != NO_DOMINANT)]
    out["baskin_tip_dogrulugu"] = float((abn["baskin_tip"] == abn["etiket"]).mean()) if len(abn) else np.nan
    return out


# ---------------------------------------------------------------------------
# Geri dönüşün etkisi (test seti)
# ---------------------------------------------------------------------------

def fallback_effect(test: pd.DataFrame, proba: np.ndarray, test_images: pd.DataFrame) -> pd.DataFrame:
    """
    Otsu geri dönüşünün test seti görüntü düzeyi sınıflandırma sonuçlarına etkisini ölçer.

    "Geri dönüşsüz": yalnızca K-means hücreleri (Adım 5'teki değerlendirme). "Geri dönüşlü": K-means boşsa
    Otsu hücreleri. Hücre bulunamayan görüntü kaçırma sayılır ("tüm görüntüler" metrikleri).

    Parametreler:
        test: Test hücreleri (yontem sütunuyla).
        proba: Test hücre olasılıkları.
        test_images: Test görüntüleri.

    Döndürür:
        Satırlar: geri dönüşsüz / geri dönüşlü; sütunlar: kapsama, makro F1, sınıf duyarlılıkları vb.
    """
    rows = []
    primary = (test["yontem"] == SEGMENTATION_METHOD).to_numpy()
    for name, sel in [("geri donussuz (yalnizca K-means)", primary), ("geri donuslu (K-means, bossa Otsu)", np.ones_like(primary))]:
        sub = test[sel].reset_index(drop=True)
        img = aggregate_images(sub, proba[sel], test_images)
        m = label_metrics(img["etiket"].to_numpy(), img["tahmin"].to_numpy())
        rows.append({"durum": name, "hucresiz_goruntu": int((img["tahmin"] == NO_CELL).sum()),
                     "kapsama": float((img["tahmin"] != NO_CELL).mean()),
                     **{k: v for k, v in m.items() if k != "karisiklik"}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figürler
# ---------------------------------------------------------------------------

def plot_risk_distribution(dev_img: pd.DataFrame, test_img: pd.DataFrame, thresholds: tuple) -> None:
    """
    Sınıf bazında risk skoru (anormal hücre oranı) dağılımını eşik çizgileriyle çizer (geliştirme ve test).

    Satırlar sınıflar, sütunlar veri bölümü; çubuklar o sınıftaki görüntülerin yüzdesi.

    Parametreler:
        dev_img: Geliştirme görüntü risk tablosu (katlar dışı).
        test_img: Test görüntü risk tablosu.
        thresholds: (düşük/orta, orta/yüksek).
    """
    bins = np.linspace(0, 1, 21)  # 0.05 genişlik: 1-20 hücreli görüntülerin oranları ayrışır
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(9.2, 6.4), sharex=True, facecolor=SURFACE)
    for j, (title, img) in enumerate([("Geliştirme (katlar dışı)", dev_img), ("Test", test_img)]):
        for i, cls in enumerate(CLASS_NAMES):
            ax = axes[i, j]
            s = img.loc[(img["etiket"] == cls) & (img["toplam_hucre"] > 0), "anormal_oran"].clip(0, 1 - 1e-9)
            weights = np.full(len(s), 100 / max(len(s), 1))
            ax.hist(s, bins=bins, weights=weights, color=CLASS_COLORS[cls], edgecolor=SURFACE, linewidth=0.6)
            for t, lab in zip(thresholds, ["Düşük | Orta", "Orta | Yüksek"]):
                ax.axvline(t, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
                if i == 0:
                    ax.text(t, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=8, color=TEXT_SECONDARY)
            ax.set_ylim(0, 100)
            ax.set_ylabel(f"{cls}\n% görüntü (n={len(s)})", color=TEXT_SECONDARY, fontsize=9)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, pad=16)
        axes[-1, j].set_xlabel("Risk skoru (anormal hücre oranı)", color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "risk", f"risk_skoru_dagilimi{RS}")


def plot_level_confusion(img: pd.DataFrame, name: str, title: str) -> None:
    """
    Gerçek sınıf × risk seviyesi karışıklık matrisini (satır yüzdesi renkli, ham sayılı) çizer.

    Parametreler:
        img: Görüntü risk tablosu.
        name: Dosya adı (uzantısız).
        title: Başlık.
    """
    levels = RISK_LEVELS + ([RISK_NO_CELL_LEVEL] if (img["risk_seviyesi"] == RISK_NO_CELL_LEVEL).any() else [])
    cm = pd.crosstab(img["etiket"], img["risk_seviyesi"]).reindex(index=CLASS_NAMES, columns=levels, fill_value=0).to_numpy()
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(levels) + 1.6, 3.6), facecolor=SURFACE)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n%{100 * pct[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.55 else TEXT_PRIMARY)
    ax.set_xticks(range(len(levels)), ["Değerlendi-\nrilemedi" if lv == RISK_NO_CELL_LEVEL else lv for lv in levels])
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Risk seviyesi", color=TEXT_SECONDARY)
    ax.set_ylabel("Gerçek sınıf", color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "risk", f"{name}{RS}")


# ---------------------------------------------------------------------------
# Örnek raporlar
# ---------------------------------------------------------------------------

def choose_examples(test_img: pd.DataFrame) -> pd.DataFrame:
    """
    Örnek PDF'ler için test görüntüleri seçer: her sınıftan 2 normal güvenli + 1 düşük güvenli görüntü.

    Seçim rastgeledir (RANDOM_STATE), doğru/yanlış tahmine bakılmaz (iyi örnek seçme yanlılığı olmasın).
    Düşük güvenli örnek için öncelik Otsu geri dönüşüyle hücre bulunan görüntüdür (geri dönüşü göstermek için).
    Yalnızca kaynağı "bilinmiyor" olan görüntüler aday olur: ALL-IDB türevleri lisans gereği paylaşılamaz.

    Parametreler:
        test_img: Test görüntü risk tablosu.

    Döndürür:
        Seçilen satırlar + "ornek_turu" sütunu.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    test_img = test_img[test_img["kaynak"] == "bilinmiyor"]
    picks = []
    for cls in CLASS_NAMES:
        pool = test_img[(test_img["etiket"] == cls) & (test_img["guven"] == "Normal")]
        idx = rng.choice(pool.index.to_numpy(), size=min(2, len(pool)), replace=False)
        picks.append(test_img.loc[idx].assign(ornek_turu=cls))
    low = test_img[(test_img["guven"] == "Düşük") & test_img["geri_donus"] & (test_img["toplam_hucre"] > 0)]
    if low.empty:
        low = test_img[(test_img["guven"] == "Düşük") & (test_img["toplam_hucre"] > 0)]
    picks.append(test_img.loc[[rng.choice(low.index.to_numpy())]].assign(ornek_turu="dusuk_guven"))
    return pd.concat(picks)


def safe_name(text: str) -> str:
    """
    Dosya adında sorun çıkarabilecek karakterleri (boşluk, parantez) alt çizgiye çevirir.

    Parametreler:
        text: Ham ad.

    Döndürür:
        Güvenli ad.
    """
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", text)).strip("_")


def make_example_reports(examples: pd.DataFrame) -> pd.DataFrame:
    """
    Seçilen görüntülerde uçtan uca pipeline'ı çalıştırıp PDF üretir ve tablo tabanlı sonuçla karşılaştırır.

    Pipeline görüntü tipini kendisi tahmin eder (gerçek kullanımdaki gibi); tablo tabanlı sonuç veri seti
    indeksindeki tipi kullanmıştı. Hücre sayısı, risk skoru ve seviyesi karşılaştırılır.

    Parametreler:
        examples: choose_examples çıktısı.

    Döndürür:
        Örnek başına: dosya, PDF yolu, tablo ve pipeline sonuçları.
    """
    from src.pipeline import run_pipeline
    EXAMPLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (_, ex) in enumerate(examples.iterrows(), 1):
        result = run_pipeline(ex["dosya_yolu"])
        stem = safe_name(f"{ex['ornek_turu']}_{ex['etiket']}_{result['goruntu_adi'].rsplit('.', 1)[0]}")
        pdf = generate_pdf_report(result, EXAMPLE_REPORTS_DIR / f"{stem}.pdf")
        r = result["risk"]
        rows.append({
            "ornek": ex["ornek_turu"], "gercek": ex["etiket"], "goruntu": result["goruntu_adi"], "pdf": pdf.name,
            "hucre_tablo": ex["toplam_hucre"], "hucre_pipeline": r["toplam_hucre"],
            "skor_tablo": ex["anormal_oran"], "skor_pipeline": r["anormal_oran"],
            "seviye_tablo": ex["risk_seviyesi"], "seviye_pipeline": r["risk_seviyesi"],
            "baskin_tip": r["baskin_tip"], "guven": r["guven"], "yontem": r["segmentasyon_yontemi"],
            "nedenler": " | ".join(r["guven_nedenleri"]),
        })
        pd.DataFrame(rows).to_csv(stage_dir("risk") / f"ornek_raporlar{RS}.csv", index=False, encoding="utf-8")
        print(f"  [{k}/{len(examples)}] {pdf.name}: {r['risk_seviyesi']} (skor {r['anormal_oran']:.2f}, "
              f"güven {r['guven']})", flush=True)
    return pd.DataFrame(rows)


def consistency_check(test_img: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Rastgele test görüntülerinde uçtan uca pipeline sonucunu tablo tabanlı sonuçla karşılaştırır.

    Parametreler:
        test_img: Test görüntü risk tablosu.
        n: Kontrol edilecek görüntü sayısı.

    Döndürür:
        Görüntü başına hücre sayısı / skor / seviye eşleşmesi tablosu.
    """
    from src.pipeline import run_pipeline
    sample = test_img.sample(n=min(n, len(test_img)), random_state=RANDOM_STATE)
    rows = []
    for _, row in sample.iterrows():
        r = run_pipeline(row["dosya_yolu"])["risk"]
        rows.append({"goruntu": Path(row["dosya_yolu"]).name, "hucre_tablo": row["toplam_hucre"],
                     "hucre_pipeline": r["toplam_hucre"], "skor_tablo": row["anormal_oran"],
                     "skor_pipeline": r["anormal_oran"], "seviye_esit": row["risk_seviyesi"] == r["risk_seviyesi"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    """Geri dönüş etkisini ölçer, eşikleri seçer, test risk sonuçlarını ve figürleri üretir (yalnızca tablolar)."""
    out_dir = stage_dir("risk")
    start = time.time()
    split = load_split()
    cells = load_cells_with_fallback()
    _, meta = load_model()
    columns = meta["oznitelikler"]
    images = {b: split[split["bolum"] == b].reset_index(drop=True) for b in ["gelistirme", "test"]}
    parts = {b: cells[cells["bolum"] == b].reset_index(drop=True) for b in ["gelistirme", "test"]}
    for b in parts:
        empty_primary = len(images[b]) - parts[b].loc[~parts[b]["geri_donus"], "meta_kaynak_dosya"].nunique()
        rescued = parts[b].loc[parts[b]["geri_donus"], "meta_kaynak_dosya"].nunique()
        print(f"{b}: K-means hücresiz görüntü {empty_primary}, Otsu ile hücre bulunan {rescued}")

    print("Geliştirme seti katlar dışı tahminler...")
    proba = {"gelistirme": dev_oof_proba(parts["gelistirme"], columns), "test": test_proba(parts["test"])}

    # 1) Geri dönüşün etkisi (test, görüntü düzeyi sınıflandırma)
    effect = fallback_effect(parts["test"], proba["test"], images["test"])
    effect.to_csv(out_dir / f"geri_donus_etkisi{RS}.csv", index=False, encoding="utf-8")
    fb_test = parts["test"][parts["test"]["geri_donus"]].copy()
    fb_test["tahmin"] = np.asarray(CLASS_NAMES)[proba["test"][parts["test"]["geri_donus"].to_numpy()].argmax(axis=1)]
    print("Test — Otsu ile kurtarılan görüntülerin hücre tahminleri:\n"
          + fb_test.groupby(["meta_kaynak_dosya", "etiket"])["tahmin"].agg(lambda x: dict(x.value_counts()))
          .reset_index().assign(meta_kaynak_dosya=lambda d: d["meta_kaynak_dosya"].map(lambda p: Path(p).name))
          .to_string(index=False))
    cols = ["durum", "hucresiz_goruntu", "kapsama", "dogruluk", "makro_f1"] + [f"duyarlilik_{c}" for c in CLASS_NAMES] \
        + ["anormal_duyarlilik", "healthy_ozgulluk"]
    print("\n=== Otsu geri dönüşünün etkisi (test, görüntü düzeyi; hücresiz = hata) ===\n"
          + effect[cols].round(3).to_string(index=False))

    # 2) Eşik seçimi (geliştirme, katlar dışı)
    dev_img = image_risk_table(parts["gelistirme"], proba["gelistirme"], images["gelistirme"], (None, None))
    scored = dev_img[dev_img["toplam_hucre"] > 0]
    low, high, cand = select_thresholds(scored["anormal_oran"].to_numpy(), scored["etiket"].isin(ABNORMAL_CLASSES).to_numpy())
    cand.to_csv(out_dir / f"esik_adaylari_gelistirme{RS}.csv", index=False, encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    ab = scored["etiket"].isin(ABNORMAL_CLASSES)
    print(f"\nGeliştirme AUC (lösemi vs Healthy): sert skor {roc_auc_score(ab, scored['anormal_oran']):.3f}, "
          f"yumuşak skor {roc_auc_score(ab, scored['anormal_oran_yumusak']):.3f}")
    print(f"Seçilen eşikler (geliştirme): Düşük/Orta {low}, Orta/Yüksek {high}")
    print("Aday eşikler:\n" + cand.round(3).to_string(index=False))
    thresholds = RISK_THRESHOLDS
    if thresholds != (low, high):
        print(f"UYARI: config.RISK_THRESHOLDS {thresholds} seçilenden farklı; config güncellenmeli → ({low}, {high}). "
              "Bu çalıştırmada seçilen eşikler kullanılıyor.")
        thresholds = (low, high)

    dev_img = image_risk_table(parts["gelistirme"], proba["gelistirme"], images["gelistirme"], thresholds)
    test_img = image_risk_table(parts["test"], proba["test"], images["test"], thresholds)
    pd.concat([dev_img.assign(bolum="gelistirme"), test_img.assign(bolum="test")]).to_csv(
        out_dir / f"risk_goruntu_sonuclari{RS}.csv", index=False, encoding="utf-8")
    dev_m, test_m = level_metrics(dev_img, thresholds), level_metrics(test_img, thresholds)
    summary = pd.DataFrame([{"bolum": "gelistirme (katlar disi)", **dev_m}, {"bolum": "test", **test_m}])
    summary.to_csv(out_dir / f"risk_seviyesi_metrikleri{RS}.csv", index=False, encoding="utf-8")
    (out_dir / f"risk_esikleri{RS}.json").write_text(json.dumps({
        "skor": "anormal_oran", "dusuk_orta": thresholds[0], "orta_yuksek": thresholds[1],
        "hedef_duyarlilik": RISK_TARGET_SENSITIVITY, "hedef_ozgulluk": RISK_TARGET_SPECIFICITY,
        "gelistirme_duyarlilik_dusuk_orta": dev_m["duyarlilik_dusuk_orta"],
        "gelistirme_ozgulluk_dusuk_orta": dev_m["ozgulluk_dusuk_orta"],
        "gelistirme_duyarlilik_orta_yuksek": dev_m["duyarlilik_orta_yuksek"],
        "gelistirme_ozgulluk_orta_yuksek": dev_m["ozgulluk_orta_yuksek"],
        "test_duyarlilik_dusuk_orta": test_m["duyarlilik_dusuk_orta"],
        "test_ozgulluk_dusuk_orta": test_m["ozgulluk_dusuk_orta"],
        "test_duyarlilik_orta_yuksek": test_m["duyarlilik_orta_yuksek"],
        "test_ozgulluk_orta_yuksek": test_m["ozgulluk_orta_yuksek"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Risk seviyesi metrikleri ===\n" + summary.round(3).T.to_string(header=False))

    for name, img in [("Geliştirme (katlar dışı)", dev_img), ("Test", test_img)]:
        print(f"\n{name} — gerçek sınıf × risk seviyesi:\n"
              + pd.crosstab(img["etiket"], img["risk_seviyesi"]).to_string())
        print(f"{name} — güven × sınıf:\n" + pd.crosstab(img["guven"], img["etiket"]).to_string())
        low_conf = img[img["guven"] == "Düşük"]
        reasons = pd.Series([x.split(" (")[0].split(";")[0] for r in low_conf["guven_nedenleri"] for x in r.split(" | ")])
        print("Düşük güven nedenleri:\n" + reasons.str.slice(0, 60).value_counts().to_string())
        normal = img[img["guven"] == "Normal"]
        print(f"Güveni normal görüntülerde seviye:\n" + pd.crosstab(normal["etiket"], normal["risk_seviyesi"]).to_string())
    fb = test_img[test_img["geri_donus"]]
    print("\nTest — geri dönüş kullanılan görüntüler:\n"
          + fb[["dosya_yolu", "etiket", "toplam_hucre", "anormal_oran", "risk_seviyesi", "baskin_tip"]]
          .assign(dosya_yolu=fb["dosya_yolu"].map(lambda p: Path(p).name)).to_string(index=False))

    plot_risk_distribution(dev_img, test_img, thresholds)
    plot_level_confusion(test_img, "test_risk_seviyesi_karisiklik", "Test — gerçek sınıf × risk seviyesi")
    plot_level_confusion(dev_img, "gelistirme_risk_seviyesi_karisiklik", "Geliştirme (katlar dışı) — sınıf × risk seviyesi")
    choose_examples(test_img).to_csv(out_dir / f"ornek_secimi{RS}.csv", index=False, encoding="utf-8")
    print(f"\nSüre: {time.time() - start:.0f} sn")


def run_examples() -> None:
    """Analiz çıktısından örnek görüntüleri alıp PDF'leri üretir ve pipeline tutarlılığını kontrol eder."""
    out_dir = stage_dir("risk")
    start = time.time()
    results = pd.read_csv(out_dir / f"risk_goruntu_sonuclari{RS}.csv")
    test_img = results[results["bolum"] == "test"].reset_index(drop=True)
    examples = pd.read_csv(out_dir / f"ornek_secimi{RS}.csv")
    print("Örnek PDF raporlar (uçtan uca pipeline)...")
    examples = make_example_reports(examples)
    print(examples.drop(columns=["nedenler"]).round(3).to_string(index=False))
    check = consistency_check(test_img)
    check.to_csv(out_dir / f"pipeline_tutarlilik{RS}.csv", index=False, encoding="utf-8")
    same_n = (check["hucre_tablo"] == check["hucre_pipeline"]).mean()
    diff = (check["skor_tablo"] - check["skor_pipeline"]).abs().max()
    print(f"\nTutarlılık ({len(check)} test görüntüsü): hücre sayısı eşit %{100 * same_n:.0f}, "
          f"risk seviyesi eşit %{100 * check['seviye_esit'].mean():.0f}, en büyük skor farkı {diff:.3f}")
    if same_n < 1:
        print(check[check["hucre_tablo"] != check["hucre_pipeline"]].to_string(index=False))
    print(f"\nSüre: {time.time() - start:.0f} sn")


PARTS = {"analiz": run_analysis, "ornekler": run_examples}


def main() -> None:
    """Komut satırında adı verilen bölümleri (argümansız: analiz ve ornekler) sırayla çalıştırır."""
    pd.set_option("display.width", 220)
    for name in sys.argv[1:] or list(PARTS):
        print(f"\n=== {name} ===", flush=True)
        PARTS[name]()


if __name__ == "__main__":
    main()
