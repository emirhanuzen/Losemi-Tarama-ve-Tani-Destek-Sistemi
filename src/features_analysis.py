"""
Öznitelik analizi ve sunum figürleri (betimleyici; sınıflandırma YAPMAZ).

- Kısayol testi: ham ve eritrosite oranlanmış ölçülerle görüntü tipi (yayma / yakın çekim)
  tahmin edilebiliyor mu? (lojistik regresyon, grup bazlı çapraz doğrulama)
- Öznitelik önemi: ANOVA F-skoru ve karşılıklı bilgi. YALNIZCA BETİMLEYİCİ: tüm veriyle
  hesaplanır; gerçek öznitelik seçimi Adım 5'te çapraz doğrulamanın içinde (yalnızca eğitim
  katında) yapılacaktır, aksi hâlde test bilgisi seçime sızar.
- Korelasyon, sınıf bazında dağılımlar, GLCM örnekleri, kenara değen hücrelerin etkisi.

Analiz seçilen yöntemin (K-means) öznitelik tablosu üzerinde yapılır.

Çalıştırma (proje kökünden):
    python -m src.features_analysis
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASS_NAMES,
    FEATURE_REF_RBC_DIAMETER,
    FEATURES_CSVS,
    GLCM_DISTANCES,
    GLCM_LEVELS,
    RANDOM_STATE,
    SEGMENTATION_METHOD,
    SEGMENTATION_METHODS,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes
from src.features import (
    feature_columns,
    glcm_properties,
    largest_component,
    masked_glcm,
    quantize_gray,
    read_png,
    texture_features,
    to_reference_scale,
)
from src.segmentation import MASK_NUCLEUS
from src.utils import save_figure, stage_dir

TYPE_COLORS = {"yayma": "#2a78d6", "yakin_cekim": "#eb6834"}
TYPE_NAMES = {"yayma": "Yayma", "yakin_cekim": "Yakın çekim"}
CV_FOLDS = 5          # 5 kat: her katta ~%20 grup test; yakın çekim grupları (245) katlara yeterince dağılır
HIGH_CORRELATION = 0.95
SHAPE_EDGE_FEATURES = ["sekil_alan", "sekil_cevre", "sekil_dairesellik", "sekil_eksantrisite", "sekil_solidity",
                       "sekil_extent", "sekil_buyuk_eksen", "sekil_kucuk_eksen", "sekil_eksen_orani"]


# ---------------------------------------------------------------------------
# Kısayol testi
# ---------------------------------------------------------------------------

def type_predictability(df: pd.DataFrame, column: str) -> dict:
    """
    Tek bir sütunla görüntü tipini (yakın çekim = 1) tahmin eden lojistik regresyonu grup bazlı
    çapraz doğrulamayla değerlendirir.

    Aynı kaynağın augmentasyonları aynı katta kalsın diye StratifiedGroupKFold (grup = grup_id)
    kullanılır. ROC AUC 0.5 → sütun görüntü tipi hakkında bilgi taşımıyor; 1.0 → tipi tam ele veriyor.

    Parametreler:
        df: Öznitelik tablosu (meta_tip, grup_id ve sütun dahil).
        column: Test edilecek sütun (pozitif değerli ölçüler log ile verilir).

    Döndürür:
        {"roc_auc", "dengeli_dogruluk"} (katların ortalaması).
    """
    data = df[[column, "meta_tip", "grup_id"]].dropna()
    x = data[[column]].to_numpy(np.float64)
    y = (data["meta_tip"] == "yakin_cekim").to_numpy().astype(int)
    folds = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, baccs = [], []
    for train, test in folds.split(x, y, data["grup_id"]):
        # class_weight="balanced": yakın çekim hücreleri azınlıkta; aksi hâlde model hep "yayma" der
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        model.fit(x[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
        baccs.append(balanced_accuracy_score(y[test], model.predict(x[test])))
    return {"roc_auc": float(np.mean(aucs)), "dengeli_dogruluk": float(np.mean(baccs))}


def native_texture_sample(df: pd.DataFrame, n_per_type: int = 400) -> pd.DataFrame:
    """
    Karşılaştırma için örnek hücrelerde GLCM kontrastını YEREL çözünürlükte (ölçeklemeden) hesaplar.

    Parametreler:
        df: Öznitelik tablosu.
        n_per_type: Görüntü tipi başına örnek hücre sayısı.

    Döndürür:
        Örnek satırlar + "yerel_doku_contrast_d1" sütunu.
    """
    sample = pd.concat([g.sample(min(len(g), n_per_type), random_state=RANDOM_STATE) for _, g in df.groupby("meta_tip")])
    values = []
    for _, row in sample.iterrows():
        crop = read_png(row["meta_kirpim_yolu"], color=True)
        nucleus = largest_component(read_png(row["meta_maske_yolu"], color=False) == MASK_NUCLEUS)
        values.append(texture_features(crop, nucleus)["doku_contrast_d1"])
    return sample.assign(yerel_doku_contrast_d1=values)


def shortcut_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham ve ölçek normalize ölçülerin görüntü tipini ne kadar ele verdiğini karşılaştırır.

    Tüm hücrelerde ve yalnızca ALL'da (ALL'da iki tip dengeli; sınıf farkı tip tahminine karışmaz).

    Parametreler:
        df: Öznitelik tablosu.

    Döndürür:
        Ölçü × veri kümesi tablo (roc_auc, dengeli_dogruluk).
    """
    df = df.assign(log_ham_alan=np.log(df["meta_ham_alan_px"]), log_oranli_alan=np.log(df["sekil_alan"]),
                   log_ham_cevre=np.log(df["meta_ham_cevre_px"].clip(lower=1)), log_oranli_cevre=np.log(df["sekil_cevre"]))
    tests = [("Alan (ham piksel, log)", "log_ham_alan"), ("Alan / eritrosit çapı² (log)", "log_oranli_alan"),
             ("Çevre (ham piksel, log)", "log_ham_cevre"), ("Çevre / eritrosit çapı (log)", "log_oranli_cevre")]
    rows = []
    for scope, part in [("tüm hücreler", df), ("yalnızca ALL", df[df["etiket"] == "ALL"])]:
        for name, col in tests:
            rows.append({"olcu": name, "kapsam": scope, **type_predictability(part, col)})
        tex = native_texture_sample(part)
        for name, col in [("GLCM contrast d1 (yerel ölçek)", "yerel_doku_contrast_d1"),
                          ("GLCM contrast d1 (referans ölçek)", "doku_contrast_d1")]:
            rows.append({"olcu": name, "kapsam": scope + " (örnek)", **type_predictability(tex, col)})
    return pd.DataFrame(rows).round(3)


def top_feature_type_dependence(df: pd.DataFrame, importance: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    En ayırt edici n özniteliğin her birinin, YALNIZCA ALL içinde görüntü tipini ne kadar ele verdiğini ölçer.

    ALL'da yayma ve yakın çekim dengeli olduğu için sınıf farkı karışmaz. Yüksek AUC: öznitelik
    hâlâ büyütme/çekim koşulunu da taşıyor (ölçek normalizasyonunun yakalamadığı artık bağımlılık).

    Parametreler:
        df: Öznitelik tablosu.
        importance: feature_importance çıktısı.
        n: Öznitelik sayısı.

    Döndürür:
        Öznitelik başına roc_auc ve dengeli doğruluk.
    """
    part = df[df["etiket"] == "ALL"]
    rows = [{"oznitelik": col, **type_predictability(part, col)} for col in importance["oznitelik"].head(n)]
    return pd.DataFrame(rows).round(3)


def plot_scale_normalization(df: pd.DataFrame, shortcut: pd.DataFrame) -> None:
    """
    Ham ve eritrosite oranlanmış alanın yayma / yakın çekim dağılımlarını yan yana çizer.

    Parametreler:
        df: Öznitelik tablosu.
        shortcut: shortcut_test çıktısı (başlıklarda AUC).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), facecolor=SURFACE)
    panels = [("meta_ham_alan_px", "Çekirdek alanı (ham piksel)", "Alan (ham piksel, log)"),
              ("sekil_alan", "Çekirdek alanı / eritrosit çapı²", "Alan / eritrosit çapı² (log)")]
    for r, (scope, part) in enumerate([("tüm hücreler", df), ("yalnızca ALL", df[df["etiket"] == "ALL"])]):
        for c, (col, label, test_name) in enumerate(panels):
            ax = axes[r, c]
            values = np.log10(part[col].clip(lower=1e-3))
            bins = np.linspace(values.min(), values.max(), 50)
            for kind in ("yayma", "yakin_cekim"):
                v = values[part["meta_tip"] == kind]
                ax.hist(v, bins=bins, density=True, alpha=0.55, color=TYPE_COLORS[kind],
                        label=f"{TYPE_NAMES[kind]} ({len(v)})")
            auc = shortcut[(shortcut["olcu"] == test_name) & (shortcut["kapsam"] == scope)]["roc_auc"].iloc[0]
            ax.set_title(f"{label} — {scope}\ngörüntü tipi tahmini ROC AUC = {auc:.2f}", fontsize=9, color=TEXT_PRIMARY, loc="left")
            ax.set_xlabel(f"log10({'piksel²' if c == 0 else 'alan / d²'})", fontsize=8, color=TEXT_SECONDARY)
            ax.set_yticks([])
            _style_axes(ax)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Ölçek normalizasyonu: ham piksel alanı görüntü tipini (büyütmeyi) ele verir; eritrosite oranlanmış alan vermez "
                 "(AUC 0.5 = bilgi yok)", fontsize=9.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "features", "olcek_normalizasyonu")


# ---------------------------------------------------------------------------
# Önem, dağılım, korelasyon
# ---------------------------------------------------------------------------

def feature_importance(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    ANOVA F-skoru ve karşılıklı bilgiyi (sınıf etiketiyle) hesaplar — yalnızca betimleyici.

    Eksik değerler (ör. halkası olmayan hücre) sütun medyanıyla doldurulur.

    Parametreler:
        df: Öznitelik tablosu.
        columns: Öznitelik sütunları.

    Döndürür:
        F-skoruna göre sıralı tablo (anova_f, anova_p, karsilikli_bilgi, sınıf medyanları).
    """
    x = df[columns].fillna(df[columns].median())
    f, p = f_classif(x, df["etiket"])
    mi = mutual_info_classif(x, df["etiket"], random_state=RANDOM_STATE)
    out = pd.DataFrame({"oznitelik": columns, "anova_f": f, "anova_p": p, "karsilikli_bilgi": mi})
    medians = df.groupby("etiket")[columns].median().T.add_prefix("medyan_")
    out = out.join(medians, on="oznitelik").sort_values("anova_f", ascending=False).reset_index(drop=True)
    out["f_sirasi"] = np.arange(1, len(out) + 1)
    out["mi_sirasi"] = out["karsilikli_bilgi"].rank(ascending=False).astype(int)
    return out


def plot_distributions(df: pd.DataFrame, importance: pd.DataFrame, n: int = 12) -> None:
    """
    En ayırt edici n özniteliğin sınıf bazında kutu grafiklerini çizer.

    Parametreler:
        df: Öznitelik tablosu.
        importance: feature_importance çıktısı.
        n: Gösterilecek öznitelik sayısı.
    """
    top = importance["oznitelik"].head(n).tolist()
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), facecolor=SURFACE)
    for ax, col in zip(axes.ravel(), top):
        data = [df.loc[df["etiket"] == c, col].dropna() for c in CLASS_NAMES]
        box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                         medianprops=dict(color=TEXT_PRIMARY, lw=1.2))
        for patch, c in zip(box["boxes"], CLASS_NAMES):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3], CLASS_NAMES, fontsize=8)
        f = importance.loc[importance["oznitelik"] == col, "anova_f"].iloc[0]
        ax.set_title(f"{col}\nF = {f:,.0f}", fontsize=8.5, color=TEXT_PRIMARY)
        _style_axes(ax)
        ax.tick_params(labelsize=7)
    fig.suptitle("En ayırt edici 12 öznitelik (ANOVA F-skoru; betimleyici) — sınıf bazında kutu grafikleri "
                 "(aykırı değerler gösterilmiyor)", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "features", "oznitelik_dagilimlari")


def correlation_analysis(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Öznitelikler arası Pearson korelasyon ısı haritasını çizer ve |r| > HIGH_CORRELATION çiftleri listeler.

    Parametreler:
        df: Öznitelik tablosu.
        columns: Öznitelik sütunları.

    Döndürür:
        Yüksek korelasyonlu çiftler tablosu.
    """
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(13, 11.5), facecolor=SURFACE)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), [c.split("_", 1)[1] for c in columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(columns)), [c.split("_", 1)[1] for c in columns], fontsize=6)
    # Grup sınırları (şekil | renk | doku)
    for i, prefix in enumerate(columns[1:], start=1):
        if prefix.split("_")[0] != columns[i - 1].split("_")[0]:
            ax.axhline(i - 0.5, color="black", lw=1)
            ax.axvline(i - 0.5, color="black", lw=1)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Pearson r")
    ax.set_title(f"Öznitelik korelasyonu ({len(columns)} öznitelik; çizgiler: şekil | renk | doku)",
                 fontsize=11, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    save_figure(fig, "features", "korelasyon")
    upper = corr.where(np.triu(np.ones(corr.shape, bool), k=1)).stack()
    high = upper[upper.abs() > HIGH_CORRELATION].rename("r").reset_index()
    high.columns = ["oznitelik_1", "oznitelik_2", "r"]
    return high.reindex(high["r"].abs().sort_values(ascending=False).index).round(4)


def edge_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kenara değen hücrelerin şekil özniteliklerinin kenara değmeyenlerden farkını ölçer.

    Etki büyüklüğü: Cohen's d = (ortalama_kenar − ortalama_iç) / birleşik std (|d| ≥ 0.8 büyük).
    Sınıf bazında hesaplanır (kenara değme oranı sınıfa göre değiştiği için sınıflar karışmasın).

    Parametreler:
        df: Öznitelik tablosu.

    Döndürür:
        (sınıf, öznitelik) başına medyan oranı ve Cohen's d tablosu.
    """
    rows = []
    for c in CLASS_NAMES:
        part = df[df["etiket"] == c]
        edge, inner = part[part["meta_kenara_degiyor"]], part[~part["meta_kenara_degiyor"]]
        for col in SHAPE_EDGE_FEATURES:
            a, b = edge[col].dropna(), inner[col].dropna()
            pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1))
            rows.append({"etiket": c, "oznitelik": col, "kenar_n": len(a), "ic_n": len(b),
                         "medyan_kenar": a.median(), "medyan_ic": b.median(),
                         "medyan_orani": a.median() / b.median() if b.median() else np.nan,
                         "cohen_d": (a.mean() - b.mean()) / pooled if pooled else np.nan})
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------------------
# GLCM örnekleri
# ---------------------------------------------------------------------------

def plot_glcm_examples(df: pd.DataFrame) -> None:
    """
    Her sınıftan bir çekirdeğin nicelenmiş görüntüsünü, GLCM'sini (d=1, 4 açı ortalaması) ve özellik değerlerini çizer.

    Örnek: kenara değmeyen ve referans ölçekte alanı sınıf medyanına en yakın çekirdek.

    Parametreler:
        df: Öznitelik tablosu.
    """
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25, 1]})
    for r, c in enumerate(CLASS_NAMES):
        part = df[(df["etiket"] == c) & ~df["meta_kenara_degiyor"] & (df["meta_tip"] == "yayma")]
        row = part.iloc[(part["sekil_alan"] - part["sekil_alan"].median()).abs().argsort().iloc[0]]
        crop = read_png(row["meta_kirpim_yolu"], color=True)
        coded = read_png(row["meta_maske_yolu"], color=False)
        image, coded_ref, _ = to_reference_scale(crop, coded, row["meta_eritrosit_capi"])
        nucleus = largest_component(coded_ref == MASK_NUCLEUS)
        gray = quantize_gray(image)
        shown = np.where(nucleus, gray, np.nan)
        ax = axes[r, 0]
        ax.imshow(shown, cmap="gray", vmin=0, vmax=GLCM_LEVELS - 1, interpolation="nearest")
        ax.set_title(f"{c}: çekirdek ({GLCM_LEVELS} gri seviye, maske dışı boş)", fontsize=8.5, color=TEXT_PRIMARY)
        ax.axis("off")
        glcm = masked_glcm(gray, nucleus)
        mean_glcm = glcm[:, :, 0, :].sum(axis=2)  # d=1, 4 açının toplamı
        ax = axes[r, 1]
        im = ax.imshow(np.log1p(mean_glcm), cmap="magma", interpolation="nearest")
        ax.set_title("GLCM (d = 1, 4 açı toplamı; log(1 + sayım))", fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_xlabel("komşu pikselin gri seviyesi j", fontsize=7.5, color=TEXT_SECONDARY)
        ax.set_ylabel("pikselin gri seviyesi i", fontsize=7.5, color=TEXT_SECONDARY)
        ax.tick_params(labelsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
        props = glcm_properties(glcm)
        ax = axes[r, 2]
        ax.axis("off")
        lines = [f"{p:<14} {np.nanmean(v[0]):8.3f}" for p, v in props.items()]
        lines.append(f"{'entropi':<14} {row['doku_entropi']:8.3f}")
        ax.text(0, 0.95, "d = 1, açı ortalaması\n\n" + "\n".join(lines), family="monospace", fontsize=9,
                va="top", color=TEXT_PRIMARY)
        ax.text(0, 0.12, f"çekirdek: {int(nucleus.sum())} px (referans ölçek)\nalan/d² = {row['sekil_alan']:.2f}",
                fontsize=8, color=TEXT_SECONDARY, va="top")
    fig.suptitle(f"GLCM örnekleri — her sınıftan bir yayma çekirdeği (eritrosit çapı {FEATURE_REF_RBC_DIAMETER} px ölçeğinde)",
                 fontsize=10.5, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, "features", "glcm_ornekleri")


# ---------------------------------------------------------------------------
# Uçtan uca
# ---------------------------------------------------------------------------

def main() -> None:
    """Analizleri çalıştırır; tabloları ve figürleri outputs/features/ altına kaydeder."""
    pd.set_option("display.width", 220)
    out_dir = stage_dir("features")
    df = pd.read_csv(FEATURES_CSVS[SEGMENTATION_METHOD])
    columns = feature_columns(df)
    groups = {p: sum(c.startswith(p) for c in columns) for p in ("sekil_", "renk_", "doku_")}
    print(f"{SEGMENTATION_METHOD}: {len(df)} hücre, {len(columns)} öznitelik {groups}")
    counts = {m: len(pd.read_csv(FEATURES_CSVS[m], usecols=["hucre_id"])) for m in SEGMENTATION_METHODS}
    print("Hücre sayıları:", counts)
    nan = df[columns].isna().sum()
    print("Eksik değerli öznitelikler:", nan[nan > 0].to_dict())

    shortcut = shortcut_test(df)
    shortcut.to_csv(out_dir / "kisayol_testi.csv", index=False, encoding="utf-8")
    print("\n=== Kısayol testi (görüntü tipi tahmini) ===\n" + shortcut.to_string(index=False))
    plot_scale_normalization(df, shortcut)

    importance = feature_importance(df, columns)
    importance.to_csv(out_dir / "oznitelik_onemi.csv", index=False, encoding="utf-8")
    print("\n=== En ayırt edici 15 öznitelik (betimleyici) ===\n" + importance.head(15).round(3).to_string(index=False))
    plot_distributions(df, importance)
    dependence = top_feature_type_dependence(df, importance)
    dependence.to_csv(out_dir / "ust_oznitelik_tip_bagimliligi.csv", index=False, encoding="utf-8")
    print("\n=== En ayırt edici 10 özniteliğin görüntü tipi bağımlılığı (yalnızca ALL) ===\n"
          + dependence.to_string(index=False))

    high = correlation_analysis(df, columns)
    high.to_csv(out_dir / "yuksek_korelasyon.csv", index=False, encoding="utf-8")
    print(f"\n=== |r| > {HIGH_CORRELATION}: {len(high)} çift ===\n" + high.to_string(index=False))

    edge = edge_effect(df)
    edge.to_csv(out_dir / "kenar_etkisi.csv", index=False, encoding="utf-8")
    print("\n=== Kenara değen hücreler (Cohen's d) ===\n" +
          edge.pivot(index="oznitelik", columns="etiket", values="cohen_d").round(2).to_string())
    print("Kenara değen oranı:", df.groupby("etiket")["meta_kenara_degiyor"].mean().round(3).to_dict())
    plot_glcm_examples(df)
    print("\nN/C durumu:\n" + df.groupby(["etiket", "meta_tip"])["meta_nc_durum"].value_counts().unstack(fill_value=0).to_string())
    print("\nFigürler:", ", ".join(sorted(p.name for p in out_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
