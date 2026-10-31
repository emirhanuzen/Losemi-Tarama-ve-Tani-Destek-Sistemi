"""
Sınıflandırma deneyleri (yalnızca geliştirme seti, sabit 5 katlı grup CV; test seti KULLANILMAZ).

Deneyler:
- modeller:     5 klasik model × küçük hiperparametre ızgarası (K-means öznitelikleri)
- segmentasyon: seçilen modelle Otsu / K-means / Watershed öznitelik tabloları
- ablasyon:     öznitelik grupları (şekil, renk, doku ve birleşimleri)
- kisayol:      görüntü tipine göre başarı, yalnız-yayma modeli, çok öznitelikli tip tahmini
- kenar:        kenara değen hücreler eğitime dahil / hariç

Her deney tablosunu ve figürünü outputs/classification/ altına yazar. Tek bir deney çalıştırmak için
adını argüman olarak verin; argümansız hepsi sırayla çalışır.

Çalıştırma (proje kökünden):
    python -m src.classification_analysis [modeller] [segmentasyon] [ablasyon] [kisayol] [kenar]
"""

import ast
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.classifier import (
    MODEL_NAMES,
    build_model,
    load_cells,
    load_split,
    out_of_fold_proba,
    proba_to_labels,
    label_metrics,
    select_features,
    summarize,
)
from src.config import (
    CLASS_NAMES,
    CLASSIFIER_FEATURE_SET,
    CLASSIFIER_MODEL,
    CLASSIFIER_PARAMS,
    CLASSIFIER_TRAIN_EDGE_CELLS,
    CV_FOLDS,
    IMAGE_EXCLUDE_EDGE_CELLS,
    N_JOBS,
    RANDOM_STATE,
    SEGMENTATION_METHOD,
    SEGMENTATION_METHODS,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes
from src.features import feature_columns
from src.utils import save_figure, stage_dir

# Hiperparametre ızgaraları bilinçli olarak küçük: 13 yapılandırma × 5 kat. Geniş ızgara CV skorunu
# iyimserleştirir (çok sayıda denemeden en iyisini seçmek, doğrulama katına uyum sağlamaktır).
PARAM_GRID = {
    # C: düzenlileştirmenin tersi; 0.1-10 aralığı, standartlaştırılmış 58 öznitelik için tipik
    "lojistik": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    # k: 5 (yerel, gürültüye duyarlı) – 31 (düzgün); uzaklık ağırlığı yakın komşuyu öne çıkarır
    "knn": [{"n_neighbors": k, "weights": "distance"} for k in (5, 15, 31)],
    # C: 1-30; gamma="scale" = 1 / (öznitelik sayısı × varyans), standartlaştırılmış veride ≈ 1/58
    "svm": [{"C": c, "gamma": "scale"} for c in (1.0, 10.0, 30.0)],
    # 300 ağaç: OOB hatası ~200 ağaçtan sonra düzleşir; yaprak ≥ 5 hücre aşırı uyumu sınırlar
    "rf": [{"n_estimators": 300, "min_samples_leaf": m} for m in (1, 5)],
    # 300 iterasyon, erken durdurma kapalı (iç rastgele doğrulama bölmesi grup ayrımı gözetmez)
    "hgb": [{"learning_rate": lr, "max_iter": 300, "early_stopping": False} for lr in (0.05, 0.1)],
}
ABLATION_SETS = {
    "tumu": "Tümü (58)",
    "sekil": "Yalnız şekil",
    "renk": "Yalnız renk",
    "doku": "Yalnız doku",
    "sekil_doku": "Şekil + doku (renksiz)",
    "sekil_renk": "Şekil + renk",
    "renk_doku": "Renk + doku",
    "std_haric": "Std ve GLCM hariç",
}
METHOD_NAMES = {"kmeans": "K-means", "otsu": "Otsu", "watershed": "Watershed"}
SUMMARY_COLUMNS = ["hucre_makro_f1", "hucre_makro_f1_kat_std", "hucre_dengeli_dogruluk", "hucre_duyarlilik_ALL",
                   "hucre_duyarlilik_AML", "hucre_duyarlilik_Healthy", "goruntu_makro_f1_tum",
                   "goruntu_makro_f1_kapsanan", "goruntu_kapsama"]


def dev_data(method: str = SEGMENTATION_METHOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Geliştirme setinin hücrelerini ve görüntülerini döndürür.

    Parametreler:
        method: Segmentasyon yöntemi.

    Döndürür:
        (hücre tablosu, görüntü tablosu); indeksler sıfırlanmış.
    """
    cells = load_cells(method)
    cells = cells[cells["bolum"] == "gelistirme"].reset_index(drop=True)
    split = load_split()
    images = split[split["bolum"] == "gelistirme"].reset_index(drop=True)
    return cells, images


def fold_f1_std(cells: pd.DataFrame, proba: np.ndarray, exclude_edge: bool) -> float:
    """
    Hücre düzeyi makro F1'in katlar arası standart sapmasını hesaplar (sonucun kararlılığı).

    Parametreler:
        cells: Geliştirme hücreleri (kat sütunu dahil).
        proba: Katlar dışı olasılıklar.
        exclude_edge: Kenara değen hücreleri dışla.

    Döndürür:
        Standart sapma.
    """
    keep = ~cells["meta_kenara_degiyor"].to_numpy(bool) if exclude_edge else np.ones(len(cells), bool)
    pred = proba_to_labels(proba)
    scores = [label_metrics(cells["etiket"].to_numpy()[sel], pred[sel])["makro_f1"]
              for fold in range(CV_FOLDS) if (sel := keep & (cells["kat"].to_numpy() == fold)).any()]
    return float(np.std(scores))


def run_config(cells: pd.DataFrame, images: pd.DataFrame, features: list[str], name: str, params: dict,
               train_edge: bool = CLASSIFIER_TRAIN_EDGE_CELLS,
               eval_exclude_edge: bool = IMAGE_EXCLUDE_EDGE_CELLS) -> tuple[dict, np.ndarray]:
    """
    Bir model yapılandırmasını katlar dışı tahminle değerlendirir.

    Parametreler:
        cells, images: Geliştirme hücreleri ve görüntüleri.
        features: Öznitelik sütunları.
        name, params: Model adı ve hiperparametreleri.
        train_edge: Kenara değen hücreler eğitime girsin mi.
        eval_exclude_edge: Hücre metriklerinde ve görüntü ortalamasında kenar hücreleri dışlansın mı.

    Döndürür:
        (özet metrik sözlüğü, katlar dışı olasılıklar).
    """
    start = time.time()
    mask = None if train_edge else ~cells["meta_kenara_degiyor"]
    proba = out_of_fold_proba(cells, features, name, params, train_mask=mask)
    row = summarize(cells, proba, images, eval_exclude_edge)
    row["hucre_makro_f1_kat_std"] = fold_f1_std(cells, proba, eval_exclude_edge)
    row["sure_sn"] = round(time.time() - start, 1)
    return row, proba


def bar_panel(ax: plt.Axes, labels: list[str], values: dict[str, list[float]], colors: dict[str, str],
              title: str, errors: list[float] | None = None) -> None:
    """
    Gruplanmış dikey çubuk grafiği çizer (her etiket için birden çok seri, değer yazılı).

    Parametreler:
        ax: Eksen.
        labels: X ekseni etiketleri.
        values: {seri adı: değerler}.
        colors: {seri adı: renk}.
        title: Başlık.
        errors: İlk seri için hata çubukları (katlar arası std).
    """
    n = len(values)
    width = 0.8 / n
    x = np.arange(len(labels))
    for i, (series, vals) in enumerate(values.items()):
        pos = x + (i - (n - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, color=colors[series], label=series,
                      yerr=errors if (i == 0 and errors is not None) else None,
                      error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2})
        errs = errors if (i == 0 and errors is not None) else [0] * len(vals)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_SECONDARY)
    _style_axes(ax)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left", pad=20)  # pad: altında lejant satırına yer
    # Lejant eksenin dışında (sağ üst): içeride çerçevesiz lejant aynı renkteki çubuğun üstüne düşüp kayboluyordu
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(values))


SERIES_COLORS = {"Hücre makro F1": "#2a78d6", "Görüntü makro F1": "#eb6834", "Kapsama": "#9a9890"}


# ---------------------------------------------------------------------------
# Deneyler
# ---------------------------------------------------------------------------

def model_comparison() -> pd.DataFrame:
    """
    Beş klasik modeli küçük hiperparametre ızgarasıyla geliştirme CV'sinde karşılaştırır.

    Döndürür:
        Tüm yapılandırmaların metrik tablosu (model başına en iyi yapılandırma "en_iyi" ile işaretli).
    """
    cells, images = dev_data()
    features = select_features(feature_columns(cells), "tumu")
    rows = []
    for name, grid in PARAM_GRID.items():
        for params in grid:
            row, _ = run_config(cells, images, features, name, params)
            rows.append({"model": name, "model_adi": MODEL_NAMES[name], "parametreler": str(params), **row})
            print(f"  {MODEL_NAMES[name]:<22} {str(params):<60} hücre F1 {row['hucre_makro_f1']:.3f} "
                  f"görüntü F1 {row['goruntu_makro_f1_tum']:.3f} ({row['sure_sn']} sn)")
    table = pd.DataFrame(rows)
    best = table.groupby("model")["hucre_makro_f1"].idxmax()
    table["en_iyi"] = table.index.isin(best)
    table.to_csv(stage_dir("classification") / "model_karsilastirma.csv", index=False, encoding="utf-8")

    plot_model_comparison(table)
    return table


def short_params(params: str) -> str:
    """
    Hiperparametre metninden figür etiketi için yalnızca ızgarada değişen değerleri kısaltır.

    Parametreler:
        params: str(dict) biçiminde hiperparametreler.

    Döndürür:
        Ör. "C=10.0" veya "lr=0.1".
    """
    names = {"C": "C", "n_neighbors": "k", "min_samples_leaf": "yaprak", "learning_rate": "lr"}
    return ", ".join(f"{names[k]}={v}" for k, v in ast.literal_eval(params).items() if k in names)


def plot_model_comparison(table: pd.DataFrame) -> None:
    """
    Her modelin en iyi yapılandırmasının hücre ve görüntü düzeyi makro F1'ini çizer.

    Parametreler:
        table: model_comparison çıktısı (veya outputs/classification/model_karsilastirma.csv).
    """
    top = table[table["en_iyi"]].sort_values("hucre_makro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 3.8), facecolor=SURFACE)
    bar_panel(ax, [f"{r.model_adi}\n{short_params(r.parametreler)}" for r in top.itertuples()],
              {"Hücre makro F1": top["hucre_makro_f1"].tolist(), "Görüntü makro F1": top["goruntu_makro_f1_tum"].tolist()},
              SERIES_COLORS, "Model karşılaştırması — geliştirme seti, 5 katlı grup CV (en iyi yapılandırma)",
              errors=top["hucre_makro_f1_kat_std"].tolist())
    fig.tight_layout()
    save_figure(fig, "classification", "model_karsilastirma")


def segmentation_comparison() -> pd.DataFrame:
    """
    Seçilen modelle üç segmentasyon yönteminin öznitelik tablolarını aynı katlarda karşılaştırır.

    Hücre kümeleri yöntemden yönteme farklı olduğundan asıl ölçüt görüntü düzeyi makro F1'dir
    (hücre bulunamayan görüntü hata sayılır → kapsama da ölçüte yansır).

    Döndürür:
        Yöntem başına metrik tablosu.
    """
    rows = []
    for method in SEGMENTATION_METHODS:
        cells, images = dev_data(method)
        features = select_features(feature_columns(cells), CLASSIFIER_FEATURE_SET)
        row, _ = run_config(cells, images, features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
        rows.append({"yontem": method, "hucre_sayisi": len(cells), **row})
        print(f"  {METHOD_NAMES[method]:<10} hücre F1 {row['hucre_makro_f1']:.3f} görüntü F1 (tüm) "
              f"{row['goruntu_makro_f1_tum']:.3f} kapsama {row['goruntu_kapsama']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir("classification") / "segmentasyon_karsilastirma.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    bar_panel(ax, [METHOD_NAMES[m] for m in table["yontem"]],
              {"Hücre makro F1": table["hucre_makro_f1"].tolist(),
               "Görüntü makro F1": table["goruntu_makro_f1_tum"].tolist(),
               "Kapsama": table["goruntu_kapsama"].tolist()},
              SERIES_COLORS, f"Segmentasyon yöntemine göre sınıflandırma ({MODEL_NAMES[CLASSIFIER_MODEL]})",
              errors=table["hucre_makro_f1_kat_std"].tolist())
    fig.tight_layout()
    save_figure(fig, "classification", "segmentasyon_karsilastirma")
    return table


def feature_ablation() -> pd.DataFrame:
    """
    Seçilen modeli farklı öznitelik gruplarıyla eğitip başarı değişimini ölçer.

    Döndürür:
        Öznitelik kümesi başına metrik tablosu.
    """
    cells, images = dev_data()
    columns = feature_columns(cells)
    rows = []
    for key, title in ABLATION_SETS.items():
        features = select_features(columns, key)
        row, _ = run_config(cells, images, features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
        rows.append({"kume": key, "kume_adi": title, "oznitelik_sayisi": len(features), **row})
        print(f"  {title:<24} ({len(features):>2}) hücre F1 {row['hucre_makro_f1']:.3f} "
              f"görüntü F1 {row['goruntu_makro_f1_tum']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir("classification") / "oznitelik_ablasyonu.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.5, 3.8), facecolor=SURFACE)
    bar_panel(ax, [f"{t}\n({n})" for t, n in zip(table["kume_adi"], table["oznitelik_sayisi"])],
              {"Hücre makro F1": table["hucre_makro_f1"].tolist(),
               "Görüntü makro F1": table["goruntu_makro_f1_tum"].tolist()},
              SERIES_COLORS, f"Öznitelik ablasyonu ({MODEL_NAMES[CLASSIFIER_MODEL]}, geliştirme CV)",
              errors=table["hucre_makro_f1_kat_std"].tolist())
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    save_figure(fig, "classification", "oznitelik_ablasyonu")
    return table


def type_from_features_auc(cells: pd.DataFrame, features: list[str]) -> dict:
    """
    Tüm özniteliklerle birlikte görüntü tipinin (yakın çekim) tahmin edilebilirliğini ölçer (yalnızca ALL).

    ALL'da iki tip dengeli (230 / 250 görüntü) olduğundan sınıf farkı tip tahminine karışmaz. Adım 4'teki
    tek öznitelikli testin çok değişkenli hâli: öznitelikler birlikte büyütmeyi ne kadar ele veriyor?

    Parametreler:
        cells: Geliştirme hücreleri.
        features: Öznitelik sütunları.

    Döndürür:
        {"lojistik_auc", "rf_auc"} (5 katlı grup CV ortalaması).
    """
    data = cells[cells["etiket"] == "ALL"]
    X = data[features].to_numpy(np.float64)
    y = (data["meta_tip"] == "yakin_cekim").to_numpy(int)
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "lojistik_auc": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "rf_auc": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                                 n_jobs=N_JOBS, random_state=RANDOM_STATE),
    }
    out = {}
    with threadpool_limits(limits=N_JOBS):
        for key, factory in models.items():
            scores = []
            for train, test in cv.split(X, y, data["grup_id"]):
                model = factory().fit(X[train], y[train])
                scores.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
            out[key] = float(np.mean(scores))
    return out


def shortcut_check() -> pd.DataFrame:
    """
    Görüntü tipi (büyütme) kısayolunu üç yoldan kontrol eder.

    1. Seçilen modelin katlar dışı başarısı görüntü tipine göre ayrı ayrı.
    2. Yalnızca yayma görüntüleriyle eğitilen/değerlendirilen model: yakın çekimler tamamen çıkarılınca
       "yakın çekim → Healthy değil" ipucu ortadan kalkar. Başarı korunuyorsa model tipe dayanmıyor.
    3. Çok öznitelikli tip tahmini (yalnızca ALL): öznitelikler birlikte tipi ne kadar ele veriyor.

    Döndürür:
        Kontrol satırlarından oluşan tablo.
    """
    cells, images = dev_data()
    features = select_features(feature_columns(cells), CLASSIFIER_FEATURE_SET)
    _, proba = run_config(cells, images, features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
    rows = []
    inner = ~cells["meta_kenara_degiyor"].to_numpy(bool)
    pred = proba_to_labels(proba)
    for tip in ["yayma", "yakin_cekim"]:
        sel = inner & (cells["meta_tip"].to_numpy() == tip)
        m = label_metrics(cells["etiket"].to_numpy()[sel], pred[sel])
        rows.append({"kontrol": f"tam model, {tip} hücreleri", "n": m["n"], "makro_f1": m["makro_f1"],
                     "dogruluk": m["dogruluk"], **{f"duyarlilik_{c}": m[f"duyarlilik_{c}"] for c in CLASS_NAMES}})

    smear = cells["meta_tip"].to_numpy() == "yayma"
    full_smear = summarize(cells[smear].reset_index(drop=True), proba[smear],
                           images[images["tip"] == "yayma"])
    smear_cells = cells[smear].reset_index(drop=True)
    row, _ = run_config(smear_cells, images[images["tip"] == "yayma"], features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
    for title, r in [("tam model → yayma (hücre/görüntü)", full_smear), ("yalnız-yayma modeli → yayma", row)]:
        rows.append({"kontrol": title, "n": r["hucre_n"], "makro_f1": r["hucre_makro_f1"],
                     "dogruluk": r["hucre_dogruluk"], "goruntu_makro_f1_tum": r["goruntu_makro_f1_tum"],
                     **{f"duyarlilik_{c}": r[f"hucre_duyarlilik_{c}"] for c in CLASS_NAMES}})

    auc = type_from_features_auc(cells, features)
    rows.append({"kontrol": "ALL içinde tip tahmini (tüm öznitelikler) ROC AUC", "lojistik_auc": auc["lojistik_auc"],
                 "rf_auc": auc["rf_auc"]})
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir("classification") / "kisayol_kontrolu.csv", index=False, encoding="utf-8")
    print(table.round(3).to_string(index=False))
    return table


def edge_experiment() -> pd.DataFrame:
    """
    Kenara değen (kesik) hücrelerin eğitime dahil edilmesinin etkisini ölçer.

    2 eğitim seçeneği (dahil / hariç) × değerlendirme: kenara değmeyen hücreler, kenara değen hücreler
    ve görüntü düzeyi (kenar hücreleri ortalamadan dışlanarak ve dahil edilerek).

    Döndürür:
        Deney tablosu.
    """
    cells, images = dev_data()
    features = select_features(feature_columns(cells), CLASSIFIER_FEATURE_SET)
    edge = cells["meta_kenara_degiyor"].to_numpy(bool)
    y = cells["etiket"].to_numpy()
    rows = []
    for train_edge in (False, True):
        mask = None if train_edge else ~cells["meta_kenara_degiyor"]
        proba = out_of_fold_proba(cells, features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS, train_mask=mask)
        pred = proba_to_labels(proba)
        inner_m = label_metrics(y[~edge], pred[~edge])
        edge_m = label_metrics(y[edge], pred[edge])
        img_excl = summarize(cells, proba, images, exclude_edge=True)
        img_incl = summarize(cells, proba, images, exclude_edge=False)
        rows.append({
            "egitim": "kenar hücreleri dahil" if train_edge else "kenar hücreleri hariç",
            "egitim_hucre": int(len(cells) if train_edge else (~edge).sum()),
            "hucre_f1_kenara_degmeyen": inner_m["makro_f1"],
            "hucre_f1_kenara_degen": edge_m["makro_f1"],
            **{f"kenara_degen_duyarlilik_{c}": edge_m[f"duyarlilik_{c}"] for c in CLASS_NAMES},
            "goruntu_f1_kenar_ortalamadan_haric": img_excl["goruntu_makro_f1_tum"],
            "goruntu_f1_kenar_ortalamaya_dahil": img_incl["goruntu_makro_f1_tum"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir("classification") / "kenar_hucreleri_deneyi.csv", index=False, encoding="utf-8")
    print(table.round(3).T.to_string())

    fig, ax = plt.subplots(figsize=(7.4, 3.6), facecolor=SURFACE)
    labels = ["Hücre F1\n(kenara değmeyen)", "Hücre F1\n(kenara değen)", "Görüntü F1\n(kenar ortalamadan hariç)",
              "Görüntü F1\n(kenar dahil)"]
    keys = ["hucre_f1_kenara_degmeyen", "hucre_f1_kenara_degen", "goruntu_f1_kenar_ortalamadan_haric",
            "goruntu_f1_kenar_ortalamaya_dahil"]
    bar_panel(ax, labels, {r["egitim"].capitalize(): [r[k] for k in keys] for r in rows},
              {"Kenar hücreleri hariç": "#2a78d6", "Kenar hücreleri dahil": "#9a9890"},
              "Kenara değen hücreler: eğitimde hariç / dahil (geliştirme CV)")
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    save_figure(fig, "classification", "kenar_hucreleri_deneyi")
    return table


EXPERIMENTS = {
    "modeller": model_comparison,
    "segmentasyon": segmentation_comparison,
    "ablasyon": feature_ablation,
    "kisayol": shortcut_check,
    "kenar": edge_experiment,
}


def main() -> None:
    """Komut satırında adı verilen deneyleri (argümansız: hepsini) sırayla çalıştırır."""
    pd.set_option("display.width", 220)
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        print(f"\n=== {name} ===")
        start = time.time()
        table = EXPERIMENTS[name]()
        if name in ("modeller", "segmentasyon", "ablasyon"):
            cols = [c for c in table.columns if c in ("model_adi", "parametreler", "yontem", "kume_adi")] + SUMMARY_COLUMNS
            print(table[cols].round(3).to_string(index=False))
        print(f"({time.time() - start:.0f} sn)")


if __name__ == "__main__":
    main()
