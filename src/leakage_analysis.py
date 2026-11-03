"""
Adım 5 düzeltmesi - Sızıntı analizi ve kaynak kısayolu testi.

Bölümler (her biri tablosunu outputs/ altına yazar; ad verilerek tek tek çalıştırılabilir):
- eski:       Eski (v1) kayıtlı modelin eski kilitli testteki sonucu, tüm test görüntülerinde (sızıntılı) ve yalnızca
              sızıntısız görüntülerde (kaynak grubu geliştirme setinde görülmeyen). Model yeniden eğitilmez.
- kaynak_auc: (a) Öznitelikler görüntünün ALL-IDB kaynaklı olup olmadığını ele veriyor mu? v2 geliştirme setinde,
              sınıf içinde (ALL, Healthy; AML'de ALL-IDB yok), kaynak_grup_id ile grup bazlı 5 katlı CV, ROC AUC.
- kaynak_basari: (b) ALL ve Healthy'de, ALL-IDB kaynaklı ve kaynağı bilinmeyen görüntülerde ayrı başarı:
              v2 geliştirme CV'si (katlar dışı) ve v2 kilitli testi (kayıtlı v2 modeli).
- karsilastirma: Eski / temiz / yeni sonuçların yan yana tablosu ve figürü (önceki bölümlerin CSV'lerinden).

Çalıştırma (proje kökünden; argümansız hepsi sırayla):
    python -m src.leakage_analysis [eski] [kaynak_auc] [kaynak_basari] [karsilastirma]
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.classification_analysis import SERIES_COLORS, bar_panel
from src.classifier import (
    NO_CELL,
    aggregate_images,
    label_metrics,
    load_cells,
    load_model,
    load_split,
    out_of_fold_proba,
    proba_to_labels,
    reorder_proba,
)
from src.config import (
    CLASS_NAMES,
    CLASSIFIER_MODEL,
    CLASSIFIER_PARAMS,
    CV_FOLDS,
    DATASET_INDEX_CSV,
    MODEL_FILES,
    N_JOBS,
    RANDOM_STATE,
    SEGMENTATION_METHOD,
)
from src.dataset import SURFACE
from src.source_groups import SOURCE_ALLIDB, SOURCE_UNKNOWN, leaky_images
from src.utils import feature_columns, save_figure, stage_dir

EVAL_DIR = "evaluation"
CLS_DIR = "classification"
OLD_CSV = "sizinti_eski_model_temiz_test.csv"
AUC_CSV = "kaynak_kisayolu_auc.csv"
SOURCE_PERF_CSV = "kaynaga_gore_basari_v2.csv"
COMPARE_CSV = "sizinti_karsilastirma.csv"
METRIC_KEYS = ["n", "dogruluk", "dengeli_dogruluk", "makro_f1"] + [f"duyarlilik_{c}" for c in CLASS_NAMES] \
    + ["anormal_duyarlilik", "healthy_ozgulluk"]


def flat(m: dict) -> dict:
    """
    label_metrics çıktısından tabloya yazılacak ölçüleri seçer (karışıklık matrisi hariç).

    Parametreler:
        m: label_metrics çıktısı.

    Döndürür:
        Düz metrik sözlüğü.
    """
    return {k: m[k] for k in METRIC_KEYS}


def level_metrics(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame) -> dict[str, dict]:
    """
    Classifier.main ile aynı üç düzeyde metrik: kenara değmeyen hücreler, hücre bulunan görüntüler, tüm görüntüler.

    Parametreler:
        cells: Hücre tablosu (proba ile aynı sırada).
        proba: Hücre olasılıkları (CLASS_NAMES sırasında).
        images: Değerlendirilen görüntüler (dosya_yolu, etiket, tip).

    Döndürür:
        {"hucre", "goruntu_kapsanan", "goruntu_tum"} → label_metrics çıktısı; ayrıca "goruntu_tablosu".
    """
    inner = ~cells["meta_kenara_degiyor"].to_numpy(bool)
    pred = proba_to_labels(proba)
    img = aggregate_images(cells, proba, images)
    covered = img[img["tahmin"] != NO_CELL]
    return {
        "hucre": label_metrics(cells["etiket"].to_numpy()[inner], pred[inner]),
        "goruntu_kapsanan": label_metrics(covered["etiket"].to_numpy(), covered["tahmin"].to_numpy()),
        "goruntu_tum": label_metrics(img["etiket"].to_numpy(), img["tahmin"].to_numpy()),
        "goruntu_tablosu": img,
    }


# ---------------------------------------------------------------------------
# 1. Eski modelin temiz değerlendirmesi
# ---------------------------------------------------------------------------

def old_model_clean() -> pd.DataFrame:
    """
    Eski kayıtlı modeli (v1) eski kilitli testin tamamında ve yalnızca sızıntısız görüntülerinde değerlendirir.

    Sızıntılı test görüntüsü: kaynak grubu (kaynak_grup_id) eski ayrımda geliştirme setine de düşmüş görüntü,
    yani aynı ALL-IDB1 yaymasının bir türevi eski modelin eğitiminde vardı. Model yeniden eğitilmez; tahminler
    kayıtlı modelden alınır.

    Döndürür:
        Alt küme (tum / temiz / sizintili) × düzey (hucre, goruntu_kapsanan, goruntu_tum) × sınıf metrik tablosu.
    """
    model, meta = load_model(MODEL_FILES["v1"])
    split = load_split("v1")
    index = pd.read_csv(DATASET_INDEX_CSV)
    split = split.merge(index[["dosya_yolu", "kaynak_grup_id", "kaynak"]], on="dosya_yolu", how="left")
    split["sizintili"] = leaky_images(split, split).to_numpy()
    cells = load_cells(meta["segmentasyon"], version="v1")
    test = cells[cells["bolum"] == "test"].reset_index(drop=True)
    test = test.join(split.set_index("dosya_yolu")[["sizintili", "kaynak"]], on="meta_kaynak_dosya")
    with threadpool_limits(limits=N_JOBS):
        proba = reorder_proba(model, model.predict_proba(test[meta["oznitelikler"]].to_numpy(np.float64)))
    test_images = split[split["bolum"] == "test"]

    rows = []
    for subset, keep in [("tum (sizintili)", None), ("temiz (sizintisiz)", False), ("yalniz sizintili", True)]:
        cmask = np.ones(len(test), bool) if keep is None else (test["sizintili"] == keep).to_numpy()
        imgs = test_images if keep is None else test_images[test_images["sizintili"] == keep]
        sub = test[cmask].reset_index(drop=True)
        lv = level_metrics(sub, proba[cmask], imgs)
        for level in ("hucre", "goruntu_kapsanan", "goruntu_tum"):
            rows.append({"alt_kume": subset, "duzey": level, "goruntu_sayisi": len(imgs),
                         **{f"goruntu_{c}": int((imgs["etiket"] == c).sum()) for c in CLASS_NAMES},
                         **flat(lv[level])})
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir(EVAL_DIR) / OLD_CSV, index=False, encoding="utf-8")
    cols = ["alt_kume", "duzey", "n", "dogruluk", "makro_f1"] + [f"duyarlilik_{c}" for c in CLASS_NAMES]
    print(table[cols].round(3).to_string(index=False))
    leak = test_images.groupby("etiket")["sizintili"].agg(["sum", "size"])
    print("Sızıntılı test görüntüsü / test görüntüsü:\n" + leak.to_string())
    return table


# ---------------------------------------------------------------------------
# 4a. Kaynak tahmini (öznitelikler kaynağı ele veriyor mu?)
# ---------------------------------------------------------------------------

def source_models() -> dict:
    """
    Kaynak tahmininde kullanılan iki ikili sınıflandırıcıyı döndürür.

    - Lojistik regresyon: doğrusal; düşük AUC "basit bir renk kayması yok" demektir.
    - HGB: son modelle aynı aile ve parametreler; sınıflandırıcının kullanabileceği doğrusal olmayan ipuçlarını da yakalar.

    Döndürür:
        {ad: model oluşturucu} sözlüğü.
    """
    return {
        "lojistik": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(max_iter=2000, class_weight="balanced")),
        "hgb": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                                      **CLASSIFIER_PARAMS),
    }


def source_auc(cells: pd.DataFrame, features: list[str]) -> dict:
    """
    Bir hücre alt kümesinde ALL-IDB kaynağının öznitelikten tahmin edilebilirliğini grup bazlı CV ile ölçer.

    Katlar kaynak_grup_id ile kurulur (aynı yaymanın türevleri hem eğitim hem doğrulamada olamaz); katman = kaynak.
    Hücre düzeyi AUC'nin yanında görüntü düzeyi AUC de verilir (görüntünün hücre olasılıklarının ortalaması).

    Parametreler:
        cells: Hücre alt kümesi (kaynak, kaynak_grup_id, meta_kaynak_dosya sütunlarıyla).
        features: Öznitelik sütunları.

    Döndürür:
        {model}_hucre_auc, {model}_goruntu_auc (kat ortalaması) ve {model}_hucre_auc_std.
    """
    X = cells[features].to_numpy(np.float64)
    y = (cells["kaynak"] == SOURCE_ALLIDB).to_numpy(int)
    groups = cells["kaynak_grup_id"].to_numpy()
    files = cells["meta_kaynak_dosya"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    with threadpool_limits(limits=N_JOBS):
        for name, factory in source_models().items():
            cell_auc, img_auc = [], []
            for train, val in cv.split(X, y, groups):
                if len(set(y[val])) < 2:
                    continue
                p = factory().fit(X[train], y[train]).predict_proba(X[val])[:, 1]
                cell_auc.append(roc_auc_score(y[val], p))
                img = pd.DataFrame({"f": files[val], "p": p, "y": y[val]}).groupby("f").mean()
                img_auc.append(roc_auc_score(img["y"].round().astype(int), img["p"]))
            out[f"{name}_hucre_auc"] = float(np.mean(cell_auc))
            out[f"{name}_hucre_auc_std"] = float(np.std(cell_auc))
            out[f"{name}_goruntu_auc"] = float(np.mean(img_auc))
            out["kat_sayisi"] = len(cell_auc)
    return out


def source_shortcut_auc() -> pd.DataFrame:
    """
    v2 geliştirme setinde sınıf içinde kaynak tahmini (ALL-IDB / bilinmiyor) yapar.

    Sınıf içinde yapılır: AML'de hiç ALL-IDB görüntüsü yok; sınıflar karışık olsaydı kaynak tahmini sınıf tahminine
    dönüşürdü. ALL'da yakın çekimler kaynakla ilişkili olduğundan (ALL-IDB2 kırpımları / kaynağı bilinmeyen yakın
    çekimler) yayma ve yakın çekim ayrıca raporlanır. Test seti kullanılmaz.

    Döndürür:
        Alt küme başına AUC tablosu.
    """
    cells = load_cells(SEGMENTATION_METHOD, version="v2")
    cells = cells[cells["bolum"] == "gelistirme"].reset_index(drop=True)
    features = feature_columns(cells)
    subsets = {
        "ALL (tum)": cells["etiket"] == "ALL",
        "ALL (yalniz yayma)": (cells["etiket"] == "ALL") & (cells["meta_tip"] == "yayma"),
        "ALL (yalniz yakin cekim)": (cells["etiket"] == "ALL") & (cells["meta_tip"] == "yakin_cekim"),
        "Healthy (tum)": cells["etiket"] == "Healthy",
    }
    rows = []
    for title, mask in subsets.items():
        start = time.time()
        sub = cells[mask].reset_index(drop=True)
        r = source_auc(sub, features)
        rows.append({"alt_kume": title, "hucre": len(sub), "allidb_hucre": int((sub["kaynak"] == SOURCE_ALLIDB).sum()),
                     "goruntu": sub["meta_kaynak_dosya"].nunique(),
                     "allidb_goruntu": sub.loc[sub["kaynak"] == SOURCE_ALLIDB, "meta_kaynak_dosya"].nunique(),
                     "kaynak_grubu": sub["kaynak_grup_id"].nunique(), **r})
        pd.DataFrame(rows).to_csv(stage_dir(CLS_DIR) / AUC_CSV, index=False, encoding="utf-8")
        print(f"  [{len(rows)}/{len(subsets)}] {title:<26} lojistik AUC {r['lojistik_hucre_auc']:.3f} "
              f"(görüntü {r['lojistik_goruntu_auc']:.3f}), HGB AUC {r['hgb_hucre_auc']:.3f} "
              f"(görüntü {r['hgb_goruntu_auc']:.3f}) ({time.time() - start:.0f} sn)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4b. Kaynağa göre sınıflandırma başarısı
# ---------------------------------------------------------------------------

def per_source_rows(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame, setting: str) -> list[dict]:
    """
    ALL ve Healthy'de kaynağa göre (ALL-IDB / bilinmiyor) hücre ve görüntü düzeyinde başarıyı hesaplar.

    Tek sınıflı alt kümede (ör. ALL, ALL-IDB) duyarlılık = doğruluk; ALL + Healthy satırında dengeli doğruluk
    (iki sınıfın duyarlılık ortalaması). Makro F1 kullanılmaz: alt kümede olmayan ama tahmin edilen sınıf (AML)
    F1 = 0 ile girip skoru anlamsız biçimde düşürür. Hatanın yönü için tahmin dağılımı da yazılır.

    Parametreler:
        cells: Hücreler (kaynak sütunuyla), proba ile aynı sırada.
        proba: Hücre olasılıkları.
        images: Görüntüler (kaynak, kaynak_grup_id sütunlarıyla).
        setting: Tablo etiketi (ör. "gelistirme CV (katlar disi)").

    Döndürür:
        Satır listesi.
    """
    inner = ~cells["meta_kenara_degiyor"].to_numpy(bool)
    pred = proba_to_labels(proba)
    img = aggregate_images(cells, proba, images).merge(images[["dosya_yolu", "kaynak", "kaynak_grup_id"]],
                                                       on="dosya_yolu")
    rows = []
    for classes in (["ALL"], ["Healthy"], ["ALL", "Healthy"]):
        for source in (SOURCE_ALLIDB, SOURCE_UNKNOWN):
            csel = inner & cells["etiket"].isin(classes).to_numpy() & (cells["kaynak"] == source).to_numpy()
            isel = img["etiket"].isin(classes) & (img["kaynak"] == source)
            cov = isel & (img["tahmin"] != NO_CELL)
            cm = label_metrics(cells["etiket"].to_numpy()[csel], pred[csel])
            im = label_metrics(img.loc[cov, "etiket"].to_numpy(), img.loc[cov, "tahmin"].to_numpy())
            rows.append({
                "ayar": setting, "sinif": "+".join(classes), "kaynak": source,
                "goruntu": int(isel.sum()), "kaynak_grubu": int(img.loc[isel, "kaynak_grup_id"].nunique()),
                "hucre": int(csel.sum()),
                "hucre_dogruluk": cm["dogruluk"], "hucre_dengeli_dogruluk": cm["dengeli_dogruluk"],
                "goruntu_dogruluk_kapsanan": im["dogruluk"], "goruntu_dengeli_dogruluk_kapsanan": im["dengeli_dogruluk"],
                "goruntu_dogruluk_tum": float((img.loc[isel, "tahmin"] == img.loc[isel, "etiket"]).mean()),
                "hucre_yok": int((isel & (img["tahmin"] == NO_CELL)).sum()),
                "goruntu_tahmin_dagilimi": img.loc[cov, "tahmin"].value_counts().to_dict(),
            })
    return rows


def source_performance() -> pd.DataFrame:
    """
    v2 geliştirme CV'si (katlar dışı) ve v2 kilitli testinde kaynağa göre başarıyı raporlar.

    Geliştirme CV'si daha çok kaynak grubu içerir (daha kararlı); test, kayıtlı v2 modelinin tek değerlendirmesidir.

    Döndürür:
        per_source_rows satırlarından oluşan tablo.
    """
    split = load_split("v2")
    cells = load_cells(SEGMENTATION_METHOD, version="v2")
    features = feature_columns(cells)
    dev = cells[cells["bolum"] == "gelistirme"].reset_index(drop=True)
    print("  geliştirme CV (katlar dışı) hesaplanıyor...", flush=True)
    oof = out_of_fold_proba(dev, features, CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
    rows = per_source_rows(dev, oof, split[split["bolum"] == "gelistirme"], "gelistirme CV (katlar disi)")

    model, meta = load_model(MODEL_FILES["v2"])
    test = cells[cells["bolum"] == "test"].reset_index(drop=True)
    with threadpool_limits(limits=N_JOBS):
        proba = reorder_proba(model, model.predict_proba(test[meta["oznitelikler"]].to_numpy(np.float64)))
    rows += per_source_rows(test, proba, split[split["bolum"] == "test"], "kilitli test (v2)")
    table = pd.DataFrame(rows)
    table.to_csv(stage_dir(EVAL_DIR) / SOURCE_PERF_CSV, index=False, encoding="utf-8")
    cols = ["ayar", "sinif", "kaynak", "goruntu", "kaynak_grubu", "hucre", "hucre_dogruluk", "hucre_dengeli_dogruluk",
            "goruntu_dogruluk_kapsanan", "goruntu_dengeli_dogruluk_kapsanan", "goruntu_dogruluk_tum", "hucre_yok", "goruntu_tahmin_dagilimi"]
    print(table[cols].round(3).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Karşılaştırma tablosu ve figürü
# ---------------------------------------------------------------------------

def comparison() -> pd.DataFrame:
    """
    Eski (v1) testin tamamı, sızıntısız alt kümesi ve yeni (v2) test sonuçlarını; ayrıca v1 ve v2 geliştirme
    CV'lerini tek tabloda ve figürde yan yana koyar.

    Girdi CSV'leri: sizinti_eski_model_temiz_test.csv, test_metrikleri_v2.csv,
    model_karsilastirma.csv ve model_karsilastirma_v2.csv (seçilen yapılandırma satırları).

    Döndürür:
        Karşılaştırma tablosu.
    """
    ev, cl = stage_dir(EVAL_DIR), stage_dir(CLS_DIR)
    old = pd.read_csv(ev / OLD_CSV)
    new = pd.read_csv(ev / "test_metrikleri_v2.csv")
    level_names = {"hucre (kenara degmeyen)": "hucre", "goruntu (hucre bulunan)": "goruntu_kapsanan",
                   "goruntu (tum; hucre yok = hata)": "goruntu_tum"}
    new = new[new["duzey"].isin(level_names)].assign(duzey=lambda d: d["duzey"].map(level_names),
                                                    alt_kume="yeni test v2 (sizintisiz)")
    rows = pd.concat([old[old["alt_kume"] != "yalniz sizintili"], new], ignore_index=True)
    keep = ["alt_kume", "duzey", "n", "dogruluk", "dengeli_dogruluk", "makro_f1"] \
        + [f"duyarlilik_{c}" for c in CLASS_NAMES] + ["anormal_duyarlilik", "healthy_ozgulluk"]
    table = rows[keep].copy()
    params = str(CLASSIFIER_PARAMS)
    for version, suffix in (("v1", ""), ("v2", "_v2")):
        cv = pd.read_csv(cl / f"model_karsilastirma{suffix}.csv")
        r = cv[(cv["model"] == CLASSIFIER_MODEL) & (cv["parametreler"] == params)].iloc[0]
        for level, col in (("hucre", "hucre_makro_f1"), ("goruntu_tum", "goruntu_makro_f1_tum"),
                           ("goruntu_kapsanan", "goruntu_makro_f1_kapsanan")):
            table.loc[len(table)] = {"alt_kume": f"gelistirme CV {version}", "duzey": level, "makro_f1": r[col]}
    table.to_csv(ev / COMPARE_CSV, index=False, encoding="utf-8")

    order = ["tum (sizintili)", "temiz (sizintisiz)", "yeni test v2 (sizintisiz)", "gelistirme CV v1", "gelistirme CV v2"]
    labels = ["Eski test\n(tümü, sızıntılı)", "Eski test\n(yalnız sızıntısız)", "Yeni test v2",
              "Geliştirme CV\nv1", "Geliştirme CV\nv2"]
    f1 = table.pivot_table(index="alt_kume", columns="duzey", values="makro_f1").reindex(order)
    fig, ax = plt.subplots(figsize=(9.0, 3.8), facecolor=SURFACE)
    bar_panel(ax, labels, {"Hücre makro F1": f1["hucre"].tolist(), "Görüntü makro F1": f1["goruntu_tum"].tolist()},
              SERIES_COLORS, "Sızıntı düzeltmesi: eski ve yeni sonuçlar (HGB; görüntü = tüm görüntüler)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save_figure(fig, EVAL_DIR, "sizinti_karsilastirma")
    print(table.round(3).to_string(index=False))
    return table


PARTS = {"eski": old_model_clean, "kaynak_auc": source_shortcut_auc, "kaynak_basari": source_performance,
         "karsilastirma": comparison}


def main() -> None:
    """Komut satırında adı verilen bölümleri (argümansız: hepsini) sırayla çalıştırır."""
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    for name in sys.argv[1:] or list(PARTS):
        print(f"\n=== {name} ===", flush=True)
        start = time.time()
        PARTS[name]()
        print(f"({time.time() - start:.0f} sn)", flush=True)


if __name__ == "__main__":
    main()
