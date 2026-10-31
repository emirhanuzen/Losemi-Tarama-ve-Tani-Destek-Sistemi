"""
Aşama 4 ve 7 - Klasik Makine Öğrenmesi ile Sınıflandırma ve Değerlendirme.

Derin öğrenme kullanılmaz; lojistik regresyon, SVM, Random Forest, k-NN ve gradyan artırma
gibi klasik yöntemler (scikit-learn) karşılaştırılır.

- Veri ayrımı görüntü/grup düzeyinde bir kez yapılır ve kilitlenir (data/processed/veri_bolumu.csv):
  aynı kaynağın augmentasyonları (grup_id) hep aynı tarafta kalır. Test seti yalnızca son
  değerlendirmede bir kez kullanılır; tüm seçimler geliştirme setinde çapraz doğrulamayla yapılır.
- Hücre düzeyi tahminler görüntü düzeyine olasılık ortalamasıyla birleştirilir.

Deneyler (model/segmentasyon karşılaştırması, ablasyon, kısayol, kenar): src/classification_analysis.py

Çalıştırma (proje kökünden; son modeli eğitir ve kilitli test setinde değerlendirir):
    python -m src.classifier
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.config import (
    CLASS_NAMES,
    CLASSIFIER_FEATURE_SET,
    CLASSIFIER_MODEL,
    CLASSIFIER_PARAMS,
    CLASSIFIER_TRAIN_EDGE_CELLS,
    CV_FOLDS,
    DATASET_INDEX_CSV,
    IMAGE_EXCLUDE_EDGE_CELLS,
    FEATURES_CSVS,
    MODELS_DIR,
    N_JOBS,
    RANDOM_STATE,
    SEGMENTATION_METHOD,
    SPLIT_CSV,
    TEST_FRACTION_FOLDS,
)
from src.dataset import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.features import feature_columns
from src.utils import save_figure, stage_dir

NO_CELL = "Hücre yok"   # Hiç çekirdek bulunamayan görüntünün görüntü düzeyi "tahmini"
MODEL_FILE = "losemi_siniflandirici"

# Model adları (sunum ve tablolar için) → oluşturucu. Parametre ızgaraları classification_analysis'te.
MODEL_NAMES = {
    "lojistik": "Lojistik regresyon",
    "knn": "k-NN",
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "hgb": "Gradyan artırma (HGB)",
}


# ---------------------------------------------------------------------------
# Veri ayrımı (kilitli test seti)
# ---------------------------------------------------------------------------

def make_split(index: pd.DataFrame) -> pd.DataFrame:
    """
    Görüntüleri grup düzeyinde geliştirme / test olarak ayırır ve geliştirme setini CV katlarına böler.

    - StratifiedGroupKFold (grup = grup_id, katman = etiket) ile görüntüler 1/TEST_FRACTION_FOLDS
      oranında test, kalanı geliştirme olur. Grup ayrımı, aynı görüntünün döndürülmüş/renk değiştirilmiş
      kopyalarının iki tarafa düşmesini (veri sızıntısını) engeller.
    - Geliştirme grupları aynı yöntemle CV_FOLDS kata ayrılır. Katlar görüntü düzeyinde sabitlendiği için
      üç segmentasyon yönteminin öznitelik tabloları ve tüm deneyler aynı katları kullanır.

    Parametreler:
        index: Veri seti indeksi (dosya_yolu, etiket, grup_id, tip).

    Döndürür:
        dosya_yolu, etiket, grup_id, tip, bolum ("gelistirme"/"test"), kat (test için -1) tablosu.
    """
    split = index[["dosya_yolu", "etiket", "grup_id", "tip"]].copy().reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    _, test_idx = next(outer.split(split, split["etiket"], split["grup_id"]))
    split["bolum"] = "gelistirme"
    split.loc[test_idx, "bolum"] = "test"
    split["kat"] = -1
    dev = split[split["bolum"] == "gelistirme"]
    inner = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, val_idx) in enumerate(inner.split(dev, dev["etiket"], dev["grup_id"])):
        split.loc[dev.index[val_idx], "kat"] = fold
    return split


def load_split() -> pd.DataFrame:
    """
    Kilitli veri ayrımını okur; dosya yoksa bir kez oluşturup kaydeder.

    Dosya varsa ASLA yeniden üretilmez: test seti bir kez belirlenir ve tüm deneylerde aynı kalır.

    Döndürür:
        make_split çıktısı biçiminde tablo.
    """
    if SPLIT_CSV.exists():
        return pd.read_csv(SPLIT_CSV)
    split = make_split(pd.read_csv(DATASET_INDEX_CSV))
    split.to_csv(SPLIT_CSV, index=False, encoding="utf-8")
    return split


def check_split(split: pd.DataFrame) -> None:
    """
    Veri ayrımında grup sızıntısı olmadığını doğrular (her grup tek bölümde ve tek katta).

    Parametreler:
        split: load_split çıktısı.
    """
    per_group = split.groupby("grup_id")[["bolum", "kat"]].nunique()
    if (per_group > 1).any().any():
        raise ValueError("Veri ayrımında sızıntı: bir grup birden fazla bölüme/kata düşmüş.")


def load_cells(method: str = SEGMENTATION_METHOD) -> pd.DataFrame:
    """
    Bir segmentasyon yönteminin öznitelik tablosunu okuyup bölüm/kat bilgisini ekler.

    Bölüm ve kat grup_id üzerinden eşlenir (grup tek bölümde olduğu için birebir).

    Parametreler:
        method: "kmeans", "otsu" veya "watershed".

    Döndürür:
        Öznitelik tablosu + bolum, kat sütunları.
    """
    cells = pd.read_csv(FEATURES_CSVS[method])
    split = load_split()
    group_map = split.drop_duplicates("grup_id").set_index("grup_id")[["bolum", "kat"]]
    return cells.join(group_map, on="grup_id")


def select_features(columns: list[str], feature_set: str) -> list[str]:
    """
    Adlandırılmış bir öznitelik alt kümesini döndürür (ablasyon ve son model için).

    Parametreler:
        columns: Tüm öznitelik sütunları (features.feature_columns).
        feature_set: "tumu", "sekil", "renk", "doku", "sekil_doku" (renksiz),
            "sekil_renk", "renk_doku", "std_haric" (tip bağımlı std ve GLCM ölçüleri hariç).

    Döndürür:
        Sütun adları listesi.
    """
    groups = {
        "sekil": [c for c in columns if c.startswith("sekil_")],
        "renk": [c for c in columns if c.startswith("renk_")],
        "doku": [c for c in columns if c.startswith("doku_")],
    }
    if feature_set == "tumu":
        return list(columns)
    if feature_set in groups:
        return groups[feature_set]
    if feature_set == "sekil_doku":
        return groups["sekil"] + groups["doku"]
    if feature_set == "sekil_renk":
        return groups["sekil"] + groups["renk"]
    if feature_set == "renk_doku":
        return groups["renk"] + groups["doku"]
    if feature_set == "std_haric":
        # Adım 4: std tabanlı renk ölçüleri ve GLCM görüntü tipini kısmen taşıyor (AUC 0.74-0.81)
        return groups["sekil"] + [c for c in groups["renk"] if not c.endswith("_std")]
    raise ValueError(f"Bilinmeyen öznitelik kümesi: {feature_set}")


# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------

def build_model(name: str, params: dict | None = None) -> Pipeline:
    """
    Tek bir klasik sınıflandırıcıyı ön işleme adımlarıyla birlikte Pipeline olarak oluşturur.

    - Medyan doldurma: tablolarda eksik değer yok, ama tahmin sırasında halkası < 5 piksel kalan
      hücrede halka öznitelikleri NaN olabilir; medyan aykırı değerlerden etkilenmez.
    - Standartlaştırma: SVM, k-NN ve lojistik regresyon mesafe/ölçek duyarlıdır (öznitelik aralıkları
      0-1 ile 0-360 arasında değişiyor). Ağaç modellerini etkilemez, tutarlılık için hep var.
    - Sınıf dengesizliği (hücre: AML 14.058, ALL 3.080, Healthy 1.178): class_weight="balanced"
      ile her sınıfın toplam ağırlığı eşitlenir; aksi hâlde model Healthy'yi neredeyse hiç tahmin etmez.
      k-NN ağırlık desteklemez (dengesizlikten en çok etkilenecek model; karşılaştırmada görülür).

    Parametreler:
        name: MODEL_NAMES anahtarlarından biri.
        params: Modele özgü hiperparametreler (verilmezse varsayılan).

    Döndürür:
        Eğitilmemiş sklearn Pipeline (imputer → scaler → model).
    """
    params = dict(params or {})
    if name == "lojistik":
        # max_iter 2000: 58 öznitelik ve 3 sınıfta lbfgs varsayılan 100 adımda yakınsamıyor
        model = LogisticRegression(class_weight="balanced", max_iter=2000, **params)
    elif name == "knn":
        model = KNeighborsClassifier(n_jobs=N_JOBS, **params)
    elif name == "svm":
        # Görüntü düzeyi birleştirme sınıf olasılıklarının ortalamasını kullanır; SVM karar değerleri
        # olasılık değildir. Platt (sigmoid) kalibrasyonu, eğitim setinin içinde 3 katlı CV ile öğrenilir
        # (sklearn 1.9'da SVC(probability=True) yerine önerilen yol). ensemble=False: tek SVM tüm
        # eğitim verisiyle eğitilir, kalibrasyon katlar dışı karar değerlerinden öğrenilir.
        model = CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", cache_size=500, **params),
                                       method="sigmoid", cv=3, ensemble=False)
    elif name == "rf":
        # balanced_subsample: ağaç başına bootstrap örneğinde yeniden dengeler
        model = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=N_JOBS,
                                       random_state=RANDOM_STATE, **params)
    elif name == "hgb":
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")
    return Pipeline([("doldurma", SimpleImputer(strategy="median")),
                     ("olcekleme", StandardScaler()),
                     ("model", model)])


def build_models() -> dict:
    """
    Karşılaştırılacak tüm klasik sınıflandırıcıları varsayılan parametrelerle oluşturur.

    Döndürür:
        {model adı: sklearn Pipeline} sözlüğü.
    """
    return {name: build_model(name) for name in MODEL_NAMES}


def train_models(X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Tüm modelleri eğitim verisi üzerinde eğitir.

    Parametreler:
        X: Öznitelik tablosu.
        y: Etiketler (ALL / AML / Healthy).

    Döndürür:
        {model adı: eğitilmiş model} sözlüğü.
    """
    models = build_models()
    with threadpool_limits(limits=N_JOBS):
        for model in models.values():
            model.fit(X, y)
    return models


def out_of_fold_proba(cells: pd.DataFrame, features: list[str], name: str, params: dict | None = None,
                      train_mask: pd.Series | None = None) -> np.ndarray:
    """
    Geliştirme setinde sabit katlarla katlar dışı (out-of-fold) sınıf olasılıklarını hesaplar.

    Her katta model diğer katların hücreleriyle eğitilir ve o katın TÜM hücreleri (kenara değenler
    dahil) için olasılık üretir; böylece hücre ve görüntü düzeyi metrikler aynı tahminlerden çıkar.

    Parametreler:
        cells: Yalnızca geliştirme hücreleri (kat sütunu 0..CV_FOLDS-1).
        features: Kullanılacak öznitelik sütunları.
        name: Model adı.
        params: Model hiperparametreleri.
        train_mask: Eğitime girebilecek hücreler (ör. kenara değmeyenler); None → tümü.

    Döndürür:
        (hücre sayısı, 3) olasılık dizisi; sütun sırası CLASS_NAMES.
    """
    proba = np.full((len(cells), len(CLASS_NAMES)), np.nan)
    allowed = np.ones(len(cells), bool) if train_mask is None else train_mask.to_numpy(bool)
    X = cells[features].to_numpy(np.float64)
    y = cells["etiket"].to_numpy()
    folds = cells["kat"].to_numpy()
    with threadpool_limits(limits=N_JOBS):
        for fold in range(CV_FOLDS):
            train = (folds != fold) & allowed
            test = folds == fold
            model = build_model(name, params).fit(X[train], y[train])
            proba[test] = reorder_proba(model, model.predict_proba(X[test]))
    return proba


def reorder_proba(model, proba: np.ndarray) -> np.ndarray:
    """
    Modelin predict_proba sütunlarını CLASS_NAMES sırasına getirir.

    Parametreler:
        model: Eğitilmiş model (classes_ özniteliği olan).
        proba: predict_proba çıktısı.

    Döndürür:
        CLASS_NAMES sırasında olasılık dizisi.
    """
    order = [list(model.classes_).index(c) for c in CLASS_NAMES]
    return proba[:, order]


# ---------------------------------------------------------------------------
# Değerlendirme
# ---------------------------------------------------------------------------

def label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Etiket dizilerinden doğruluk, dengeli doğruluk, makro F1, sınıf bazında duyarlılık/kesinlik/F1
    ve karışıklık matrisini hesaplar.

    Makro F1 birincil ölçüttür: sınıflar dengesiz (hücrelerin %77'si AML), doğruluk çoğunluk sınıfını ödüllendirir.
    Tahmin CLASS_NAMES dışında olabilir (ör. "Hücre yok"); bu durumda gerçek sınıfın kaçırılması sayılır.

    Parametreler:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.

    Döndürür:
        Metrik sözlüğü; "karisiklik" anahtarı CLASS_NAMES sırasında matris.
    """
    # Makro ortalamalar yalnızca gerçekte ya da tahminde bulunan sınıflar üzerinden: alt kümede olmayan
    # bir sınıf (ör. yakın çekimlerde Healthy yok) ortalamaya 0 olarak girip skoru haksızca düşürmesin.
    # Tahmin edilip gerçekte olmayan sınıf F1 = 0 ile girer (yanlış alarm cezalandırılır).
    present = [c for c in CLASS_NAMES if c in set(y_true) | set(y_pred)]
    in_true = [c for c in CLASS_NAMES if c in set(y_true)]
    out = {
        "n": len(y_true),
        "dogruluk": accuracy_score(y_true, y_pred),
        "dengeli_dogruluk": float(np.mean(recall_score(y_true, y_pred, labels=in_true, average=None,
                                                       zero_division=0))),
        "makro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
    }
    recalls = recall_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    for cls, r, p, f in zip(CLASS_NAMES, recalls, precisions, f1s):
        out[f"duyarlilik_{cls}"] = r if cls in in_true else np.nan   # gerçekte olmayan sınıfın duyarlılığı tanımsız
        out[f"kesinlik_{cls}"] = p if cls in present else np.nan
        out[f"f1_{cls}"] = f if cls in present else np.nan
    # Anormal (ALL+AML) ve Healthy ikili ayrımı: risk skorunun dayandığı ayrım
    abn_true = np.isin(y_true, ["ALL", "AML"])
    abn_pred = np.isin(y_pred, ["ALL", "AML"])
    out["anormal_duyarlilik"] = recall_score(abn_true, abn_pred, zero_division=0)
    out["healthy_ozgulluk"] = recall_score(~abn_true, ~abn_pred, zero_division=0) if (~abn_true).any() else np.nan
    out["karisiklik"] = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    return out


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """
    Modeli doğruluk, duyarlılık (recall), F1-skoru ve karışıklık matrisi ile değerlendirir.

    Parametreler:
        model: Eğitilmiş sınıflandırıcı.
        X_test: Test öznitelikleri.
        y_test: Test etiketleri.

    Döndürür:
        Metrikleri ve karışıklık matrisini içeren sözlük (label_metrics).
    """
    return label_metrics(np.asarray(y_test), model.predict(X_test))


def proba_to_labels(proba: np.ndarray) -> np.ndarray:
    """
    Olasılık dizisini en olası sınıf etiketine çevirir.

    Parametreler:
        proba: (n, 3) olasılıklar, CLASS_NAMES sırasında.

    Döndürür:
        Etiket dizisi.
    """
    return np.asarray(CLASS_NAMES)[np.argmax(proba, axis=1)]


def aggregate_images(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame,
                     exclude_edge: bool = IMAGE_EXCLUDE_EDGE_CELLS) -> pd.DataFrame:
    """
    Hücre olasılıklarını görüntü düzeyine birleştirir.

    - Görüntü tahmini = hücre olasılıklarının ortalamasının en büyüğü. Çoğunluk oyu yerine ortalama:
      Healthy görüntülerinde çoğunlukla 1-2 hücre var; oy eşitliğinde olasılık karar verir.
    - exclude_edge: kenara değen (kesik) hücreler, görüntüde kenara değmeyen hücre varsa ortalamaya
      girmez; yalnızca kesik hücreler varsa onlar kullanılır (düşük güvenli tahmin).
    - Hiç hücresi olmayan görüntüler "Hücre yok" tahmini alır (metrikte kaçırma sayılır).
    - Risk skoru (Adım 6) için anormal hücre oranı ve baskın lösemi tipi de hesaplanır.

    Parametreler:
        cells: Hücre tablosu (meta_kaynak_dosya, meta_kenara_degiyor dahil), proba ile aynı sırada.
        proba: Hücre olasılıkları (CLASS_NAMES sırasında).
        images: Değerlendirilecek görüntüler (dosya_yolu, etiket, tip).
        exclude_edge: Kenara değen hücreleri ortalamadan dışla.

    Döndürür:
        Görüntü başına: dosya_yolu, etiket, tip, hucre_sayisi, tahmin, olasılıklar,
        anormal_oran, baskin_tip.
    """
    df = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASS_NAMES])
    df["dosya_yolu"] = cells["meta_kaynak_dosya"].to_numpy()
    df["kenar"] = cells["meta_kenara_degiyor"].to_numpy(bool)
    df["hucre_tahmin"] = proba_to_labels(proba)
    if exclude_edge:
        has_inner = df.groupby("dosya_yolu")["kenar"].transform(lambda s: (~s).any())
        df = df[~df["kenar"] | ~has_inner]
    pcols = [f"p_{c}" for c in CLASS_NAMES]
    agg = df.groupby("dosya_yolu")[pcols].mean()
    agg["hucre_sayisi"] = df.groupby("dosya_yolu").size()
    agg["anormal_oran"] = df.assign(a=df["hucre_tahmin"].isin(["ALL", "AML"])).groupby("dosya_yolu")["a"].mean()
    counts = df.groupby("dosya_yolu")["hucre_tahmin"].value_counts().unstack(fill_value=0)
    abn = counts.reindex(columns=["ALL", "AML"], fill_value=0)
    agg["baskin_tip"] = np.where(abn.sum(axis=1) == 0, "-", abn.idxmax(axis=1)).astype(object)
    out = images[["dosya_yolu", "etiket", "tip"]].merge(agg, left_on="dosya_yolu", right_index=True, how="left")
    out["hucre_sayisi"] = out["hucre_sayisi"].fillna(0).astype(int)
    out["tahmin"] = NO_CELL
    has = out["hucre_sayisi"] > 0
    out.loc[has, "tahmin"] = proba_to_labels(out.loc[has, pcols].to_numpy())
    return out


def summarize(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame,
              exclude_edge: bool = IMAGE_EXCLUDE_EDGE_CELLS) -> dict:
    """
    Hücre ve görüntü düzeyinde özet metrikleri tek sözlükte toplar (karşılaştırma tabloları için).

    - Hücre düzeyi: exclude_edge ise yalnızca kenara değmeyen hücreler (kesik şekiller değerlendirmeyi
      sınıfa göre farklı bozar; Adım 4).
    - Görüntü düzeyi iki biçimde: "kapsanan" (≥1 hücre bulunan görüntüler) ve "tüm" (hücre bulunamayan
      görüntü kaçırma sayılır). Segmentasyon yöntemlerini adil karşılaştırmak için "tüm" kullanılır.

    Parametreler:
        cells: Hücre tablosu (proba ile aynı sırada).
        proba: Hücre olasılıkları.
        images: Değerlendirilen görüntüler.
        exclude_edge: Kenara değen hücreleri hücre metriklerinden ve görüntü ortalamasından dışla.

    Döndürür:
        Düz metrik sözlüğü (hucre_*, goruntu_*, kapsama).
    """
    mask = ~cells["meta_kenara_degiyor"].to_numpy(bool) if exclude_edge else np.ones(len(cells), bool)
    cell = label_metrics(cells["etiket"].to_numpy()[mask], proba_to_labels(proba[mask]))
    img = aggregate_images(cells, proba, images, exclude_edge)
    covered = img[img["tahmin"] != NO_CELL]
    img_all = label_metrics(img["etiket"].to_numpy(), img["tahmin"].to_numpy())
    img_cov = label_metrics(covered["etiket"].to_numpy(), covered["tahmin"].to_numpy())
    return {
        "hucre_n": cell["n"],
        "hucre_makro_f1": cell["makro_f1"],
        "hucre_dengeli_dogruluk": cell["dengeli_dogruluk"],
        "hucre_dogruluk": cell["dogruluk"],
        **{f"hucre_duyarlilik_{c}": cell[f"duyarlilik_{c}"] for c in CLASS_NAMES},
        "goruntu_kapsama": len(covered) / len(img),
        "goruntu_makro_f1_kapsanan": img_cov["makro_f1"],
        "goruntu_makro_f1_tum": img_all["makro_f1"],
        "goruntu_dengeli_dogruluk_tum": img_all["dengeli_dogruluk"],
        "goruntu_dogruluk_tum": img_all["dogruluk"],
    }


# ---------------------------------------------------------------------------
# Figürler ve model kaydı
# ---------------------------------------------------------------------------

def draw_confusion(ax: plt.Axes, cm: np.ndarray, labels: list[str], title: str) -> None:
    """
    Karışıklık matrisini satır yüzdesi (duyarlılık) renkleriyle ve ham sayılarla bir eksene çizer.

    Parametreler:
        ax: matplotlib ekseni.
        cm: Karışıklık matrisi (satır = gerçek, sütun = tahmin).
        labels: Sınıf adları.
        title: Eksen başlığı.
    """
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if pct[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{cm[i, j]}\n%{100 * pct[i, j]:.0f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Tahmin", color=TEXT_SECONDARY)
    ax.set_ylabel("Gerçek", color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], name: str, title: str = "",
                          stage: str = "evaluation"):
    """
    Karışıklık matrisini görselleştirip outputs/<stage>/ altına kaydeder.

    Parametreler:
        cm: Karışıklık matrisi.
        labels: Sınıf adları.
        name: Kaydedilecek dosya adı (uzantısız).
        title: Figür başlığı.
        stage: Çıktı aşaması (varsayılan evaluation).

    Döndürür:
        Kaydedilen figürün yolu.
    """
    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor=SURFACE)
    draw_confusion(ax, cm, labels, title)
    fig.tight_layout()
    return save_figure(fig, stage, name)


def save_model(model, name: str, metadata: dict | None = None):
    """
    Eğitilmiş modeli (ve kullanım için gereken meta veriyi) models/ klasörüne kaydeder.

    Parametreler:
        model: Eğitilmiş sınıflandırıcı.
        name: Dosya adı (uzantısız).
        metadata: Öznitelik sütunları, sınıflar, yöntem vb. (yanına JSON olarak da yazılır).

    Döndürür:
        .joblib dosyasının yolu.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "meta": metadata or {}}, path)
    if metadata:
        with open(MODELS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path


def load_model(name: str = MODEL_FILE):
    """
    models/ klasöründen kayıtlı bir modeli yükler.

    Parametreler:
        name: Dosya adı (uzantısız).

    Döndürür:
        (model, meta veri sözlüğü).
    """
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    return bundle["model"], bundle["meta"]


# ---------------------------------------------------------------------------
# Son model: geliştirme setinin tamamıyla eğitim, kilitli test setinde TEK değerlendirme
# ---------------------------------------------------------------------------

def plot_test_results(cell_metrics: dict, img_metrics: dict, per_type: pd.DataFrame) -> None:
    """
    Test seti karışıklık matrislerini (hücre ve görüntü) yan yana ve görüntü tipine göre makro F1'i çizer.

    Parametreler:
        cell_metrics: Hücre düzeyi label_metrics çıktısı.
        img_metrics: Görüntü düzeyi (kapsanan görüntüler) label_metrics çıktısı.
        per_type: Görüntü tipine göre hücre metrikleri tablosu.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), facecolor=SURFACE)
    draw_confusion(axes[0], cell_metrics["karisiklik"], CLASS_NAMES,
                   f"Hücre düzeyi (n={cell_metrics['n']}, makro F1 {cell_metrics['makro_f1']:.2f})")
    draw_confusion(axes[1], img_metrics["karisiklik"], CLASS_NAMES,
                   f"Görüntü düzeyi (n={img_metrics['n']}, makro F1 {img_metrics['makro_f1']:.2f})")
    fig.suptitle("Kilitli test seti", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "evaluation", "test_karisiklik_matrisleri")
    plot_confusion_matrix(cell_metrics["karisiklik"], CLASS_NAMES, "test_karisiklik_hucre", "Test — hücre düzeyi")
    plot_confusion_matrix(img_metrics["karisiklik"], CLASS_NAMES, "test_karisiklik_goruntu", "Test — görüntü düzeyi")


def main() -> None:
    """Son modeli geliştirme setinin tamamıyla eğitir, kaydeder ve kilitli test setinde bir kez değerlendirir."""
    pd.set_option("display.width", 220)
    out_dir = stage_dir("evaluation")
    split = load_split()
    check_split(split)
    cells = load_cells(SEGMENTATION_METHOD)
    columns = select_features(feature_columns(cells), CLASSIFIER_FEATURE_SET)
    dev = cells[cells["bolum"] == "gelistirme"]
    test = cells[cells["bolum"] == "test"].reset_index(drop=True)
    train = dev if CLASSIFIER_TRAIN_EDGE_CELLS else dev[~dev["meta_kenara_degiyor"]]
    print(f"Model: {MODEL_NAMES[CLASSIFIER_MODEL]} {CLASSIFIER_PARAMS}, öznitelik: {CLASSIFIER_FEATURE_SET} "
          f"({len(columns)}), yöntem: {SEGMENTATION_METHOD}, eğitim hücresi: {len(train)}")

    model = build_model(CLASSIFIER_MODEL, CLASSIFIER_PARAMS)
    with threadpool_limits(limits=N_JOBS):
        model.fit(train[columns].to_numpy(np.float64), train["etiket"].to_numpy())
        proba = reorder_proba(model, model.predict_proba(test[columns].to_numpy(np.float64)))
    save_model(model, MODEL_FILE, {
        "model": CLASSIFIER_MODEL, "parametreler": CLASSIFIER_PARAMS, "oznitelikler": columns,
        "siniflar": CLASS_NAMES, "segmentasyon": SEGMENTATION_METHOD, "oznitelik_kumesi": CLASSIFIER_FEATURE_SET,
        "kenar_hucreleri_egitimde": CLASSIFIER_TRAIN_EDGE_CELLS, "egitim_hucre_sayisi": int(len(train)),
    })

    test_images = split[split["bolum"] == "test"]
    inner = ~test["meta_kenara_degiyor"].to_numpy(bool)
    y_cell = test["etiket"].to_numpy()
    pred_cell = proba_to_labels(proba)
    cell_m = label_metrics(y_cell[inner], pred_cell[inner])
    cell_all = label_metrics(y_cell, pred_cell)
    img = aggregate_images(test, proba, test_images)
    img.to_csv(out_dir / "test_goruntu_tahminleri.csv", index=False, encoding="utf-8")
    covered = img[img["tahmin"] != NO_CELL]
    img_m = label_metrics(covered["etiket"].to_numpy(), covered["tahmin"].to_numpy())
    img_all = label_metrics(img["etiket"].to_numpy(), img["tahmin"].to_numpy())

    rows = []
    for title, m in [("hucre (kenara degmeyen)", cell_m), ("hucre (tum)", cell_all),
                     ("goruntu (hucre bulunan)", img_m), ("goruntu (tum; hucre yok = hata)", img_all)]:
        rows.append({"duzey": title, **{k: v for k, v in m.items() if k != "karisiklik"}})
    per_type = []
    for tip in ["yayma", "yakin_cekim"]:
        sel = inner & (test["meta_tip"].to_numpy() == tip)
        if sel.any():
            m = label_metrics(y_cell[sel], pred_cell[sel])
            per_type.append({"duzey": f"hucre ({tip})", **{k: v for k, v in m.items() if k != "karisiklik"}})
        isel = covered["tip"] == tip
        if isel.any():
            m = label_metrics(covered.loc[isel, "etiket"].to_numpy(), covered.loc[isel, "tahmin"].to_numpy())
            per_type.append({"duzey": f"goruntu ({tip})", **{k: v for k, v in m.items() if k != "karisiklik"}})
    table = pd.DataFrame(rows + per_type)
    table.to_csv(out_dir / "test_metrikleri.csv", index=False, encoding="utf-8")
    plot_test_results(cell_m, img_m, pd.DataFrame(per_type))

    cols = ["duzey", "n", "dogruluk", "dengeli_dogruluk", "makro_f1"] + [f"duyarlilik_{c}" for c in CLASS_NAMES] \
        + ["anormal_duyarlilik", "healthy_ozgulluk"]
    print("\n=== Kilitli test seti ===\n" + table[cols].round(3).to_string(index=False))
    print("\nHücre karışıklık matrisi (satır gerçek, sütun tahmin; " + "/".join(CLASS_NAMES) + "):\n", cell_m["karisiklik"])
    print("Görüntü karışıklık matrisi:\n", img_m["karisiklik"])
    print(f"Hücre bulunamayan test görüntüsü: {(img['tahmin'] == NO_CELL).sum()} / {len(img)}")
    # Ayrıntılı yanlış sınıflanan görüntüler (sunumda örnek göstermek için)
    wrong = covered[covered["etiket"] != covered["tahmin"]]
    print(f"Yanlış sınıflanan görüntü: {len(wrong)}\n" + wrong.groupby(["etiket", "tahmin"]).size().to_string())


if __name__ == "__main__":
    main()
