"""
Adım 8.4-8.5 — Alan kayması düzeltme denemeleri ve son değerlendirme.

Denemeler (her biri ayrı raporlanır; eğitim YALNIZCA Kaggle v2 geliştirme setiyle, Münih yalnızca AYAR kısmında
değerlendirme için kullanılır):
a) olcek     : Yüksek büyütmeli görüntüler (yakın çekim) yayma çözünürlüğüne (eritrosit çapı ≈ 10 px) indirilip
               işlenir. Model DEĞİŞMEZ; yalnızca çıkarım değişir.
b) macenko   : Macenko boya ayrıştırmasıyla normalizasyon (Kaggle referans boya matrisine), ardından mevcut zemin
               hizalaması; öznitelikler yeniden çıkarılır, model Kaggle v2 ayrımıyla yeniden eğitilir.
c) artirma   : Eğitimde her Kaggle görüntüsünün 2 rastgele renk/parlaklık/kontrast kopyası (external_validation.
               augment_color) eklenir; çıkarım değişmez.
d) goreli    : Renge daha az bağımlı model: şekil + doku + yalnızca göreli renk (çekirdek − çevre farkları).
Birleşimler (ör. a + d) yalnızca tek tek denemelerden sonra, teşhise dayanarak eklenir.

Seçim ölçütü: Münih AYAR dengeli doğruluğu ve Kaggle geliştirme CV'si birlikte (bkz. select_config). Kaggle kilitli
test her deneme için raporlanır (Kaggle başarısı ne kadar düşüyor?) ama seçimde kullanılmaz. Münih TEST'e yalnızca
`sonuc` bölümünde, seçilen yapılandırma ve mevcut model için bir kez bakılır.

Çalıştırma (proje kökünden):
    python -m src.domain_shift denemeler     # tüm denemeler: geliştirme CV, Münih AYAR, Kaggle test
    python -m src.domain_shift sonuc         # seçilen + mevcut: Münih TEST, ALL-IDB, Kaggle test; çok merkezli ek deney
"""

import json
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import (
    CALIBRATION_FOLDS,
    CLASS_NAMES,
    CLASSIFIER_CALIBRATION,
    CLASSIFIER_MODEL,
    CLASSIFIER_PARAMS,
    CV_FOLDS,
    FEATURE_REF_RBC_DIAMETER,
    IMAGE_SIZE,
    MODEL_FILES,
    N_JOBS,
    PADDING_VALUE,
    RANDOM_STATE,
    SPLIT_VERSION,
)
from src.utils import feature_columns, resolve_path, stage_dir

# Yakın çekim eşiği: eritrosit çapı > 20 px (256 px uzayında). Kaggle yaymalarında çap ~10 px (yedek değer 10),
# yakın çekimlerde ~62 px; 20 px iki dağılımın arasında güvenli bir sınır (Adım 8 ölçek teşhisi).
CLOSE_UP_MIN_DIAMETER = 20.0
# Hedef çap: yayma çözünürlüğü = öznitelik referans ölçeği (10 px); yakın çekim bu ölçeğe indirilir.
SMEAR_DIAMETER = float(FEATURE_REF_RBC_DIAMETER)
# Macenko: optik yoğunluk eşiği (β = 0.15; Macenko ve ark. 2009) ve açı yüzdelikleri. Özgün α = %1 bu veride
# seyrek artefaktları (turuncu leke, dolgu kenarı) boya yönü olarak seçiyordu; α = %5 daha kararlı. Referans için
# Kaggle geliştirme setinden sınıf başına 20 görüntü.
MACENKO_BETA = 0.15
MACENKO_ALPHA = 5.0
MACENKO_REF_PER_CLASS = 20
AUG_COPIES = 2   # Renk artırması: görüntü başına 2 kopya (eğitim verisi 3 katı; süre makul)
EXPERIMENTS = {
    # ad: (eğitim öznitelik varyantı, artırma kopyası, çıkarım varyantı, öznitelik kümesi)
    "mevcut": ("varsayilan", 0, "varsayilan", "tumu"),
    "a_olcek": ("varsayilan", 0, "olcek", "tumu"),
    "b_macenko": ("macenko", 0, "macenko", "tumu"),
    "c_artirma": ("varsayilan", AUG_COPIES, "varsayilan", "tumu"),
    "d_goreli": ("varsayilan", 0, "varsayilan", "goreli"),
}


# ---------------------------------------------------------------------------
# a) Ölçek uyarlaması
# ---------------------------------------------------------------------------

def adapt_scale(rgb: np.ndarray) -> tuple[np.ndarray, float | None]:
    """
    Yüksek büyütmeli görüntüyü yayma çözünürlüğüne indirir.

    Eritrosit çapı (256 px uzayında) CLOSE_UP_MIN_DIAMETER'dan büyükse görüntü, 256 px uzayındaki çap
    SMEAR_DIAMETER olacak şekilde küçültülür ve 256x256 dolgulu tuvalin ortasına konur (dolgu maskesi tuvali
    hesaplardan dışlar). Kaggle yaymaları da mikroskop görüntüsünden ~10 kat küçültülerek oluşmuştu; bu dönüşüm
    yakın çekimdeki hücreyi aynı çözünürlüğe ve aynı örnekleme bulanıklığına getirir.

    Parametreler:
        rgb: RGB görüntü.

    Döndürür:
        (işlenecek görüntü, zorlanan eritrosit çapı veya yayma ise None).
    """
    from src.preprocessing import preprocess
    from src.segmentation import image_rbc_diameter

    _, steps = preprocess(rgb)
    diameter, _ = image_rbc_diameter(steps)
    if diameter <= CLOSE_UP_MIN_DIAMETER:
        return rgb, None
    h, w = rgb.shape[:2]
    s = IMAGE_SIZE / max(h, w) * SMEAR_DIAMETER / diameter
    new_w, new_h = max(8, round(w * s)), max(8, round(h * s))
    small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left, top = (IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2
    canvas = cv2.copyMakeBorder(small, top, IMAGE_SIZE - new_h - top, left, IMAGE_SIZE - new_w - left,
                                cv2.BORDER_CONSTANT, value=(PADDING_VALUE,) * 3)
    return canvas, SMEAR_DIAMETER


# ---------------------------------------------------------------------------
# b) Macenko boya normalizasyonu
# ---------------------------------------------------------------------------

def optical_density(rgb: np.ndarray) -> np.ndarray:
    """
    RGB'yi optik yoğunluğa çevirir: OD = −log((I + 1) / 256).

    Parametreler:
        rgb: RGB uint8 görüntü.

    Döndürür:
        (H, W, 3) float OD.
    """
    return -np.log((rgb.astype(np.float64) + 1) / 256)


def macenko_stains(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Macenko yöntemiyle iki boya vektörünü ve boya yoğunluklarının %99'luk değerlerini kestirir.

    Parametreler:
        rgb: RGB görüntü.
        mask: Geçerli piksel maskesi.

    Döndürür:
        ((3, 2) boya matrisi, (2,) %99 yoğunluklar) veya yeterli boyalı piksel yoksa None.
    """
    od = optical_density(rgb)[mask]
    return macenko_from_od(od[od.sum(axis=1) > MACENKO_BETA])


def macenko_from_od(od: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Boyalı piksellerin optik yoğunluklarından Macenko boya vektörlerini kestirir.

    Pikseller OD uzayında en büyük iki özvektörün düzlemine izdüşürülür; düzlemdeki açıların %α ve %(100−α)
    yüzdelikleri iki boya yönünü verir. Kan yaymasında (May-Grünwald-Giemsa) bunlar çekirdek boyası (mor: kırmızı
    ve yeşili soğurur, maviyi geçirir) ve eritrosit boyası (pembe: yeşili soğurur, kırmızı ve maviyi geçirir).
    Sıra OD(R) − OD(B) ile sabitlenir: çekirdek boyasında büyüktür (çekirdek boyası ilk sütun). Kırmızı bileşene
    göre sıralamak görüntüden görüntüye boyaları yer değiştiriyordu.

    Parametreler:
        od: (N, 3) boyalı piksel OD'leri.

    Döndürür:
        ((3, 2) boya matrisi, (2,) %99 yoğunluklar) veya piksel azsa None.
    """
    if len(od) < 200:
        return None
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]
    proj = od @ plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(angles, [MACENKO_ALPHA, 100 - MACENKO_ALPHA])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    stains = np.stack([v * np.sign(v.sum()) for v in (v1, v2)], axis=1)
    # Boya soğurması fiziksel olarak negatif olamaz: negatif bileşenler sıfırlanır
    stains = np.clip(stains, 0, None)
    stains /= np.maximum(np.linalg.norm(stains, axis=0, keepdims=True), 1e-9)
    if stains[0, 0] - stains[2, 0] < stains[0, 1] - stains[2, 1]:   # Çekirdek boyası (OD(R) − OD(B) büyük) önce
        stains = stains[:, ::-1]
    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    return stains, np.percentile(conc, 99, axis=1)


@lru_cache(maxsize=1)
def macenko_reference() -> tuple[np.ndarray, np.ndarray]:
    """
    Kaggle geliştirme setinden (sınıf başına MACENKO_REF_PER_CLASS görüntü) referans boya matrisi ve yoğunlukları.

    Görüntülerin boyalı pikselleri havuzlanır (görüntü başına en fazla 5.000, sabit tohum) ve boyalar bu havuzdan
    tek seferde kestirilir: görüntü başına vektörlerin medyanı, az çekirdekli Healthy yaymalarındaki kararsız
    kestirimler yüzünden iki boyayı karıştırıyordu. outputs/external/macenko_referans.json'a yazılır (yalnızca sayılar).

    Döndürür:
        ((3, 2) boya matrisi, (2,) yoğunluklar).
    """
    from src.classifier import load_split
    from src.preprocessing import resize_with_padding, valid_pixel_mask
    from src.utils import load_image

    path = stage_dir("external") / "macenko_referans.json"
    if path.exists():
        ref = json.loads(path.read_text(encoding="utf-8"))
        return np.array(ref["boya_matrisi"]), np.array(ref["yogunluk_99"])
    split = load_split()
    dev = split[split["bolum"] == "gelistirme"]
    rng = np.random.default_rng(RANDOM_STATE)
    pool, n_images = [], 0
    for c in CLASS_NAMES:
        sub = dev[dev["etiket"] == c]
        for p in sub["dosya_yolu"].iloc[rng.choice(len(sub), MACENKO_REF_PER_CLASS, replace=False)]:
            img = resize_with_padding(load_image(resolve_path(p)))
            od = optical_density(img)[valid_pixel_mask(img)]
            od = od[od.sum(axis=1) > MACENKO_BETA]
            pool.append(od[rng.choice(len(od), min(5000, len(od)), replace=False)])
            n_images += 1
    ref_s, ref_c = macenko_from_od(np.concatenate(pool))
    path.write_text(json.dumps({"boya_matrisi": ref_s.tolist(), "yogunluk_99": ref_c.tolist(),
                                "goruntu_sayisi": n_images, "sutunlar": ["cekirdek_boyasi", "eritrosit_boyasi"]},
                               indent=2), encoding="utf-8")
    return ref_s, ref_c


def macenko_normalize(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Görüntünün boyalarını Kaggle referans boya matrisine ve yoğunluklarına taşır (Macenko).

    Parametreler:
        rgb: RGB görüntü.
        mask: Geçerli piksel maskesi (dolgu 0 kalır).

    Döndürür:
        Normalize RGB görüntü; boya kestirilemezse görüntünün kopyası.
    """
    est = macenko_stains(rgb, mask)
    if est is None:
        return rgb.copy()
    stains, maxc = est
    ref_s, ref_c = macenko_reference()
    # Kestirim çöktüyse (iki boya neredeyse aynı yön; ör. tek lökositli yayma) görüntünün boyaları yerine referans
    # boyalar kullanılır, yalnızca yoğunluklar ölçeklenir. 0.98 ≈ 11° açı farkı.
    if abs(float(stains[:, 0] @ stains[:, 1])) > 0.98:
        stains = ref_s
        conc = np.linalg.lstsq(stains, optical_density(rgb)[mask].T, rcond=None)[0]
        maxc = np.percentile(conc, 99, axis=1)
    od = optical_density(rgb).reshape(-1, 3)
    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    conc *= (ref_c / np.maximum(maxc, 1e-6))[:, None]
    out = np.clip(256 * np.exp(-(ref_s @ conc)).T - 1, 0, 255).reshape(rgb.shape).astype(np.uint8)
    out[~mask] = 0
    return out


def macenko_color(steps: dict) -> np.ndarray:
    """
    Öznitelik çıkarımında kullanılacak Macenko normalize renk görüntüsü.

    Boyutlandırılmış görüntüye Macenko boya normalizasyonu, ardından mevcut zemin hizalaması (preprocessing.
    normalize_color) uygulanır; böylece renk öznitelikleri varsayılan pipeline'la aynı zemin hedefine göre ölçülür.
    Segmentasyon bu görüntüden YAPILMAZ: iki boyalı Macenko modeli Giemsa yaymasındaki üç renk bileşenini (pembe
    eritrosit, mor çekirdek, mavi sitoplazma) temsil edemiyor; eritrositler grileşip çekirdekten ayrılamıyordu
    (ör. bir AML yaymasında 8 yerine 75 "çekirdek").

    Parametreler:
        steps: Varsayılan preprocess ara çıktıları (boyutlandirilmis, gecerli_maske).

    Döndürür:
        RGB görüntü (dolgu 0).
    """
    from src.preprocessing import normalize_color

    mask = steps["gecerli_maske"]
    return normalize_color(macenko_normalize(steps["boyutlandirilmis"], mask), mask)


# ---------------------------------------------------------------------------
# Kaggle öznitelikleri, eğitim ve değerlendirme
# ---------------------------------------------------------------------------

def feature_set(columns: list[str], name: str) -> list[str]:
    """
    Deneme öznitelik kümesi.

    Parametreler:
        columns: Tüm öznitelik sütunları.
        name: "tumu" veya "goreli" (şekil + doku + çekirdek−çevre renk farkları; mutlak renk yok).

    Döndürür:
        Sütun listesi.
    """
    if name == "goreli":
        return [c for c in columns if c.startswith(("sekil_", "doku_", "renk_fark_"))]
    return list(columns)


def kaggle_features(variant: str, copies: int = 0) -> pd.DataFrame:
    """
    Kaggle görüntülerinin (geliştirme + test) bir varyanttaki hücre özniteliklerini v2 ayrım bilgisiyle döndürür.

    "varsayilan" ve kopyasız için Adım 4 tablosu (features_kmeans.csv) kullanılır: uçtan uca pipeline ile aynı
    hesaplama (Adım 6 tutarlılık kontrolü). Diğer varyantlar external_validation.extract_dataset ile çıkarılır.

    Parametreler:
        variant: Çıkarım varyantı.
        copies: Renk artırmalı kopya sayısı.

    Döndürür:
        Öznitelikler + id, meta_kopya, meta_kenara_degiyor, etiket, bolum, kat, kaynak_grup_id.
    """
    from src.classifier import load_cells, load_split
    from src.external_validation import extract_dataset

    split = load_split()
    if variant == "varsayilan" and copies == 0:
        cells = load_cells("kmeans").rename(columns={"meta_kaynak_dosya": "id"})
        cells["meta_kopya"] = 0
        return cells
    items = pd.DataFrame({"id": split["dosya_yolu"], "yol": [str(resolve_path(p)) for p in split["dosya_yolu"]]})
    cells, _ = extract_dataset("kaggle", items, variant, copies)
    info = split.rename(columns={"dosya_yolu": "id"})[["id", "etiket", "bolum", "kat", "kaynak_grup_id", "tip"]]
    return cells.merge(info, on="id")


def kaggle_images() -> pd.DataFrame:
    """
    Kaggle v2 ayrımındaki görüntüler (aggregate_images için dosya_yolu, etiket, tip, bolum, kat).

    Döndürür:
        Ayrım tablosu.
    """
    from src.classifier import load_split
    return load_split()


def fit(train: pd.DataFrame, cols: list[str], calibration: str | None = CLASSIFIER_CALIBRATION):
    """
    Seçili modeli (HGB, Adım 5 parametreleri) eğitir; istenirse temperature kalibrasyonu (kaynak gruplu iç katlar).

    Parametreler:
        train: Eğitim hücreleri.
        cols: Öznitelik sütunları.
        calibration: Kalibrasyon yöntemi (None: yok).

    Döndürür:
        Eğitilmiş model.
    """
    from src.classifier import fit_model

    with threadpool_limits(limits=N_JOBS):
        return fit_model(train[cols].to_numpy(np.float64), train["etiket"].to_numpy(),
                         train["kaynak_grup_id"].to_numpy(), CLASSIFIER_MODEL, CLASSIFIER_PARAMS, calibration)


def kaggle_scores(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame) -> dict:
    """
    Kaggle hücre ve görüntü düzeyi ölçüleri (Adım 5 tanımları: hücre = kenara değmeyen, görüntü = hücre bulunan
    ve tümü).

    Parametreler:
        cells: Değerlendirilen hücreler (kopya 0).
        proba: Olasılıklar.
        images: Değerlendirilen görüntüler.

    Döndürür:
        {hucre_makro_f1, goruntu_makro_f1_kapsanan, goruntu_makro_f1_tum, hucre_duyarlilik_<sınıf>}.
    """
    from src.classifier import NO_CELL, aggregate_images, label_metrics, proba_to_labels

    inner = ~cells["meta_kenara_degiyor"].astype(bool).to_numpy()
    cm = label_metrics(cells["etiket"].to_numpy()[inner], proba_to_labels(proba)[inner])
    agg = aggregate_images(cells.rename(columns={"id": "meta_kaynak_dosya"}), proba, images)
    cov = agg[agg["tahmin"] != NO_CELL]
    return {"hucre_makro_f1": cm["makro_f1"],
            **{f"hucre_duyarlilik_{c}": cm[f"duyarlilik_{c}"] for c in CLASS_NAMES},
            "goruntu_makro_f1_kapsanan": label_metrics(cov["etiket"].to_numpy(), cov["tahmin"].to_numpy())["makro_f1"],
            "goruntu_makro_f1_tum": label_metrics(agg["etiket"].to_numpy(), agg["tahmin"].to_numpy())["makro_f1"]}


def experiment_data(name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """
    Bir denemenin eğitim ve çıkarım Kaggle tablolarını ve öznitelik listesini hazırlar.

    Parametreler:
        name: EXPERIMENTS anahtarı.

    Döndürür:
        (eğitim hücreleri (artırmalı kopyalar dahil), çıkarım hücreleri (kopya 0), öznitelik sütunları, çıkarım varyantı).
    """
    train_variant, copies, infer_variant, fset = EXPERIMENTS[name]
    train = kaggle_features(train_variant, copies)
    infer = train if (infer_variant == train_variant and copies == 0) else kaggle_features(infer_variant)
    infer = infer[infer["meta_kopya"] == 0].reset_index(drop=True)
    cols = feature_set(feature_columns(train), fset)
    return train, infer, cols, infer_variant


def dev_cv(train: pd.DataFrame, infer: pd.DataFrame, cols: list[str]) -> dict:
    """
    Kaggle geliştirme setinde sabit 5 katlı kaynak grubu CV (Adım 5 katları). Eğitim katında artırmalı kopyalar
    dahil; değerlendirme yalnızca kopya 0 (çıkarım varyantıyla). Kalibrasyonsuz (sınıf kararı aynı).

    Parametreler:
        train, infer, cols: experiment_data çıktıları.

    Döndürür:
        kaggle_scores ölçüleri ("cv_" önekli).
    """
    tr = train[train["bolum"] == "gelistirme"]
    ev = infer[infer["bolum"] == "gelistirme"].reset_index(drop=True)
    proba = np.zeros((len(ev), len(CLASS_NAMES)))
    from src.classifier import reorder_proba
    for k in range(CV_FOLDS):
        model = fit(tr[tr["kat"] != k], cols, calibration=None)
        sel = (ev["kat"] == k).to_numpy()
        proba[sel] = reorder_proba(model, model.predict_proba(ev.loc[sel, cols].to_numpy(np.float64)))
    images = kaggle_images()
    return {f"cv_{k}": v for k, v in kaggle_scores(ev, proba, images[images["bolum"] == "gelistirme"]).items()}


def final_model(name: str, train: pd.DataFrame, cols: list[str], infer_variant: str,
                extra: pd.DataFrame | None = None) -> tuple:
    """
    Denemenin son modelini geliştirme setinin tamamıyla (ve istenirse ek hücrelerle) temperature kalibrasyonlu eğitir.

    Parametreler:
        name: Deneme adı.
        train: Eğitim hücreleri.
        cols: Öznitelikler.
        infer_variant: Çıkarım varyantı (meta veriye yazılır).
        extra: Eğitime eklenecek hücreler (çok merkezli ek deney; etiket ve kaynak_grup_id ile).

    Döndürür:
        (model, meta) — external_validation.predict_cells ile kullanılabilir.
    """
    data = train[train["bolum"] == "gelistirme"]
    if extra is not None:
        data = pd.concat([data, extra], ignore_index=True)
        # Kaggle grup kimlikleri tamsayı, Münih'inkiler metin: iç kat ayırıcısı sıralayabilsin diye hepsi metin
        data["kaynak_grup_id"] = data["kaynak_grup_id"].astype(str)
    model = fit(data, cols)
    meta = {"model": CLASSIFIER_MODEL, "parametreler": CLASSIFIER_PARAMS, "oznitelikler": cols,
            "siniflar": CLASS_NAMES, "segmentasyon": "kmeans", "deneme": name, "egitim_varyanti": EXPERIMENTS.get(
                name, (None,))[0], "cikarim_varyanti": infer_variant, "artirma_kopyasi": EXPERIMENTS.get(name, (0, 0))[1],
            "veri_ayrimi": SPLIT_VERSION, "kalibrasyon": CLASSIFIER_CALIBRATION, "kalibrasyon_ic_kat": CALIBRATION_FOLDS,
            "egitim_hucre_sayisi": int(len(data)), "munih_egitimde": extra is not None}
    return model, meta


def kaggle_test(bundle: tuple, infer: pd.DataFrame) -> dict:
    """
    Kaggle v2 kilitli test setinde ölçüler (raporlama için; seçimde kullanılmaz).

    Parametreler:
        bundle: (model, meta).
        infer: Çıkarım varyantındaki Kaggle hücreleri (kopya 0).

    Döndürür:
        kaggle_scores ölçüleri ("test_" önekli).
    """
    from src.external_validation import predict_cells

    test = infer[infer["bolum"] == "test"].reset_index(drop=True)
    images = kaggle_images()
    scores = kaggle_scores(test, predict_cells(bundle, test), images[images["bolum"] == "test"])
    return {f"test_{k}": v for k, v in scores.items()}


def munich_scores(results: pd.DataFrame, prefix: str) -> dict:
    """
    Münih sonuçlarından özet ölçüler (merkez hücre).

    Parametreler:
        results: evaluate_munich çıktısı.
        prefix: Sütun öneki ("ayar_" / "test_munih_").

    Döndürür:
        {<önek>dengeli_dogruluk, <önek>duyarlilik_<sınıf>, <önek>MYO_vs_normal_f1/duyarlilik/ozgulluk}.
    """
    from src.external_validation import munich_metrics

    m = munich_metrics(results[results["beklenen"] != ""]).set_index("sinif")
    out = {f"{prefix}dengeli_dogruluk": m.loc["Dengeli doğruluk (MYO, LYT, NGS)", "duyarlilik"]}
    for cls in ("MYO", "LYT", "NGS"):
        out[f"{prefix}duyarlilik_{cls}"] = m.loc[cls, "duyarlilik"]
    b = m.loc["MYO vs LYT+NGS (merkez hücre)"]
    out.update({f"{prefix}anormal_duyarlilik": b["duyarlilik"], f"{prefix}normal_ozgulluk": b["ozgulluk"],
                f"{prefix}anormal_f1": b["f1"]})
    return out


# ---------------------------------------------------------------------------
# Bölümler
# ---------------------------------------------------------------------------

def run_experiments(names: list[str] | None = None) -> pd.DataFrame:
    """
    Denemeleri çalıştırır: Kaggle geliştirme CV, son model, Münih AYAR ve Kaggle kilitli test. Her denemeden sonra
    sonuç tablosu güncellenir (outputs/external/duzeltme_denemeleri.csv).

    Parametreler:
        names: Çalıştırılacak denemeler (None: hepsi).

    Döndürür:
        Deneme başına ölçü tablosu.
    """
    from src.classifier import load_model
    from src.external_validation import evaluate_munich, munich_metrics

    out = stage_dir("external")
    path = out / "duzeltme_denemeleri.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["deneme"])
    for name in names or list(EXPERIMENTS):
        start = time.time()
        print(f"\n--- {name} {EXPERIMENTS[name]} ---", flush=True)
        train, infer, cols, infer_variant = experiment_data(name)
        row = {"deneme": name, "egitim_varyanti": EXPERIMENTS[name][0], "artirma": EXPERIMENTS[name][1],
               "cikarim_varyanti": infer_variant, "oznitelik_kumesi": EXPERIMENTS[name][3], "oznitelik_sayisi": len(cols)}
        row.update(dev_cv(train, infer, cols))
        bundle = load_model() if name == "mevcut" else final_model(name, train, cols, infer_variant)
        if name not in ("mevcut", "a_olcek"):
            save_bundle(bundle, name)
        if name == "a_olcek":
            bundle = load_model()   # Model değişmez
        res = evaluate_munich(bundle, name, ("ayar",), infer_variant)
        res.drop(columns="yol").to_csv(out / f"munih_ayar_sonuclari_{name}.csv", index=False, encoding="utf-8")
        munich_metrics(res).to_csv(out / f"munih_ayar_metrikleri_{name}.csv", index=False, encoding="utf-8")
        row.update(munich_scores(res, "ayar_"))
        row.update(kaggle_test(bundle, infer))
        row["sure_sn"] = round(time.time() - start)
        table = pd.concat([table[table["deneme"] != name], pd.DataFrame([row])], ignore_index=True)
        table.to_csv(path, index=False, encoding="utf-8")
        show = ["cv_hucre_makro_f1", "cv_goruntu_makro_f1_tum", "ayar_dengeli_dogruluk", "ayar_duyarlilik_MYO",
                "ayar_duyarlilik_LYT", "ayar_duyarlilik_NGS", "test_hucre_makro_f1", "test_goruntu_makro_f1_tum"]
        print(pd.Series({k: row[k] for k in show}).round(3).to_string(), flush=True)
    return table


def save_bundle(bundle: tuple, name: str) -> None:
    """
    Deneme modelini models/ altına yeni bir sürüm olarak kaydeder (varsayılan model DEĞİŞMEZ).

    Parametreler:
        bundle: (model, meta).
        name: Deneme adı → models/<v2 model adı>_<deneme>_kalibre.joblib.
    """
    from src.classifier import save_model

    save_model(bundle[0], f"{MODEL_FILES[SPLIT_VERSION]}_{name}_kalibre", bundle[1])


# Seçim kuralı (sonuçlar görülmeden önce sabitlendi): Münih AYAR dengeli doğruluğu en yüksek yapılandırma; ancak
# Kaggle geliştirme CV görüntü makro F1'i (tümü) mevcut modele göre en fazla 0.02 düşebilir ("Kaggle başarısını
# belirgin düşürmeden"). 0.02: Adım 5'te görüntü düzeyi katlar arası sapma düzeyi.
MAX_KAGGLE_CV_DROP = 0.02


def select_config(table: pd.DataFrame) -> str:
    """
    Deneme tablosundan seçim kuralına göre yapılandırmayı seçer (Kaggle testi ve Münih TEST kullanılmaz).

    Parametreler:
        table: duzeltme_denemeleri.csv.

    Döndürür:
        Seçilen deneme adı (koşulu sağlayan yoksa "mevcut").
    """
    base = table.loc[table["deneme"] == "mevcut", "cv_goruntu_makro_f1_tum"].iat[0]
    ok = table[table["cv_goruntu_makro_f1_tum"] >= base - MAX_KAGGLE_CV_DROP]
    return ok.sort_values(["ayar_dengeli_dogruluk", "cv_goruntu_makro_f1_tum"], ascending=False)["deneme"].iat[0]


def experiment_bundle(name: str):
    """
    Bir denemenin kayıtlı modelini yükler ("mevcut" ve "a_olcek": varsayılan kalibre model).

    Parametreler:
        name: Deneme adı.

    Döndürür:
        (model, meta).
    """
    from src.classifier import load_model

    if name in ("mevcut", "a_olcek"):
        return load_model()
    return load_model(f"{MODEL_FILES[SPLIT_VERSION]}_{name}_kalibre")


def munich_center_training(variant: str) -> pd.DataFrame:
    """
    Çok merkezli ek deney için Münih AYAR merkez hücrelerini eğitim satırlarına çevirir (MYO → AML, LYT/NGS →
    Healthy). Her Münih görüntüsü ayrı kaynak grubudur (kalibrasyon iç katları için).

    Parametreler:
        variant: Çıkarım varyantı.

    Döndürür:
        Öznitelikler + etiket, bolum="gelistirme", kaynak_grup_id.
    """
    from src.external_validation import extract_dataset, munich_split

    split = munich_split()
    tune = split[split["bolum"] == "ayar"]
    cells, _ = extract_dataset("munich", tune, variant)
    cells = cells[cells["meta_kopya"] == 0]
    center = cells.loc[cells.groupby("id")["meta_merkez_uzaklik"].idxmin()].merge(tune[["id", "beklenen"]], on="id")
    return center.assign(etiket=center["beklenen"], bolum="gelistirme",
                         kaynak_grup_id=["munih_" + i for i in center["id"]], meta_kopya=0)


def run_final() -> None:
    """
    Son değerlendirme: mevcut model ve seçilen yapılandırma için Münih TEST, güvenli ALL-IDB ve Kaggle kilitli test
    tek tabloda (Münih TEST'e burada, yapılandırma başına bir kez bakılır). Ayrıca açıkça etiketli çok merkezli ek
    deney: Münih AYAR eğitime eklenir; bu artık dış doğrulama DEĞİLDİR.
    """
    from src.external_validation import evaluate_allidb, evaluate_munich, munich_metrics, plot_confusion

    out = stage_dir("external")
    table = pd.read_csv(out / "duzeltme_denemeleri.csv")
    chosen = select_config(table)
    print(f"Seçilen yapılandırma (AYAR + geliştirme CV): {chosen}")
    rows = []
    configs = ["mevcut"] + ([chosen] if chosen != "mevcut" else [])
    for name in configs:
        bundle = experiment_bundle(name)
        _, infer, _, infer_variant = experiment_data(name)
        res = evaluate_munich(bundle, name, ("test",), infer_variant)
        res.drop(columns="yol").to_csv(out / f"munih_test_sonuclari_{name}.csv", index=False, encoding="utf-8")
        munich_metrics(res).to_csv(out / f"munih_test_metrikleri_{name}.csv", index=False, encoding="utf-8")
        plot_confusion(res, f"munih_test_karisiklik_{name}", f"Münih TEST — {name} (merkez hücre)")
        idb_res, idb_m = evaluate_allidb(bundle, name, infer_variant)
        idb_res.to_csv(out / f"allidb_goruntu_sonuclari_{name}.csv", index=False, encoding="utf-8")
        idb_m.to_csv(out / f"allidb_metrikleri_{name}.csv", index=False, encoding="utf-8")
        safe = idb_m[(idb_m["alt_kume"] == "Güvenli (birlikte)") & (idb_m["kural"] == "risk Orta/Yüksek")].iloc[0]
        all_imgs = idb_res[idb_res["guvenli"] & (idb_res["etiket"] == "ALL")]
        rows.append({"yapilandirma": name, "tur": "dış doğrulama (Kaggle ile eğitim)",
                     **munich_scores(res, "munih_test_"),
                     "allidb_duyarlilik": safe["duyarlilik"], "allidb_ozgulluk": safe["ozgulluk"], "allidb_f1": safe["f1"],
                     "allidb_ALL_tahmin_ALL_orani": float((all_imgs["goruntu_tahmin"] == "ALL").mean()),
                     **kaggle_test(bundle, infer),
                     **table.loc[table["deneme"] == name, ["cv_hucre_makro_f1", "cv_goruntu_makro_f1_tum",
                                                          "ayar_dengeli_dogruluk"]].iloc[0].to_dict()})
        print(pd.Series(rows[-1]).to_string(), flush=True)

    # Ek deney: çok merkezli eğitim (Münih AYAR eğitimde) — dış doğrulama değil
    for name in configs:
        train, infer, cols, infer_variant = experiment_data(name)
        extra = munich_center_training(infer_variant)
        bundle = final_model(f"{name}_cok_merkezli", train, cols, infer_variant, extra=extra)
        res = evaluate_munich(bundle, f"{name}_cok_merkezli", ("test",), infer_variant)
        res.drop(columns="yol").to_csv(out / f"munih_test_sonuclari_{name}_cok_merkezli.csv", index=False,
                                       encoding="utf-8")
        rows.append({"yapilandirma": f"{name} + Münih AYAR eğitimde", "tur": "EK DENEY: çok merkezli eğitim "
                     "(Münih AYAR eğitimde; dış doğrulama DEĞİL)", **munich_scores(res, "munih_test_"),
                     **kaggle_test(bundle, infer)})
        print(pd.Series(rows[-1]).to_string(), flush=True)
    final = pd.DataFrame(rows)
    final.to_csv(out / "dis_dogrulama_sonuc.csv", index=False, encoding="utf-8")
    print("\n" + final.round(3).to_string(index=False))


PARTS = {"denemeler": lambda: run_experiments(sys.argv[2:] or None), "sonuc": run_final}


def main() -> None:
    """Komut satırı: python -m src.domain_shift <bölüm> [deneme adları]."""
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "denemeler"
    start = time.time()
    PARTS[part]()
    print(f"\nToplam süre: {time.time() - start:.0f} sn")


if __name__ == "__main__":
    main()
