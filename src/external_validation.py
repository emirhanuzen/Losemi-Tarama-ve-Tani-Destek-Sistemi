"""
Adım 8 — Dış doğrulama: ALL-IDB ve Münih AML sitomorfoloji veri seti.

Bölümler (python -m src.external_validation <bölüm> [...]):
- allidb       : Eğitimde hiç görülmemiş ALL-IDB görüntülerinde (allidb_goruntu_durumu.csv, dis_dogrulama_guvenli)
                 uçtan uca pipeline; görüntü düzeyi risk seviyesi dağılımı, anormal/sağlıklı ayrımı, ALL → ALL/AML.
- segmentasyon : ALL-IDB1 .xyc blast merkezleriyle Otsu, K-means ve Watershed'in gerçek etiketle değerlendirmesi.
- munich_bolum : Münih örneklemi (sınıf başına 300, sabit tohum) ve kilitli AYAR (%40) / TEST (%60) ayrımı.
- munich_teshis: Mevcut modelin Münih AYAR kısmındaki hatasının teşhisi (karışıklık, ölçek, segmentasyon, renk,
                 öznitelik kayması).
- allidb2_teshis: ALL-IDB2 Healthy yakın çekimlerinin "Yüksek" risk almasının sebebi (hücre sayısı, görüntü tipi,
                 Kaggle tip × sınıf dengesi, tip kısayolu).
Düzeltme denemeleri ve son değerlendirme src/domain_shift.py içindedir; ikisi de buradaki çıkarım altyapısını
(extract_dataset, predict_cells, image_results) kullanır.

Lisans: ALL-IDB görüntüsü içeren figürler yalnızca data/processed/external/ altına (git dışı) yazılır. Sayılar ve
Münih (CC BY 3.0) görüntülü figürler atıfla outputs/external/ altına yazılır.
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import (
    ABNORMAL_CLASSES,
    ALLIDB_DIR,
    CLASS_NAMES,
    EXTERNAL_PROCESSED_DIR,
    MUNICH_CLASSES,
    MUNICH_DIR,
    MUNICH_EXTRA_CLASSES,
    MUNICH_SAMPLE_PER_CLASS,
    MUNICH_TUNE_FRACTION,
    N_JOBS,
    RANDOM_STATE,
    RISK_LEVELS,
    RISK_NO_CELL_LEVEL,
    SEGMENTATION_METHOD,
    XYC_MATCH_RADIUS_RATIO,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.utils import load_image, stage_dir

FEATURE_DIR = EXTERNAL_PROCESSED_DIR / "ozellik"     # Dış veri öznitelik tabloları (git dışı)
ALLIDB_STATUS_CSV = EXTERNAL_PROCESSED_DIR.parent / "allidb_goruntu_durumu.csv"
MUNICH_SPLIT_NAME = "munich_bolum.csv"
PROGRESS_EVERY = 100
CHUNK = 20
PROBA_COLUMNS = [f"p_{c}" for c in CLASS_NAMES]
RISK_ORDER = RISK_LEVELS + [RISK_NO_CELL_LEVEL]

# Çıkarım varyantları: (renk normalizasyonu, ölçek stratejisi, eğitim artırması). "varsayilan" uçtan uca
# pipeline'ın birebir aynısıdır (src.pipeline.run_pipeline). Diğerleri src/domain_shift.py denemeleridir.
VARIANTS = {
    "varsayilan": {"normalizasyon": "varsayilan", "olcek": "varsayilan"},
    "olcek": {"normalizasyon": "varsayilan", "olcek": "uyarlanmis"},
    "macenko": {"normalizasyon": "macenko", "olcek": "varsayilan"},
}


# ---------------------------------------------------------------------------
# Veri listeleri
# ---------------------------------------------------------------------------

def allidb_images() -> pd.DataFrame:
    """
    ALL-IDB1 ve ALL-IDB2 görüntülerini dış doğrulama durumuyla listeler.

    Döndürür:
        id (dosya adı + set), yol (mutlak), idb_set, etiket (ALL/Healthy), guvenli (eğitimde hiç görülmedi) tablosu.
    """
    from src.allidb_overlap import list_allidb

    idb = list_allidb(ALLIDB_DIR)
    status = pd.read_csv(ALLIDB_STATUS_CSV)[["idb_dosya", "idb_set", "dis_dogrulama_guvenli"]]
    idb = idb.merge(status, on=["idb_dosya", "idb_set"], how="left")
    return pd.DataFrame({
        "id": idb["idb_set"] + "/" + idb["idb_dosya"], "yol": idb["idb_yolu"], "idb_set": idb["idb_set"],
        "etiket": idb["idb_etiket"], "guvenli": idb["dis_dogrulama_guvenli"].fillna(False).astype(bool),
    })


def munich_split(rebuild: bool = False) -> pd.DataFrame:
    """
    Münih örneklemini ve AYAR/TEST ayrımını oluşturur veya kilitli dosyadan okur.

    - Kaynak tespitinde (src.munich_overlap) Kaggle ile eşleşen görüntüler havuzdan çıkarılır.
    - Her MUNICH_CLASSES sınıfından MUNICH_SAMPLE_PER_CLASS görüntü (azsa tamamı) sabit tohumla seçilir; sınıf
      içinde karıştırılıp ilk %MUNICH_TUNE_FRACTION AYAR, kalanı TEST olur.
    - MUNICH_EXTRA_CLASSES (MYB, beklenen sınıf tanımsız) tamamıyla "ek" bölümüne girer; seçimde kullanılmaz.
    - Dosya bir kez yazılır ve yeniden üretilmez (rebuild=True hariç): AYAR/TEST ayrımı kilitlidir.

    Parametreler:
        rebuild: Var olan ayrımın üzerine yaz (yalnızca açıkça istenirse).

    Döndürür:
        sinif, beklenen, id (MUNICH_DIR'e göre göreli yol), yol (mutlak), bolum (ayar/test/ek) tablosu.
    """
    from src.munich_overlap import list_munich

    path = stage_dir("external") / MUNICH_SPLIT_NAME
    if path.exists() and not rebuild:
        split = pd.read_csv(path, keep_default_na=False)
    else:
        pool = list_munich()
        excluded_path = stage_dir("external") / "munich_dislanan.csv"
        if not excluded_path.exists():
            raise FileNotFoundError("Önce kaynak ön kontrolünü çalıştırın: python -m src.munich_overlap")
        excluded = set(pd.read_csv(excluded_path)["munich"])
        pool = pool[~pool["goreli_yol"].isin(excluded)]
        missing = [c for c in MUNICH_CLASSES if c not in set(pool["sinif"])]
        if missing:
            raise FileNotFoundError(f"Münih sınıf klasörleri eksik: {missing} ({MUNICH_DIR})")
        rows = []
        for i, (cls, sub) in enumerate(pool.groupby("sinif", sort=True)):
            if cls not in MUNICH_CLASSES:
                rows.append(sub.assign(bolum="ek"))
                continue
            rng = np.random.default_rng(RANDOM_STATE + i)
            pick = sub.iloc[rng.permutation(len(sub))[:MUNICH_SAMPLE_PER_CLASS]]
            n_tune = round(MUNICH_TUNE_FRACTION * len(pick))
            rows.append(pick.assign(bolum=["ayar"] * n_tune + ["test"] * (len(pick) - n_tune)))
        split = pd.concat(rows).rename(columns={"goreli_yol": "id"})[["sinif", "beklenen", "id", "bolum"]]
        split.to_csv(path, index=False, encoding="utf-8")
        print(f"Münih ayrımı yazıldı: {path}")
    split["yol"] = [str(MUNICH_DIR / p) for p in split["id"]]
    return split


# ---------------------------------------------------------------------------
# Çıkarım (varyant destekli)
# ---------------------------------------------------------------------------

def image_cells(rgb: np.ndarray, variant: str = "varsayilan") -> tuple[list[dict], dict]:
    """
    Bir görüntüde (varyanta göre) ön işleme, segmentasyon, hücre çıkarma ve öznitelik çıkarma yapar.

    "varsayilan" varyantı src.pipeline.run_pipeline ile aynı hesaplamadır (aynı fonksiyonlar, aynı sıra).

    Parametreler:
        rgb: RGB görüntü.
        variant: VARIANTS anahtarı.

    Döndürür:
        (hücre başına öznitelik + meta sözlükleri, görüntü bilgisi sözlüğü).
        Meta: meta_kenara_degiyor, meta_merkez_uzaklik (256 px uzayında, hücre kutusu merkezi), meta_alan,
        meta_eritrosit_capi, meta_yontem, meta_geri_donus.
    """
    from src.features import crop_features
    from src.pipeline import segment_with_fallback
    from src.segmentation import extract_cells

    from src.preprocessing import preprocess

    spec = VARIANTS[variant]
    diameter = None
    if spec["olcek"] == "uyarlanmis":   # Yakın çekim → yayma çözünürlüğü (src.domain_shift.adapt_scale)
        from src.domain_shift import adapt_scale
        rgb, diameter = adapt_scale(rgb)
    # Segmentasyon her varyantta varsayılan ön işlemeyle yapılır; boya normalizasyonu varyantı yalnızca öznitelik
    # çıkarılan renk görüntüsünü değiştirir (Macenko çıktısında eritrositler çekirdekten ayrılamıyordu).
    steps = preprocess(rgb)[1]
    seg = segment_with_fallback(steps, diameter=diameter)
    color = steps["renk_normalize"]
    if spec["normalizasyon"] == "macenko":
        from src.domain_shift import macenko_color
        color = macenko_color(steps)
    cells = extract_cells(color, seg["etiketler"], steps["gecerli_maske"], seg["eritrosit_capi"])
    d = round(seg["eritrosit_capi"], 2)   # Eğitim tablosuyla aynı yuvarlama (pipeline.run_pipeline)
    rows = []
    for c in cells:
        x, y, w, h = c["kutu"]
        rows.append({**crop_features(c["kirpim"], c["maske"], d),
                     "meta_kenara_degiyor": bool(c["kenara_degiyor"]), "meta_alan": int(c["alan"]),
                     "meta_merkez_uzaklik": float(np.hypot(x + w / 2 - 128, y + h / 2 - 128)),
                     "meta_eritrosit_capi": d, "meta_yontem": seg["yontem"], "meta_geri_donus": seg["geri_donus"]})
    info = {"hucre_sayisi": len(cells), "eritrosit_capi": seg["eritrosit_capi"],
            "eritrosit_capi_olculen": seg["eritrosit_capi_olculen"], "eritrosit_guvenilir": seg["eritrosit_tahmini_guvenilir"],
            "eritrosit_bilesen": seg["eritrosit_bilesen"], "tip": seg["tip"], "yontem": seg["yontem"],
            "geri_donus": seg["geri_donus"]}
    return rows, info


def augment_color(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Renk dayanıklılığı denemesi için görüntüye rastgele renk, parlaklık ve kontrast değişimi uygular.

    Lab uzayında: L* çarpımsal (×0.8-1.2, pozlama) ve kontrast (L* ortalaması etrafında ×0.8-1.2); a*/b* toplamsal
    kayma (±15, boya tonu farkı) ve doygunluk (a*/b* ×0.75-1.25, boya yoğunluğu). Aralıklar Kaggle, ALL-IDB ve Münih
    arasındaki zemin/çekirdek renk farklarını kapsayacak genişlikte seçildi (Adım 8 renk teşhisi).

    Parametreler:
        rgb: RGB görüntü.
        rng: Rastgele sayı üreteci.

    Döndürür:
        Değiştirilmiş RGB görüntü.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    mean_l = lab[..., 0].mean()
    lab[..., 0] = (lab[..., 0] - mean_l) * rng.uniform(0.8, 1.2) + mean_l
    lab[..., 0] *= rng.uniform(0.8, 1.2)
    lab[..., 1:] = (lab[..., 1:] - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-15, 15, 2)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def extract_chunk(items: list[tuple[str, str]], variant: str, copies: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Bir parça görüntünün hücre özniteliklerini çıkarır (paralel işçide). Görüntüler tek tek okunur.

    Parametreler:
        items: (id, yol) listesi.
        variant: VARIANTS anahtarı.
        copies: >0 ise her görüntünün ayrıca bu kadar renk artırmalı kopyası işlenir (meta_kopya 1..copies;
            tohum görüntü kimliğinden türetilir, tekrarlanabilir).

    Döndürür:
        (hücre satırları, görüntü satırları); her satırda id ve meta_kopya bulunur.
    """
    cells, images = [], []
    for item_id, path in items:
        rgb = load_image(path)
        for k in range(copies + 1):
            img = rgb
            if k > 0:
                seed = (RANDOM_STATE * 1_000_003 + sum(map(ord, item_id)) * 31 + k) % (2 ** 32)
                img = augment_color(rgb, np.random.default_rng(seed))
            rows, info = image_cells(img, variant)
            cells += [{"id": item_id, "meta_kopya": k, **r} for r in rows]
            images.append({"id": item_id, "meta_kopya": k, **info})
    return cells, images


def extract_dataset(name: str, items: pd.DataFrame, variant: str = "varsayilan", copies: int = 0,
                    force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bir görüntü kümesinin hücre özniteliklerini paralel çıkarır ve CSV'ye önbellekler (git dışı klasör).

    Önbellek, aynı görüntü kümesi için yeniden kullanılır; eksik görüntüler varsa yalnızca onlar işlenir.

    Parametreler:
        name: Küme adı (ör. "allidb", "munich", "kaggle").
        items: id ve yol sütunlu tablo.
        variant: VARIANTS anahtarı.
        copies: Renk artırmalı kopya sayısı (extract_chunk).
        force: Önbelleği yok say.

    Döndürür:
        (hücre tablosu, görüntü tablosu); yalnızca items'taki görüntüler.
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{variant}" + (f"_artirma{copies}" if copies else "")
    cell_path, img_path = FEATURE_DIR / f"{tag}_hucre.csv", FEATURE_DIR / f"{tag}_goruntu.csv"
    cells = pd.read_csv(cell_path) if cell_path.exists() and not force else pd.DataFrame(columns=["id"])
    images = pd.read_csv(img_path) if img_path.exists() and not force else pd.DataFrame(columns=["id"])
    todo = items[~items["id"].isin(images["id"])]
    if len(todo):
        pairs = list(zip(todo["id"], todo["yol"]))
        chunks = [pairs[i:i + CHUNK] for i in range(0, len(pairs), CHUNK)]
        new_c, new_i, start, done = [], [], time.time(), 0
        print(f"  [{tag}] {len(pairs)} görüntü işlenecek", flush=True)
        with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
            for c, (rc, ri) in zip(chunks, par(delayed(extract_chunk)(c, variant, copies) for c in chunks)):
                new_c += rc
                new_i += ri
                done += len(c)
                if done % PROGRESS_EVERY < CHUNK or done == len(pairs):
                    print(f"  [{tag}] {done}/{len(pairs)} ({time.time() - start:.0f} sn)", flush=True)
        cells = pd.concat([cells, pd.DataFrame(new_c)], ignore_index=True) if len(cells) else pd.DataFrame(new_c)
        images = pd.concat([images, pd.DataFrame(new_i)], ignore_index=True) if len(images) else pd.DataFrame(new_i)
        cells.to_csv(cell_path, index=False, encoding="utf-8")
        images.to_csv(img_path, index=False, encoding="utf-8")
    keep = set(items["id"])
    return cells[cells["id"].isin(keep)].reset_index(drop=True), images[images["id"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tahmin ve görüntü düzeyi sonuçlar
# ---------------------------------------------------------------------------

def predict_cells(bundle: tuple, cells: pd.DataFrame) -> np.ndarray:
    """
    Kayıtlı bir modelle hücre olasılıklarını hesaplar.

    Parametreler:
        bundle: (model, meta) — classifier.load_model çıktısı; meta["oznitelikler"] kullanılır.
        cells: Öznitelik tablosu.

    Döndürür:
        (hücre, 3) olasılık dizisi (CLASS_NAMES sırasında).
    """
    from src.classifier import reorder_proba

    model, meta = bundle
    if cells.empty:
        return np.zeros((0, len(CLASS_NAMES)))
    return reorder_proba(model, model.predict_proba(cells[meta["oznitelikler"]].to_numpy(np.float64)))


def image_results(cells: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame) -> pd.DataFrame:
    """
    Hücre tahminlerinden görüntü başına risk (src.risk_report.compute_risk), görüntü tahmini ve merkez hücre
    tahminini hesaplar.

    - Görüntü tahmini: hücre olasılıklarının ortalamasının en büyüğü (kenara değmeyen hücre varsa yalnızca onlar;
      classifier.aggregate_images ile aynı kural). Hücre yoksa "Hücre yok".
    - Merkez hücre: görüntü merkezine en yakın hücre (Münih görüntüleri tek hücre merkezli kırpımlardır).

    Parametreler:
        cells: Hücre tablosu (id, meta_kenara_degiyor, meta_merkez_uzaklik, meta_yontem, meta_geri_donus).
        proba: Hücre olasılıkları (cells ile aynı sırada).
        images: Görüntü tablosu (id + bilgi sütunları).

    Döndürür:
        images + hucre_sayisi, n_ALL/n_AML/n_Healthy, anormal_oran, risk_seviyesi, guven, baskin_tip_ham,
        goruntu_tahmin, p_* (görüntü), merkez_tahmin, merkez_p_* sütunları.
    """
    from src.classifier import NO_CELL
    from src.risk_report import compute_risk

    df = cells[["id", "meta_kenara_degiyor", "meta_merkez_uzaklik", "meta_yontem", "meta_geri_donus"]].copy()
    df[PROBA_COLUMNS] = proba
    df["sinif"] = np.asarray(CLASS_NAMES)[proba.argmax(axis=1)] if len(proba) else []
    df["kenara_degiyor"] = df["meta_kenara_degiyor"].astype(bool)
    rows = []
    for _, img in images.iterrows():
        sub = df[df["id"] == img["id"]]
        method = sub["meta_yontem"].iat[0] if len(sub) else SEGMENTATION_METHOD
        fallback = bool(sub["meta_geri_donus"].iat[0]) if len(sub) else False
        risk = compute_risk(sub, method, fallback)
        row = {"id": img["id"],
               **{f"n_{c}": risk["hucre_sayilari"][c] for c in CLASS_NAMES},
               "anormal_oran": risk["anormal_oran"], "risk_seviyesi": risk["risk_seviyesi"], "guven": risk["guven"],
               "baskin_tip_ham": risk["baskin_tip_ham"]}
        if len(sub):
            inner = sub[~sub["kenara_degiyor"]]
            use = inner if len(inner) else sub
            mean = use[PROBA_COLUMNS].mean()
            center = sub.loc[sub["meta_merkez_uzaklik"].idxmin()]
            row.update({"goruntu_tahmin": CLASS_NAMES[int(np.argmax(mean.to_numpy()))], **mean.to_dict(),
                        "merkez_tahmin": center["sinif"], **{f"merkez_{c}": center[c] for c in PROBA_COLUMNS},
                        "merkez_uzaklik": center["meta_merkez_uzaklik"]})
        else:
            row.update({"goruntu_tahmin": NO_CELL, "merkez_tahmin": NO_CELL})
        rows.append(row)
    return images.merge(pd.DataFrame(rows), on="id")


def binary_metrics(abnormal_true: np.ndarray, abnormal_pred: np.ndarray) -> dict:
    """
    Anormal (pozitif) / sağlıklı (negatif) ikili ayrım ölçüleri.

    Parametreler:
        abnormal_true: Gerçek anormal mi (bool).
        abnormal_pred: Anormal tahmin edildi mi (bool).

    Döndürür:
        {n, tp, fn, tn, fp, duyarlilik, ozgulluk, kesinlik, f1, dogruluk}; tanımsız oranlar NaN.
    """
    t, p = np.asarray(abnormal_true, bool), np.asarray(abnormal_pred, bool)
    tp, fn, tn, fp = int((t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum()), int((~t & p).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    return {"n": len(t), "tp": tp, "fn": fn, "tn": tn, "fp": fp, "duyarlilik": rec, "ozgulluk": div(tn, tn + fp),
            "kesinlik": prec, "f1": f1, "dogruluk": div(tp + tn, len(t))}


# ---------------------------------------------------------------------------
# 1) ALL-IDB
# ---------------------------------------------------------------------------

def allidb_metrics(results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    ALL-IDB görüntü sonuçlarından alt küme başına ölçüleri hesaplar.

    Pozitif = ALL hastası görüntüsü. İki karar kuralı: risk seviyesi Orta/Yüksek (tarama kuralı) ve görüntü
    tahmini ALL/AML. Hücresiz ("Değerlendirilemedi") görüntü pozitif sayılmaz; ayrıca raporlanır.

    Parametreler:
        results: image_results çıktısı + etiket, idb_set, guvenli.
        model_name: Tablodaki model etiketi.

    Döndürür:
        Alt küme × kural başına ölçü tablosu (risk dağılımı ve ALL → ALL/AML sayıları dahil).
    """
    rows = []
    subsets = [("IDB1 güvenli", results["guvenli"] & (results["idb_set"] == "IDB1")),
               ("IDB2 güvenli", results["guvenli"] & (results["idb_set"] == "IDB2")),
               ("Güvenli (birlikte)", results["guvenli"]),
               ("Eğitimde görülmüş (dış doğrulama DEĞİL)", ~results["guvenli"])]
    for title, sel in subsets:
        sub = results[sel]
        truth = sub["etiket"].eq("ALL").to_numpy()
        for rule, pred in [("risk Orta/Yüksek", sub["risk_seviyesi"].isin(["Orta", "Yüksek"])),
                           ("görüntü tahmini ALL/AML", sub["goruntu_tahmin"].isin(ABNORMAL_CLASSES))]:
            m = binary_metrics(truth, pred.to_numpy())
            all_imgs = sub[truth]
            rows.append({
                "model": model_name, "alt_kume": title, "kural": rule, **m,
                **{f"ALL_risk_{lv}": int((all_imgs["risk_seviyesi"] == lv).sum()) for lv in RISK_ORDER},
                **{f"Healthy_risk_{lv}": int((sub[~truth]["risk_seviyesi"] == lv).sum()) for lv in RISK_ORDER},
                "ALL_tahmin_ALL": int((all_imgs["goruntu_tahmin"] == "ALL").sum()),
                "ALL_tahmin_AML": int((all_imgs["goruntu_tahmin"] == "AML").sum()),
                "ALL_baskin_ALL": int((all_imgs["baskin_tip_ham"] == "ALL").sum()),
                "ALL_baskin_AML": int((all_imgs["baskin_tip_ham"] == "AML").sum()),
                "hucresiz": int((sub["hucre_sayisi"] == 0).sum()),
            })
    return pd.DataFrame(rows)


def evaluate_allidb(bundle: tuple, model_name: str, variant: str = "varsayilan") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tüm ALL-IDB görüntülerini (güvenli ve görülmüş) bir modelle değerlendirir.

    Parametreler:
        bundle: (model, meta).
        model_name: Çıktı etiketi.
        variant: Çıkarım varyantı.

    Döndürür:
        (görüntü sonuçları, ölçü tablosu).
    """
    items = allidb_images()
    cells, images = extract_dataset("allidb", items, variant)
    cells, images = cells[cells["meta_kopya"] == 0], images[images["meta_kopya"] == 0]
    results = image_results(cells.reset_index(drop=True), predict_cells(bundle, cells), images)
    results = items[["id", "idb_set", "etiket", "guvenli"]].merge(results, on="id")
    return results, allidb_metrics(results, model_name)


def run_allidb() -> None:
    """Mevcut (kalibre) modeli ALL-IDB'de değerlendirir ve sonuçları outputs/external/ altına yazar."""
    from src.classifier import load_model

    out = stage_dir("external")
    results, metrics = evaluate_allidb(load_model(), "mevcut")
    results.to_csv(out / "allidb_goruntu_sonuclari_mevcut.csv", index=False, encoding="utf-8")
    metrics.to_csv(out / "allidb_metrikleri_mevcut.csv", index=False, encoding="utf-8")
    cols = ["alt_kume", "kural", "n", "tp", "fn", "tn", "fp", "duyarlilik", "ozgulluk", "f1", "hucresiz"]
    print(metrics[cols].round(3).to_string(index=False))
    risk_cols = ["alt_kume"] + [c for c in metrics.columns if "_risk_" in c] + \
        ["ALL_tahmin_ALL", "ALL_tahmin_AML", "ALL_baskin_ALL", "ALL_baskin_AML"]
    print(metrics[metrics["kural"] == "risk Orta/Yüksek"][risk_cols].to_string(index=False))
    plot_allidb_levels(results)


def plot_allidb_levels(results: pd.DataFrame) -> None:
    """
    Güvenli ALL-IDB görüntülerinde gerçek sınıfa göre risk seviyesi dağılımını çizer (yalnızca sayılar; görüntü yok).

    Parametreler:
        results: Görüntü sonuçları.
    """
    from src.utils import save_figure

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    colors = {"Düşük": "#1baf7a", "Orta": "#e08a00", "Yüksek": "#c0392b", RISK_NO_CELL_LEVEL: "#9a9890"}
    for ax, subset in zip(axes, ["IDB1", "IDB2"]):
        sub = results[results["guvenli"] & (results["idb_set"] == subset)]
        table = pd.crosstab(sub["etiket"], sub["risk_seviyesi"]).reindex(columns=RISK_ORDER, fill_value=0) \
            .reindex(["ALL", "Healthy"], fill_value=0)
        left = np.zeros(len(table))
        for lv in RISK_ORDER:
            ax.barh(table.index, table[lv], left=left, color=colors[lv], label=lv)
            for y, (l, v) in enumerate(zip(left, table[lv])):
                if v:
                    ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=8, color="white")
            left += table[lv].to_numpy()
        ax.set_title(f"ALL-IDB{subset[-1]} (güvenli, n={len(sub)})", fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Görüntü sayısı", fontsize=8, color=TEXT_SECONDARY)
        ax.invert_yaxis()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "external", "allidb_risk_seviyeleri")


# ---------------------------------------------------------------------------
# 1b) Segmentasyonun ALL-IDB1 .xyc blast merkezleriyle değerlendirilmesi
# ---------------------------------------------------------------------------

def read_xyc(path: Path) -> np.ndarray:
    """
    ALL-IDB1 .xyc dosyasındaki blast merkezlerini okur (sağlıklı görüntülerde dosya boştur).

    Parametreler:
        path: .xyc yolu.

    Döndürür:
        (N, 2) float (x, y) orijinal piksel koordinatları.
    """
    text = path.read_text(encoding="latin-1").strip() if path.exists() else ""
    if not text:   # Sağlıklı görüntülerin dosyası boş veya yalnızca boşluk içerir
        return np.zeros((0, 2))
    return np.array([[float(v) for v in line.split()[:2]] for line in text.splitlines() if line.strip()])


def to_256(points: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Orijinal koordinatları preprocessing.resize_with_padding'in 256x256 uzayına çevirir (ölçek + ortalama payı).

    Parametreler:
        points: (N, 2) x, y.
        shape: Orijinal görüntü şekli.

    Döndürür:
        (N, 2) 256 px uzayında koordinatlar.
    """
    from src.config import IMAGE_SIZE

    h, w = shape[:2]
    s = IMAGE_SIZE / max(h, w)
    new_w, new_h = max(1, round(w * s)), max(1, round(h * s))
    left, top = (IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2
    return points * s + [left, top] if len(points) else points


def match_blasts(labels: np.ndarray, centers: np.ndarray, radius: float) -> list[int]:
    """
    Her blast merkezini, merkezde veya radius içinde en yakın etiketli çekirdeğe atar.

    Parametreler:
        labels: Etiketli çekirdek maskesi (256 px).
        centers: (N, 2) blast merkezleri (256 px).
        radius: Eşleşme yarıçapı (px).

    Döndürür:
        Her blast için çekirdek etiketi (bulunamadıysa 0).
    """
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for x, y in centers:
        xi, yi = int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1)))
        if labels[yi, xi] > 0:
            out.append(int(labels[yi, xi]))
            continue
        near = ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) & (labels > 0)
        if not near.any():
            out.append(0)
            continue
        d = np.where(near, (xx - x) ** 2 + (yy - y) ** 2, np.inf)
        out.append(int(labels.flat[int(np.argmin(d))]))
    return out


def xyc_chunk(items: list[tuple[str, str, str]]) -> list[dict]:
    """
    Bir parça ALL-IDB1 görüntüsünde üç segmentasyon yöntemini blast merkezleriyle karşılaştırır (paralel işçide).

    Parametreler:
        items: (id, görüntü yolu, xyc yolu) listesi.

    Döndürür:
        Görüntü × yöntem başına {id, yontem, blast, tespit, birlesik_blast, cekirdek, eritrosit_capi}.
    """
    from src.preprocessing import preprocess
    from src.segmentation import METHODS, image_rbc_diameter

    rows = []
    for item_id, path, xyc in items:
        rgb = load_image(path)
        _, steps = preprocess(rgb)
        diameter, _ = image_rbc_diameter(steps)
        centers = to_256(read_xyc(Path(xyc)), rgb.shape)
        for method, func in METHODS.items():
            labels = func(steps, diameter)
            hit = match_blasts(labels, centers, XYC_MATCH_RADIUS_RATIO * diameter)
            found = [h for h in hit if h > 0]
            counts = pd.Series(found).value_counts() if found else pd.Series(dtype=int)
            rows.append({"id": item_id, "yontem": method, "blast": len(centers), "tespit": len(found),
                         "birlesik_blast": int(sum(counts[counts >= 2])) if found else 0,
                         "blastli_cekirdek": int(len(counts)), "cekirdek": int(len(np.unique(labels)) - 1),
                         "eritrosit_capi": diameter})
    return rows


def run_segmentation_check() -> None:
    """108 ALL-IDB1 görüntüsünde üç yöntemin blast tespit duyarlılığını ve birleşik çekirdek oranını hesaplar."""
    out = stage_dir("external")
    items = allidb_images()
    items = items[items["idb_set"] == "IDB1"]
    xyc_dir = Path(items["yol"].iat[0]).parent.parent / "xyc"
    jobs = [(r["id"], r["yol"], str(xyc_dir / (Path(r["yol"]).stem + ".xyc"))) for _, r in items.iterrows()]
    chunks = [jobs[i:i + 6] for i in range(0, len(jobs), 6)]
    rows, start = [], time.time()
    with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
        for res in par(delayed(xyc_chunk)(c) for c in chunks):
            rows += res
            print(f"  [xyc] {len(rows) // 3}/{len(jobs)} ({time.time() - start:.0f} sn)", flush=True)
    per = pd.DataFrame(rows).merge(items[["id", "etiket", "guvenli"]], on="id")
    per.to_csv(out / "segmentasyon_xyc_goruntu.csv", index=False, encoding="utf-8")
    summary = []
    for (method, label), sub in per.groupby(["yontem", "etiket"]):
        blasts = sub["blast"].sum()
        summary.append({
            "yontem": method, "etiket": label, "goruntu": len(sub), "blast": int(blasts),
            "blast_duyarlilik": sub["tespit"].sum() / blasts if blasts else np.nan,
            "birlesik_blast_orani": sub["birlesik_blast"].sum() / sub["tespit"].sum() if sub["tespit"].sum() else np.nan,
            "goruntu_basina_cekirdek": sub["cekirdek"].mean(),
            "blastsiz_cekirdek_orani": 1 - sub["blastli_cekirdek"].sum() / sub["cekirdek"].sum()
            if sub["cekirdek"].sum() and blasts else np.nan,
            "hic_cekirdek_yok": int((sub["cekirdek"] == 0).sum()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "segmentasyon_xyc_ozet.csv", index=False, encoding="utf-8")
    print(summary.round(3).to_string(index=False))
    plot_xyc_examples(per)


def plot_xyc_examples(per: pd.DataFrame, n: int = 3) -> None:
    """
    Blast merkezleri ve üç yöntemin konturları örnek figürü (ALL-IDB görüntüsü içerir → data/processed/external/).

    Parametreler:
        per: Görüntü × yöntem sonuçları.
        n: Örnek görüntü sayısı (en çok blastlı ALL görüntüleri arasından).
    """
    from skimage.segmentation import find_boundaries

    from src.preprocessing import preprocess
    from src.segmentation import METHODS, image_rbc_diameter

    items = allidb_images().set_index("id")
    pick = per[per["etiket"] == "ALL"].groupby("id")["blast"].first().sort_values(ascending=False).index[:n]
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.4 * n), facecolor=SURFACE)
    for r, item_id in enumerate(pick):
        path = items.loc[item_id, "yol"]
        rgb = load_image(path)
        _, steps = preprocess(rgb)
        diameter, _ = image_rbc_diameter(steps)
        centers = to_256(read_xyc(Path(path).parent.parent / "xyc" / (Path(path).stem + ".xyc")), rgb.shape)
        for c, (method, func) in enumerate(METHODS.items()):
            labels = func(steps, diameter)
            img = steps["renk_normalize"].copy()
            img[find_boundaries(labels, mode="inner")] = (255, 214, 0)
            ax = axes[r, c]
            ax.imshow(img)
            ax.scatter(centers[:, 0], centers[:, 1], s=40, facecolors="none", edgecolors="#00e5ff", linewidths=1.2)
            row = per[(per["id"] == item_id) & (per["yontem"] == method)].iloc[0]
            ax.set_title(f"{item_id} — {method}: {row['tespit']}/{row['blast']} blast, "
                         f"{row['birlesik_blast']} birleşik", fontsize=8, color=TEXT_PRIMARY, loc="left")
            ax.axis("off")
    fig.suptitle("ALL-IDB1 blast merkezleri (camgöbeği daire) ve çekirdek konturları (sarı)", fontsize=10,
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    EXTERNAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXTERNAL_PROCESSED_DIR / "segmentasyon_xyc_ornekleri.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-3) Münih: değerlendirme ve teşhis
# ---------------------------------------------------------------------------

def evaluate_munich(bundle: tuple, model_name: str, parts: tuple = ("ayar",), variant: str = "varsayilan") -> pd.DataFrame:
    """
    Münih ayrımının istenen bölümlerini bir modelle değerlendirir (görüntü başına sonuç).

    Parametreler:
        bundle: (model, meta).
        model_name: Çıktı etiketi.
        parts: Bölümler ("ayar", "test", "ek"). TEST yalnızca son değerlendirmede istenir.
        variant: Çıkarım varyantı.

    Döndürür:
        sinif, beklenen, bolum + image_results sütunları + model.
    """
    split = munich_split()
    items = split[split["bolum"].isin(parts)]
    cells, images = extract_dataset("munich", items, variant)
    cells, images = cells[cells["meta_kopya"] == 0].reset_index(drop=True), images[images["meta_kopya"] == 0]
    results = image_results(cells, predict_cells(bundle, cells), images)
    return items[["id", "yol", "sinif", "beklenen", "bolum"]].merge(results, on="id").assign(model=model_name)


def munich_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Münih sonuçlarından sınıf başına ve genel ölçüleri hesaplar (esas: merkez hücre tahmini).

    Parametreler:
        results: evaluate_munich çıktısı (tek model, tek bölüm grubu).

    Döndürür:
        Satır başına bir sınıf (+ "MYO vs normal" ikili ve "dengeli doğruluk" satırları): n, tahmin sayıları,
        duyarlilik (merkez hücre beklenen sınıfta), tüm hücrelerde beklenen sınıf oranı, risk dağılımı.
    """
    from src.classifier import NO_CELL

    rows = []
    for cls, sub in results.groupby("sinif", sort=False):
        exp = sub["beklenen"].iat[0]
        row = {"sinif": cls, "beklenen": exp or "-", "n": len(sub),
               **{f"merkez_{c}": int((sub["merkez_tahmin"] == c).sum()) for c in CLASS_NAMES + [NO_CELL]},
               "duyarlilik": float((sub["merkez_tahmin"] == exp).mean()) if exp else np.nan,
               "tum_hucrelerde_beklenen_orani": float(sub[f"n_{exp}"].sum() / max(sub["hucre_sayisi"].sum(), 1))
               if exp else np.nan,
               **{f"risk_{lv}": int((sub["risk_seviyesi"] == lv).sum()) for lv in RISK_ORDER}}
        rows.append(row)
    scored = results[results["beklenen"] != ""]
    if len(scored):
        truth = scored["beklenen"].eq("AML").to_numpy()
        m = binary_metrics(truth, scored["merkez_tahmin"].isin(ABNORMAL_CLASSES).to_numpy())
        rows.append({"sinif": "MYO vs LYT+NGS (merkez hücre)", "beklenen": "anormal/normal", "n": m["n"],
                     "duyarlilik": m["duyarlilik"], "ozgulluk": m["ozgulluk"], "f1": m["f1"]})
        sens = [r["duyarlilik"] for r in rows if r["sinif"] in MUNICH_CLASSES]
        rows.append({"sinif": "Dengeli doğruluk (MYO, LYT, NGS)", "n": len(scored), "duyarlilik": float(np.mean(sens))})
    return pd.DataFrame(rows)


def kaggle_dev_features() -> pd.DataFrame:
    """
    Kaggle geliştirme seti hücre özniteliklerini (K-means, v2 ayrımı) okur. Teşhiste test seti kullanılmaz.

    Döndürür:
        Öznitelik tablosu (etiket, meta_tip, kaynak_grup_id, kat dahil).
    """
    from src.classifier import load_cells

    cells = load_cells(SEGMENTATION_METHOD)
    return cells[cells["bolum"] == "gelistirme"].reset_index(drop=True)


def subset_model_check(dev: pd.DataFrame, munich_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Öznitelik grubu teşhisi: yalnız şekil / renk / doku (ve mutlak renk hariç) ile Kaggle geliştirme setinde
    eğitilen modellerin Münih AYAR merkez hücrelerindeki tahminleri.

    Hangi grup Münih normal hücrelerini AML'ye itiyor sorusunu doğrudan yanıtlar.

    Parametreler:
        dev: Kaggle geliştirme hücreleri.
        munich_cells: Münih AYAR merkez hücreleri (öznitelikler + sinif).

    Döndürür:
        Grup × Münih sınıfı başına tahmin oranları tablosu.
    """
    from threadpoolctl import threadpool_limits

    from src.classifier import fit_model, reorder_proba
    from src.config import CLASSIFIER_MODEL, CLASSIFIER_PARAMS
    from src.utils import feature_columns

    cols = feature_columns(dev)
    groups = {
        "tümü": cols, "yalnız şekil": [c for c in cols if c.startswith("sekil_")],
        "yalnız renk": [c for c in cols if c.startswith("renk_")], "yalnız doku": [c for c in cols if c.startswith("doku_")],
        "şekil + doku + renk farkı": [c for c in cols if c.startswith(("sekil_", "doku_", "renk_fark_"))],
    }
    rows = []
    with threadpool_limits(limits=N_JOBS):
        for name, feats in groups.items():
            model = fit_model(dev[feats].to_numpy(np.float64), dev["etiket"].to_numpy(), None, CLASSIFIER_MODEL,
                              CLASSIFIER_PARAMS)
            pred = np.asarray(CLASS_NAMES)[reorder_proba(model, model.predict_proba(
                munich_cells[feats].to_numpy(np.float64))).argmax(axis=1)]
            for cls in munich_cells["sinif"].unique():
                sel = munich_cells["sinif"].to_numpy() == cls
                rows.append({"oznitelik_grubu": name, "oznitelik_sayisi": len(feats), "sinif": cls, "n": int(sel.sum()),
                             **{f"tahmin_{c}": float((pred[sel] == c).mean()) for c in CLASS_NAMES}})
    return pd.DataFrame(rows)


def type_shortcut_check(dev: pd.DataFrame, munich_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Görüntü tipi kısayolu teşhisi: Kaggle hücrelerinden "yayma mı yakın çekim mi" tahmin eden bir model kurulur ve
    Münih hücrelerine uygulanır. Kaggle'da yakın çekimlerin neredeyse tamamı ALL (ALL-IDB2), Healthy'de hiç yakın
    çekim yok; Münih hücreleri "yakın çekim" görünüyorsa model onları Healthy'den uzak tutar.

    Parametreler:
        dev: Kaggle geliştirme hücreleri (meta_tip ile).
        munich_cells: Münih merkez hücreleri.

    Döndürür:
        Kaggle tip × sınıf dağılımı ve Münih sınıfı başına ortalama P(yakın çekim) tablosu.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    from src.utils import feature_columns

    cols = feature_columns(dev)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(dev[cols].to_numpy(np.float64), (dev["meta_tip"] == "yakin_cekim").to_numpy())
    p = clf.predict_proba(munich_cells[cols].to_numpy(np.float64))[:, 1]
    rows = [{"grup": f"Kaggle {t} {e}", "hucre": int(n)} for (t, e), n in dev.groupby(["meta_tip", "etiket"]).size().items()]
    for cls in munich_cells["sinif"].unique():
        sel = munich_cells["sinif"].to_numpy() == cls
        rows.append({"grup": f"Münih {cls}", "hucre": int(sel.sum()), "ort_p_yakin_cekim": float(p[sel].mean()),
                     "yakin_cekim_orani": float((p[sel] >= 0.5).mean())})
    return pd.DataFrame(rows)


def feature_shift(dev: pd.DataFrame, munich_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Öznitelik kayması: her öznitelikte Münih normal (LYT, NGS) ve MYO medyanlarının Kaggle Healthy ile Kaggle AML
    medyanları arasındaki konumu.

    konum = (medyan_Münih − medyan_Kaggle_Healthy) / (medyan_Kaggle_AML − medyan_Kaggle_Healthy):
    0 → Kaggle Healthy gibi, 1 → Kaggle AML gibi, > 1 veya < 0 → iki Kaggle sınıfının da dışında.

    Parametreler:
        dev: Kaggle geliştirme hücreleri.
        munich_cells: Münih merkez hücreleri.

    Döndürür:
        Öznitelik başına medyanlar ve konumlar (ANOVA F sırasıyla).
    """
    from src.utils import feature_columns

    importance = pd.read_csv(stage_dir("features") / "oznitelik_onemi.csv")[["oznitelik", "f_sirasi"]]
    rows = []
    for col in feature_columns(dev):
        med = {f"Kaggle_{c}": dev.loc[dev["etiket"] == c, col].median() for c in CLASS_NAMES}
        med["Kaggle_ALL_yakin_cekim"] = dev.loc[(dev["etiket"] == "ALL") & (dev["meta_tip"] == "yakin_cekim"), col].median()
        for cls in ("LYT", "NGS", "MYO"):
            med[f"Münih_{cls}"] = munich_cells.loc[munich_cells["sinif"] == cls, col].median()
        span = med["Kaggle_AML"] - med["Kaggle_Healthy"]
        pos = {f"konum_{cls}": (med[f"Münih_{cls}"] - med["Kaggle_Healthy"]) / span if abs(span) > 1e-9 else np.nan
               for cls in ("LYT", "NGS", "MYO")}
        rows.append({"oznitelik": col, **med, **pos})
    return pd.DataFrame(rows).merge(importance, on="oznitelik", how="left").sort_values("f_sirasi")


def color_sample() -> pd.DataFrame:
    """
    Renk teşhisi için görüntü örneklemi: Kaggle geliştirme (sınıf başına 50), güvenli ALL-IDB (IDB1 ve IDB2'den
    30'ar), Münih AYAR (sınıf başına 50). Sabit tohum.

    Döndürür:
        veri_seti, grup, id, yol tablosu.
    """
    from src.classifier import load_split
    from src.utils import resolve_path

    rng = np.random.default_rng(RANDOM_STATE)
    split = load_split()
    dev = split[split["bolum"] == "gelistirme"]
    parts = []
    for c in CLASS_NAMES:
        sub = dev[dev["etiket"] == c]
        pick = sub.iloc[rng.choice(len(sub), 50, replace=False)]
        parts.append(pd.DataFrame({"veri_seti": "Kaggle", "grup": f"Kaggle {c}", "id": pick["dosya_yolu"],
                                   "yol": [str(resolve_path(p)) for p in pick["dosya_yolu"]]}))
    idb = allidb_images()
    for s in ("IDB1", "IDB2"):
        sub = idb[idb["guvenli"] & (idb["idb_set"] == s)]
        pick = sub.iloc[rng.choice(len(sub), min(30, len(sub)), replace=False)]
        parts.append(pd.DataFrame({"veri_seti": "ALL-IDB", "grup": f"ALL-IDB{s[-1]}", "id": pick["id"], "yol": pick["yol"]}))
    mu = munich_split()
    for cls in MUNICH_CLASSES:
        sub = mu[(mu["bolum"] == "ayar") & (mu["sinif"] == cls)]
        pick = sub.iloc[rng.choice(len(sub), min(50, len(sub)), replace=False)]
        parts.append(pd.DataFrame({"veri_seti": "Münih", "grup": f"Münih {cls}", "id": pick["id"], "yol": pick["yol"]}))
    return pd.concat(parts, ignore_index=True)


def lab_true(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Maskedeki piksellerin gerçek CIE Lab medyanını (L* 0-100, a*/b* işaretli) döndürür.

    Parametreler:
        rgb: RGB görüntü.
        mask: bool maske.

    Döndürür:
        (3,) medyan [L*, a*, b*]; maske boşsa NaN.
    """
    if not mask.any():
        return np.full(3, np.nan)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float64)
    return np.median(lab, axis=0) * [100 / 255, 1, 1] - [0, 128, 128]


def color_chunk(items: list[tuple[str, str]]) -> list[dict]:
    """
    Görüntü başına normalizasyon öncesi ve sonrası zemin ve çekirdek Lab medyanlarını hesaplar (paralel işçide).

    Zemin: geçerli pikseller − (çekirdek maskesi 5 px genişletilmiş). Çekirdek: pipeline segmentasyonu.

    Parametreler:
        items: (id, yol) listesi.

    Döndürür:
        Görüntü başına {id, once_/sonra_ × zemin_/cekirdek_ × L/a/b}.
    """
    from src.pipeline import segment_with_fallback
    from src.preprocessing import preprocess

    rows = []
    for item_id, path in items:
        _, steps = preprocess(load_image(path))
        seg = segment_with_fallback(steps)
        nuc = seg["maske"]
        bg = steps["gecerli_maske"] & ~cv2.dilate(nuc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        row = {"id": item_id}
        for when, img in (("once", steps["boyutlandirilmis"]), ("sonra", steps["renk_normalize"])):
            for region, m in (("zemin", bg), ("cekirdek", nuc)):
                row.update(dict(zip([f"{when}_{region}_{k}" for k in "Lab"], lab_true(img, m))))
        rows.append(row)
    return rows


def plot_color(colors: pd.DataFrame) -> None:
    """
    Normalizasyon öncesi/sonrası zemin ve çekirdek a*-b* dağılımlarını veri setlerine göre çizer (yalnızca sayılar).

    Parametreler:
        colors: color_chunk sonuçları + grup.
    """
    from src.utils import save_figure

    palette = {"Kaggle ALL": CLASS_COLORS["ALL"], "Kaggle AML": CLASS_COLORS["AML"],
               "Kaggle Healthy": CLASS_COLORS["Healthy"], "ALL-IDB1": "#7a4fbf", "ALL-IDB2": "#c86fd6",
               "Münih MYO": "#8a1c1c", "Münih LYT": "#3a3a3a", "Münih NGS": "#9a9890"}
    markers = {"Kaggle": "o", "ALL-IDB": "s", "Münih": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), facecolor=SURFACE)
    for r, when in enumerate(("once", "sonra")):
        for c, region in enumerate(("zemin", "cekirdek")):
            ax = axes[r, c]
            for grp, sub in colors.groupby("grup"):
                ax.scatter(sub[f"{when}_{region}_a"], sub[f"{when}_{region}_b"], s=12, alpha=0.7,
                           color=palette.get(grp, "#888"), marker=markers[sub["veri_seti"].iat[0]], label=grp)
            ax.set_title(f"{'Normalizasyon öncesi' if when == 'once' else 'Normalizasyon sonrası'} — "
                         f"{'zemin' if region == 'zemin' else 'çekirdek'}", fontsize=10, color=TEXT_PRIMARY, loc="left")
            ax.set_xlabel("a* (medyan)", fontsize=8, color=TEXT_SECONDARY)
            ax.set_ylabel("b* (medyan)", fontsize=8, color=TEXT_SECONDARY)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "external", "renk_dagilimi_kaggle_allidb_munih")


def plot_feature_shift(dev: pd.DataFrame, munich_cells: pd.DataFrame, features: list[str]) -> None:
    """
    Seçilen özniteliklerin Kaggle (Healthy, AML, ALL yakın çekim) ve Münih (LYT, NGS, MYO) dağılımlarını kutu
    grafiğiyle çizer.

    Parametreler:
        dev: Kaggle geliştirme hücreleri.
        munich_cells: Münih merkez hücreleri.
        features: Öznitelik sütunları (en fazla 15).
    """
    from app.ortak import feature_name_tr
    from src.utils import save_figure

    groups = [("Kaggle Healthy", dev[dev["etiket"] == "Healthy"], CLASS_COLORS["Healthy"]),
              ("Kaggle AML", dev[dev["etiket"] == "AML"], CLASS_COLORS["AML"]),
              ("Kaggle ALL yakın", dev[(dev["etiket"] == "ALL") & (dev["meta_tip"] == "yakin_cekim")], CLASS_COLORS["ALL"]),
              ("Münih LYT", munich_cells[munich_cells["sinif"] == "LYT"], "#3a3a3a"),
              ("Münih NGS", munich_cells[munich_cells["sinif"] == "NGS"], "#9a9890"),
              ("Münih MYO", munich_cells[munich_cells["sinif"] == "MYO"], "#8a1c1c")]
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5), facecolor=SURFACE)
    for ax, col in zip(axes.ravel(), features):
        data = [g[col].dropna().to_numpy() for _, g, _ in groups]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, (_, _, color) in zip(bp["boxes"], groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("white")
        ax.set_xticks(range(1, len(groups) + 1), [g for g, _, _ in groups], rotation=60, fontsize=6.5)
        ax.set_title(feature_name_tr(col), fontsize=8, color=TEXT_PRIMARY, loc="left")
        ax.tick_params(axis="y", labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("En ayırt edici 15 öznitelik: Kaggle ve Münih (merkez hücre) dağılımları", fontsize=11,
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    save_figure(fig, "external", "oznitelik_kaymasi")


def plot_munich_examples(results: pd.DataFrame, per_class: int = 2) -> None:
    """
    Münih örneklerinde ölçek ve segmentasyonun elle kontrolü: eritrosit maskesi, tahmini eritrosit çapı (daire),
    çekirdek konturları (merkez hücre kalın) ve tahmin. Münih görüntüleri CC BY 3.0; kaynak figürde belirtilir.

    Parametreler:
        results: Münih AYAR görüntü sonuçları (varsayılan varyant).
        per_class: Sınıf başına örnek.
    """
    from skimage.segmentation import find_boundaries

    from src.pipeline import segment_with_fallback
    from src.preprocessing import preprocess
    from src.segmentation import estimate_rbc_diameter
    from src.utils import save_figure

    rng = np.random.default_rng(RANDOM_STATE)
    picks = []
    for cls in MUNICH_CLASSES:
        sub = results[results["sinif"] == cls]
        picks += list(sub.iloc[rng.choice(len(sub), per_class, replace=False)].itertuples())
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), facecolor=SURFACE)
    for c, row in enumerate(picks):
        _, steps = preprocess(load_image(row.yol))
        est = estimate_rbc_diameter(steps["renk_normalize"], steps["gecerli_maske"])
        seg = segment_with_fallback(steps)
        top = steps["renk_normalize"].copy()
        top[find_boundaries(est["maske"], mode="inner")] = (0, 229, 255)
        ax = axes[0, c]
        ax.imshow(top)
        d = seg["eritrosit_capi"]
        ax.add_patch(plt.Circle((128, 128), d / 2, fill=False, color="#ffd600", lw=1.5))
        ax.set_title(f"{row.sinif}: çap {d:.0f} px" + ("" if seg["eritrosit_tahmini_guvenilir"] else " (yedek)"),
                     fontsize=8, color=TEXT_PRIMARY, loc="left")
        ax.axis("off")
        bottom = steps["renk_normalize"].copy()
        bottom[find_boundaries(seg["etiketler"], mode="inner")] = (255, 214, 0)
        ax = axes[1, c]
        ax.imshow(bottom)
        ax.set_title(f"tahmin (merkez): {row.merkez_tahmin}", fontsize=8, color=TEXT_PRIMARY, loc="left")
        ax.axis("off")
    fig.suptitle("Münih örnekleri — üst: eritrosit maskesi (camgöbeği) ve tahmini eritrosit çapı (sarı daire); "
                 "alt: çekirdek konturları\nGörüntüler: Matek ve ark. 2019, AML-Cytomorphology_LMU (TCIA), CC BY 3.0",
                 fontsize=9, color=TEXT_PRIMARY)
    fig.tight_layout()
    save_figure(fig, "external", "munih_olcek_segmentasyon_ornekleri")


def plot_confusion(results: pd.DataFrame, name: str, title: str) -> None:
    """
    Münih sınıfı × merkez hücre tahmini sayım matrisini çizer.

    Parametreler:
        results: Münih görüntü sonuçları.
        name: Dosya adı (outputs/external/<ad>.png).
        title: Başlık.
    """
    from src.classifier import NO_CELL
    from src.utils import save_figure

    rows = [c for c in list(MUNICH_CLASSES) + MUNICH_EXTRA_CLASSES if c in set(results["sinif"])]
    cols = CLASS_NAMES + [NO_CELL]
    table = pd.crosstab(results["sinif"], results["merkez_tahmin"]).reindex(index=rows, columns=cols, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 0.7 * len(rows) + 1.4), facecolor=SURFACE)
    norm = table.div(table.sum(axis=1), axis=0).fillna(0)
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = table.iat[i, j]
            ax.text(j, i, f"{v}\n%{100 * norm.iat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if norm.iat[i, j] > 0.55 else TEXT_PRIMARY)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), [f"{r} (→ {MUNICH_CLASSES.get(r, '?')})" for r in rows], fontsize=8)
    ax.set_xlabel("Merkez hücre tahmini", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=9, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    save_figure(fig, "external", name)


def run_munich_diagnosis() -> None:
    """Mevcut modelin Münih AYAR (ve ek MYB) hatasını teşhis eder; tüm sonuçları outputs/external/ altına yazar."""
    from src.classifier import load_model

    out = stage_dir("external")
    bundle = load_model()
    print("1) Mevcut model, Münih AYAR + ek (MYB)")
    res = evaluate_munich(bundle, "mevcut", ("ayar", "ek"))
    res.drop(columns="yol").to_csv(out / "munih_ayar_sonuclari_mevcut.csv", index=False, encoding="utf-8")
    metrics = munich_metrics(res)
    metrics.to_csv(out / "munih_ayar_metrikleri_mevcut.csv", index=False, encoding="utf-8")
    print(metrics.round(3).to_string(index=False))
    plot_confusion(res, "munih_ayar_karisiklik_mevcut", "Münih AYAR — mevcut model (merkez hücre)")
    tune = res[res["bolum"] == "ayar"]

    print("2) Ölçek: eritrosit çapı dağılımı")
    from src.config import CELLS_INDEX_CSV
    ci = pd.read_csv(CELLS_INDEX_CSV).groupby("kaynak_dosya").agg(
        etiket=("etiket", "first"), tip=("tip", "first"), cap=("eritrosit_capi", "first"),
        guvenilir=("eritrosit_tahmini_guvenilir", "first"))
    rows = [{"grup": f"Kaggle {t}", "n": len(s), "cap_medyan": s["cap"].median(), "cap_p10": s["cap"].quantile(0.1),
             "cap_p90": s["cap"].quantile(0.9), "guvenilir_orani": s["guvenilir"].mean()} for t, s in ci.groupby("tip")]
    for cls, s in res.groupby("sinif"):
        rows.append({"grup": f"Münih {cls}", "n": len(s), "cap_medyan": s["eritrosit_capi"].median(),
                     "cap_p10": s["eritrosit_capi"].quantile(0.1), "cap_p90": s["eritrosit_capi"].quantile(0.9),
                     "guvenilir_orani": s["eritrosit_guvenilir"].astype(bool).mean(),
                     "olculen_medyan": s["eritrosit_capi_olculen"].median(),
                     "tip_yakin_cekim_orani": (s["tip"] == "yakin_cekim").mean()})
    scale = pd.DataFrame(rows)
    # Biyolojik tutarlılık: merkez çekirdeğin eşdeğer çapı / eritrosit çapı (lenfosit çekirdeği ~0.8-1.0 eritrosit)
    split = munich_split()
    mcells, _ = extract_dataset("munich", split[split["bolum"].isin(["ayar", "ek"])])
    mcells = mcells[mcells["meta_kopya"] == 0]
    center = mcells.loc[mcells.groupby("id")["meta_merkez_uzaklik"].idxmin()].merge(split[["id", "sinif", "bolum"]], on="id")
    center["cekirdek_cap_orani"] = np.sqrt(4 * center["sekil_alan"] / np.pi)
    scale = scale.merge(center.groupby("sinif")["cekirdek_cap_orani"].median().rename("cekirdek_cap_orani_medyan")
                        .rename(lambda s: f"Münih {s}").reset_index().rename(columns={"sinif": "grup"}), on="grup", how="left")
    dev = kaggle_dev_features()
    dev["cekirdek_cap_orani"] = np.sqrt(4 * dev["sekil_alan"] / np.pi)
    for (t, e), s in dev.groupby(["meta_tip", "etiket"]):
        scale = pd.concat([scale, pd.DataFrame([{"grup": f"Kaggle {t} {e} (hücre)", "n": len(s),
                                                 "cekirdek_cap_orani_medyan": s["cekirdek_cap_orani"].median()}])])
    scale.to_csv(out / "olcek_eritrosit_capi.csv", index=False, encoding="utf-8")
    print(scale.round(3).to_string(index=False))
    plot_munich_examples(tune)

    print("3) Öznitelik kayması, grup modelleri, tip kısayolu")
    center_tune = center[center["bolum"] == "ayar"].reset_index(drop=True)
    shift = feature_shift(dev, center_tune)
    shift.to_csv(out / "oznitelik_kaymasi.csv", index=False, encoding="utf-8")
    feats = pd.read_csv(stage_dir("features") / "oznitelik_onemi.csv")
    feats = feats.assign(g=feats["oznitelik"].str.split("_").str[0]).sort_values("f_sirasi").groupby("g").head(5)
    plot_feature_shift(dev, center_tune, feats["oznitelik"].tolist())
    cols = ["oznitelik", "Kaggle_Healthy", "Kaggle_AML", "Münih_LYT", "Münih_NGS", "Münih_MYO", "konum_LYT", "konum_NGS"]
    print(shift[shift["oznitelik"].isin(feats["oznitelik"])][cols].round(3).to_string(index=False))
    normal_pos = shift[["konum_LYT", "konum_NGS"]].mean(axis=1)
    print(f"  58 öznitelikten Münih normal hücrelerin Kaggle AML'ye daha yakın olduğu (konum > 0.5): "
          f"{int((normal_pos > 0.5).sum())}; Kaggle aralığının dışında (konum < -0.5 veya > 1.5): "
          f"{int(((normal_pos < -0.5) | (normal_pos > 1.5)).sum())}")
    subset = subset_model_check(dev, center_tune)
    subset.to_csv(out / "oznitelik_grubu_munih.csv", index=False, encoding="utf-8")
    print(subset.round(3).to_string(index=False))
    shortcut = type_shortcut_check(dev, center_tune)
    shortcut.to_csv(out / "tip_kisayolu_munih.csv", index=False, encoding="utf-8")
    print(shortcut.round(3).to_string(index=False))

    print("4) Renk: normalizasyon öncesi/sonrası Lab")
    sample = color_sample()
    pairs = list(zip(sample["id"], sample["yol"]))
    chunks = [pairs[i:i + 10] for i in range(0, len(pairs), 10)]
    with Parallel(n_jobs=N_JOBS) as par:
        colors = pd.DataFrame([r for res_c in par(delayed(color_chunk)(c) for c in chunks) for r in res_c])
    colors = sample[["veri_seti", "grup", "id"]].merge(colors, on="id")
    colors.drop(columns=["id"]).to_csv(out / "renk_lab_ozet_goruntu.csv", index=False, encoding="utf-8")
    summary = colors.groupby("grup")[[c for c in colors.columns if c.startswith(("once_", "sonra_"))]].median()
    summary.to_csv(out / "renk_lab_ozet.csv", encoding="utf-8")
    print(summary.round(1).to_string())
    plot_color(colors)


def run_allidb2_diagnosis() -> None:
    """
    ALL-IDB2 Healthy yakın çekimlerinin neden "Yüksek" risk aldığını teşhis eder (yalnızca sayılar; görüntü yok).

    1) Görüntü düzeyi: güvenli ALL-IDB alt küme × sınıf başına görüntü tipi, hücre sayısı, tek hücreli görüntü
       oranı, risk dağılımı ve merkez hücre olasılıkları (mevcut model sonuçlarından).
    2) Eğitim verisi: Kaggle geliştirme setinde görüntü tipi × sınıf sayıları (Healthy yakın çekim var mı?).
    3) Tip kısayolu: Kaggle hücrelerinden eğitilen "yakın çekim mi?" modeli güvenli ALL-IDB hücrelerine uygulanır.
    Çıktılar: outputs/external/allidb2_teshis_goruntu.csv, allidb2_teshis_kaggle_tip.csv, allidb2_tip_kisayolu.csv.
    """
    out = stage_dir("external")
    res = pd.read_csv(out / "allidb_goruntu_sonuclari_mevcut.csv")
    res = res[res["guvenli"]]
    rows = []
    for (s, e), sub in res.groupby(["idb_set", "etiket"]):
        scored = sub[sub["hucre_sayisi"] > 0]
        rows.append({"alt_kume": f"ALL-IDB{s[-1]} {e}", "goruntu": len(sub),
                     "yakin_cekim_orani": float((sub["tip"] == "yakin_cekim").mean()),
                     "hucre_medyan": float(sub["hucre_sayisi"].median()),
                     "tek_hucreli_orani": float((sub["hucre_sayisi"] == 1).mean()),
                     "eritrosit_capi_medyan": float(sub["eritrosit_capi"].median()),
                     **{f"risk_{lv}": int((sub["risk_seviyesi"] == lv).sum()) for lv in RISK_ORDER},
                     **{f"merkez_{c}": int((scored["merkez_tahmin"] == c).sum()) for c in CLASS_NAMES},
                     **{f"merkez_ort_{c}": float(scored[f"merkez_{c}"].mean()) for c in PROBA_COLUMNS},
                     # Tek hücreli görüntüde risk skoru yalnızca 0 veya 1 olabilir: hücre anormal → doğrudan Yüksek
                     "tek_hucre_yuksek": int(((sub["hucre_sayisi"] == 1) & (sub["risk_seviyesi"] == "Yüksek")).sum())})
    images = pd.DataFrame(rows)
    images.to_csv(out / "allidb2_teshis_goruntu.csv", index=False, encoding="utf-8")
    print(images.round(3).to_string(index=False))

    from src.classifier import load_split
    split = load_split()
    kag = pd.crosstab([split["bolum"], split["tip"]], split["etiket"]).reset_index()
    kag.to_csv(out / "allidb2_teshis_kaggle_tip.csv", index=False, encoding="utf-8")
    print(kag.to_string(index=False))

    dev = kaggle_dev_features()
    items = allidb_images()
    cells, _ = extract_dataset("allidb", items)
    cells = cells[cells["meta_kopya"] == 0].merge(items[["id", "idb_set", "etiket", "guvenli"]], on="id")
    cells = cells[cells["guvenli"]]
    groups = cells.assign(sinif=cells["idb_set"].str.replace("IDB", "ALL-IDB") + " " + cells["etiket"])
    shortcut = type_shortcut_check(dev, groups.reset_index(drop=True))
    shortcut["grup"] = shortcut["grup"].str.replace("Münih ", "")
    shortcut.to_csv(out / "allidb2_tip_kisayolu.csv", index=False, encoding="utf-8")
    print(shortcut.round(3).to_string(index=False))


PARTS = {"allidb": run_allidb, "segmentasyon": run_segmentation_check, "munich_bolum": lambda: munich_split(),
         "munich_teshis": run_munich_diagnosis, "allidb2_teshis": run_allidb2_diagnosis}


def main() -> None:
    """Komut satırında adı verilen bölümleri sırayla çalıştırır."""
    pd.set_option("display.width", 250)
    for name in sys.argv[1:] or ["allidb", "segmentasyon"]:
        print(f"\n=== {name} ===", flush=True)
        start = time.time()
        PARTS[name]()
        print(f"({name}: {time.time() - start:.0f} sn)")


if __name__ == "__main__":
    main()
