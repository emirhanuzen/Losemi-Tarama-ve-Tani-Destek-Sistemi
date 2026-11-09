"""
Adım 9 — Çok merkezli eğitim ve laboratuvar dışarıda bırakma (LOSO, leave-one-source-out) değerlendirmesi.

Adım 8: Kaggle ile eğitilen model Kaggle AML kaynağının çekim imzasını (parlak zemin, doygun çekirdek) AML olarak
öğrenmişti; Münih'te şans düzeyinde kaldı, ALL-IDB'de ALL'ı neredeyse hep AML diye tipledi. Burada üç kaynak
(Kaggle, ALL-IDB, Münih) birlikte kullanılır ve başarı, test kaynağı eğitime HİÇ girmeden ölçülür:

- Kat A: Kaggle geliştirme (kaynağı ALL-IDB olan Kaggle görüntüleri HARİÇ) + Münih → ALL-IDB güvenli 121 görüntü.
  Karşılaştırma: aynı kat, ALL-IDB kaynaklı Kaggle görüntüleri dahil ("A_idb_dahil").
- Kat B: Kaggle geliştirme + ALL-IDB → Münih TEST (Adım 8'deki kilitli %60).
- Kat C: Kaggle geliştirme + Münih + ALL-IDB (Kaggle testiyle kaynak grubu paylaşanlar hariç) → Kaggle v2 kilitli test.

Yapılandırmalar: (i) mevcut model, (ii) çok merkezli + 58 öznitelik, (iii) + 15 biyolojik öznitelik
(src/bio_features.py), (iv) (iii) eksi aynı sınıf içinde kaynağı ele veren öznitelikler (AUC ≥ 0.90; her katın
kendi eğitim verisinde ölçülür). Hiperparametreler her katın eğitim verisinde iç CV ile seçilir.

Bölümler (python -m src.multicenter <bölüm>):
    oznitelik    Üç kaynağın hücre öznitelikleri (58 + 15) → data/processed/multicenter/ (git dışı)
    kisayol      Öznitelik başına kaynak ve görüntü tipi AUC'si → outputs/multicenter/oznitelik_kisayol.csv
    kural        Ön kayıtlı seçim kuralı → outputs/multicenter/secim_kurali.csv (LOSO'dan ÖNCE; üzerine yazılmaz)
    loso         Katlar × yapılandırmalar → loso_sonuclari.csv (+ kat doğrulaması, kaynak bağımlı tipleme)
    laboratuvar  Laboratuvar kısayolu testi (öznitelikten kaynak tahmini) → laboratuvar_kisayolu.csv
    son_model    Seçilen yapılandırmayla v3 (üç kaynağın tamamı, temperature, yeni risk eşikleri; varsayılan DEĞİŞMEZ)
    gorseller    outputs/multicenter/ figürleri
    hepsi        Sırayla hepsi

Lisans: ALL-IDB görüntüsü içeren figür üretilmez; outputs/ altına yalnızca sayılar ve sayım figürleri yazılır.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.bio_features import BIO_FEATURES, BIO_GROUPS
from src.config import (
    CLASS_NAMES,
    FEATURES_CSVS,
    MULTICENTER_DIR,
    MULTICENTER_GRID,
    MULTICENTER_INNER_FOLDS,
    MULTICENTER_MAX_KAGGLE_DROP,
    MULTICENTER_MODEL_FILE,
    MULTICENTER_SOURCE_AUC_MAX,
    MULTICENTER_SOURCE_CAP,
    MUNICH_CLASSES,
    MUNICH_DIR,
    MUNICH_TRAIN_IMAGES,
    N_JOBS,
    RANDOM_STATE,
    RISK_LEVELS,
    RISK_TARGET_SENSITIVITY,
    RISK_TARGET_SPECIFICITY,
    RISK_THRESHOLDS,
    XYC_MATCH_RADIUS_RATIO,
)
from src.dataset import CLASS_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.utils import load_image, save_figure, stage_dir

SOURCES = ["Kaggle", "ALL-IDB", "Münih"]
SOURCE_COLORS = {"Kaggle": "#7a5af5", "ALL-IDB": "#2a78d6", "Münih": "#eb6834"}
PROBA_COLUMNS = [f"p_{c}" for c in CLASS_NAMES]
CHUNK = 20
PROGRESS_EVERY = 100
BASE_FEATURES_N = 58
CONFIGS = {
    # ad: öznitelik kümesi (None: mevcut kayıtlı model)
    "i_mevcut": None,
    "ii_cok_merkezli_58": "temel",
    "iii_cok_merkezli_bio": "bio",
    "iv_bio_kaynak_ayikli": "bio_kaynak_ayikli",
}
CONFIG_LABELS = {"i_mevcut": "(i) Mevcut model", "ii_cok_merkezli_58": "(ii) Çok merkezli, 58",
                 "iii_cok_merkezli_bio": "(iii) Çok merkezli + biyolojik",
                 "iv_bio_kaynak_ayikli": "(iv) (iii) − kaynak AUC ≥ 0.90"}
FOLDS = {
    "A": "Kaggle gel. (ALL-IDB kaynaklılar hariç) + Münih → ALL-IDB güvenli",
    "A_idb_dahil": "Kaggle gel. (tümü) + Münih → ALL-IDB güvenli (karşılaştırma)",
    "B": "Kaggle gel. + ALL-IDB → Münih TEST",
    "C": "Kaggle gel. + Münih + ALL-IDB → Kaggle test",
}
MAIN_FOLDS = ["A", "B", "C"]
OUT_STAGE = "multicenter"
RULE_CSV = "secim_kurali.csv"
LOSO_CSV = "loso_sonuclari.csv"


def out_dir() -> Path:
    """outputs/multicenter/ klasörünü döndürür (yoksa oluşturur)."""
    return stage_dir(OUT_STAGE)


# ---------------------------------------------------------------------------
# 1) Öznitelik tabloları
# ---------------------------------------------------------------------------

def kaggle_bio_chunk(rows: list[tuple[str, str, float]]) -> list[dict]:
    """
    Kaydedilmiş Kaggle hücre kırpımlarından biyolojik öznitelikleri çıkarır (paralel işçide, kırpımlar tek tek okunur).

    Parametreler:
        rows: (kırpım yolu, maske yolu, eritrosit çapı) listesi.

    Döndürür:
        Öznitelik sözlükleri.
    """
    from src.bio_features import bio_features
    from src.features import read_png

    return [bio_features(read_png(c, True), read_png(m, False), d) for c, m, d in rows]


def kaggle_table() -> pd.DataFrame:
    """
    Kaggle hücreleri: Adım 4 tablosundaki 58 öznitelik (features_kmeans.csv; uçtan uca pipeline ile aynı) + aynı
    kırpımlardan 15 biyolojik öznitelik. Biyolojik kısım data/processed/multicenter/kaggle_bio.csv'de önbelleklenir.

    Döndürür:
        Hücre tablosu (id = kaynak görüntü, etiket, kaggle_bolum, kat, kaynak_grup_id, kaggle_kaynak, tip ...).
    """
    from src.classifier import load_split

    MULTICENTER_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(FEATURES_CSVS["kmeans"])
    bio_path = MULTICENTER_DIR / "kaggle_bio.csv"
    if bio_path.exists():
        bio = pd.read_csv(bio_path)
    else:
        jobs = list(zip(cells["meta_kirpim_yolu"], cells["meta_maske_yolu"], cells["meta_eritrosit_capi"]))
        chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
        rows, start = [], time.time()
        print(f"  [kaggle bio] {len(jobs)} hücre", flush=True)
        with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
            for res in par(delayed(kaggle_bio_chunk)(c) for c in chunks):
                rows += res
                print(f"  [kaggle bio] {len(rows)}/{len(jobs)} hücre ({time.time() - start:.0f} sn)", flush=True)
        bio = pd.DataFrame(rows)
        bio.insert(0, "hucre_id", cells["hucre_id"].to_numpy())
        bio.to_csv(bio_path, index=False, encoding="utf-8")
    cells = cells.merge(bio, on="hucre_id", how="left", validate="one_to_one")
    split = load_split()[["dosya_yolu", "bolum", "kat", "kaynak_grup_id", "kaynak", "tip"]]
    split = split.rename(columns={"dosya_yolu": "id", "bolum": "kaggle_bolum", "kaynak": "kaggle_kaynak"})
    cells = cells.rename(columns={"meta_kaynak_dosya": "id"}).merge(split, on="id", how="left", validate="many_to_one")
    # Merkez uzaklığı (256 px uzayında, hücre kutusu merkezi): dış verideki tanımla aynı
    center = np.hypot(cells["meta_kutu_x"] + cells["meta_kutu_w"] / 2 - 128, cells["meta_kutu_y"] + cells["meta_kutu_h"] / 2 - 128)
    return cells.assign(kaynak="Kaggle", etiketli=True, meta_merkez_uzaklik=center, meta_yontem="kmeans",
                        meta_geri_donus=False, meta_xyc_blast=False)


def external_image_rows(item_id: str, path: str, xyc_path: str = "") -> tuple[list[dict], dict]:
    """
    Bir dış görüntüde varsayılan uçtan uca hesaplama (ön işleme → K-means + Otsu geri dönüşü → hücre kırpımı → 58
    öznitelik; external_validation.image_cells "varsayilan" ile aynı) + biyolojik öznitelikler + ALL-IDB1 .xyc eşleşmesi.

    Parametreler:
        item_id: Görüntü kimliği.
        path: Görüntü yolu.
        xyc_path: ALL-IDB1 blast merkezleri dosyası ("" ise eşleşme yapılmaz).

    Döndürür:
        (hücre satırları, görüntü bilgisi).
    """
    from src.bio_features import bio_features
    from src.external_validation import match_blasts, read_xyc, to_256
    from src.features import crop_features
    from src.pipeline import segment_with_fallback
    from src.preprocessing import preprocess
    from src.segmentation import extract_cells

    rgb = load_image(path)
    steps = preprocess(rgb)[1]
    seg = segment_with_fallback(steps)
    labels = seg["etiketler"]
    cells = extract_cells(steps["renk_normalize"], labels, steps["gecerli_maske"], seg["eritrosit_capi"])
    regions = [r for r, sl in enumerate(ndi.find_objects(labels), start=1) if sl is not None]   # extract_cells sırası
    blasts, n_xyc = set(), 0
    if xyc_path:
        centers = to_256(read_xyc(Path(xyc_path)), rgb.shape)
        n_xyc = len(centers)
        blasts = {h for h in match_blasts(labels, centers, XYC_MATCH_RADIUS_RATIO * seg["eritrosit_capi"]) if h > 0}
    d = round(seg["eritrosit_capi"], 2)   # Eğitim tablosuyla aynı yuvarlama (pipeline.run_pipeline)
    rows = []
    for region, c in zip(regions, cells):
        x, y, w, h = c["kutu"]
        rows.append({"id": item_id, **crop_features(c["kirpim"], c["maske"], d), **bio_features(c["kirpim"], c["maske"], d),
                     "meta_kenara_degiyor": bool(c["kenara_degiyor"]), "meta_alan": int(c["alan"]),
                     "meta_merkez_uzaklik": float(np.hypot(x + w / 2 - 128, y + h / 2 - 128)),
                     "meta_eritrosit_capi": d, "meta_yontem": seg["yontem"], "meta_geri_donus": seg["geri_donus"],
                     "meta_bolge": region, "meta_xyc_blast": region in blasts})
    info = {"id": item_id, "hucre_sayisi": len(cells), "eritrosit_capi": seg["eritrosit_capi"], "tip_tahmini": seg["tip"],
            "yontem": seg["yontem"], "geri_donus": seg["geri_donus"], "xyc_blast": n_xyc,
            "xyc_eslesen_cekirdek": len(blasts)}
    return rows, info


def external_chunk(items: list[tuple[str, str, str]]) -> tuple[list[dict], list[dict]]:
    """
    Bir parça dış görüntüyü işler (paralel işçide; görüntüler tek tek okunur).

    Parametreler:
        items: (id, yol, xyc yolu) listesi.

    Döndürür:
        (hücre satırları, görüntü satırları).
    """
    cells, images = [], []
    for item in items:
        rows, info = external_image_rows(*item)
        cells += rows
        images.append(info)
    return cells, images


def extract_external(name: str, items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dış görüntü kümesinin hücre özniteliklerini paralel çıkarır; data/processed/multicenter/<ad>_{hucre,goruntu}.csv
    önbelleğine yazar ve eksik görüntüleri tamamlar.

    Parametreler:
        name: "allidb" veya "munich".
        items: id, yol, xyc sütunlu tablo.

    Döndürür:
        (hücre tablosu, görüntü tablosu); yalnızca items'taki görüntüler.
    """
    MULTICENTER_DIR.mkdir(parents=True, exist_ok=True)
    cell_path, img_path = MULTICENTER_DIR / f"{name}_hucre.csv", MULTICENTER_DIR / f"{name}_goruntu.csv"
    cells = pd.read_csv(cell_path) if cell_path.exists() else pd.DataFrame(columns=["id"])
    images = pd.read_csv(img_path) if img_path.exists() else pd.DataFrame(columns=["id"])
    todo = items[~items["id"].isin(images["id"])]
    if len(todo):
        jobs = list(zip(todo["id"], todo["yol"], todo["xyc"]))
        chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
        new_c, new_i, start, done = [], [], time.time(), 0
        print(f"  [{name}] {len(jobs)} görüntü işlenecek", flush=True)
        with Parallel(n_jobs=N_JOBS, return_as="generator") as par:
            for c, (rc, ri) in zip(chunks, par(delayed(external_chunk)(c) for c in chunks)):
                new_c += rc
                new_i += ri
                done += len(c)
                if done % PROGRESS_EVERY < CHUNK or done == len(jobs):
                    print(f"  [{name}] {done}/{len(jobs)} ({time.time() - start:.0f} sn)", flush=True)
        cells = pd.concat([cells, pd.DataFrame(new_c)], ignore_index=True) if len(cells) else pd.DataFrame(new_c)
        images = pd.concat([images, pd.DataFrame(new_i)], ignore_index=True) if len(images) else pd.DataFrame(new_i)
        cells.to_csv(cell_path, index=False, encoding="utf-8")
        images.to_csv(img_path, index=False, encoding="utf-8")
    keep = set(items["id"])
    return cells[cells["id"].isin(keep)].reset_index(drop=True), images[images["id"].isin(keep)].reset_index(drop=True)


def allidb_items() -> pd.DataFrame:
    """
    Tüm ALL-IDB görüntüleri (güvenli ve Kaggle'da görülmüş) + ALL-IDB1 için .xyc yolu.

    Döndürür:
        id, yol, xyc, idb_set, etiket, guvenli, idb_dosya.
    """
    from src.external_validation import allidb_images

    items = allidb_images()
    xyc = []
    for path, subset in zip(items["yol"], items["idb_set"]):
        p = Path(path)
        xyc.append(str(p.parent.parent / "xyc" / (p.stem + ".xyc")) if subset == "IDB1" else "")
    return items.assign(xyc=xyc, idb_dosya=[i.split("/", 1)[1] for i in items["id"]])


def munich_items() -> pd.DataFrame:
    """
    Münih eğitim / test örneklemi. Kilitli AYAR/TEST ayrımının 900 görüntüsü (munich_bolum.csv) + sınıf başına
    MUNICH_TRAIN_IMAGES'e tamamlayan ek görüntüler (sabit tohum; Adım 8'de Kaggle ile eşleşip dışlanan 14 görüntü ve
    MYB hariç). Liste outputs/multicenter/munih_ornegi.csv'ye bir kez yazılır, sonra oradan okunur.

    Döndürür:
        id (MUNICH_DIR'e göre göreli), yol, xyc (""), sinif, beklenen, munih_bolum (ayar/test/ek_egitim).
    """
    from src.external_validation import munich_split
    from src.munich_overlap import list_munich

    path = out_dir() / "munih_ornegi.csv"
    if path.exists():
        sample = pd.read_csv(path, keep_default_na=False)
    else:
        split = munich_split()
        split = split[split["sinif"].isin(MUNICH_CLASSES)][["id", "sinif", "beklenen", "bolum"]]
        excluded = set(pd.read_csv(stage_dir("external") / "munich_dislanan.csv")["munich"])
        pool = list_munich().rename(columns={"goreli_yol": "id"})
        pool = pool[pool["sinif"].isin(MUNICH_CLASSES) & ~pool["id"].isin(excluded) & ~pool["id"].isin(split["id"])]
        parts = [split.rename(columns={"bolum": "munih_bolum"})]
        for i, (cls, target) in enumerate(MUNICH_TRAIN_IMAGES.items()):
            sub = pool[pool["sinif"] == cls]
            n_extra = max(0, target - int((split["sinif"] == cls).sum()))
            rng = np.random.default_rng(RANDOM_STATE + 100 + i)
            pick = sub.iloc[rng.permutation(len(sub))[:n_extra]]
            parts.append(pick[["id", "sinif", "beklenen"]].assign(munih_bolum="ek_egitim"))
        sample = pd.concat(parts, ignore_index=True)
        sample.to_csv(path, index=False, encoding="utf-8")
    return sample.assign(yol=[str(MUNICH_DIR / p) for p in sample["id"]], xyc="")


def unified_groups(split: pd.DataFrame, idb: pd.DataFrame) -> dict[str, str]:
    """
    Kaggle ve ALL-IDB görüntüleri için birleşik kaynak grubu (union-find).

    Bağlar: aynı Kaggle kaynak_grup_id (Adım 1 kopyaları + ALL-IDB eşleşmeleri), kabul edilmiş her Kaggle ↔ ALL-IDB
    eşleşmesi (o ALL-IDB görüntüsünün kaynak ALL-IDB1 yayma düğümüne), her ALL-IDB görüntüsü ↔ kaynak yayma düğümü
    (ALL-IDB2 kırpımı → kesildiği ALL-IDB1 yayması; source_groups.idb1_root_map). Aynı yaymanın türevleri tek grupta
    toplanır: iç CV katları ve LOSO kat doğrulaması bu gruplara göre yapılır.

    Parametreler:
        split: Kaggle v2 ayrımı.
        idb: allidb_items çıktısı.

    Döndürür:
        {görüntü kimliği: "g<n>"} (Kaggle: dosya_yolu, ALL-IDB: "IDB1/Im001_1.jpg").
    """
    from src.allidb_overlap import load_kaggle, load_saved_matches
    from src.dataset import assign_groups
    from src.external_validation import ALLIDB_STATUS_CSV
    from src.source_groups import idb1_root_map

    roots = idb1_root_map(pd.read_csv(ALLIDB_STATUS_CSV))
    matches = load_saved_matches(load_kaggle())
    acc = matches[matches["kabul"].astype(bool)]
    nodes: dict[str, int] = {}
    node = lambda name: nodes.setdefault(name, len(nodes))  # noqa: E731
    pairs = [(node("K:" + p), node(f"KG:{g}")) for p, g in zip(split["dosya_yolu"], split["kaynak_grup_id"])]
    pairs += [(node("K:" + r.kaggle_yolu), node("R:" + root)) for r in acc.itertuples(index=False)
              for root in roots[r.idb_dosya]]
    pairs += [(node("I:" + i), node("R:" + root)) for i, f in zip(idb["id"], idb["idb_dosya"]) for root in roots[f]]
    comp = assign_groups(len(nodes), pairs)
    return {name[2:]: f"g{comp[k]}" for name, k in nodes.items() if name[:2] in ("K:", "I:")}


def build_tables(extract: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Üç kaynağın birleşik hücre ve görüntü tablolarını kurar (gerekirse öznitelikleri çıkarır).

    Etiketler (yalnızca etiketli hücreler eğitime girer; etiketli=False hücreler sadece görüntü düzeyi tahminde):
    - Kaggle: tüm hücreler, görüntü etiketi (Adım 5 ile aynı).
    - Münih: yalnızca merkez hücre (görüntü merkezine en yakın çekirdek); MYO → AML, LYT / NGS → Healthy.
    - ALL-IDB2: merkez hücre; _1 → ALL, _0 → Healthy.
    - ALL-IDB1: _1 görüntülerinde .xyc blast merkeziyle eşleşen çekirdekler → ALL (eşleşmeyenler etiketsiz); _0
      görüntülerindeki tüm çekirdekler → Healthy.

    Parametreler:
        extract: Eksik öznitelikleri çıkar (False: yalnızca önbellekten oku).

    Döndürür:
        (hücreler, görüntüler).
    """
    from src.classifier import load_split

    split = load_split()
    kag = kaggle_table()
    idb = allidb_items()
    mun = munich_items()
    idb_cells, idb_imgs = extract_external("allidb", idb)
    mun_cells, mun_imgs = extract_external("munich", mun)
    groups = unified_groups(split, idb)

    # ALL-IDB etiketleri
    idb_cells = idb_cells.merge(idb[["id", "idb_set", "etiket", "guvenli"]].rename(columns={"etiket": "goruntu_etiket"}),
                                on="id", how="left")
    center = idb_cells.groupby("id")["meta_merkez_uzaklik"].transform("min") == idb_cells["meta_merkez_uzaklik"]
    first_center = center & ~idb_cells.assign(c=center).duplicated(["id", "c"])   # eşit uzaklıkta tek hücre
    is1, all_img = idb_cells["idb_set"] == "IDB1", idb_cells["goruntu_etiket"] == "ALL"
    lab = np.where(is1 & all_img & idb_cells["meta_xyc_blast"].astype(bool), "ALL",
                   np.where(is1 & ~all_img, "Healthy",
                            np.where(~is1 & first_center, idb_cells["goruntu_etiket"], "")))
    idb_cells = idb_cells.assign(etiket=lab, etiketli=lab != "", kaynak="ALL-IDB", grup=idb_cells["id"].map(groups),
                                 merkez=first_center)
    # Münih etiketleri: yalnızca merkez hücre
    mun_cells = mun_cells.merge(mun[["id", "sinif", "beklenen", "munih_bolum"]].rename(columns={"sinif": "munih_sinif"}),
                                on="id", how="left")
    center = mun_cells.groupby("id")["meta_merkez_uzaklik"].transform("min") == mun_cells["meta_merkez_uzaklik"]
    first_center = center & ~mun_cells.assign(c=center).duplicated(["id", "c"])
    mun_cells = mun_cells.assign(etiket=np.where(first_center, mun_cells["beklenen"], ""), etiketli=first_center,
                                 kaynak="Münih", grup="m:" + mun_cells["id"], merkez=first_center)
    kag = kag.assign(grup=kag["id"].map(groups), merkez=False)
    if kag["grup"].isna().any() or idb_cells["grup"].isna().any():
        raise ValueError("Birleşik grup bulunamayan görüntü var.")
    cells = pd.concat([kag, idb_cells, mun_cells], ignore_index=True, sort=False)
    cells["meta_kenara_degiyor"] = cells["meta_kenara_degiyor"].astype(bool)
    cells["etiket"] = cells["etiket"].fillna("").astype(str)
    cells["etiketli"] = cells["etiketli"].astype(bool)

    kimg = split.rename(columns={"dosya_yolu": "id", "bolum": "kaggle_bolum", "kaynak": "kaggle_kaynak"})
    kimg = kimg.assign(kaynak="Kaggle", grup=kimg["id"].map(groups))
    iimg = idb_imgs.merge(idb[["id", "idb_set", "etiket", "guvenli"]], on="id").assign(kaynak="ALL-IDB")
    iimg["grup"] = iimg["id"].map(groups)
    mimg = mun_imgs.merge(mun[["id", "sinif", "beklenen", "munih_bolum"]].rename(columns={"sinif": "munih_sinif"}),
                          on="id").assign(kaynak="Münih")
    mimg = mimg.assign(etiket=mimg["beklenen"], grup="m:" + mimg["id"])
    images = pd.concat([kimg, iimg, mimg], ignore_index=True, sort=False)
    return cells, images


def feature_sets(cells: pd.DataFrame) -> dict[str, list[str]]:
    """
    Temel (58) ve biyolojik dahil (73) öznitelik listeleri.

    Parametreler:
        cells: Birleşik hücre tablosu.

    Döndürür:
        {"temel": [...], "bio": [...]}.
    """
    from src.utils import feature_columns

    base = [c for c in feature_columns(cells) if c not in BIO_FEATURES]
    if len(base) != BASE_FEATURES_N:
        raise ValueError(f"Beklenen {BASE_FEATURES_N} temel öznitelik, bulunan {len(base)}")
    return {"temel": base, "bio": base + BIO_FEATURES}


def run_features() -> None:
    """Üç kaynağın özniteliklerini çıkarır ve etiketli hücre sayılarını kaynak × sınıf olarak yazar."""
    cells, images = build_tables()
    lab = cells[cells["etiketli"]]
    counts = pd.crosstab(lab["kaynak"], lab["etiket"]).reindex(index=SOURCES, columns=CLASS_NAMES, fill_value=0)
    counts["toplam"] = counts.sum(axis=1)
    counts.to_csv(out_dir() / "etiketli_hucre_sayilari.csv", encoding="utf-8")
    print(counts.to_string())
    idb = images[images["kaynak"] == "ALL-IDB"]
    print(f"ALL-IDB1 .xyc: {int(idb['xyc_blast'].sum())} blast merkezi, {int(idb['xyc_eslesen_cekirdek'].sum())} eşleşen "
          f"çekirdek; hücresiz görüntü: {int((images['hucre_sayisi'] == 0).sum())}")


# ---------------------------------------------------------------------------
# 2) Öznitelik kısayol testi (kaynak ve görüntü tipi AUC'si)
# ---------------------------------------------------------------------------

def folded_auc(y: np.ndarray, x: np.ndarray) -> float:
    """
    Tek özniteliğin ikili hedefi ayırma AUC'si, yönden bağımsız (|AUC − 0.5| + 0.5). Eksik değerler medyanla doldurulur.

    Parametreler:
        y: bool hedef.
        x: Öznitelik değerleri.

    Döndürür:
        0.5-1 arası AUC (tek sınıf varsa NaN).
    """
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    x = np.asarray(x, np.float64)
    x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def within_class_source_auc(cells: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Her öznitelik için AYNI SINIF içinde kaynak ayırma AUC'si: en az iki kaynağı olan her sınıfta her kaynak için
    bire-karşı-diğerleri; en büyüğü ve hangi sınıf / kaynakta olduğu.

    Parametreler:
        cells: Etiketli hücreler (kaynak, etiket).
        cols: Öznitelikler.

    Döndürür:
        oznitelik, kaynak_auc, kaynak_auc_sinif, kaynak_auc_kaynak + sınıf başına sütunlar.
    """
    rows = []
    for col in cols:
        row, best = {"oznitelik": col}, (np.nan, "", "")
        for cls in CLASS_NAMES:
            sub = cells[cells["etiket"] == cls]
            srcs = [s for s in SOURCES if (sub["kaynak"] == s).sum() >= 20]
            if len(srcs) < 2:
                continue
            vals = {s: folded_auc(sub["kaynak"].eq(s).to_numpy(), sub[col].to_numpy()) for s in srcs}
            s_max = max(vals, key=lambda s: vals[s])
            row[f"kaynak_auc_{cls}"] = vals[s_max]
            if not np.isfinite(best[0]) or vals[s_max] > best[0]:
                best = (vals[s_max], cls, s_max)
        row.update({"kaynak_auc": best[0], "kaynak_auc_sinif": best[1], "kaynak_auc_kaynak": best[2]})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_category(col: str) -> str:
    """
    Öznitelik kategorisi (raporda işaretleme için).

    Parametreler:
        col: Sütun adı.

    Döndürür:
        "biyolojik", "mutlak zemin rengi" (halka ortalamaları: yaymada halka çoğunlukla zemin), "mutlak çekirdek rengi",
        "göreli renk", "şekil", "doku".
    """
    if col in BIO_FEATURES:
        return "biyolojik"
    if col.startswith("renk_halka_") and col.endswith("_ort"):
        return "mutlak zemin rengi"
    if col.startswith("renk_cekirdek_") and col.endswith("_ort"):
        return "mutlak çekirdek rengi"
    if col.startswith("renk_"):
        return "renk (std / fark)"
    return "şekil" if col.startswith("sekil_") else "doku"


def run_shortcut() -> pd.DataFrame:
    """
    Öznitelik başına kısayol tablosu (tüm etiketli hücrelerde): sınıf içi kaynak AUC'si, Kaggle'da görüntü tipi
    (yayma / yakın çekim) AUC'si (yakın çekimi olan ALL ve AML sınıflarında, en büyüğü) ve sınıf ayırma AUC'si
    (ALL vs AML, anormal hücrelerde; biyolojik bilgi taşıyor mu?). Kaynak AUC ≥ MULTICENTER_SOURCE_AUC_MAX ve mutlak
    zemin rengine dayananlar işaretlenir. Yazılan: outputs/multicenter/oznitelik_kisayol.csv.

    Döndürür:
        Kısayol tablosu.
    """
    cells, _ = build_tables()
    lab = cells[cells["etiketli"]]
    cols = feature_sets(cells)["bio"]
    table = within_class_source_auc(lab, cols)
    kag = lab[lab["kaynak"] == "Kaggle"]
    tip = []
    for col in cols:
        vals = [folded_auc(kag.loc[kag["etiket"] == c, "tip"].eq("yakin_cekim").to_numpy(),
                           kag.loc[kag["etiket"] == c, col].to_numpy()) for c in ("ALL", "AML")]
        tip.append(np.nanmax(vals))
    table["tip_auc_kaggle"] = tip
    abn = lab[lab["etiket"].isin(["ALL", "AML"])]
    table["ALL_AML_auc_tum_kaynaklar"] = [folded_auc(abn["etiket"].eq("ALL").to_numpy(), abn[c].to_numpy()) for c in cols]
    table["kategori"] = [feature_category(c) for c in cols]
    table["kaynak_auc_yuksek"] = table["kaynak_auc"] >= MULTICENTER_SOURCE_AUC_MAX
    table["mutlak_zemin_rengi"] = table["kategori"] == "mutlak zemin rengi"
    table = table.sort_values("kaynak_auc", ascending=False)
    table.to_csv(out_dir() / "oznitelik_kisayol.csv", index=False, encoding="utf-8")
    show = ["oznitelik", "kategori", "kaynak_auc", "kaynak_auc_sinif", "kaynak_auc_kaynak", "tip_auc_kaggle",
            "ALL_AML_auc_tum_kaynaklar"]
    print(table[table["oznitelik"].isin(BIO_FEATURES)][show].round(3).to_string(index=False))
    print(f"Kaynak AUC ≥ {MULTICENTER_SOURCE_AUC_MAX}: {int(table['kaynak_auc_yuksek'].sum())}/{len(table)} öznitelik "
          f"(biyolojik: {int(table.loc[table['kategori'] == 'biyolojik', 'kaynak_auc_yuksek'].sum())}/{len(BIO_FEATURES)})")
    return table


# ---------------------------------------------------------------------------
# 3) Eğitim: kaynak ağırlıkları, model, iç CV
# ---------------------------------------------------------------------------

def source_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Kaynak × sınıf tavanlı, sınıf dengeli örnek ağırlıkları (gerekçe: config.MULTICENTER_SOURCE_CAP).

    1) Kaynak × sınıf grubunun hücre sayısı n > tavan ise her hücre tavan / n, değilse 1.
    2) Her sınıfın toplam ağırlığı eşitlenir. 3) Ortalama ağırlık 1 olacak şekilde ölçeklenir.

    Parametreler:
        df: Etiketli eğitim hücreleri (kaynak, etiket).

    Döndürür:
        (n,) ağırlıklar.
    """
    key = df["kaynak"].astype(str) + "|" + df["etiket"].astype(str)
    n = key.map(key.value_counts())
    w = np.minimum(1.0, MULTICENTER_SOURCE_CAP / n.to_numpy(np.float64))
    totals = pd.Series(w).groupby(df["etiket"].to_numpy()).sum()
    w = w * (totals.mean() / df["etiket"].map(totals).to_numpy())
    return w / w.mean()


def weight_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaynak × sınıf hücre sayısı ve toplam ağırlık payı (raporlama).

    Parametreler:
        df: Etiketli eğitim hücreleri.

    Döndürür:
        kaynak, etiket, hucre, toplam_agirlik_payi.
    """
    w = source_weights(df)
    t = df.assign(w=w).groupby(["kaynak", "etiket"]).agg(hucre=("w", "size"), agirlik=("w", "sum")).reset_index()
    t["toplam_agirlik_payi"] = t["agirlik"] / t["agirlik"].sum()
    return t.drop(columns="agirlik")


def make_model(params: dict) -> Pipeline:
    """
    HGB modeli (Adım 5 ile aynı yapı: medyan doldurma → standartlaştırma → HGB). class_weight YOK: sınıf dengesi
    source_weights ile örnek ağırlığında sağlanır (ikisi birden sınıf dengesini iki kez uygulardı).

    Parametreler:
        params: HGB hiperparametreleri.

    Döndürür:
        Eğitilmemiş Pipeline.
    """
    return Pipeline([("doldurma", SimpleImputer(strategy="median")), ("olcekleme", StandardScaler()),
                     ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params))])


def fit_weighted(df: pd.DataFrame, cols: list[str], params: dict, target: str = "etiket") -> Pipeline:
    """
    Ağırlıklı eğitim.

    Parametreler:
        df: Etiketli eğitim hücreleri.
        cols: Öznitelikler.
        params: Hiperparametreler.
        target: Hedef sütunu.

    Döndürür:
        Eğitilmiş model.
    """
    model = make_model(params)
    with threadpool_limits(limits=N_JOBS):
        model.fit(df[cols].to_numpy(np.float64), df[target].to_numpy(), model__sample_weight=source_weights(df))
    return model


def proba_of(model, df: pd.DataFrame, cols: list[str], classes: list[str] = CLASS_NAMES) -> np.ndarray:
    """
    Olasılıkları istenen sınıf sırasında döndürür (modelde olmayan sınıf 0).

    Parametreler:
        model: Eğitilmiş model.
        df: Hücreler.
        cols: Öznitelikler.
        classes: Sınıf sırası.

    Döndürür:
        (n, len(classes)) olasılık.
    """
    if df.empty:
        return np.zeros((0, len(classes)))
    with threadpool_limits(limits=N_JOBS):
        p = model.predict_proba(df[cols].to_numpy(np.float64))
    known = list(model.classes_)
    return np.stack([p[:, known.index(c)] if c in known else np.zeros(len(df)) for c in classes], axis=1)


def inner_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    İç CV katları: StratifiedGroupKFold (grup = birleşik kaynak grubu, katman = kaynak × sınıf).

    Parametreler:
        df: Etiketli hücreler (grup, kaynak, etiket).

    Döndürür:
        (eğitim indeksleri, doğrulama indeksleri) listesi.
    """
    strata = (df["kaynak"] + "|" + df["etiket"]).to_numpy()
    cv = StratifiedGroupKFold(n_splits=MULTICENTER_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(df)), strata, df["grup"].to_numpy()))


def tune(df: pd.DataFrame, cols: list[str], tag: str) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """
    Küçük ızgarada (MULTICENTER_GRID) iç CV ile hiperparametre seçer. Ölçüt: ağırlıklı makro F1 (eğitimle aynı
    kaynak / sınıf ağırlıkları; hiçbir kaynak ölçüte baskın olmasın). Test kaynağı bu veride YOKTUR.

    Parametreler:
        df: Katın etiketli eğitim hücreleri.
        cols: Öznitelikler.
        tag: İlerleme etiketi.

    Döndürür:
        (en iyi parametreler, ızgara tablosu, en iyi parametrelerin katlar dışı olasılıkları).
    """
    folds = inner_folds(df)
    w = source_weights(df)
    y = df["etiket"].to_numpy()
    rows, best = [], (-1.0, None, None)
    for params in MULTICENTER_GRID:
        oof = np.zeros((len(df), len(CLASS_NAMES)))
        for tr, va in folds:
            model = fit_weighted(df.iloc[tr], cols, params)
            oof[va] = proba_of(model, df.iloc[va], cols)
        pred = np.asarray(CLASS_NAMES)[oof.argmax(axis=1)]
        score = f1_score(y, pred, labels=CLASS_NAMES, average="macro", sample_weight=w)
        rows.append({**params, "ic_cv_agirlikli_makro_f1": score})
        print(f"    [{tag}] lr={params['learning_rate']} yaprak={params['max_leaf_nodes']}: F1 {score:.3f}", flush=True)
        if score > best[0]:
            best = (score, params, oof)
    return best[1], pd.DataFrame(rows), best[2]


# ---------------------------------------------------------------------------
# 4) LOSO katları
# ---------------------------------------------------------------------------

def fold_split(cells: pd.DataFrame, images: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bir katın eğitim hücrelerini, test hücrelerini ve test görüntülerini seçer.

    Parametreler:
        cells, images: build_tables çıktıları.
        fold: FOLDS anahtarı.

    Döndürür:
        (etiketli eğitim hücreleri, test hücreleri (etiketsizler dahil), test görüntüleri).
    """
    lab = cells["etiketli"]
    kag_dev = (cells["kaynak"] == "Kaggle") & (cells["kaggle_bolum"] == "gelistirme")
    mun, idb = cells["kaynak"] == "Münih", cells["kaynak"] == "ALL-IDB"
    if fold in ("A", "A_idb_dahil"):
        kag = kag_dev & ((cells["kaggle_kaynak"] == "bilinmiyor") if fold == "A" else True)
        train = lab & (kag | mun)
        test_imgs = images[(images["kaynak"] == "ALL-IDB") & images["guvenli"].astype(bool)]
        if fold == "A_idb_dahil":
            # Adım 8'de "güvenli" sayılan IDB1/Im036_0, Kaggle'da görülmüş Im103_0 ile aynı ALL-IDB2 kırpımlarını
            # (Im134/135/244/245_0) paylaşıyor: iki yayma aynı bölgeyi örtüyor. ALL-IDB kaynaklı Kaggle görüntüleri
            # eğitimdeyken bu tür test görüntüleri çıkarılır (ana Kat A'da bu Kaggle görüntüleri zaten yok).
            test_imgs = test_imgs[~test_imgs["grup"].isin(set(cells.loc[train, "grup"]))]
    elif fold == "B":
        train = lab & (kag_dev | idb)
        test_imgs = images[(images["kaynak"] == "Münih") & (images["munih_bolum"] == "test")]
    elif fold == "C":
        test_imgs = images[(images["kaynak"] == "Kaggle") & (images["kaggle_bolum"] == "test")]
        test_groups = set(test_imgs["grup"])
        # Kaggle geliştirme görüntüleri de birleşik gruba göre ayıklanır: v2 kaynak grupları 17 (geliştirme) ve 18
        # (test), aynı bölgeyi örten ALL-IDB1 Im020_1 / Im021_1 yaymalarından (ortak ALL-IDB2 kırpımları Im056/057_1)
        # geliyor. Kilitli v2 ayrımı değiştirilmez; bu katta 3 geliştirme görüntüsü eğitimden çıkar.
        train = lab & (kag_dev | mun | idb) & ~cells["grup"].isin(test_groups)
    else:
        raise ValueError(fold)
    test_cells = cells[cells["id"].isin(set(test_imgs["id"]))]
    return cells[train].reset_index(drop=True), test_cells.reset_index(drop=True), test_imgs.reset_index(drop=True)


def verify_fold(fold: str, train: pd.DataFrame, test_imgs: pd.DataFrame) -> dict:
    """
    Test kaynağının eğitime hiç girmediğini kodla doğrular; ihlal varsa hata verir.

    Kontroller: ortak görüntü yok; ortak birleşik kaynak grubu yok; A'da eğitimde ALL-IDB hücresi ve (A için) kaynağı
    ALL-IDB olan Kaggle görüntüsü yok; B'de Münih hücresi yok; C'de Kaggle test görüntüsü yok.

    Parametreler:
        fold: Kat adı.
        train: Eğitim hücreleri.
        test_imgs: Test görüntüleri.

    Döndürür:
        Kontrol sonuçları sözlüğü (tümü 0 olmalı) + sayılar.
    """
    checks = {
        "ortak_goruntu": len(set(train["id"]) & set(test_imgs["id"])),
        "ortak_kaynak_grubu": len(set(train["grup"]) & set(test_imgs["grup"])),
    }
    if fold.startswith("A"):
        checks["egitimde_ALL-IDB_hucresi"] = int((train["kaynak"] == "ALL-IDB").sum())
        if fold == "A":
            checks["egitimde_ALL-IDB_kaynakli_Kaggle"] = int(((train["kaynak"] == "Kaggle")
                                                              & (train["kaggle_kaynak"] == "ALL-IDB")).sum())
    elif fold == "B":
        checks["egitimde_Munih_hucresi"] = int((train["kaynak"] == "Münih").sum())
    else:
        checks["egitimde_Kaggle_test_hucresi"] = int(((train["kaynak"] == "Kaggle")
                                                      & (train["kaggle_bolum"] == "test")).sum())
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise AssertionError(f"Kat {fold}: test kaynağı eğitime sızmış: {bad}")
    counts = train.groupby(["kaynak", "etiket"]).size()
    return {"kat": fold, **checks, "egitim_hucre": len(train), "egitim_goruntu": train["id"].nunique(),
            "test_goruntu": len(test_imgs),
            **{f"egitim_{s}_{c}": int(counts.get((s, c), 0)) for s in SOURCES for c in CLASS_NAMES}}


def evaluate_fold(fold: str, test_cells: pd.DataFrame, test_imgs: pd.DataFrame, proba: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """
    Katın test ölçüleri.

    - A (ALL-IDB, görüntü düzeyi; Adım 8 ile aynı tanımlar): ALL görüntülerinde görüntü tahmini ALL oranı (ana ölçü,
      "ALL tiplemesi"), AML oranı, tarama duyarlılığı (risk Orta/Yüksek), Healthy özgüllüğü (risk Düşük veya
      değerlendirilemedi), Healthy görüntü tahmini oranı; IDB1 / IDB2 ayrı. Risk eşikleri config.RISK_THRESHOLDS.
    - B (Münih TEST, merkez hücre): MYO → AML, LYT → Healthy, NGS → Healthy duyarlılıkları, dengeli doğruluk.
    - C (Kaggle test): hücre / görüntü makro F1 (domain_shift.kaggle_scores; Adım 8 ile aynı tanım).

    Parametreler:
        fold: Kat.
        test_cells, test_imgs: fold_split çıktıları.
        proba: test_cells olasılıkları.

    Döndürür:
        (ölçü sözlüğü, görüntü / hücre düzeyi sonuç tablosu).
    """
    from src.external_validation import image_results

    if fold == "C":
        from src.classifier import load_split
        from src.domain_shift import kaggle_scores

        split = load_split()
        scores = kaggle_scores(test_cells, proba, split[split["bolum"] == "test"])
        res = pd.DataFrame({"id": test_cells["id"], "etiket": test_cells["etiket"],
                            "tahmin": np.asarray(CLASS_NAMES)[proba.argmax(axis=1)],
                            "kenara_degiyor": test_cells["meta_kenara_degiyor"]})
        return {"C_hucre_makro_f1": scores["hucre_makro_f1"], "C_goruntu_makro_f1": scores["goruntu_makro_f1_tum"],
                "C_goruntu_makro_f1_kapsanan": scores["goruntu_makro_f1_kapsanan"],
                **{f"C_hucre_duyarlilik_{c}": scores[f"hucre_duyarlilik_{c}"] for c in CLASS_NAMES}}, res
    res = image_results(test_cells, proba, test_imgs[["id"]])
    res = test_imgs.drop(columns=[c for c in ("hucre_sayisi",) if c in test_imgs]).merge(res, on="id")
    if fold.startswith("A"):
        p = "A_" if fold == "A" else "Aidb_"
        out = {}
        for tag, sub in [("", res), ("IDB1_", res[res["idb_set"] == "IDB1"]), ("IDB2_", res[res["idb_set"] == "IDB2"])]:
            a, h = sub[sub["etiket"] == "ALL"], sub[sub["etiket"] == "Healthy"]
            out.update({f"{p}{tag}ALL_tipleme": float((a["goruntu_tahmin"] == "ALL").mean()),
                        f"{p}{tag}ALL_AML_orani": float((a["goruntu_tahmin"] == "AML").mean()),
                        f"{p}{tag}duyarlilik": float(a["risk_seviyesi"].isin(["Orta", "Yüksek"]).mean()),
                        f"{p}{tag}healthy_ozgulluk": float((~h["risk_seviyesi"].isin(["Orta", "Yüksek"])).mean()),
                        f"{p}{tag}healthy_goruntu_tahmini": float((h["goruntu_tahmin"] == "Healthy").mean())})
        out[f"{p}n_ALL"], out[f"{p}n_Healthy"] = int((res["etiket"] == "ALL").sum()), int((res["etiket"] == "Healthy").sum())
        return out, res
    rec = {cls: float((res.loc[res["munih_sinif"] == cls, "merkez_tahmin"] == exp).mean())
           for cls, exp in MUNICH_CLASSES.items()}
    myo = res[res["munih_sinif"] == "MYO"]
    return {"B_dengeli_dogruluk": float(np.mean(list(rec.values()))),
            **{f"B_duyarlilik_{c}": v for c, v in rec.items()},
            "B_MYO_ALL_orani": float((myo["merkez_tahmin"] == "ALL").mean()),
            "B_normal_ozgulluk": float(res.loc[res["munih_sinif"] != "MYO", "merkez_tahmin"].eq("Healthy").mean())}, res


def config_features(name: str, sets: dict, train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Yapılandırmanın öznitelik listesi. (iv): biyolojik dahil küme eksi, BU katın eğitim verisinde sınıf içi kaynak
    AUC'si ≥ MULTICENTER_SOURCE_AUC_MAX olanlar (test kaynağına bakılmaz).

    Parametreler:
        name: CONFIGS anahtarı (i hariç).
        sets: feature_sets çıktısı.
        train: Katın etiketli eğitim hücreleri.

    Döndürür:
        (kullanılan öznitelikler, çıkarılan öznitelikler).
    """
    kind = CONFIGS[name]
    if kind != "bio_kaynak_ayikli":
        return sets[kind], []
    auc = within_class_source_auc(train, sets["bio"]).set_index("oznitelik")["kaynak_auc"]
    drop = [c for c in sets["bio"] if auc.get(c, 0) >= MULTICENTER_SOURCE_AUC_MAX]
    return [c for c in sets["bio"] if c not in drop], drop


def source_typing(train: pd.DataFrame, oof: np.ndarray, fold: str, config: str) -> pd.DataFrame:
    """
    Kaynak bağımlı tipleme (iç CV katlar dışı tahminlerinden): aynı sınıftaki hücreler kaynağa göre farklı mı
    tipleniyor? Sınıf × kaynak başına duyarlılık ve anormal hücrelerde "anormal tahmin edilenler içinde doğru tip".

    Parametreler:
        train: Etiketli eğitim hücreleri.
        oof: Katlar dışı olasılıklar.
        fold, config: Etiketler.

    Döndürür:
        kat, yapilandirma, sinif, kaynak, n, duyarlilik, tip_dogrulugu_anormalde, tahmin_<sınıf>_orani.
    """
    pred = np.asarray(CLASS_NAMES)[oof.argmax(axis=1)]
    rows = []
    for (cls, src), idx in train.groupby(["etiket", "kaynak"]).indices.items():
        p = pred[idx]
        abn = np.isin(p, ["ALL", "AML"])
        rows.append({"kat": fold, "yapilandirma": config, "sinif": cls, "kaynak": src, "n": len(idx),
                     "duyarlilik": float((p == cls).mean()),
                     "tip_dogrulugu_anormalde": float((p[abn] == cls).mean()) if cls != "Healthy" and abn.any() else np.nan,
                     **{f"tahmin_{c}_orani": float((p == c).mean()) for c in CLASS_NAMES}})
    return pd.DataFrame(rows)


def write_rule() -> None:
    """
    Ön kayıtlı seçim kuralını outputs/multicenter/secim_kurali.csv'ye yazar. LOSO sonuç dosyası varsa veya kural
    zaten yazılmışsa DOKUNMAZ (kural sonuçlardan önce sabitlenir).
    """
    path = out_dir() / RULE_CSV
    if path.exists():
        print(f"Kural zaten kayıtlı, değiştirilmedi: {path}")
        return
    if (out_dir() / LOSO_CSV).exists():
        raise RuntimeError("LOSO sonuçları kuraldan önce üretilmiş; kural artık ön kayıtlı sayılamaz.")
    rows = [
        {"madde": "skor", "tanim": "(Kat A ALL tiplemesi oranı + Kat B dengeli doğruluk) / 2; en yüksek seçilir",
         "deger": ""},
        {"madde": "Kat A ALL tiplemesi", "tanim": "ALL-IDB güvenli ALL görüntülerinde görüntü tahmini = ALL oranı "
         "(kenara değmeyen hücrelerin ortalama olasılığı; Adım 8 tanımı)", "deger": ""},
        {"madde": "Kat B dengeli doğruluk", "tanim": "Münih TEST merkez hücre: MYO→AML, LYT→Healthy, NGS→Healthy "
         "duyarlılıklarının ortalaması", "deger": ""},
        {"madde": "kısıt", "tanim": "Kat C (Kaggle kilitli test) görüntü makro F1 ≥ mevcut model − eşik",
         "deger": MULTICENTER_MAX_KAGGLE_DROP},
        {"madde": "adaylar", "tanim": "; ".join(CONFIGS), "deger": ""},
        {"madde": "eşitlik", "tanim": "skor eşitse Kat C görüntü makro F1 yüksek olan", "deger": ""},
        {"madde": "hiçbiri sağlamazsa", "tanim": "i_mevcut", "deger": ""},
        {"madde": "(iv) kaynak AUC eşiği", "tanim": "sınıf içi kaynak AUC'si ≥ eşik olan öznitelikler çıkarılır "
         "(her katın eğitim verisinde)", "deger": MULTICENTER_SOURCE_AUC_MAX},
        {"madde": "kayıt zamanı", "tanim": time.strftime("%Y-%m-%d %H:%M:%S"), "deger": ""},
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"Ön kayıtlı seçim kuralı yazıldı: {path}")


def run_loso(folds: list[str] | None = None) -> pd.DataFrame:
    """
    LOSO değerlendirmesi: her kat × yapılandırma için iç CV ile hiperparametre seçimi, katın eğitim verisiyle eğitim
    ve test kaynağında değerlendirme. Her kat-yapılandırmadan sonra CSV'ler güncellenir.

    Yazılanlar (outputs/multicenter/): loso_sonuclari.csv, kat_dogrulama.csv, ic_cv_izgara.csv,
    kaynak_bagimli_tipleme.csv, kat_agirliklari.csv; sonuç tabloları loso_tahmin_<kat>_<yapılandırma>.csv
    (ALL-IDB içerenler data/processed/multicenter/ altına: yalnızca kimlik + tahmin, görüntü yok).

    Parametreler:
        folds: Çalıştırılacak katlar (None: hepsi).

    Döndürür:
        Sonuç tablosu.
    """
    from src.classifier import load_model
    from src.external_validation import predict_cells

    if not (out_dir() / RULE_CSV).exists():
        raise RuntimeError("Önce ön kayıtlı kuralı yazın: python -m src.multicenter kural")
    cells, images = build_tables()
    sets = feature_sets(cells)
    od = out_dir()
    paths = {k: od / f for k, f in [("res", LOSO_CSV), ("ver", "kat_dogrulama.csv"), ("grid", "ic_cv_izgara.csv"),
                                    ("typ", "kaynak_bagimli_tipleme.csv"), ("w", "kat_agirliklari.csv")]}
    old = {k: pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists()
           else pd.DataFrame(columns=["kat", "yapilandirma"]) for k, p in paths.items()}
    current = load_model()
    for fold in folds or list(FOLDS):
        start = time.time()
        train, test_cells, test_imgs = fold_split(cells, images, fold)
        ver = verify_fold(fold, train, test_imgs)
        print(f"\n=== Kat {fold}: {FOLDS[fold]} | eğitim {ver['egitim_hucre']} hücre, test {ver['test_goruntu']} "
              f"görüntü | sızıntı kontrolü: geçti ===", flush=True)
        old["ver"] = pd.concat([old["ver"][old["ver"]["kat"] != fold], pd.DataFrame([ver])], ignore_index=True)
        old["w"] = pd.concat([old["w"][old["w"]["kat"] != fold], weight_table(train).assign(kat=fold)], ignore_index=True)
        for name in CONFIGS:
            t0 = time.time()
            row = {"kat": fold, "yapilandirma": name}
            if CONFIGS[name] is None:
                proba = predict_cells(current, test_cells)
                row.update({"oznitelik_sayisi": len(current[1]["oznitelikler"]), "cikarilan": ""})
            else:
                cols, dropped = config_features(name, sets, train)
                params, grid, oof = tune(train, cols, f"{fold}/{name}")
                model = fit_weighted(train, cols, params)
                proba = proba_of(model, test_cells, cols)
                row.update({"oznitelik_sayisi": len(cols), "cikarilan": ";".join(dropped),
                            "learning_rate": params["learning_rate"], "max_leaf_nodes": params["max_leaf_nodes"],
                            "ic_cv_f1": float(grid["ic_cv_agirlikli_makro_f1"].max())})
                old["grid"] = pd.concat([old["grid"][(old["grid"]["kat"] != fold) | (old["grid"]["yapilandirma"] != name)],
                                         grid.assign(kat=fold, yapilandirma=name)], ignore_index=True)
                typ = source_typing(train, oof, fold, name)
                old["typ"] = pd.concat([old["typ"][(old["typ"]["kat"] != fold) | (old["typ"]["yapilandirma"] != name)],
                                        typ], ignore_index=True)
            metrics, res = evaluate_fold(fold, test_cells, test_imgs, proba)
            row.update(metrics)
            row["sure_sn"] = round(time.time() - t0)
            pred_dir = MULTICENTER_DIR if fold != "B" else od
            res.drop(columns=[c for c in ("yol",) if c in res]).to_csv(
                pred_dir / f"loso_tahmin_{fold}_{name}.csv", index=False, encoding="utf-8")
            old["res"] = pd.concat([old["res"][(old["res"]["kat"] != fold) | (old["res"]["yapilandirma"] != name)],
                                    pd.DataFrame([row])], ignore_index=True)
            for k in ("res", "ver", "grid", "typ", "w"):
                old[k].to_csv(paths[k], index=False, encoding="utf-8")
            show = {k: v for k, v in metrics.items() if k.split("_", 1)[1] in (
                "ALL_tipleme", "healthy_ozgulluk", "dengeli_dogruluk", "goruntu_makro_f1", "duyarlilik_MYO",
                "duyarlilik_LYT", "duyarlilik_NGS", "hucre_makro_f1")}
            print(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in show.items()) + f" ({row['sure_sn']} sn)", flush=True)
        print(f"Kat {fold} süresi: {time.time() - start:.0f} sn", flush=True)
    return old["res"]


def summary_table(res: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Yapılandırma başına tek satırlık özet (Kat A, A_idb_dahil, B, C ölçüleri yan yana) ve ön kayıtlı skor / kısıt.

    Parametreler:
        res: loso_sonuclari.csv (None: dosyadan).

    Döndürür:
        yapilandirma, ölçüler, skor, kisit_saglandi, secilen.
    """
    res = pd.read_csv(out_dir() / LOSO_CSV) if res is None else res
    rows = []
    for name in CONFIGS:
        sub = res[res["yapilandirma"] == name]
        row = {"yapilandirma": name, "ad": CONFIG_LABELS[name]}
        for _, r in sub.iterrows():
            row.update({k: v for k, v in r.items() if isinstance(k, str) and k[:2] in ("A_", "Ai", "B_", "C_")
                        and pd.notna(v)})
            if r["kat"] in MAIN_FOLDS:
                row[f"oznitelik_sayisi_{r['kat']}"] = r["oznitelik_sayisi"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table["skor"] = (table["A_ALL_tipleme"] + table["B_dengeli_dogruluk"]) / 2
    base = table.loc[table["yapilandirma"] == "i_mevcut", "C_goruntu_makro_f1"].iat[0]
    table["kisit_saglandi"] = table["C_goruntu_makro_f1"] >= base - MULTICENTER_MAX_KAGGLE_DROP
    ok = table[table["kisit_saglandi"]].sort_values(["skor", "C_goruntu_makro_f1"], ascending=False)
    chosen = ok["yapilandirma"].iat[0] if len(ok) else "i_mevcut"
    table["secilen"] = table["yapilandirma"] == chosen
    return table


def run_summary() -> pd.DataFrame:
    """Özet tabloyu ve seçimi yazdırır / kaydeder (outputs/multicenter/loso_ozet.csv)."""
    table = summary_table()
    table.to_csv(out_dir() / "loso_ozet.csv", index=False, encoding="utf-8")
    cols = ["ad", "A_ALL_tipleme", "A_healthy_ozgulluk", "A_duyarlilik", "Aidb_ALL_tipleme", "B_dengeli_dogruluk",
            "B_duyarlilik_MYO", "B_duyarlilik_LYT", "B_duyarlilik_NGS", "C_goruntu_makro_f1", "C_hucre_makro_f1",
            "skor", "kisit_saglandi", "secilen"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# 5) Laboratuvar kısayolu testi
# ---------------------------------------------------------------------------

def run_lab_shortcut() -> pd.DataFrame:
    """
    Laboratuvar kısayolu: her katın eğitim verisinde, yalnızca özniteliklerden hücrenin KAYNAĞINI tahmin eden HGB
    (iç CV katları, katlar dışı). Ağırlıklar kaynak × sınıf grupları eşit olacak şekilde (sınıf karışımı kaynağı ele
    vermesin). Ölçüler: kaynaklar arası makro AUC (bire-karşı-diğerleri) ve sınıf içi kaynak AUC'si (aynı sınıftaki
    hücrelerde; biyolojiden bağımsız laboratuvar imzası). Öznitelik kümeleri: temel 58, biyolojik dahil 73, (iv) ayıklı.

    Döndürür:
        kat, oznitelik_kumesi, n_oznitelik, kaynak_makro_auc, sinif_ici_kaynak_auc_<sınıf> tablosu
        (outputs/multicenter/laboratuvar_kisayolu.csv).
    """
    cells, images = build_tables()
    sets = feature_sets(cells)
    rows = []
    params = {"learning_rate": 0.1, "max_iter": 200, "early_stopping": False}
    for fold in MAIN_FOLDS:
        train, _, _ = fold_split(cells, images, fold)
        srcs = sorted(s for s in SOURCES if (train["kaynak"] == s).any())   # roc_auc_score sıralı etiket ister
        key = train["kaynak"] + "|" + train["etiket"]
        w = (1.0 / key.map(key.value_counts())).to_numpy()
        w = w / w.mean()
        folds = inner_folds(train)
        cols_iv, _ = config_features("iv_bio_kaynak_ayikli", sets, train)
        for set_name, cols in [("temel_58", sets["temel"]), ("biyolojik_dahil_73", sets["bio"]),
                               ("iv_ayikli", cols_iv), ("yalniz_biyolojik_15", BIO_FEATURES)]:
            oof = np.zeros((len(train), len(srcs)))
            for tr, va in folds:
                model = make_model(params)
                with threadpool_limits(limits=N_JOBS):
                    model.fit(train.iloc[tr][cols].to_numpy(np.float64), train["kaynak"].iloc[tr].to_numpy(),
                              model__sample_weight=w[tr])
                oof[va] = proba_of(model, train.iloc[va], cols, srcs)
            y = train["kaynak"].to_numpy()
            if len(srcs) == 2:
                macro = roc_auc_score(y == srcs[1], oof[:, 1])
            else:
                macro = roc_auc_score(y, oof, multi_class="ovr", labels=srcs, average="macro")
            row = {"kat": fold, "oznitelik_kumesi": set_name, "n_oznitelik": len(cols), "kaynaklar": "+".join(srcs),
                   "kaynak_makro_auc": float(macro)}
            for cls in CLASS_NAMES:
                sel = (train["etiket"] == cls).to_numpy()
                present = [s for s in srcs if (y[sel] == s).sum() >= 20]
                if len(present) < 2:
                    continue
                aucs = [roc_auc_score(y[sel] == s, oof[sel, srcs.index(s)]) for s in present]
                row[f"sinif_ici_kaynak_auc_{cls}"] = float(np.mean(aucs))
            rows.append(row)
            print(f"  [{fold}] {set_name}: kaynak AUC {macro:.3f} | " + ", ".join(
                f"{k[-7:]}={v:.3f}" for k, v in row.items() if k.startswith("sinif_ici")), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir() / "laboratuvar_kisayolu.csv", index=False, encoding="utf-8")
    return table


# ---------------------------------------------------------------------------
# 6) Son model (v3)
# ---------------------------------------------------------------------------

class TemperatureScaledModel:
    """
    Temperature kalibrasyonlu sarmalayıcı: p' = softmax(log p / T). Tek parametre; en olası sınıf DEĞİŞMEZ.

    sklearn'ün CalibratedClassifierCV'si Pipeline'a örnek ağırlığı iletmediği için (kaynak dengesi ağırlıkla
    sağlanıyor) kalibrasyon burada elle yapılır: T, iç CV katlar dışı olasılıklarında ağırlıklı log loss'u en aza
    indiren değerdir.
    """

    def __init__(self, base, temperature: float):
        """
        Parametreler:
            base: Eğitilmiş model (predict_proba, classes_).
            temperature: Sıcaklık (> 1: olasılıklar yumuşar).
        """
        self.base = base
        self.temperature = float(temperature)
        self.classes_ = base.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Kalibre olasılıklar.

        Parametreler:
            X: Öznitelik matrisi.

        Döndürür:
            (n, sınıf) olasılık.
        """
        return temperature_scale(self.base.predict_proba(X), self.temperature)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        En olası sınıf.

        Parametreler:
            X: Öznitelik matrisi.

        Döndürür:
            Etiketler.
        """
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(axis=1)]


def temperature_scale(p: np.ndarray, t: float) -> np.ndarray:
    """
    softmax(log p / t).

    Parametreler:
        p: Olasılıklar.
        t: Sıcaklık.

    Döndürür:
        Ölçeklenmiş olasılıklar.
    """
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(oof: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Ağırlıklı log loss'u en aza indiren sıcaklık (log T ∈ [−3, 3]).

    Parametreler:
        oof: Katlar dışı olasılıklar (CLASS_NAMES sırası).
        y: Etiketler.
        w: Ağırlıklar.

    Döndürür:
        T.
    """
    idx = np.array([CLASS_NAMES.index(c) for c in y])

    def loss(logt: float) -> float:
        p = temperature_scale(oof, np.exp(logt))
        return float(-(w * np.log(np.clip(p[np.arange(len(y)), idx], 1e-12, 1))).sum() / w.sum())

    return float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))


def balanced_thresholds(img: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """
    Risk eşiklerini çok kaynaklı veriden seçer: risk_analysis.select_thresholds ile aynı iki hedef (duyarlılık ≥
    RISK_TARGET_SENSITIVITY olan en büyük Düşük/Orta eşiği; özgüllük ≥ RISK_TARGET_SPECIFICITY olan en küçük
    Orta/Yüksek eşiği), ama duyarlılık / özgüllük KAYNAK ORTALAMASI olarak: tek hücreli Münih görüntüleri sayıca
    baskın olup eşiği belirlemesin. Hedef sağlanamazsa en yakın aday seçilir ve tabloda belirtilir.

    Parametreler:
        img: anormal_oran (risk skoru), anormal (bool), kaynak sütunlu görüntü tablosu (hücresizler hariç).

    Döndürür:
        (düşük/orta, orta/yüksek, aday tablosu).
    """
    from src.risk_analysis import threshold_candidates

    cands = threshold_candidates(img["anormal_oran"].to_numpy())
    # Üst uç adayı (en büyük skor + 0.01) çıkarılır: o eşikte hiçbir görüntü "Yüksek" olamaz (özgüllük kendiliğinden 1).
    # Tek hücreli görüntülerde hedef özgüllük sağlanamazsa en az skor = 1 olanlar "Yüksek" kalsın.
    cands = cands[cands <= 1.0]
    rows = []
    for t in cands:
        sens, spec = [], []
        for _, sub in img.groupby("kaynak"):
            pos, neg = sub[sub["anormal"]], sub[~sub["anormal"]]
            if len(pos):
                sens.append(float((pos["anormal_oran"] >= t).mean()))
            if len(neg):
                spec.append(float((neg["anormal_oran"] < t).mean()))
        rows.append({"esik": t, "duyarlilik": np.mean(sens), "ozgulluk": np.mean(spec)})
    table = pd.DataFrame(rows)
    ok_s, ok_p = table[table["duyarlilik"] >= RISK_TARGET_SENSITIVITY], table[table["ozgulluk"] >= RISK_TARGET_SPECIFICITY]
    low = float(ok_s["esik"].max()) if len(ok_s) else float(table.loc[table["duyarlilik"].idxmax(), "esik"])
    high = float(ok_p["esik"].min()) if len(ok_p) else float(table.loc[table["ozgulluk"].idxmax(), "esik"])
    table["hedef_duyarlilik_saglandi"] = len(ok_s) > 0
    table["hedef_ozgulluk_saglandi"] = len(ok_p) > 0
    if high < low:
        low, high = high, low
    return round(low, 4), round(high, 4), table


def run_final_model() -> None:
    """
    Seçilen yapılandırmayla son model (v3): üç kaynağın TÜM etiketli hücreleri (Kaggle geliştirme + test, ALL-IDB,
    Münih örneklemi), iç CV ile hiperparametre, temperature kalibrasyonu, kaynak dengeli risk eşikleri. Kaydedilir:
    models/losemi_siniflandirici_v3_kalibre.joblib (+ .json). Varsayılan model (config.ACTIVE_MODEL_FILE) DEĞİŞMEZ.
    Dış test seti kalmaz: başarı tahmini LOSO sonuçlarıdır.
    """
    from src.classifier import save_model
    from src.external_validation import image_results

    table = summary_table()
    chosen = table.loc[table["secilen"], "yapilandirma"].iat[0]
    if CONFIGS[chosen] is None:
        print("Seçilen yapılandırma mevcut model: v3 eğitilmedi.")
        return
    print(f"Seçilen yapılandırma: {chosen}", flush=True)
    cells, images = build_tables()
    sets = feature_sets(cells)
    train = cells[cells["etiketli"]].reset_index(drop=True)
    cols, dropped = config_features(chosen, sets, train)
    params, grid, oof = tune(train, cols, f"v3/{chosen}")
    base = fit_weighted(train, cols, params)
    w = source_weights(train)
    temp = fit_temperature(oof, train["etiket"].to_numpy(), w)
    model = TemperatureScaledModel(base, temp)

    # Risk eşikleri: tüm hücrelerin (etiketsizler dahil) katlar dışı tahminleri → görüntü risk skoru
    folds = inner_folds(train)
    group_fold = {}
    for k, (_, va) in enumerate(folds):
        group_fold.update(dict.fromkeys(train["grup"].iloc[va], k))
    rng = np.random.default_rng(RANDOM_STATE)
    unseen = sorted(set(cells["grup"]) - set(group_fold))   # yalnızca etiketsiz hücresi olan grup (olmamalı)
    group_fold.update({g: int(rng.integers(len(folds))) for g in unseen})
    cell_fold = cells["grup"].map(group_fold).to_numpy()
    proba = np.zeros((len(cells), len(CLASS_NAMES)))
    for k, (tr, _) in enumerate(folds):
        m = fit_weighted(train.iloc[tr], cols, params)
        sel = cell_fold == k
        proba[sel] = proba_of(m, cells[sel], cols)
    img = image_results(cells, temperature_scale(proba, temp), images[["id"]])
    img = img.merge(images[["id", "kaynak", "etiket"]], on="id")
    img = img[img[[f"n_{c}" for c in CLASS_NAMES]].sum(axis=1) > 0]   # hücresiz görüntüler eşik seçimine girmez
    img["anormal"] = img["etiket"].isin(["ALL", "AML"])
    low, high, cand = balanced_thresholds(img)
    cand.to_csv(out_dir() / "v3_risk_esik_adaylari.csv", index=False, encoding="utf-8")
    per_source = []
    for src, sub in img.groupby("kaynak"):
        pos, neg = sub[sub["anormal"]], sub[~sub["anormal"]]
        per_source.append({"kaynak": src, "goruntu": len(sub), "lösemi_dusuk_orani": float((pos["anormal_oran"] < low).mean()),
                           "healthy_yuksek_orani": float((neg["anormal_oran"] >= high).mean())})
    pd.DataFrame(per_source).to_csv(out_dir() / "v3_risk_esikleri_kaynak.csv", index=False, encoding="utf-8")

    meta = {"model": "hgb", "parametreler": params, "oznitelikler": cols, "cikarilan_oznitelikler": dropped,
            "siniflar": CLASS_NAMES, "segmentasyon": "kmeans", "yapilandirma": chosen, "veri_ayrimi": "yok (üç kaynağın tamamı)",
            "kalibrasyon": "temperature (elle; ağırlıklı iç CV katlar dışı)", "sicaklik": temp,
            "kaynak_agirligi_tavani": MULTICENTER_SOURCE_CAP, "ic_cv_f1": float(grid["ic_cv_agirlikli_makro_f1"].max()),
            "risk_esikleri": [low, high], "risk_esikleri_varsayilan_model": list(RISK_THRESHOLDS),
            "egitim_hucre_sayisi": int(len(train)),
            "egitim_kaynak_sinif": {f"{s}|{c}": int(n) for (s, c), n in train.groupby(["kaynak", "etiket"]).size().items()},
            "dis_test_seti": "YOK: üç kaynağın tamamı eğitimde; başarı tahmini LOSO sonuçlarıdır (loso_ozet.csv)",
            "varsayilan_model_degil": True}
    save_model(model, MULTICENTER_MODEL_FILE, meta)
    pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in meta.items()}]
                 ).to_csv(out_dir() / "v3_model_ozeti.csv", index=False, encoding="utf-8")
    grid.to_csv(out_dir() / "v3_ic_cv_izgara.csv", index=False, encoding="utf-8")
    print(f"v3 kaydedildi: models/{MULTICENTER_MODEL_FILE}.joblib | T={temp:.3f} | risk eşikleri {low} / {high} | "
          f"{len(cols)} öznitelik, parametreler {params}")
    print(pd.DataFrame(per_source).round(3).to_string(index=False))


def permutation_importance_table(chosen: str) -> pd.DataFrame:
    """
    Seçilen yapılandırmanın öznitelik önemi: tüm etiketli verinin ilk iç katı dışarıda bırakılarak eğitilen modelde,
    dışarıdaki katta permütasyon önemi (ağırlıklı makro F1 düşüşü, 3 tekrar).

    Parametreler:
        chosen: Yapılandırma adı.

    Döndürür:
        oznitelik, onem, onem_std, kategori.
    """
    cells, _ = build_tables()
    sets = feature_sets(cells)
    train = cells[cells["etiketli"]].reset_index(drop=True)
    cols, _ = config_features(chosen, sets, train)
    from src.config import MODELS_DIR

    meta_path = MODELS_DIR / f"{MULTICENTER_MODEL_FILE}.json"
    params = MULTICENTER_GRID[0]
    if meta_path.exists():
        params = json.loads(meta_path.read_text(encoding="utf-8")).get("parametreler", params)
    tr, va = inner_folds(train)[0]
    model = fit_weighted(train.iloc[tr], cols, params)
    val = train.iloc[va].reset_index(drop=True)
    w = source_weights(val)
    y = val["etiket"].to_numpy()
    X = val[cols].to_numpy(np.float64)

    def score(Xm: np.ndarray) -> float:
        with threadpool_limits(limits=N_JOBS):
            p = model.predict(Xm)
        return f1_score(y, p, labels=CLASS_NAMES, average="macro", sample_weight=w)

    base = score(X)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for j, col in enumerate(cols):
        drops = []
        for _ in range(3):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - score(Xp))
        rows.append({"oznitelik": col, "onem": float(np.mean(drops)), "onem_std": float(np.std(drops)),
                     "kategori": feature_category(col)})
    return pd.DataFrame(rows).sort_values("onem", ascending=False)


# ---------------------------------------------------------------------------
# 7) Görseller
# ---------------------------------------------------------------------------

def style_axis(ax: plt.Axes) -> None:
    """Üst / sağ çerçeveyi kaldırır, yazı renklerini ayarlar."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_loso(table: pd.DataFrame) -> None:
    """
    LOSO sonuç grafiği: dört ana ölçü × yapılandırma (mevcut modelle karşılaştırmalı) → loso_sonuclari.png.

    Parametreler:
        table: summary_table çıktısı.
    """
    metrics = [("A_ALL_tipleme", "Kat A: ALL-IDB ALL\ngörüntüsü → ALL"),
               ("A_healthy_ozgulluk", "Kat A: ALL-IDB Healthy\nözgüllüğü (risk)"),
               ("B_dengeli_dogruluk", "Kat B: Münih TEST\ndengeli doğruluk"),
               ("C_goruntu_makro_f1", "Kat C: Kaggle test\ngörüntü makro F1")]
    colors = ["#9a9890", "#7fb2e5", "#2a78d6", "#0d3f7a"]
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    x = np.arange(len(metrics))
    width = 0.8 / len(table)
    for k, (_, r) in enumerate(table.iterrows()):
        vals = [r[m] for m, _ in metrics]
        bars = ax.bar(x + (k - (len(table) - 1) / 2) * width, vals, width * 0.92, color=colors[k % len(colors)],
                      label=r["ad"] + ("  ← seçilen" if r["secilen"] else ""))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=TEXT_PRIMARY)
    ax.axhline(1 / 3, color="#c0392b", lw=0.8, ls=":")
    ax.text(len(metrics) - 0.5, 1 / 3 + 0.01, "şans (3 sınıf)", fontsize=7, color="#c0392b", ha="right")
    ax.set_xticks(x, [m[1] for m in metrics], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Oran / skor", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("Laboratuvar dışarıda bırakma (LOSO): test kaynağı eğitimde yok", fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.2))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "loso_sonuclari")


def plot_confusions() -> None:
    """
    Kat bazında karışıklık matrisleri (yalnızca sayılar; görüntü yok) → karisiklik_kat_<A/B/C>.png. Satırlar gerçek
    sınıf (Münih'te Münih sınıfı), sütunlar tahmin; her yapılandırma bir panel.
    """
    from src.classifier import NO_CELL

    od = out_dir()
    for fold in MAIN_FOLDS:
        fig, axes = plt.subplots(1, len(CONFIGS), figsize=(3.1 * len(CONFIGS), 2.9), facecolor=SURFACE)
        for ax, name in zip(axes, CONFIGS):
            path = (MULTICENTER_DIR if fold != "B" else od) / f"loso_tahmin_{fold}_{name}.csv"
            if not path.exists():
                ax.axis("off")
                continue
            res = pd.read_csv(path, keep_default_na=False)
            if fold == "A":
                rows_key, pred_key, rows = "etiket", "goruntu_tahmin", ["ALL", "Healthy"]
            elif fold == "B":
                rows_key, pred_key, rows = "munih_sinif", "merkez_tahmin", list(MUNICH_CLASSES)
            else:
                res = res[~res["kenara_degiyor"].astype(str).str.lower().eq("true")]
                rows_key, pred_key, rows = "etiket", "tahmin", CLASS_NAMES
            cols = CLASS_NAMES + ([NO_CELL] if fold != "C" and (res[pred_key] == NO_CELL).any() else [])
            t = pd.crosstab(res[rows_key], res[pred_key]).reindex(index=rows, columns=cols, fill_value=0)
            norm = t.div(t.sum(axis=1).replace(0, 1), axis=0)
            ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(len(rows)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{t.iat[i, j]}\n%{100 * norm.iat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if norm.iat[i, j] > 0.55 else TEXT_PRIMARY)
            ax.set_xticks(range(len(cols)), cols, fontsize=7)
            ax.set_yticks(range(len(rows)), [f"{r} (→{MUNICH_CLASSES[r]})" if fold == "B" else r for r in rows], fontsize=7)
            ax.set_title(CONFIG_LABELS[name], fontsize=8, color=TEXT_PRIMARY, loc="left")
        unit = {"A": "görüntü tahmini", "B": "merkez hücre", "C": "hücre (kenara değmeyen)"}[fold]
        fig.suptitle(f"Kat {fold}: {FOLDS[fold]} — {unit}", fontsize=9, color=TEXT_PRIMARY, x=0.01, ha="left")
        fig.tight_layout()
        save_figure(fig, OUT_STAGE, f"karisiklik_kat_{fold}")


def plot_importance(imp: pd.DataFrame, top: int = 25) -> None:
    """
    Öznitelik önemi (biyolojik öznitelikler vurgulu) → oznitelik_onemi.png.

    Parametreler:
        imp: permutation_importance_table çıktısı.
        top: Gösterilecek öznitelik sayısı.
    """
    sub = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.26 * top + 1), facecolor=SURFACE)
    colors = [CLASS_COLORS["AML"] if k == "biyolojik" else "#b9b7b0" for k in sub["kategori"]]
    ax.barh(sub["oznitelik"], sub["onem"], xerr=sub["onem_std"], color=colors, error_kw={"lw": 0.6})
    ax.set_xlabel("Permütasyon önemi (ağırlıklı makro F1 düşüşü)", fontsize=8, color=TEXT_SECONDARY)
    bio = imp.reset_index(drop=True)
    ranks = [int(i) + 1 for i in bio.index[bio["kategori"] == "biyolojik"]]
    ax.set_title("Öznitelik önemi, (iii) 73 öznitelik — turuncu: yeni biyolojik\n"
                 f"Biyolojik özniteliklerin sırası: {', '.join(map(str, ranks))} / {len(bio)}",
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    ax.tick_params(axis="y", labelsize=7)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "oznitelik_onemi")


def plot_lab_shortcut(lab: pd.DataFrame) -> None:
    """
    Laboratuvar kısayolu testi grafiği (kaynak AUC'si; 0.5 = kaynak ayırt edilemiyor) → laboratuvar_kisayolu.png.

    Parametreler:
        lab: run_lab_shortcut çıktısı.
    """
    sets = list(dict.fromkeys(lab["oznitelik_kumesi"]))
    fig, axes = plt.subplots(1, len(MAIN_FOLDS), figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    palette = ["#9a9890", "#2a78d6", "#1baf7a", "#eb6834"]
    for ax, fold in zip(axes, MAIN_FOLDS):
        sub = lab[lab["kat"] == fold].set_index("oznitelik_kumesi")
        metrics = ["kaynak_makro_auc"] + [f"sinif_ici_kaynak_auc_{c}" for c in CLASS_NAMES
                                          if f"sinif_ici_kaynak_auc_{c}" in sub and sub[f"sinif_ici_kaynak_auc_{c}"].notna().any()]
        x = np.arange(len(metrics))
        width = 0.8 / len(sets)
        for k, s in enumerate(sets):
            vals = [sub.loc[s, m] if s in sub.index else np.nan for m in metrics]
            ax.bar(x + (k - (len(sets) - 1) / 2) * width, vals, width * 0.92, color=palette[k], label=s)
        ax.axhline(0.5, color=TEXT_SECONDARY, lw=0.7, ls=":")
        ax.set_xticks(x, ["tüm hücreler"] + [f"sınıf içi\n{m.rsplit('_', 1)[1]}" for m in metrics[1:]], fontsize=7)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"Kat {fold} eğitimi: {sub['kaynaklar'].iat[0]}", fontsize=8, color=TEXT_PRIMARY, loc="left")
        style_axis(ax)
    axes[0].set_ylabel("Kaynak tahmini AUC", fontsize=8, color=TEXT_SECONDARY)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22))
    fig.suptitle("Laboratuvar kısayolu: öznitelikler hücrenin hangi laboratuvardan geldiğini ne kadar ele veriyor?",
                 fontsize=9, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "laboratuvar_kisayolu")


def plot_feature_source_auc(table: pd.DataFrame) -> None:
    """
    Öznitelik başına sınıf içi kaynak AUC'si (biyolojik vurgulu, eşik çizgili) → oznitelik_kaynak_auc.png.

    Parametreler:
        table: oznitelik_kisayol.csv.
    """
    t = table.sort_values("kaynak_auc")
    colors = [CLASS_COLORS["AML"] if k == "biyolojik" else ("#c0392b" if k == "mutlak zemin rengi" else "#b9b7b0")
              for k in t["kategori"]]
    fig, ax = plt.subplots(figsize=(7, 0.15 * len(t) + 1), facecolor=SURFACE)
    ax.barh(t["oznitelik"], t["kaynak_auc"], color=colors)
    ax.axvline(MULTICENTER_SOURCE_AUC_MAX, color=TEXT_PRIMARY, lw=0.8, ls="--")
    ax.set_xlim(0.5, 1.0)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("Sınıf içi kaynak AUC'si (en büyük sınıf × kaynak)", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(f"Turuncu: biyolojik, kırmızı: mutlak zemin rengi; kesikli: eşik {MULTICENTER_SOURCE_AUC_MAX}",
                 fontsize=8, color=TEXT_PRIMARY, loc="left")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STAGE, "oznitelik_kaynak_auc")


def run_figures() -> None:
    """Tüm Adım 9 figürlerini üretir; öznitelik önemini de CSV'ye yazar."""
    table = summary_table()
    plot_loso(table)
    plot_confusions()
    # Önem, biyolojik öznitelikleri içeren (iii) yapılandırmasında ölçülür (yeni özniteliklerin katkısı görünsün);
    # parametreler v3'ünkiler (yoksa ızgaranın ilki).
    imp_cfg = "iii_cok_merkezli_bio"
    imp = permutation_importance_table(imp_cfg)
    imp.assign(yapilandirma=imp_cfg).to_csv(out_dir() / "oznitelik_onemi.csv", index=False, encoding="utf-8")
    plot_importance(imp)
    if (out_dir() / "laboratuvar_kisayolu.csv").exists():
        plot_lab_shortcut(pd.read_csv(out_dir() / "laboratuvar_kisayolu.csv"))
    if (out_dir() / "oznitelik_kisayol.csv").exists():
        plot_feature_source_auc(pd.read_csv(out_dir() / "oznitelik_kisayol.csv"))
    print("Figürler: outputs/multicenter/")


PARTS = {"oznitelik": run_features, "kisayol": run_shortcut, "kural": write_rule,
         "loso": lambda: run_loso(sys.argv[2:] or None), "ozet": run_summary, "laboratuvar": run_lab_shortcut,
         "son_model": run_final_model, "gorseller": run_figures}


def main() -> None:
    """Komut satırı: python -m src.multicenter <bölüm> [katlar]."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    part = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    start = time.time()
    if part == "hepsi":
        for p in ("oznitelik", "kisayol", "kural", "loso", "ozet", "laboratuvar", "son_model", "gorseller"):
            print(f"\n##### {p} #####", flush=True)
            PARTS[p]() if p != "loso" else run_loso()
    else:
        PARTS[part]()
    print(f"\nToplam süre: {time.time() - start:.0f} sn")


if __name__ == "__main__":
    main()
