"""
Adım 5 düzeltmesi - ALL-IDB kaynak grupları ve sızıntısız veri ayrımı (veri_bolumu_v2.csv).

Sorun: Adım 1'in kopya tespiti (pHash + ORB) aynı ALL-IDB1 yaymasının tam boy hâlini, 256x256'ya sıkıştırılmış
hâlini, kırpımlarını ve ondan kesilen ALL-IDB2 hücre kırpımlarını her zaman aynı gruba (grup_id) bağlayamadı.
Bu yüzden eski kilitli ayrımda (veri_bolumu.csv) aynı yaymanın türevleri hem geliştirme hem test tarafına düştü.

Çözüm: Kaggle ↔ ALL-IDB eşleşmeleri (src/allidb_overlap.py çıktıları) üzerinden grafik kurulur:
- düğümler: Kaggle görüntüleri + ALL-IDB1 "kaynak yayma" düğümleri (ALL-IDB1'in kendi içindeki kopyaları tek düğüm),
- kenarlar: aynı grup_id (Adım 1 kopyaları), kabul edilmiş her eşleşme (tam / sıkıştırılmış / kırpım / hücre kırpımı
  → o ALL-IDB1 yayması; ALL-IDB2 kırpımı → kaynak ALL-IDB1 yayması).
Bağlantılı bileşenler kaynak_grup_id olur. Eski grup_id silinmez; kaynak_grup_id grup_id'lerin birleşimidir.
Her görüntüye kaynak sütunu eklenir: bileşende bir ALL-IDB düğümü varsa "ALL-IDB", yoksa "bilinmiyor".

Yeni ayrım kaynak_grup_id ile yapılır, katman = etiket × kaynak (ALL-IDB kaynaklı ve kaynağı bilinmeyen görüntüler
test ve katlarda orantılı kalsın; kaynak kısayolu testi her iki alt kümeyi de gerektiriyor).
veri_bolumu_v2.csv bir kez üretilir, sonra ASLA yeniden üretilmez; eski veri_bolumu.csv'ye dokunulmaz.

Çalıştırma (proje kökünden):
    python -m src.source_groups
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.allidb_overlap import MATCH_CSV, STATUS_CSV, load_saved_matches
from src.config import (
    CV_FOLDS,
    DATASET_INDEX_CSV,
    RANDOM_STATE,
    SPLIT_CSV,
    SPLIT_CSV_V2,
    TEST_FRACTION_FOLDS,
)
from src.dataset import assign_groups

SOURCE_ALLIDB = "ALL-IDB"
SOURCE_UNKNOWN = "bilinmiyor"


def idb1_root_map(status: pd.DataFrame) -> dict[str, list[str]]:
    """
    Her ALL-IDB dosyasını kaynak ALL-IDB1 düğüm(ler)ine eşler.

    - ALL-IDB1 yayması: kendi iç kopya grubu (idb1_ic_grup; Im093_0 ↔ Im108_0 gibi kopyalar tek düğüm).
    - ALL-IDB2 kırpımı: kaynak yayma(lar)ının (kaynak_idb1, ";" ile ayrılmış) düğümleri. Kaynağı bulunamayan
      kırpım (olmamalı; 260/260 bulundu) kendi başına bir düğüm olur.

    Parametreler:
        status: allidb_goruntu_durumu.csv tablosu.

    Döndürür:
        {idb_dosya: [düğüm adı, ...]} sözlüğü.
    """
    idb1 = status[status["idb_set"] == "IDB1"]
    node_of_idb1 = {r.idb_dosya: f"IDB1_grup_{int(r.idb1_ic_grup)}" for r in idb1.itertuples(index=False)}
    roots = dict((f, [n]) for f, n in node_of_idb1.items())
    for r in status[status["idb_set"] == "IDB2"].itertuples(index=False):
        parents = [p for p in str(r.kaynak_idb1).split(";") if p and p != "nan"]
        roots[r.idb_dosya] = [node_of_idb1[p] for p in parents] or [f"IDB2_{r.idb_dosya}"]
    return roots


def build_source_groups(index: pd.DataFrame, matches: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle grupları ile ALL-IDB eşleşmelerini birleştirip kaynak_grup_id ve kaynak sütunlarını hesaplar.

    Yalnızca kabul edilmiş eşleşmeler (kabul=True) kullanılır; tek başına şablon taraması (kabul=False) bağlantı
    kurmaz (allidb_overlap: geometrik destek yok, etiketle çelişen eşleşmeler).

    Parametreler:
        index: Veri seti indeksi (dosya_yolu, etiket, grup_id, tip).
        matches: load_saved_matches çıktısı (kaggle_yolu sütunuyla).
        status: allidb_goruntu_durumu.csv tablosu.

    Döndürür:
        index sırasında: dosya_yolu, etiket, grup_id, tip, kaynak_grup_id, kaynak, idb1_kaynaklari
        (görüntünün bağlandığı ALL-IDB1 düğümleri, ";" ile).
    """
    acc = matches[matches["kabul"].astype(bool)]
    if acc["kaggle_yolu"].isna().any():
        raise ValueError("Eşleşme tablosunda veri seti indeksinde bulunmayan Kaggle dosyası var.")
    roots = idb1_root_map(status)
    n = len(index)
    pos = {p: i for i, p in enumerate(index["dosya_yolu"])}
    node_names = sorted({node for f in acc["idb_dosya"] for node in roots[f]})
    node_pos = {name: n + k for k, name in enumerate(node_names)}

    pairs = []
    # Adım 1 kopya grupları: grubun her görüntüsü ilk görüntüsüne bağlanır
    first = index.groupby("grup_id")["dosya_yolu"].transform("first")
    pairs += [(pos[a], pos[b]) for a, b in zip(index["dosya_yolu"], first) if a != b]
    # ALL-IDB eşleşmeleri: Kaggle görüntüsü ↔ kaynak ALL-IDB1 düğümü
    linked: dict[int, set[str]] = {}
    for r in acc.itertuples(index=False):
        for node in roots[r.idb_dosya]:
            pairs.append((pos[r.kaggle_yolu], node_pos[node]))
            linked.setdefault(pos[r.kaggle_yolu], set()).add(node)

    comp = assign_groups(n + len(node_names), pairs)
    allidb_comps = set(comp[n:])
    out = index[["dosya_yolu", "etiket", "grup_id", "tip"]].copy().reset_index(drop=True)
    # Kimlikleri yalnızca görüntüler üzerinden 0'dan ardışık yeniden numarala
    remap = {c: k for k, c in enumerate(dict.fromkeys(comp[:n]))}
    out["kaynak_grup_id"] = [remap[c] for c in comp[:n]]
    out["kaynak"] = np.where(np.isin(comp[:n], list(allidb_comps)), SOURCE_ALLIDB, SOURCE_UNKNOWN)
    # Bileşen düzeyinde ALL-IDB1 düğümleri (doğrudan eşleşmeyen grup üyeleri de bileşenin kaynağını taşır)
    comp_nodes: dict[int, set[str]] = {}
    for i, nodes in linked.items():
        comp_nodes.setdefault(out.at[i, "kaynak_grup_id"], set()).update(nodes)
    out["idb1_kaynaklari"] = [";".join(sorted(comp_nodes.get(g, ()))) for g in out["kaynak_grup_id"]]
    return out


def leaky_images(groups: pd.DataFrame, split: pd.DataFrame, group_col: str = "kaynak_grup_id") -> pd.Series:
    """
    Eski ayrımda kaynak grubu iki tarafa (geliştirme ve test) düşmüş görüntüleri işaretler.

    Bir test görüntüsü, kaynak grubundaki herhangi bir görüntü geliştirme setindeyse "sızıntılı" sayılır
    (aynı ALL-IDB yayması modelin eğitiminde görüldü).

    Parametreler:
        groups: build_source_groups çıktısı.
        split: Eski veri ayrımı (dosya_yolu, bolum).
        group_col: Kaynak grup sütunu.

    Döndürür:
        groups sırasında bool seri (True = kaynak grubu iki bölümde).
    """
    b = groups[["dosya_yolu", group_col]].merge(split[["dosya_yolu", "bolum"]], on="dosya_yolu", how="left")
    n_parts = b.groupby(group_col)["bolum"].transform("nunique")
    return pd.Series((n_parts > 1).to_numpy(), index=groups.index)


def test_fold_score(test: pd.DataFrame, strata: pd.Series) -> tuple[int, int]:
    """
    Aday test katının yapısal puanı: (katmanlardaki en küçük kaynak grubu sayısı, toplam kaynak grubu sayısı).

    ALL-IDB bileşenleri büyük (bir ALL-IDB1 yaymasının ~10 türevi; en büyüğü 53 görüntü). İlk dış katta ALL-IDB
    kaynaklı ALL'ın 56 test görüntüsü yalnızca 2 kaynak grubundan geliyordu (biri 53 görüntülük bileşen): o alt
    kümedeki başarı fiilen tek yaymanın başarısı olurdu. Puan yalnızca grup yapısına bakar, hiçbir model sonucuna
    bakmaz; bu yüzden seçim test başarısını iyimserleştirmez. Eşitlikte ilk kat (max ilk en büyüğü döndürür).

    Parametreler:
        test: Aday test katının görüntüleri (kaynak_grup_id sütunuyla).
        strata: Aynı satırların katman etiketi (etiket|kaynak).

    Döndürür:
        Karşılaştırılabilir (min, toplam) ikilisi; büyük olan daha iyi.
    """
    per_stratum = test.groupby(strata.to_numpy())["kaynak_grup_id"].nunique()
    return int(per_stratum.min()), int(test["kaynak_grup_id"].nunique())


def make_split_v2(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Görüntüleri kaynak_grup_id düzeyinde geliştirme / test olarak ayırır ve geliştirmeyi CV katlarına böler.

    - StratifiedGroupKFold (grup = kaynak_grup_id). Katman = etiket × kaynak: ALL ve Healthy'nin yaklaşık yarısı
      ALL-IDB kaynaklı; katman olmadan büyük ALL-IDB bileşenleri bir tarafta toplanabilir ve kaynak kısayolu testi
      (ALL-IDB kaynaklı / kaynağı bilinmeyen başarı) test setinde bir alt kümeyi boş bırakabilirdi.
    - Test oranı ve kat sayısı eski ayrımla aynı (1/TEST_FRACTION_FOLDS, CV_FOLDS): sonuçlar karşılaştırılabilir.
    - Test, TEST_FRACTION_FOLDS dış kattan yapısal puanı (test_fold_score) en yüksek olandır.

    Parametreler:
        groups: build_source_groups çıktısı.

    Döndürür:
        dosya_yolu, etiket, grup_id, kaynak_grup_id, kaynak, tip, bolum, kat (test için -1) tablosu.
    """
    split = groups[["dosya_yolu", "etiket", "grup_id", "kaynak_grup_id", "kaynak", "tip"]].copy().reset_index(drop=True)
    strata = split["etiket"] + "|" + split["kaynak"]
    outer = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = [test for _, test in outer.split(split, strata, split["kaynak_grup_id"])]
    test_idx = max(folds, key=lambda idx: test_fold_score(split.iloc[idx], strata.iloc[idx]))
    split["bolum"] = "gelistirme"
    split.loc[test_idx, "bolum"] = "test"
    split["kat"] = -1
    dev = split[split["bolum"] == "gelistirme"]
    inner = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, val_idx) in enumerate(inner.split(dev, strata[dev.index], dev["kaynak_grup_id"])):
        split.loc[dev.index[val_idx], "kat"] = fold
    return split


def verify_split_v2(split: pd.DataFrame) -> None:
    """
    Yeni ayrımda hiçbir kaynak grubunun (ve dolayısıyla hiçbir Adım 1 grubunun) iki tarafa düşmediğini doğrular.

    Kontroller: her kaynak_grup_id ve her grup_id tek bölümde ve tek katta; her ALL-IDB1 düğümü tek bölümde.

    Parametreler:
        split: make_split_v2 çıktısı (idb1_kaynaklari sütunu varsa ALL-IDB1 düğüm kontrolü de yapılır).
    """
    for col in ("kaynak_grup_id", "grup_id"):
        per_group = split.groupby(col)[["bolum", "kat"]].nunique()
        if (per_group > 1).any().any():
            bad = per_group[(per_group > 1).any(axis=1)].index.tolist()[:5]
            raise ValueError(f"Veri ayrımında sızıntı: {col} birden fazla bölüme/kata düşmüş (ör. {bad}).")
    if "idb1_kaynaklari" in split:
        nodes = split.assign(dugum=split["idb1_kaynaklari"].str.split(";")).explode("dugum")
        nodes = nodes[nodes["dugum"].fillna("") != ""]
        if (nodes.groupby("dugum")["bolum"].nunique() > 1).any():
            raise ValueError("Veri ayrımında sızıntı: bir ALL-IDB1 yaymasının türevleri iki bölüme düşmüş.")


def update_dataset_index(groups: pd.DataFrame) -> None:
    """
    dataset_index.csv'ye kaynak_grup_id ve kaynak sütunlarını ekler (diğer sütunlara, grup_id dahil, dokunmaz).

    Parametreler:
        groups: build_source_groups çıktısı.
    """
    index = pd.read_csv(DATASET_INDEX_CSV)
    index = index.drop(columns=[c for c in ("kaynak_grup_id", "kaynak") if c in index])
    index = index.merge(groups[["dosya_yolu", "kaynak_grup_id", "kaynak"]], on="dosya_yolu", how="left")
    index.to_csv(DATASET_INDEX_CSV, index=False, encoding="utf-8")


def summary_table(split: pd.DataFrame) -> pd.DataFrame:
    """
    Bölüm × sınıf × kaynak başına görüntü, Adım 1 grubu ve kaynak grubu sayılarını özetler.

    Parametreler:
        split: make_split_v2 çıktısı.

    Döndürür:
        Özet tablo.
    """
    return (split.groupby(["bolum", "etiket", "kaynak"])
            .agg(goruntu=("dosya_yolu", "size"), grup=("grup_id", "nunique"),
                 kaynak_grup=("kaynak_grup_id", "nunique"))
            .reset_index())


def main() -> None:
    """Kaynak gruplarını hesaplar, veri seti indeksine ekler, yeni kilitli ayrımı (bir kez) üretip doğrular."""
    pd.set_option("display.width", 200)
    index = pd.read_csv(DATASET_INDEX_CSV)
    matches = load_saved_matches(index)
    status = pd.read_csv(STATUS_CSV)
    print(f"Eşleşme: {MATCH_CSV.name} ({matches['kabul'].sum()} kabul), durum: {STATUS_CSV.name}")
    groups = build_source_groups(index, matches, status)
    update_dataset_index(groups)

    sizes = groups.groupby("kaynak_grup_id").size()
    print(f"Adım 1 grubu: {groups['grup_id'].nunique()} → kaynak grubu: {groups['kaynak_grup_id'].nunique()} "
          f"(en büyük kaynak grubu {sizes.max()} görüntü)")
    merged = groups.groupby("kaynak_grup_id")["grup_id"].nunique()
    print(f"Birden fazla Adım 1 grubunu birleştiren kaynak grubu: {(merged > 1).sum()} "
          f"({merged[merged > 1].sum()} Adım 1 grubu)")
    print("Kaynak dağılımı (görüntü):\n" + pd.crosstab(groups["etiket"], groups["kaynak"]).to_string())

    old = pd.read_csv(SPLIT_CSV)
    leak = leaky_images(groups, old)
    old_test = groups.assign(bolum=old.set_index("dosya_yolu").loc[groups["dosya_yolu"], "bolum"].to_numpy(),
                             sizinti=leak.to_numpy())
    old_test = old_test[old_test["bolum"] == "test"]
    print("Eski test setinde sızıntılı görüntü (kaynağı geliştirmede görülen):\n"
          + old_test.groupby("etiket")["sizinti"].agg(["sum", "size"]).to_string())

    if SPLIT_CSV_V2.exists():
        split = pd.read_csv(SPLIT_CSV_V2)
        print(f"\n{SPLIT_CSV_V2.name} zaten var; yeniden üretilmedi (kilitli).")
        same = split.set_index("dosya_yolu")["kaynak_grup_id"].eq(
            groups.set_index("dosya_yolu").loc[split["dosya_yolu"], "kaynak_grup_id"])
        if not same.all():
            raise ValueError("Kayıtlı v2 ayrımının kaynak grupları güncel eşleşmelerle uyuşmuyor.")
    else:
        split = make_split_v2(groups)
        split.to_csv(SPLIT_CSV_V2, index=False, encoding="utf-8")
        print(f"\nYeni kilitli ayrım yazıldı: {SPLIT_CSV_V2}")
    verify_split_v2(split.merge(groups[["dosya_yolu", "idb1_kaynaklari"]], on="dosya_yolu"))
    print("Doğrulama: hiçbir kaynak grubu / Adım 1 grubu / ALL-IDB1 yayması iki bölüme düşmüyor.")
    print(summary_table(split).to_string(index=False))
    print("Kat başına görüntü (geliştirme):\n"
          + pd.crosstab(split.loc[split["kat"] >= 0, "kat"], split.loc[split["kat"] >= 0, "etiket"]).to_string())


if __name__ == "__main__":
    main()
