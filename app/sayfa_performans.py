"""
Model performansı sayfası: outputs/ altındaki mevcut figürler ve CSV'lerden üretilen tablolar.

Hiçbir model burada yeniden eğitilmez veya değerlendirilmez; sayılar ilgili analiz komutlarının çıktılarıdır.
"""

import json

import pandas as pd
import streamlit as st

from app.ortak import disclaimer, output_csv, show_figure
from src.config import ACTIVE_EVAL_SUFFIX, RISK_RESULT_SUFFIX, STAGE_OUTPUT_DIRS

CLS, EVAL, RISK = STAGE_OUTPUT_DIRS["classification"], STAGE_OUTPUT_DIRS["evaluation"], STAGE_OUTPUT_DIRS["risk"]
NUM3 = st.column_config.NumberColumn(format="%.3f")


def number_config(table: pd.DataFrame, skip: tuple = ()) -> dict:
    """
    Tablodaki sayısal sütunlar için 3 ondalıklı gösterim ayarı üretir.

    Parametreler:
        table: Gösterilecek tablo.
        skip: Biçimlenmeyecek sütunlar (ör. sayımlar).

    Döndürür:
        st.dataframe column_config sözlüğü.
    """
    return {c: NUM3 for c in table.select_dtypes("number").columns if c not in skip}


def show_table(table: pd.DataFrame | None, skip: tuple = ()) -> None:
    """
    Tabloyu gösterir; CSV yoksa bilgi mesajı verir.

    Parametreler:
        table: Tablo veya None.
        skip: Ondalık biçimlenmeyecek sütunlar.
    """
    if table is None:
        st.info("İlgili CSV bulunamadı; analiz komutunu çalıştırın (README → Kurulum ve kullanım).")
        return
    st.dataframe(table, hide_index=True, width="stretch", column_config=number_config(table, skip))


def model_comparison() -> None:
    """Geliştirme CV'sinde model karşılaştırmasını (her modelin en iyi yapılandırması) gösterir."""
    st.subheader("Model karşılaştırması")
    show_figure(CLS / "model_karsilastirma_v2.png",
                "Her modelin en iyi yapılandırması; hata çubukları katlar arası standart sapma.")
    table = output_csv("classification", "model_karsilastirma_v2.csv")
    if table is not None:
        table = table[table["en_iyi"]].sort_values("hucre_makro_f1", ascending=False)
        table = table[["model_adi", "parametreler", "hucre_makro_f1", "hucre_makro_f1_kat_std",
                       "hucre_dengeli_dogruluk", "hucre_duyarlilik_Healthy", "goruntu_makro_f1_tum"]].rename(columns={
            "model_adi": "Model", "parametreler": "Parametre", "hucre_makro_f1": "Hücre makro F1",
            "hucre_makro_f1_kat_std": "Kat std", "hucre_dengeli_dogruluk": "Dengeli doğruluk",
            "hucre_duyarlilik_Healthy": "Healthy duyarlılığı", "goruntu_makro_f1_tum": "Görüntü makro F1"})
    show_table(table)
    st.markdown("Geliştirme seti, 5 katlı kaynak grubu çapraz doğrulaması. Gradyan artırma (HGB) hem hücre hem "
                "görüntü düzeyinde en üst grupta; SVM ile farkı katlar arası sapmadan küçük. Random Forest ve k-NN "
                "normal hücreleri çoğunluk sınıfı AML'ye kaydırıyor.")


def locked_test() -> None:
    """Kalibre modelin kilitli test seti karışıklık matrislerini ve metriklerini gösterir."""
    st.subheader("Kilitli test seti (kalibre model)")
    c1, c2 = st.columns(2)
    with c1:
        show_figure(EVAL / f"test_karisiklik_hucre{ACTIVE_EVAL_SUFFIX}.png", "Hücre düzeyi")
    with c2:
        show_figure(EVAL / f"test_karisiklik_goruntu{ACTIVE_EVAL_SUFFIX}.png", "Görüntü düzeyi")
    table = output_csv("evaluation", f"test_metrikleri{ACTIVE_EVAL_SUFFIX}.csv")
    if table is not None:
        table = table.head(4)[["duzey", "n", "dogruluk", "makro_f1", "duyarlilik_ALL", "duyarlilik_AML",
                               "duyarlilik_Healthy", "anormal_duyarlilik"]].rename(columns={
            "duzey": "Düzey", "n": "n", "dogruluk": "Doğruluk", "makro_f1": "Makro F1", "duyarlilik_ALL": "Duyarlılık ALL",
            "duyarlilik_AML": "Duyarlılık AML", "duyarlilik_Healthy": "Duyarlılık Healthy",
            "anormal_duyarlilik": "Anormal duyarlılık"})
    show_table(table, skip=("n",))
    st.markdown("Test seti (361 görüntü) yalnızca son değerlendirmede bir kez kullanıldı; tüm seçimler geliştirme "
                "setinde yapıldı. Lösemi hücrelerinin %98'i anormal olarak işaretlendi. Kalibrasyon hiçbir hücrenin "
                "sınıfını değiştirmediği için metrikler kalibrasyonsuz modelle aynıdır.")


def risk_levels() -> None:
    """Test setinde risk seviyesi karışıklık matrisini ve eşik metriklerini gösterir."""
    st.subheader("Risk seviyesi (test seti)")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_figure(RISK / f"test_risk_seviyesi_karisiklik{RISK_RESULT_SUFFIX}.png",
                    "Satır gerçek sınıf, sütun risk seviyesi.")
    with c2:
        thr_path = RISK / f"risk_esikleri{RISK_RESULT_SUFFIX}.json"
        if thr_path.exists():
            thr = json.loads(thr_path.read_text(encoding="utf-8"))
            st.markdown(f"**Eşikler:** Düşük < {thr['dusuk_orta']:.4f} ≤ Orta < {thr['orta_yuksek']:.4f} ≤ Yüksek")
            st.markdown(f"- Düşük/Orta: lösemi görüntülerinin en az %{100 * thr['hedef_duyarlilik']:.0f}'i Düşük "
                        "dışında kalmalı.\n"
                        f"- Orta/Yüksek: Healthy görüntülerinin en az %{100 * thr['hedef_ozgulluk']:.0f}'i Yüksek "
                        "dışında kalmalı.\n- Eşikler yalnızca geliştirme setinin katlar dışı tahminlerinden seçildi.")
    table = output_csv("risk", f"risk_seviyesi_metrikleri{RISK_RESULT_SUFFIX}.csv")
    if table is not None:
        table = table[["bolum", "n_goruntu", "n_hucresiz", "duyarlilik_dusuk_orta", "ozgulluk_orta_yuksek",
                       "lösemi_dusuk_orani", "healthy_yuksek_orani", "baskin_tip_dogrulugu"]].rename(columns={
            "bolum": "Bölüm", "n_goruntu": "Görüntü", "n_hucresiz": "Hücresiz", "duyarlilik_dusuk_orta": "Duyarlılık @ t₁",
            "ozgulluk_orta_yuksek": "Özgüllük @ t₂", "lösemi_dusuk_orani": "Lösemi → Düşük oranı",
            "healthy_yuksek_orani": "Healthy → Yüksek oranı", "baskin_tip_dogrulugu": "Baskın tip doğruluğu"})
    show_table(table, skip=("Görüntü", "Hücresiz"))
    st.markdown("Testte iki hedef de sağlandı: lösemi görüntülerinin %0.8'i Düşük, Healthy görüntülerinin %1.7'si "
                "Yüksek çıktı. Kaçırılan 2 AML görüntüsünde çok sayıda hücrenin bir kısmı Healthy sınıflandı; güven "
                "bayrağının yakalayamadığı tek hata türü bu.")


def ablation() -> None:
    """Öznitelik grubu ablasyonunu gösterir."""
    st.subheader("Öznitelik ablasyonu")
    show_figure(CLS / "oznitelik_ablasyonu_v2.png", "Seçilen modelin farklı öznitelik gruplarıyla geliştirme başarısı.")
    table = output_csv("classification", "oznitelik_ablasyonu_v2.csv")
    if table is not None:
        table = table[["kume_adi", "oznitelik_sayisi", "hucre_makro_f1", "hucre_makro_f1_kat_std",
                       "goruntu_makro_f1_tum"]].rename(columns={
            "kume_adi": "Öznitelik kümesi", "oznitelik_sayisi": "Sayı", "hucre_makro_f1": "Hücre makro F1",
            "hucre_makro_f1_kat_std": "Kat std", "goruntu_makro_f1_tum": "Görüntü makro F1"})
    show_table(table, skip=("Sayı",))
    st.markdown("Bilginin çoğu renk özniteliklerinde; şekil ve doku tek başına zayıf ama renge eklendiklerinde "
                "başarıyı artırıyor. Renk baskınlığı biyolojik farkların yanında farklı boyama protokollerinden de "
                "kaynaklanabilir.")


def learning_curve() -> None:
    """Öğrenme eğrisini ve oran başına ortalama skorları gösterir."""
    st.subheader("Öğrenme eğrisi")
    show_figure(CLS / "ogrenme_egrisi.png", "Eğitim verisi oranına göre geliştirme CV başarısı (3 tekrar).")
    table = output_csv("classification", "ogrenme_egrisi_v2.csv")
    if table is not None:
        cols = ["hucre_makro_f1", "goruntu_makro_f1_tum", "hucre_f1_ALL", "hucre_f1_AML", "hucre_f1_Healthy"]
        table = table.groupby("oran", as_index=False).agg(
            hucre=("secilen_hucre", "mean"), **{c: (c, "mean") for c in cols})
        table["oran"] = (100 * table["oran"]).round().astype(int).map(lambda x: f"%{x}")
        table["hucre"] = table["hucre"].round().astype(int)
        table = table.rename(columns={"oran": "Grup oranı", "hucre": "Ort. hücre", "hucre_makro_f1": "Hücre makro F1",
                                      "goruntu_makro_f1_tum": "Görüntü makro F1", "hucre_f1_ALL": "Hücre F1 ALL",
                                      "hucre_f1_AML": "Hücre F1 AML", "hucre_f1_Healthy": "Hücre F1 Healthy"})
    show_table(table, skip=("Ort. hücre",))
    st.markdown("Görüntü düzeyinde eğri düzleşti (%80 → %100: 0.951 → 0.952). Hücre düzeyinde, özellikle Healthy "
                "hücrelerinde artış sürüyor; en büyük kazancı yeni ve bağımsız sağlıklı örnekler sağlar.")


def calibration() -> None:
    """Olasılık kalibrasyonu karşılaştırmasını ve test sonucunu gösterir."""
    st.subheader("Olasılık kalibrasyonu")
    show_figure(CLS / "kalibrasyon_guvenilirlik.png",
                "Güvenilirlik diyagramları: köşegene yakın eğri, olasılıkların gerçek doğruluğu yansıttığını gösterir.")
    table = output_csv("classification", "kalibrasyon_v2.csv")
    if table is not None:
        table = table[["yontem", "brier", "log_loss", "ece", "guven_099_ustu_orani", "sinifi_degisen_hucre",
                       "goruntu_makro_f1_tum"]].rename(columns={
            "yontem": "Yöntem", "brier": "Brier", "log_loss": "Log loss", "ece": "ECE",
            "guven_099_ustu_orani": "Güven ≥ 0.99 oranı", "sinifi_degisen_hucre": "Sınıfı değişen hücre",
            "goruntu_makro_f1_tum": "Görüntü makro F1"})
    show_table(table, skip=("Sınıfı değişen hücre",))
    test = output_csv("classification", "kalibrasyon_test_v2.csv")
    if test is not None:
        st.markdown("**Kilitli test:**")
        test = test.assign(model=test["model"].map(lambda m: "Temperature (kalibre)" if m.endswith("_kalibre")
                                                   else "Kalibrasyonsuz"))
        show_table(test.rename(columns={"model": "Model", "brier": "Brier", "log_loss": "Log loss", "ece": "ECE",
                                        "ort_guven": "Ort. güven", "guven_099": "Güven ≥ 0.99", "dogruluk": "Doğruluk"}))
    st.markdown("Sigmoid ve isotonic olasılıkları iyi kalibre ediyor ama ~550 hücrenin sınıfını AML'ye kaydırıyor. "
                "Temperature scaling sınıfları hiç değiştirmeyen tek yöntem; testte ECE 0.026 → 0.006, log loss "
                "yarıya indi. Arayüzdeki ve PDF'teki olasılıklar bu kalibre modelden gelir.")


def leakage() -> None:
    """Sızıntı tespiti ve düzeltmesinin özetini (eski ve yeni sonuçlar) gösterir."""
    st.subheader("Sızıntı tespiti ve düzeltme")
    st.markdown("İlk test seti kopya gruplarına göre ayrılmıştı. Kaggle görüntüleri ALL-IDB ile karşılaştırıldığında "
                "aynı ALL-IDB1 yaymasının bazı türevlerinin (sıkıştırılmış, kırpılmış hâller ve hücre kırpımları) "
                "farklı gruplara düştüğü görüldü: eski testteki ALL görüntülerinin 44/120'sinin, normal görüntülerin "
                "63/120'sinin kaynağı eğitimde de vardı. Ayrım kaynak gruplarıyla yeniden kuruldu, model yeniden "
                "eğitildi.")
    show_figure(EVAL / "sizinti_karsilastirma.png", "Eski ve yeni test sonuçları ile iki ayrımın geliştirme CV'si.")
    table = output_csv("evaluation", "sizinti_karsilastirma.csv")
    if table is not None:
        table = table.dropna(subset=["n"]).pivot(index="alt_kume", columns="duzey", values="makro_f1")
        table = table.reindex(["tum (sizintili)", "temiz (sizintisiz)", "yeni test v2 (sizintisiz)"])
        table.index = ["Eski model, eski test (tümü, sızıntılı)", "Eski model, eski testin sızıntısız kısmı",
                       "Yeni model, yeni sızıntısız test"]
        table = table[["hucre", "goruntu_kapsanan", "goruntu_tum"]].rename(columns={
            "hucre": "Hücre makro F1", "goruntu_kapsanan": "Görüntü makro F1 (hücre bulunan)",
            "goruntu_tum": "Görüntü makro F1 (tümü)"}).reset_index(names="Değerlendirme")
    show_table(table)
    st.markdown("Sızıntı eski test skorunu ölçülebilir biçimde şişirmemiş; seçilen model değişmedi. Yeni ayrım "
                "`data/processed/veri_bolumu_v2.csv` olarak kilitlendi.")


EXT = STAGE_OUTPUT_DIRS["external"]
EXPERIMENT_NAMES = {"mevcut": "Mevcut model", "a_olcek": "a) Ölçek uyarlaması", "b_macenko": "b) Macenko",
                    "c_artirma": "c) Renk artırması", "d_goreli": "d) Göreli renk"}


def external_validation() -> None:
    """Dış doğrulama (Adım 8): ön kontrol, ALL-IDB, segmentasyon, Münih teşhisi, düzeltme denemeleri ve sonuç."""
    st.subheader("Dış doğrulama")
    st.markdown(DIS_DOGRULAMA_OZET)

    st.markdown("**Sonuç (Münih TEST, güvenli ALL-IDB, Kaggle kilitli test)**")
    final = output_csv("external", "dis_dogrulama_sonuc.csv")
    if final is not None:
        cols = ["yapilandirma", "tur", "munih_test_dengeli_dogruluk", "munih_test_duyarlilik_MYO",
                "munih_test_duyarlilik_LYT", "munih_test_duyarlilik_NGS", "allidb_duyarlilik", "allidb_ozgulluk",
                "allidb_ALL_tahmin_ALL_orani", "test_hucre_makro_f1", "test_goruntu_makro_f1_tum"]
        final = final[[c for c in cols if c in final.columns]].rename(columns={
            "yapilandirma": "Yapılandırma", "tur": "Tür", "munih_test_dengeli_dogruluk": "Münih TEST dengeli doğ.",
            "munih_test_duyarlilik_MYO": "MYO→AML", "munih_test_duyarlilik_LYT": "LYT→Healthy",
            "munih_test_duyarlilik_NGS": "NGS→Healthy", "allidb_duyarlilik": "ALL-IDB duyarlılık",
            "allidb_ozgulluk": "ALL-IDB özgüllük", "allidb_ALL_tahmin_ALL_orani": "ALL-IDB ALL→ALL",
            "test_hucre_makro_f1": "Kaggle test hücre F1", "test_goruntu_makro_f1_tum": "Kaggle test görüntü F1"})
    show_table(final)

    st.markdown("**Düzeltme denemeleri (Münih AYAR + Kaggle geliştirme CV; seçim bunlarla yapıldı)**")
    exp = output_csv("external", "duzeltme_denemeleri.csv")
    if exp is not None:
        exp = exp.assign(deneme=exp["deneme"].map(lambda d: EXPERIMENT_NAMES.get(d, d)))[
            ["deneme", "cv_hucre_makro_f1", "cv_goruntu_makro_f1_tum", "ayar_dengeli_dogruluk", "ayar_duyarlilik_MYO",
             "ayar_duyarlilik_LYT", "ayar_duyarlilik_NGS", "test_goruntu_makro_f1_tum"]].rename(columns={
            "deneme": "Deneme", "cv_hucre_makro_f1": "Kaggle CV hücre F1", "cv_goruntu_makro_f1_tum": "Kaggle CV görüntü F1",
            "ayar_dengeli_dogruluk": "Münih AYAR dengeli doğ.", "ayar_duyarlilik_MYO": "MYO→AML",
            "ayar_duyarlilik_LYT": "LYT→Healthy", "ayar_duyarlilik_NGS": "NGS→Healthy",
            "test_goruntu_makro_f1_tum": "Kaggle test görüntü F1"})
    show_table(exp)

    c1, c2 = st.columns(2)
    with c1:
        show_figure(EXT / "munih_test_karisiklik_mevcut.png", "Münih TEST, mevcut model (merkez hücre).")
    with c2:
        show_figure(EXT / "allidb_risk_seviyeleri.png", "Güvenli ALL-IDB görüntülerinde risk seviyeleri (mevcut model).")

    st.markdown("**Ön kontrol: Münih ↔ Kaggle kaynak örtüşmesi**")
    pre = output_csv("external", "munich_kaynak_ozet.csv")
    if pre is not None:
        pre = pre[["sinif", "goruntu", "eslesen_A_goruntu", "eslesen_C_kapsamli", "hucre_B_esik_ustu_raporlanan",
                   "dislanan"]].rename(columns={
            "sinif": "Münih sınıfı", "goruntu": "Görüntü", "eslesen_A_goruntu": "A) görüntü eşleşmesi",
            "eslesen_C_kapsamli": "C) kapsamlı tarama", "hucre_B_esik_ustu_raporlanan": "B) hücre NCC ≥ 0.95 (yalnızca rapor)",
            "dislanan": "Dışlanan"})
    show_table(pre, skip=("Görüntü", "A) görüntü eşleşmesi", "C) kapsamlı tarama", "B) hücre NCC ≥ 0.95 (yalnızca rapor)",
                          "Dışlanan"))
    st.markdown("**ALL-IDB2 Healthy yakın çekimleri neden \"Yüksek\"?**")
    idb2 = output_csv("external", "allidb2_teshis_goruntu.csv")
    if idb2 is not None:
        idb2 = idb2[["alt_kume", "goruntu", "yakin_cekim_orani", "hucre_medyan", "tek_hucreli_orani", "risk_Yüksek",
                     "tek_hucre_yuksek", "merkez_ALL", "merkez_AML", "merkez_Healthy"]].rename(columns={
            "alt_kume": "Alt küme", "goruntu": "Görüntü", "yakin_cekim_orani": "Yakın çekim oranı",
            "hucre_medyan": "Hücre (medyan)", "tek_hucreli_orani": "Tek hücreli oranı", "risk_Yüksek": "Yüksek",
            "tek_hucre_yuksek": "Tek hücreli + Yüksek", "merkez_ALL": "Merkez→ALL", "merkez_AML": "Merkez→AML",
            "merkez_Healthy": "Merkez→Healthy"})
    show_table(idb2, skip=("Görüntü", "Hücre (medyan)", "Yüksek", "Tek hücreli + Yüksek", "Merkez→ALL", "Merkez→AML",
                           "Merkez→Healthy"))
    st.markdown("**ALL-IDB (mevcut model, eğitimde hiç görülmemiş görüntüler)**")
    idb = output_csv("external", "allidb_metrikleri_mevcut.csv")
    if idb is not None:
        idb = idb[idb["kural"] == "risk Orta/Yüksek"][["alt_kume", "n", "duyarlilik", "ozgulluk", "f1", "ALL_tahmin_ALL",
                                                       "ALL_tahmin_AML"]].rename(columns={
            "alt_kume": "Alt küme", "duyarlilik": "Duyarlılık", "ozgulluk": "Özgüllük", "f1": "F1",
            "ALL_tahmin_ALL": "ALL→ALL", "ALL_tahmin_AML": "ALL→AML"})
    show_table(idb, skip=("n", "ALL→ALL", "ALL→AML"))
    st.markdown("**Segmentasyonun gerçek etiketle değerlendirmesi (ALL-IDB1 .xyc blast merkezleri)**")
    seg = output_csv("external", "segmentasyon_xyc_ozet.csv")
    if seg is not None:
        seg = seg[seg["etiket"] == "ALL"][["yontem", "goruntu", "blast", "blast_duyarlilik", "birlesik_blast_orani",
                                            "goruntu_basina_cekirdek"]].rename(columns={
            "yontem": "Yöntem", "goruntu": "Görüntü", "blast": "Blast", "blast_duyarlilik": "Blast duyarlılığı",
            "birlesik_blast_orani": "Birleşik blast oranı", "goruntu_basina_cekirdek": "Görüntü başına çekirdek"})
    show_table(seg, skip=("Görüntü", "Blast"))
    st.markdown("**Münih hatasının teşhisi**")
    show_figure(EXT / "oznitelik_kaymasi.png", "En ayırt edici 15 öznitelik: Kaggle ve Münih dağılımları.")
    show_figure(EXT / "munih_olcek_segmentasyon_ornekleri.png",
                "Münih örneklerinde ölçek ve segmentasyon (Matek ve ark. 2019, TCIA, CC BY 3.0).")
    show_figure(EXT / "renk_dagilimi_kaggle_allidb_munih.png", "Normalizasyon öncesi ve sonrası zemin/çekirdek rengi.")


MC = STAGE_OUTPUT_DIRS["multicenter"]
LOSO_COLUMNS = {
    "ad": "Yapılandırma", "A_ALL_tipleme": "A: ALL-IDB ALL→ALL", "A_healthy_ozgulluk": "A: Healthy özgüllük",
    "A_duyarlilik": "A: lösemi duyarlılık", "Aidb_ALL_tipleme": "A (ALL-IDB kaynaklı Kaggle dahil): ALL→ALL",
    "B_dengeli_dogruluk": "B: Münih dengeli doğ.", "B_duyarlilik_MYO": "B: MYO→AML", "B_duyarlilik_LYT": "B: LYT→Healthy",
    "B_duyarlilik_NGS": "B: NGS→Healthy", "C_goruntu_makro_f1": "C: Kaggle test görüntü F1",
    "C_hucre_makro_f1": "C: Kaggle test hücre F1", "skor": "Ön kayıtlı skor", "kisit_saglandi": "Kısıt",
    "secilen": "Seçilen",
}


def multicenter() -> None:
    """Çok merkezli eğitim ve laboratuvar dışarıda bırakma (Adım 9): LOSO tablosu, laboratuvar kısayolu, v3."""
    st.subheader("Çok merkezli eğitim (LOSO)")
    st.markdown(COK_MERKEZLI_OZET)
    summary = output_csv("multicenter", "loso_ozet.csv")
    if summary is not None:
        summary = summary[[c for c in LOSO_COLUMNS if c in summary.columns]].rename(columns=LOSO_COLUMNS)
    show_table(summary)
    show_figure(MC / "loso_sonuclari.png", "Her katta test kaynağı eğitime hiç girmedi (kodla doğrulandı).")
    for fold, caption in [("A", "Kat A: ALL-IDB güvenli görüntüler (görüntü tahmini)."),
                          ("B", "Kat B: Münih TEST (merkez hücre)."), ("C", "Kat C: Kaggle kilitli test (hücre).")]:
        show_figure(MC / f"karisiklik_kat_{fold}.png", caption)
    st.markdown("**Laboratuvar kısayolu testi (öznitelikten kaynak tahmini; 0.5 = ayırt edilemiyor)**")
    lab = output_csv("multicenter", "laboratuvar_kisayolu.csv")
    if lab is not None:
        lab = lab.rename(columns={"kat": "Kat eğitimi", "oznitelik_kumesi": "Öznitelik kümesi", "n_oznitelik": "n",
                                  "kaynaklar": "Kaynaklar", "kaynak_makro_auc": "Kaynak AUC (tümü)",
                                  "sinif_ici_kaynak_auc_ALL": "Sınıf içi AUC: ALL",
                                  "sinif_ici_kaynak_auc_AML": "Sınıf içi AUC: AML",
                                  "sinif_ici_kaynak_auc_Healthy": "Sınıf içi AUC: Healthy"})
    show_table(lab, skip=("n",))
    show_figure(MC / "laboratuvar_kisayolu.png")
    c1, c2 = st.columns(2)
    with c1:
        show_figure(MC / "oznitelik_onemi.png", "Permütasyon önemi; turuncu: yeni biyolojik öznitelikler.")
    with c2:
        show_figure(MC / "oznitelik_kaynak_auc.png", "Öznitelik başına sınıf içi kaynak AUC'si.")
    st.markdown("**Kaynağa bağlı tipleme (iç CV, katlar dışı; aynı sınıf farklı kaynaklarda nasıl tipleniyor?)**")
    typ = output_csv("multicenter", "kaynak_bagimli_tipleme.csv")
    if typ is not None:
        typ = typ[typ["kat"].isin(["A", "B", "C"])][["kat", "yapilandirma", "sinif", "kaynak", "n", "duyarlilik",
                                                      "tip_dogrulugu_anormalde"]].rename(columns={
            "kat": "Kat", "yapilandirma": "Yapılandırma", "sinif": "Sınıf", "kaynak": "Kaynak", "duyarlilik": "Duyarlılık",
            "tip_dogrulugu_anormalde": "Anormal tahminde doğru tip"})
    show_table(typ, skip=("n",))
    st.markdown("**Son model (v3; varsayılan DEĞİL)**")
    show_table(output_csv("multicenter", "v3_model_ozeti.csv"))


LA = STAGE_OUTPUT_DIRS["leukemiaattri"]
LA_COLUMNS = {
    "ad": "Yapılandırma", "A_ALL_tipleme": "LOSO A: ALL-IDB ALL→ALL", "A_healthy_ozgulluk": "LOSO A: Healthy özgüllük",
    "B_dengeli_dogruluk": "LOSO B: Münih dengeli doğ.", "LA_blast_dengeli_dogruluk": "LA-CV: ALL/AML dengeli doğ.",
    "LA_duyarlilik_Healthy": "LA-CV: Healthy duyarlılığı", "LA_hasta_blast_cogunluk_dogru": "LA-CV: hasta tipi doğru",
    "C_goruntu_makro_f1": "Kaggle test görüntü F1", "skor": "Ön kayıtlı skor", "kisit_saglandi": "Kısıt",
    "secilen": "Seçilen",
}
LA_TEST_NAMES = {"blast_dengeli_dogruluk": "Blast ALL/AML dengeli doğruluk", "duyarlilik_ALL": "ALL→ALL",
                 "duyarlilik_AML": "AML→AML", "duyarlilik_Healthy": "Healthy→Healthy", "makro_f1": "Hücre makro F1",
                 "hasta_blast_cogunluk_dogru": "Hasta: blast çoğunluğu doğru",
                 "hasta_baskin_tip_dogru": "Hasta: baskın tip doğru"}


def leukemiaattri() -> None:
    """LeukemiaAttri ve laboratuvardan bağımsız ALL/AML ayrımı (Adım 10): başlangıç, yapılandırmalar, LA TEST, v4."""
    st.subheader("LeukemiaAttri: laboratuvardan bağımsız ALL/AML ayrımı")
    st.markdown(LEUKEMIAATTRI_OZET)
    summary = output_csv("leukemiaattri", "yapilandirma_sonuclari.csv")
    if summary is not None:
        summary = summary[[c for c in LA_COLUMNS if c in summary.columns]].rename(columns=LA_COLUMNS)
    show_table(summary)
    show_figure(LA / "yapilandirma_sonuclari.png", "İlk iki satır referans modeller; ► ön kayıtlı kurala göre seçilen.")
    st.markdown("**LeukemiaAttri TEST (4 ALL + 5 AML hastası; hasta bootstrap %95 güven aralığı)**")
    comp = output_csv("leukemiaattri", "la_test_karsilastirma.csv")
    if comp is not None:
        comp = comp[comp["olcu"].isin(LA_TEST_NAMES)].assign(olcu=lambda t: t["olcu"].map(LA_TEST_NAMES)).rename(
            columns={"model": "Model", "olcu": "Ölçü", "deger": "Değer", "alt": "GA alt", "ust": "GA üst"})
    show_table(comp)
    show_figure(LA / "la_test_karsilastirma.png", "v4'ün eğitiminde bu laboratuvar vardı (hastalar yoktu).")
    show_figure(LA / "la_test_karisiklik.png", "Hücre düzeyi karışıklık matrisleri (yalnızca sayılar).")
    st.markdown("**Laboratuvar kısayolu testi (aynı sınıf içinde kaynak AUC'si; 0.5 = ayırt edilemiyor)**")
    lab = output_csv("leukemiaattri", "laboratuvar_kisayolu.csv")
    if lab is not None:
        lab = lab.rename(columns={"yapilandirma": "Temsil", "n_oznitelik": "n", "kaynak_makro_auc": "Kaynak AUC (tümü)",
                                  "sinif_ici_kaynak_auc_ALL": "Sınıf içi: ALL", "sinif_ici_kaynak_auc_AML": "Sınıf içi: AML",
                                  "sinif_ici_kaynak_auc_Healthy": "Sınıf içi: Healthy"})
        lab = lab[[c for c in lab.columns if not c.startswith("sinif_ici_LA") and c not in ("kaynaklar", "sure_sn")]]
    show_table(lab, skip=("n",))
    show_figure(LA / "laboratuvar_kisayolu.png")
    st.markdown("**v4 (varsayılan DEĞİL)**")
    show_table(output_csv("leukemiaattri", "v4_model_ozeti.csv"))


# Adım 8 özeti (docs/sunum_notlari.md "Adım 8 — Dış Doğrulama ve Alan Kayması" ile aynı sonuçlar)
DIS_DOGRULAMA_OZET = """
Model hiç görmediği iki veri setinde sınandı. **Kaggle'daki başarı başka merkezlere genellenmiyor.**

- **ALL-IDB** (eğitimde hiç görülmemiş 121 görüntü; yalnızca ALL ve sağlıklı): lösemi duyarlılığı **0.92**, ancak
  özgüllük **0.43**. ALL hastalarının görüntüleri neredeyse hep **AML** olarak tiplendi (75 görüntüden 2'si ALL).
  ALL-IDB2 sağlıklı yakın çekimlerinin çoğu "Yüksek" çıktı: görüntülerin %89'unda tek hücre var, bu yüzden risk
  skoru yalnızca 0 ya da 1 olabiliyor. Ayrıca Kaggle'da hiç sağlıklı yakın çekim yok.
- **Münih AML veri seti** (Matek ve ark. 2019; 540 TEST görüntüsü): model hemen her hücreye AML diyor. Miyeloblastlar
  %92 AML, ama lenfositlerin %96'sı ve nötrofillerin %100'ü de AML. Dengeli doğruluk **0.32** (şans düzeyi 0.33).
- **Sebep: renk ve kaynak karışması.** Kaggle AML görüntüleri, ALL ve sağlıklı görüntülerden (çoğu ALL-IDB) farklı
  bir çekim kaynağından geliyor: parlak zemin, doygun mor çekirdek. Münih görüntüleri de böyle. Model bu kaynak
  imzasını "AML" olarak öğrenmiş. Segmentasyon ve ölçek sorunun kaynağı değil.
- **Düzeltme denemeleri** (ölçek uyarlaması, Macenko, renk artırması, göreli renk) Münih'te anlamlı iyileşme
  sağlamadı. Önceden sabitlenen kurala göre ölçek uyarlaması seçildi, ama TEST'teki fark +0.01.
- **Ek deney (dış doğrulama DEĞİL):** Münih AYAR kısmındaki 360 hücre eğitime eklenince Münih TEST dengeli doğruluğu
  **0.91**'e çıkıyor; Kaggle testi değişmiyor (görüntü F1 0.967). Sorun model kapasitesinde değil, tek kaynaklı
  eğitim verisinde.

Varsayılan model değiştirilmedi. Bu sonuçlar sistemin yalnızca Kaggle'a benzer görüntülerde sınandığını ve
kliniğe uygun olmadığını gösterir.
"""


def page() -> None:
    """Model performansı sayfasını çizer."""
    st.title("Model Performansı")
    disclaimer()
    tabs = st.tabs(["Model karşılaştırması", "Kilitli test", "Risk seviyesi", "Ablasyon", "Öğrenme eğrisi",
                    "Kalibrasyon", "Sızıntı", "Dış doğrulama", "Çok merkezli (LOSO)", "LeukemiaAttri"])
    for tab, section in zip(tabs, [model_comparison, locked_test, risk_levels, ablation, learning_curve,
                                   calibration, leakage, external_validation, multicenter, leukemiaattri]):
        with tab:
            section()


# Adım 10 özeti (outputs/leukemiaattri/ sonuçları)
LEUKEMIAATTRI_OZET = """
LeukemiaAttri'de (MICCAI 2024) lenfoblast, miyeloblast ve normal lökositler aynı laboratuvardan geliyor. Hücreler
uzman kutularından kırpıldı; seçim kuralı sonuçlardan önce sabitlendi.

- **Başlangıç (LA TEST, hiç görülmemiş laboratuvar):** mevcut model ve v3 neredeyse her hücreye AML diyor; blast
  ALL/AML dengeli doğruluk 0.51 (şans).
- **Seçilen (a) → v4, LA TEST:** blast ALL/AML dengeli doğruluk 0.61 (%95 GA 0.51-0.70), Healthy duyarlılığı 0.89.
  Laboratuvar eğitimde görüldü, hastalar görülmedi.
- **Görülmemiş laboratuvar:** ALL-IDB'de ALL→ALL hâlâ 0.03-0.08; Münih dengeli doğruluk 0.32 → 0.56.
- **Darboğaz ölçüm:** yalnızca LA ile eğitimde bile blast tipi 0.51-0.61; uzman öznitelik etiketleriyle 0.995.
  Nükleol ve sitoplazma bazofilisi bu çözünürlükte ölçülemiyor.
- v4 kaydedildi, **varsayılan yapılmadı**.
"""


# Adım 9 özeti (docs/sunum_notlari.md "Adım 9 — Çok Merkezli Eğitim" ile aynı sonuçlar)
COK_MERKEZLI_OZET = """
Kaggle, ALL-IDB ve Münih birlikte kullanıldı. Başarı **laboratuvar dışarıda bırakma (LOSO)** ile ölçüldü: her katta
test laboratuvarı eğitime hiç girmez (kodla doğrulandı).

- **Kat A:** Kaggle (ALL-IDB kaynaklı görüntüler hariç) + Münih ile eğitim → ALL-IDB'de test.
- **Kat B:** Kaggle + ALL-IDB ile eğitim → Münih TEST'te test.
- **Kat C:** üç kaynakla eğitim → Kaggle kilitli testte test.

Mevcut 58 özniteliğe 15 biyolojik öznitelik eklendi: nükleol, sitoplazma granülerliği, kromatin dokusu, çekirdek
kenar düzensizliği. Seçim kuralı sonuçlardan önce sabitlendi.

- **Çok merkezli eğitim, laboratuvardan bağımsız bir ALL/AML ayrımı sağlamadı.** Görülmemiş laboratuvarda ALL
  görüntülerinin %91-93'ü hâlâ AML tipleniyor (ALL→ALL 0.03-0.07). Münih'te dengeli doğruluk 0.32-0.34, yani şans
  düzeyi.
- Model **laboratuvar içinde** çalışıyor: eğitimdeki her kaynak × sınıfta duyarlılık 0.88-0.98. Ama görmediği
  laboratuvara genellenmiyor.
- **Laboratuvar kısayolu:** öznitelikler hücrenin hangi laboratuvardan geldiğini AUC 0.98-0.999 ile ele veriyor,
  aynı sınıf içinde de. Model sınıf yerine laboratuvarı öğrenebiliyor.
- Biyolojik özniteliklerin laboratuvarlar arası genellemeye ölçülebilir bir katkısı yok.
- Kurala göre seçilen (ii), mevcut modelden yalnızca gürültü düzeyinde iyi: 75 ALL görüntüsünün 5'ine karşı 2'si.
- Üç kaynağın tamamıyla eğitilen **v3 kaydedildi ama varsayılan yapılmadı**. Dış test seti kalmadı; başarısı ancak
  LOSO ile tahmin edilebilir.
"""
