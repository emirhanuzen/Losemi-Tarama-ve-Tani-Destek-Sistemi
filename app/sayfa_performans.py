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


def page() -> None:
    """Model performansı sayfasını çizer."""
    st.title("Model Performansı")
    disclaimer()
    tabs = st.tabs(["Model karşılaştırması", "Kilitli test", "Risk seviyesi", "Ablasyon", "Öğrenme eğrisi",
                    "Kalibrasyon", "Sızıntı"])
    for tab, section in zip(tabs, [model_comparison, locked_test, risk_levels, ablation, learning_curve,
                                   calibration, leakage]):
        with tab:
            section()
