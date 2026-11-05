"""
Yöntem sayfası: pipeline akış şeması, her aşamanın kısa açıklaması ve Adım 2-4'ün karşılaştırma figürleri.

Yalnızca outputs/ altındaki figürler kullanılır; data/ veya ALL-IDB dizininden görüntü okunmaz.
"""

import streamlit as st

from app.ortak import disclaimer, show_figure
from src.config import STAGE_OUTPUT_DIRS

PRE, SEG, FEAT = STAGE_OUTPUT_DIRS["preprocessing"], STAGE_OUTPUT_DIRS["segmentation"], STAGE_OUTPUT_DIRS["features"]

# Akış şeması (Graphviz DOT; tarayıcıda çizilir, ek paket gerektirmez)
FLOW_DOT = """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#52514e", fontname="Helvetica", fontsize=11];
  edge [color="#52514e"];
  A [label="Kan yayması\\ngörüntüsü"];
  B [label="Ön işleme\\n256x256, dolgu maskesi,\\nrenk normalizasyonu,\\nLab b*, medyan, CLAHE"];
  C [label="Segmentasyon\\nK-means (k=3)\\nhücre yoksa Otsu"];
  D [label="Hücre çıkarma\\nçekirdek + çevresi"];
  E [label="Öznitelik çıkarma\\n58: şekil, renk,\\nGLCM doku"];
  F [label="Sınıflandırma\\nHGB + temperature\\nALL / AML / Healthy"];
  G [label="Risk skoru\\nanormal hücre oranı,\\nbaskın tip, güven"];
  H [label="PDF rapor"];
  A -> B -> C -> D -> E -> F -> G -> H;
}
"""


def section(title: str, text: str, figures: list[tuple[str, str]]) -> None:
    """
    Bir yöntem aşamasını başlık, kısa açıklama ve figürlerle gösterir.

    Parametreler:
        title: Bölüm başlığı.
        text: Markdown açıklama.
        figures: (figür yolu, figür altı açıklama) listesi.
    """
    st.subheader(title)
    st.markdown(text)
    for path, caption in figures:
        show_figure(path, caption)


def page() -> None:
    """Yöntem sayfasını çizer."""
    st.title("Yöntem")
    disclaimer()
    st.graphviz_chart(FLOW_DOT, width="stretch")
    st.markdown("Sistem yalnızca klasik görüntü işleme ve klasik makine öğrenmesiyle çalışır; derin öğrenme "
                "kullanılmaz. Her aşamada alternatif yöntemler ölçülerek karşılaştırıldı ve seçimler sayısal "
                "ölçütlere dayandırıldı. Ayrıntılar `docs/sunum_notlari.md` dosyasındadır.")

    tabs = st.tabs(["1. Ön işleme", "2. Segmentasyon", "3. Öznitelikler", "4. Sınıflandırma ve risk"])
    with tabs[0]:
        section("Renk kanalı seçimi",
                "Çekirdek kanalı 11 aday arasından, çekirdek ile arka plan arasındaki **Fisher oranına** göre seçildi. "
                "**Ters Lab b\\*** en yüksek ayrışmayı verdi (22.9; B−G 18.9, gri 8.8): mor çekirdek bu kanalda "
                "belirgin, pembe eritrosit ve plazma nötr.",
                [(PRE / "renk_kanallari.png", "Aday kanallar ve çekirdek–arka plan ayrışma skorları.")])
        section("Renk normalizasyonu",
                "Normalizasyon yalnızca **zemin** (eritrosit + plazma) istatistiklerine dayanır: L\\* çarpımsal, "
                "a\\*/b\\* toplamsal hizalanır. Klasik Reinhard (tüm pikseller) bu veride sınıfa bağlı yapay bir "
                "renk ipucu üretiyordu, çünkü görüntünün renk dağılımı boyadan çok içerikten (çekirdek yoğunluğu) "
                "etkileniyor.",
                [(PRE / "renk_normalizasyonu.png", "Normalizasyon öncesi ve sonrası; zemin rengi tutarlı hâle gelir.")])
        section("Gürültü azaltma",
                "**Medyan 5x5**, Gaussian 5x5 ile aynı oranda gürültü siler (%38) ama kenarı daha iyi korur "
                "(0.94'e karşı 0.89). Filtre CLAHE'den önce uygulanır, çünkü CLAHE var olan gürültüyü de büyütür.",
                [(PRE / "gurultu_filtreleri.png", "Filtre türleri ve Canny kenar haritaları.")])
        section("Kontrast iyileştirme",
                "Global histogram eşitleme arka plan gürültüsünü ~26 kat büyüttü (Fisher 22.0 → 3.4). **CLAHE "
                "(clipLimit 1.0, 64 px karo)** gürültüyü düşük tutarken ayrışmayı korudu (Fisher 18.2).",
                [(PRE / "kontrast.png", "Orijinal, histogram eşitleme ve farklı clipLimit değerleriyle CLAHE.")])
        section("Tüm aşamalar", "Her sınıftan örneklerde ön işleme zinciri:",
                [(PRE / "pipeline_ornekleri.png", "Orijinal → 256x256 → dolgu maskesi → renk normalizasyonu → "
                                                  "ters Lab b* → medyan → CLAHE.")])
    with tabs[1]:
        section("Eritrosit çapı ile ölçek",
                "Görüntüler farklı büyütmelerde. Tüm boyut parametreleri (en küçük çekirdek, marker yüksekliği, "
                "kırpım payı) her görüntüde tahmin edilen **eritrosit çapına** oranlanır.",
                [(SEG / "eritrosit_boyutu.png", "Eritrosit çapı tahmini.")])
        section("Üç yöntemin karşılaştırması",
                "Otsu eşikleme, Lab uzayında K-means (k = 3) ve marker kontrollü Watershed, elle işaretlenmiş maske "
                "olmadığından etiket gerektirmeyen altı ölçütle karşılaştırıldı. **K-means** nihai sıra skorunda öne "
                "çıktı: parlaklığı da kullandığı için yoğun AML yaymalarında koyu çekirdeği açık sitoplazmadan ayırıyor. "
                "Zayıf yanı, soluk normal yaymalarda bazen hiç çekirdek bulamaması; bu durumda Otsu'ya dönülür.",
                [(SEG / "yontem_karsilastirma.png", "Üç yöntemin bulduğu çekirdek konturları."),
                 (SEG / "kmeans_k_secimi.png", "K-means küme sayısı seçimi."),
                 (SEG / "watershed_asamalari.png", "Watershed aşamaları: mesafe dönüşümü, markerlar, sonuç.")])
    with tabs[2]:
        section("Ölçek normalizasyonu",
                "Yakın çekimlerde aynı çekirdek ~40 kat fazla piksel kaplar ve yakın çekimlerin neredeyse tamamı "
                "ALL'dır. Ham piksel ölçüleri bu yüzden \"büyütmeyi\" öğretir. Boyutlar eritrosit çapına oranlandı "
                "ve her hücre eritrosit çapı 10 px olacak şekilde ortak ölçeğe getirildi: görüntü tipi tahmininde "
                "ham alan AUC 0.99, oranlanmış alan 0.46.",
                [(FEAT / "olcek_normalizasyonu.png", "Ham ve oranlanmış ölçülerin görüntü tipini ele verme düzeyi.")])
        section("GLCM doku öznitelikleri",
                "Gri seviye eş-oluşum matrisi, yan yana iki pikselin ton çiftlerinin sayım tablosudur. Düzgün "
                "kromatinde sayılar köşegende toplanır, kaba ve benekli kromatinde köşegenden uzağa yayılır. "
                "Contrast, dissimilarity, homogeneity, energy, correlation ve ASM; d = 1, 2, 3 mesafelerinde ve "
                "4 açının ortalamasıyla hesaplandı.",
                [(FEAT / "glcm_ornekleri.png", "Sınıf örneklerinde GLCM matrisleri ve doku değerleri.")])
        section("Öznitelik dağılımları", "En ayırt edici özniteliklerin sınıflara göre dağılımı:",
                [(FEAT / "oznitelik_dagilimlari.png", "Sınıflara göre öznitelik dağılımları.")])
    with tabs[3]:
        st.markdown(
            "- **Model:** HistGradientBoosting (58 öznitelik), sınıf ağırlıkları dengeli (hücrelerin %77'si AML).\n"
            "- **Ayrım:** aynı kaynağın türevleri (augmentasyonlar, ALL-IDB türevleri) hiçbir zaman hem eğitimde hem "
            "testte bulunmaz (`StratifiedGroupKFold`, grup = kaynak grubu). Test seti %20, bir kez kullanıldı.\n"
            "- **Kalibrasyon:** temperature scaling; olasılıklar iyileşir, hiçbir hücrenin sınıfı değişmez.\n"
            "- **Risk skoru:** anormal (ALL + AML) hücre oranı. Eşikler geliştirme setinden: lösemi görüntülerinin "
            "≥ %98'i Düşük dışında, Healthy görüntülerinin ≥ %95'i Yüksek dışında kalacak şekilde.\n"
            "- **Güven bayrağı:** 3'ten az hücre, Otsu geri dönüşü veya hücrelerin yarısından fazlasının kenara "
            "değmesi sonucu düşük güvenli yapar.\n"
            "- Sayısal sonuçlar **Model Performansı** sayfasındadır.")
        show_figure(STAGE_OUTPUT_DIRS["risk"] / "risk_skoru_dagilimi_kalibre.png",
                    "Geliştirme setinde sınıflara göre risk skoru dağılımı ve seçilen eşikler.")
