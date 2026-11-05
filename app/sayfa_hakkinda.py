"""
Hakkında sayfası: proje özeti, veri kaynakları, kaynakça ve kısıtlar.
"""

import streamlit as st

from app.ortak import disclaimer

REFERENCES = [
    "\"Leukemia Types\" veri seti (sheikhlubna), Kaggle. "
    "https://www.kaggle.com/datasets/sheikhlubna/leukemia-types",
    "R. Donida Labati, V. Piuri, F. Scotti, \"ALL-IDB: The Acute Lymphoblastic Leukemia Image Database for Image "
    "Processing,\" *Proc. IEEE International Conference on Image Processing (ICIP)*, pp. 2045–2048, 2011.",
    "C. Matek, S. Schwarz, K. Spiekermann, C. Marr, \"Human-level recognition of blast cells in acute myeloid "
    "leukaemia with convolutional neural networks,\" *Nature Machine Intelligence*, vol. 1, pp. 538–544, 2019 "
    "(Münih LMU Hastanesi AML sitomorfoloji veri seti, The Cancer Imaging Archive).",
    "N. Otsu, \"A Threshold Selection Method from Gray-Level Histograms,\" *IEEE Trans. Systems, Man, and "
    "Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.",
    "R. M. Haralick, K. Shanmugam, I. Dinstein, \"Textural Features for Image Classification,\" *IEEE Trans. "
    "Systems, Man, and Cybernetics*, vol. SMC-3, no. 6, pp. 610–621, 1973.",
    "E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley, \"Color Transfer between Images,\" *IEEE Computer Graphics "
    "and Applications*, vol. 21, no. 5, pp. 34–41, 2001.",
    "K. Zuiderveld, \"Contrast Limited Adaptive Histogram Equalization,\" *Graphics Gems IV*, Academic Press, "
    "pp. 474–485, 1994.",
    "L. Vincent, P. Soille, \"Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion "
    "Simulations,\" *IEEE Trans. PAMI*, vol. 13, no. 6, pp. 583–598, 1991.",
    "J. H. Friedman, \"Greedy Function Approximation: A Gradient Boosting Machine,\" *Annals of Statistics*, "
    "vol. 29, no. 5, pp. 1189–1232, 2001.",
    "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" *Proc. ICML*, "
    "pp. 1321–1330, 2017 (temperature scaling).",
    "F. Pedregosa ve ark., \"Scikit-learn: Machine Learning in Python,\" *JMLR*, vol. 12, pp. 2825–2830, 2011.",
]


def page() -> None:
    """Hakkında sayfasını çizer."""
    st.title("Hakkında")
    disclaimer()
    st.subheader("Proje")
    st.markdown(
        "**Görüntü İşleme Tabanlı Akıllı Lösemi Tarama ve Tanı Destek Sistemi** — Sayısal Görüntü İşleme dersi "
        "projesi. Mikroskobik kan yayması görüntülerinde beyaz kan hücresi çekirdeklerini bulur, her hücreyi "
        "**ALL** (akut lenfoblastik lösemi), **AML** (akut miyeloid lösemi) veya **Healthy** (normal) olarak "
        "sınıflandırır ve görüntü için bir risk skoru ile PDF rapor üretir.\n\n"
        "- Yalnızca klasik görüntü işleme ve klasik makine öğrenmesi (scikit-learn) kullanılır; derin öğrenme yoktur.\n"
        "- Risk skoru, görüntüdeki anormal (ALL + AML) hücre oranıdır; raporda baskın lösemi tipi de yazılır.\n"
        "- Sistem hastanın yaşını tahmin etmez. ALL'ın çoğunlukla çocuklarda, AML'nin yetişkinlerde görülmesi "
        "yalnızca klinik arka plan bilgisidir.")

    st.subheader("Veri kaynakları")
    st.markdown(
        "- **Kaggle \"Leukemia Types\"** (`sheikhlubna/leukemia-types`): yalnızca `acute_lymphocytic` (ALL), "
        "`acute_myeloid` (AML) ve `healthy_blood_cell` (Healthy) klasörleri kullanıldı. Veri setinin klinik kaynağı "
        "belirtilmemiş; aynı görüntünün döndürme, renk ve kırpma augmentasyonlarını içeriyor.\n"
        "- **ALL-IDB (Milano Üniversitesi):** Kaggle'daki ALL ve Healthy görüntülerinin bir kısmının ALL-IDB1/2 "
        "türevi olduğu bu projede tespit edildi. Eğitim/test ayrımı bu kaynak eşleşmelerine göre yapıldı. ALL-IDB "
        "lisans gereği repoya girmez ve arayüzde gösterilmez; hazır örnekler yalnızca kaynağı bilinmeyen "
        "görüntülerden seçildi.\n"
        "- **Münih AML sitomorfoloji veri seti (Matek ve ark., 2019):** AML blast morfolojisi için literatürdeki "
        "başlıca açık veri setlerinden biri; arka plan ve karşılaştırma kaynağı olarak listelendi. Kaggle AML "
        "görüntülerinin bu veri setiyle eşleşmesi bu projede test edilmedi.")

    st.subheader("Kısıtlar")
    st.markdown(
        "- **Klinik doğrulama yok.** Sonuçlar aynı veri setinden ayrılmış bir test setine ait; başka laboratuvar "
        "ve mikroskop görüntülerinde başarı düşebilir.\n"
        "- **Kaynak belirsizliği.** Veri setinin klinik kaynağı ve boyama protokolü bilinmiyor; görüntülerin bağımsız "
        "hastalardan gelip gelmediği doğrulanamıyor. Kaynak tespitinin kaçırdığı türevler kalmış olabilir.\n"
        "- **Renk baskınlığı.** Ayrımın büyük kısmı renk özniteliklerinden geliyor; model morfolojinin yanında "
        "boyanma tarzını da öğreniyor olabilir.\n"
        "- **Görüntü düzeyi etiketler.** Her hücre görüntünün etiketini alır; lösemi görüntüsündeki normal lökosit "
        "de anormal sayılır. Bu yüzden risk skoru bir \"blast oranı\" değildir.\n"
        "- **Segmentasyon kapsaması.** K-means soluk normal yaymalarda çekirdek kaçırabilir (Otsu geri dönüşü "
        "kısmen telafi eder); hiç hücre bulunamazsa sonuç \"değerlendirilemedi\" olur.\n"
        "- **Az hücreli görüntüler.** 1-2 hücrede tek bir hata skoru büyük ölçüde değiştirir; bu sonuçlar düşük "
        "güvenli olarak işaretlenir.")

    st.subheader("Kaynakça")
    st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(REFERENCES, start=1)))
