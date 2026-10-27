# Görüntü İşleme Tabanlı Akıllı Lösemi Tarama ve Tanı Destek Sistemi

Mikroskobik kan yayması görüntülerinden lösemi taraması yapan, çok aşamalı ve karşılaştırmalı
bir tanı destek sistemi (Sayısal Görüntü İşleme dersi projesi).

> ⚠️ **Bu sistem araştırma ve eğitim amaçlı bir prototiptir; klinik teşhis aracı değildir.**
> Ürettiği sınıflandırma, risk skoru ve raporlar tıbbi karar için kullanılamaz.

## Ne yapar?

- Görüntüyü **3 sınıftan** birine atar:
  - **ALL** — akut lenfoblastik lösemi (çoğunlukla çocuklarda görülür)
  - **AML** — akut miyeloid lösemi (çoğunlukla yetişkinlerde görülür)
  - **Healthy** — normal kan hücresi
- Hastanın yaşını **tahmin etmez**; yalnızca lösemi tipini sınıflandırır.
- **Risk skoru:** görüntüdeki anormal hücre (ALL veya AML) oranı. PDF raporda baskın lösemi tipi de yer alır.
- Yalnızca klasik görüntü işleme + klasik makine öğrenmesi (scikit-learn) kullanılır; derin öğrenme yoktur.

## Veri Seti

- **Kaynak:** Kaggle — "leukemia types" (`sheikhlubna/leukemia-types`).
- Veri setindeki klasörlerden yalnızca üçü kullanılır:

  | Kaggle klasörü        | Projedeki klasör     | Sınıf   |
  |-----------------------|----------------------|---------|
  | `acute_lymphocytic`   | `data/raw/ALL`       | ALL     |
  | `acute_myeloid`       | `data/raw/AML`       | AML     |
  | `healthy_blood_cell`  | `data/raw/Healthy`   | Healthy |

- Veri dosyaları repoya **yüklenmez** (`.gitignore`). Kurulumda Kaggle'dan indirip yukarıdaki
  klasörlere yerleştirin.
- **Kısıt:** Veri setinin klinik kaynağı (hastane, mikroskop, boyama protokolü, hasta bilgisi)
  belirtilmemiştir. Görüntülerin bağımsız hastalardan gelip gelmediği doğrulanamaz.
- **Kısıt:** Veri setinde aynı kaynak görüntünün augmentasyonları (döndürme, renk kaydırma,
  kırpma) ve birebir kopyalar bulunur. Bu nedenle train/test ayrımı `src/dataset.py` ile atanan
  `grup_id`'ye göre yapılır. Her sınıfta 600 görüntü olmasına rağmen benzersiz kaynak sayısı
  ALL 421, AML 163, Healthy 248'dir. Ayrıntılar: [docs/sunum_notlari.md](docs/sunum_notlari.md) — Adım 1.

## Kurulum

```bash
pip install -r requirements.txt
python -m src.dataset              # veri seti indeksi ve analizi (outputs/dataset/)
python -m src.preprocessing_analysis  # ön işleme karşılaştırmaları (outputs/preprocessing/)
python -m src.segmentation_analysis   # segmentasyon karşılaştırması (outputs/segmentation/)
python -m src.segmentation            # hücre çıkarma (data/processed/cells/, cells_index.csv)
python -m src.segmentation --hepsi    # üç yöntemle hücre çıkarma (Adım 5 karşılaştırması)
python -m src.features               # öznitelik tabloları (data/processed/features_<yöntem>.csv)
python -m src.features_analysis      # kısayol testi, öznitelik önemi, figürler (outputs/features/)
streamlit run app/streamlit_app.py
```
