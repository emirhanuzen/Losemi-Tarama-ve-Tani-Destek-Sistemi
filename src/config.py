"""Proje genelinde kullanılan dizin yolları ve ortak sabitler."""

from pathlib import Path

# Proje kök dizini (src/ klasörünün bir üstü)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

# Her pipeline aşamasının ara çıktılarının kaydedileceği alt klasörler
STAGE_OUTPUT_DIRS = {
    "dataset": OUTPUTS_DIR / "dataset",
    "preprocessing": OUTPUTS_DIR / "preprocessing",
    "segmentation": OUTPUTS_DIR / "segmentation",
    "features": OUTPUTS_DIR / "features",
    "classification": OUTPUTS_DIR / "classification",
    "evaluation": OUTPUTS_DIR / "evaluation",
}

# Sınıflandırma hedefi: 3 sınıf. Klasör adları data/raw/<sınıf>/ ile birebir aynıdır.
# ALL = akut lenfoblastik lösemi, AML = akut miyeloid lösemi, Healthy = normal hücre.
CLASS_NAMES = ["ALL", "AML", "Healthy"]
ABNORMAL_CLASSES = ["ALL", "AML"]  # Risk skorunda "anormal" sayılan sınıflar

# Veri seti indeksi (dosya yolu, etiket, grup_id, görüntü tipi ...) bu CSV'ye yazılır
DATASET_INDEX_CSV = PROCESSED_DATA_DIR / "dataset_index.csv"
DUPLICATE_PAIRS_CSV = PROCESSED_DATA_DIR / "kopya_ciftleri.csv"
CELLS_DIR = PROCESSED_DATA_DIR / "cells"                 # Hücre kırpımları (Adım 3, seçilen yöntem: K-means)
CELLS_INDEX_CSV = PROCESSED_DATA_DIR / "cells_index.csv"
# Üç segmentasyon yönteminin hücre kırpımları, hücre indeksleri ve öznitelik tabloları (Adım 4-5)
SEGMENTATION_METHODS = ["kmeans", "otsu", "watershed"]
CELLS_DIRS = {"kmeans": CELLS_DIR, "otsu": PROCESSED_DATA_DIR / "cells_otsu",
              "watershed": PROCESSED_DATA_DIR / "cells_watershed"}
CELLS_INDEX_CSVS = {"kmeans": CELLS_INDEX_CSV, "otsu": PROCESSED_DATA_DIR / "cells_index_otsu.csv",
                    "watershed": PROCESSED_DATA_DIR / "cells_index_watershed.csv"}
FEATURES_CSVS = {m: PROCESSED_DATA_DIR / f"features_{m}.csv" for m in SEGMENTATION_METHODS}

RANDOM_STATE = 42  # Tekrarlanabilir sonuçlar için sabit rastgelelik tohumu

# ---------------------------------------------------------------------------
# Ön işleme parametreleri (seçim gerekçeleri: docs/sunum_notlari.md — Adım 2,
# sayısal karşılaştırmalar: python -m src.preprocessing_analysis)
# ---------------------------------------------------------------------------

# Siyah dolgu eşiği: döndürme augmentasyonunun köşeleri ve kenar doldurma saf siyahtır;
# JPEG bunları 0-10 aralığına dağıtır. Boyanmış yaymada her kanalı < 15 olan doku yoktur.
# Bazı augmentasyonlar köşelere gürültü de eklemiş; maske 5x5 açma ile temizlenir
# (preprocessing.valid_pixel_mask; bir örnekte 5437 px benek silindi).
PADDING_THRESHOLD = 15
# Koyu renkli dolgu: görüntülerin ~%3'ünde (ALL 25, AML 13, Healthy 22) döndürme köşeleri siyah
# yerine koyu gri/mor sabit renkle (ör. RGB ≈ 70, 60, 80) doldurulmuş ve 15 eşiğini geçiyor.
# - Her kanal < 100: bu dolgular 20-80 aralığında; plazma/eritrosit > 130.
# - 5x5 yerel std < 3: dolgu sabit renk; çekirdek koyu olsa da kromatin dokusu std'yi yükseltir.
# - Kenara bağlı ve alanı ≥ %1 (≈ 655 px): döndürme köşeleri büyük üçgenlerdir; kenardaki
#   küçük koyu nesneler (çekirdek parçası) dolgu sayılmaz.
DARK_PADDING_MAX = 100
DARK_PADDING_MAX_STD = 3.0
# - Renk doygunluğu (maks − min kanal) < 30: dolgu nötr gridir (ör. 70,60,80 → 20); görüntü
#   kenarındaki koyu, pürüzsüz bir çekirdek ise doygun mordur (ör. 60,30,110 → 80). Bu koşul
#   olmadan kenara değen 2 çekirdek siyah dolguyla birleşip dolgu sayılıyordu.
DARK_PADDING_MAX_CHROMA = 30
DARK_PADDING_MIN_AREA = 0.01

# Hedef boyut 256x256: görüntülerin %91'i zaten ~256 px (ALL 551/600, AML 554/600,
# Healthy 535/600). Büyük görüntülerin yarısının 256 px'lik kopyaları aynı grupta bulundu
# (yani 256'lık görüntüler aynı görüş alanının küçültülmüş hâli); Healthy'de çekirdeğin alan
# oranı iki boyutta da aynı (0.0018). Büyük boyuta (ör. 512) çıkmak çoğunluğu büyütmek
# olurdu: yeni bilgi eklemez, yalnızca enterpolasyon ve 4 kat hesaplama getirir.
IMAGE_SIZE = 256
# Kenar doldurma değeri 0 (siyah): veri setindeki döndürme dolgusu da siyah olduğu için
# tek bir "geçerli piksel" maskesi (PADDING_THRESHOLD) her iki dolguyu birden dışlar.
PADDING_VALUE = 0

# Renk normalizasyonu: arka plan tabanlı Reinhard — zemin hizalama (L* çarpımsal, a*/b* toplamsal).
# Veri setinde güçlü renk augmentasyonları (yeşil/mavi/kahverengi tonlar) ve sınıfa göre farklı
# zemin rengi/parlaklığı var (arka plan b*: Healthy 135, AML 125; L*: AML 206, Healthy 142).
# 89 görüntüde (sınıf başına 30 grup) ölçüm (ayrıntı: outputs/preprocessing/renk_normalizasyonu_skorlari.csv):
# - Klasik Reinhard (tüm pikseller, ortalama+std): std'yi içerik belirliyor (AML çekirdekle dolu,
#   Healthy'de çekirdek yok); Healthy'nin b*'si ×4.7 gerildi, Healthy çekirdeklerinin ~%100'ü
#   RGB gamutunda kırpıldı, görüntüler arası çekirdek rengi tutarlılığı kötüleşti (std 12.3 → 28).
# - Arka plan Reinhard (ortalama+std): zemin hizalandı (std 7.4 → 0.8) ama germe hâlâ sınıfa
#   bağlı (Healthy ×2.2, AML ×0.6); çekirdek−zemin farkı sınıfa göre bozuldu.
# - Arka plan kaydırma (Lab'ın üç kanalı toplamsal): zemin b* std 7.4 → 1.3, ama parlak AML
#   zeminini hedefe indirmek koyu çekirdekleri 0'a dayadı (AML çekirdek piksellerinin %31'i RGB'de kırpıldı).
# - Zemin hizalama (seçilen): aynı hizalama (zemin b* std 1.3, zemin L* std 41.9 → 5.4), AML
#   çekirdek kırpması %14. Pozlama farkı çarpımsal olduğundan L* ölçeklenir (siyah 0'da kalır),
#   renk kayması toplamsal olduğundan a*/b* kaydırılır. Fisher oranı değişmez (22.9 → 22.9).
COLOR_NORM_METHOD = "arka_plan_hizalama"
# Çekirdek maskesi 5x5 genişletilerek dışlanır (≈2 px geçiş halkası zemin istatistiğine girmesin)
REINHARD_BG_MARGIN = 5
# Hedef Lab istatistikleri (OpenCV 8 bit ölçeği): 832 farklı kopya grubunun ilk görüntüsünden,
# önce sınıf içinde sonra sınıflar arasında ortalanarak (sınıf dengeli). Yeniden hesaplamak için:
# python -m src.preprocessing_analysis (konsola yazdırır).
TARGET_LAB_BACKGROUND = {"ortalama": (168.36, 132.21, 129.43), "std": (22.78, 4.82, 3.33)}
TARGET_LAB_ALL = {"ortalama": (154.90, 137.75, 122.91), "std": (33.89, 10.80, 11.60)}

# Renk kanalı: Lab uzayının b* (sarı-mavi) kanalı, ters çevrilmiş (çekirdek parlak).
# Renk normalizasyonu sonrası 89 görüntüde (sınıf başına 30 farklı grup) 11 kanal içinde
# çekirdek/arka plan Fisher oranı en yüksek kanal (medyan 22.9; B−G 18.9, Lab-a 14.7, HSV-S 14.2,
# gri 8.8). Bağımsız S referans maskesiyle de birinci (19.1). Mor çekirdek b*'de
# negatif (mavi), pembe eritrosit ve plazma pozitif/nötrdür; eritrositi plazmadan ayırmadığı
# için yalnızca çekirdeğe duyarlıdır.
COLOR_SPACE = "LAB"
COLOR_CHANNEL = 2          # Lab içinde 0=L, 1=a, 2=b
INVERT_CHANNEL = True      # b* çekirdekte düşüktür; ters çevrilince çekirdek parlak olur

# Gürültü azaltma: 5x5 medyan. Karşılaştırmada arka plan gürültüsünü %38 azaltıp çekirdek
# sınırındaki Sobel gradyanının %94'ünü korudu. Aynı gürültü azaltımında Gaussian 5x5 kenarı
# daha çok yumuşattı (%89). Bilateral d5 σc10 çok yakın sonuç verdi (%34 / %97) ama σc'nin
# kanalın dinamik aralığına göre ayarlanması gerekir; renk augmentasyonları bu aralığı görüntüden
# görüntüye değiştirdiği için parametresiz medyan tercih edildi. Medyan 5x5, alanı < 13 px olan
# yapıları siler; 256 px'de en küçük çekirdek ~33 px (görüntü alanının %0.05'i), yani
# çekirdekler korunur. 3x3 gürültünün yalnızca %25'ini sildi, 7x7 kenar korumayı %89'a düşürdü.
DENOISE_METHOD = "median"
MEDIAN_KSIZE = 5

# CLAHE: clipLimit 1.0, karo boyu 64 px (256 px'de 4x4 ızgara).
# - 64 px karo, yaymada çekirdek (~10-25 px) ve eritrositleri (~15-20 px) birlikte içerir;
#   yakın çekimde çekirdek (~100 px) en fazla 2-3 karoya yayılır. 16-32 px karolar çekirdeğin
#   içini ayrı ayrı eşitleyip yakın çekimlerde ayrışmayı yarıdan fazla düşürdü.
# - Bu kanalda çekirdek zaten iyi ayrıştığı için tüm kontrast yöntemleri Fisher oranını biraz
#   düşürdü (arka plan gürültüsü de büyür; orijinal 22.0). clipLimit 1.0 / 64 px: Fisher 18.2
#   (16 px ile başa baş: 18.3, ama yakın çekimde 30.3'e karşı 20.4 — 16 px çekirdeğin içini
#   ayrı ayrı eşitliyor); düşük kontrastlı görüntülerin (10. persentil) çekirdek−arka plan
#   farkını 30'dan 37 gri seviyeye çıkardı. clipLimit 2 / 4 gürültüyü 3 / 5 katına çıkarıp
#   Fisher'ı 16.7 / 14.1'e düşürdü; global histogram eşitleme gürültüyü 26 katına çıkardı (Fisher 3.4).
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_PX = 64

# ---------------------------------------------------------------------------
# Segmentasyon parametreleri (seçim gerekçeleri: docs/sunum_notlari.md — Adım 3,
# karşılaştırmalar: python -m src.segmentation_analysis, hücre çıkarma: python -m src.segmentation)
# ---------------------------------------------------------------------------

# Seçilen yöntem: K-means. Etiketsiz 6 ölçütte (sınır gradyanı, Fisher, küçük parça, birleşik
# kalma, boş görüntü, sınıf tutarsızlığı) yayma ve yakın çekimde ayrı ayrı sıralama; iki tipin
# ortalama sırası K-means 1.75, Otsu 2.12, Watershed 2.12. Otsu ve Watershed kodda kalır
# (Adım 5'te sınıflandırma başarısıyla da karşılaştırılacak).
SEGMENTATION_METHOD = "kmeans"

# Eritrosit çapı tahmini (ölçek). Tüm boyut parametreleri bu çapa oranlanır, çünkü büyütme
# görüntüden görüntüye değişiyor (yayma ~10 px, yakın çekim ~62 px).
# - İç teğet çember yarıçapı < 2 px olan bileşenler sayılmaz (çap < 4 px: en küçük eritrosit
#   bile ~8 px; bunlar JPEG lekesi veya kesik parça).
# - En az 3 bileşen ve eritrosit kapsaması ≥ %5 değilse tahmin güvenilmez: çekirdekle dolu AML
#   yaymalarında eritrosit maskesi sitoplazma/plazma artıklarını yakalıyordu (kapsama %1.6).
RBC_MIN_INSCRIBED_RADIUS = 2
RBC_MIN_COMPONENTS = 3
RBC_MIN_COVERAGE = 0.05
# Güvenilmez tahminde görüntü tipinin tipik çapı: 832 farklı grubun güvenilir tahminlerinin
# medyanı (yayma %92, yakın çekim %94 güvenilir). python -m src.segmentation_analysis yeniden hesaplar.
RBC_FALLBACK_DIAMETER = {"yayma": 10.0, "yakin_cekim": 62.0}

# Ortak son işlem (üç yöntemde aynı):
# - Açma çekirdeği yarıçapı = 0.1 × eritrosit çapı (yayma 1 px, yakın çekim 6 px): çekirdekler
#   arasındaki ince köprüleri ve JPEG beneklerini siler, çekirdek şeklini bozmaz.
# - Minimum çekirdek alanı = 0.2 × eritrosit alanı (çap oranı ≈ 0.45; eritrosit ~7.5 µm ise
#   ~3.4 µm). Trombositler (2-3 µm) elenir; lenfosit çekirdeği (~eritrosit boyu) ve blastlar korunur.
OPENING_RADIUS_RATIO = 0.1
MIN_NUCLEUS_AREA_RATIO = 0.2
# Kontrast tabanı: normalize ters b* ≥ zemin + 15. Zemin piksellerinin %99.9'u zeminin en fazla
# ~13 gri seviye üstünde (medyan görüntü); çekirdeklerin en koyu %10'u görüntülerin %95'inde
# 18.6'nın üstünde. Çekirdeği çok az olan Healthy yaymalarında Otsu/K-means'in zemini bölüp
# sahte "çekirdek" üretmesini engeller (renk normalizasyonu zemini sabitlediği için mutlak eşik anlamlı).
NUCLEUS_MIN_CONTRAST = 15

# K-means: k = 3 (çekirdek / eritrosit + sitoplazma / plazma). k=2'de çekirdek kümesi
# sitoplazmayı da alıyor (Fisher 16.5), k=4-5'te çekirdeğin yalnızca koyu merkezi kalıyor
# (sınır gradyanı 137 → 120 → 108, sınıf tutarsızlığı 0.19-0.28). k=3 en düşük sınıf
# tutarsızlığını (0.12) ve gradyan/Fisher dengesini verdi.
# Model 4000 rastgele pikselle eğitilir (tüm piksellerle eğitimde boş görüntü sayısı aynı, ~2.3 kat yavaş);
# n_init=3: farklı başlangıçlarla yerel minimum riskini azaltır, süreyi makul tutar.
KMEANS_K = 3
KMEANS_MAX_PIXELS = 4000
KMEANS_N_INIT = 3

# Watershed markerları: mesafe dönüşümünün h-maksimumları, h = 0.15 × eritrosit çapı.
# min_distance = 0.5 × çap ile AML'de görüntü başına 2.3 bölme yapıldı ama ayırma çizgileri
# çekirdek içinden geçiyordu (çizgi kontrastı 0.41); h-maxima h=0.15 bölmeyi 0.9'a indirip
# çizgi kontrastını 0.72'ye çıkardı (h=0.05-0.3 arası denendi).
WATERSHED_H_RATIO = 0.15

# Hücre kırpımı: çekirdeğin sınır kutusu her yönde 0.5 × eritrosit çapı genişletilir; blastların
# dar sitoplazması ve lökositlerin sitoplazma halkası kırpıma girer, komşu hücreler az girer.
CELL_MARGIN_RATIO = 0.5

# ---------------------------------------------------------------------------
# Öznitelik çıkarma parametreleri (gerekçeler: docs/sunum_notlari.md — Adım 4,
# öznitelik tabloları: python -m src.features, analiz: python -m src.features_analysis)
# ---------------------------------------------------------------------------
# Yalnızca bu öneklere sahip sütunlar öznitelik sayılır; meta_* sütunları (görüntü tipi/boyutu,
# konum, yöntem, ham piksel ölçüleri, N/C) sınıflandırmada KULLANILMAZ.
FEATURE_PREFIXES = ("sekil_", "renk_", "doku_")

# Referans ölçek: her hücre eritrosit çapı 10 px olacak şekilde yeniden örneklenir, tüm
# öznitelikler bu ölçekte hesaplanır. 10 px yaymaların doğal ölçeği (hücrelerin %98'i yaymadan);
# yakın çekimler (~62 px) küçültülür, böylece piksel mesafeli doku ölçüleri (GLCM d=1-3 px) her
# hücrede aynı fiziksel uzunluğa karşılık gelir. Ölçeklemeden GLCM kontrastı görüntü tipini
# AUC 0.95 ile ele veriyordu; referans ölçekte 0.57.
FEATURE_REF_RBC_DIAMETER = 10

# GLCM: 32 gri seviye. Referans ölçekte yayma çekirdekleri ~100-250 px; 256 seviyede matris
# 65.536 hücreli olur ve neredeyse tamamen boş kalır (olasılıklar güvenilmez). 32 seviyede
# 1.024 hücre; seviye genişliği ~4 L birimi (8 bit), zemin gürültüsü (std ~3-5) düzeyinde, yani
# daha ince nicelemenin ekleyeceği ayrım çoğunlukla gürültü olurdu.
GLCM_LEVELS = 32
# Mesafeler 1-3 px: referans ölçekte çekirdek çapı ~12-20 px; 3 px kromatin topaklarının ölçeği,
# daha büyük mesafeler çekirdeğin dışına taşan çiftler yüzünden seyrekleşir.
GLCM_DISTANCES = (1, 2, 3)
# 4 açının ortalaması: veri setinde döndürülmüş kopyalar var; ortalama dönmeye duyarsızdır.
GLCM_ANGLES_DEG = (0, 45, 90, 135)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
# Niceleme aralığı (L*, OpenCV 8 bit): referans ölçekte çekirdek piksellerinin %0.5-%99.5
# yüzdelikleri 18-142. Sabit aralık: renk normalizasyonu sonrası aynı seviye her görüntüde aynı
# koyuluk demektir (hücre başına min-max germe soluk çekirdeklerin gürültüsünü doku gibi büyütürdü).
GLCM_GRAY_RANGE = (15, 145)

# Lob sayısı: derinliği ≥ 0.15 × eritrosit çapı (referans ölçekte 1.5 px) olan dışbükeylik
# kusurları. 1 px'lik girintiler piksel basamaklanmasıdır. Sınırlılık: 10 px ölçeğinde nötrofil
# lobları ~3 px; ayırt edilemiyor (Healthy çekirdeklerinin %99'u tek lob çıktı).
LOBE_DEPTH_RATIO = 0.15

# Çekirdek çevresi halkası (sitoplazma renk öznitelikleri): çekirdekten 1 px boşluk (sınır
# bulanıklığı çekirdek rengi taşır), sonra 0.2 × eritrosit çapı (2 px) genişlik. Lenfosit ve
# blastların sitoplazması dardır; daha geniş halka çoğunlukla zemini ölçerdi (yaymalarda 0.3-0.5 d
# halkasının medyan rengi zemine eşit çıktı). Halkada < 5 piksel kalırsa halka öznitelikleri NaN.
RING_GAP_PX = 1
RING_WIDTH_RATIO = 0.2
RING_MIN_PIXELS = 5

# N/C (çekirdek / hücre alanı) — META VERİ, öznitelik değil:
# - Yalnızca yerel eritrosit çapı ≥ 25 px olan hücrelerde (pratikte yakın çekimler) hesaplanır;
#   yaymalarda sitoplazma 1-2 px'lik bant, sınır bulanıklığından ayrılamaz.
# - Değerin var olup olmaması görüntü tipini (ve dolayısıyla çoğunlukla ALL'ı) ele verir; bu
#   yüzden sınıflandırmada kullanılmaz.
# - Sitoplazma: çekirdeğe ≤ 0.5 d uzaklıkta (kırpım payı), normalize ters b* değeri zeminden
#   ≥ 5 yüksek (mavimsi) ve çekirdeğe bağlı pikseller. 5: yaymalarda 0.3-0.5 d halkası (zemin)
#   medyanı 2-5, yakın çekimde sitoplazma içeren iç halka 6-21.
NC_MIN_NATIVE_RBC_DIAMETER = 25
NC_MAX_DISTANCE_RATIO = 0.5
CYTOPLASM_MIN_CONTRAST = 5

# ---------------------------------------------------------------------------
# Sınıflandırma parametreleri (gerekçeler: docs/sunum_notlari.md — Adım 5,
# deneyler: python -m src.classification_analysis, son model + test: python -m src.classifier)
# ---------------------------------------------------------------------------
# Kilitli veri ayrımı: grup düzeyinde geliştirme / test ve geliştirme içi CV katları.
# Bir kez üretilir, sonra hiç değiştirilmez (classifier.load_split).
SPLIT_CSV = PROCESSED_DATA_DIR / "veri_bolumu.csv"
# Test oranı 1/5 (%20): 832 grubun ~166'sı test. AML'de yalnızca 163 farklı grup var; %20 test
# ~33 AML grubu bırakır (yakın çekimli birkaç grup dahil). Daha küçük test sınıf başı metrikleri
# oynak yapar, daha büyüğü geliştirme setini (model seçimi) zayıflatır.
TEST_FRACTION_FOLDS = 5
# Geliştirme setinde 5 katlı grup CV (StratifiedGroupKFold): her katta ~%20 grup doğrulama.
CV_FOLDS = 5
# Paralel iş sayısı üst sınırı (bellek ve makine yükü için)
N_JOBS = 4
# Son model: gradyan artırma (HistGradientBoosting), öğrenme oranı 0.1, 300 iterasyon.
# Geliştirme CV'sinde 13 yapılandırma içinde hücre makro F1 (0.924, katlar arası std 0.041) ve dengeli
# doğrulukta (0.948) birinci; görüntü makro F1 0.954 (en iyi 0.956, HGB lr 0.05). İkinci SVM (C=30):
# 0.912 / 0.939 / 0.945. Fark katlar arası std'den küçük; ama HGB hem hücre hem görüntü düzeyinde en üst
# grupta, ölçek/kalibrasyon gerektirmez ve doğal olasılık verir. Erken durdurma kapalı: iç doğrulama
# bölmesi grup gözetmez (augmentasyon kopyaları iki tarafa düşerdi).
CLASSIFIER_MODEL = "hgb"
CLASSIFIER_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "early_stopping": False}
CLASSIFIER_FEATURE_SET = "tumu"
# Kenara değen (kesik) hücreler EĞİTİME DAHİL. Adım 4 dışlamayı önermişti (kesik şekiller sınıfa göre
# farklı bozuluyor); kenar deneyi (geliştirme CV) bunu desteklemedi: dahil edilince kenara değmeyen
# hücrelerde makro F1 değişmedi (0.920 → 0.924), kenara değen hücrelerde 0.616 → 0.719 arttı (ALL
# duyarlılığı 0.72 → 0.88). Risk skoru (Adım 6) kesik hücreleri de saydığı için bu önemli.
CLASSIFIER_TRAIN_EDGE_CELLS = True
# Değerlendirme ve görüntü düzeyi birleştirme: kenara değen hücreler, görüntüde tam hücre varsa
# olasılık ortalamasına girmez (görüntü F1 0.954'e karşı dahil edilince 0.948); birincil hücre
# metrikleri kenara değmeyen hücrelerde raporlanır (kesik şekil sınıflandırmayı zorlaştırıyor).
IMAGE_EXCLUDE_EDGE_CELLS = True
