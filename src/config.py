"""Proje genelinde kullanılan dizin yolları ve ortak sabitler."""

import os
import platform
from pathlib import Path

# Proje kök dizini (src/ klasörünün bir üstü). CSV'lerdeki tüm yollar bu köke göre göreli ve ileri eğik çizgilidir
# (ör. data/raw/ALL/acute_lymphocytic (1).jpg); böylece aynı tablolar Windows'ta ve WSL'de birlikte kullanılır.
# Okurken utils.resolve_path, yazarken utils.to_relative kullanılır.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_ROOT  # eski ad; mevcut modüller için


def is_wsl() -> bool:
    """
    Kodun WSL (Windows Subsystem for Linux) içinde çalışıp çalışmadığını döndürür.

    Döndürür:
        Linux çekirdek sürümünde "microsoft" geçiyorsa True.
    """
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


# ALL-IDB (ALL_IDB1 ve ALL_IDB2 klasörlerini içeren dizin) proje dışında tutulur: lisans gereği repoya girmez.
# Öncelik: LOSEMI_ALLIDB_DIR ortam değişkeni; yoksa platform varsayılanı (Windows ve WSL aynı klasörü görür).
ALLIDB_DEFAULT_DIRS = {
    "windows": Path(r"C:\Users\Emirhan\Downloads"),
    "wsl": Path("/mnt/c/Users/Emirhan/Downloads"),
    "diger": PROJECT_ROOT / "data" / "external",
}
ALLIDB_DIR = Path(os.environ.get("LOSEMI_ALLIDB_DIR") or ALLIDB_DEFAULT_DIRS[
    "windows" if os.name == "nt" else ("wsl" if is_wsl() else "diger")])

# Münih AML sitomorfoloji veri seti (Matek ve ark., 2019; TCIA "AML-Cytomorphology_LMU", CC BY 3.0): sınıf
# klasörleri (MYO, LYT, NGS, MYB) bu dizinin altında. Öncelik LOSEMI_MUNICH_DIR; yoksa ALL-IDB ile aynı varsayılan.
MUNICH_DIR = Path(os.environ.get("LOSEMI_MUNICH_DIR") or ALLIDB_DEFAULT_DIRS[
    "windows" if os.name == "nt" else ("wsl" if is_wsl() else "diger")])

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
    "risk": OUTPUTS_DIR / "risk",
    "external": OUTPUTS_DIR / "external",
    "multicenter": OUTPUTS_DIR / "multicenter",
}
# ALL-IDB görüntüsü içeren figürler ve dış veri ara dosyaları (git dışı: data/ .gitignore kapsamında)
EXTERNAL_PROCESSED_DIR = DATA_DIR / "processed" / "external"

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
# v2 (sızıntı düzeltmesi, src/source_groups.py): ALL-IDB kaynak tespiti, eski ayrımda aynı ALL-IDB1 yaymasının
# türevlerinin (tam / sıkıştırılmış / kırpım / ALL-IDB2 hücre kırpımı) iki tarafa düştüğünü gösterdi. v2 ayrımı
# kaynak_grup_id'ye (Adım 1 grupları + ALL-IDB eşleşmeleriyle birleştirilmiş gruplar) göre yapılır, o da kilitlidir.
# Eski ayrım (v1) silinmez; eski modelin değerlendirmesi ve karşılaştırma için kalır.
SPLIT_CSV_V2 = PROCESSED_DATA_DIR / "veri_bolumu_v2.csv"
SPLIT_CSVS = {"v1": SPLIT_CSV, "v2": SPLIT_CSV_V2}
SPLIT_GROUP_COLUMNS = {"v1": "grup_id", "v2": "kaynak_grup_id"}
# Etkin ayrım: tüm deneyler, son model ve risk analizi bunu kullanır.
SPLIT_VERSION = "v2"
# Sürüme göre model dosyası ve çıktı dosyası soneki (v1 çıktılarının üzerine yazılmasın)
MODEL_FILES = {"v1": "losemi_siniflandirici", "v2": "losemi_siniflandirici_v2"}
RESULT_SUFFIXES = {"v1": "", "v2": "_v2"}
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
# v2 ayrımıyla (sızıntısız kaynak grupları) karşılaştırma yeniden yapıldı, seçim değişmedi: HGB lr 0.1 hücre makro F1
# 0.903 (kat std 0.089) ile yine birinci (HGB lr 0.05: 0.901, RF yaprak 1: 0.889, SVM C=30: 0.887); görüntü makro F1
# 0.952 (HGB lr 0.05 ile eşit, en yüksek). Ablasyonda "şekil + renk" 0.905, tümü 0.903: fark kat std'nin çok altında,
# öznitelik kümesi değiştirilmedi. Model: models/losemi_siniflandirici_v2.joblib.
CLASSIFIER_MODEL = "hgb"
# Olasılık kalibrasyonu (python -m src.classification_analysis kalibrasyon): HGB'nin ham olasılıkları aşırı
# kendinden emin (çoğu hücrede 1.000 / 0.000). CalibratedClassifierCV, ensemble=False: temel model tüm eğitim
# verisiyle eğitilir, kalibratör iç katların katlar dışı tahminlerinden öğrenilir. İç katlar kaynak grubu düzeyinde
# (StratifiedGroupKFold): aynı yaymanın türevleri hem kalibratörün eğitiminde hem tahmininde olursa olasılıklar yine
# aşırı iyimser kalır. None: kalibrasyon yok. Seçim ve gerekçe: aşağıda CLASSIFIER_CALIBRATION.
CALIBRATION_FOLDS = 5
CALIBRATION_SUFFIX = "_kalibre"
# Seçim: "temperature" (geliştirme CV, katlar dışı, 14.729 hücre; outputs/classification/kalibrasyon_v2.csv):
# - Kalibrasyonsuz: Brier 0.114, log loss 0.366, ECE 0.051; hücrelerin %90'ında güven ≥ 0.99.
# - Temperature: Brier 0.098, log loss 0.185, ECE 0.013; güven ≥ 0.99 %63. Tek parametre olduğu için en olası sınıf
#   DEĞİŞMEZ (0 hücre; görüntü ortalamasında 1 görüntü), hücre/görüntü makro F1 aynı (0.903 / 0.951).
# - Sigmoid / isotonic (ağırlıksız): Brier 0.095 (biraz daha iyi) ama hücrelerin %3.6-3.8'i AML'ye kaydı
#   (≈ 300 ALL→AML, 230 Healthy→AML), görüntü F1 0.952 → 0.93: kalibratör AML'nin önselini (%77) geri getiriyor.
# - Dengeli ağırlıklı sigmoid / isotonic: ters yöne kaydı (≈ 260-300 AML→Healthy), Brier 0.127-0.128 (kötüleşti).
CLASSIFIER_CALIBRATION = "temperature"
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
# Etkin son model ve onun sonuç dosyalarının soneki (pipeline, risk analizi ve PDF bunları kullanır).
# Kalibre model ayrı dosyadır; kalibrasyonsuz model ve sonuçları silinmez.
_CAL = CALIBRATION_SUFFIX if CLASSIFIER_CALIBRATION else ""
ACTIVE_MODEL_FILE = MODEL_FILES[SPLIT_VERSION] + _CAL          # ör. losemi_siniflandirici_v2_kalibre
ACTIVE_EVAL_SUFFIX = RESULT_SUFFIXES[SPLIT_VERSION] + _CAL     # outputs/evaluation/test_metrikleri<sonek>.csv
RISK_RESULT_SUFFIX = _CAL                                      # outputs/risk/<ad><sonek>.csv

# ---------------------------------------------------------------------------
# Uçtan uca pipeline, risk skoru ve PDF rapor (gerekçeler: docs/sunum_notlari.md — Adım 6,
# analiz: python -m src.risk_analysis, tek görüntü: python -m src.pipeline --goruntu <yol> --pdf)
# ---------------------------------------------------------------------------
# Otsu geri dönüşü: K-means bir görüntüde hiç çekirdek bulamazsa aynı görüntüde Otsu denenir.
# K-means, çekirdeği çok az ve soluk Healthy yaymalarında kümeyi zemine/eritrosite harcayıp boş
# kalabiliyor (testte 11 görüntü hücresiz: 9 Healthy). Adım 5 karşılaştırmasında Otsu'nun kapsaması
# daha yüksekti; tek eşik, tek bir çekirdek bile zeminden ayrışıyorsa onu yakalar. Model yeniden
# eğitilmez (K-means hücreleriyle eğitildi); geri dönüş yalnızca çıkarımda uygulanır ve sonuçta
# hangi yöntemin kullanıldığı yazılır (düşük güven nedeni sayılır).
SEGMENTATION_FALLBACK_METHOD = "otsu"

# Risk skoru: anormal (ALL + AML) olarak sınıflanan hücrelerin oranı. Kenara değen (kesik) hücreler
# DAHİL: model onlarla da eğitildi (Adım 5.7) ve Healthy görüntülerinde 1-2 hücre olduğundan bir
# hücreyi atmak skoru belirsiz bırakır. Çoğu kenara değiyorsa güven bayrağı kalkar (aşağıda).
# Yumuşak skor: 1 − ortalama P(Healthy) = ortalama P(ALL) + P(AML); raporda ek bilgi olarak verilir.
RISK_INCLUDE_EDGE_CELLS = True

# Risk seviyesi eşikleri (Düşük < RISK_THRESHOLDS[0] ≤ Orta < RISK_THRESHOLDS[1] ≤ Yüksek).
# Geliştirme setinin katlar dışı (out-of-fold) görüntü düzeyi skorlarından, iki hedefle seçildi
# (src.risk_analysis.select_thresholds; aday eşikler komşu skor değerlerinin orta noktaları):
# - Düşük/Orta eşiği: lösemi (ALL+AML) görüntülerinin en az %RISK_TARGET_SENSITIVITY'i "Düşük"
#   DIŞINDA kalacak en büyük eşik. Tarama aracında lösemiyi "düşük risk" demek en pahalı hatadır.
# - Orta/Yüksek eşiği: Healthy görüntülerinin en az %RISK_TARGET_SPECIFICITY'i "Yüksek" DIŞINDA
#   kalacak en küçük eşik. "Yüksek" denilen görüntü gerçekten şüpheli olmalı (yanlış alarm sınırlı).
# Değerler geliştirme setinde hesaplanıp buraya sabitlendi; test seti eşik seçiminde kullanılmadı.
RISK_TARGET_SENSITIVITY = 0.98
RISK_TARGET_SPECIFICITY = 0.95
# Sonuç (v2 ayrımı, losemi_siniflandirici_v2 yapılandırması, geliştirme 1.426 hücreli görüntü):
# - Duyarlılık ≥ 0.98 olan en büyük eşik 0.8229 (duyarlılık 0.981); özgüllük ≥ 0.95 olan en küçük eşik 0.7591
#   (özgüllük 0.952). İkincisi birincisinden küçük → yer değiştirildi (select_thresholds): iki hedef de sağlanır.
# - Geliştirmede: lösemi görüntülerinin %1.0'ı "Düşük", Healthy görüntülerinin %4.8'i "Yüksek".
# - "Orta" bandı dar (0.76-0.82): skor dağılımları iyi ayrışıyor (sert skor AUC 0.974); 0.55-0.99 aralığında
#   özgüllük 0.947-0.952'de sabit, yani bu aralıktaki Healthy görüntüleri tek hücreli yanlış alarmlar.
# Kalibre modelle (temperature) yeniden seçildi: aynı eşikler. Risk skoru en olası sınıfa dayanan hücre oranıdır ve
# temperature en olası sınıfı değiştirmez; yalnızca yumuşak skor iyileşti (geliştirme AUC 0.995 → 0.997).
RISK_THRESHOLDS = (0.7591, 0.8229)  # python -m src.risk_analysis analiz ile seçildi
RISK_LEVELS = ["Düşük", "Orta", "Yüksek"]
RISK_NO_CELL_LEVEL = "Değerlendirilemedi"  # Geri dönüşle de hiç hücre bulunamazsa

# Güven bayrağı ("düşük güven") koşulları:
# - Hücre sayısı < 3: Adım 5'te yanlış sınıflanan 9 Healthy test görüntüsünün hepsinde tek hücre
#   vardı; 1-2 hücrede tek bir hücre hatası skoru 0'dan 0.5-1'e taşır. 3 hücrede tek hata skoru
#   en fazla 0.33 değiştirir.
# - Geri dönüş kullanıldı: model Otsu kırpımlarıyla eğitilmedi (sınır ve boyut dağılımı farklı).
# - Hücrelerin yarısından fazlası kenara değiyor: kesik hücrelerde makro F1 0.72'ye karşı
#   tam hücrelerde 0.92 (Adım 5.7); skorun çoğu daha güvenilmez tahminlerden gelir.
LOW_CONFIDENCE_MIN_CELLS = 3
LOW_CONFIDENCE_EDGE_FRACTION = 0.5

# PDF rapor
EXAMPLE_REPORTS_DIR = REPORTS_DIR / "ornekler"
REPORT_MAX_CELL_ROWS = 30      # Hücre tablosu bir sayfaya sığsın; kalanlar sınıf sayılarıyla özetlenir
# Kontur görselinin uzun kenarı (px): A4'te ~8 cm genişliğe 250 dpi civarı, dosya boyutu küçük kalır
REPORT_IMAGE_MAX_SIDE = 800
REPORT_DISCLAIMER = ("Bu rapor araştırma ve eğitim amaçlı bir prototip tarafından üretilmiştir. "
                     "Klinik teşhis için kullanılamaz; kesin tanı uzman hekim tarafından konmalıdır.")


# ---------------------------------------------------------------------------
# Dış doğrulama (Adım 8): ALL-IDB ve Münih
# ---------------------------------------------------------------------------
# Münih sınıf klasörü → beklenen proje sınıfı. Kısaltmalar veri setinin abbreviations.txt dosyasından:
# MYO = myeloblast (AML blastı), LYT = tipik lenfosit, NGS = segmentli nötrofil (ikisi de normal hücre).
# MYB = myelocyte (olgunlaşan granülosit öncülü; blast DEĞİL, veri setinde yalnızca 42 görüntü): beklenen sınıf
# tanımsız, yalnızca ayrıca raporlanır.
MUNICH_CLASSES = {"MYO": "AML", "LYT": "Healthy", "NGS": "Healthy"}
MUNICH_EXTRA_CLASSES = ["MYB"]
# Sınıf başına örnek: 300 (900 görüntü); uçtan uca pipeline görüntü başına ~1 sn, tüm denemeler makul sürede biter.
MUNICH_SAMPLE_PER_CLASS = 300
# AYAR (düzeltme seçimi) / TEST (yalnızca en sonda, seçilen yapılandırmalar için bir kez) oranı: %40 / %60.
# AYAR kısmı 120 görüntü/sınıf: sınıf duyarlılığında ±0.09'luk (%95 GA) belirsizlik; seçim için yeterli,
# TEST'e daha büyük pay bırakılır.
MUNICH_TUNE_FRACTION = 0.4
# ALL-IDB1 .xyc blast merkezlerinin çekirdekle eşleşme yarıçapı = 0.5 × eritrosit çapı (256 px uzayında).
# Blast çekirdeği ~1-1.5 eritrosit çapında; merkez işaretleri elle konduğu için çekirdeğin içinde ama tam ortada
# olmayabilir. Yarım çap, komşu çekirdeğe yanlış atamayı önleyecek kadar dar.
XYC_MATCH_RADIUS_RATIO = 0.5


# ---------------------------------------------------------------------------
# Çok merkezli eğitim ve laboratuvar dışarıda bırakma (Adım 9; python -m src.multicenter)
# ---------------------------------------------------------------------------
# Çok merkezli öznitelik tabloları (Kaggle + ALL-IDB + Münih, 58 + 15 biyolojik öznitelik; git dışı)
MULTICENTER_DIR = PROCESSED_DATA_DIR / "multicenter"
# Münih eğitim örneklemi (görüntü; yalnızca merkez hücre etiketli): kilitli AYAR/TEST ayrımının 300 görüntüsü + aynı
# sınıftan sabit tohumla ek görüntüler. MYO 1000: AML'nin ikinci kaynağı (Kaggle AML ile aynı ağırlık tavanı, aşağıda).
# LYT + NGS = 1000: ikisi birlikte Münih Healthy'si; tavan proje sınıfı düzeyinde olduğu için yarı yarıya.
# Daha fazlası tavan yüzünden ağırlığı düşürülürdü; yalnızca çeşitlilik ekler, çıkarım süresini uzatır.
MUNICH_TRAIN_IMAGES = {"MYO": 1000, "LYT": 500, "NGS": 500}
# Kaynak dengesi: kaynak × sınıf başına toplam örnek ağırlığı tavanı (hücre). n > tavan ise her hücrenin ağırlığı
# tavan / n (ör. Kaggle AML 11.441 hücre → 0.087), n ≤ tavan ise 1. 1000 ≈ Kaggle Healthy hücre sayısı (964), yani
# Kaggle'ın üç sınıfı eşit ağırlık alır ve Münih bir sınıfta Kaggle'dan ağır basamaz. Tavanın altındaki küçük gruplar
# (ALL-IDB: ~600 blast, ~300 sağlıklı) YUKARI ağırlıklandırılmaz: birkaç yüz hücreye 3-10 kat ağırlık vermek modeli o
# hücreleri ezberlemeye iterdi. Ardından sınıf toplamları eşitlenir (her sınıf toplam ağırlığın 1/3'ü).
MULTICENTER_SOURCE_CAP = 1000
# İç CV (hiperparametre seçimi, her katın kendi eğitim verisinde): 3 kat StratifiedGroupKFold (grup = birleşik kaynak
# grubu, katman = kaynak × sınıf). 3 kat: küçük kaynak × sınıf hücreleri (ALL-IDB Healthy ~300) her katta temsil edilir.
MULTICENTER_INNER_FOLDS = 3
# Küçük ızgara (HGB): öğrenme oranı 0.05 / 0.1 (Adım 5'te ikisi başa baş) × yaprak sayısı 15 / 31 (31 = sklearn
# varsayılanı; 15 daha düzenli, kaynak imzasını ezberlemeye daha az yatkın). 300 iterasyon, erken durdurma kapalı
# (iç doğrulama bölmesi grup gözetmez).
MULTICENTER_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 300, "early_stopping": False}
                    for lr in (0.05, 0.1) for leaves in (15, 31)]
# Kaynak kısayolu eşiği: bir özniteliğin AYNI SINIF içinde kaynağı ayırma AUC'si (|AUC − 0.5| + 0.5, sınıf ve kaynak
# çiftleri üzerinden en büyük) ≥ 0.90 ise "kaynak AUC'si çok yüksek" sayılır ve (iv) yapılandırmasında çıkarılır.
# Sınıf içi ölçülür: kaynaklar farklı sınıf karışımına sahip (Münih'te ALL yok), sınıflar arası AUC biyolojiyi de
# ölçerdi. 0.90: Adım 4'te görüntü tipi kısayolu için "belirgin" sayılan düzey (GLCM 0.95), sonuçlardan önce sabitlendi.
MULTICENTER_SOURCE_AUC_MAX = 0.90
# Ön kayıtlı seçim kuralı (outputs/multicenter/secim_kurali.csv; LOSO sonuçlarından ÖNCE yazılır):
# skor = (Kat A ALL tiplemesi oranı + Kat B dengeli doğruluk) / 2 en yüksek; Kat C görüntü makro F1 mevcut modele
# göre en fazla 0.02 düşebilir.
MULTICENTER_MAX_KAGGLE_DROP = 0.02
# Çok merkezli son model (üç kaynağın tamamı, temperature kalibrasyonlu). Varsayılan model (ACTIVE_MODEL_FILE) DEĞİŞMEZ.
MULTICENTER_MODEL_FILE = "losemi_siniflandirici_v3_kalibre"
