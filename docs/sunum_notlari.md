# Sunum Notları — Görüntü İşleme Tabanlı Akıllı Lösemi Tarama ve Tanı Destek Sistemi

Bu dosya her geliştirme adımı tamamlandığında güncellenir. Her adım için
**Ne yapıldı**, **Yöntemin teorisi** ve **Elde edilen sonuçlar** başlıkları yer alır.

---

## Adım 0 — Proje İskeleti

**Proje kapsamı:**
- Sistem 3 sınıflı sınıflandırma yapar: **ALL** (akut lenfoblastik lösemi, çoğunlukla çocuklarda),
  **AML** (akut miyeloid lösemi, çoğunlukla yetişkinlerde), **Healthy** (normal hücre).
- Sistem hastanın **yaşını tahmin ETMEZ**; yalnızca lösemi tipini sınıflandırır. ALL/AML'nin yaş
  gruplarıyla ilişkisi yalnızca klinik arka plan bilgisidir.
- **Risk skoru:** görüntüdeki anormal hücre (ALL veya AML) oranı. PDF raporda risk skoruna ek
  olarak baskın lösemi tipi (ALL veya AML) de yazılır.
- Sistem araştırma ve eğitim amaçlı bir prototiptir; **klinik teşhis aracı değildir**.

**Ne yapıldı:**
- Klasör yapısı oluşturuldu: `data/`, `src/`, `app/`, `notebooks/`, `outputs/`, `reports/`, `docs/`, `models/`.
- `src/` altında her pipeline aşaması için ayrı modül tanımlandı (yalnızca fonksiyon imzaları ve Türkçe docstring'ler).
- `requirements.txt` ve Streamlit arayüz iskeleti hazırlandı.

**Yöntemin teorisi:** — (bu adımda algoritma yok)

**Elde edilen sonuçlar:** — (bu adımda sonuç yok)

---

## Adım 1 — Veri Seti

**Ne yapıldı:**
- Kaggle "leukemia types" (`sheikhlubna/leukemia-types`) veri setinden `acute_lymphocytic`,
  `acute_myeloid` ve `healthy_blood_cell` klasörleri `data/raw/ALL`, `data/raw/AML`,
  `data/raw/Healthy` olarak alındı. Veri setinin klinik kaynağı belirtilmemiştir (kısıt).
- `src/dataset.py` yazıldı: görüntü listesi (`dosya_yolu`, `etiket`), temel özellikler, kopya
  tespiti, `grup_id` ataması ve görüntü tipi tespiti. Çıktılar:
  `data/processed/dataset_index.csv` (görüntü başına bir satır) ve
  `data/processed/kopya_ciftleri.csv` (kopya çiftleri, yöntem ve kapsam bilgisiyle).
- Hiçbir görüntü silinmedi veya değiştirilmedi; kopyalar yalnızca listelendi.
- Analiz bellek dostu çalışır: görüntüler tek tek okunur, bellekte yalnızca hash ve sayısal
  özellikler tutulur (1800 görüntü için ~1 MB), ilerleme her 100 görüntüde bir yazdırılır.

**Yöntemin teorisi:**
- **Birebir kopya (MD5):** Dosya baytlarının MD5 özeti aynıysa dosyalar bit düzeyinde aynıdır.
- **Algısal hash (pHash):** Görüntü 32x32'ye küçültülür, 2B DCT alınır, düşük frekanslı 8x8
  katsayıların medyana göre işaretinden 64 bitlik bir parmak izi çıkar. Renk/parlaklık/JPEG
  değişimlerine dayanıklıdır; iki hash arasındaki Hamming mesafesi görsel benzerliği ölçer.
- **Döndürmeye dayanıklı pHash:** pHash döndürmeye dayanıklı değildir. Bu yüzden her görüntünün
  aynasız ve aynalı hâli 10° adımlarla 36 açıda döndürülüp (72 varyant) merkezdeki 180x180
  bölgenin (256/√2; hangi açıda olursa olsun siyah dolgu köşelerini içermez) hash'i alınır.
  d(i, j) = i'nin orijinal hash'i ile j'nin 72 varyantı arasındaki en küçük mesafe.
- **ORB + RANSAC doğrulama:** pHash tek başına yetersiz kaldı — Healthy yaymalarındaki tekdüze
  eritrosit dokusu yüzünden mesafe ≤ 4'te bile zincirleme eşleşmeyle 198 farklı görüntü tek
  gruba düşüyordu. Bu nedenle hash mesafesi ≤ 14 olan 45 358 aday çift ORB anahtar noktaları
  (döndürme/ölçek değişmez ikili tanımlayıcılar) ile eşleştirildi, Lowe oran testi (0.75) ve
  RANSAC ile benzerlik dönüşümü uyduruldu. ORB aynalamaya değişmez olmadığı için j'nin aynalı
  hâli de denendi (ilk denemede aynalı kopyaların çoğu bu yüzden kaçıyordu).
  Kabul kuralı: *(hash ≤ 10 ve inlier ≥ 15) veya inlier ≥ 30*. Hash 12–14 bandında tekrarlayan
  eritrosit dokusu 25 inlier'e kadar yanlış eşleşme üretti (ör. ALL yayması ↔ Healthy yayması);
  eşikler bu çiftlerin görsel incelemesiyle seçildi.
- **Gruplama:** Kabul edilen çiftler bir grafiğin kenarları kabul edilip bağlantılı bileşenler
  (union-find) bulundu; her bileşen bir `grup_id`'dir (A~B, B~C ⇒ A, B, C aynı kaynak).
  Train/test ayrımı bu gruplara göre yapılacak (`StratifiedGroupKFold`), aksi hâlde aynı
  görüntünün döndürülmüş hâli hem eğitimde hem testte bulunur ve başarı yapay olarak şişer.
- **Görüntü tipi:** Mor lökosit çekirdekleri B kanalında yüksek, G kanalında düşüktür; B−G
  indeksine Otsu eşiği uygulanarak kaba çekirdek maskesi çıkarıldı. En büyük çekirdek görüntü
  alanının ≥ %3'ü, toplam çekirdek alanının ≥ %60'ı ve solidity ≥ 0.85 ise "tek hücre yakın
  çekim", aksi hâlde "çok hücreli yayma" sayıldı.

**Elde edilen sonuçlar:**

| Sınıf   | Görüntü | Format / kanal | Çözünürlük (en yaygın; min–max; ort.) | Siyah dolgulu (döndürülmüş) |
|---------|--------:|----------------|----------------------------------------|----------------------------:|
| ALL     | 600 | JPEG, 3 kanal RGB | 256x256 (513); 256x253 – 1712x1368; ort. 375x347 | 365 |
| AML     | 600 | JPEG, 3 kanal RGB | 256x256 (554); 256x256 – 1600x1200; ort. 314x293 | 145 |
| Healthy | 600 | JPEG, 3 kanal RGB | 256x256 (535); 256x256 – 2592x1944; ort. 509x439 | 323 |

Kopyalar (tümü sınıf içi):

| Sınıf   | Birebir fazlalık dosya (MD5) | Kopyası olan görüntü | Benzersiz grup | En büyük grup |
|---------|-----------------------------:|---------------------:|---------------:|--------------:|
| ALL     | 19  | 268 | **421** | 11 |
| AML     | 112 | 477 | **163** | 38 |
| Healthy | 84  | 451 | **248** | 24 |
| Toplam  | 215 | 1196 | **832** | — |

- 354 birebir (MD5) + 5 566 neredeyse aynı (pHash + ORB) kopya çifti bulundu.
- **Sınıflar arası kopya yok:** 13 807 sınıflar arası hash adayının hiçbiri ORB doğrulamasını
  geçmedi; MD5 düzeyinde de sınıflar arası kopya yok. Hiçbir grup birden fazla sınıf içermiyor.
- 600'er görüntülük dengeli görünen sınıflar aslında dengesizdir: AML'de yalnızca 163 benzersiz
  kaynak var (görüntülerin %80'i bir augmentasyon grubunda).

Görüntü tipi:

| Sınıf   | Yayma | Yakın çekim | Benzersiz grup: yayma / yakın çekim |
|---------|------:|------------:|------------------------------------:|
| ALL     | 310 | 290 | 185 / 236 |
| AML     | 570 | 30  | 153 / 10  |
| Healthy | 600 | 0   | 248 / 0   |

- **Önemli kısıt:** Yakın çekimlerin neredeyse tamamı ALL sınıfında, Healthy'de hiç yok. Bir
  sınıflandırıcı lösemi tipi yerine "büyütme/görüntü tipi"ni öğrenebilir (kısayol öğrenmesi).
  Hücre bazlı öznitelikler (segmentasyon sonrası tek tek hücreler) ve tip bazlı ayrı
  değerlendirme bu riski azaltmak için kullanılacak.
- Not: Görüntü tipi ve kopya tespiti sezgisel yöntemlerdir; birkaç hatalı atama olabilir.
  Hash adayı olmayan ağır kırpma/ölçek augmentasyonları kaçmış olabilir.

Görseller (`outputs/dataset/`):
- [Sınıf dağılımı (ham vs. benzersiz grup)](../outputs/dataset/sinif_dagilimi.png)
- [Görüntü tipi dağılımı](../outputs/dataset/goruntu_tipleri.png)
- [Her sınıftan örnekler](../outputs/dataset/ornek_goruntuler.png)
- [Kopya grubu örnekleri](../outputs/dataset/kopya_grup_ornekleri.png)
- [Yayma ve yakın çekim örnekleri](../outputs/dataset/yayma_ve_yakin_cekim.png)

---

## Adım 2 — Ön İşleme

**Ne yapıldı:**
- `src/preprocessing.py` dolduruldu. `preprocess(goruntu)` aşağıdaki adımları sırayla uygular ve
  `(son_görüntü, ara_çıktılar)` döndürür. Sözlükteki her anahtar arayüzde bir aşama olarak gösterilecek:

  | Anahtar | İçerik |
  |---------|--------|
  | `orijinal` | Giriş görüntüsü (RGB, orijinal boyut) |
  | `boyutlandirilmis` | 256x256, en-boy oranı korunmuş, siyah kenar dolgulu RGB |
  | `gecerli_maske` | Dolgu olmayan pikseller (bool); sonraki aşamalar bu maskeyi kullanır |
  | `renk_normalize` | Zemin hizalama sonrası RGB (segmentasyon ve Adım 4'teki renk öznitelikleri için) |
  | `kanal` | Lab b\*, ters çevrilmiş (çekirdek parlak) |
  | `filtreli` | Medyan 5x5 |
  | `kontrast` | CLAHE (clipLimit 1.0, 64 px karo) = son görüntü |

- Tüm parametreler gerekçeleriyle birlikte `src/config.py` içinde.
- `src/preprocessing_analysis.py` her tekniği karşılaştırır ve figürleri üretir (`python -m src.preprocessing_analysis`, ~30 sn).
  Her sınıftan 30 görüntü (toplam 90) seçildi; her biri **farklı kopya grubundan** (augmentasyonlar
  sonucu şişirmesin diye). Görüntüler tek tek işlendi. Bir Healthy görüntüsünde referans çekirdek
  bulunamadığı (< 20 px) için ölçümler **89 görüntüde** yapıldı.

**Ölçütler:**
- **Fisher oranı** `(μ_çekirdek − μ_arka plan)² / (σ²_çekirdek + σ²_arka plan)`: iki piksel kümesinin
  eşiklemeyle ne kadar ayrılabildiğini ölçer; büyük olması iyidir. Doğrusal yoğunluk dönüşümlerinden
  (kaydırma, ölçekleme) etkilenmez.
- **Otsu ayrışabilirliği** `η = σ²_sınıflar arası / σ²_toplam`: etiket gerektirmez, yalnızca histogramın
  iki moda ne kadar iyi bölündüğünü ölçer.
- Elle işaretlenmiş maske olmadığı için iki bağımsız kaba **referans çekirdek maskesi** kullanıldı:
  B−G + Otsu (Adım 1) ve HSV-S + Otsu. Çekirdeğin 2 px'lik geçiş halkası arka plana dahil edilmedi.
- **Gürültü:** düz arka planda ortalama |Laplace| değeri. **Kenar koruma:** referans çekirdek sınırındaki
  Sobel gradyanı, filtresiz görüntüye oranlanarak. **Canny:** kenar pikselleri içinde sınıra yakın
  olanların oranı (hassasiyet) ve sınırın yakalanan oranı (duyarlılık).

### 2.1 Boyut standartlaştırma

- **Seçim:** 256x256. En-boy oranı korunur, kısa kenar ortalanıp siyah (0) dolguyla tamamlanır.
  Küçültmede `INTER_AREA` (piksel ortalaması, örtüşme yapmaz), büyütmede `INTER_CUBIC` kullanılır.
- **Gerekçe:**
  - Görüntülerin %91'i zaten ~256 px.
  - Büyük görüntü içeren 86 grubun 43'ünde aynı alanın 256 px'lik kopyası da var. Healthy'de çekirdeğin
    görüntüdeki alan oranı her iki boyutta da 0.0018; yani 256 px'lik görüntüler büyük orijinallerin
    aynı görüş alanının küçültülmüş hâli.
  - 512'ye çıkmak görüntülerin %91'ini büyütmek olurdu: yeni bilgi eklemez, yalnızca enterpolasyon ve
    4 kat hesaplama getirir.

### 2.2 Dolgu maskesi

[dolgu_maskesi.png](../outputs/preprocessing/dolgu_maskesi.png)

- **Sorun:** Görüntülerin 833'ünde döndürme augmentasyonundan kalan köşeler var; boyutlandırma da siyah
  kenar ekliyor. Bu pikseller histogram, Otsu eşiği, CLAHE ve renk istatistiklerini bozar
  (ör. Otsu dolguyu "arka plan" sanar).
- **Siyah dolgu:**
  - Her kanalı < 15 olan pikseller dolgu sayılır. JPEG saf siyahı 0-10 aralığına dağıtır; boyanmış
    yaymada bu kadar koyu doku yoktur.
  - Bazı augmentasyonlar siyah köşelere gürültü de eklemiş. Figürün ilk satırındaki örnekte ham eşik
    dolgu bölgesinde 5437 benek piksel bırakıyor. **5x5 morfolojik açma** bu benekleri siler.
- **Koyu renkli dolgu** (Adım 3 sırasında fark edildi):
  - Görüntülerin ~%3'ünde (ALL 25, AML 13, Healthy 22; 54 farklı grup) döndürme köşeleri siyah yerine
    koyu gri/mor sabit renkle (ör. RGB ≈ 70, 60, 80) doldurulmuş. Bu renk 15 eşiğini geçtiği için
    "doku" sayılıyordu ve eritrosit tahmini ile renk istatistiklerini bozuyordu.
  - Ek kural: **koyu** (her kanal < 100), **dokusuz** (5x5 yerel std < 3), **nötr renkli**
    (maks − min kanal < 30) ve **görüntü kenarına bağlı büyük** (alanın ≥ %1'i) bölgeler dolgu sayılır.
  - Renk koşulu olmadan kenara değen 2 koyu, pürüzsüz çekirdek siyah dolguyla birleşip dolgu
    sayılıyordu. Renk koşuluyla dolguya düşen çekirdek pikseli 0, 60 adaydan 57'si yakalanıyor.
  - Kural toplam 71 görüntüde ek alan dışlıyor; bunların 14'ü siyah dolgudaki gürültü benekleri.
- **Kullanım:** Filtre ve CLAHE dolgu sınırında sahte, çok güçlü bir kenar görmesin diye dolgu önce
  geçerli piksellerin medyanıyla doldurulur, işlem sonunda yeniden 0'a çekilir.

### 2.3 Renk normalizasyonu (Reinhard)

[renk_normalizasyonu.png](../outputs/preprocessing/renk_normalizasyonu.png),
[renk_dagilimi.png](../outputs/preprocessing/renk_dagilimi.png),
[renk_normalizasyonu_skorlari.csv](../outputs/preprocessing/renk_normalizasyonu_skorlari.csv),
[normalizasyon_fisher.csv](../outputs/preprocessing/normalizasyon_fisher.csv)

**Teori:**
- Reinhard (2001) yöntemi bir görüntünün renk dağılımını hedef bir dağılıma taşır. Lab gibi kanalları
  birbirinden az bağımlı bir uzayda her kanal için `x' = (x − μ_görüntü) / σ_görüntü · σ_hedef + μ_hedef`
  uygulanır, yani birinci ve ikinci momentler hizalanır.
- Mikroskopide boya yoğunluğu, ışık ve kamera farklarını gidermek için kullanılır.
- Hedef istatistikler 832 farklı kopya grubundan **sınıf dengeli** olarak hesaplandı: önce sınıf içinde,
  sonra sınıflar arasında ortalandı.

**Problem — sınıfa göre farklı zemin rengi ve parlaklığı:** Normalizasyon öncesi Healthy'nin b\* dağılımı
136 civarında, ALL ve AML'ninki 128 civarında tepe yapıyor
([renk_dagilimi.png](../outputs/preprocessing/renk_dagilimi.png), sol sütun). Zemin parlaklığı da sınıfa
göre farklı (L\*: AML ≈ 206, Healthy ≈ 142). Bu, büyük olasılıkla farklı çekim kaynağından gelen bir
farktır. Böyle bir fark sınıflandırıcıya hücre morfolojisi yerine **zemin rengini** öğretebilir.

**Beş varyant karşılaştırıldı** (b\* = ters b\* kanalı; std = 89 görüntü arasında b\* medyanlarının yayılımı, düşük = tutarlı):

| Yöntem | Fisher (b\*) | Zemin b\* std | Çekirdek b\* std | Çekirdek−zemin farkı ALL / AML / Healthy | b\* germe ALL / AML / Healthy |
|--------|------:|------:|------:|:------:|:------:|
| Normalizasyonsuz | 22.9 | 7.4 | 12.3 | 44 / 49 / 39 | 1 / 1 / 1 |
| Klasik Reinhard (tüm pikseller) | 44.1 | 5.4 | 28.0 | 38 / 34 / 82 | 0.92 / 0.67 / **4.71** |
| Arka plan Reinhard (ortalama+std) | 26.3 | **0.8** | 20.2 | 52 / 31 / 70 | 1.26 / 0.61 / 2.15 |
| Arka plan kaydırma (Lab toplamsal) | 23.3 | 1.3 | 9.4 | 44 / 47 / 39 | 1 / 1 / 1 |
| **Zemin hizalama (seçilen)** | **22.9** | **1.3** | **9.9** | **44 / 49 / 39** | 1 / 1 / 1 |

- **Klasik Reinhard'ın sorunu:**
  - Bir görüntünün renk std'sini boyadan çok **içerik** belirliyor: AML yaymaları mor çekirdekle dolu
    (b\* std 18.6), Healthy'de neredeyse hiç çekirdek yok (2.8).
  - Bu yüzden Healthy'nin rengi ×4.7 geriliyor; [renk_normalizasyonu.png](../outputs/preprocessing/renk_normalizasyonu.png)
    dosyasında Healthy örneklerinde tuz-biber renk gürültüsü ve mavi çekirdekler görülüyor.
  - Dönüşüm sınıfa bağlı olduğu için yeni bir sahte sınıf ipucu üretiyor.
- **Arka plan Reinhard:** İstatistikleri yalnızca zeminden (eritrosit + plazma, kaba çekirdek maskesi
  dışı) almak zemini mükemmel hizalıyor. Ancak std ölçeklemesi hâlâ sınıfa bağlı (Healthy ×2.2,
  AML ×0.6) ve çekirdek−zemin farkını sınıfa göre bozuyor (44/49/39 → 52/31/70).
- **Arka plan kaydırma:** Yalnızca ortalamayı hizalamak (`x' = x − μ_zemin + μ_hedef`, üç kanal toplamsal)
  sınıfa bağlı germeyi ortadan kaldırıyor. Ancak parlak AML zeminini hedefe indirmek koyu çekirdekleri 0'a
  dayıyor: AML çekirdek piksellerinin **%34'ü** RGB'de kırpılıyor (aşağıdaki tablo).
- **Seçilen: Zemin hizalama** — zemin ortalaması hizalanır; **L\* çarpımsal**, **a\*/b\* toplamsal**:
  `L' = L · μ_hedef / μ_zemin`, `a' = a − μ_zemin + μ_hedef`, `b' = b − μ_zemin + μ_hedef`.
  - Fiziksel gerekçe: pozlama/aydınlatma farkı bir ölçek etkisidir (siyah siyah kalır), renk kayması
    ise toplamsaldır.
  - Zemin b\* tutarlılığı 7.4 → 1.3, zemin L\* tutarlılığı 41.9 → 5.4. Çekirdek−zemin farkı her sınıfta
    birebir korunuyor (44/49/39), AML çekirdek kırpması %34'ten %17'ye iniyor.
  - [renk_dagilimi.png](../outputs/preprocessing/renk_dagilimi.png) sağ sütununda üç sınıfın b\* tepe
    noktaları aynı yere (≈129) oturuyor.
- **Hedef (zemin):** L, a\*, b\* = 168.36, 132.21, 129.43 (OpenCV 8 bit ölçeği).

**Ters b\* kanalında çekirdek−arka plan Fisher oranı** (aynı 89 görüntü, görüntü bazında medyan;
[normalizasyon_fisher.csv](../outputs/preprocessing/normalizasyon_fisher.csv)):

| Durum | Tümü | ALL | AML | Healthy | Yayma | Yakın çekim | Artan / azalan görüntü (±%5) | Çekirdekte RGB kırpma ALL / AML / Healthy |
|-------|-----:|----:|----:|--------:|------:|------------:|:------:|:------:|
| Normalizasyonsuz | 22.9 | 23.6 | 17.8 | 37.0 | 21.8 | 53.4 | — | %10 / %8 / %2 |
| Klasik Reinhard | 44.1 | 38.8 | 17.7 | **109.5** | 29.0 | 53.1 | 37 / 5 | %30 / %7 / **%100** |
| **Zemin hizalama (seçilen)** | **22.9** | **23.6** | **17.9** | **36.9** | **21.7** | **53.4** | 8 / 2 | %12 / %17 / %5 |

- **Seçilen yöntem ayrışmayı değiştirmiyor** (22.9 → 22.9). Beklenen de bu: kanal başına doğrusal dönüşüm
  Fisher oranını matematiksel olarak değiştirmez. Normalizasyonun amacı ayrışmayı artırmak değil,
  renkleri **görüntüler arasında tutarlı** yapmak.
- **Klasik Reinhard'ın artışı yapaydır.** Doğrulama:
  - Normalizasyon Lab'da (float) uygulanıp RGB'ye dönülmeden ölçülünce Fisher oranı **tam olarak aynı**
    kalıyor (22.88 → 22.88).
  - Artışın tamamı Lab→RGB dönüşümündeki **gamut kırpmasından** geliyor. Healthy'de ×4.7 gerilen çekirdek
    renkleri sRGB gamutunun dışına taşıyor ve çekirdek piksellerinin **%100'ü** RGB'de 0/255'e kırpılıyor.
    Kırpılan çekirdekler tek bir doygun maviye çöküyor, çekirdek varyansı sıfıra yaklaşıyor ve Fisher
    oranı 37'den 110'a fırlıyor.
  - Bu bir bilgi kaybıdır: çekirdek içi doku (kromatin deseni) silinir. Ayrıca yalnızca Healthy'yi
    etkilediği için sınıfa bağlı bir yapaylık üretir.

### 2.4 Çekirdek kanalı seçimi

[renk_kanallari.png](../outputs/preprocessing/renk_kanallari.png) (5 aday kanal + skor tablosu),
[renk_uzaylari.png](../outputs/preprocessing/renk_uzaylari.png) (RGB/HSV/Lab/gri'nin 10 kanalının tamamı),
[kanal_skorlari.csv](../outputs/preprocessing/kanal_skorlari.csv)

**Teori:**
- **RGB:** donanımın kanalları. Parlaklık ve renk bilgisi üç kanala karışık dağılmıştır.
- **HSV:** renk tonu (H, dairesel bir açı), doygunluk (S) ve parlaklık (V) birbirinden ayrılır.
- **CIE Lab:** algısal olarak düzgün bir uzaydır. L parlaklığı, a\* yeşil↔kırmızı eksenini, b\* mavi↔sarı
  eksenini taşır.
- **Gri seviye:** 0.299R + 0.587G + 0.114B ağırlıklı toplamı.
- **B−G:** mor çekirdekte mavi yüksek, yeşil düşük; eritrosit ve plazmada B ≈ G.

**Sonuç** (renk normalizasyonu sonrası, 89 görüntünün medyanları):

| Kanal | Fisher (B−G ref.) | Fisher (S ref.) | Otsu η | Fisher ALL | Fisher AML | Fisher Healthy |
|-------|------:|------:|-----:|-----:|-----:|-----:|
| **Lab-b** | **22.9** | **19.1** | 0.82 | **23.6** | **17.9** | **36.9** |
| B−G | 18.9\* | 15.8 | 0.83 | 21.7 | 14.9 | 29.5 |
| Lab-a | 14.7 | 13.7 | 0.75 | 15.1 | 11.1 | 19.8 |
| HSV-S | 14.2 | 16.2 | 0.83 | 14.0 | 10.1 | 24.5 |
| RGB-G | 11.2 | 12.5 | 0.71 | 11.3 | 7.3 | 16.6 |
| Lab-L | 9.4 | 10.2 | 0.71 | 10.2 | 5.1 | 14.7 |
| Gri | 8.8 | 9.9 | 0.71 | 10.1 | 5.3 | 13.6 |
| RGB-R | 5.9 | 5.3 | 0.68 | 9.1 | 3.2 | 10.2 |
| HSV-V | 1.8 | 1.3 | 0.66 | 3.4 | 0.3 | 2.5 |
| RGB-B | 1.0 | 1.4 | 0.70 | 1.5 | 0.3 | 1.2 |
| HSV-H | 0.4 | 0.4 | 0.96 | 0.4 | 0.8 | 0.2 |

\* B−G referans maskesi B−G kanalından üretildiği için bu değer daireseldir (kanal kendini ölçüyor).
Bağımsız S referansında B−G 15.8, b\* ise 19.1.

- **Seçilen kanal: Lab b\*, ters çevrilmiş** (`255 − b*`, çekirdek parlak görünsün diye). Her iki
  referans maskede ve her üç sınıfta da en yüksek Fisher oranını veriyor.
- **Neden b\*:** Mor çekirdek b\*'de negatif (mavi), pembe eritrosit ve plazma nötr/pozitif. b\* eritrositi
  plazmadan neredeyse hiç ayırmaz; yani **yalnızca çekirdeğe** tepki verir. Gri seviye ve L kanalı
  eritrositleri de koyu gösterir, bu da segmentasyonu zorlaştırır.
- **Otsu η yanıltıcıdır:** HSV-H en yüksek η'yı (0.96) verdi, çünkü kırmızı ton 0/179 sınırında ikiye
  bölünüyor. Histogram çekirdek/arka plan olarak değil, bu sınırda iki moda ayrılıyor; Fisher oranı ise 0.4.
  Bu yüzden karar Fisher oranına göre verildi.
  - Normalizasyon öncesi H, Healthy'de tek başına 129 çıkıyordu, normalizasyondan sonra 0.2'ye düştü.
    Healthy'nin zemin tonu bu dairesel sınıra göre farklı yerde duruyordu.

### 2.5 Kontrast iyileştirme

[kontrast.png](../outputs/preprocessing/kontrast.png) (orijinal, HE, CLAHE clipLimit 1/2/4 + histogramlar),
[clahe_parametre_izgarasi.png](../outputs/preprocessing/clahe_parametre_izgarasi.png),
[kontrast_skorlari.csv](../outputs/preprocessing/kontrast_skorlari.csv)

**Teori:**
- **Histogram eşitleme (HE):** Tüm görüntünün kümülatif dağılımı (CDF) dönüşüm fonksiyonu olarak
  kullanılır, histogram 0-255 aralığına düzgün yayılır. Global ve kırpmasız olduğu için kalabalık arka
  plan modunu açar, bu da arka plandaki küçük farkları (gürültüyü) büyütür.
- **CLAHE:** Görüntü karolara bölünür, her karoda ayrı histogram eşitlenir, karo sınırları bilineer
  enterpolasyonla birleştirilir. Her bin `clipLimit × (karo piksel sayısı / 256)` değerinde kırpılır,
  taşan kısım binlere eşit dağıtılır. Bu kırpma dönüşüm eğimini, yani gürültü büyütmeyi sınırlar.
  - OpenCV'de `tileGridSize` piksel değil **karo sayısıdır**: 256 px'de 4x4 = 64 px, 8x8 = 32 px, 16x16 = 16 px karo.
- Ölçümler medyan filtreden sonra yapıldı (pipeline sırası).

| Yöntem | Fisher | Çekirdek−arka plan farkı | En düşük %10 fark | Arka plan gürültüsü | Fisher yakın çekim | Fisher yayma |
|--------|------:|------:|------:|------:|------:|------:|
| Orijinal | 22.0 | 39.1 | 30.1 | 0.82 | 55.8 | 20.7 |
| Histogram eşitleme | 3.4 | 114.4 | 86.9 | 21.39 | 4.0 | 2.6 |
| **CLAHE c1, 64 px** | **18.2** | **51.8** | **37.1** | **1.70** | **30.3** | **17.6** |
| CLAHE c2, 64 px | 16.7 | 59.5 | 38.9 | 2.51 | 19.9 | 15.3 |
| CLAHE c4, 64 px | 14.1 | 65.1 | 40.9 | 3.96 | 12.0 | 15.1 |
| CLAHE c1, 16 px | 18.3 | 62.5 | 50.4 | 1.58 | 20.4 | 17.6 |
| CLAHE c2, 32 px | 12.2 | 56.4 | 42.2 | 2.61 | 14.0 | 12.1 |

- **Dürüst bulgu:** b\* kanalında çekirdek zaten çok iyi ayrışıyor ve **hiçbir** kontrast yöntemi Fisher
  oranını artırmadı. Kontrast artışı arka plan gürültüsünü de büyütüyor.
  - HE gürültüyü 26 kat artırıyor, Fisher oranı 3.4'e düşüyor. Histogramda arka plan modu 0-130 aralığına yayılıyor.
  - CLAHE'de clipLimit 1 → 2 → 4 arttıkça gürültü 2 → 3 → 5 katına çıkıyor, Fisher 18.2 → 16.7 → 14.1'e düşüyor.
- **Seçilen: CLAHE `clipLimit = 1.0`, karo 64 px (`tileGridSize = 4x4`).**
  - En az gürültü büyütmesiyle en az Fisher kaybını veren ayarlardan biri (18.2). 16 px karo genel
    ölçümde başa baş (18.3), ama çekirdeğin içini ayrı ayrı eşitlediği için **yakın çekimlerde** Fisher
    oranını 20.4'e düşürüyor (64 px: 30.3). Yakın çekimler ALL'ın yarısı.
  - En düşük kontrastlı görüntülerde çekirdek-arka plan farkını 30'dan 37 gri seviyeye çıkarıyor, yani
    görüntüler arası kontrast farkını bir ölçüde dengeliyor.
  - 64 px'lik karo yaymada çekirdeği (~10-25 px) ve eritrositleri (~10-20 px) birlikte içerir; yakın
    çekimde çekirdek (~100 px) en fazla 2-3 karoya yayılır.
  - Not: 16 px'lik karoda c1 ve c1.5 aynı sonucu verir; kırpma sınırı `int(clip × 256/256)` her ikisinde de 1'e yuvarlanıyor.

### 2.6 Gürültü azaltma

[gurultu_filtreleri.png](../outputs/preprocessing/gurultu_filtreleri.png) (filtre türleri + Canny),
[gurultu.png](../outputs/preprocessing/gurultu.png) (çekirdek boyutu 3/5/7 + Canny),
[gurultu_skorlari.csv](../outputs/preprocessing/gurultu_skorlari.csv)

**Teori:**
- **Gaussian:** komşuları uzaklığa göre Gauss ağırlıklarıyla ortalar (doğrusal alçak geçiren filtre).
  Gürültüyle birlikte kenarı da yumuşatır.
- **Medyan:** komşuluğun medyanını alır (doğrusal olmayan bir filtredir). Tuz-biber ve JPEG gürültüsünü
  siler, basamak kenarları kaydırmadan korur. Çekirdek alanının yarısından küçük yapıları yok eder.
- **Bilateral:** uzamsal ağırlığı yoğunluk farkı ağırlığıyla çarpar. Yoğunluk farkı `sigmaColor`'dan
  büyük kenarlar korunur; bu yüzden σc kanalın dinamik aralığına göre ayarlanmalıdır.
- **Sıra:** Filtre CLAHE'den **önce** uygulanır, çünkü CLAHE önceden var olan gürültüyü de büyütür.

| Filtre | Gürültü (düşük iyi) | Kenar koruma (1'e yakın iyi) | Canny hassasiyet | Canny duyarlılık |
|--------|------:|------:|------:|------:|
| Filtresiz | 1.00 | 1.00 | 0.94 | 0.93 |
| Gaussian 3x3 | 0.65 | 0.94 | 0.97 | 0.91 |
| Gaussian 5x5 | 0.62 | 0.89 | 0.98 | 0.89 |
| Medyan 3x3 | 0.75 | 0.98 | 0.95 | 0.93 |
| **Medyan 5x5** | **0.62** | **0.94** | **0.97** | **0.90** |
| Medyan 7x7 | 0.55 | 0.89 | 0.99 | 0.86 |
| Bilateral d5 σc10 | 0.66 | 0.97 | 0.98 | 0.93 |
| Bilateral d5 σc25 | 0.66 | 0.91 | 0.98 | 0.89 |
| Bilateral d7 σc15 | 0.58 | 0.92 | 0.99 | 0.90 |
| Bilateral d7 σc50 | 0.58 | 0.80 | 1.00 | 0.79 |

- **Seçilen: Medyan 5x5.**
  - Gaussian 5x5 ile aynı oranda gürültü siliyor (%38), ama kenarı daha iyi koruyor (0.94'e karşı 0.89).
  - Kenar haritalarında filtresiz görüntüdeki arka plan kırıntıları kayboluyor, çekirdek konturları kalıyor.
- **Neden 3x3 değil:** 3x3 kenarı en iyi koruyan filtre (0.98), ama gürültünün yalnızca %25'ini siliyor.
  [gurultu_filtreleri.png](../outputs/preprocessing/gurultu_filtreleri.png) dosyasındaki AML örneğinde 3x3
  sonrası Canny haritası hâlâ filtresize yakın kırıntılı.
- **Çekirdek boyutu çözünürlüğe göre seçildi:**
  - 256 px'de en küçük çekirdek ~33 px alanlı (görüntünün %0.05'i, çapı ~6.5 px).
  - Medyan 5x5 yalnızca alanı < 13 px olan yapıları siler, yani çekirdekleri korur.
  - 7x7 kenar korumayı 0.89'a, Canny duyarlılığını 0.86'ya düşürüyor.
- **Bilateral:** Doğru ayarla (d5, σc10) medyana çok yakın sonuç verdi (gürültü 0.66, kenar 0.97).
  Ancak σc büyüdükçe sonuç çabuk bozuluyor (σc50'de kenar 0.80): b\*'de çekirdek-arka plan farkı yalnızca
  ~40 gri seviye, σc bu farka yaklaştığında filtre kenarın iki tarafını karıştırıyor. Renk
  augmentasyonları bu farkı görüntüden görüntüye değiştirdiği için parametresiz ve sağlam olan medyan tercih edildi.

### 2.7 Pipeline örnekleri

- [pipeline_ornekleri.png](../outputs/preprocessing/pipeline_ornekleri.png): her sınıftan 3 örneğin tüm aşamaları
  (orijinal → boyutlandırılmış → maske → renk normalizasyonu → b\* → medyan → CLAHE). Örnekler büyük
  çözünürlüklü (1712x1368, 1500x1167, 2592x1944), yakın çekim ve döndürülmüş görüntüleri de kapsıyor.
- b\* kanalında eritrositler neredeyse kayboluyor, çekirdekler parlak lekeler olarak öne çıkıyor.
  Healthy yaymalarında tek tük lenfosit/trombosit kalıyor.
- **Not:** Bazı görüntülerde preparat üzerindeki turuncu ok işaretleri b\*'de koyu leke olarak görünüyor.
  Ters kanalda koyu kaldıkları için çekirdekle karışmıyorlar.

### Özet: seçilen parametreler

| Adım | Seçim | Ana gerekçe |
|------|-------|-------------|
| Boyut | 256x256, siyah kenar dolgusu | Görüntülerin %91'i 256 px; büyük görüntüler aynı alanın yüksek çözünürlüklü hâli |
| Dolgu maskesi | her kanal ≥ 15 + koyu nötr dolgu kuralı + 5x5 açma | Siyah ve koyu renkli döndürme köşeleri ile gürültülü dolgu tüm hesaplardan dışlanır |
| Renk normalizasyonu | Zemin hizalama (L\* çarpımsal, a\*/b\* toplamsal) | Zemin hizalanır (b\* std 7.4 → 1.3), sınıfa bağlı germe yok, Fisher değişmez |
| Kanal | Lab b\* (ters) | En yüksek çekirdek/arka plan Fisher oranı (22.9), yalnızca çekirdeğe duyarlı |
| Gürültü | Medyan 5x5 | %38 gürültü azaltımı, %94 kenar koruma; parametresiz ve sağlam |
| Kontrast | CLAHE clipLimit 1.0, 64 px karo | En az gürültü büyütmesi; yakın çekimlerde en az ayrışma kaybı |

---

## Adım 3 — Segmentasyon

**Ne yapıldı:**
- `src/segmentation.py` dolduruldu. Üç yöntem de `preprocess()` ara çıktılarını alır ve **etiketli maske**
  döndürür (0 = arka plan, her çekirdek ayrı bir tam sayı):
  - `segment_otsu`: `kontrast` (ters b\*, CLAHE) üzerinde Otsu; eşik yalnızca `gecerli_maske` içindeki piksellerden.
  - `segment_kmeans`: `renk_normalize` görüntüsünde Lab K-means.
  - `segment_watershed`: Otsu maskesinden marker kontrollü watershed.
- **Ortak son işlem** (üç yöntemde aynı): açma → delik doldurma → küçük nesne temizliği.
- **Ölçek:** Görüntüler farklı büyütmelerde (yaymada eritrosit ~10 px, yakın çekimde ~62 px). Bu yüzden
  tüm boyut parametreleri sabit piksel yerine her görüntüdeki **tahmini eritrosit çapına** oranlanır.
- **Kontrast tabanı** (üç yöntemde ortak): bir piksel ancak normalize ters b\* değeri zeminden en az 15 gri
  seviye yüksekse çekirdek olabilir. Renk normalizasyonu zemini her görüntüde aynı değere taşıdığı için bu
  mutlak eşik anlamlıdır. Çekirdeği çok az olan Healthy yaymalarında Otsu ve K-means'in histogramı zemin
  içinde bölüp sahte "çekirdek" üretmesini engeller.
  - Seçim: zemin piksellerinin %99.9'u zeminin en fazla ~13 gri seviye üstünde; çekirdeklerin en koyu
    %10'u görüntülerin %95'inde 18.6'nın üstünde.
- `src/segmentation_analysis.py` karşılaştırmayı yapar ve figürleri üretir (`python -m src.segmentation_analysis`, ~1 dk).
- `python -m src.segmentation` tüm veri setinden hücreleri çıkarır (~2.5 dk).

### 3.1 Eritrosit boyutu (ölçek) tahmini

[eritrosit_boyutu.png](../outputs/segmentation/eritrosit_boyutu.png),
[eritrosit_capi_yedek.csv](../outputs/segmentation/eritrosit_capi_yedek.csv)

- **Yöntem (`estimate_rbc_diameter`):**
  1. Eritrosit maskesi: lökosit çekirdekleri (B−G maskesi, genişletilmiş) dışlanır, L kanalında Otsu
     uygulanır (eritrositler plazmadan koyu). Eritrositin soluk merkezi delik doldurmayla kapatılır.
  2. Her eritrosit bileşeninin **en büyük iç teğet çemberi** (mesafe dönüşümünün maksimumu × 2) ölçülür.
  3. Çaplar bileşen alanıyla ağırlıklandırılıp **medyanı** alınır.
- **Neden iç teğet çember:**
  - Birbirine yapışmış eritrositlerde bile tek hücrenin çapını verir.
  - Görüntü kenarında kesilmiş hücrede merkez görünüyorsa çapı doğru ölçer (görüntü sınırı ve dolgu
    mesafe dönüşümünde "arka plan" sayılmaz).
  - Yaymalarda eşdeğer çapla uyumlu (oran 0.94, korelasyon 0.88). Eşdeğer çap ise yakın çekimlerde hiç
    tahmin üretemedi, çünkü eritrositlerin hepsi kenara değiyordu.
- **Neden alan ağırlığı:** Kenarda kalan küçük eritrosit parçaları ağırlıksız medyanı yakın çekimlerde
  29 px'e çekiyordu; alan ağırlığıyla 62 px'e çıktı, gözle ölçülen değerle uyumlu.
- **Tahmin edilemeyen görüntüler:** En az 3 eritrosit bileşeni yoksa veya eritrosit kapsaması < %5 ise
  tahmin **güvenilmez** işaretlenir ve görüntü tipinin tipik çapı kullanılır: yayma **10 px**, yakın çekim
  **62 px**. Bu değerler 832 farklı grubun güvenilir tahminlerinin medyanıdır.
  - Güvenilir tahmin oranı: yayma %92, yakın çekim %94.
  - Tipik güvenilmez durumlar:
    - Çekirdekle dolu AML yaymaları: neredeyse hiç eritrosit yok, maske sitoplazma/plazma artıklarını
      yakalıyor (kapsama %1.6).
    - Dolgu sınırındaki geçiş şeridini eritrosit sanan yakın çekimler.
  - Görüntü tipi biliniyorsa indeksten alınır; yeni bir görüntüde (arayüz) `dataset.estimate_image_type` ile tahmin edilir.
  - Hücre indeksinde `eritrosit_tahmini_guvenilir` sütunu tutulur. Güvenilir hücre oranı: ALL %97,
    AML %82, Healthy %97.
- **Adım 4 için:** Alan ve çevre öznitelikleri bu çapa oranlanacak. Böylece sınıflandırıcı "büyütme oranı"
  kısayolunu (yakın çekimlerin neredeyse tamamı ALL) kullanamayacak.

### 3.2 Üç yöntemin teorisi

- **Otsu eşikleme** ([otsu_histogram.png](../outputs/segmentation/otsu_histogram.png)):
  - Histogramı t eşiğiyle iki sınıfa ayırır ve sınıflar arası varyansı en büyük yapan t'yi seçer:
    `σ²_B(t) = w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²`.
  - Sınıflar arası varyansı maksimize etmek, sınıf içi varyansı minimize etmekle eşdeğerdir.
  - Parametresizdir ve hızlıdır (~4 ms). Ama tek bir global eşik kullanır ve histogramın iki modlu olduğunu varsayar.
- **K-means renk kümeleme** ([kmeans_k_secimi.png](../outputs/segmentation/kmeans_k_secimi.png)):
  - Pikselleri renk uzayında k kümeye ayırır. Her piksel en yakın küme merkezine atanır, merkezler
    kümelerin ortalamasına güncellenir ve bu, küme içi kareler toplamı (SSE) azalmayı bırakana kadar tekrarlanır.
  - Gerçek CIE Lab ölçeği kullanılır (L\* 0-100). OpenCV'nin 8 bit Lab'ında L\*, 0-255'e gerildiği için
    Öklid mesafesinde a\*/b\*'ye göre 2.55 kat ağırlık kazanırdı. Lab'da Öklid mesafesi algısal renk farkına (ΔE) karşılık gelir.
  - **Çekirdek kümesi otomatik seçilir:** b\* merkezi en düşük (= ters b\* en yüksek, en mavi/mor) küme.
  - Model 4000 rastgele pikselle eğitilir, tüm piksellere uygulanır. `random_state = RANDOM_STATE`, `n_init = 3`.
- **Marker kontrollü Watershed** ([watershed_asamalari.png](../outputs/segmentation/watershed_asamalari.png)):
  - **Topografik benzetme:** Görüntü bir arazi yüzeyi gibi düşünülür. Yüzey en dip noktalardan suyla
    doldurulur; farklı havzalardan yükselen sular buluşacağı yerde bir set (watershed çizgisi) çekilir.
  - Burada yüzey **−mesafe dönüşümüdür**: Otsu maskesinde her pikselin arka plana uzaklığı çekirdek
    merkezlerinde tepe yapar; işaret çevrilince bu tepeler çukur olur. Birbirine değen iki yuvarlak çekirdeğin
    arasındaki dar boyun bir sırt oluşturur ve set oraya çekilir.
  - **Marker mantığı:** Her yerel minimumdan su basılırsa gürültü aşırı bölünmeye yol açar. Bu yüzden su
    yalnızca **markerlardan** basılır.
  - Markerlar mesafe dönüşümünün **h-maksimumlarıdır**, yani çevresinden en az `h = 0.15 × eritrosit çapı`
    yüksek tepeler. İki çekirdek ancak aralarındaki boyun yeterince darsa ayrılır.
  - Markersız kalan bileşene en yüksek noktasından bir marker eklenir, böylece hiçbir çekirdek kaybolmaz.

### 3.3 Parametre seçimleri

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Açma çekirdeği yarıçapı | 0.1 × eritrosit çapı (yayma 1 px, yakın çekim 6 px) | İnce köprüleri ve JPEG beneklerini siler, çekirdek şeklini bozmaz |
| Minimum çekirdek alanı | 0.2 × eritrosit alanı | Çap oranı ≈ 0.45 (eritrosit 7.5 µm ise ~3.4 µm): trombositler (2-3 µm) elenir, lenfosit çekirdeği (~eritrosit boyu) ve blastlar korunur |
| Kontrast tabanı | zemin + 15 (normalize ters b\*) | Zemin gürültüsünün (%99.9 ≈ +13) üstünde, çekirdeklerin en koyu %10'unun (görüntülerin %95'inde > 18.6) altında |
| K-means k | 3 | Aşağıdaki tablo |
| Watershed h | 0.15 × eritrosit çapı | Aşağıdaki tablo |
| Kırpım payı | 0.5 × eritrosit çapı (her yönde) | Blastların dar sitoplazması ve lökositlerin sitoplazma halkası kırpıma girer |

**K-means k seçimi** ([kmeans_k_karsilastirma.csv](../outputs/segmentation/kmeans_k_karsilastirma.csv); 90 görüntü):

| k | Sınır gradyanı | Fisher | Boş görüntü | Sınıf tutarsızlığı |
|---|------:|------:|------:|------:|
| 2 | 137.2 | 16.5 | %4.4 | 0.15 |
| **3** | **129.8** | **22.4** | **%3.3** | **0.12** |
| 4 | 119.9 | 25.8 | %2.2 | 0.19 |
| 5 | 107.9 | 27.5 | %1.1 | 0.28 |

- k = 2'de çekirdek kümesi sitoplazmayı da alıyor (düşük Fisher).
- k ≥ 4'te çekirdek kümesi yalnızca koyu çekirdek merkezine daralıyor: Fisher yükseliyor ama sınır gradyanı
  düşüyor, yani sınır çekirdeğin içine kayıyor.
- k = 3 (çekirdek / eritrosit + sitoplazma / plazma) en düşük sınıf tutarsızlığını ve gradyan/Fisher dengesini veriyor.

**Watershed marker ayarı** (ön ayar denemesi; 90 örnek + 30 ek yoğun AML yayması; dolgu ve normalizasyon
düzeltmelerinden önce ölçüldü). Ayırma çizgisi kontrastı = (medyan çekirdek içi − medyan çizgi) /
std_çekirdek içi; doğru bir bölme çizgisi daha soluk boşluktan geçer, tek çekirdeği bölen çizgi ise geçmez:

| Marker kuralı | Görüntü başına bölme (tümü / AML) | Çizgi kontrastı (tümü / AML) |
|---------------|:------:|:------:|
| min_distance = 0.5 × çap | 1.20 / 2.28 | 0.35 / 0.41 |
| h = 0.05 × çap | 0.68 / 1.27 | 0.32 / 0.50 |
| h = 0.10 × çap | 0.55 / 1.03 | 0.41 / 0.64 |
| **h = 0.15 × çap** | **0.47 / 0.88** | **0.35 / 0.72** |
| h = 0.20 × çap | 0.40 / 0.77 | 0.34 / 0.68 |
| h = 0.30 × çap | 0.29 / 0.57 | 0.46 / 0.60 |

- `min_distance` ile yapılan bölmelerin çoğu tek bir büyük blast çekirdeğinin içinden geçiyordu.
- h = 0.15, dokunan çekirdeklerin en sık olduğu AML'de en yüksek çizgi kontrastını verdi.

### 3.4 Karşılaştırma (etiket gerektirmeyen ölçütler)

[karsilastirma.csv](../outputs/segmentation/karsilastirma.csv),
[karsilastirma.md](../outputs/segmentation/karsilastirma.md),
[karsilastirma_sinif.csv](../outputs/segmentation/karsilastirma_sinif.csv),
[yontem_karsilastirma.png](../outputs/segmentation/yontem_karsilastirma.png),
[dokunan_cekirdekler.png](../outputs/segmentation/dokunan_cekirdekler.png)

Her sınıftan farklı kopya gruplarından 30'ar görüntü (76 yayma, 14 yakın çekim). Ölçütler:
- **Sınır gradyanı:** Çekirdek dış sınırı boyunca ortalama Lab renk gradyanı. Yüksek değer, sınırın gerçek
  kenara oturduğunu gösterir. Renk gradyanı kullanıldı ki b\* tabanlı yöntemler kendi kanallarında
  ölçülerek kayırılmasın.
- **Fisher oranı:** Çekirdek ile arka plan arasında, ters b\* üzerinde.
- **Küçük parça oranı** (aşırı segmentasyon): alanı < 0.5 eritrosit alanı olan nesnelerin oranı.
- **Birleşik oranı** (dokunan çekirdeklerin ayrılamaması): en az 0.5 eritrosit alanındaki nesnelerden
  solidity'si < 0.85 olanların oranı. Tek çekirdekler genelde > 0.9; birleşik çekirdekler girintili sınır oluşturur.
- **Boş görüntü oranı** (eksik segmentasyon): hiç çekirdek bulunamayan görüntüler. Her görüntüde en az bir lökosit var.
- **Sınıf tutarsızlığı:** Sınır gradyanı ve Fisher'ın sınıf medyanlarının değişim katsayısı (std / ortalama).
  0 = her sınıfta aynı başarı.
- **Süre:** görüntü başına medyan ms.

| Yöntem | Tip | Sınır gradyanı ↑ | Fisher ↑ | Küçük parça ↓ | Birleşik ↓ | Boş görüntü ↓ | Sınıf tutarsızlığı ↓ | Nesne/görüntü | Süre (ms) |
|--------|-----|------:|------:|------:|------:|------:|------:|------:|------:|
| Otsu | yayma | **144.5** | 16.0 | 0.050 | 0.072 | **%0** | 0.161 | 3.0 | **3.7** |
| K-means | yayma | 135.8 | **20.7** | **0.047** | 0.069 | %2.6 | **0.134** | 3.5 | 16.5 |
| Watershed | yayma | 144.5 | 16.0 | 0.048 | **0.067** | **%0** | 0.163 | 3.5 | 35.0 |
| Otsu | yakın çekim | 96.6 | 54.9 | **0.077** | 0 | %7.1 | 0.179 | 1.0 | **3.9** |
| K-means | yakın çekim | **98.2** | **59.4** | 0.115 | 0 | %7.1 | **0.141** | 1.0 | 16.3 |
| Watershed | yakın çekim | 96.6 | 54.9 | **0.077** | 0 | %7.1 | 0.179 | 1.0 | 38.2 |

**Sınıf bazında** (tümü):

| Yöntem | Sınıf | Sınır gradyanı | Fisher | Birleşik | Boş görüntü | Nesne/görüntü |
|--------|-------|------:|------:|------:|------:|------:|
| Otsu | ALL / AML / Healthy | 142.3 / 131.8 / 139.4 | 18.6 / 16.4 / 10.2 | 0.013 / 0.177 / 0 | %0 / %3.3 / %0 | 3 / 9.5 / 1 |
| K-means | ALL / AML / Healthy | 141.9 / 115.2 / 140.7 | 24.4 / 20.9 / 16.8 | 0.029 / 0.137 / 0.009 | %0 / %3.3 / %6.7 | 3 / 17 / 1 |
| Watershed | ALL / AML / Healthy | 142.3 / 131.8 / 139.4 | 18.5 / 16.5 / 10.1 | 0.013 / 0.164 / 0 | %0 / %3.3 / %0 | 3 / 12 / 1 |

**Seçim kuralı** ([secim_siralamasi.csv](../outputs/segmentation/secim_siralamasi.csv)):
1. Her görüntü tipinde (yayma, yakın çekim) üç yöntem altı ölçütün her birinde sıralanır (1 = en iyi;
   eşitlikte ortalama sıra). Ölçütler: sınır gradyanı, Fisher, küçük parça, birleşik, boş görüntü, sınıf tutarsızlığı.
2. Tipin ortalama sırası alınır.
3. Nihai skor = iki tipin ortalama sıralarının ortalaması. İki tip eşit ağırlıklıdır: yakın çekimler az
   olsa da ALL'ın yarısıdır.
4. En düşük skor kazanır; eşitlikte daha hızlı yöntem seçilir.
5. Süre ölçütlere girmez, yalnızca eşitlik bozucudur. Hepsi < 40 ms ve segmentasyon bir kez yapılıyor.

| Yöntem | Yayma ort. sıra | Yakın çekim ort. sıra | **Nihai skor** |
|--------|------:|------:|------:|
| **K-means** | 1.83 | 1.67 | **1.75** |
| Otsu | 2.08 | 2.17 | 2.12 |
| Watershed | 2.08 | 2.17 | 2.12 |

**Seçilen yöntem: K-means (k = 3).**
- Fisher ve sınıf tutarlılığında her iki görüntü tipinde birinci; küçük parça oranında (yayma) ve
  sınır gradyanında (yakın çekim) da birinci. Boş görüntü oranında ve yaymada sınır gradyanında sonuncu.
- Asıl üstünlüğü **AML'de**: yoğun AML yaymalarında blastların bazofilik (mavimsi) sitoplazması da ters
  b\*'de parlak. Bu yüzden yalnızca b\* kullanan Otsu ve Watershed çekirdekleri sitoplazma üzerinden
  birleştirip bütün hücre kümesini tek nesne olarak alıyor (AML'de 9.5-12 nesne/görüntü).
- K-means L\* (parlaklık) bilgisini de kullandığı için koyu çekirdeği daha açık sitoplazmadan ayırabiliyor
  (AML'de 17 nesne/görüntü, en düşük birleşik oranı 0.137). Bu yorum nesne sayısı ve birleşik oranına
  dayanıyor; elle işaretlenmiş maskeyle doğrulanmadı.
- Otsu ve Watershed kodda kalıyor; Adım 5'te sınıflandırma başarısıyla da karşılaştırılacak.

**Sınır gradyanı ölçütünün sınırlılığı:** K-means'in AML'deki düşük sınır gradyanı (115'e karşı 132)
kısmen ölçütün yapısından kaynaklanıyor olabilir. Çekirdek−sitoplazma sınırı, hücre−zemin sınırından
daha zayıf bir renk kenarıdır. Çekirdeği doğru ayıran bir yöntem bu ölçütte bütün hücreyi alan yönteme
göre daha düşük puan alır.

### 3.5 Yöntemlerin zayıf yönleri

- **Otsu:**
  - Tek global eşik kullanır. Yalnızca b\* kanalını kullandığı için bazofilik sitoplazmayı çekirdekten ayıramıyor.
  - Dokunan çekirdekleri birleştiriyor (AML birleşik oranı 0.177).
  - Çekirdek alanı çok küçük olduğunda histogram iki modlu olmuyor; kontrast tabanı olmadan zemini bölüyordu.
- **K-means:**
  - Çekirdek görüntünün ~%0.2'si kadarsa (Healthy yayma) k = 3 kümenin hiçbiri çekirdeğe ayrılmıyor;
    çekirdeği ayrı kümeye almak toplam kare hatayı anlamlı düşürmüyor. Sonuç: örnek setteki Healthy
    görüntülerinin **%6.7'sinde**, tüm veri setinde **63/600 Healthy görüntüsünde** hiç hücre bulunamadı.
  - Bunun örnekleme hatası olmadığı doğrulandı: tüm piksellerle eğitimde de aynı görüntüler boş kaldı.
  - Rastgele başlangıca bağlıdır (sabit `random_state` ile tekrarlanabilir). Otsu'dan ~4 kat yavaş.
- **Watershed:**
  - Otsu maskesine dayandığı için onun birleşik bölgelerini devralıyor. Sitoplazma üzerinden birleşmiş,
    boyunsuz yuvarlak kümelerde mesafe dönüşümü tek tepe verdiği için bölemiyor (AML birleşik oranı yalnızca 0.177 → 0.164).
  - Marker eşiği (h) hem aşırı hem eksik bölmeye duyarlı. En yavaş yöntem (~35 ms).
  - [dokunan_cekirdekler.png](../outputs/segmentation/dokunan_cekirdekler.png) dosyasında Otsu'nun
    birleştirdiği ve Watershed'in ayırdığı örnekler var. Dördüncü örnekteki gibi yoğun kümeler ayrılamadan kalıyor.
- **Ortak sınırlılıklar:**
  - Etiketli çekirdek maskesi olmadığı için ölçütlerin hepsi dolaylı.
  - Fisher oranı daralan maskeleri ödüllendirir (yüksek k'de görüldü); sınır gradyanı ise çekirdek−sitoplazma
    sınırını cezalandırabilir.
- **Olası iyileştirme (uygulanmadı):** K-means maskesinin üzerine watershed uygulamak, ya da K-means boş
  kaldığında Otsu'ya geri dönmek.

### 3.6 Hücre çıkarma

[hucre_ornekleri.png](../outputs/segmentation/hucre_ornekleri.png); hücre indeksi `data/processed/cells_index.csv`,
kırpımlar `data/processed/cells/<sınıf>/` (repoya yüklenmez).

- `extract_cells` her çekirdeğin sınır kutusunu her yönde 0.5 × eritrosit çapı genişletip kırpar.
  Kırpım renk normalize RGB görüntüden alınır.
- Her hücre için kaydedilenler:
  - kırpım (`<sınıf>_<görüntü>_<hücre>.png`) ve çekirdek maskesi (`..._maske.png`),
  - kaynak dosya, `grup_id`, etiket, görüntü tipi,
  - eritrosit çapı ve tahminin güvenilir olup olmadığı,
  - **görüntü/dolgu sınırına değip değmediği** (`kenara_degiyor`; hücre eksik olabilir),
  - çekirdek alanı ve kutu koordinatları.
- Tüm 1800 görüntü tek tek işlendi (~2.5 dk); **18.317 hücre** (36.634 dosya) çıkarıldı.

| Sınıf | Tip | Görüntü | Hücresiz görüntü | Hücre | Hücre/görüntü (medyan) | Kenara değen | Hücreli grup |
|-------|-----|------:|------:|------:|------:|------:|------:|
| ALL | yayma | 310 | 6 | 2.743 | 9 | %20 | 183 |
| ALL | yakın çekim | 290 | 0 | 337 | 1 | %13 | 236 |
| AML | yayma | 570 | 4 | 14.027 | 19 | %27 | 151 |
| AML | yakın çekim | 30 | 1 | 32 | 1 | %6 | 9 |
| Healthy | yayma | 600 | **63** | 1.178 | 2 | %5 | 229 |
| **Toplam** | | 1800 | 74 | **18.317** | | | |

- Sınıf toplamları: **ALL 3.080, AML 14.059, Healthy 1.178** hücre. Hücre içeren grup: ALL 419/421,
  AML 160/163, Healthy 229/248.
- **Adım 4-5 için önemli notlar:**
  - **Ciddi dengesizlik:** hücrelerin %77'si AML'den geliyor. AML yaymaları blastla dolu, Healthy
    yaymalarında görüntü başına 1-2 lökosit var. Hücre bazlı sınıflandırmada sınıf ağırlıkları ya da
    grup bazlı örnekleme gerekecek.
  - Aynı kaynağın augmentasyonları aynı hücreleri tekrar üretiyor (ör. AML yakın çekimde 32 hücre yalnızca
    9 gruptan). Train/test ayrımı yine `grup_id`'ye göre yapılmalı.
  - Kenara değen hücreler (%5-27) eksik çekirdek içerebilir; şekil özniteliklerinde dışlanmaları veya
    işaretlenmeleri gerekir.
  - [hucre_ornekleri.png](../outputs/segmentation/hucre_ornekleri.png) incelemesinde gözlenen hatalar:
    - AML'de kenarda ince şeritler, birleşik kalan çekirdek çiftleri ve bir zincir şeklinde nesne,
    - Healthy'de birkaç küçük ve soluk nesne (trombosit kümesi veya soluk lökosit olabilir; doğrulanmadı).

---
