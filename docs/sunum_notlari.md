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
