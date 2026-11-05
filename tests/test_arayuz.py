"""
Streamlit arayüzünün uçtan uca testi (streamlit.testing.v1.AppTest; tarayıcı gerekmez).

Kontroller:
1. Ana uygulama (varsayılan Analiz sayfası) ve dört sayfanın her biri hatasız açılıyor.
2. 7 hazır örnek görüntünün hepsi analiz ediliyor; risk seviyesi outputs/risk/ornek_raporlar_kalibre.csv ile aynı;
   PDF indirme butonu geçerli bir PDF veriyor.
3. İlk örnekte "Üç yöntemi karşılaştır" anahtarı ve hücre seçimi çalışıyor.
4. Bozuk dosya anlaşılır hata mesajı veriyor; hücresiz görüntü uygulamayı çökertmeden uyarı veriyor.

Çalıştırma (proje kökünden):
    python -m tests.test_arayuz
Çıkış kodu: tüm kontroller geçerse 0, aksi hâlde 1.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit.testing.v1.app_test as app_test_module
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["sayfa_analiz", "sayfa_performans", "sayfa_yontem", "sayfa_hakkinda"]
TIMEOUT = 300   # sn; ilk analizde model yükleme ve büyük görüntüler (2592x1944) dahil
INPUT_KEY = "analiz_girdisi"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RecordingMediaStorage(MemoryMediaFileStorage):
    """
    AppTest'in her çalıştırmada oluşturduğu bellek içi medya deposunu kaydeder.

    AppTest çalıştırma bitince sahte runtime'ı bıraktığı için indirme butonunun dosyasına sonradan
    ulaşmanın yolu, depoyu oluşturulurken yakalamaktır.
    """

    instances: list["RecordingMediaStorage"] = []

    def __init__(self, *args, **kwargs):
        """Depoyu oluşturur ve sınıf listesine ekler (parametreler MemoryMediaFileStorage ile aynı)."""
        super().__init__(*args, **kwargs)
        RecordingMediaStorage.instances.append(self)


app_test_module.MemoryMediaFileStorage = RecordingMediaStorage


def run_page(root: str, module: str) -> None:
    """
    AppTest.from_function için: tek bir sayfa modülünün page() fonksiyonunu çalıştırır.

    Parametreler:
        root: Proje kökü (sys.path'e eklenir).
        module: app paketi içindeki sayfa modülünün adı.
    """
    import importlib
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    importlib.import_module(f"app.{module}").page()


def problems(at: AppTest) -> list[str]:
    """
    Bir çalıştırmadaki yakalanmış istisnaları ve st.error mesajlarını toplar.

    Parametreler:
        at: Çalıştırılmış AppTest.

    Döndürür:
        Sorun metinleri listesi (boşsa sorun yok).
    """
    return [f"istisna: {e.value}" for e in at.exception] + [f"hata: {e.value}" for e in at.error]


def risk_level_shown(at: AppTest) -> str | None:
    """
    Sayfadaki risk seviyesi rozetinin metnini bulur.

    Parametreler:
        at: Çalıştırılmış AppTest.

    Döndürür:
        Rozet metni (ör. "Yüksek") veya bulunamazsa None.
    """
    for md in at.markdown:
        if md.value.startswith("Risk seviyesi"):
            return md.value.rsplit(">", 2)[-2].split("<")[0].strip()
    return None


def pdf_from_button(at: AppTest) -> bytes | None:
    """
    PDF indirme butonunun sunduğu dosyanın baytlarını AppTest'in bellek içi medya deposundan okur.

    Parametreler:
        at: Çalıştırılmış AppTest.

    Döndürür:
        PDF baytları veya buton yoksa None.
    """
    if not at.download_button:
        return None
    url = at.download_button[0].proto.url
    return RecordingMediaStorage.instances[-1].get_file(url.rsplit("/", 1)[-1]).content


def check_pages() -> list[str]:
    """
    Ana uygulamayı ve her sayfayı ayrı ayrı çalıştırır.

    Döndürür:
        Hata metinleri listesi.
    """
    errors = []
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    found = problems(at)
    print(f"  ana uygulama: {'OK' if not found else found}")
    errors += [f"ana uygulama: {p}" for p in found]
    for module in PAGES:
        at = AppTest.from_function(run_page, args=(str(ROOT), module), default_timeout=TIMEOUT).run()
        found = problems(at)
        n_img = len(at.get("image"))
        print(f"  {module}: {'OK' if not found else found} (başlık: {at.title[0].value if at.title else '-'}, "
              f"görsel öğesi: {n_img})")
        errors += [f"{module}: {p}" for p in found]
    return errors


def check_examples() -> list[str]:
    """
    7 hazır örneğin her birini arayüzden analiz eder; risk seviyesi ve PDF'i doğrular.

    Döndürür:
        Hata metinleri listesi.
    """
    from app.ortak import example_images

    expected = pd.read_csv(ROOT / "outputs" / "risk" / "ornek_raporlar_kalibre.csv")
    expected = dict(zip(expected["goruntu"], expected["seviye_pipeline"]))
    examples = example_images()
    errors = []
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    for i, row in examples.iterrows():
        start = time.time()
        at.selectbox[0].set_value(i)
        at.button[0].click()
        at.run(timeout=TIMEOUT)
        name = Path(row["dosya_yolu"]).name
        found = problems(at)
        level = risk_level_shown(at)
        pdf = pdf_from_button(at)
        ok_pdf = pdf is not None and pdf.startswith(b"%PDF") and len(pdf) > 10_000
        ok_level = level == expected.get(name)
        print(f"  [{i + 1}/{len(examples)}] {row['metin']}: risk {level} (beklenen {expected.get(name)}), "
              f"PDF {'OK' if ok_pdf else 'YOK'} ({0 if pdf is None else len(pdf) // 1024} KB), "
              f"{time.time() - start:.1f} sn" + (f", sorun: {found}" if found else ""))
        errors += [f"{name}: {p}" for p in found]
        if not ok_level:
            errors.append(f"{name}: risk seviyesi {level}, beklenen {expected.get(name)}")
        if not ok_pdf:
            errors.append(f"{name}: PDF indirme butonu geçerli PDF vermedi")
        if i == 0:
            errors += check_interactions(at)
    return errors


def check_interactions(at: AppTest) -> list[str]:
    """
    Analiz edilmiş bir görüntüde üç yöntem karşılaştırmasını ve hücre seçimini dener.

    Parametreler:
        at: Bir örneğin analiz edildiği AppTest.

    Döndürür:
        Hata metinleri listesi.
    """
    errors = []
    at.toggle[0].set_value(True)
    at.run(timeout=TIMEOUT)
    methods = [m.value for m in at.markdown if " çekirdek" in m.value and m.value.startswith("**")
               and "—" in m.value]
    found = problems(at)
    print(f"    üç yöntem karşılaştırması: {methods}" + (f", sorun: {found}" if found else ""))
    if len(methods) != 3 or found:
        errors.append(f"yöntem karşılaştırması: {len(methods)} yöntem, {found}")
    cell_box = at.selectbox[1]
    cell_box.set_value(cell_box.options and int(cell_box.options[-1].split()[1])).run(timeout=TIMEOUT)
    found = problems(at)
    tables = [d.value for d in at.dataframe]
    n_feat = max((len(t) for t in tables if "Öznitelik" in t.columns), default=0)
    print(f"    hücre seçimi: öznitelik tablosu {n_feat} satır" + (f", sorun: {found}" if found else ""))
    if n_feat < 10 or found:
        errors.append(f"hücre seçimi: öznitelik tablosu {n_feat} satır, {found}")
    at.toggle[0].set_value(False)
    return errors


def encode_png(image: np.ndarray) -> bytes:
    """
    RGB görüntüyü PNG baytlarına çevirir.

    Parametreler:
        image: RGB uint8 görüntü.

    Döndürür:
        PNG baytları.
    """
    return cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()


def check_bad_inputs() -> list[str]:
    """
    Bozuk dosyanın anlaşılır hata verdiğini ve hücresiz görüntünün çökmeden uyarı verdiğini doğrular.

    Döndürür:
        Hata metinleri listesi.
    """
    errors = []
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.session_state[INPUT_KEY] = {"veri": b"bu bir goruntu degil", "ad": "bozuk.png"}
    at.run(timeout=TIMEOUT)
    msgs = [e.value for e in at.error]
    ok = not at.exception and any("okunamadı" in m for m in msgs)
    print(f"  bozuk dosya: {'OK' if ok else 'BAŞARISIZ'} ({msgs[:1]})")
    if not ok:
        errors.append(f"bozuk dosya: beklenen hata mesajı yok ({msgs}, {[e.value for e in at.exception]})")

    blank = np.full((400, 400, 3), (232, 190, 200), np.uint8)   # düz pembe zemin: çekirdek yok
    at.session_state[INPUT_KEY] = {"veri": encode_png(blank), "ad": "bos_zemin.png"}
    at.run(timeout=TIMEOUT)
    msgs = [e.value for e in at.error]
    level = risk_level_shown(at)
    ok = not at.exception and level == "Değerlendirilemedi" and any("bulunamadı" in m for m in msgs)
    print(f"  hücresiz görüntü: {'OK' if ok else 'BAŞARISIZ'} (risk {level}, PDF "
          f"{'var' if pdf_from_button(at) else 'yok'})")
    if not ok:
        errors.append(f"hücresiz görüntü: risk {level}, mesajlar {msgs}, istisna {[e.value for e in at.exception]}")
    return errors


def main() -> int:
    """
    Tüm kontrolleri çalıştırır ve özeti yazdırır.

    Döndürür:
        Çıkış kodu (0 başarılı, 1 başarısız).
    """
    start = time.time()
    errors = []
    print("1) Sayfalar")
    errors += check_pages()
    print("2) Hazır örnekler")
    errors += check_examples()
    print("3) Hatalı girdiler")
    errors += check_bad_inputs()
    print(f"\nSüre: {time.time() - start:.0f} sn")
    if errors:
        print(f"BAŞARISIZ: {len(errors)} sorun")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("TÜM KONTROLLER GEÇTİ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
