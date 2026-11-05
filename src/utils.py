"""Görüntü okuma/yazma ve ara çıktı kaydetme gibi ortak yardımcı fonksiyonlar."""

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import matplotlib

matplotlib.use("Agg")  # Ekransız ortamda (betik/Streamlit) figür kaydetmek için
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURE_PREFIXES, PROJECT_ROOT, STAGE_OUTPUT_DIRS

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def resolve_path(path: str | Path) -> Path:
    """
    CSV'den veya kullanıcıdan gelen bir yolu bu platformda açılabilecek mutlak yola çevirir.

    - Göreli yol (ör. "data/raw/ALL/x.jpg" ya da ters eğik çizgili hâli) proje köküne (config.PROJECT_ROOT) eklenir.
    - Mutlak yol bu platformun yoluysa olduğu gibi döner.
    - Windows mutlak yolu (ör. "C:\\...\\Losemi-Tarama-ve-Tani-Destek-Sistemi\\data\\raw\\...") Windows dışında
      açılırsa: proje klasörünün adı yolda geçiyorsa geri kalanı proje köküne eklenir (proje başka yere kopyalansa
      da çalışır); geçmiyorsa WSL bağlama noktasına çevrilir (C:\\x → /mnt/c/x).

    Parametreler:
        path: Göreli veya mutlak yol (str veya Path).

    Döndürür:
        Mutlak Path.
    """
    text = str(path)
    if _WINDOWS_ABSOLUTE.match(text) and os.name != "nt":
        parts = PureWindowsPath(text).parts
        if PROJECT_ROOT.name in parts:
            rest = parts[len(parts) - parts[::-1].index(PROJECT_ROOT.name):]
            return PROJECT_ROOT.joinpath(*rest)
        return Path("/mnt", parts[0][0].lower(), *parts[1:])
    # Windows dışında ters eğik çizgi ayraç değildir; göreli Windows yolları da çözülebilsin diye çevrilir
    candidate = Path(text) if os.name == "nt" else Path(text.replace("\\", "/"))
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def to_relative(path: str | Path) -> str:
    """
    Bir yolu CSV'ye yazılacak biçime getirir: proje kökü altındaysa köke göre göreli, ileri eğik çizgili.

    Proje dışındaki yollar (ör. ALL-IDB) mutlak kalır, yine ileri eğik çizgiyle yazılır.

    Parametreler:
        path: Göreli veya mutlak yol.

    Döndürür:
        Ör. "data/raw/ALL/acute_lymphocytic (1).jpg".
    """
    full = resolve_path(path)
    try:
        return PurePosixPath(*full.relative_to(PROJECT_ROOT).parts).as_posix()
    except ValueError:
        return full.as_posix()


def load_image(path: str | Path) -> np.ndarray:
    """
    Diskteki bir görüntüyü RGB formatında okur.

    cv2.imread Windows'ta Türkçe/Unicode karakterli yolları okuyamadığı için dosya
    önce bayt dizisi olarak okunur, ardından cv2.imdecode ile çözülür. Yol resolve_path ile çözülür
    (göreli CSV yolları ve Windows yolları her platformda açılır).

    Parametreler:
        path: Görüntü dosyasının yolu (göreli veya mutlak).

    Döndürür:
        RGB formatında (H, W, 3) uint8 numpy dizisi.
    """
    data = np.fromfile(str(resolve_path(path)), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stage_dir(stage: str) -> Path:
    """
    Bir aşamanın çıktı klasörünü döndürür; yoksa oluşturur.

    Parametreler:
        stage: Aşama adı (config.STAGE_OUTPUT_DIRS anahtarlarından biri).

    Döndürür:
        outputs/<stage>/ klasörünün yolu.
    """
    out_dir = STAGE_OUTPUT_DIRS[stage]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_stage_output(image: np.ndarray, stage: str, name: str) -> Path:
    """
    Bir pipeline aşamasının ara çıktısını outputs/<stage>/ klasörüne kaydeder.

    Parametreler:
        image: Kaydedilecek görüntü (RGB, gri veya ikili maske).
        stage: Aşama adı (ör. "preprocessing", "segmentation").
        name: Dosya adı (uzantısız).

    Döndürür:
        Kaydedilen dosyanın yolu.
    """
    path = stage_dir(stage) / f"{name}.png"
    img = image
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Görüntü kodlanamadı: {name}")
    buf.tofile(str(path))
    return path


def save_figure(fig: plt.Figure, stage: str, name: str) -> Path:
    """
    Hazır bir matplotlib figürünü outputs/<stage>/ klasörüne PNG olarak kaydeder ve kapatır.

    Parametreler:
        fig: Kaydedilecek matplotlib figürü.
        stage: Aşama adı.
        name: Dosya adı (uzantısız).

    Döndürür:
        Kaydedilen figürün yolu.
    """
    path = stage_dir(stage) / f"{name}.png"
    # dpi=150: sunum slaytında net görünecek, dosya boyutu makul kalacak çözünürlük
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_comparison_figure(images: dict[str, np.ndarray], stage: str, name: str) -> Path:
    """
    Birden fazla görüntüyü başlıklarıyla yan yana tek bir figürde kaydeder (sunum için).

    Parametreler:
        images: {başlık: görüntü} sözlüğü.
        stage: Aşama adı.
        name: Dosya adı (uzantısız).

    Döndürür:
        Kaydedilen figürün yolu.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return save_figure(fig, stage, name)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Sınıflandırmada kullanılabilecek öznitelik sütunlarını döndürür (sekil_/renk_/doku_ önekli).

    Meta veri (meta_*), etiket, grup_id ve hucre_id bilinçli olarak hariçtir.

    Parametreler:
        df: Öznitelik tablosu.

    Döndürür:
        Sütun adları listesi.
    """
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
