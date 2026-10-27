"""
Aşama 1 - Ön İşleme.

Renk uzayı dönüşümü, kontrast iyileştirme ve gürültü azaltma adımlarını içerir.
"""

import numpy as np


def convert_color_space(image: np.ndarray, target: str = "HSV") -> np.ndarray:
    """
    RGB görüntüyü hedef renk uzayına (HSV, LAB vb.) dönüştürür.

    Parametreler:
        image: RGB giriş görüntüsü.
        target: Hedef renk uzayı adı.

    Döndürür:
        Dönüştürülmüş görüntü.
    """
    raise NotImplementedError


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Görüntünün kontrastını iyileştirir (ör. CLAHE / histogram eşitleme).

    Parametreler:
        image: Giriş görüntüsü.

    Döndürür:
        Kontrastı iyileştirilmiş görüntü.
    """
    raise NotImplementedError


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """
    Görüntüdeki gürültüyü azaltır (ör. Gauss / medyan / bilateral filtre).

    Parametreler:
        image: Giriş görüntüsü.

    Döndürür:
        Gürültüsü azaltılmış görüntü.
    """
    raise NotImplementedError


def preprocess(image: np.ndarray, save_outputs: bool = True) -> dict[str, np.ndarray]:
    """
    Ön işleme adımlarını sırayla uygular ve her adımın ara sonucunu döndürür.

    Parametreler:
        image: RGB giriş görüntüsü.
        save_outputs: True ise ara çıktılar outputs/preprocessing/ altına kaydedilir.

    Döndürür:
        {adım adı: görüntü} sözlüğü.
    """
    raise NotImplementedError
