"""
Aşama 3 - Öznitelik Çıkarma.

Her hücre için şekil, renk ve doku (GLCM) öznitelikleri hesaplanır.
"""

import numpy as np
import pandas as pd


def extract_shape_features(cell_mask: np.ndarray) -> dict[str, float]:
    """
    Şekil özniteliklerini hesaplar: alan, çevre, dairesellik, eksantrisite, solidity.

    Parametreler:
        cell_mask: Tek bir hücreye ait ikili maske.

    Döndürür:
        {öznitelik adı: değer} sözlüğü.
    """
    raise NotImplementedError


def extract_color_features(cell_image: np.ndarray, cell_mask: np.ndarray) -> dict[str, float]:
    """
    Renk özniteliklerini hesaplar: RGB ve HSV kanallarının ortalama, std vb. istatistikleri.

    Parametreler:
        cell_image: Hücrenin RGB görüntüsü.
        cell_mask: Hücre maskesi (yalnızca hücre pikselleri kullanılır).

    Döndürür:
        {öznitelik adı: değer} sözlüğü.
    """
    raise NotImplementedError


def extract_texture_features(cell_image: np.ndarray, cell_mask: np.ndarray) -> dict[str, float]:
    """
    GLCM tabanlı doku özniteliklerini hesaplar: contrast, correlation, energy, homogeneity.

    Parametreler:
        cell_image: Hücrenin RGB görüntüsü.
        cell_mask: Hücre maskesi.

    Döndürür:
        {öznitelik adı: değer} sözlüğü.
    """
    raise NotImplementedError


def extract_all_features(cells: list[dict]) -> pd.DataFrame:
    """
    Tüm hücreler için şekil, renk ve doku özniteliklerini birleştirip tablo oluşturur.

    Parametreler:
        cells: segmentation.extract_cells çıktısı.

    Döndürür:
        Her satırı bir hücre, her sütunu bir öznitelik olan DataFrame.
    """
    raise NotImplementedError
