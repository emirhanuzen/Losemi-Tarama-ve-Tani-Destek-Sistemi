"""
Aşama 2 - Segmentasyon.

Otsu, K-means ve Watershed yöntemleri ayrı ayrı uygulanır, sonuçları
karşılaştırılarak en uygun yöntem seçilir.
"""

import numpy as np


def segment_otsu(image: np.ndarray) -> np.ndarray:
    """
    Otsu eşikleme ile hücre bölgelerini arka plandan ayırır.

    Parametreler:
        image: Ön işlenmiş giriş görüntüsü.

    Döndürür:
        İkili maske (hücre = 1, arka plan = 0).
    """
    raise NotImplementedError


def segment_kmeans(image: np.ndarray, k: int = 3) -> np.ndarray:
    """
    K-means kümeleme ile pikselleri renk benzerliğine göre gruplayarak segmentasyon yapar.

    Parametreler:
        image: Ön işlenmiş giriş görüntüsü.
        k: Küme sayısı.

    Döndürür:
        İkili maske (hücre = 1, arka plan = 0).
    """
    raise NotImplementedError


def segment_watershed(image: np.ndarray) -> np.ndarray:
    """
    Watershed algoritması ile birbirine değen hücreleri ayırarak etiketli maske üretir.

    Parametreler:
        image: Ön işlenmiş giriş görüntüsü.

    Döndürür:
        Etiketli maske (her hücre farklı bir tam sayı etiketi).
    """
    raise NotImplementedError


def compare_methods(image: np.ndarray, ground_truth: np.ndarray | None = None) -> dict:
    """
    Üç segmentasyon yöntemini aynı görüntü üzerinde çalıştırıp karşılaştırır.

    Parametreler:
        image: Ön işlenmiş giriş görüntüsü.
        ground_truth: Varsa referans maske (Dice/IoU hesabı için).

    Döndürür:
        Yöntem adı -> {maske, metrikler} içeren sözlük.
    """
    raise NotImplementedError


def select_best_method(comparison: dict) -> str:
    """
    Karşılaştırma sonuçlarına göre en uygun segmentasyon yöntemini seçer.

    Parametreler:
        comparison: compare_methods çıktısı.

    Döndürür:
        Seçilen yöntemin adı.
    """
    raise NotImplementedError


def extract_cells(image: np.ndarray, mask: np.ndarray) -> list[dict]:
    """
    Maskedeki her bağlı bileşeni ayrı bir hücre olarak çıkarır.

    Parametreler:
        image: Orijinal RGB görüntü.
        mask: İkili ya da etiketli maske.

    Döndürür:
        Her hücre için {etiket, sınır kutusu, hücre maskesi, hücre görüntüsü} listesi.
    """
    raise NotImplementedError
