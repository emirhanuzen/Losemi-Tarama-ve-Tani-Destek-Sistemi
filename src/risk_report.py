"""
Aşama 5 - Risk Skoru ve Otomatik PDF Rapor.

Anormal hücre oranından risk skoru hesaplanır ve ReportLab ile PDF rapor üretilir.
"""

from pathlib import Path

import numpy as np


def compute_risk_score(predictions: np.ndarray) -> dict:
    """
    Anormal hücre oranına göre risk skorunu ve risk seviyesini hesaplar.

    Parametreler:
        predictions: Hücre bazlı tahminler (0 = normal, 1 = anormal).

    Döndürür:
        {toplam hücre, anormal hücre, oran, risk seviyesi} sözlüğü.
    """
    raise NotImplementedError


def generate_pdf_report(risk: dict, stage_images: dict[str, Path], output_name: str) -> Path:
    """
    Risk skoru ve pipeline ara görüntülerini içeren PDF raporu reports/ klasörüne üretir.

    Parametreler:
        risk: compute_risk_score çıktısı.
        stage_images: {aşama adı: görüntü yolu} sözlüğü.
        output_name: PDF dosya adı (uzantısız).

    Döndürür:
        Oluşturulan PDF dosyasının yolu.
    """
    raise NotImplementedError
