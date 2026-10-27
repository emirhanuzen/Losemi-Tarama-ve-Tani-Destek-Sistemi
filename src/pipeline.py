"""Tüm aşamaları uçtan uca birleştiren ana pipeline."""

from pathlib import Path


def run_pipeline(image_path: str | Path, model_name: str, save_outputs: bool = True) -> dict:
    """
    Tek bir kan yayması görüntüsü için ön işleme -> segmentasyon -> öznitelik çıkarma
    -> sınıflandırma -> risk skoru adımlarını sırayla çalıştırır.

    Parametreler:
        image_path: Giriş görüntüsünün yolu.
        model_name: Kullanılacak kayıtlı modelin adı.
        save_outputs: True ise ara çıktılar outputs/ altına kaydedilir.

    Döndürür:
        Her aşamanın sonuçlarını içeren sözlük.
    """
    raise NotImplementedError
