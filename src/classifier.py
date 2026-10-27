"""
Aşama 4 ve 7 - Klasik Makine Öğrenmesi ile Sınıflandırma ve Değerlendirme.

Derin öğrenme kullanılmaz; SVM, Random Forest, k-NN gibi klasik yöntemler karşılaştırılır.
"""

import numpy as np
import pandas as pd


def build_models() -> dict:
    """
    Karşılaştırılacak klasik sınıflandırıcıları (ör. SVM, Random Forest, k-NN) oluşturur.

    Döndürür:
        {model adı: sklearn Pipeline} sözlüğü.
    """
    raise NotImplementedError


def train_models(X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Tüm modelleri eğitim verisi üzerinde eğitir.

    Parametreler:
        X: Öznitelik tablosu.
        y: Etiketler (normal / anormal).

    Döndürür:
        {model adı: eğitilmiş model} sözlüğü.
    """
    raise NotImplementedError


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """
    Modeli doğruluk, duyarlılık (recall), F1-skoru ve karışıklık matrisi ile değerlendirir.

    Parametreler:
        model: Eğitilmiş sınıflandırıcı.
        X_test: Test öznitelikleri.
        y_test: Test etiketleri.

    Döndürür:
        Metrikleri ve karışıklık matrisini içeren sözlük.
    """
    raise NotImplementedError


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], name: str):
    """
    Karışıklık matrisini görselleştirip outputs/evaluation/ altına kaydeder.

    Parametreler:
        cm: Karışıklık matrisi.
        labels: Sınıf adları.
        name: Kaydedilecek dosya adı (uzantısız).
    """
    raise NotImplementedError


def save_model(model, name: str):
    """
    Eğitilmiş modeli models/ klasörüne kaydeder.

    Parametreler:
        model: Eğitilmiş sınıflandırıcı.
        name: Dosya adı (uzantısız).
    """
    raise NotImplementedError


def load_model(name: str):
    """
    models/ klasöründen kayıtlı bir modeli yükler.

    Parametreler:
        name: Dosya adı (uzantısız).

    Döndürür:
        Yüklenen model.
    """
    raise NotImplementedError


def predict_cells(model, features: pd.DataFrame) -> np.ndarray:
    """
    Her hücre için normal / anormal tahmini üretir.

    Parametreler:
        model: Eğitilmiş sınıflandırıcı.
        features: Hücre öznitelik tablosu.

    Döndürür:
        Tahmin edilen etiket dizisi.
    """
    raise NotImplementedError
