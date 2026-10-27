"""
Streamlit arayüzü - pipeline'ın her aşamasını interaktif olarak gösterir.

Çalıştırma: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# src paketini import edebilmek için proje kökünü yola ekle
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    """Arayüzün ana giriş noktası: sayfa düzenini ve aşama sekmelerini oluşturur."""
    st.set_page_config(page_title="Lösemi Tarama Sistemi", layout="wide")
    st.title("Görüntü İşleme Tabanlı Akıllı Lösemi Tarama ve Tanı Destek Sistemi")

    uploaded = st.file_uploader("Kan yayması görüntüsü yükleyin", type=["png", "jpg", "jpeg", "bmp", "tif"])

    tabs = st.tabs([
        "1. Ön İşleme",
        "2. Segmentasyon",
        "3. Öznitelikler",
        "4. Sınıflandırma",
        "5. Risk ve Rapor",
        "6. Değerlendirme",
    ])
    for tab in tabs:
        with tab:
            st.info("Bu aşama henüz uygulanmadı.")

    if uploaded is None:
        st.caption("Başlamak için bir görüntü yükleyin.")


if __name__ == "__main__":
    main()
