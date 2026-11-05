"""
Streamlit arayüzü: Analiz, Model Performansı, Yöntem ve Hakkında sayfaları.

Çalıştırma (proje kökünden):
    streamlit run app/streamlit_app.py
Tarayıcıda: http://localhost:8501
"""

import sys
from pathlib import Path

import streamlit as st

# src ve app paketlerini import edebilmek için proje kökünü yola ekle
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import sayfa_analiz, sayfa_hakkinda, sayfa_performans, sayfa_yontem  # noqa: E402

APP_TITLE = "Lösemi Tarama ve Tanı Destek Sistemi"


def main() -> None:
    """Sayfa ayarlarını yapar, kenar çubuğunu ve sayfa gezinmesini oluşturup seçili sayfayı çalıştırır."""
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/biotech:", layout="wide")
    pages = [
        st.Page(sayfa_analiz.page, title="Analiz", icon=":material/biotech:", url_path="analiz", default=True),
        st.Page(sayfa_performans.page, title="Model Performansı", icon=":material/monitoring:",
                url_path="performans"),
        st.Page(sayfa_yontem.page, title="Yöntem", icon=":material/schema:", url_path="yontem"),
        st.Page(sayfa_hakkinda.page, title="Hakkında", icon=":material/info:", url_path="hakkinda"),
    ]
    nav = st.navigation(pages)
    with st.sidebar:
        st.markdown(f"**{APP_TITLE}**")
        st.caption("Sayısal Görüntü İşleme dersi projesi. Araştırma ve eğitim amaçlı prototip; klinik teşhis aracı "
                   "değildir.")
    nav.run()


main()
