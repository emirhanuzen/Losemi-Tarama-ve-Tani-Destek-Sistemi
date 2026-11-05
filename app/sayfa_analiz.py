"""
Analiz sayfası: görüntü seçimi/yükleme, uçtan uca pipeline ve aşamaların gösterimi, PDF rapor indirme.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.ortak import (
    COMPARE_CONTOUR_COLOR,
    DISCLAIMER,
    GROUP_NAMES,
    analyze_with_progress,
    boundary_overlay,
    compare_methods,
    disclaimer,
    example_bytes,
    example_images,
    image_key,
    label_mask_image,
    legend_html,
    load_model_cached,
    pdf_bytes,
    preprocessing_panels,
    risk_badge,
    to_display,
    top_features,
)
from src.config import CLASS_NAMES, RISK_NO_CELL_LEVEL
from src.dataset import CLASS_COLORS
from src.risk_report import METHOD_NAMES, dominant_label, fmt_pct

UPLOAD_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
GRID_COLUMNS = 6       # Hücre ızgarasında satır başına kırpım: 1080p ekranda kırpımlar ~150 px kalır
INPUT_KEY = "analiz_girdisi"


def choose_input() -> None:
    """
    Görüntü kaynağını (hazır örnek veya yükleme) seçtirir; "Analiz et" ile seçimi oturum durumuna yazar.

    Analiz girdisi st.session_state'te tutulur; böylece sayfadaki başka bir etkileşim (hücre seçimi, yöntem
    karşılaştırması anahtarı) sonuçları kaybettirmez.
    """
    source = st.radio("Görüntü kaynağı", ["Hazır örnek", "Görüntü yükle"], horizontal=True,
                      help="Hazır örnekler, kaynağı bilinmeyen kilitli test görüntülerinden seçildi "
                           "(reports/ornekler/ PDF'lerinin görüntüleri).")
    data, name = None, None
    if source == "Hazır örnek":
        examples = example_images()
        if examples.empty:
            st.error("Hazır örnek listesi bulunamadı (outputs/risk/ornek_secimi_kalibre.csv). "
                     "`python -m src.risk_analysis analiz` ile üretilebilir.")
            return
        idx = st.selectbox("Örnek görüntü", examples.index, format_func=lambda i: examples.loc[i, "metin"])
        row = examples.loc[idx]
        data = example_bytes(row["dosya_yolu"])
        name = Path(row["dosya_yolu"]).name
        if data is None:
            st.error(f"Örnek görüntü diskte yok: `{row['dosya_yolu']}`. Veri setini `data/raw/` altına "
                     "yerleştirin (README → Veri seti) veya kendi görüntünüzü yükleyin.")
    else:
        uploaded = st.file_uploader("Kan yayması görüntüsü (JPG, PNG, TIF)", type=UPLOAD_TYPES)
        if uploaded is not None:
            data, name = uploaded.getvalue(), uploaded.name
    if st.button("Analiz et", type="primary", disabled=data is None, icon=":material/play_arrow:"):
        st.session_state[INPUT_KEY] = {"veri": data, "ad": name}


def show_preprocessing(result: dict) -> None:
    """
    Ön işleme ara çıktılarını açıklamalarıyla ızgarada gösterir.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    panels = preprocessing_panels(result["on_isleme"])
    for start in range(0, len(panels), 4):
        cols = st.columns(4)
        for col, (title, image, text) in zip(cols, panels[start:start + 4]):
            with col:
                shown = image if title == "Orijinal" else to_display(image)
                st.image(shown, width="stretch")
                st.markdown(f"**{title}**")
                st.caption(text)


def show_segmentation(result: dict) -> None:
    """
    Kullanılan segmentasyon yönteminin maskesini ve konturlarını, istenirse üç yöntemin karşılaştırmasını gösterir.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    seg, steps = result["segmentasyon"], result["on_isleme"]
    n_cells = len(result["hucreler"])
    method = METHOD_NAMES[seg["yontem"]]
    if seg["geri_donus"]:
        st.warning(f"K-means bu görüntüde hiç çekirdek bulamadı; **Otsu geri dönüşü** kullanıldı. Model Otsu "
                   "kırpımlarıyla eğitilmediği için sonuç düşük güvenli sayılır.", icon=":material/info:")
    st.markdown(f"Kullanılan yöntem: **{method}** · bulunan çekirdek: **{n_cells}** · tahmini eritrosit çapı "
                f"**{seg['eritrosit_capi']:.1f} px** "
                f"({'görüntüden ölçüldü' if seg['eritrosit_tahmini_guvenilir'] else 'görüntü tipi yedek değeri'}) · "
                f"görüntü tipi: **{'yayma' if seg['tip'] == 'yayma' else 'yakın çekim'}**")
    c1, c2 = st.columns(2)
    with c1:
        st.image(to_display(label_mask_image(seg["etiketler"])), width="stretch")
        st.caption("Çekirdek maskesi (beyaz). Boyut eşikleri eritrosit çapına oranlıdır; çok küçük parçalar atılır.")
    with c2:
        st.image(boundary_overlay(steps["renk_normalize"], seg["etiketler"], COMPARE_CONTOUR_COLOR), width="stretch")
        st.caption(f"{method} çekirdek konturları, renk normalize görüntü üzerinde.")

    if st.toggle("Üç yöntemi karşılaştır", help="Aynı görüntüde Otsu, K-means ve Watershed sonuçları yan yana."):
        labels = compare_methods(image_key(st.session_state[INPUT_KEY]["veri"]), steps, seg["eritrosit_capi"])
        cols = st.columns(3)
        for col, (m, lab) in zip(cols, labels.items()):
            with col:
                n = int(len(np.unique(lab)) - 1)
                st.image(boundary_overlay(steps["renk_normalize"], lab, COMPARE_CONTOUR_COLOR), width="stretch")
                st.markdown(f"**{METHOD_NAMES[m]}** — {n} çekirdek" + (" (seçilen)" if m == "kmeans" else ""))
        st.caption("Otsu ve Watershed yalnızca b* kanalına dayanır; yoğun AML yaymalarında komşu çekirdekleri "
                   "birleştirebilir. K-means parlaklığı da kullanır ama soluk normal yaymalarda çekirdek "
                   "kaçırabilir. Bu yüzden pipeline K-means kullanır, hücre bulamazsa Otsu'ya döner.")


def probability_chart(row: pd.Series) -> alt.Chart:
    """
    Bir hücrenin sınıf olasılıkları için sınıf renkli yatay çubuk grafiği oluşturur.

    Parametreler:
        row: Hücre tablosu satırı (p_ALL, p_AML, p_Healthy).

    Döndürür:
        Altair grafiği.
    """
    data = pd.DataFrame({"Sınıf": CLASS_NAMES, "Olasılık": [float(row[f"p_{c}"]) for c in CLASS_NAMES]})
    base = alt.Chart(data).encode(
        y=alt.Y("Sınıf:N", sort=CLASS_NAMES, title=None),
        x=alt.X("Olasılık:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Olasılık"),
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("Sınıf:N", scale=alt.Scale(domain=CLASS_NAMES, range=[CLASS_COLORS[c] for c in CLASS_NAMES]),
                        legend=None))
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("Olasılık:Q", format=".1%"))
    return (bars + text).properties(height=140)


def feature_table(features: pd.Series) -> pd.DataFrame:
    """
    Seçilen hücrenin en ayırt edici öznitelik değerlerini sınıf medyanlarıyla birlikte tabloya koyar.

    Parametreler:
        features: Hücrenin öznitelik satırı (features.crop_features çıktısı).

    Döndürür:
        Grup, Öznitelik, Bu hücre, Medyan ALL/AML/Healthy sütunlu tablo.
    """
    top = top_features()
    return pd.DataFrame({
        "Grup": top["grup"].map(GROUP_NAMES),
        "Öznitelik": top["ad"],
        "Bu hücre": [float(features.get(c, np.nan)) for c in top["oznitelik"]],
        **{f"Medyan {c}": top[f"medyan_{c}"].astype(float) for c in CLASS_NAMES},
    })


def show_cells(result: dict) -> None:
    """
    Hücre kırpımlarını numara ve tahminle ızgarada gösterir; seçilen hücrenin öznitelik ve olasılıklarını verir.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    cells, crops = result["hucreler"], result["kirpimlar"]
    if cells.empty:
        st.warning("Bu görüntüde hücre çekirdeği bulunamadı; gösterilecek hücre yok.", icon=":material/warning:")
        return
    for start in range(0, len(cells), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (_, c) in zip(cols, cells.iloc[start:start + GRID_COLUMNS].iterrows()):
            with col:
                st.image(to_display(crops[int(c["hucre_no"]) - 1], 160), width="stretch")
                color = CLASS_COLORS[c["sinif"]]
                st.markdown(f"**#{c['hucre_no']}** <span style='color:{color};font-weight:600'>{c['sinif']}</span> "
                            f"<span style='color:#52514e;font-size:0.85rem'>{fmt_pct(c[f'p_{c.sinif}'])}"
                            f"{' · kenarda' if c['kenara_degiyor'] else ''}</span>", unsafe_allow_html=True)

    st.divider()
    no = st.selectbox("Hücre seç", cells["hucre_no"].tolist(),
                      format_func=lambda k: f"Hücre {k} — {cells.loc[cells['hucre_no'] == k, 'sinif'].iloc[0]}")
    row = cells[cells["hucre_no"] == no].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.image(to_display(crops[no - 1], 320), width="stretch")
        st.caption("Renk normalize kırpım (çekirdek ve çevresi). "
                   + ("Hücre görüntü kenarına değiyor; kesik olabilir." if row["kenara_degiyor"] else ""))
        st.markdown("**Sınıf olasılıkları** (kalibre)")
        st.altair_chart(probability_chart(row), width="stretch")
    with right:
        st.markdown("**En ayırt edici 15 öznitelik** (her gruptan 5; sıralama Adım 4'teki ANOVA F)")
        table = feature_table(result["oznitelikler"].iloc[no - 1])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in table.columns if c not in ("Grup", "Öznitelik")})
        st.caption("Medyanlar veri setindeki tüm hücrelerden (eritrosit çapı 10 px referans ölçeği; d = eritrosit "
                   "çapı). Model 58 özniteliğin tamamını kullanır.")


def show_result(result: dict) -> None:
    """
    Sonucu gösterir: sınıf renkli konturlu görüntü, lejant, hücre sayıları, risk skoru ve seviyesi, baskın tip,
    güven durumu.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    risk = result["risk"]
    counts = risk["hucre_sayilari"]
    left, right = st.columns([3, 2])
    with left:
        c1, c2 = st.columns(2)
        c1.image(result["gorsel_orijinal"], caption="Orijinal", width="stretch")
        c2.image(result["gorsel_kontur"], caption="Hücre çekirdekleri (numaralar hücre tablosundaki No)",
                 width="stretch")
        st.markdown(legend_html(), unsafe_allow_html=True)
    with right:
        st.markdown(f"Risk seviyesi &nbsp; {risk_badge(risk['risk_seviyesi'])}", unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Anormal hücre oranı (risk skoru)", fmt_pct(risk["anormal_oran"]),
                  help="(ALL + AML hücre sayısı) / toplam hücre.")
        m2.metric("Yumuşak oran", fmt_pct(risk["anormal_oran_yumusak"]),
                  help="Hücrelerin ortalama P(ALL) + P(AML) değeri (kalibre olasılıklar).")
        k1, k2, k3 = st.columns(3)
        k1.metric("ALL", counts["ALL"])
        k2.metric("AML", counts["AML"])
        k3.metric("Healthy", counts["Healthy"])
        st.markdown(f"**Baskın lösemi tipi:** {dominant_label(risk)}")
        low, high = risk["esikler"]
        st.caption(f"Eşikler: Düşük < {low:.2f} ≤ Orta < {high:.2f} ≤ Yüksek (geliştirme setinden seçildi). "
                   "Baskın tip yalnızca Orta/Yüksek riskte gösterilir.")
        st.markdown(f"**Güven durumu:** {risk['guven']}")
        for reason in risk["guven_nedenleri"]:
            st.markdown(f"- {reason}")
    with st.expander("Hücre tablosu"):
        table = result["hucreler"][["hucre_no", "sinif", "p_ALL", "p_AML", "p_Healthy", "kenara_degiyor"]]
        st.dataframe(table.rename(columns={"hucre_no": "No", "sinif": "Sınıf", "p_ALL": "P(ALL)", "p_AML": "P(AML)",
                                           "p_Healthy": "P(Healthy)", "kenara_degiyor": "Kenara değiyor"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("P(ALL)", "P(AML)", "P(Healthy)")})


def show_status(result: dict) -> None:
    """
    Hücre bulunamayan veya düşük güvenli sonuçlar için sonuç bölümlerinin üstünde açık bir uyarı gösterir.

    Parametreler:
        result: run_pipeline çıktısı.
    """
    risk = result["risk"]
    if risk["risk_seviyesi"] == RISK_NO_CELL_LEVEL:
        st.error("Bu görüntüde hiç hücre çekirdeği bulunamadı; risk **değerlendirilemedi**. Görüntü bir kan "
                 "yayması değilse, çok soluksa veya lökosit içermiyorsa bu beklenen bir sonuçtur.",
                 icon=":material/error:")
    elif risk["guven"] == "Düşük":
        st.warning("**Düşük güvenli sonuç:** " + " ".join(risk["guven_nedenleri"]), icon=":material/warning:")


def show_pdf_button(result: dict, key: str) -> None:
    """
    PDF rapor indirme butonunu, yanında klinik kullanım uyarısıyla gösterir.

    Parametreler:
        result: run_pipeline çıktısı.
        key: Görüntü özeti (PDF önbellek anahtarı).
    """
    left, right = st.columns([1, 3], vertical_alignment="center")
    with left:
        stem = Path(result["goruntu_adi"]).stem.replace(" ", "_")
        st.download_button("PDF raporu indir", data=pdf_bytes(key, result), file_name=f"{stem}_rapor.pdf",
                           mime="application/pdf", type="primary", icon=":material/download:", width="stretch")
    with right:
        st.warning(DISCLAIMER, icon=":material/warning:")


def page() -> None:
    """Analiz sayfasını çizer."""
    st.title("Lösemi Tarama — Analiz")
    disclaimer()
    try:
        load_model_cached()
    except Exception as exc:  # noqa: BLE001 — model dosyası yoksa sayfa anlaşılır mesajla durur
        st.error(f"Sınıflandırıcı yüklenemedi: {exc}. `python -m src.classifier` ile modeli üretin.")
        return

    choose_input()
    current = st.session_state.get(INPUT_KEY)
    if current is None:
        st.info("Bir hazır örnek seçin veya görüntü yükleyin, ardından **Analiz et**'e basın.",
                icon=":material/info:")
        return

    st.subheader(f"Sonuçlar: {current['ad']}")
    try:
        result = analyze_with_progress(current["veri"], current["ad"])
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    except Exception as exc:  # noqa: BLE001 — beklenmeyen hata uygulamayı çökertmesin
        st.error(f"Analiz sırasında beklenmeyen bir hata oluştu: {type(exc).__name__}: {exc}",
                 icon=":material/error:")
        return

    show_status(result)
    with st.expander("a) Ön işleme", icon=":material/tune:"):
        show_preprocessing(result)
    with st.expander("b) Segmentasyon", icon=":material/blur_on:"):
        show_segmentation(result)
    with st.expander("c) Hücreler", icon=":material/grid_view:"):
        show_cells(result)
    with st.expander("d) Sonuç", expanded=True, icon=":material/assignment:"):
        show_result(result)
    show_pdf_button(result, image_key(current["veri"]))
