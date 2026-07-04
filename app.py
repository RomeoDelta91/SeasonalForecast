"""
Suriname Seizoens-Outlook — Streamlit app.

Start met:  streamlit run app.py

Kies in de zijbalk welke plots je wil maken en voor welke maand. Klik op de kaart
op een district (of kies het uit de lijst) om de plots op de extent van dat district
bij te snijden. Titels zijn per plot aan te passen. Alle velden worden getekend met
contourf en (standaard) 6 niveaus.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium

import outlook as ol

st.set_page_config(page_title="Suriname Seizoens-Outlook", layout="wide",
                   page_icon="🌧️")


# --------------------------------------------------------------------------- #
# Data laden (gecached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Districten laden…")
def _districts():
    return ol.load_districts()


@st.cache_resource(show_spinner="Neerslagdata laden…")
def _rainfall():
    return ol.load_rainfall()


@st.cache_resource(show_spinner="Nino3.4 laden…")
def _nina():
    return ol.load_nina()


@st.cache_resource(show_spinner="SST laden…")
def _sst():
    return ol.load_sst()


gdf = _districts()
precip = _rainfall()
nina = _nina()

district_names = gdf[ol.DISTRICT_NAME_FIELD].tolist()


# --------------------------------------------------------------------------- #
# Kop
# --------------------------------------------------------------------------- #
st.title("🌧️ Suriname Seizoens-Outlook")
st.caption(
    "Maandelijkse neerslag (1982–2026), ENSO-composieten op basis van Niño3.4 "
    "en NOAA ERSST v5 zeewatertemperatuur."
)


# --------------------------------------------------------------------------- #
# Zijbalk — keuzes
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.header("⚙️ Instellingen")

PLOT_OPTIES = [
    "Neerslag klimatologie",
    "Neerslag anomalie (jaar)",
    "ENSO-composiet",
    "ENSO-anomalie (outlook)",
    "SST-anomalie",
]
gekozen_plots = sb.multiselect("Welke plots wil je maken?", PLOT_OPTIES,
                               default=["Neerslag klimatologie"])

maand_naam = sb.selectbox("Maand", ol.MAANDEN, index=5)  # standaard Juni
maand = ol.MAANDEN.index(maand_naam) + 1

# alleen jaren tonen die deze maand daadwerkelijk in de neerslagreeks hebben
jaren_maand = sorted(set(
    precip.sel(time=precip["time"].dt.month == maand)["time"].dt.year.values.tolist()
))
jaar = sb.selectbox("Jaar (voor anomalie/SST)", jaren_maand[::-1], index=0)

fase = sb.selectbox("ENSO-fase (voor composiet/outlook)", ol.ENSO_FASEN)
thr = sb.slider("Niño3.4 drempel (°C)", 0.3, 1.0, 0.5, 0.1)

sb.markdown("---")
sb.subheader("Weergave")
n_levels = sb.number_input("Aantal niveaus (contourf)", min_value=2, max_value=20,
                           value=6, step=1)
mask_district = sb.checkbox("Bijsnijden op districtsvorm (clip)", value=True)

cmap_seq = sb.selectbox("Kleurenschaal neerslag", ["YlGnBu", "Blues", "GnBu", "viridis"])
cmap_div = sb.selectbox("Kleurenschaal anomalie", ["BrBG", "RdBu", "PuOr", "coolwarm_r"])
cmap_sst = sb.selectbox("Kleurenschaal SST", ["RdBu_r", "coolwarm", "seismic"])

sst_domein = sb.selectbox("SST-domein", list(ol.SST_DOMEINEN.keys()))

# basisperiode voor anomalieën / klimatologie
base_opt = sb.selectbox("Basisperiode (klimatologie)",
                        ["Volledige reeks", "1991–2020"], index=0)
base = None if base_opt == "Volledige reeks" else (1991, 2020)


# --------------------------------------------------------------------------- #
# District kiezen — klikbare kaart + selectbox
# --------------------------------------------------------------------------- #
st.subheader("1 · Kies een district")
col_map, col_pick = st.columns([2, 1])

if "sel_district" not in st.session_state:
    st.session_state["sel_district"] = district_names[0]

with col_map:
    center = [gdf.geometry.union_all().centroid.y, gdf.geometry.union_all().centroid.x]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron",
                      control_scale=True)

    def _style(feat):
        naam = feat["properties"][ol.DISTRICT_NAME_FIELD]
        selected = naam == st.session_state["sel_district"]
        return {
            "fillColor": "#e34a33" if selected else "#4292c6",
            "color": "#333333",
            "weight": 2 if selected else 1,
            "fillOpacity": 0.55 if selected else 0.25,
        }

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=_style,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=[ol.DISTRICT_NAME_FIELD],
                                      aliases=["District:"]),
    ).add_to(fmap)

    map_state = st_folium(fmap, height=430, width=None, key="distmap",
                          returned_objects=["last_active_drawing"])

    clicked = None
    if map_state and map_state.get("last_active_drawing"):
        props = map_state["last_active_drawing"].get("properties", {})
        clicked = props.get(ol.DISTRICT_NAME_FIELD)
    if clicked and clicked in district_names and clicked != st.session_state["sel_district"]:
        st.session_state["sel_district"] = clicked
        st.rerun()

with col_pick:
    st.selectbox("District", district_names, key="sel_district")
    sel_naam = st.session_state["sel_district"]
    district_row = gdf[gdf[ol.DISTRICT_NAME_FIELD] == sel_naam].iloc[0]
    st.metric("Gekozen district", sel_naam)
    st.caption(
        f"Maand: **{maand_naam}**  ·  Niveaus: **{n_levels}**  ·  "
        f"Clip: **{'aan' if mask_district else 'uit'}**"
    )
    # ENSO-info voor deze maand
    phases = ol.enso_phase_for_month(nina, maand, thr)
    if jaar in phases:
        st.caption(f"ENSO-fase {maand_naam} {jaar}: **{phases[jaar]}**")

sel_naam = st.session_state["sel_district"]
district_row = gdf[gdf[ol.DISTRICT_NAME_FIELD] == sel_naam].iloc[0]


# --------------------------------------------------------------------------- #
# Helper voor het tekenen + tonen van één neerslagplot
# --------------------------------------------------------------------------- #
def render_rain_plot(key: str, field, default_title: str, cbar_label: str,
                     cmap: str, diverging: bool, subtitle: str = ""):
    if field is None:
        st.warning(f"Geen data beschikbaar voor deze selectie ({default_title}).")
        return
    title = st.text_input("Titel", value=default_title, key=f"title_{key}")
    if subtitle:
        st.caption(subtitle)
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ol.plot_suriname_field(
        ax, field, gdf, district_row, title=title, cbar_label=cbar_label,
        cmap=cmap, n_levels=int(n_levels), diverging=diverging,
        mask_to_district=mask_district,
    )
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
st.subheader("2 · Plots")

if not gekozen_plots:
    st.info("Kies links minstens één plot-type.")

for plot_naam in gekozen_plots:
    st.markdown(f"### {plot_naam}")

    if plot_naam == "Neerslag klimatologie":
        fld = ol.climatology(precip, maand, base=base)
        per = "1982–2025" if base is None else f"{base[0]}–{base[1]}"
        render_rain_plot(
            "clim", fld,
            default_title=f"Gemiddelde neerslag {maand_naam} – {sel_naam}",
            cbar_label="Neerslag (mm)", cmap=cmap_seq, diverging=False,
            subtitle=f"Klimatologie {maand_naam} ({per}).",
        )

    elif plot_naam == "Neerslag anomalie (jaar)":
        fld = ol.year_anomaly(precip, maand, jaar, base=base)
        render_rain_plot(
            "yanom", fld,
            default_title=f"Neerslaganomalie {maand_naam} {jaar} – {sel_naam}",
            cbar_label="Anomalie (mm)", cmap=cmap_div, diverging=True,
            subtitle=f"{maand_naam} {jaar} minus klimatologie.",
        )

    elif plot_naam == "ENSO-composiet":
        fld, n, years = ol.enso_composite(precip, maand, fase, nina, thr)
        sub = (f"Gemiddelde over {n} {fase}-jaren: "
               f"{', '.join(map(str, years))}" if n else "Geen jaren gevonden.")
        render_rain_plot(
            "ecomp", fld,
            default_title=f"{fase}-composiet neerslag {maand_naam} – {sel_naam}",
            cbar_label="Neerslag (mm)", cmap=cmap_seq, diverging=False,
            subtitle=sub,
        )

    elif plot_naam == "ENSO-anomalie (outlook)":
        fld, n, years = ol.enso_anomaly(precip, maand, fase, nina, thr, base=base)
        sub = (f"{fase}-composiet minus klimatologie · {n} jaren: "
               f"{', '.join(map(str, years))}" if n else "Geen jaren gevonden.")
        render_rain_plot(
            "eanom", fld,
            default_title=f"Outlook {maand_naam} ({fase}) – {sel_naam}",
            cbar_label="Anomalie (mm)", cmap=cmap_div, diverging=True,
            subtitle=sub,
        )

    elif plot_naam == "SST-anomalie":
        if not ol.sst_available():
            st.warning("SST-data (sst.mnmean.nc) is nog niet gedownload.")
            if st.button("⬇️ SST-releasebestand downloaden (~154 MB)", key="dl_sst"):
                bar = st.progress(0.0, text="Downloaden…")
                ol.download_sst(progress_cb=lambda f: bar.progress(f, text=f"Downloaden… {f*100:.0f}%"))
                bar.empty()
                st.cache_resource.clear()
                st.rerun()
        else:
            sst = _sst()
            anom = ol.sst_anomaly(sst, maand, jaar, sst_domein, base=(1991, 2020))
            if anom is None:
                st.warning(f"Geen SST voor {maand_naam} {jaar}.")
            else:
                title = st.text_input(
                    "Titel",
                    value=f"SST-anomalie {maand_naam} {jaar} – {sst_domein}",
                    key="title_sst",
                )
                st.caption("Anomalie t.o.v. basisperiode 1991–2020. Niet op district bijgesneden.")
                fig, ax = plt.subplots(figsize=(9, 5))
                cf = ol.plot_sst_field(ax, anom, title=title, cmap=cmap_sst,
                                       n_levels=int(n_levels))
                cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label("SST-anomalie (°C)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("---")

st.caption(
    "Databronnen: neerslag (repo), Niño3.4 (`nina34.anom.nc`), districten "
    "(`DistriktenSuriname.shp`), SST (NOAA ERSST v5, GitHub-release `sstTemp`)."
)
