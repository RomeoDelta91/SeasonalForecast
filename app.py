"""
Suriname Seizoens-Outlook — Streamlit GUI.

Start met:  streamlit run app.py

De app draait de forecast-motor (v3 — hybride statistisch-dynamisch, EOF + ML,
terciel-kansen, GROCS) op de data in deze repo, of laadt een bestaand
`suriname_forecast_v3_spatial.nc`. Daarna kies je welke variabelen je wil
plotten en voor welke voorspelde maand, klik je op de kaart op een district om
de plots op de extent van dat district bij te snijden, pas je de titel aan, en
worden de velden getekend met contourf (standaard 6 niveaus).
"""
from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import streamlit as st
import folium
from streamlit_folium import st_folium

import outlook as ol
import forecast_engine as fe

st.set_page_config(page_title="Suriname Seizoens-Outlook", layout="wide", page_icon="🌧️")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Districten laden…")
def _districts():
    return ol.load_districts()


@st.cache_data(show_spinner=False)
def _cached_forecast(forecast_start, n_months, val_mode, val_step, use_nmme,
                     n_eof, grocs_threshold, consolidate, _progress):
    cfg = fe.ForecastConfig(
        rain_file=str(ol.RAINFALL_NC),
        enso_file=str(ol.NINA_NC),
        sst_file=str(ol.SST_NC),
        forecast_start=forecast_start,
        forecast_n_months=n_months,
        use_nmme=use_nmme,
        n_eof=n_eof,
        val_mode=val_mode,
        val_step=val_step,
        grocs_threshold=grocs_threshold,
        consolidate=consolidate,
    )
    return fe.run_forecast(cfg, progress=_progress)


gdf = _districts()
district_names = gdf[ol.DISTRICT_NAME_FIELD].tolist()

# forecast-start = maand ná de laatste datamaand
_last = pd.Timestamp(ol.rainfall_last_month())
FORECAST_START = (_last + pd.DateOffset(months=1)).strftime("%Y-%m-01")


# --------------------------------------------------------------------------- #
# Kop
# --------------------------------------------------------------------------- #
st.title("🌧️ Suriname Seizoens-Outlook")
st.caption(
    "Hybride statistisch-dynamische neerslagvoorspelling (EOF + ML + optioneel "
    "NMME), terciel-kansen en GROCS-skill — MDS / HydroMet Suriname."
)


# --------------------------------------------------------------------------- #
# Zijbalk — forecast genereren of laden
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.header("① Forecast")

bron = sb.radio("Bron van de forecast", ["Genereren", "Bestand uploaden"], index=0)

if "forecast_ds" not in st.session_state:
    st.session_state["forecast_ds"] = None

if bron == "Genereren":
    sb.caption(f"Uitgiftemoment: **{_last:%B %Y}** → start **{pd.Timestamp(FORECAST_START):%B %Y}**")
    n_months = sb.slider("Aantal voorspelmaanden", 1, 12, 7)
    val_label = sb.selectbox(
        "Rekenmodus (skill/GROCS)",
        ["Snel — geen validatie (~1 min)",
         "Standaard — nationale GROCS (enkele min.)",
         "Volledig — ruimtelijke GROCS (traag)"],
        index=0,
    )
    val_mode = {"Snel": "none", "Standaard": "national", "Volledig": "spatial"}[val_label.split(" —")[0]]
    val_step = sb.select_slider("Validatie-stap (maanden)", [3, 6, 12], value=6,
                                disabled=(val_mode == "none"),
                                help="Kleiner = meer uitgiftemomenten = nauwkeuriger skill, maar trager.")
    with sb.expander("Geavanceerd"):
        n_eof = st.number_input("Aantal EOF's", 2, 8, 4)
        use_nmme = st.checkbox("NMME dynamische predictoren (internet, langzamer)", value=False)
        consolidate = st.checkbox("GBM ⊕ Ridge consolidatie", value=True)
        grocs_threshold = st.slider("GROCS-drempel (masking)", 0.4, 0.7, 0.5, 0.05)

    # SST is nodig voor de Atlantische indices
    if not ol.sst_available():
        sb.warning("SST-data (sst.mnmean.nc) ontbreekt — nodig voor de forecast.")
        if sb.button("⬇️ SST-releasebestand downloaden (~154 MB)"):
            bar = sb.progress(0.0, text="SST downloaden…")
            ol.download_sst(progress_cb=lambda f: bar.progress(f, text=f"SST downloaden… {f*100:.0f}%"))
            bar.empty()
            st.rerun()

    can_run = ol.sst_available()
    if sb.button("▶️ Forecast genereren", type="primary", disabled=not can_run):
        bar = st.progress(0.0, text="Forecast starten…")
        def _pg(frac, msg):
            bar.progress(frac, text=msg)
        try:
            ds = _cached_forecast(FORECAST_START, int(n_months), val_mode, int(val_step),
                                  bool(use_nmme), int(n_eof), float(grocs_threshold),
                                  bool(consolidate), _progress=_pg)
            st.session_state["forecast_ds"] = ds
            bar.empty()
            st.success(f"Forecast klaar: {ds.sizes['time']} maanden.")
        except Exception as e:
            bar.empty()
            st.error(f"Forecast mislukt: {e}")

else:  # uploaden
    up = sb.file_uploader("suriname_forecast_v3_spatial.nc", type=["nc"])
    if up is not None:
        try:
            st.session_state["forecast_ds"] = xr.open_dataset(io.BytesIO(up.read()))
            sb.success("Forecast-bestand geladen.")
        except Exception as e:
            sb.error(f"Kon bestand niet lezen: {e}")

ds = st.session_state["forecast_ds"]

# --------------------------------------------------------------------------- #
# Zijbalk — weergave-instellingen
# --------------------------------------------------------------------------- #
sb.markdown("---")
sb.header("② Weergave")
n_levels = sb.number_input("Aantal niveaus (contourf)", 2, 20, 6, 1)
mask_district = sb.checkbox("Bijsnijden op districtsvorm (clip)", value=True)


# --------------------------------------------------------------------------- #
# Geen forecast? -> uitleg en stop
# --------------------------------------------------------------------------- #
if ds is None:
    st.info(
        "👈 Genereer links een forecast (of upload een bestaand "
        "`suriname_forecast_v3_spatial.nc`). Daarna kies je variabelen en maand, "
        "en klik je op een district om de plots bij te snijden."
    )
    st.stop()

# Beschikbare voorspelmaanden
times = pd.to_datetime(ds["time"].values)
maand_labels = [f"{ol.MAANDEN[t.month - 1]} {t.year}" for t in times]
available_vars = [v for v in ol.VARIABLE_SPECS if v in ds.data_vars]


# --------------------------------------------------------------------------- #
# 1 · District kiezen (klikbare kaart + selectbox)
# --------------------------------------------------------------------------- #
st.subheader("1 · Kies een district")
col_map, col_pick = st.columns([2, 1])

if "sel_district" not in st.session_state:
    st.session_state["sel_district"] = district_names[0]

with col_map:
    uni = gdf.geometry.union_all()
    fmap = folium.Map(location=[uni.centroid.y, uni.centroid.x], zoom_start=7,
                      tiles="CartoDB positron", control_scale=True)

    def _style(feat):
        naam = feat["properties"][ol.DISTRICT_NAME_FIELD]
        selected = naam == st.session_state["sel_district"]
        return {"fillColor": "#e34a33" if selected else "#4292c6",
                "color": "#333333", "weight": 2 if selected else 1,
                "fillOpacity": 0.55 if selected else 0.25}

    folium.GeoJson(
        gdf.__geo_interface__, style_function=_style,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=[ol.DISTRICT_NAME_FIELD], aliases=["District:"]),
    ).add_to(fmap)

    map_state = st_folium(fmap, height=430, width=None, key="distmap",
                          returned_objects=["last_active_drawing"])
    clicked = None
    if map_state and map_state.get("last_active_drawing"):
        clicked = map_state["last_active_drawing"].get("properties", {}).get(ol.DISTRICT_NAME_FIELD)
    if clicked and clicked in district_names and clicked != st.session_state["sel_district"]:
        st.session_state["sel_district"] = clicked
        st.rerun()

with col_pick:
    st.selectbox("District", district_names, key="sel_district")
    st.selectbox("Voorspelde maand", maand_labels, key="sel_month")
    sel_naam = st.session_state["sel_district"]
    st.metric("Gekozen district", sel_naam)
    st.caption(f"Uitgifte: {ds.attrs.get('forecast_issue_month', '?')} · "
               f"Modus: {ds.attrs.get('validation_mode', '?')} · "
               f"NMME: {ds.attrs.get('nmme_models', '?')}")

sel_naam = st.session_state["sel_district"]
district_row = gdf[gdf[ol.DISTRICT_NAME_FIELD] == sel_naam].iloc[0]
month_idx = maand_labels.index(st.session_state.get("sel_month", maand_labels[0]))
month_label = maand_labels[month_idx]


# --------------------------------------------------------------------------- #
# 2 · Variabelen kiezen
# --------------------------------------------------------------------------- #
st.subheader("2 · Kies plots (variabelen)")
default_vars = [v for v in ol.DEFAULT_VARS if v in available_vars]
gekozen = st.multiselect(
    "Welke variabelen wil je plotten?",
    options=available_vars,
    default=default_vars,
    format_func=lambda v: ol.VARIABLE_SPECS[v]["label"],
)


# --------------------------------------------------------------------------- #
# 3 · Plots
# --------------------------------------------------------------------------- #
st.subheader(f"3 · Plots — {month_label} · {sel_naam}")

if not gekozen:
    st.info("Kies hierboven minstens één variabele.")

cols = st.columns(2)
for i, var in enumerate(gekozen):
    spec = ol.VARIABLE_SPECS[var]
    with cols[i % 2]:
        default_title = f"{spec['label']} — {month_label} — {sel_naam}"
        title = st.text_input("Titel", value=default_title, key=f"title_{var}")
        field = ds[var].isel(time=month_idx)
        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ol.plot_field(
            ax, field, gdf, district_row, title=title, cbar_label=spec["unit"],
            cmap=spec["cmap"], mode=spec["mode"], n_levels=int(n_levels),
            mask_to_district=mask_district,
        )
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(spec["unit"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Download van de gegenereerde forecast
# --------------------------------------------------------------------------- #
st.markdown("---")
with st.expander("💾 Forecast-NetCDF downloaden"):
    try:
        buf = ds.to_netcdf()
        st.download_button("Download suriname_forecast_v3_spatial.nc", data=buf,
                           file_name="suriname_forecast_v3_spatial.nc",
                           mime="application/x-netcdf")
    except Exception as e:
        st.caption(f"Download niet beschikbaar: {e}")

st.caption(
    "Motor: GradientBoosting-kwantielen ⊕ Ridge (consolidatie), RandomForest per "
    "EOF-PC, terciel-kansen met GROCS-masking. Predictoren: CHIRPS, Niño3.4, "
    "ERSSTv5 TNA/TSA, optioneel NMME. Plots: contourf, geclipt op districtsextent."
)
