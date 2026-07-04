"""
Data-toegang en plot-helpers voor de Suriname Seizoens-Outlook app.

Databronnen (allemaal in deze repo / als release):
  * Suriname_monthly_rainfall_jan_1982-may_2026.nc  -> maandelijkse neerslag (mm) op 0.05deg grid
  * nina34.anom.nc                                   -> Nino3.4 anomalie (degC), maandelijks
  * DistriktenSuriname.shp                           -> 10 districten van Suriname
  * sst.mnmean.nc  (GitHub release 'sstTemp')        -> NOAA ERSST v5 SST (degC), maandelijks, globaal
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# Paden
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent

RAINFALL_NC = BASE / "Suriname_monthly_rainfall_jan_1982-may_2026.nc"
NINA_NC = BASE / "nina34.anom.nc"
SHP = BASE / "DistriktenSuriname.shp"

SST_DIR = BASE / ".sst_cache"
SST_NC = SST_DIR / "sst.mnmean.nc"
SST_URL = "https://github.com/RomeoDelta91/SeasonalForecast/releases/download/sstTemp/sst.mnmean.nc"

DISTRICT_NAME_FIELD = "DISTR_NM"

MAANDEN = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December",
]


# --------------------------------------------------------------------------- #
# Laden van data
# --------------------------------------------------------------------------- #
def load_districts() -> gpd.GeoDataFrame:
    """Districten inlezen en herprojecteren naar WGS84 (EPSG:4326)."""
    gdf = gpd.read_file(SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.sort_values(DISTRICT_NAME_FIELD).reset_index(drop=True)
    return gdf


def load_rainfall() -> xr.DataArray:
    """Neerslag inlezen; coordinaten hernoemen naar lon/lat, lat oplopend sorteren."""
    ds = xr.open_dataset(RAINFALL_NC)
    da = ds["precipitation"]
    rename = {}
    if "x" in da.dims:
        rename["x"] = "lon"
    if "y" in da.dims:
        rename["y"] = "lat"
    if rename:
        da = da.rename(rename)
    da = da.sortby("lat")
    da.name = "precipitation"
    return da


def load_nina() -> pd.Series:
    """Nino3.4 anomalie als pandas Series (index = maand-timestamp)."""
    ds = xr.open_dataset(NINA_NC)
    s = ds["value"].to_series()
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def sst_available() -> bool:
    return SST_NC.exists()


def load_sst() -> xr.DataArray:
    """NOAA ERSST v5 SST inlezen (globaal, lon 0-360, lat oplopend gesorteerd)."""
    ds = xr.open_dataset(SST_NC)
    return ds["sst"].sortby("lat")


def download_sst(progress_cb=None) -> Path:
    """
    SST-releasebestand downloaden naar .sst_cache/ als het nog niet bestaat.

    progress_cb: optionele callback f(fractie 0..1) voor een voortgangsbalk.
    """
    if SST_NC.exists():
        return SST_NC
    import requests

    SST_DIR.mkdir(exist_ok=True)
    tmp = SST_NC.with_suffix(".part")
    with requests.get(SST_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(min(done / total, 1.0))
    tmp.rename(SST_NC)
    return SST_NC


# --------------------------------------------------------------------------- #
# ENSO-classificatie
# --------------------------------------------------------------------------- #
ENSO_FASEN = ["El Nino", "La Nina", "Neutraal"]


def enso_phase_for_month(nina: pd.Series, month: int, threshold: float = 0.5) -> dict[int, str]:
    """
    Voor iedere jaar-waarde van de gekozen maand de ENSO-fase bepalen op basis van
    de Nino3.4 anomalie (>= +thr = El Nino, <= -thr = La Nina, anders Neutraal).

    Geeft {jaar: fase} terug.
    """
    sub = nina[nina.index.month == month]
    out = {}
    for ts, val in sub.items():
        if val >= threshold:
            out[ts.year] = "El Nino"
        elif val <= -threshold:
            out[ts.year] = "La Nina"
        else:
            out[ts.year] = "Neutraal"
    return out


# --------------------------------------------------------------------------- #
# Neerslag-velden afleiden
# --------------------------------------------------------------------------- #
def _month_slice(da: xr.DataArray, month: int) -> xr.DataArray:
    return da.sel(time=da["time"].dt.month == month)


def climatology(da: xr.DataArray, month: int, base=None) -> xr.DataArray:
    """Gemiddelde neerslag voor de gekozen maand (optioneel over een basisperiode)."""
    m = _month_slice(da, month)
    if base is not None:
        y0, y1 = base
        m = m.sel(time=(m["time"].dt.year >= y0) & (m["time"].dt.year <= y1))
    return m.mean("time", keep_attrs=True)


def month_year_field(da: xr.DataArray, month: int, year: int) -> xr.DataArray | None:
    """Neerslag van een specifieke maand+jaar (of None als niet aanwezig)."""
    m = _month_slice(da, month)
    m = m.sel(time=m["time"].dt.year == year)
    if m["time"].size == 0:
        return None
    return m.isel(time=0)


def year_anomaly(da: xr.DataArray, month: int, year: int, base=None) -> xr.DataArray | None:
    fld = month_year_field(da, month, year)
    if fld is None:
        return None
    return fld - climatology(da, month, base=base)


def enso_composite(da: xr.DataArray, month: int, phase: str, nina: pd.Series,
                   threshold: float = 0.5):
    """
    Composiet (gemiddelde) neerslag voor de gekozen maand over alle jaren met de
    gekozen ENSO-fase. Geeft (veld, aantal_jaren, [jaren]) terug.
    """
    phases = enso_phase_for_month(nina, month, threshold)
    years = sorted([y for y, ph in phases.items() if ph == phase])
    m = _month_slice(da, month)
    data_years = set(np.unique(m["time"].dt.year.values).tolist())
    years = [y for y in years if y in data_years]
    if not years:
        return None, 0, []
    sel = m.sel(time=np.isin(m["time"].dt.year.values, years))
    return sel.mean("time", keep_attrs=True), len(years), years


def enso_anomaly(da: xr.DataArray, month: int, phase: str, nina: pd.Series,
                 threshold: float = 0.5, base=None):
    comp, n, years = enso_composite(da, month, phase, nina, threshold)
    if comp is None:
        return None, 0, []
    return comp - climatology(da, month, base=base), n, years


# --------------------------------------------------------------------------- #
# SST-velden
# --------------------------------------------------------------------------- #
SST_DOMEINEN = {
    "Globaal": dict(lon=(0, 360), lat=(-88, 88)),
    "Tropen (30N-30S)": dict(lon=(0, 360), lat=(-30, 30)),
    "Stille Oceaan (ENSO)": dict(lon=(120, 290), lat=(-25, 25)),
    "Tropische Atlantische Oceaan": dict(lon=(280, 360), lat=(-25, 25)),
}


def sst_anomaly(sst: xr.DataArray, month: int, year: int, domein: str,
                base=(1991, 2020)) -> xr.DataArray | None:
    """SST-anomalie t.o.v. basisperiode voor gekozen maand+jaar, bijgesneden op domein."""
    m = sst.sel(time=sst["time"].dt.month == month)
    fld = m.sel(time=m["time"].dt.year == year)
    if fld["time"].size == 0:
        return None
    fld = fld.isel(time=0)
    y0, y1 = base
    clim = m.sel(time=(m["time"].dt.year >= y0) & (m["time"].dt.year <= y1)).mean("time")
    anom = fld - clim
    d = SST_DOMEINEN[domein]
    anom = anom.sel(lat=slice(d["lat"][0], d["lat"][1]))
    anom = anom.sel(lon=slice(d["lon"][0], d["lon"][1]))
    return anom


# --------------------------------------------------------------------------- #
# Geometrie -> matplotlib clip-pad
# --------------------------------------------------------------------------- #
def geom_to_path(geom) -> MplPath:
    """(Multi)Polygon omzetten naar een matplotlib Path (voor clippen)."""
    verts, codes = [], []

    def add_ring(coords):
        pts = np.asarray(coords)
        n = len(pts)
        if n < 3:
            return
        verts.extend(pts.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY])

    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        add_ring(poly.exterior.coords)
        for interior in poly.interiors:
            add_ring(interior.coords)
    return MplPath(verts, codes)


# --------------------------------------------------------------------------- #
# Contour-niveaus
# --------------------------------------------------------------------------- #
def compute_levels(data: np.ndarray, n_levels: int, diverging: bool):
    """
    n_levels gevulde banden -> n_levels+1 grenzen.
    diverging: symmetrisch rond 0.
    """
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.linspace(0, 1, n_levels + 1)
    if diverging:
        vmax = float(np.nanmax(np.abs(vals)))
        if vmax == 0:
            vmax = 1.0
        return np.linspace(-vmax, vmax, n_levels + 1)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    locator = MaxNLocator(nbins=n_levels, prune=None)
    lv = locator.tick_values(vmin, vmax)
    # zorg voor precies n_levels+1 grenzen als het kan, anders nette lineaire fallback
    if len(lv) < 3:
        lv = np.linspace(vmin, vmax, n_levels + 1)
    return lv


# --------------------------------------------------------------------------- #
# Kaart tekenen (neerslagveld over Suriname)
# --------------------------------------------------------------------------- #
def plot_suriname_field(
    ax,
    field: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    district_row,
    title: str,
    cbar_label: str,
    cmap: str,
    n_levels: int = 6,
    diverging: bool = False,
    mask_to_district: bool = True,
    margin: float = 0.15,
):
    """
    Een neerslagveld tekenen met contourf (n_levels banden), bijgesneden op het
    gekozen district. district_row = None -> heel Suriname.
    Geeft de contourf-set terug (voor de colorbar).
    """
    if district_row is not None:
        minx, miny, maxx, maxy = district_row.geometry.bounds
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
    minx, maxx = minx - margin, maxx + margin
    miny, maxy = miny - margin, maxy + margin

    # data bijsnijden op de (gebufferde) bounding box
    sub = field.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
    if sub["lon"].size < 2 or sub["lat"].size < 2:
        # te weinig cellen in de box -> val terug op het volledige veld
        sub = field

    lon = sub["lon"].values
    lat = sub["lat"].values
    data = sub.values
    levels = compute_levels(data, n_levels, diverging)

    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend="both")

    # districtsgrenzen
    gdf.boundary.plot(ax=ax, color="0.6", linewidth=0.5, zorder=3)
    if district_row is not None:
        sel = gpd.GeoDataFrame(geometry=[district_row.geometry], crs=gdf.crs)
        sel.boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=4)

        if mask_to_district:
            clip_path = geom_to_path(district_row.geometry)
            patch = PathPatch(clip_path, transform=ax.transData,
                              facecolor="none", edgecolor="none")
            ax.add_patch(patch)
            try:
                cf.set_clip_path(patch)
            except AttributeError:
                for coll in cf.collections:
                    coll.set_clip_path(patch)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xlabel("Lengtegraad")
    ax.set_ylabel("Breedtegraad")
    ax.set_title(title, fontsize=13, fontweight="bold")
    return cf


def plot_sst_field(ax, anom: xr.DataArray, title: str, cmap: str, n_levels: int = 6):
    """SST-anomalieveld tekenen (diverging, symmetrisch rond 0)."""
    lon = anom["lon"].values
    lat = anom["lat"].values
    data = anom.values
    levels = compute_levels(data, n_levels, diverging=True)
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend="both")
    ax.set_xlabel("Lengtegraad (0-360)")
    ax.set_ylabel("Breedtegraad")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    return cf
