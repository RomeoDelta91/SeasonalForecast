"""
Plot- en geodata-helpers voor de Suriname Seizoens-Outlook GUI.

De forecast zelf komt uit `forecast_engine.run_forecast(...)` (of uit een
geüpload `suriname_forecast_v3_spatial.nc`). Dit bestand levert:
  * districtgeometrie (voor het clippen van de plots),
  * de weergavespecificatie per forecast-variabele,
  * de contourf-plotfunctie (n niveaus, standaard 6) met district-clip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# Paden
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent

# Neerslagdata: vaste naam zodat een nieuwe maand alleen een bestandsupdate
# vergt, geen code-aanpassing. De reeks mag op elke willekeurige maand eindigen;
# de forecast start automatisch op de eerstvolgende maand.
RAINFALL_NC = BASE / "data.nc"
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
# Districten
# --------------------------------------------------------------------------- #
def load_districts() -> gpd.GeoDataFrame:
    """Districten inlezen en herprojecteren naar WGS84 (EPSG:4326)."""
    gdf = gpd.read_file(SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.sort_values(DISTRICT_NAME_FIELD).reset_index(drop=True)
    return gdf


# --------------------------------------------------------------------------- #
# SST-release (nodig voor de forecast-motor: Atlantische SST-indices)
# --------------------------------------------------------------------------- #
def sst_available() -> bool:
    return SST_NC.exists()


def download_sst(progress_cb=None) -> Path:
    """SST-releasebestand downloaden naar .sst_cache/ als het nog niet bestaat."""
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


def rainfall_last_month() -> "np.datetime64":
    """Laatste datamaand in de neerslagreeks (voor het bepalen van FORECAST_START)."""
    ds = xr.open_dataset(RAINFALL_NC)
    t = ds["time"].values[-1]
    ds.close()
    return t


# --------------------------------------------------------------------------- #
# Weergavespecificatie per forecast-variabele
#   mode bepaalt kleurschaal/niveaus: seq0 (0..max), center0, center100,
#   prob (0..1), center05 (GROCS 0..1 rond 0.5)
# --------------------------------------------------------------------------- #
VARIABLE_SPECS = {
    "precipitation_p50": dict(label="Neerslag — mediaan (P50)", cmap="YlGnBu", mode="seq0", unit="mm/maand"),
    "precipitation_p10": dict(label="Neerslag — 10e percentiel (droog)", cmap="YlGnBu", mode="seq0", unit="mm/maand"),
    "precipitation_p90": dict(label="Neerslag — 90e percentiel (nat)", cmap="YlGnBu", mode="seq0", unit="mm/maand"),
    "precipitation_climatology": dict(label="Klimatologie 1991–2020", cmap="YlGnBu", mode="seq0", unit="mm/maand"),
    "precipitation_anomaly": dict(label="Anomalie (P50 − klimatologie)", cmap="BrBG", mode="center0", unit="mm/maand"),
    "precipitation_pct_normal": dict(label="Percentage van normaal", cmap="BrBG", mode="center100", unit="%"),
    "prob_below": dict(label="Kans op onder-normaal", cmap="Oranges", mode="prob", unit="kans"),
    "prob_normal": dict(label="Kans op rond-normaal", cmap="Purples", mode="prob", unit="kans"),
    "prob_above": dict(label="Kans op boven-normaal", cmap="Greens", mode="prob", unit="kans"),
    # Afgeleide varianten met de legenda in procenten (0–100 %)
    "prob_below_pct": dict(label="% kans op onder-normaal", cmap="Oranges", mode="prob_pct",
                           unit="%", source="prob_below", scale=100.0),
    "prob_normal_pct": dict(label="% kans op rond-normaal", cmap="Purples", mode="prob_pct",
                            unit="%", source="prob_normal", scale=100.0),
    "prob_above_pct": dict(label="% kans op boven-normaal", cmap="Greens", mode="prob_pct",
                           unit="%", source="prob_above", scale=100.0),
    "grocs": dict(label="GROCS-skill (walk-forward)", cmap="RdYlGn", mode="center05", unit="GROCS"),
    "skill_mask": dict(label="Skill-masker (1 = skill)", cmap="Greys", mode="prob", unit="1 = skill"),
}

# volgorde/standaardselectie
DEFAULT_VARS = [
    "precipitation_p50", "precipitation_p10", "precipitation_p90",
    "precipitation_anomaly", "precipitation_pct_normal",
    "prob_below_pct", "prob_normal_pct", "prob_above_pct",
]


# --------------------------------------------------------------------------- #
# Geometrie -> matplotlib clip-pad
# --------------------------------------------------------------------------- #
def geom_to_path(geom) -> MplPath:
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
def compute_levels(data: np.ndarray, n_levels: int, mode: str):
    """n_levels gevulde banden -> n_levels+1 grenzen, afhankelijk van de modus."""
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.linspace(0, 1, n_levels + 1), "neither"
    if mode == "prob" or mode == "center05":
        return np.linspace(0, 1, n_levels + 1), "neither"
    if mode == "prob_pct":
        return np.linspace(0, 100, n_levels + 1), "neither"
    if mode == "center0":
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        return np.linspace(-vmax, vmax, n_levels + 1), "both"
    if mode == "center100":
        half = float(np.nanmax(np.abs(vals - 100))) or 1.0
        return np.linspace(100 - half, 100 + half, n_levels + 1), "both"
    # seq0 en overig: nette lineaire niveaus vanaf (min 0)
    vmin = min(0.0, float(np.nanmin(vals))) if mode == "seq0" else float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    lv = MaxNLocator(nbins=n_levels).tick_values(vmin, vmax)
    if len(lv) < 3:
        lv = np.linspace(vmin, vmax, n_levels + 1)
    return lv, "max"


# --------------------------------------------------------------------------- #
# Forecast-DataArray voorbereiden (y/x -> lat/lon, lat oplopend)
# --------------------------------------------------------------------------- #
def prep_field(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "y" in da.dims:
        rename["y"] = "lat"
    if "x" in da.dims:
        rename["x"] = "lon"
    if rename:
        da = da.rename(rename)
    return da.sortby("lat")


# --------------------------------------------------------------------------- #
# Kaart tekenen (forecast-veld over Suriname, contourf + district-clip)
# --------------------------------------------------------------------------- #
def plot_field(
    ax,
    field: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    district_row,
    title: str,
    cbar_label: str,
    cmap: str,
    mode: str,
    n_levels: int = 6,
    mask_to_district: bool = True,
    margin: float = 0.15,
):
    """
    forecast-veld tekenen met contourf (n_levels banden), bijgesneden op het
    gekozen district (district_row=None -> heel Suriname). Retourneert de
    contourf-set voor de colorbar.
    """
    field = prep_field(field)

    if district_row is not None:
        minx, miny, maxx, maxy = district_row.geometry.bounds
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
    minx, maxx = minx - margin, maxx + margin
    miny, maxy = miny - margin, maxy + margin

    sub = field.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
    if sub["lon"].size < 2 or sub["lat"].size < 2:
        sub = field

    lon = sub["lon"].values
    lat = sub["lat"].values
    data = sub.values
    levels, extend = compute_levels(data, n_levels, mode)

    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend=extend)

    gdf.boundary.plot(ax=ax, color="0.6", linewidth=0.5, zorder=3)
    if district_row is not None:
        sel = gpd.GeoDataFrame(geometry=[district_row.geometry], crs=gdf.crs)
        sel.boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=4)
        if mask_to_district:
            patch = PathPatch(geom_to_path(district_row.geometry),
                              transform=ax.transData, facecolor="none", edgecolor="none")
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
