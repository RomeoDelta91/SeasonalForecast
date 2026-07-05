"""
Forecast-motor voor de Suriname Seizoens-Outlook.

Getrouwe port van het notebook `LatestSeasonFcast_1.ipynb` (v3 — hybride
statistisch-dynamisch + terciel-kansen + GROCS) naar een herbruikbare module,
zodat de Streamlit-GUI de forecast kan genereren zonder dat er een notebook
gedraaid hoeft te worden.

Kernstappen (identiek aan het notebook):
  1. Data laden (CHIRPS-neerslag, Niño3.4, ERSSTv5 → TNA/TSA/gradiënt)
  1b. Optioneel NMME dynamische predictoren (IRI Data Library)
  2. Klimatologie 1991–2020 + anomalieën + terciel-grenzen + σ-schaling
  3. EOF/PCA van het anomalieveld
  4. Feature-tabel (alleen info bekend op t0)
  5. Walk-forward validatie → MSESS + GROCS (optioneel)
  6. Definitieve forecast per lead (GBM-kwantielen ⊕ Ridge-consolidatie, RF per PC)
  7. Ruimtelijke velden + terciel-kansen + GROCS-masking
  8. xarray.Dataset met 11 variabelen (dims time, y, x)

`run_forecast(cfg, progress=...)` geeft de xarray.Dataset terug.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Configuratie
# --------------------------------------------------------------------------- #
@dataclass
class ForecastConfig:
    rain_file: str
    enso_file: str
    sst_file: str
    forecast_start: str            # "YYYY-MM-01" — maand ná de laatste datamaand
    forecast_n_months: int = 7
    clim_start: str = "1991-01-01"
    clim_end: str = "2020-12-31"
    n_eof: int = 4
    # Atlantische SST-boxen (0–360°)
    tna_box: dict = field(default_factory=lambda: dict(lat_n=23.5, lat_s=5.5, lon_w=302.5, lon_e=345.0))
    tsa_box: dict = field(default_factory=lambda: dict(lat_n=0.0, lat_s=-20.0, lon_w=330.0, lon_e=10.0))
    # NMME
    use_nmme: bool = False
    nmme_models: tuple = ("NCEP-CFSv2", "GFDL-SPEAR")
    sur_box: dict = field(default_factory=lambda: dict(lat_s=1.5, lat_n=6.5, lon_w=302.0, lon_e=306.5))
    nino34_box: dict = field(default_factory=lambda: dict(lat_s=-5.0, lat_n=5.0, lon_w=190.0, lon_e=240.0))
    nmme_cache: str = ".nmme_cache"
    # Validatie: "none" | "national" | "spatial"
    val_mode: str = "national"
    val_start: str = "2012-01-01"
    val_step: int = 6
    grocs_threshold: float = 0.5
    smooth_sigma: float = 1.5
    consolidate: bool = True


# --------------------------------------------------------------------------- #
# STAP 1 — data & EOF hulpfuncties
# --------------------------------------------------------------------------- #
def load_rainfall(path):
    """CHIRPS: 3D-veld + nationaal gemiddelde. Normaliseert naar (tijd, lat, lon)."""
    ds = xr.open_dataset(path)
    if "y" not in ds.coords:
        ds = ds.rename({"lat": "y", "lon": "x"})
    da = ds["precipitation"].transpose("time", "y", "x")
    rain3d = da.values.astype(np.float32)
    lats = ds["y"].values
    lons = ds["x"].values
    times = pd.to_datetime(ds.time.values).to_period("M").to_timestamp()
    nat = np.nanmean(rain3d, axis=(1, 2))
    nat_df = pd.DataFrame({"rainfall": nat}, index=times)
    return nat_df, rain3d, lats, lons, times


def load_enso(path):
    ds = xr.open_dataset(path)
    df = pd.DataFrame({"enso": ds["value"].values},
                      index=pd.to_datetime(ds.time.values))
    df.index = df.index.to_period("M").to_timestamp()
    return df


def _box_mean_sst(ds, box):
    sst = ds["sst"]
    lat_slice = slice(box["lat_n"], box["lat_s"])   # ERSSTv5: lat aflopend
    if box["lon_w"] <= box["lon_e"]:
        sub = sst.sel(lat=lat_slice, lon=slice(box["lon_w"], box["lon_e"]))
        return sub.mean(dim=["lat", "lon"], skipna=True)
    sub1 = sst.sel(lat=lat_slice, lon=slice(box["lon_w"], 360))
    sub2 = sst.sel(lat=lat_slice, lon=slice(0, box["lon_e"]))
    both = xr.concat([sub1, sub2], dim="lon")
    return both.mean(dim=["lat", "lon"], skipna=True)


def load_atlantic_indices(path, tna_box, tsa_box, clim_start, clim_end):
    ds = xr.open_dataset(path)
    times = pd.to_datetime(ds.time.values)
    out = pd.DataFrame(index=pd.DatetimeIndex(times).to_period("M").to_timestamp())
    for name, box in [("tna", tna_box), ("tsa", tsa_box)]:
        raw = _box_mean_sst(ds, box).values
        s = pd.Series(raw, index=out.index)
        cmask = (out.index >= clim_start) & (out.index <= clim_end)
        clim = s[cmask].groupby(s[cmask].index.month).mean()
        out[name + "_anom"] = s - s.index.month.map(clim)
    out["atl_grad"] = out["tna_anom"] - out["tsa_anom"]
    return out


def monthly_climatology_3d(rain3d, times, clim_start, clim_end):
    mask = (times >= pd.Timestamp(clim_start)) & (times <= pd.Timestamp(clim_end))
    ny, nx = rain3d.shape[1], rain3d.shape[2]
    clim = np.zeros((12, ny, nx), dtype=np.float32)
    for mo in range(1, 13):
        sel = mask & (times.month == mo)
        clim[mo - 1] = np.nanmean(rain3d[sel], axis=0)
    return clim


def compute_anomaly_3d(rain3d, times, clim3d):
    anom = np.empty_like(rain3d)
    for i, t in enumerate(times):
        anom[i] = rain3d[i] - clim3d[t.month - 1]
    return anom


def eof_decompose(anom3d, n_eof):
    nt, ny, nx = anom3d.shape
    flat = anom3d.reshape(nt, ny * nx)
    valid = ~np.isnan(flat).any(axis=0)
    X = flat[:, valid]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    expl = (S ** 2) / np.sum(S ** 2)
    pcs = U[:, :n_eof] * S[:n_eof]
    eofs = np.full((n_eof, ny * nx), np.nan, dtype=np.float32)
    eofs[:, valid] = Vt[:n_eof]
    return pcs, eofs.reshape(n_eof, ny, nx), expl[:n_eof], valid.reshape(ny, nx)


def reconstruct_field(pc_values, eofs):
    return np.tensordot(pc_values, eofs, axes=(0, 0))


# --------------------------------------------------------------------------- #
# STAP 4/6 — features & modellen
# --------------------------------------------------------------------------- #
def build_feature_table(nat_anom, pcs_df, enso_df, atl_df):
    df = pd.DataFrame(index=nat_anom.index)
    for lag in [0, 1, 2, 3, 6, 12]:
        df[f"nat_anom_lag{lag}"] = nat_anom.shift(lag)
    df["nat_ma3"] = nat_anom.rolling(3).mean()
    df["nat_ma6"] = nat_anom.rolling(6).mean()
    df["nat_ma12"] = nat_anom.rolling(12).mean()
    for k in pcs_df.columns:
        for lag in [0, 1, 2]:
            df[f"{k}_lag{lag}"] = pcs_df[k].shift(lag)
    for lag in [0, 1, 2, 3]:
        df[f"enso_lag{lag}"] = enso_df["enso"].reindex(df.index).shift(lag)
    for col in ["tna_anom", "tsa_anom", "atl_grad"]:
        for lag in [0, 1, 2]:
            df[f"{col}_lag{lag}"] = atl_df[col].reindex(df.index).shift(lag)
    yr = df.index.year
    df["year_norm"] = (yr - yr.min()) / (yr.max() - yr.min() + 1e-6)
    return df


def make_direct_dataset(feat_df, target_series, lead):
    y = target_series.shift(-lead)
    target_month = pd.Series(feat_df.index.month, index=feat_df.index)
    tm = ((target_month - 1 + lead) % 12) + 1
    X = feat_df.copy()
    X["sin_tmonth"] = np.sin(2 * np.pi * tm / 12)
    X["cos_tmonth"] = np.cos(2 * np.pi * tm / 12)
    data = X.join(y.rename("target")).dropna()
    return data.drop(columns="target"), data["target"]


def fit_quantile_models(X, y, quantiles=(0.1, 0.5, 0.9), n_est=400):
    models = {}
    for q in quantiles:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=n_est, max_depth=3,
            learning_rate=0.04, subsample=0.8,
            min_samples_leaf=8, random_state=42)
        m.fit(X, y)
        models[q] = m
    return models


def fit_pc_model(X, y, n_est=400):
    m = RandomForestRegressor(
        n_estimators=n_est, max_depth=8, min_samples_leaf=4,
        max_features=0.5, random_state=42, n_jobs=-1)
    m.fit(X, y)
    return m


def pinball_loss(y_true, y_pred, q):
    d = y_true - y_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))


# --------------------------------------------------------------------------- #
# STAP 1b — NMME (IRI Data Library) — optioneel, faalt zacht
# --------------------------------------------------------------------------- #
import urllib.request

IRI_BASE = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"


def _iri_box_url(model, dataset, var, box):
    return (f"{IRI_BASE}/.{model}/.{dataset}/.MONTHLY/.{var}/"
            f"X/{box['lon_w']}/{box['lon_e']}/RANGEEDGES/"
            f"Y/{box['lat_s']}/{box['lat_n']}/RANGEEDGES/"
            f"%5BX/Y%5Daverage/%5BM%5Daverage/data.nc")


def _fetch_iri(url, cache_path, log=print):
    if os.path.exists(cache_path):
        return xr.open_dataset(cache_path, decode_times=False)
    try:
        log(f"    download: {os.path.basename(cache_path)} ...")
        urllib.request.urlretrieve(url, cache_path)
        return xr.open_dataset(cache_path, decode_times=False)
    except Exception as e:
        log(f"    MISLUKT ({e})")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None


def _decode_iri_time(ds):
    s_vals = ds["S"].values
    origin = pd.Timestamp("1960-01-01")
    return pd.DatetimeIndex([origin + pd.DateOffset(months=int(round(v))) for v in s_vals])


def load_nmme_index(model, var, box, cache_dir, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for dataset in ("HINDCAST", "FORECAST"):
        tag = f"{model}_{dataset}_{var}_{box['lon_w']}_{box['lat_s']}".replace(".", "p")
        ds = _fetch_iri(_iri_box_url(model, dataset, var, box),
                        os.path.join(cache_dir, tag + ".nc"), log=log)
        if ds is None:
            continue
        vname = [v for v in ds.data_vars][0]
        da = ds[vname].squeeze(drop=True)
        times = _decode_iri_time(ds)
        df = pd.DataFrame(da.values, index=times, columns=np.round(ds["L"].values, 1))
        frames.append(df)
        ds.close()
    if not frames:
        return None
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def build_nmme_feature_table(models, max_lead, cache_dir, clim_start, clim_end,
                             sur_box, nino34_box, log=print):
    spec = {"prec": sur_box, "sst": nino34_box}
    per_var = {}
    for var, box in spec.items():
        model_dfs = []
        for m in models:
            log(f"  NMME {m} / {var}:")
            df = load_nmme_index(m, var, box, cache_dir, log=log)
            if df is not None and len(df) > 60:
                model_dfs.append(df)
            else:
                log(f"    {m}/{var} overgeslagen (geen of te weinig data)")
        if not model_dfs:
            per_var[var] = None
            continue
        common_L = sorted(set.intersection(*[set(d.columns) for d in model_dfs]))
        mmm = pd.concat([d[common_L] for d in model_dfs]).groupby(level=0).mean()
        mmm = mmm.dropna(how="all")
        cmask = (mmm.index >= clim_start) & (mmm.index <= clim_end)
        anom = mmm.copy()
        for L in common_L:
            clim_ml = mmm.loc[cmask, L].groupby(mmm.index[cmask].month).mean()
            anom[L] = mmm[L] - mmm.index.month.map(clim_ml)
        per_var[var] = anom
    name_map = {"prec": "nmme_prec", "sst": "nmme_nino34"}
    tables = []
    for var, anom in per_var.items():
        if anom is None:
            continue
        t0_index = anom.index - pd.DateOffset(months=1)
        cols = {}
        for h in range(1, max_lead + 1):
            L = h - 0.5
            if L in anom.columns:
                cols[f"{name_map[var]}_lead{h}"] = anom[L].values
        tables.append(pd.DataFrame(cols, index=t0_index))
    if not tables:
        return None
    nmme_df = pd.concat(tables, axis=1).sort_index()
    return nmme_df[~nmme_df.index.duplicated(keep="last")]


def add_nmme_to_lead(X, nmme_df, lead):
    if nmme_df is None:
        return X
    X = X.copy()
    for base in ("nmme_prec", "nmme_nino34"):
        col = f"{base}_lead{lead}"
        if col in nmme_df.columns:
            X[base] = nmme_df[col].reindex(X.index)
    return X


# --------------------------------------------------------------------------- #
# STAP 2b/6c — tercielen, kansen, GROCS
# --------------------------------------------------------------------------- #
def tercile_bounds_3d(rain3d, times, clim_start, clim_end):
    mask = (times >= pd.Timestamp(clim_start)) & (times <= pd.Timestamp(clim_end))
    ny, nx = rain3d.shape[1], rain3d.shape[2]
    tb = np.full((12, 2, ny, nx), np.nan, dtype=np.float32)
    for mo in range(1, 13):
        sel = mask & (times.month == mo)
        tb[mo - 1, 0] = np.nanpercentile(rain3d[sel], 100 / 3, axis=0)
        tb[mo - 1, 1] = np.nanpercentile(rain3d[sel], 200 / 3, axis=0)
    return tb


def cell_sigma_ratio(anom3d, times, clim_start, clim_end):
    mask = (times >= pd.Timestamp(clim_start)) & (times <= pd.Timestamp(clim_end))
    cell_std = np.nanstd(anom3d[mask], axis=0)
    nat_std = np.nanstd(np.nanmean(anom3d[mask], axis=(1, 2)))
    ratio = cell_std / max(nat_std, 1e-6)
    return np.clip(ratio, 0.3, 6.0)


def tercile_probs_field(anom_field, clim_mo, tb_mo, sigma_nat, sigma_ratio):
    sigma = np.maximum(sigma_nat * sigma_ratio, 1e-3)
    t1_anom = tb_mo[0] - clim_mo
    t2_anom = tb_mo[1] - clim_mo
    p_below = norm.cdf((t1_anom - anom_field) / sigma)
    p_above = 1.0 - norm.cdf((t2_anom - anom_field) / sigma)
    p_norm = np.clip(1.0 - p_below - p_above, 0.0, 1.0)
    tot = p_below + p_norm + p_above
    return p_below / tot, p_norm / tot, p_above / tot


def observed_tercile_field(rain_mo, tb_mo):
    cat = np.where(rain_mo <= tb_mo[0], 0, np.where(rain_mo > tb_mo[1], 2, 1)).astype(np.float32)
    cat = np.where(np.isnan(rain_mo) | np.isnan(tb_mo[0]), np.nan, cat)
    return cat


def smooth_nan(field, sigma):
    v = np.nan_to_num(field, nan=0.0)
    w = np.where(np.isnan(field), 0.0, 1.0)
    vs = gaussian_filter(v, sigma)
    ws = gaussian_filter(w, sigma)
    out = np.where(ws > 1e-6, vs / ws, np.nan)
    return np.where(np.isnan(field), np.nan, out)


def _auc_vectorized(scores, labels):
    valid = ~np.isnan(labels) & ~np.isnan(scores)
    s = np.where(valid, scores, np.nan)
    ranks = rankdata(np.where(valid, s, np.inf), axis=0).astype(np.float64)
    ranks = np.where(valid, ranks, np.nan)
    pos = np.where(valid, labels, 0.0)
    npos = np.nansum(pos, axis=0)
    nvalid = valid.sum(axis=0)
    nneg = nvalid - npos
    sum_r = np.nansum(ranks * pos, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_r - npos * (npos + 1) / 2.0) / (npos * nneg)
    auc[(npos < 2) | (nneg < 2)] = np.nan
    return auc


def grocs_maps(prob_stack, obs_cat_stack):
    n, _, ny, nx = prob_stack.shape
    p_below = prob_stack[:, 0].reshape(n, -1)
    p_above = prob_stack[:, 2].reshape(n, -1)
    obs = obs_cat_stack.reshape(n, -1)
    auc_b = _auc_vectorized(p_below, np.where(np.isnan(obs), np.nan, (obs == 0).astype(float)))
    auc_a = _auc_vectorized(p_above, np.where(np.isnan(obs), np.nan, (obs == 2).astype(float)))
    grocs = np.nanmean(np.vstack([auc_b, auc_a]), axis=0)
    return grocs.reshape(ny, nx)


def grocs_series(probs, obs_cat):
    g = grocs_maps(probs[:, :, None, None], obs_cat[:, None, None])
    return float(g[0, 0])


# --------------------------------------------------------------------------- #
# Orchestratie
# --------------------------------------------------------------------------- #
def run_forecast(cfg: ForecastConfig, progress=None, log=None) -> xr.Dataset:
    """Draai de volledige forecast en geef de xarray.Dataset (11 variabelen) terug."""
    def _p(frac, msg):
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)
    def _log(msg):
        if log:
            log(msg)

    spatial_val = cfg.val_mode == "spatial"
    run_val = cfg.val_mode in ("national", "spatial")

    # ---- STAP 1: data laden ----
    _p(0.02, "Data laden…")
    nat_df, rain3d, lats, lons, rain_times = load_rainfall(cfg.rain_file)
    enso_df = load_enso(cfg.enso_file)
    atl_df = load_atlantic_indices(cfg.sst_file, cfg.tna_box, cfg.tsa_box,
                                   cfg.clim_start, cfg.clim_end)
    n_lat, n_lon = len(lats), len(lons)

    last = rain_times[-1]
    fs = pd.Timestamp(cfg.forecast_start)
    if fs != last + pd.DateOffset(months=1):
        raise ValueError(
            f"FORECAST_START ({fs:%Y-%m}) moet de maand ná de laatste "
            f"datamaand ({last:%Y-%m}) zijn.")

    # ---- STAP 1b: NMME ----
    nmme_df = None
    if cfg.use_nmme:
        _p(0.06, "NMME-predictoren downloaden…")
        try:
            nmme_df = build_nmme_feature_table(
                list(cfg.nmme_models), cfg.forecast_n_months, cfg.nmme_cache,
                cfg.clim_start, cfg.clim_end, cfg.sur_box, cfg.nino34_box, log=_log)
        except Exception as e:
            _log(f"NMME-opbouw mislukt: {e}")
            nmme_df = None
        if nmme_df is not None:
            dekking = nmme_df.reindex([last]).notna().sum(axis=1).iloc[0]
            if dekking == 0:
                _log("Geen NMME-forecast voor huidig uitgiftemoment → v2-features.")
                nmme_df = None

    # ---- STAP 2: klimatologie & anomalieën ----
    _p(0.12, "Klimatologie & anomalieën…")
    sp_clim = monthly_climatology_3d(rain3d, rain_times, cfg.clim_start, cfg.clim_end)
    anom3d = compute_anomaly_3d(rain3d, rain_times, sp_clim)
    nat_clim_by_month = pd.Series(
        [np.nanmean(sp_clim[m]) for m in range(12)], index=range(1, 13))
    nat_anom = nat_df["rainfall"] - nat_df.index.month.map(nat_clim_by_month)

    # ---- STAP 2b: tercielen & sigma ----
    terc_bounds = tercile_bounds_3d(rain3d, rain_times, cfg.clim_start, cfg.clim_end)
    sigma_ratio = cell_sigma_ratio(anom3d, rain_times, cfg.clim_start, cfg.clim_end)
    nat_terc = {}
    cmask = (nat_anom.index >= cfg.clim_start) & (nat_anom.index <= cfg.clim_end)
    for mo in range(1, 13):
        v = nat_anom[cmask & (nat_anom.index.month == mo)]
        nat_terc[mo] = (np.percentile(v, 100 / 3), np.percentile(v, 200 / 3))

    # ---- STAP 3: EOF ----
    _p(0.18, "EOF-decompositie…")
    pcs, eofs, expl_var, valid_mask = eof_decompose(anom3d, cfg.n_eof)
    pcs_df = pd.DataFrame(pcs, index=rain_times,
                          columns=[f"pc{k}" for k in range(cfg.n_eof)])

    # ---- STAP 4: features ----
    feat_df = build_feature_table(nat_anom, pcs_df, enso_df, atl_df)

    leads = range(1, cfg.forecast_n_months + 1)
    time_index = {t: i for i, t in enumerate(rain_times)}

    # ---- STAP 5: walk-forward validatie ----
    skill_rows = []
    grocs_lead = np.full((cfg.forecast_n_months, n_lat, n_lon), np.nan, dtype=np.float32)
    if run_val:
        origins = pd.date_range(
            cfg.val_start,
            rain_times[-1] - pd.DateOffset(months=cfg.forecast_n_months),
            freq=f"{cfg.val_step}MS")
        records = []
        sp_probs = {l: [] for l in leads}
        sp_obscat = {l: [] for l in leads}
        n_iter = max(len(list(leads)) * len(origins), 1)
        step = 0
        for lead in leads:
            feat_lead = add_nmme_to_lead(feat_df, nmme_df, lead)
            X_lead, y_lead = make_direct_dataset(feat_lead, nat_anom, lead)
            pc_data = ([make_direct_dataset(feat_lead, pcs_df[f"pc{k}"], lead)
                        for k in range(cfg.n_eof)] if spatial_val else None)
            for t0 in origins:
                step += 1
                _p(0.20 + 0.45 * step / n_iter,
                   f"Walk-forward validatie… lead {lead}, {t0:%Y-%m}")
                tr = X_lead.index <= t0
                if tr.sum() < 120 or t0 not in X_lead.index:
                    continue
                X_tr, y_tr = X_lead[tr], y_lead[tr]
                X_te = X_lead.loc[[t0]]
                t_target = t0 + pd.DateOffset(months=lead)
                if t_target not in nat_anom.index:
                    continue
                y_true_anom = nat_anom.loc[t_target]
                clim_val = nat_clim_by_month[t_target.month]

                qm = fit_quantile_models(X_tr, y_tr, n_est=250)
                p10 = qm[0.1].predict(X_te)[0]
                p50 = qm[0.5].predict(X_te)[0]
                p90 = qm[0.9].predict(X_te)[0]
                p10, p50, p90 = np.sort([p10, p50, p90])

                scaler = StandardScaler().fit(X_tr)
                ridge = Ridge(alpha=10.0).fit(scaler.transform(X_tr), y_tr)
                p_ridge = ridge.predict(scaler.transform(X_te))[0]

                sigma_nat = max((p90 - p10) / 2.5631, 1e-3)
                t1, t2 = nat_terc[t_target.month]
                pb = norm.cdf((t1 - p50) / sigma_nat)
                pa = 1.0 - norm.cdf((t2 - p50) / sigma_nat)
                pn = max(1.0 - pb - pa, 0.0)
                obs_cat_nat = 0 if y_true_anom <= t1 else (2 if y_true_anom > t2 else 1)

                records.append(dict(
                    origin=t0, lead=lead, target=t_target, month=t_target.month,
                    y_anom=y_true_anom, p50=p50, p10=p10, p90=p90, ridge=p_ridge,
                    prob_b=pb, prob_n=pn, prob_a=pa, obs_cat=obs_cat_nat,
                    y_mm=y_true_anom + clim_val, p50_mm=p50 + clim_val))

                if spatial_val and t_target in time_index:
                    pc_hat = np.zeros(cfg.n_eof)
                    for k in range(cfg.n_eof):
                        Xk, yk = pc_data[k]
                        trk = Xk.index <= t0
                        if t0 not in Xk.index or trk.sum() < 120:
                            pc_hat[:] = np.nan
                            break
                        m = RandomForestRegressor(
                            n_estimators=150, max_depth=8, min_samples_leaf=4,
                            max_features=0.5, random_state=42, n_jobs=-1)
                        m.fit(Xk[trk], yk[trk])
                        pc_hat[k] = m.predict(Xk.loc[[t0]])[0]
                    if not np.any(np.isnan(pc_hat)):
                        anom_rec = reconstruct_field(pc_hat, eofs)
                        anom_adj = anom_rec + (p50 - np.nanmean(anom_rec))
                        anom_s = smooth_nan(anom_adj, cfg.smooth_sigma)
                        mo = t_target.month
                        pB, pN, pA = tercile_probs_field(
                            anom_s, sp_clim[mo - 1], terc_bounds[mo - 1], sigma_nat, sigma_ratio)
                        obs_s = smooth_nan(rain3d[time_index[t_target]], cfg.smooth_sigma)
                        obs_cat = observed_tercile_field(obs_s, terc_bounds[mo - 1])
                        sp_probs[lead].append(np.stack([pB, pN, pA]).astype(np.float32))
                        sp_obscat[lead].append(obs_cat.astype(np.float32))

        val = pd.DataFrame(records)
        for lead in leads:
            v = val[val.lead == lead]
            if len(v) == 0:
                skill_rows.append(dict(lead=lead, msess=0.0, msess_ridge=0.0, grocs=np.nan))
                continue
            mse_m = mean_squared_error(v.y_anom, v.p50)
            mse_r = mean_squared_error(v.y_anom, v.ridge)
            mse_c = np.mean(v.y_anom ** 2)
            msess = 1 - mse_m / mse_c
            msess_r = 1 - mse_r / mse_c
            g_nat = grocs_series(v[["prob_b", "prob_n", "prob_a"]].values.astype(float),
                                 v.obs_cat.values.astype(float))
            skill_rows.append(dict(lead=lead, msess=msess, msess_ridge=msess_r, grocs=g_nat))
        if spatial_val:
            for l_i, lead in enumerate(leads):
                if len(sp_probs[lead]) >= 8:
                    g = grocs_maps(np.stack(sp_probs[lead]), np.stack(sp_obscat[lead]))
                    grocs_lead[l_i] = smooth_nan(g.astype(np.float32), cfg.smooth_sigma)
            grocs_lead[:, ~valid_mask] = np.nan
    else:
        for lead in leads:
            skill_rows.append(dict(lead=lead, msess=0.0, msess_ridge=0.0, grocs=np.nan))

    # ---- consolidatie-gewichten ----
    cons_w = {}
    for r in skill_rows:
        wg = max(r["msess"], 0.0) ** 2
        wr = max(r["msess_ridge"], 0.0) ** 2
        if not cfg.consolidate or not run_val:
            wg, wr = 1.0, 0.0
        cons_w[r["lead"]] = (wg, wr)

    def consolidate_p50(lead, p50_gbm, p50_ridge):
        wg, wr = cons_w.get(lead, (1.0, 0.0))
        tot = wg + wr
        if tot == 0:
            return 0.0
        return (wg * p50_gbm + wr * p50_ridge) / tot

    # ---- STAP 6: definitieve forecast ----
    _p(0.68, "Definitieve modellen trainen…")
    t0 = rain_times[-1]
    future_months = pd.date_range(cfg.forecast_start, periods=cfg.forecast_n_months, freq="MS")
    n_fut = len(future_months)
    nat_p10, nat_p50, nat_p90 = np.zeros(n_fut), np.zeros(n_fut), np.zeros(n_fut)
    pc_pred = np.zeros((n_fut, cfg.n_eof))

    for i, lead in enumerate(range(1, cfg.forecast_n_months + 1)):
        _p(0.68 + 0.22 * (i + 1) / n_fut, f"Forecast lead {lead}…")
        feat_lead = add_nmme_to_lead(feat_df, nmme_df, lead)
        X_lead, y_lead = make_direct_dataset(feat_lead, nat_anom, lead)
        qm = fit_quantile_models(X_lead, y_lead)

        X_t0 = feat_lead.loc[[t0]].copy()
        tm = ((t0.month - 1 + lead) % 12) + 1
        X_t0["sin_tmonth"] = np.sin(2 * np.pi * tm / 12)
        X_t0["cos_tmonth"] = np.cos(2 * np.pi * tm / 12)
        X_t0 = X_t0[X_lead.columns].dropna()
        if len(X_t0) != 1:
            raise ValueError(f"Ontbrekende features op t0 voor lead {lead}")

        p10 = qm[0.1].predict(X_t0)[0]
        p50 = qm[0.5].predict(X_t0)[0]
        p90 = qm[0.9].predict(X_t0)[0]
        p10, p50, p90 = np.sort([p10, p50, p90])

        scaler = StandardScaler().fit(X_lead)
        ridge = Ridge(alpha=10.0).fit(scaler.transform(X_lead), y_lead)
        p_ridge = ridge.predict(scaler.transform(X_t0))[0]
        p50_c = consolidate_p50(lead, p50, p_ridge)
        nat_p10[i] = p10 + (p50_c - p50)
        nat_p50[i] = p50_c
        nat_p90[i] = p90 + (p50_c - p50)

        for k in range(cfg.n_eof):
            Xk, yk = make_direct_dataset(feat_lead, pcs_df[f"pc{k}"], lead)
            pc_m = fit_pc_model(Xk, yk)
            pc_pred[i, k] = pc_m.predict(X_t0[Xk.columns])[0]

    stack = np.sort(np.vstack([nat_p10, nat_p50, nat_p90]), axis=0)
    nat_p10, nat_p50, nat_p90 = stack[0], stack[1], stack[2]

    # ---- STAP 7: ruimtelijke velden ----
    _p(0.92, "Ruimtelijke velden & terciel-kansen…")
    forecast_p50 = np.zeros((n_fut, n_lat, n_lon), dtype=np.float32)
    forecast_p10 = np.zeros_like(forecast_p50)
    forecast_p90 = np.zeros_like(forecast_p50)
    clim_3d = np.zeros_like(forecast_p50)
    anom_3d = np.zeros_like(forecast_p50)
    pct_3d = np.zeros_like(forecast_p50)
    prob_below = np.zeros_like(forecast_p50)
    prob_normal = np.zeros_like(forecast_p50)
    prob_above = np.zeros_like(forecast_p50)
    skill_mask3d = np.zeros_like(forecast_p50)

    for i, fdate in enumerate(future_months):
        mo = fdate.month
        clim_mo = sp_clim[mo - 1]
        anom_rec = reconstruct_field(pc_pred[i], eofs)
        offset = nat_p50[i] - np.nanmean(anom_rec)
        anom_adj = anom_rec + offset

        f50 = np.clip(clim_mo + anom_adj, 0, None)
        f10 = np.clip(f50 + (nat_p10[i] - nat_p50[i]), 0, None)
        f90 = np.clip(f50 + (nat_p90[i] - nat_p50[i]), 0, None)

        sigma_nat = max((nat_p90[i] - nat_p10[i]) / 2.5631, 1e-3)
        pB, pN, pA = tercile_probs_field(
            anom_adj, clim_mo, terc_bounds[mo - 1], sigma_nat, sigma_ratio)

        if spatial_val and not np.all(np.isnan(grocs_lead[i])):
            skill = grocs_lead[i] > cfg.grocs_threshold
        elif run_val:
            skill = np.full((n_lat, n_lon), skill_rows[i]["grocs"] > cfg.grocs_threshold)
        else:
            skill = np.full((n_lat, n_lon), True)   # geen validatie → geen masking
        pB = np.where(skill, pB, 1 / 3)
        pN = np.where(skill, pN, 1 / 3)
        pA = np.where(skill, pA, 1 / 3)

        forecast_p50[i], forecast_p10[i], forecast_p90[i] = f50, f10, f90
        clim_3d[i] = clim_mo
        anom_3d[i] = f50 - clim_mo
        pct_3d[i] = np.where(clim_mo > 1e-3, f50 / clim_mo * 100, np.nan)
        prob_below[i], prob_normal[i], prob_above[i] = pB, pN, pA
        skill_mask3d[i] = skill.astype(np.float32)

    for arr in (forecast_p50, forecast_p10, forecast_p90, clim_3d, anom_3d, pct_3d,
                prob_below, prob_normal, prob_above, skill_mask3d):
        arr[:, ~valid_mask] = np.nan

    # ---- STAP 8: Dataset opbouwen ----
    _p(0.98, "NetCDF-dataset opbouwen…")

    def _da(arr, long_name, units, extra=None):
        at = {"long_name": long_name, "units": units, "grid_mapping": "crs"}
        if extra:
            at.update(extra)
        return xr.DataArray(arr, dims=["time", "y", "x"], attrs=at)

    ds_out = xr.Dataset(
        {
            "precipitation_p50": _da(forecast_p50, "Neerslagvoorspelling — mediaan (P50)", "mm month-1"),
            "precipitation_p10": _da(forecast_p10, "Neerslagvoorspelling — 10e percentiel", "mm month-1",
                                     {"description": "Droog scenario: 10% kans op minder dan deze waarde."}),
            "precipitation_p90": _da(forecast_p90, "Neerslagvoorspelling — 90e percentiel", "mm month-1",
                                     {"description": "Nat scenario: 10% kans op meer dan deze waarde."}),
            "precipitation_climatology": _da(clim_3d, "Maandelijkse klimatologie 1991–2020", "mm month-1"),
            "precipitation_anomaly": _da(anom_3d, "P50-anomalie t.o.v. klimatologie 1991–2020", "mm month-1",
                                         {"positive_note": "Positief = natter dan normaal"}),
            "precipitation_pct_normal": _da(pct_3d, "P50 als percentage van de klimatologie", "%"),
            "prob_below": _da(prob_below, "Kans op onder-normale neerslag (onderste terciel)", "1"),
            "prob_normal": _da(prob_normal, "Kans op rond-normale neerslag (middelste terciel)", "1"),
            "prob_above": _da(prob_above, "Kans op boven-normale neerslag (bovenste terciel)", "1"),
            "grocs": _da(grocs_lead, "GROCS-skill uit walk-forward hindcasts (per lead)", "1"),
            "skill_mask": _da(skill_mask3d, "Skill-masker (1 = GROCS > drempel)", "1"),
        },
        coords={
            "time": ("time", pd.DatetimeIndex(future_months),
                     {"long_name": "Tijd", "standard_name": "time", "axis": "T"}),
            "y": ("y", lats, {"long_name": "Latitude", "units": "degrees_north",
                              "standard_name": "latitude", "axis": "Y"}),
            "x": ("x", lons, {"long_name": "Longitude", "units": "degrees_east",
                              "standard_name": "longitude", "axis": "X"}),
        },
        attrs={
            "title": (f"Suriname Maandelijkse Neerslagvoorspelling v3 "
                      f"{future_months[0]:%B %Y} – {future_months[-1]:%B %Y}"),
            "institution": "Meteorologische Dienst Suriname (MDS / HydroMet)",
            "Conventions": "CF-1.8",
            "history": f"Aangemaakt op {pd.Timestamp.now():%Y-%m-%d %H:%M} via Streamlit-GUI",
            "forecast_issue_month": f"{t0:%Y-%m}",
            "forecast_start": cfg.forecast_start,
            "forecast_months": str(cfg.forecast_n_months),
            "climatology_period": f"{cfg.clim_start[:7]} t/m {cfg.clim_end[:7]} (WMO-normaal)",
            "n_eof": str(cfg.n_eof),
            "eof_explained_variance": ", ".join(f"{v*100:.1f}%" for v in expl_var),
            "nmme_models": ", ".join(cfg.nmme_models) if nmme_df is not None else "geen (v2-modus)",
            "grocs_threshold": str(cfg.grocs_threshold),
            "validation_mode": cfg.val_mode,
        })

    ds_out.attrs["_skill_rows"] = str(skill_rows)
    _p(1.0, "Klaar.")
    return ds_out
