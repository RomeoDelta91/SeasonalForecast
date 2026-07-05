# 🌧️ Suriname Seizoens-Outlook — GUI

Een [Streamlit](https://streamlit.io)-app die de **v3 hybride statistisch-dynamische
neerslagvoorspelling** voor Suriname genereert én interactief plot — zonder dat je
nog een notebook hoeft te draaien.

De app is de GUI-versie van de twee notebooks:
`LatestSeasonFcast_1.ipynb` (de forecast-motor) en `suriname_forecast_plots.ipynb`
(de plotter).

## Wat kan de app?

**① Forecast genereren of laden**
- **Genereren**: de app draait de forecast-motor (`forecast_engine.py`) op de data in
  deze repo en produceert de 11-variabelen dataset (`suriname_forecast_v3_spatial.nc`):
  P50/P10/P90, klimatologie, anomalie, %-van-normaal, terciel-kansen (onder/rond/boven),
  GROCS en skill-masker.
  - Rekenmodus instelbaar: **Snel** (geen validatie, ~1 min), **Standaard**
    (nationale GROCS) of **Volledig** (ruimtelijke GROCS-kaarten, traag). Resultaten
    worden gecachet.
  - Optioneel **NMME** dynamische predictoren (IRI Data Library; valt zacht terug als
    er geen internet/data is).
- **Uploaden**: laad een bestaand `suriname_forecast_v3_spatial.nc`.

**② + ③ Plotten**
- **Kies welke variabelen** je wil plotten (meerdere tegelijk).
- **Kies de voorspelde maand** (bv. Juni 2026).
- **Klik op de kaart op een district** (of kies het uit de lijst) → de plots worden
  bijgesneden op de **extent én de vorm** van dat district (bv. Juni voor Nickerie).
- **Pas de titel** van elke plot aan.
- Alle velden worden getekend met **`contourf`** en standaard **6 niveaus** (instelbaar).
- De gegenereerde forecast is als NetCDF te downloaden.

## Data

| Bestand | Inhoud |
|---|---|
| `data.nc` | CHIRPS maandneerslag (mm), 0.05° — mag op elke maand eindigen |
| `nina34.anom.nc` | Niño3.4-anomalie (°C) |
| `DistriktenSuriname.shp` (+ `.dbf/.shx/.prj`) | 10 districten van Suriname |
| `sst.mnmean.nc` | NOAA ERSST v5 — **GitHub-release `sstTemp`** (TNA/TSA-indices) |

De SST wordt automatisch vanuit de release naar `.sst_cache/` gedownload (knop in de
app) en niet in git bewaard.

**Data bijwerken**: vervang `data.nc` door de nieuwste reeks (zelfde structuur,
zelfde naam) — meer is niet nodig. De reeks mag op elke willekeurige maand eindigen;
de app bepaalt zelf de laatste datamaand en start de forecast op de eerstvolgende
maand (eindigt de data op mei, dan begint de verwachting in juni, enz.).

## Installeren & starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Bij "Genereren" heb je eenmalig de SST-download (~154 MB) nodig; die wordt lokaal
gecachet.

## Bestanden

| Bestand | Rol |
|---|---|
| `app.py` | Streamlit-GUI (forecast draaien + interactief plotten) |
| `forecast_engine.py` | Forecast-motor, port van `LatestSeasonFcast_1.ipynb` |
| `outlook.py` | Geodata- en plot-helpers (district-clip, contourf, variabele-specs) |

## Methode (samengevat)

GradientBoosting-kwantielen (P10/P50/P90) op de nationale neerslaganomalie ⊕ Ridge
(skill-gewogen consolidatie, MSESS²); RandomForest per EOF-PC voor het ruimtelijke
patroon; terciel-kansen via normale benadering met GROCS-masking. Predictoren:
gelagde neerslag-/PC-anomalieën, Niño3.4, ERSSTv5 TNA/TSA/gradiënt en optioneel NMME
box-indices. Klimatologie-referentie: WMO-normaal 1991–2020.
