# 🌧️ Suriname Seizoens-Outlook

Een [Streamlit](https://streamlit.io)-app om de seizoens-outlook voor Suriname te
maken op basis van de historische data in deze repository.

## Wat kan de app?

- **Kies welke plots je wil maken** (meerdere tegelijk mogelijk):
  1. **Neerslag klimatologie** – gemiddelde neerslag voor de gekozen maand
  2. **Neerslag anomalie (jaar)** – een specifiek jaar/maand t.o.v. de klimatologie
  3. **ENSO-composiet** – gemiddelde neerslag in El Niño / La Niña / Neutrale jaren
  4. **ENSO-anomalie (outlook)** – de outlook-signaal: composiet minus klimatologie
  5. **SST-anomalie** – zeewatertemperatuur-anomalie (NOAA ERSST v5)
- **Kies de maand** (bv. Juni).
- **Klik op de kaart op een district** (of kies het uit de lijst). De neerslagplots
  worden bijgesneden op de **extent van dat district** en op de districtsvorm
  geclipt — klik dus op Nickerie voor de outlook van Nickerie in juni.
- **Pas de titel** van elke plot aan.
- Alle velden worden getekend met **`contourf`** en standaard **6 niveaus**
  (instelbaar).

## Data

Alles zit in de repository, behalve de SST (te groot voor git):

| Bestand | Inhoud |
|---|---|
| `Suriname_monthly_rainfall_jan_1982-may_2026.nc` | Maandelijkse neerslag (mm), 0.05° grid |
| `nina34.anom.nc` | Niño3.4-anomalie (°C), voor ENSO-fasering |
| `DistriktenSuriname.shp` (+ `.dbf/.shx/.prj`) | 10 districten van Suriname |
| `sst.mnmean.nc` | NOAA ERSST v5 SST — **GitHub-release `sstTemp`** |

De SST wordt **automatisch gedownload** vanuit de release naar `.sst_cache/`
zodra je voor het eerst een SST-plot maakt (knop in de app). Het bestand wordt
niet in git bewaard.

## Installeren & starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

De app opent in je browser. Kies links de plots en de maand, klik een district
op de kaart, en de plots verschijnen eronder.

## ENSO-classificatie

Per jaar wordt de ENSO-fase van de gekozen maand bepaald met de Niño3.4-anomalie:
`≥ +0.5 °C` = El Niño, `≤ −0.5 °C` = La Niña, anders Neutraal. De drempel is
instelbaar in de zijbalk.
