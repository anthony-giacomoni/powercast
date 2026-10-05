# ============================================================
#  PowerCast — France Power Price Predictor — Weather Client (Open-Meteo)
#
#  Open-Meteo is free, requires no API key, and has 40+ years of
#  historical archive data. Used here as a proxy for demand drivers:
#  temperature strongly correlates with electricity demand (heating
#  in winter, air conditioning in summer) — a key input for price
#  prediction.
#
#  Queries several major French cities and computes a population-
#  weighted national average temperature — a more realistic proxy for
#  nationwide electricity demand than one city alone. Population
#  weighting is a reasonable, defensible choice; it doesn't capture
#  industrial-heavy regions with lower population but disproportionate
#  consumption, which would be a further refinement if pursued
#  (weighting by RTE regional consumption data instead of population,
#  once RTE access works).
#
#  CONFIRMED WORKING against the real API (both single-city and
#  multi-city weighted versions tested successfully).
# ============================================================

import requests
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Previous Runs API — donne accès, pour chaque heure passée, à la valeur
# QUI AVAIT ÉTÉ PRÉVUE 24h avant (temperature_2m_previous_day1), pas la
# valeur réalisée. Nécessaire pour un vrai forecast day-ahead sans fuite
# temporelle. Couverture limitée à ~2024+ selon les modèles (confirmé
# via la doc officielle Open-Meteo) — d'où le dataset séparé "clean"
# 2024-2026, distinct du dataset historique 2015-2026 basé sur l'archive
# réalisée (voir build_dataset.py pour le détail des deux pipelines).
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Grandes villes françaises avec coordonnées + population approximative
# (aire urbaine, ordre de grandeur suffisant pour une pondération —
# source : INSEE, arrondi). Choisies pour une bonne couverture
# géographique (nord/sud/est/ouest) en plus du poids démographique.
FRENCH_CITIES = {
    "Paris":     {"lat": 48.85, "lon": 2.35,  "population": 10_800_000},
    "Marseille": {"lat": 43.30, "lon": 5.37,  "population": 1_900_000},
    "Lyon":      {"lat": 45.76, "lon": 4.83,  "population": 1_700_000},
    "Toulouse":  {"lat": 43.60, "lon": 1.44,  "population": 1_000_000},
    "Lille":     {"lat": 50.63, "lon": 3.06,  "population": 1_200_000},
    "Nantes":    {"lat": 47.22, "lon": -1.55, "population": 970_000},
    "Strasbourg": {"lat": 48.58, "lon": 7.75, "population": 800_000},
}

_TOTAL_POPULATION = sum(c["population"] for c in FRENCH_CITIES.values())


def _fetch_city_weather(url: str, lat: float, lon: float, extra_params: Dict) -> Optional[Dict]:
    """Appel générique Open-Meteo pour une seule ville."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,cloud_cover",
        "timezone": "Europe/Paris",
        **extra_params,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Open-Meteo error for lat={lat},lon={lon}: {e}")
        return None


def get_historical_weather(start_date: str, end_date: str,
                            lat: float = FRENCH_CITIES["Paris"]["lat"],
                            lon: float = FRENCH_CITIES["Paris"]["lon"]) -> Optional[Dict]:
    """
    Historique météo horaire pour UNE ville (comportement d'origine,
    conservé pour compatibilité — utiliser get_national_weather ci-dessous
    pour la version multi-villes pondérée).
    Dates au format 'YYYY-MM-DD'.
    """
    return _fetch_city_weather(ARCHIVE_URL, lat, lon, {
        "start_date": start_date, "end_date": end_date,
    })


def get_forecast_weather(days: int = 7,
                          lat: float = FRENCH_CITIES["Paris"]["lat"],
                          lon: float = FRENCH_CITIES["Paris"]["lon"]) -> Optional[Dict]:
    """Prévisions météo à venir pour une ville (utile pour un prix J+1 réel)."""
    return _fetch_city_weather(FORECAST_URL, lat, lon, {"forecast_days": days})


def _weighted_mean_ignoring_nan(df_wide: pd.DataFrame, prefix: str, city_pop: list) -> pd.Series:
    """
    Moyenne pondérée par population, robuste aux trous ville par ville.

    BUG CORRIGÉ (historique) : une version précédente faisait
    sum(df[f"temp_{name}"] * poids for chaque ville), ce qui propage un
    NaN d'UNE SEULE ville à toute la ligne (NaN + x = NaN en arithmétique
    standard), même si les autres villes avaient de vraies valeurs.
    Fix : pour chaque ligne, on ne pondère que sur les villes qui ont une
    vraie valeur, en renormalisant les poids pour que leur somme fasse
    toujours 1.
    """
    cols = [f"{prefix}_{name}" for name, _, _ in city_pop]
    weights = np.array([pop for _, pop, _ in city_pop], dtype=float)

    values = df_wide[cols].to_numpy()
    mask = ~np.isnan(values)

    weighted_sum = np.nansum(values * weights, axis=1)
    weight_total = (mask * weights).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = weighted_sum / weight_total
    result[weight_total == 0] = np.nan
    return pd.Series(result, index=df_wide.index)


def get_national_weather(start_date: str, end_date: str,
                          cities: Dict = None) -> Optional[pd.DataFrame]:
    """
    Récupère la météo horaire pour plusieurs villes françaises et calcule
    une température (+ vent, nébulosité) nationale pondérée par population.

    C'est un meilleur proxy de la demande électrique nationale qu'un
    point unique (Paris) : la demande de chauffage/clim dépend de la
    météo locale partout en France, pas seulement en Île-de-France.

    IMPORTANT: Open-Meteo limite chaque appel à ~366 jours (confirmé par
    la doc officielle). Pour une plage plus longue, on découpe par année
    civile par ville, comme pour ODRÉ/ENTSO-E dans build_dataset.py —
    sinon l'appel échoue silencieusement sur une plage multi-années.

    Utilise l'API ARCHIVE (données consolidées) — a un délai de
    traitement de quelques heures à un jour, donc pas adaptée pour "la
    température actuelle, maintenant" (voir get_national_weather_now
    pour ça). Bonne pour l'historique d'entraînement du modèle.

    Retourne un DataFrame avec colonnes: datetime, temperature_c,
    wind_speed_kmh, cloud_cover_pct (toutes pondérées par population),
    ou None si aucune ville n'a pu être récupérée.
    """
    from datetime import datetime as _dt

    cities = cities or FRENCH_CITIES
    city_dfs = []

    start_dt = _dt.strptime(start_date, "%Y-%m-%d")
    end_dt = _dt.strptime(end_date, "%Y-%m-%d")

    for name, info in cities.items():
        year_dfs = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start.replace(year=chunk_start.year + 1), end_dt)
            raw = _fetch_city_weather(ARCHIVE_URL, info["lat"], info["lon"], {
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
            })
            if raw and "hourly" in raw:
                hourly = raw["hourly"]
                year_dfs.append(pd.DataFrame({
                    "datetime": pd.to_datetime(hourly["time"]),
                    f"temp_{name}": hourly.get("temperature_2m"),
                    f"wind_{name}": hourly.get("wind_speed_10m"),
                    f"cloud_{name}": hourly.get("cloud_cover"),
                }))
            else:
                print(f"⚠️  {name}: no data for {chunk_start.date()}–{chunk_end.date()}")
            chunk_start = chunk_end

        if not year_dfs:
            print(f"⚠️  Skipping {name} entirely — no data for any year.")
            continue

        city_dfs.append((name, info["population"], pd.concat(year_dfs, ignore_index=True)))

    if not city_dfs:
        print("❌ No city weather data could be retrieved at all.")
        return None

    # Assemble toutes les villes sur le même axe temporel
    merged = city_dfs[0][2][["datetime"]].copy()
    for name, population, df in city_dfs:
        merged = merged.merge(df, on="datetime", how="outer")

    merged["temperature_c"] = _weighted_mean_ignoring_nan(merged, "temp", city_dfs)
    merged["wind_speed_kmh"] = _weighted_mean_ignoring_nan(merged, "wind", city_dfs)
    merged["cloud_cover_pct"] = _weighted_mean_ignoring_nan(merged, "cloud", city_dfs)

    result = merged[["datetime", "temperature_c", "wind_speed_kmh", "cloud_cover_pct"]].copy()
    return result


def get_national_weather_forecast_j1_historical(start_date: str, end_date: str,
                                                  cities: Dict = None) -> Optional[pd.DataFrame]:
    """
    Météo nationale pondérée, mais PRÉVUE 24H À L'AVANCE (Previous Runs
    API, temperature_2m_previous_day1) plutôt que réalisée (Archive API).
    C'est l'information réellement disponible ex-ante pour un vrai
    forecast day-ahead — nécessaire pour éviter la fuite temporelle
    identifiée sur le dataset historique 2015-2026 (qui utilise la météo
    réalisée, voir get_national_weather).

    LIMITE CONNUE : couverture ~2024+ selon les modèles (confirmé via la
    doc officielle Open-Meteo) — pas utilisable pour reconstruire un
    historique propre avant cette date. D'où l'existence de deux
    pipelines séparés dans ce projet plutôt qu'un seul mélangeant les
    deux régimes (voir build_dataset.py).

    Simplification volontaire : l'API compte 'past_days' à partir
    d'AUJOURD'HUI, pas de end_date — donc on demande systématiquement
    past_days = (aujourd'hui - start_date), en une ou plusieurs tranches
    de 92 jours max (limite documentée), puis on filtre après coup à la
    plage [start_date, end_date] réellement voulue. Un peu redondant en
    requêtes mais beaucoup plus simple et robuste qu'un calcul de
    forecast_days ad hoc.
    """
    cities = cities or FRENCH_CITIES
    city_dfs = []

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    today = pd.Timestamp.now().normalize()

    total_past_days = (today - start_dt).days
    if total_past_days <= 0:
        print("❌ start_date must be in the past for historical J-1 forecast data.")
        return None

    for name, info in cities.items():
        raw = None
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params={
                "latitude": info["lat"], "longitude": info["lon"],
                "hourly": "temperature_2m_previous_day1,wind_speed_10m_previous_day1,cloud_cover_previous_day1",
                "past_days": total_past_days,
                "forecast_days": 1,
                "timezone": "Europe/Paris",
            }, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
        except requests.exceptions.HTTPError as e:
            # Le vrai plafond de past_days pour cet endpoint n'est pas
            # clairement documenté par Open-Meteo (contrairement à d'autres
            # limites confirmées ailleurs dans ce projet) — plutôt que de
            # deviner un chiffre, on laisse l'API elle-même signaler un
            # dépassement via son message d'erreur réel.
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass
            print(f"Previous Runs API HTTP error for {name}: {e} — {body}")
        except requests.exceptions.RequestException as e:
            print(f"Previous Runs API network error for {name}: {e}")

        if not raw or "hourly" not in raw:
            print(f"⚠️  Skipping {name} — no J-1 forecast data returned.")
            continue

        hourly = raw["hourly"]
        city_dfs.append((name, info["population"], pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            f"temp_{name}": hourly.get("temperature_2m_previous_day1"),
            f"wind_{name}": hourly.get("wind_speed_10m_previous_day1"),
            f"cloud_{name}": hourly.get("cloud_cover_previous_day1"),
        })))

    if not city_dfs:
        print("❌ No city J-1 forecast data could be retrieved at all — check the "
              "error messages above (the range may exceed this endpoint's real limit).")
        return None

    merged = city_dfs[0][2][["datetime"]].copy()
    for name, population, df in city_dfs:
        merged = merged.merge(df, on="datetime", how="outer")

    merged["temperature_c"] = _weighted_mean_ignoring_nan(merged, "temp", city_dfs)
    merged["wind_speed_kmh"] = _weighted_mean_ignoring_nan(merged, "wind", city_dfs)
    merged["cloud_cover_pct"] = _weighted_mean_ignoring_nan(merged, "cloud", city_dfs)

    result = merged[["datetime", "temperature_c", "wind_speed_kmh", "cloud_cover_pct"]].copy()
    result = result[(result["datetime"] >= start_dt) & (result["datetime"] <= end_dt)]
    return result


def get_national_weather_now(cities: Dict = None) -> Optional[pd.DataFrame]:
    """
    Météo nationale pondérée VIA L'API FORECAST (pas archive), qui inclut
    les toutes dernières heures — l'API archive a un délai de traitement
    de quelques heures à un jour, donc "aujourd'hui, maintenant" y
    apparaît souvent comme un point vide, et le dernier point utilisable
    reste celui de la veille au soir (~23:00). Confirmé en pratique :
    panneau température affichant "04 Sep, 23:00" alors qu'il n'était
    pas encore 9h le lendemain.

    forecast_days=1 suffit (aujourd'hui uniquement) ; Open-Meteo inclut
    les heures passées de la journée en cours dans ce même appel.
    """
    cities = cities or FRENCH_CITIES
    city_dfs = []

    for name, info in cities.items():
        raw = _fetch_city_weather(FORECAST_URL, info["lat"], info["lon"], {"forecast_days": 1})
        if raw and "hourly" in raw:
            hourly = raw["hourly"]
            city_dfs.append((name, info["population"], pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                f"temp_{name}": hourly.get("temperature_2m"),
                f"wind_{name}": hourly.get("wind_speed_10m"),
                f"cloud_{name}": hourly.get("cloud_cover"),
            })))
        else:
            print(f"⚠️  {name}: no forecast data returned.")

    if not city_dfs:
        print("❌ No city forecast data could be retrieved at all.")
        return None

    merged = city_dfs[0][2][["datetime"]].copy()
    for name, population, df in city_dfs:
        merged = merged.merge(df, on="datetime", how="outer")

    merged["temperature_c"] = _weighted_mean_ignoring_nan(merged, "temp", city_dfs)
    merged["wind_speed_kmh"] = _weighted_mean_ignoring_nan(merged, "wind", city_dfs)
    merged["cloud_cover_pct"] = _weighted_mean_ignoring_nan(merged, "cloud", city_dfs)

    # Ne garde que les heures déjà passées (pas les prévisions futures de
    # la journée) — le panneau doit montrer "où on en est", pas une prévision.
    now = pd.Timestamp.now()
    result = merged[merged["datetime"] <= now][["datetime", "temperature_c", "wind_speed_kmh", "cloud_cover_pct"]].copy()
    return result


if __name__ == "__main__":
    from datetime import datetime, timedelta

    print("Testing single-city (Paris) historical weather (last 7 days)...")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    result = get_historical_weather(start, end)
    if result:
        print(f"  -> Got {len(result.get('hourly', {}).get('time', []))} hourly records.")
    else:
        print("  -> Failed — see error above.")

    print(f"\nTesting multi-city national weighted weather ({len(FRENCH_CITIES)} cities, last 7 days)...")
    national = get_national_weather(start, end)
    if national is not None:
        print(f"  -> Got {len(national)} hourly records, columns: {list(national.columns)}")
        print(national.head())
    else:
        print("  -> Failed — see errors above.")
