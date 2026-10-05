# ============================================================
#  France Power Price Predictor — Data Pipeline
#
#  Orchestrates the three data sources into a single hourly dataset
#  ready for feature engineering:
#    - ENTSO-E               -> EPEX day-ahead price (target variable)
#                              [CONFIRMED WORKING — real historical
#                              data, unlike RTE's API which only
#                              serves today/tomorrow]
#    - ODRÉ éCO2mix          -> production by source (incl. nuclear
#                              availability), consumption [CONFIRMED
#                              WORKING — real column names verified]
#    - Open-Meteo            -> temperature (demand driver) [CONFIRMED
#                              WORKING, multi-city population-weighted]
#
#  Usage:
#      python src/build_dataset.py --start 2024-01-01 --end 2026-08-01
# ============================================================

import argparse
import sys
import os
from datetime import datetime, timedelta

import pandas as pd

from data_utils import resample_hourly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entsoe_client import get_historical_prices
from odre_client import get_eco2mix_data, DATASET_HISTORICAL
from weather_client import get_national_weather, FRENCH_CITIES


def fetch_prices(start: str, end: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Récupère l'historique de prix day-ahead via ENTSO-E (pas RTE — voir
    entsoe_client.py : RTE ne sert que today/tomorrow, ENTSO-E a le vrai
    historique multi-années). CONFIRMED WORKING — testé avec succès sur
    janvier 2026 (665 lignes récupérées).

    Découpe la plage par année civile — ENTSO-E documente une limite
    d'environ 1 an par requête pour ce type de document, et une plage
    multi-années en un seul appel risque en plus le même genre de
    timeout déjà observé sur une plage de 5 mois.

    Chaque année est mise en cache sur disque (data/cache/), même
    raisonnement que pour ODRÉ : éviter de tout refaire en cas de
    plantage à mi-parcours sur une longue plage (ex: 2015-2026).
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start.replace(year=chunk_start.year + 1), end_dt)
        cache_file = os.path.join(cache_dir, f"entsoe_{chunk_start.year}.csv")

        if os.path.exists(cache_file):
            print(f"  ENTSO-E chunk: {chunk_start.date()} to {chunk_end.date()} (from cache)")
            df_chunk = pd.read_csv(cache_file)
        else:
            print(f"  ENTSO-E chunk: {chunk_start.date()} to {chunk_end.date()} (fetching...)")
            df_chunk = get_historical_prices(
                chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            )
            if df_chunk is not None and not df_chunk.empty:
                df_chunk.to_csv(cache_file, index=False)

        if df_chunk is not None and not df_chunk.empty:
            chunks.append(df_chunk)
        chunk_start = chunk_end

    if not chunks:
        print("⚠️  No price data returned from ENTSO-E — check API key/subscription.")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Paris")
    return resample_hourly(df)


def fetch_generation(start: str, end: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Récupère la production par filière (dont nucléaire) et la consommation
    via ODRÉ (pas RTE OAuth2 — voir odre_client.py). CONFIRMED WORKING,
    real column names verified against a live response:
    consommation, nucleaire, eolien, solaire, hydraulique, gaz, taux_co2, etc.

    Utilise le dataset historique (cons-def) plutôt que temps-réel (tr),
    car ce dernier ne garde qu'~1 mois d'historique glissant — pas
    suffisant pour entraîner un modèle sur une période étendue.

    IMPORTANT: ODRÉ plafonne strictement offset + limit <= 10000 par
    requête (confirmé par le serveur : "InvalidRESTParameterError" au-delà).
    À 15 min de résolution, 10000 lignes ~= 104 jours -- donc pour une
    plage plus large, on découpe en tranches mensuelles, chacune avec son
    propre offset repartant de 0, plutôt qu'un seul appel qui dépasserait
    la limite au milieu de la plage demandée.

    Sur une longue plage (plusieurs années), chaque tranche mensuelle est
    mise en cache sur disque (data/cache/) au fur et à mesure : si le
    script plante ou time out à mi-parcours (probable sur >100 appels
    réseau successifs), on peut relancer et ne refaire QUE les mois
    manquants au lieu de tout reprendre à zéro.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        if chunk_start.month == 12:
            chunk_end = chunk_start.replace(year=chunk_start.year + 1, month=1)
        else:
            chunk_end = chunk_start.replace(month=chunk_start.month + 1)
        chunk_end = min(chunk_end, end_dt)

        cache_file = os.path.join(cache_dir, f"odre_{chunk_start.strftime('%Y-%m')}.csv")

        if os.path.exists(cache_file):
            print(f"  ODRÉ chunk: {chunk_start.date()} to {chunk_end.date()} (from cache)")
            df_chunk = pd.read_csv(cache_file)
        else:
            print(f"  ODRÉ chunk: {chunk_start.date()} to {chunk_end.date()} (fetching...)")
            df_chunk = get_eco2mix_data(
                dataset=DATASET_HISTORICAL,
                start_date=chunk_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
                limit=10000,
            )
            if df_chunk is not None and not df_chunk.empty:
                df_chunk.to_csv(cache_file, index=False)

        if df_chunk is not None and not df_chunk.empty:
            chunks.append(df_chunk)

        chunk_start = chunk_end

    if not chunks:
        print("⚠️  No generation data returned from ODRÉ.")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # IMPORTANT — correction d'une fuite temporelle (data leakage) :
    # 'consommation' est la valeur RÉALISÉE, connue seulement une fois
    # l'heure passée — elle n'était jamais disponible au moment réel où
    # un vrai forecast day-ahead aurait dû être fait. On utilise
    # 'prevision_j1' à la place : la prévision de consommation établie
    # PAR RTE LA VEILLE (J-1), c'est-à-dire l'information réellement
    # disponible ex-ante au moment de la prédiction. Renommée
    # 'consumption_forecast_j1' pour que ce soit explicite dans le
    # dataset final (pas une valeur réalisée déguisée).
    keep_cols = ["date_heure", "prevision_j1", "nucleaire", "eolien", "solaire",
                 "hydraulique", "gaz", "charbon", "fioul", "bioenergies", "taux_co2",
                 "ech_physiques"]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].rename(columns={
        "date_heure": "datetime",
        "prevision_j1": "consumption_forecast_j1",
        # ech_physiques = solde physique import/export France (MW) —
        # positif = France importe (déficit domestique), négatif =
        # France exporte (surplus). Influence directement le prix spot :
        # un fort import signale une tension système que le prix reflète,
        # symétriquement pour un fort export en cas de surproduction.
        # NOTE: ceci reste une valeur RÉALISÉE (pas de prévision publique
        # trouvée pour les imports/exports) — fuite résiduelle assumée
        # et documentée, contrairement à consommation/nucléaire/météo
        # qui sont maintenant corrigées.
        "ech_physiques": "net_imports_mw",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Paris")
    return resample_hourly(df)


def fetch_weather(start: str, end: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Récupère la météo nationale pondérée par population (7 grandes villes
    françaises) — voir weather_client.get_national_weather pour le détail
    de la pondération (elle-même découpée par année en interne pour
    respecter la limite Open-Meteo de ~366 jours/appel). CONFIRMED WORKING.

    Mis en cache sur disque pour la même raison que ODRÉ/ENTSO-E : 77
    appels réseau (7 villes x 11 ans) prennent du temps, autant ne pas
    tout refaire si le script plante ailleurs après.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"weather_{start}_{end}.csv")

    if os.path.exists(cache_file):
        print(f"  Weather: {start} to {end} (from cache)")
        df = pd.read_csv(cache_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        print(f"  Weather: {start} to {end} (fetching, {len(FRENCH_CITIES)} cities x years...)")
        df = get_national_weather(start, end)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)

    if df is None or df.empty:
        print("⚠️  No weather data returned.")
        return pd.DataFrame()

    df["datetime"] = df["datetime"].dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="NaT")
    return resample_hourly(df)


def build_dataset(start: str, end: str, output_path: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Assemble les trois sources en un seul DataFrame joint sur 'datetime',
    et sauvegarde en CSV. Retourne le DataFrame pour inspection immédiate.

    Si les prix RTE ne sont pas encore disponibles (auth en attente),
    construit quand même un dataset partiel (ODRÉ + météo) sous
    data/dataset_partial.csv, pour permettre d'avancer sur l'exploration
    et la préparation du feature engineering sans attendre RTE.
    """
    print(f"Fetching prices from {start} to {end}...")
    prices = fetch_prices(start, end)
    print(f"  -> {len(prices)} rows")

    print(f"Fetching generation/nuclear data...")
    generation = fetch_generation(start, end)
    print(f"  -> {len(generation)} rows")

    print(f"Fetching weather...")
    weather = fetch_weather(start, end)
    print(f"  -> {len(weather)} rows")

    if prices.empty:
        print("⚠️  No price data (ENTSO-E unavailable) — building a PARTIAL dataset "
              "(ODRÉ + weather only, no target variable) for exploration purposes.")
        if generation.empty and weather.empty:
            print("❌ Nothing to save — both ODRÉ and weather also empty.")
            return pd.DataFrame()

        df = generation if not generation.empty else weather
        if not generation.empty and not weather.empty:
            df = generation.merge(weather, on="datetime", how="left", suffixes=("", "_weather"))

        partial_path = output_path.replace(".csv", "_partial.csv")
        os.makedirs(os.path.dirname(partial_path), exist_ok=True)
        df.to_csv(partial_path, index=False)
        print(f"✅ Saved {len(df)} rows to {partial_path} (partial — no price column yet)")
        return df

    df = prices.copy()
    if not generation.empty and "datetime" in generation.columns:
        df = df.merge(generation, on="datetime", how="left", suffixes=("", "_gen"))
    if not weather.empty:
        df = df.merge(weather, on="datetime", how="left", suffixes=("", "_weather"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the EDF power price dataset.")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=str, default="data/dataset.csv")
    args = parser.parse_args()

    df = build_dataset(args.start, args.end, args.output)
    if not df.empty:
        print("\nFirst rows:")
        print(df.head())
        print("\nColumns:", list(df.columns))
