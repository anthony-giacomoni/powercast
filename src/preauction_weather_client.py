# ============================================================
# PowerCast — strict pre-auction weather client
#
# Reconstructs ONE fixed ECMWF IFS 00 UTC run from D-1 for each French
# delivery day D. The 00Z run is normally distributed several hours
# before the French day-ahead auction, avoiding the rolling lead-time
# leakage of Open-Meteo Previous Runs _previous_day1 series.
#
# Open-Meteo Single Runs API; ECMWF IFS HRES archive from 2024-03-14.
# Python 3.8 compatible.
# ============================================================

import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

FRENCH_CITIES = {
    "Paris":      {"lat": 48.85, "lon": 2.35,  "population": 10_800_000},
    "Marseille":  {"lat": 43.30, "lon": 5.37,  "population": 1_900_000},
    "Lyon":       {"lat": 45.76, "lon": 4.83,  "population": 1_700_000},
    "Toulouse":   {"lat": 43.60, "lon": 1.44,  "population": 1_000_000},
    "Lille":      {"lat": 50.63, "lon": 3.06,  "population": 1_200_000},
    "Nantes":     {"lat": 47.22, "lon": -1.55, "population": 970_000},
    "Strasbourg": {"lat": 48.58, "lon": 7.75,  "population": 800_000},
}

HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "cloud_cover",
    "shortwave_radiation",
]

OUTPUT_COLS = [
    "preauction_temperature_c",
    "preauction_wind_speed_10m_kmh",
    "preauction_wind_speed_100m_kmh",
    "preauction_cloud_cover_pct",
    "preauction_shortwave_radiation_wm2",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    num = np.nansum(values * weights[None, :], axis=1)
    den = np.sum(mask * weights[None, :], axis=1)
    out = np.full(len(values), np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _request_run(run_time, session: requests.Session, max_retries: int = 6):
    names = list(FRENCH_CITIES.keys())
    lats = ",".join(str(FRENCH_CITIES[n]["lat"]) for n in names)
    lons = ",".join(str(FRENCH_CITIES[n]["lon"]) for n in names)
    params = {
        "latitude": lats,
        "longitude": lons,
        "run": (
            run_time.strftime("%Y-%m-%dT%H:%M")
            if isinstance(run_time, datetime)
            else run_time.strftime("%Y-%m-%dT00:00")
        ),
        "models": "ecmwf_ifs",
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "forecast_days": 3,
        "wind_speed_unit": "kmh",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            r = session.get(SINGLE_RUNS_URL, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = "HTTP %s" % r.status_code
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 5.0 * (attempt + 1)
                else:
                    delay = min(5.0 * (2 ** attempt), 60.0)
                time.sleep(delay)
                continue
            # Permanent error (e.g. unavailable archived run)
            snippet = r.text[:500].replace("\n", " ")
            raise RuntimeError("Open-Meteo HTTP %s: %s" % (r.status_code, snippet))
        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError("Open-Meteo temporary failure after retries: %s" % last_error)


def _parse_multicity_run(payload, delivery_date: date) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Unexpected Open-Meteo multi-location response")

    names = list(FRENCH_CITIES.keys())
    if len(payload) != len(names):
        raise ValueError("Expected %d locations, got %d" % (len(names), len(payload)))

    long_parts = []
    for idx, (name, item) in enumerate(zip(names, payload)):
        hourly = item.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
        if len(times) == 0:
            continue
        part = pd.DataFrame({"datetime": times})
        for var in HOURLY_VARS:
            vals = hourly.get(var)
            if vals is None:
                part[var] = np.nan
            else:
                part[var] = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
        part["city"] = name
        part["weight"] = float(FRENCH_CITIES[name]["population"])
        long_parts.append(part)

    if not long_parts:
        return pd.DataFrame(columns=["datetime"] + OUTPUT_COLS)

    long_df = pd.concat(long_parts, ignore_index=True)
    local_date = long_df["datetime"].dt.tz_convert("Europe/Paris").dt.date
    long_df = long_df[local_date == delivery_date].copy()
    if long_df.empty:
        return pd.DataFrame(columns=["datetime"] + OUTPUT_COLS)

    # Pivot each weather field to cities, preserving timestamps and weights.
    result = pd.DataFrame({"datetime": sorted(long_df["datetime"].dropna().unique())})
    weights = np.array([FRENCH_CITIES[n]["population"] for n in names], dtype=float)
    mapping = {
        "temperature_2m": "preauction_temperature_c",
        "wind_speed_10m": "preauction_wind_speed_10m_kmh",
        "wind_speed_100m": "preauction_wind_speed_100m_kmh",
        "cloud_cover": "preauction_cloud_cover_pct",
        "shortwave_radiation": "preauction_shortwave_radiation_wm2",
    }
    for raw_col, out_col in mapping.items():
        p = long_df.pivot_table(index="datetime", columns="city", values=raw_col, aggfunc="first")
        p = p.reindex(columns=names)
        p = p.reindex(pd.DatetimeIndex(result["datetime"]))
        result[out_col] = _weighted_mean(p.to_numpy(dtype=float), weights)

    return result.sort_values("datetime").reset_index(drop=True)


def fetch_delivery_day_weather(
    delivery_date: date,
    cache_dir: str = "data/cache/weather_preauction_ecmwf",
    session: Optional[requests.Session] = None,
) -> Tuple[pd.DataFrame, bool]:
    """
    Returns national weather for local French delivery day D using the
    ECMWF IFS 00Z run initialised on D-1. bool=True means loaded from cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, delivery_date.strftime("%Y-%m-%d") + ".csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        return df, True

    run_date = delivery_date - timedelta(days=1)
    sess = session or requests.Session()
    try:
        payload = _request_run(run_date, sess)
    except RuntimeError as e:
        # Rare archived 00Z gaps: use the latest strictly older full run,
        # D-2 12Z. It is still known well before the D-1 day-ahead auction.
        if "requested model run is not available" not in str(e):
            raise
        fallback_day = delivery_date - timedelta(days=2)
        fallback_run = datetime(
            fallback_day.year, fallback_day.month, fallback_day.day, 12, 0
        )
        print(
            "      ↳ %s 00Z unavailable; fallback to %s 12Z"
            % (run_date, fallback_day)
        )
        payload = _request_run(fallback_run, sess)
    df = _parse_multicity_run(payload, delivery_date)
    if df.empty:
        raise RuntimeError("No ECMWF 00Z data reconstructed for delivery day %s" % delivery_date)
    df.to_csv(cache_path, index=False)
    return df, False


def get_national_preauction_weather(
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache/weather_preauction_ecmwf",
) -> pd.DataFrame:
    """Fetch local delivery days in [start_date, end_date), cache/resume safe."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start < date(2024, 3, 15):
        raise ValueError(
            "Strict ECMWF 00Z reconstruction starts at delivery day 2024-03-15 "
            "because the Single Runs IFS archive starts on 2024-03-14."
        )
    if end <= start:
        raise ValueError("end_date must be after start_date")

    all_days = []
    sess = requests.Session()
    cursor = start
    active_month = None
    month_stats = {"fetched": 0, "cache": 0, "failed": 0}

    def flush_month(month_label, stats):
        if month_label is not None:
            total = stats["fetched"] + stats["cache"] + stats["failed"]
            print(
                "  ECMWF 00Z %s: %d delivery days | fetched=%d | cache=%d | failed=%d"
                % (month_label, total, stats["fetched"], stats["cache"], stats["failed"])
            )

    while cursor < end:
        month_label = cursor.strftime("%Y-%m")
        if active_month is None:
            active_month = month_label
        elif month_label != active_month:
            flush_month(active_month, month_stats)
            active_month = month_label
            month_stats = {"fetched": 0, "cache": 0, "failed": 0}

        try:
            day_df, from_cache = fetch_delivery_day_weather(cursor, cache_dir, sess)
            all_days.append(day_df)
            month_stats["cache" if from_cache else "fetched"] += 1
        except Exception as e:
            month_stats["failed"] += 1
            print("    ⚠️  %s: %s" % (cursor, e))
        cursor += timedelta(days=1)

    flush_month(active_month, month_stats)
    if not all_days:
        return pd.DataFrame(columns=["datetime"] + OUTPUT_COLS)

    out = pd.concat(all_days, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
    return out.sort_values("datetime").reset_index(drop=True)
