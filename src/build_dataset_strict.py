# ============================================================
# PowerCast — strict pre-auction dataset builder
#
# Builds the final strict training dataset using only information
# admissible before the D-1 forecasting cutoff:
#
#   - ENTSO-E A44 French day-ahead prices          -> target
#   - ODRÉ J-1 consumption forecast
#   - ENTSO-E A80 nuclear availability
#   - fixed ECMWF pre-auction weather snapshot
#   - TTF gas futures
#   - EEX EUA auction prices
#
# Explicitly excluded:
#   - realised generation
#   - realised weather
#   - A69 renewable forecast
#   - A09 scheduled commercial exchanges
#   - A61 transfer capacity
# ============================================================

import argparse
import os
from datetime import datetime

import pandas as pd

from data_utils import resample_hourly
from entsoe_client import get_historical_prices
from odre_client import (
    get_eco2mix_data,
    DATASET_HISTORICAL,
)
from entsoe_unavailability_client import (
    get_nuclear_unit_catalog,
    get_nuclear_unavailability_events,
    build_nuclear_availability_forecast_j1,
)
from preauction_weather_client import (
    get_national_preauction_weather,
)
from macro_features import add_macro_features


CACHE_DIR = "data/cache"


def _read_csv_dates(path, date_cols):

    df = pd.read_csv(path)

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce",
            )

    return df


def fetch_prices_strict(
    start,
    end,
    cache_dir=CACHE_DIR,
):

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    chunks = []
    cursor = start_dt

    while cursor < end_dt:

        next_year = pd.Timestamp(
            year=cursor.year + 1,
            month=1,
            day=1,
        )

        chunk_end = min(
            next_year,
            end_dt,
        )

        cache_file = os.path.join(
            cache_dir,
            "entsoe_prices_clean_%s_%s.csv"
            % (
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            ),
        )

        if os.path.exists(cache_file):

            print(
                "  Prices %s -> %s (cache)"
                % (
                    cursor.date(),
                    chunk_end.date(),
                )
            )

            part = pd.read_csv(
                cache_file
            )

        else:

            print(
                "  Prices %s -> %s (fetching)"
                % (
                    cursor.date(),
                    chunk_end.date(),
                )
            )

            part = get_historical_prices(
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )

            if (
                part is not None
                and not part.empty
            ):
                part.to_csv(
                    cache_file,
                    index=False,
                )

        if (
            part is not None
            and not part.empty
        ):
            chunks.append(part)

        cursor = chunk_end

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df["datetime"] = (
        pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert("Europe/Paris")
    )

    return resample_hourly(df)


def fetch_consumption_j1(
    start,
    end,
    cache_dir=CACHE_DIR,
):

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    # Delivery boundaries are expressed in Europe/Paris.
    # Determine which UTC calendar months are required, because a local
    # month can begin during the previous UTC calendar day/month.
    start_local = pd.Timestamp(start)
    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(
            "Europe/Paris"
        )
    else:
        start_local = start_local.tz_convert(
            "Europe/Paris"
        )

    end_local = pd.Timestamp(end)
    if end_local.tzinfo is None:
        end_local = end_local.tz_localize(
            "Europe/Paris"
        )
    else:
        end_local = end_local.tz_convert(
            "Europe/Paris"
        )

    start_utc_naive = (
        start_local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    end_utc_naive = (
        end_local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    month = start_utc_naive.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    chunks = []

    while month < end_utc_naive:

        next_month = (
            month
            + pd.offsets.MonthBegin(1)
        )

        cache_file = os.path.join(
            cache_dir,
            "odre_j1_%s.csv"
            % month.strftime("%Y-%m"),
        )

        if os.path.exists(cache_file):

            print(
                "  ODRÉ J-1 %s (cache)"
                % month.strftime("%Y-%m")
            )

            part = pd.read_csv(
                cache_file
            )

        else:

            print(
                "  ODRÉ J-1 %s (fetching)"
                % month.strftime("%Y-%m")
            )

            raw = get_eco2mix_data(
                dataset=DATASET_HISTORICAL,
                start_date=month.strftime(
                    "%Y-%m-%d"
                ),
                end_date=next_month.strftime(
                    "%Y-%m-%d"
                ),
                limit=10000,
            )

            if (
                raw is None
                or raw.empty
                or "prevision_j1"
                not in raw.columns
            ):
                part = pd.DataFrame()

            else:
                part = raw[
                    [
                        "date_heure",
                        "prevision_j1",
                    ]
                ].copy()

            if not part.empty:
                part.to_csv(
                    cache_file,
                    index=False,
                )

        if (
            part is not None
            and not part.empty
        ):
            chunks.append(part)

        month = next_month

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df = df.rename(
        columns={
            "date_heure": "datetime",
            "prevision_j1":
                "consumption_forecast_j1",
        }
    )

    df["datetime"] = (
        pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert("Europe/Paris")
    )

    # Monthly ODRÉ requests use [start, end). Keep this defensive
    # deduplication so pre-existing local caches created with the old
    # inclusive end boundary cannot average two revisions at midnight.
    df = (
        df.dropna(subset=["datetime"])
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df = resample_hourly(df)

    return df[
        (df["datetime"] >= start_local)
        &
        (df["datetime"] < end_local)
    ].reset_index(drop=True)


def fetch_nuclear_catalogs(
    start,
    end,
    cache_dir=CACHE_DIR,
):

    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    start_year = (
        pd.Timestamp(start).year
    )

    end_year = (
        pd.Timestamp(end)
        - pd.Timedelta(seconds=1)
    ).year

    catalogs = {}

    for year in range(
        start_year,
        end_year + 1,
    ):

        cache_file = os.path.join(
            cache_dir,
            "entsoe_nuclear_catalog_%d.csv"
            % year,
        )

        if os.path.exists(cache_file):

            print(
                "  Nuclear catalogue %d (cache)"
                % year
            )

            cat = pd.read_csv(
                cache_file
            )

        else:

            print(
                "  Nuclear catalogue %d (fetching)"
                % year
            )

            cat = get_nuclear_unit_catalog(
                year
            )

            if (
                cat is not None
                and not cat.empty
            ):
                cat.to_csv(
                    cache_file,
                    index=False,
                )

        if (
            cat is None
            or cat.empty
        ):
            raise RuntimeError(
                "Missing nuclear catalogue %d"
                % year
            )

        catalogs[year] = cat

    return catalogs


def fetch_nuclear_events(
    start,
    end,
    catalogs,
    cache_dir=CACHE_DIR,
    refresh=False,
):

    combined_catalog = pd.concat(
        catalogs.values(),
        ignore_index=True,
    )

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    month = start_dt.replace(day=1)
    chunks = []

    while month < end_dt:

        next_month = (
            month
            + pd.offsets.MonthBegin(1)
        )

        label = month.strftime(
            "%Y-%m"
        )

        cache_file = os.path.join(
            cache_dir,
            "entsoe_nuclear_a80_%s.csv"
            % label,
        )

        if (
            os.path.exists(cache_file)
            and not refresh
        ):

            print(
                "  A80 nuclear %s (cache)"
                % label
            )

            part = _read_csv_dates(
                cache_file,
                [
                    "created_doc_time",
                    "start",
                    "end",
                ],
            )

        else:

            print(
                "  A80 nuclear %s (fetching)"
                % label
            )

            part = (
                get_nuclear_unavailability_events(
                    month.strftime(
                        "%Y-%m-%d"
                    ),
                    next_month.strftime(
                        "%Y-%m-%d"
                    ),
                    catalog=combined_catalog,
                )
            )

            if (
                part is not None
                and not part.empty
            ):
                part.to_csv(
                    cache_file,
                    index=False,
                )

        if (
            part is not None
            and not part.empty
        ):
            chunks.append(part)

        month = next_month

    if not chunks:
        raise RuntimeError(
            "No A80 nuclear data available"
        )

    events = pd.concat(
        chunks,
        ignore_index=True,
    )

    for col in (
        "created_doc_time",
        "start",
        "end",
    ):
        events[col] = pd.to_datetime(
            events[col],
            utc=True,
            errors="coerce",
        )

    return events


def fetch_nuclear_preauction(
    start,
    end,
    cutoff_hour=11,
    refresh=False,
):

    catalogs = fetch_nuclear_catalogs(
        start,
        end,
    )

    events = fetch_nuclear_events(
        start,
        end,
        catalogs,
        refresh=refresh,
    )

    print(
        "  Reconstructing nuclear at D-1 %02d:00 Paris"
        % cutoff_hour
    )

    return build_nuclear_availability_forecast_j1(
        events,
        catalogs,
        start,
        end,
        cutoff_hour_local=cutoff_hour,
    )


def build_strict_dataset(
    start,
    end,
    output_path="data/dataset_strict.csv",
    nuclear_cutoff_hour=11,
    refresh_nuclear=False,
    refresh_current=False,
):

    print(
        "Building PowerCast STRICT dataset: %s -> %s"
        % (
            start,
            end,
        )
    )

    # Keep a short price-only warm-up before the strict feature period.
    # This is required to construct 24/48/72/168h price lags without
    # adding earlier observations to the actual model training sample.
    warmup_days = 8

    strict_start_local = pd.Timestamp(start)
    if strict_start_local.tzinfo is None:
        strict_start_local = strict_start_local.tz_localize(
            "Europe/Paris"
        )
    else:
        strict_start_local = strict_start_local.tz_convert(
            "Europe/Paris"
        )

    price_start_local = (
        strict_start_local
        - pd.Timedelta(days=warmup_days)
    )

    price_start = price_start_local.strftime(
        "%Y-%m-%d"
    )

    print(
        "\n[1/5] French day-ahead prices"
    )

    print(
        "  Price-lag warm-up: %d days (%s -> %s)"
        % (
            warmup_days,
            price_start,
            start,
        )
    )

    prices = fetch_prices_strict(
        price_start,
        end,
    )

    print(
        "\n[2/5] ODRÉ J-1 demand"
    )

    consumption = fetch_consumption_j1(
        start,
        end,
    )

    print(
        "\n[3/5] Nuclear A80"
    )

    nuclear = fetch_nuclear_preauction(
        start,
        end,
        cutoff_hour=nuclear_cutoff_hour,
        refresh=refresh_nuclear,
    )

    print(
        "\n[4/5] ECMWF pre-auction weather"
    )

    weather = get_national_preauction_weather(
        start,
        end,
    )

    if prices.empty:
        raise RuntimeError(
            "Price target is empty"
        )

    df = prices.copy()

    for source in (
        consumption,
        nuclear,
        weather,
    ):

        if (
            source is None
            or source.empty
        ):
            continue

        source = source.copy()

        source["datetime"] = pd.to_datetime(
            source["datetime"],
            utc=True,
            errors="coerce",
        )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce",
        )

        df = df.merge(
            source,
            on="datetime",
            how="left",
        )

    # Preserve the price-only warm-up, while enforcing the exact end
    # boundary. Strict covariates begin at `start`; warm-up rows remain
    # intentionally incomplete and therefore cannot enter model training.
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    start_local = price_start_local
    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(
            "Europe/Paris"
        )
    else:
        start_local = start_local.tz_convert(
            "Europe/Paris"
        )

    end_local = pd.Timestamp(end)
    if end_local.tzinfo is None:
        end_local = end_local.tz_localize(
            "Europe/Paris"
        )
    else:
        end_local = end_local.tz_convert(
            "Europe/Paris"
        )

    local_dt = df["datetime"].dt.tz_convert(
        "Europe/Paris"
    )

    df = df.loc[
        (local_dt >= start_local)
        &
        (local_dt < end_local)
    ].copy()

    print(
        "\n[5/5] TTF + EUA macro features"
    )

    df = add_macro_features(
        df,
        refresh_current=refresh_current,
    )

    forbidden = {
        "consommation",
        "nucleaire",
        "eolien",
        "solaire",
        "hydraulique",
        "gaz",
        "charbon",
        "fioul",
        "bioenergies",
        "taux_co2",
        "net_imports_mw",

        # Explicitly rejected strict inputs:
        "solar_forecast_mw",
        "wind_onshore_forecast_mw",
        "wind_offshore_forecast_mw",
        "wind_forecast_mw",

        "capacity_fr_to_es_mw",
        "capacity_es_to_fr_mw",
        "capacity_fr_to_ch_mw",
        "capacity_ch_to_fr_mw",
        "capacity_fr_to_it_nord_mw",
        "capacity_it_nord_to_fr_mw",
        "capacity_fr_to_gb_mw",
        "capacity_gb_to_fr_mw",
        "noncore_export_capacity_total_mw",
        "noncore_import_capacity_total_mw",
        "noncore_capacity_bias_mw",

        # Legacy Previous Runs weather:
        "temperature_c",
        "wind_speed_kmh",
        "cloud_cover_pct",
    }

    leaked = sorted(
        forbidden.intersection(
            df.columns
        )
    )

    if leaked:
        raise RuntimeError(
            "Forbidden columns entered strict dataset: %s"
            % leaked
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n✅ Saved %d rows to %s"
        % (
            len(df),
            output_path,
        )
    )

    print(
        "\nColumns (%d):"
        % len(df.columns)
    )

    for col in df.columns:
        print(
            "  - %s"
            % col
        )

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2024-03-15",
    )

    parser.add_argument(
        "--end",
        default=datetime.now().strftime(
            "%Y-%m-%d"
        ),
    )

    parser.add_argument(
        "--output",
        default="data/dataset_strict.csv",
    )

    parser.add_argument(
        "--nuclear-cutoff-hour",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--refresh-nuclear",
        action="store_true",
    )

    parser.add_argument(
        "--refresh-current",
        action="store_true",
        help=(
            "Refresh current TTF/EUA sources instead of reusing "
            "their local cache. Use this for production dataset refreshes."
        ),
    )

    args = parser.parse_args()

    build_strict_dataset(
        args.start,
        args.end,
        output_path=args.output,
        nuclear_cutoff_hour=args.nuclear_cutoff_hour,
        refresh_nuclear=args.refresh_nuclear,
        refresh_current=args.refresh_current,
    )


if __name__ == "__main__":
    main()
