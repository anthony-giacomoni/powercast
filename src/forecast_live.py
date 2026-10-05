# ============================================================
# PowerCast — live strict day-ahead inference
#
# Generates one frozen forecast for the next delivery day.
#
# Historical/model inputs follow the D-1 11:00 Europe/Paris
# information-timing policy. Live ENTSO-E A65 demand is captured
# before that cutoff as a frozen local snapshot, then reused by
# inference after the cutoff. If no valid pre-cutoff A65 snapshot
# exists, strict live inference fails closed.
#
# The forecast is saved and never overwritten unless --force
# is explicitly supplied.
# ============================================================

import argparse
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from build_dataset_strict import (
    fetch_prices_strict,
    fetch_nuclear_preauction,
)
from entsoe_client import (
    get_day_ahead_load_forecast,
)
from rte_client import (
    get_day_ahead_prices,
)
from preauction_weather_client import (
    get_national_preauction_weather,
)
from macro_features import (
    add_macro_features,
)
from strict_model import (
    LIVE_ALLOWED_FEATURE_NANS,
    engineer_strict_features,
    feature_columns,
    validate_feature_missingness,
)


PARIS_TZ = "Europe/Paris"
CUTOFF_HOUR = 11


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def delivery_cutoff(delivery):
    previous_day = delivery.date() - pd.Timedelta(days=1)
    return pd.Timestamp(
        "%s %02d:00:00" % (previous_day.isoformat(), CUTOFF_HOUR),
        tz=PARIS_TZ,
    )


def a65_snapshot_paths(output_dir, delivery):
    d = delivery.strftime("%Y-%m-%d")
    return (
        os.path.join(output_dir, "a65_snapshot_%s.csv" % d),
        os.path.join(output_dir, "a65_snapshot_%s.json" % d),
    )


def expected_delivery_utc_index(delivery):
    start = pd.Timestamp(
        delivery.date(),
        tz=PARIS_TZ,
    ).tz_convert("UTC")
    end = pd.Timestamp(
        delivery.date() + pd.Timedelta(days=1),
        tz=PARIS_TZ,
    ).tz_convert("UTC")
    return pd.date_range(
        start,
        end,
        freq="1h",
        inclusive="left",
    )


def validate_a65_snapshot_frame(frame, delivery):
    required = {
        "datetime",
        "consumption_forecast_j1",
    }

    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or not required.issubset(frame.columns)
    ):
        raise RuntimeError(
            "A65 snapshot is missing required data; refusing strict live forecast."
        )

    out = frame[
        ["datetime", "consumption_forecast_j1"]
    ].copy()

    out["datetime"] = pd.to_datetime(
        out["datetime"],
        utc=True,
        errors="coerce",
    )

    out["consumption_forecast_j1"] = pd.to_numeric(
        out["consumption_forecast_j1"],
        errors="coerce",
    )

    if out.isna().any().any():
        raise RuntimeError(
            "A65 snapshot contains malformed values; refusing strict live forecast."
        )

    out = out.sort_values("datetime").reset_index(drop=True)

    if out["datetime"].duplicated().any():
        raise RuntimeError(
            "A65 snapshot contains duplicate timestamps; refusing strict live forecast."
        )

    if not pd.DatetimeIndex(
        out["datetime"]
    ).equals(
        expected_delivery_utc_index(delivery)
    ):
        raise RuntimeError(
            "A65 snapshot does not match the complete delivery-day grid; "
            "refusing strict live forecast."
        )

    return out


def capture_a65_snapshot(
    delivery,
    output_dir,
    now_local=None,
):
    if now_local is None:
        now_local = pd.Timestamp.now(
            tz=PARIS_TZ
        )

    cutoff = delivery_cutoff(delivery)

    if now_local >= cutoff:
        raise RuntimeError(
            "A65 snapshot capture is only allowed before the "
            "D-1 11:00 Europe/Paris cutoff."
        )

    demand = get_day_ahead_load_forecast(
        delivery.strftime("%Y-%m-%d")
    )

    source_metadata = dict(demand.attrs)

    demand = validate_a65_snapshot_frame(
        demand,
        delivery,
    )

    cutoff_utc = cutoff.tz_convert("UTC")

    retrieved = pd.to_datetime(
        source_metadata.get("retrieved_at_utc"),
        utc=True,
        errors="coerce",
    )

    created = pd.to_datetime(
        source_metadata.get(
            "document_created_at_utc"
        ),
        utc=True,
        errors="coerce",
    )

    if (
        pd.isna(retrieved)
        or retrieved > cutoff_utc
    ):
        raise RuntimeError(
            "A65 snapshot retrieval timestamp is not pre-cutoff; refusing capture."
        )

    if (
        pd.notna(created)
        and created > cutoff_utc
    ):
        raise RuntimeError(
            "A65 document was created after the D-1 11:00 cutoff; refusing capture."
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    csv_path, metadata_path = (
        a65_snapshot_paths(
            output_dir,
            delivery,
        )
    )

    atomic_write_csv(
        demand,
        csv_path,
    )

    payload = {
        "status":
            "frozen_pre_cutoff",
        "delivery_date":
            delivery.strftime("%Y-%m-%d"),
        "cutoff_local":
            cutoff.isoformat(),
        "cutoff_utc":
            cutoff_utc.isoformat(),
        "snapshot_csv_sha256":
            sha256_file(csv_path),
        "hours":
            int(len(demand)),
        "source":
            source_metadata.get("source"),
        "revision_policy":
            "frozen_pre_cutoff_snapshot",
        "document_mrid":
            source_metadata.get(
                "document_mrid"
            ),
        "document_revision":
            source_metadata.get(
                "document_revision"
            ),
        "document_created_at_utc":
            source_metadata.get(
                "document_created_at_utc"
            ),
        "retrieved_at_utc":
            source_metadata.get(
                "retrieved_at_utc"
            ),
        "exact_asof_11_reconstructed":
            False,
    }

    atomic_write_json(
        payload,
        metadata_path,
    )

    return demand, payload


def load_a65_snapshot(
    delivery,
    output_dir,
):
    csv_path, metadata_path = (
        a65_snapshot_paths(
            output_dir,
            delivery,
        )
    )

    if not (
        os.path.isfile(csv_path)
        and os.path.isfile(metadata_path)
    ):
        raise RuntimeError(
            "No frozen pre-cutoff A65 snapshot exists for %s. "
            "Run --snapshot-a65 before D-1 11:00 Europe/Paris."
            % delivery.strftime("%Y-%m-%d")
        )

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception:
        raise RuntimeError(
            "A65 snapshot metadata is unreadable; refusing strict live forecast."
        ) from None

    delivery_str = delivery.strftime(
        "%Y-%m-%d"
    )

    if (
        metadata.get("delivery_date")
        != delivery_str
    ):
        raise RuntimeError(
            "A65 snapshot metadata has the wrong delivery day; refusing strict live forecast."
        )

    expected_sha = metadata.get(
        "snapshot_csv_sha256"
    )

    if (
        not expected_sha
        or sha256_file(csv_path)
        != expected_sha
    ):
        raise RuntimeError(
            "A65 snapshot CSV hash does not match metadata; refusing strict live forecast."
        )

    cutoff_utc = (
        delivery_cutoff(delivery)
        .tz_convert("UTC")
    )

    retrieved = pd.to_datetime(
        metadata.get("retrieved_at_utc"),
        utc=True,
        errors="coerce",
    )

    created = pd.to_datetime(
        metadata.get(
            "document_created_at_utc"
        ),
        utc=True,
        errors="coerce",
    )

    if (
        pd.isna(retrieved)
        or retrieved > cutoff_utc
    ):
        raise RuntimeError(
            "A65 snapshot retrieval timestamp is not pre-cutoff; refusing strict live forecast."
        )

    if (
        pd.notna(created)
        and created > cutoff_utc
    ):
        raise RuntimeError(
            "A65 snapshot document timestamp is post-cutoff; refusing strict live forecast."
        )

    frame = pd.read_csv(
        csv_path
    )

    frame = validate_a65_snapshot_frame(
        frame,
        delivery,
    )

    return frame, metadata


def normalize_datetime(df):
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    return (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .reset_index(drop=True)
    )



def atomic_write_csv(df, path):
    tmp = (
        "%s.tmp.%d"
        % (
            path,
            os.getpid(),
        )
    )

    try:
        df.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(payload, path):
    tmp = (
        "%s.tmp.%d"
        % (
            path,
            os.getpid(),
        )
    )

    try:
        with open(
            tmp,
            "w",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

        os.replace(
            tmp,
            path,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def frozen_bundle_complete(
    forecast_path,
    inputs_path,
    metadata_path,
):
    """
    A frozen live forecast is complete only when all three
    auditable artifacts exist.

    forecast CSV is written last and therefore acts as the
    completion marker for newly generated bundles.
    """
    return all(
        os.path.isfile(path)
        for path in (
            forecast_path,
            inputs_path,
            metadata_path,
        )
    )


def remove_forecast_commit_marker(forecast_path):
    """
    Remove an existing forecast completion marker before any
    rebuild/forced recalculation starts.

    This prevents a crash after rewriting inputs/metadata but
    before rewriting forecast.csv from making a stale forecast
    look like a complete new bundle on the next run.
    """
    if not os.path.exists(forecast_path):
        return False

    os.remove(forecast_path)
    return True


def choose_delivery_date(now_local):
    """
    Select tomorrow's civil delivery day after the 11:00 Paris cutoff.

    DST-safe: civil dates are constructed explicitly rather than by
    adding 24 elapsed hours to timezone-aware timestamps.
    """
    cutoff = pd.Timestamp(
        "%s %02d:00:00"
        % (
            now_local.date().isoformat(),
            CUTOFF_HOUR,
        ),
        tz=PARIS_TZ,
    )

    if now_local < cutoff:
        raise RuntimeError(
            "It is before 11:00 Europe/Paris. "
            "Do not generate a new forecast yet. "
            "The active forecast should still be yesterday's "
            "frozen forecast for today."
        )

    next_day = (
        now_local.date()
        + pd.Timedelta(days=1)
    )

    return pd.Timestamp(
        next_day,
        tz=PARIS_TZ,
    )


def make_target_grid(delivery):
    """
    Build the complete Europe/Paris civil delivery day.

    Normal day     -> 24 hourly periods
    Spring DST day -> 23 hourly periods
    Autumn DST day -> 25 hourly periods
    """
    delivery_day = delivery.date()
    next_day = delivery_day + pd.Timedelta(days=1)

    start = pd.Timestamp(
        delivery_day,
        tz=PARIS_TZ,
    )

    end = pd.Timestamp(
        next_day,
        tz=PARIS_TZ,
    )

    local_hours = pd.date_range(
        start=start,
        end=end,
        freq="1h",
        inclusive="left",
    )

    return pd.DataFrame({
        "datetime": local_hours.tz_convert("UTC")
    })


def fill_previous_day_price_gap_from_rte(
    prices,
    delivery,
):
    """
    Repair only genuinely missing D-1 hourly prices with RTE.

    A complete D-1 ENTSO-E series returns immediately without any
    RTE authentication/network call. If RTE is unavailable, the
    expected D-1 grid is retained with NaNs so the documented live
    missing-value contract can mark inference as degraded.
    """
    previous_day = (
        delivery.date()
        - pd.Timedelta(days=1)
    )

    start = pd.Timestamp(
        previous_day,
        tz=PARIS_TZ,
    )
    end = pd.Timestamp(
        previous_day + pd.Timedelta(days=1),
        tz=PARIS_TZ,
    )

    previous_day_utc_start = start.tz_convert("UTC")
    previous_day_utc_end = end.tz_convert("UTC")

    expected = pd.DataFrame({
        "datetime": pd.date_range(
            start=previous_day_utc_start,
            end=previous_day_utc_end,
            freq="1h",
            inclusive="left",
        )
    })

    existing = prices[[
        "datetime",
        "price_eur_mwh",
    ]].copy()

    existing["datetime"] = pd.to_datetime(
        existing["datetime"],
        utc=True,
        errors="coerce",
    )

    d1 = expected.merge(
        existing,
        on="datetime",
        how="left",
    )

    missing_before = int(
        d1["price_eur_mwh"].isna().sum()
    )

    if missing_before == 0:
        print(
            "  RTE fallback: D-1 ENTSO-E prices complete; no call needed"
        )
        return prices

    raw = get_day_ahead_prices(
        start.isoformat(),
        end.isoformat(),
        raise_errors=False,
    )

    if not raw:
        print(
            "  ⚠️ RTE fallback unavailable; keeping %d D-1 gap(s) as NaN"
            % missing_before
        )
        outside = existing[
            (existing["datetime"] < previous_day_utc_start)
            | (existing["datetime"] >= previous_day_utc_end)
        ]
        return (
            pd.concat([outside, d1], ignore_index=True)
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )

    rows = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        for point in block.get("values", []):
            if not isinstance(point, dict):
                continue
            dt = pd.to_datetime(
                point.get("start_date"),
                utc=True,
                errors="coerce",
            )
            value = pd.to_numeric(
                point.get("price"),
                errors="coerce",
            )
            if pd.notna(dt) and pd.notna(value):
                rows.append({
                    "datetime": dt,
                    "rte_price": float(value),
                })

    if rows:
        rte = (
            pd.DataFrame(rows)
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .set_index("datetime")["rte_price"]
            .resample("1h")
            .mean()
            .reset_index()
        )

        rte = rte[
            (rte["datetime"] >= previous_day_utc_start)
            & (rte["datetime"] < previous_day_utc_end)
        ]

        d1 = d1.merge(
            rte,
            on="datetime",
            how="left",
        )

        fill_mask = (
            d1["price_eur_mwh"].isna()
            & d1["rte_price"].notna()
        )
        filled = int(fill_mask.sum())
        d1.loc[fill_mask, "price_eur_mwh"] = d1.loc[
            fill_mask,
            "rte_price",
        ]
        d1 = d1.drop(columns=["rte_price"])
    else:
        filled = 0

    remaining = int(
        d1["price_eur_mwh"].isna().sum()
    )

    if filled:
        print(
            "  ✅ RTE hourly fallback filled %d D-1 price gap(s)"
            % filled
        )
    if remaining:
        print(
            "  ⚠️ %d D-1 price gap(s) remain after RTE fallback"
            % remaining
        )

    outside = existing[
        (existing["datetime"] < previous_day_utc_start)
        | (existing["datetime"] >= previous_day_utc_end)
    ]

    return (
        pd.concat([outside, d1], ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

def build_live_inputs(delivery, output_dir="data/live"):
    delivery_str = delivery.strftime(
        "%Y-%m-%d"
    )

    delivery_end = pd.Timestamp(
        delivery.date() + pd.Timedelta(days=1),
        tz=PARIS_TZ,
    )

    end_str = delivery_end.strftime(
        "%Y-%m-%d"
    )

    print()
    print(
        "[1/5] Historical price lags"
    )

    price_start = pd.Timestamp(
        delivery.date() - pd.Timedelta(days=8),
        tz=PARIS_TZ,
    ).strftime("%Y-%m-%d")

    prices = fetch_prices_strict(
        price_start,
        delivery_str,
    )

    required_price_columns = {
        "datetime",
        "price_eur_mwh",
    }

    if (
        not isinstance(prices, pd.DataFrame)
        or prices.empty
        or not required_price_columns.issubset(
            prices.columns
        )
    ):
        raise RuntimeError(
            "Historical day-ahead prices are unavailable or malformed; "
            "refusing live forecast."
        )

    prices = normalize_datetime(
        prices
    )

    if prices.empty:
        raise RuntimeError(
            "Historical day-ahead prices contain no valid timestamps; "
            "refusing live forecast."
        )

    prices = prices[
        [
            "datetime",
            "price_eur_mwh",
        ]
    ].copy()

    prices = fill_previous_day_price_gap_from_rte(
        prices,
        delivery,
    )


    print()
    print(
        "[2/5] Frozen pre-cutoff ENTSO-E A65 load forecast"
    )

    demand, demand_metadata = (
        load_a65_snapshot(
            delivery,
            output_dir,
        )
    )

    demand = normalize_datetime(
        demand
    )


    print()
    print(
        "[3/5] Nuclear A80 at 11:00 cutoff"
    )

    # Refresh at inference time so documents published shortly
    # before 11:00 are not missed because of an earlier cache.
    nuclear = fetch_nuclear_preauction(
        delivery_str,
        end_str,
        cutoff_hour=CUTOFF_HOUR,
        refresh=True,
    )

    nuclear = normalize_datetime(
        nuclear
    )


    print()
    print(
        "[4/5] Fixed ECMWF pre-auction weather"
    )

    weather = get_national_preauction_weather(
        delivery_str,
        end_str,
    )

    weather = normalize_datetime(
        weather
    )


    print()
    print(
        "[5/5] TTF + EUA macro features"
    )

    target = make_target_grid(
        delivery
    )

    for source in [
        demand,
        nuclear,
        weather,
    ]:
        target = target.merge(
            source,
            on="datetime",
            how="left",
        )

    target["price_eur_mwh"] = np.nan

    # Price-only warm-up is sufficient to construct
    # 24/48/72/168h lags and previous-day mean.
    #
    # Historical-price APIs can sometimes return points from the
    # delivery day itself. Those prices must never enter live inference:
    # D is precisely the day being forecast. Keep history strictly
    # before local midnight at the start of D.
    delivery_start_utc = (
        pd.Timestamp(delivery.date())
        .tz_localize(PARIS_TZ)
        .tz_convert("UTC")
    )

    prices = prices[
        prices["datetime"] < delivery_start_utc
    ].copy()

    raw = pd.concat(
        [
            prices,
            target,
        ],
        ignore_index=True,
        sort=False,
    )

    raw = normalize_datetime(
        raw
    )

    raw = add_macro_features(
        raw,
        refresh_current=True,
    )

    return raw, target, demand_metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="data/model_strict_production.pkl",
    )

    parser.add_argument(
        "--output-dir",
        default="data/live",
    )

    parser.add_argument(
        "--snapshot-a65",
        action="store_true",
        help=(
            "Capture tomorrow's ENTSO-E A65 demand forecast before "
            "the D-1 11:00 Europe/Paris cutoff, then exit."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen forecast.",
    )

    args = parser.parse_args()

    now_local = pd.Timestamp.now(
        tz=PARIS_TZ
    )

    if args.snapshot_a65:
        snapshot_delivery = pd.Timestamp(
            now_local.date()
            + pd.Timedelta(days=1),
            tz=PARIS_TZ,
        )

        demand, _ = capture_a65_snapshot(
            snapshot_delivery,
            args.output_dir,
            now_local=now_local,
        )

        csv_path, metadata_path = (
            a65_snapshot_paths(
                args.output_dir,
                snapshot_delivery,
            )
        )

        print(
            "✅ Frozen pre-cutoff A65 snapshot created"
        )
        print(
            "Delivery:",
            snapshot_delivery.strftime(
                "%Y-%m-%d"
            ),
        )
        print(
            "Hours:",
            len(demand),
        )
        print(
            "Snapshot:",
            csv_path,
        )
        print(
            "Metadata:",
            metadata_path,
        )
        return

    delivery = choose_delivery_date(
        now_local
    )

    delivery_str = delivery.strftime(
        "%Y-%m-%d"
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    forecast_path = os.path.join(
        args.output_dir,
        "forecast_%s.csv"
        % delivery_str,
    )

    metadata_path = os.path.join(
        args.output_dir,
        "forecast_%s.json"
        % delivery_str,
    )

    inputs_path = os.path.join(
        args.output_dir,
        "inputs_%s.csv"
        % delivery_str,
    )

    print(
        "PowerCast LIVE strict forecast"
    )

    print(
        "Generated at:",
        now_local,
    )

    print(
        "Delivery day:",
        delivery_str,
    )

    if not args.force:
        if frozen_bundle_complete(
            forecast_path,
            inputs_path,
            metadata_path,
        ):
            print()
            print(
                "🔒 Complete frozen forecast bundle already exists:"
            )
            print(
                forecast_path
            )
            print(
                "Nothing was recalculated."
            )
            return

        existing_parts = [
            path
            for path in (
                forecast_path,
                inputs_path,
                metadata_path,
            )
            if os.path.exists(path)
        ]

        if existing_parts:
            print()
            print(
                "⚠️ Incomplete frozen bundle found; "
                "rebuilding the full bundle."
            )

            for path in existing_parts:
                print(
                    "  existing:",
                    path,
                )

    # From this point onward we are intentionally recalculating
    # (partial rebuild or --force). Remove any old forecast first
    # so it cannot masquerade as the completion marker for new
    # inputs/metadata if execution stops before the final write.
    if remove_forecast_commit_marker(
        forecast_path
    ):
        print(
            "Removed previous forecast completion marker before rebuild:",
            forecast_path,
        )

    artifact = joblib.load(
        args.model
    )
    production_model_sha256 = sha256_file(
        args.model
    )

    model = artifact["model"]
    model_features = artifact["features"]

    print()
    print(
        "Production model trained through:",
        artifact.get(
            "trained_through"
        ),
    )

    raw, raw_target, demand_metadata = build_live_inputs(
        delivery,
        output_dir=args.output_dir,
    )

    engineered = engineer_strict_features(
        raw
    )

    current_features = feature_columns(
        engineered
    )

    if current_features != model_features:
        print()
        print(
            "MODEL FEATURES:"
        )
        print(model_features)

        print()
        print(
            "LIVE FEATURES:"
        )
        print(current_features)

        raise RuntimeError(
            "Live feature order does not exactly match "
            "the production model."
        )

    local_dt = (
        engineered["datetime"]
        .dt.tz_convert(PARIS_TZ)
    )

    target_mask = (
        local_dt.dt.date
        == delivery.date()
    )

    inference = (
        engineered.loc[
            target_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    expected_hours = len(
        make_target_grid(delivery)
    )

    if len(inference) != expected_hours:
        raise RuntimeError(
            "Expected %d delivery hours, got %d"
            % (
                expected_hours,
                len(inference),
            )
        )

    missing = validate_feature_missingness(
        inference,
        model_features,
        LIVE_ALLOWED_FEATURE_NANS,
        "live inference",
    )

    if len(missing):
        print()
        print(
            "⚠️ Source data gaps handled natively by LightGBM:"
        )
        print(
            missing.to_string()
        )

    prediction = model.predict(
        inference[model_features]
    )

    generated_at = pd.Timestamp.now(
        tz=PARIS_TZ
    )

    forecast = pd.DataFrame({
        "datetime": inference[
            "datetime"
        ],
        "predicted_price_eur_mwh":
            prediction,
        "delivery_date":
            delivery_str,
        "forecast_generated_at":
            generated_at.isoformat(),
        "forecast_status":
            "frozen",
    })

    # Preserve the exact inference snapshot first.
    atomic_write_csv(
        inference,
        inputs_path,
    )

    metadata = {
        "delivery_date":
            delivery_str,
        "generated_at":
            generated_at.isoformat(),
        "cutoff_timezone":
            PARIS_TZ,
        "cutoff_hour":
            CUTOFF_HOUR,
        "status":
            "frozen",
        "hours":
            int(len(forecast)),
        "model_trained_through":
            artifact.get(
                "trained_through"
            ),
        "model_train_rows":
            artifact.get(
                "train_rows"
            ),
        "production_model_sha256":
            production_model_sha256,
        "production_dataset_sha256":
            artifact.get(
                "dataset_sha256"
            ),
        "production_dataset_source_from":
            artifact.get(
                "dataset_source_from"
            ),
        "production_dataset_source_through":
            artifact.get(
                "dataset_source_through"
            ),
        "demand_forecast_source":
            "ENTSO-E A65/A01",
        "demand_forecast_revision_policy":
            demand_metadata.get(
                "revision_policy"
            ),
        "demand_forecast_snapshot_sha256":
            demand_metadata.get(
                "snapshot_csv_sha256"
            ),
        "demand_forecast_pre_cutoff_snapshot":
            True,
        "demand_forecast_document_mrid":
            demand_metadata.get(
                "document_mrid"
            ),
        "demand_forecast_document_revision":
            demand_metadata.get(
                "document_revision"
            ),
        "demand_forecast_document_created_at_utc":
            demand_metadata.get(
                "document_created_at_utc"
            ),
        "demand_forecast_retrieved_at_utc":
            demand_metadata.get(
                "retrieved_at_utc"
            ),
        "demand_forecast_exact_asof_11_reconstructed":
            bool(
                demand_metadata.get(
                    "exact_asof_11_reconstructed",
                    False,
                )
            ),
        "train_live_demand_source_shift":
            "historical ODRE J-1 -> live ENTSO-E A65",
        "live_allowed_nan_features":
            sorted(
                LIVE_ALLOWED_FEATURE_NANS
            ),
        "feature_count":
            int(len(model_features)),
        "data_quality_status":
            (
                "degraded"
                if len(missing)
                else "complete"
            ),
        "missing_feature_values":
            {
                str(k): int(v)
                for k, v in missing.items()
            },
    }

    # Metadata is written before the public forecast completion
    # marker. If execution stops here, the next run sees an
    # incomplete bundle and rebuilds it.
    atomic_write_json(
        metadata,
        metadata_path,
    )

    # Forecast is deliberately written LAST and acts as the
    # completion marker for the frozen auditable bundle.
    atomic_write_csv(
        forecast,
        forecast_path,
    )

    print()
    print(
        "===================================="
    )

    print(
        "✅ FROZEN POWERCAST FORECAST CREATED"
    )

    print(
        "===================================="
    )

    print(
        "Delivery:",
        delivery_str,
    )

    print(
        "Hours:",
        len(forecast),
    )

    print(
        "Mean predicted price: %.2f EUR/MWh"
        % forecast[
            "predicted_price_eur_mwh"
        ].mean()
    )

    print(
        "Min predicted price: %.2f EUR/MWh"
        % forecast[
            "predicted_price_eur_mwh"
        ].min()
    )

    print(
        "Max predicted price: %.2f EUR/MWh"
        % forecast[
            "predicted_price_eur_mwh"
        ].max()
    )

    print()
    print(
        "Forecast:",
        forecast_path,
    )

    print(
        "Inputs snapshot:",
        inputs_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )

    print()
    print(
        "🔒 Refreshing Streamlit will NOT "
        "recalculate this forecast."
    )


if __name__ == "__main__":
    main()
