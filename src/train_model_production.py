# ============================================================
# PowerCast — strict production model trainer
#
# Trains the final LightGBM model on ALL admissible historical
# observations with a known target.
#
# This artifact is deliberately separate from model_strict.pkl:
#
#   model_strict.pkl
#       -> chronological evaluation / benchmark
#
#   model_strict_production.pkl
#       -> live day-ahead inference
#
# Python 3.8 compatible.
# ============================================================

import argparse
import hashlib
import platform
from importlib import metadata as importlib_metadata
import json
import os

import joblib
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from strict_model import (
    TRAIN_ALLOWED_FEATURE_NANS,
    engineer_strict_features,
    feature_columns,
    validate_feature_missingness,
)


MACRO_FEATURES = [
    "ttf_front_month_eur_mwh",
    "ttf_change_5obs_pct",
    "eua_auction_eur_tco2",
    "eua_change_5auctions_pct",
    "ccgt_cost_proxy_eur_mwh",
    "ocgt_cost_proxy_eur_mwh",
]

REQUIRED_STRICT = [
    "consumption_forecast_j1",
    "nuclear_available_forecast_mw",
    "preauction_temperature_c",
    "preauction_wind_speed_10m_kmh",
    "preauction_wind_speed_100m_kmh",
    "preauction_cloud_cover_pct",
    "preauction_shortwave_radiation_wm2",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()

def main():
    if lgb is None:
        raise RuntimeError(
            "lightgbm is not installed"
        )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Explicit refreshed strict dataset snapshot used "
            "for production training. The tracked "
            "data/dataset_strict.csv is the static benchmark "
            "snapshot and is not implicitly reused here."
        ),
    )

    parser.add_argument(
        "--output",
        default="data/model_strict_production.pkl",
    )

    args = parser.parse_args()

    dataset_sha256 = sha256_file(
        args.data
    )

    dataset_bytes = os.path.getsize(
        args.data
    )

    generated_at_utc = (
        pd.Timestamp.now(tz="UTC")
        .isoformat()
    )

    print("PowerCast strict PRODUCTION model")
    print("Loading %s..." % args.data)

    raw = pd.read_csv(args.data)

    print(
        "Loaded %d raw rows"
        % len(raw)
    )

    missing = [
        c
        for c in REQUIRED_STRICT + MACRO_FEATURES
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            "Missing production input columns: %s"
            % missing
        )

    df = engineer_strict_features(
        raw
    )

    features = feature_columns(df)

    source_from = df["datetime"].min()
    source_through = df["datetime"].max()

    complete_features = [
        c
        for c in features
        if c not in TRAIN_ALLOWED_FEATURE_NANS
    ]

    train = df.dropna(
        subset=[
            "price_eur_mwh"
        ] + complete_features
    ).copy()

    if len(train) < 1000:
        raise RuntimeError(
            "Insufficient production training rows"
        )

    validate_feature_missingness(
        train,
        features,
        TRAIN_ALLOWED_FEATURE_NANS,
        "production training",
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=324,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["price_eur_mwh"],
    )

    trained_from = (
        train["datetime"].min()
    )

    trained_through = (
        train["datetime"].max()
    )

    artifact = {
        "model": model,
        "features": features,
        "train_rows": int(len(train)),
        "trained_from": str(trained_from),
        "trained_through": str(trained_through),
        "purpose": "live_strict_preauction_inference",
        "dataset_sha256": dataset_sha256,
        "dataset_bytes": int(dataset_bytes),
        "dataset_raw_rows": int(len(raw)),
        "dataset_source_from": str(source_from),
        "dataset_source_through": str(source_through),
        "generated_at_utc": generated_at_utc,
    }

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        args.output,
    )

    model_sha256 = sha256_file(
        args.output
    )

    metadata = {
        "purpose": "live_strict_preauction_inference",
        "generated_at_utc": generated_at_utc,
        "train_rows": int(len(train)),
        "feature_count": int(len(features)),
        "trained_from": str(trained_from),
        "trained_through": str(trained_through),
        "features": features,
        "dataset": {
            "path": args.data,
            "sha256": dataset_sha256,
            "bytes": int(dataset_bytes),
            "raw_rows": int(len(raw)),
            "source_from": str(source_from),
            "source_through": str(source_through),
        },
        "model_sha256": model_sha256,
        "missing_value_contract": {
            "training_allowed_nan_features": sorted(
                TRAIN_ALLOWED_FEATURE_NANS
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "joblib": importlib_metadata.version(
                "joblib"
            ),
        },
    }

    metadata_path = (
        os.path.splitext(args.output)[0]
        + "_metadata.json"
    )

    with open(
        metadata_path,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "Production rows: %d"
        % len(train)
    )

    print(
        "Features: %d"
        % len(features)
    )

    print(
        "Training period: %s -> %s"
        % (
            trained_from,
            trained_through,
        )
    )

    print()
    print(
        "✅ Saved %s"
        % args.output
    )


if __name__ == "__main__":
    main()
