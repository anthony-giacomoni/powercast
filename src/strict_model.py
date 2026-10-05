# ============================================================
# PowerCast — strict pre-auction model core
#
# Feature engineering, feature selection, training, evaluation
# and artifact persistence for the final strict model.
#
# Legacy/non-strict inputs such as A69 renewable forecasts,
# A61 transfer capacities and realised weather are explicitly
# excluded from the feature set.
# ============================================================

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


FORBIDDEN_INPUT_COLUMNS = {
    # ENTSO-E A69 renewable forecasts.
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "wind_forecast_mw",
    "renewables_forecast_mw",
    "renewables_to_demand_ratio",
    "residual_load_forecast_mw",
    "dispatchable_gap_forecast_mw",

    # Legacy realised / rolling weather.
    "temperature_c",
    "wind_speed_kmh",
    "cloud_cover_pct",

    # ENTSO-E A61 transfer-capacity experiment.
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
    "noncore_export_capacity_to_demand_ratio",
    "noncore_import_capacity_to_demand_ratio",
    "system_surplus_proxy_mw",
    "surplus_after_noncore_export_capacity_mw",
}

def _fr_holiday_dates(years):
    """
    Return canonical French holiday dates.

    holidays is a mandatory dependency of the frozen benchmark:
    silently replacing this feature with zeros would produce a
    different model while looking superficially successful.
    """
    try:
        import holidays
    except ImportError as exc:
        raise RuntimeError(
            "holidays==0.58 is required for the canonical "
            "PowerCast strict feature pipeline"
        ) from exc

    fr_holidays = holidays.France(
        years=years
    )

    return {
        str(d)
        for d in fr_holidays.keys()
    }


def engineer_strict_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    local_dt = df["datetime"].dt.tz_convert("Europe/Paris")
    df["hour"] = local_dt.dt.hour
    df["day_of_week"] = local_dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = local_dt.dt.month

    if "price_eur_mwh" in df.columns:
        df["price_lag_24h"] = df["price_eur_mwh"].shift(24)
        df["price_lag_48h"] = df["price_eur_mwh"].shift(48)
        df["price_lag_72h"] = df["price_eur_mwh"].shift(72)
        df["price_lag_168h"] = df["price_eur_mwh"].shift(168)
        delivery_date = local_dt.dt.date
        daily_mean = df.groupby(delivery_date)["price_eur_mwh"].mean().to_dict()
        local_naive = local_dt.dt.tz_localize(None)
        prev_dates = (local_naive.dt.normalize() - pd.Timedelta(days=1)).dt.date
        df["previous_day_price_mean"] = prev_dates.map(daily_mean)

    # Strict weather features only — all based on the same ECMWF 00Z D-1 run.
    t = "preauction_temperature_c"
    w100 = "preauction_wind_speed_100m_kmh"
    rad = "preauction_shortwave_radiation_wm2"
    comfort = 18.3
    if t in df.columns:
        df["preauction_hdd"] = (comfort - df[t]).clip(lower=0)
        df["preauction_cdd"] = (df[t] - comfort).clip(lower=0)

    if "consumption_forecast_j1" in df.columns:
        demand = df["consumption_forecast_j1"] + 1e-6

        # ------------------------------------------------------------
        # CLEAN NUCLEAR SPLIT
        #
        # Nuclear fleet nominal capacity is valid independently from
        # historical A80 outage availability.
        #
        # ENTSO-E A80 point-in-time outage history before the platform
        # migration cannot be reconstructed reliably through the
        # current API. Therefore:
        #
        #   - nominal nuclear context is kept for the full history;
        #   - A80 outage information is NaN before 2025-11-12;
        #   - from 2025-11-12 onward, the pre-auction A80 signal is used.
        #
        # LightGBM handles the historical NaN values natively.
        # ------------------------------------------------------------

        nominal = None

        if "nuclear_nominal_capacity_mw" in df.columns:
            nominal = pd.to_numeric(
                df["nuclear_nominal_capacity_mw"],
                errors="coerce",
            )

        elif (
            "nuclear_available_forecast_mw" in df.columns
            and "nuclear_unavailable_forecast_mw" in df.columns
        ):
            nominal = (
                pd.to_numeric(
                    df["nuclear_available_forecast_mw"],
                    errors="coerce",
                )
                +
                pd.to_numeric(
                    df["nuclear_unavailable_forecast_mw"],
                    errors="coerce",
                )
            )

        if nominal is not None:
            df["nuclear_nominal_capacity_safe"] = nominal

            df["nuclear_nominal_to_demand_ratio_safe"] = (
                nominal / demand
            )

            df["nuclear_nominal_demand_margin_safe"] = (
                nominal - df["consumption_forecast_j1"]
            )

        if "nuclear_unavailable_forecast_mw" in df.columns:
            df["nuclear_a80_unavailable_safe"] = pd.to_numeric(
                df["nuclear_unavailable_forecast_mw"],
                errors="coerce",
            )

            reliable_a80_start = pd.Timestamp(
                "2025-11-12",
                tz="Europe/Paris",
            )

            df.loc[
                local_dt < reliable_a80_start,
                "nuclear_a80_unavailable_safe",
            ] = np.nan

        if rad in df.columns:
            # Weather-based solar-pressure proxy, not a production forecast.
            df["solar_weather_pressure"] = df[rad] / demand
        if w100 in df.columns:
            # Wind generation is approximately cubic in wind speed below rated speed.
            wind_ms = df[w100] / 3.6
            df["wind100_cubic_proxy"] = wind_ms.clip(lower=0, upper=25) ** 3
            df["wind_weather_pressure"] = df["wind100_cubic_proxy"] / demand

    years = range(
        local_dt.dt.year.min(),
        local_dt.dt.year.max() + 1,
    )

    holiday_dates = _fr_holiday_dates(
        years
    )

    df["is_holiday"] = (
        local_dt.dt.date
        .astype(str)
        .isin(holiday_dates)
        .astype(int)
    )

    if "preauction_hdd" in df.columns:
        df["preauction_hdd_x_weekday"] = df["preauction_hdd"] * (1 - df["is_weekend"])

    return df


# Canonical strict pre-auction model schema.
#
# The exact names AND order are part of model reproducibility:
# LightGBM uses colsample_bytree < 1, so silently reordering columns
# can change the fitted model even when the feature set is identical.
#
# Never derive this list dynamically from "all numeric columns".
STRICT_FEATURES = [
    "consumption_forecast_j1",
    "preauction_temperature_c",
    "preauction_wind_speed_10m_kmh",
    "preauction_wind_speed_100m_kmh",
    "preauction_cloud_cover_pct",
    "preauction_shortwave_radiation_wm2",
    "ttf_front_month_eur_mwh",
    "ttf_change_5obs_pct",
    "eua_auction_eur_tco2",
    "eua_change_5auctions_pct",
    "ccgt_cost_proxy_eur_mwh",
    "ocgt_cost_proxy_eur_mwh",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_72h",
    "price_lag_168h",
    "previous_day_price_mean",
    "preauction_hdd",
    "preauction_cdd",
    "solar_weather_pressure",
    "wind100_cubic_proxy",
    "wind_weather_pressure",
    "is_holiday",
    "preauction_hdd_x_weekday",
    "nuclear_nominal_capacity_safe",
    "nuclear_nominal_to_demand_ratio_safe",
    "nuclear_nominal_demand_margin_safe",
    "nuclear_a80_unavailable_safe",
]



# Missing-value contracts are deliberately different in training and live
# inference. Historical benchmark/production training requires complete
# price lags, while live inference may retain genuine upstream A44 gaps
# rather than fabricate historical wholesale prices.
TRAIN_ALLOWED_FEATURE_NANS = frozenset({
    "nuclear_a80_unavailable_safe",
})

LIVE_ALLOWED_FEATURE_NANS = frozenset({
    "nuclear_a80_unavailable_safe",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_72h",
    "price_lag_168h",
})


def validate_feature_missingness(
    df,
    features,
    allowed_missing,
    context,
):
    counts = (
        df[features]
        .isna()
        .sum()
    )

    counts = counts[
        counts > 0
    ]

    forbidden = [
        c
        for c in counts.index
        if c not in allowed_missing
    ]

    if forbidden:
        details = ", ".join(
            "%s=%d" % (
                c,
                int(counts[c]),
            )
            for c in forbidden
        )

        raise RuntimeError(
            "%s has missing required feature values: %s"
            % (
                context,
                details,
            )
        )

    return counts

def feature_columns(df):
    """
    Return the exact canonical strict feature schema.

    Extra numeric columns are deliberately ignored so that adding a new
    dataset column can never silently alter the model. Missing or non-
    numeric canonical inputs fail loudly.
    """
    missing = [
        c for c in STRICT_FEATURES
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing canonical strict feature columns: %s"
            % ", ".join(missing)
        )

    non_numeric = [
        c for c in STRICT_FEATURES
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if non_numeric:
        raise TypeError(
            "Canonical strict features must be numeric: %s"
            % ", ".join(non_numeric)
        )

    return list(STRICT_FEATURES)


def score_bands(actual, pred):
    rows = []
    for label, mask in [
        ("price >= 0", actual >= 0),
        ("price < 0", actual < 0),
        ("price < -50", actual < -50),
        ("price < -100", actual < -100),
    ]:
        if mask.any():
            rmse = float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))
            mae = float(np.mean(np.abs(actual[mask] - pred[mask])))
            rows.append((label, int(mask.sum()), rmse, mae))
    return rows


def fit_variant(df, test_start, label):
    working = df.copy()
    features = feature_columns(working)
    required_strict = [
        "consumption_forecast_j1",
        "nuclear_available_forecast_mw",
        "preauction_temperature_c",
        "preauction_wind_speed_10m_kmh",
        "preauction_wind_speed_100m_kmh",
        "preauction_cloud_cover_pct",
        "preauction_shortwave_radiation_wm2",
    ]
    missing = [c for c in required_strict if c not in working.columns]
    if missing:
        raise ValueError("Missing strict pre-auction columns: %s" % missing)

    target = "price_eur_mwh"
    # A80 historical outage information is intentionally unknown
    # before 2025-11-12. LightGBM handles this NaN natively.
    allowed_feature_nans = {
        "nuclear_a80_unavailable_safe",
    }

    complete_features = [
        f for f in features
        if f not in allowed_feature_nans
    ]

    clean = working.dropna(
        subset=[target] + complete_features
    ).copy()
    split_ts = pd.Timestamp(test_start)
    if split_ts.tzinfo is None:
        split_ts = split_ts.tz_localize("UTC")
    else:
        split_ts = split_ts.tz_convert("UTC")
    train_df = clean[clean["datetime"] < split_ts].copy()
    test_df = clean[clean["datetime"] >= split_ts].copy()
    if len(train_df) < 1000 or len(test_df) < 500:
        raise ValueError("Insufficient rows after strict filtering/split")

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
    model.fit(train_df[features], train_df[target])
    pred = model.predict(test_df[features])
    actual = test_df[target].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    mean_price = float(np.mean(actual))
    rmse_pct = float(100 * rmse / mean_price) if mean_price else float("nan")

    print("\n=== %s ===" % label)
    print("Features: %d" % len(features))
    print("Train: %d rows (%s -> %s)" % (len(train_df), train_df["datetime"].min(), train_df["datetime"].max()))
    print("Test:  %d rows (%s -> %s)" % (len(test_df), test_df["datetime"].min(), test_df["datetime"].max()))
    print("RMSE: %.2f EUR/MWh" % rmse)
    print("MAE:  %.2f EUR/MWh" % mae)
    print("RMSE as %% of mean price: %.1f%%" % rmse_pct)
    for band, n, r, m in score_bands(actual, pred):
        print("  %-12s: %4d h | RMSE %7.2f | MAE %7.2f" % (band, n, r, m))

    # Same-row naive baselines.
    if "price_lag_24h" in test_df.columns:
        b24 = test_df["price_lag_24h"].to_numpy()
        br = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    b24,
                )
            )
        )
        bm = float(
            mean_absolute_error(
                actual,
                b24,
            )
        )
        print(
            "Lag-24h same-row baseline: RMSE %.2f | MAE %.2f"
            % (
                br,
                bm,
            )
        )
    else:
        br = bm = float("nan")

    if "price_lag_168h" in test_df.columns:
        b168 = test_df[
            "price_lag_168h"
        ].to_numpy()

        b168r = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    b168,
                )
            )
        )

        b168m = float(
            mean_absolute_error(
                actual,
                b168,
            )
        )

        print(
            "Lag-168h same-row baseline: RMSE %.2f | MAE %.2f"
            % (
                b168r,
                b168m,
            )
        )
    else:
        b168r = b168m = float("nan")

    imp = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Feature importance (top 25):")
    for name, value in imp[:25]:
        print("  %s: %s" % (name, value))

    return {
        "model": model,
        "features": features,
        "test_df": test_df,
        "pred": pred,
        "rmse": rmse,
        "mae": mae,
        "rmse_pct": rmse_pct,
        "baseline_rmse": br,
        "baseline_mae": bm,
        "baseline_168h_rmse": b168r,
        "baseline_168h_mae": b168m,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


def save_variant(result, stem, note):
    os.makedirs("data", exist_ok=True)
    joblib.dump({"model": result["model"], "features": result["features"]}, "data/model_%s.pkl" % stem)
    out = result["test_df"][["datetime", "price_eur_mwh"]].copy()
    out["predicted_price_eur_mwh"] = result["pred"]
    out.to_csv("data/predictions_%s.csv" % stem, index=False)
    metrics = {
        "rmse": result["rmse"],
        "mae": result["mae"],
        "rmse_pct": result["rmse_pct"],
        "baseline_rmse": result["baseline_rmse"],
        "baseline_mae": result["baseline_mae"],
        "baseline_168h_rmse": result["baseline_168h_rmse"],
        "baseline_168h_mae": result["baseline_168h_mae"],
        "train_rows": int(result["train_rows"]),
        "test_rows": int(result["test_rows"]),
        "features": result["features"],
        "note": note,
    }
    with open("data/metrics_%s.json" % stem, "w") as f:
        json.dump(metrics, f, indent=2)
