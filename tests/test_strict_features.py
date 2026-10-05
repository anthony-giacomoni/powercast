import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from strict_model import (
    STRICT_FEATURES,
    engineer_strict_features,
    feature_columns,
)


EXPECTED_STRICT_FEATURES = [
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

SAFE_NUCLEAR = EXPECTED_STRICT_FEATURES[-4:]


def make_df():
    return pd.DataFrame({
        # First hour is before the reliable A80 boundary in Paris.
        # Second hour is exactly 2025-11-12 00:00 Europe/Paris.
        "datetime": [
            "2025-11-11T22:00:00Z",
            "2025-11-11T23:00:00Z",
        ],
        "price_eur_mwh": [50.0, 60.0],
        "consumption_forecast_j1": [40000.0, 40000.0],
        "nuclear_nominal_capacity_mw": [60000.0, 60000.0],
        "nuclear_available_forecast_mw": [58800.0, 58800.0],
        "nuclear_unavailable_forecast_mw": [1200.0, 1200.0],
        "preauction_temperature_c": [10.0, 10.0],
        "preauction_wind_speed_10m_kmh": [20.0, 20.0],
        "preauction_wind_speed_100m_kmh": [30.0, 30.0],
        "preauction_cloud_cover_pct": [50.0, 50.0],
        "preauction_shortwave_radiation_wm2": [100.0, 100.0],
        "ttf_front_month_eur_mwh": [30.0, 30.0],
        "ttf_change_5obs_pct": [0.1, 0.1],
        "eua_auction_eur_tco2": [80.0, 80.0],
        "eua_change_5auctions_pct": [0.2, 0.2],
        "ccgt_cost_proxy_eur_mwh": [82.0, 82.0],
        "ocgt_cost_proxy_eur_mwh": [123.0, 123.0],

        # Deliberately forbidden legacy inputs.
        "wind_forecast_mw": [5000.0, 5000.0],
        "solar_forecast_mw": [3000.0, 3000.0],
        "temperature_c": [15.0, 15.0],
        "capacity_fr_to_es_mw": [2000.0, 2000.0],
    })


from strict_model import _fr_holiday_dates

def test_clean_nuclear_split_boundary():
    df = engineer_strict_features(make_df())

    # Nominal context remains valid throughout history.
    assert np.allclose(
        df["nuclear_nominal_capacity_safe"],
        [60000.0, 60000.0],
    )

    assert np.allclose(
        df["nuclear_nominal_to_demand_ratio_safe"],
        [1.5, 1.5],
    )

    assert np.allclose(
        df["nuclear_nominal_demand_margin_safe"],
        [20000.0, 20000.0],
    )

    # A80 is intentionally unknown before 2025-11-12 Paris.
    assert pd.isna(
        df.loc[0, "nuclear_a80_unavailable_safe"]
    )

    # From the migration boundary onward, the real A80 value is used.
    assert (
        df.loc[1, "nuclear_a80_unavailable_safe"]
        == 1200.0
    )


def test_feature_order_and_forbidden_inputs():
    df = engineer_strict_features(make_df())
    features = feature_columns(df)

    # Exact feature names AND order are part of LightGBM
    # reproducibility because colsample_bytree < 1.
    assert STRICT_FEATURES == EXPECTED_STRICT_FEATURES
    assert features == EXPECTED_STRICT_FEATURES
    assert features[-4:] == SAFE_NUCLEAR

    forbidden = {
        "wind_forecast_mw",
        "solar_forecast_mw",
        "temperature_c",
        "capacity_fr_to_es_mw",
        "nuclear_nominal_capacity_mw",
        "nuclear_available_forecast_mw",
        "nuclear_unavailable_forecast_mw",
    }

    assert forbidden.isdisjoint(features)



def test_accidental_numeric_column_is_never_auto_selected():
    df = engineer_strict_features(make_df())

    # Simulate a future numeric source column being added to the dataset.
    # It must NOT silently become a model feature.
    df["accidental_future_numeric_column"] = [123.0, 456.0]

    features = feature_columns(df)

    assert features == EXPECTED_STRICT_FEATURES
    assert "accidental_future_numeric_column" not in features


def test_missing_canonical_feature_fails_loudly():
    df = engineer_strict_features(make_df())

    df = df.drop(
        columns=["ttf_front_month_eur_mwh"]
    )

    with pytest.raises(
        ValueError,
        match="ttf_front_month_eur_mwh",
    ):
        feature_columns(df)



def test_missing_holidays_dependency_fails_loudly(
    monkeypatch,
):
    import builtins

    real_import = builtins.__import__

    def fake_import(
        name,
        *args,
        **kwargs
    ):
        if name == "holidays":
            raise ImportError(
                "simulated missing dependency"
            )

        return real_import(
            name,
            *args,
            **kwargs
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        fake_import,
    )

    with pytest.raises(
        RuntimeError,
        match="holidays==0.58",
    ):
        _fr_holiday_dates([
            2026,
        ])
