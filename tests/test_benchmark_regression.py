import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from strict_model import (
    engineer_strict_features,
    fit_variant,
)


MACRO_FEATURES = [
    "ttf_front_month_eur_mwh",
    "ttf_change_5obs_pct",
    "eua_auction_eur_tco2",
    "eua_change_5auctions_pct",
    "ccgt_cost_proxy_eur_mwh",
    "ocgt_cost_proxy_eur_mwh",
]


EXPECTED_HASHES = {
    "data/dataset_strict.csv":
        "4723efa3d456a96a4f0f61b2b1b1f9179065ff40c0a25f3494614697a7476673",
    "data/model_strict.pkl":
        "6b51327fca4414a76102eeeb6d2bcd700dc9917fbef0180f2193b660d9a0722a",
    "data/predictions_strict.csv":
        "c0e97085dcd6cd6cf5f12a655ab2dcac1ea2ea4ca46ab2847e1ba71eedf0dce2",
    "data/metrics_strict.json":
        "50471ceb98719efbc58066a3cc0798fa487ae0cd97eb6ffa86bc9cb6da65005a",
}


def _sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def test_static_benchmark_regression():
    for relative in [
        "data/model_strict.pkl",
        "data/metrics_strict.json",
    ]:
        assert _sha256(
            ROOT / relative
        ) == EXPECTED_HASHES[relative]

    private_artifacts = [
        ROOT / "data/dataset_strict.csv",
        ROOT / "data/predictions_strict.csv",
    ]

    present = [
        path.is_file()
        for path in private_artifacts
    ]

    assert all(present) or not any(present)

    if not all(present):
        pytest.skip(
            "Reference benchmark CSVs are not redistributed "
            "in the public release."
        )

    for relative in [
        "data/dataset_strict.csv",
        "data/predictions_strict.csv",
    ]:
        assert _sha256(
            ROOT / relative
        ) == EXPECTED_HASHES[relative]

    raw = pd.read_csv(
        ROOT / "data/dataset_strict.csv"
    )

    df = engineer_strict_features(
        raw
    )

    clean = df.dropna(
        subset=MACRO_FEATURES
    ).copy()

    result = fit_variant(
        clean,
        "2025-12-12T23:00:00Z",
        "PYTEST STATIC BENCHMARK REGRESSION",
    )

    assert len(
        result["features"]
    ) == 32

    assert result[
        "train_rows"
    ] == 13923

    assert result[
        "test_rows"
    ] == 3862

    assert np.isclose(
        result["rmse"],
        30.1090005847,
        atol=1e-8,
    )

    assert np.isclose(
        result["mae"],
        20.2674092458,
        atol=1e-8,
    )

    assert np.isclose(
        result["baseline_rmse"],
        39.9574667264,
        atol=1e-8,
    )

    assert np.isclose(
        result["baseline_mae"],
        26.4504460124,
        atol=1e-8,
    )

    assert np.isclose(
        result["baseline_168h_rmse"],
        52.5845292270,
        atol=1e-8,
    )

    assert np.isclose(
        result["baseline_168h_mae"],
        37.7983022613,
        atol=1e-8,
    )

    artifact = joblib.load(
        ROOT / "data/model_strict.pkl"
    )
    tracked = pd.read_csv(
        ROOT / "data/predictions_strict.csv"
    )
    tracked["datetime"] = pd.to_datetime(
        tracked["datetime"],
        utc=True,
    )

    test_df = result["test_df"].reset_index(drop=True)
    assert artifact["features"] == result["features"]
    assert tracked["datetime"].reset_index(drop=True).equals(
        test_df["datetime"].reset_index(drop=True)
    )

    predicted = artifact["model"].predict(
        test_df[artifact["features"]]
    )

    np.testing.assert_allclose(
        predicted,
        tracked["predicted_price_eur_mwh"].to_numpy(),
        rtol=0.0,
        atol=1e-10,
    )
