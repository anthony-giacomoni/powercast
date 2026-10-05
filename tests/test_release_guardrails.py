import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "src"
    ),
)

import rte_client

from forecast_live import (
    atomic_write_csv,
    atomic_write_json,
    frozen_bundle_complete,
    remove_forecast_commit_marker,
)
from strict_model import (
    LIVE_ALLOWED_FEATURE_NANS,
    TRAIN_ALLOWED_FEATURE_NANS,
    validate_feature_missingness,
)
from train_model_production import (
    sha256_file,
)
from entsoe_client import (
    _extract_load_document_metadata,
)


def test_training_and_live_missingness_contracts():
    df = pd.DataFrame({
        "nuclear_a80_unavailable_safe": [
            np.nan,
        ],
        "price_lag_168h": [
            np.nan,
        ],
        "consumption_forecast_j1": [
            45000.0,
        ],
    })

    # A80 missingness is admissible in both modes.
    validate_feature_missingness(
        df[
            [
                "nuclear_a80_unavailable_safe",
            ]
        ],
        [
            "nuclear_a80_unavailable_safe",
        ],
        TRAIN_ALLOWED_FEATURE_NANS,
        "training",
    )

    # Genuine historical-price gaps may remain live.
    validate_feature_missingness(
        df,
        list(df.columns),
        LIVE_ALLOWED_FEATURE_NANS,
        "live",
    )

    # The same missing price lag is not accepted in training.
    with pytest.raises(
        RuntimeError,
        match="price_lag_168h",
    ):
        validate_feature_missingness(
            df,
            list(df.columns),
            TRAIN_ALLOWED_FEATURE_NANS,
            "training",
        )


def test_required_live_feature_missing_fails():
    df = pd.DataFrame({
        "consumption_forecast_j1": [
            np.nan,
        ],
    })

    with pytest.raises(
        RuntimeError,
        match="consumption_forecast_j1",
    ):
        validate_feature_missingness(
            df,
            [
                "consumption_forecast_j1",
            ],
            LIVE_ALLOWED_FEATURE_NANS,
            "live",
        )


def test_atomic_csv_write(tmp_path):
    path = tmp_path / "forecast.csv"

    df = pd.DataFrame({
        "x": [
            1,
            2,
        ],
    })

    atomic_write_csv(
        df,
        str(path),
    )

    loaded = pd.read_csv(path)

    assert loaded["x"].tolist() == [1, 2]
    assert not list(
        tmp_path.glob("*.tmp.*")
    )


def test_atomic_json_write(tmp_path):
    path = tmp_path / "forecast.json"

    payload = {
        "status": "frozen",
        "hours": 24,
    }

    atomic_write_json(
        payload,
        str(path),
    )

    assert json.loads(
        path.read_text()
    ) == payload

    assert not list(
        tmp_path.glob("*.tmp.*")
    )


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")

    assert sha256_file(
        str(path)
    ) == hashlib.sha256(
        b"abc"
    ).hexdigest()



def test_a65_document_metadata_parser():
    xml = """
    <GL_MarketDocument xmlns="urn:test">
      <mRID>DOC-123</mRID>
      <revisionNumber>7</revisionNumber>
      <createdDateTime>2026-09-06T08:30:00Z</createdDateTime>
    </GL_MarketDocument>
    """

    meta = _extract_load_document_metadata(
        xml
    )

    assert meta["document_mrid"] == "DOC-123"
    assert meta["revision_number"] == 7
    assert (
        meta["created_at_utc"].isoformat()
        == "2026-09-06T08:30:00+00:00"
    )



def test_frozen_bundle_requires_all_three_artifacts(
    tmp_path,
):
    forecast = tmp_path / "forecast.csv"
    inputs = tmp_path / "inputs.csv"
    metadata = tmp_path / "forecast.json"

    forecast.write_text("forecast")

    assert not frozen_bundle_complete(
        str(forecast),
        str(inputs),
        str(metadata),
    )

    inputs.write_text("inputs")

    assert not frozen_bundle_complete(
        str(forecast),
        str(inputs),
        str(metadata),
    )

    metadata.write_text("{}")

    assert frozen_bundle_complete(
        str(forecast),
        str(inputs),
        str(metadata),
    )


def test_rte_network_error_propagates_when_requested(
    monkeypatch,
):
    monkeypatch.setattr(
        rte_client,
        "_get_access_token",
        lambda raise_errors=False: "token",
    )

    def fail_request(*args, **kwargs):
        raise rte_client.requests.exceptions.ConnectionError(
            "network down"
        )

    monkeypatch.setattr(
        rte_client.requests,
        "get",
        fail_request,
    )

    with pytest.raises(
        rte_client.RTEAPIError,
    ):
        rte_client._get(
            "test/endpoint",
            raise_errors=True,
        )


def test_rte_network_error_remains_optional_by_default(
    monkeypatch,
):
    monkeypatch.setattr(
        rte_client,
        "_get_access_token",
        lambda raise_errors=False: "token",
    )

    def fail_request(*args, **kwargs):
        raise rte_client.requests.exceptions.ConnectionError(
            "network down"
        )

    monkeypatch.setattr(
        rte_client.requests,
        "get",
        fail_request,
    )

    assert rte_client._get(
        "test/endpoint",
    ) is None



def test_forecast_commit_marker_is_removed_before_rebuild(tmp_path):
    forecast = tmp_path / "forecast.csv"
    forecast.write_text("old forecast")

    assert remove_forecast_commit_marker(
        str(forecast)
    )
    assert not forecast.exists()

    assert not remove_forecast_commit_marker(
        str(forecast)
    )


def test_rte_missing_access_token_fails_when_requested(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"expires_in": 7200}

    monkeypatch.setattr(
        rte_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        rte_client,
        "CLIENT_ID",
        "dummy",
    )

    monkeypatch.setattr(
        rte_client,
        "CLIENT_SECRET",
        "dummy",
    )

    monkeypatch.setattr(
        rte_client,
        "_token_cache",
        {
            "access_token": None,
            "expires_at": 0,
        },
    )

    with pytest.raises(
        rte_client.RTEAPIError,
        match="access token",
    ):
        rte_client._get_access_token(
            raise_errors=True
        )
