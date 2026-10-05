import ast
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )

import rte_client



@pytest.fixture(autouse=True)
def _isolate_market_result_snapshots(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(
        tmp_path
    )


def _load_market_result_function():
    source = (
        ROOT / "app.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    function = next(
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name
        == "load_market_result"
    )

    # Do not execute @st.cache_data in this isolated unit test.
    function.decorator_list = []

    module = ast.Module(
        body=[function],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    namespace = {
        "pd": pd,
    }

    exec(
        compile(
            module,
            str(ROOT / "app.py"),
            "exec",
        ),
        namespace,
    )

    return namespace[
        "load_market_result"
    ]


def test_market_result_network_error_is_api_error(monkeypatch):
    def fail(*args, **kwargs):
        raise rte_client.RTEAPIError(
            "simulated network failure"
        )

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        fail,
    )

    loader = (
        _load_market_result_function()
    )

    frame, status, detail = loader(
        pd.Timestamp(
            "2026-09-07"
        ).date()
    )

    assert frame is None
    assert status == "api_error"
    assert detail == "RTEAPIError"


def test_market_result_empty_curve_is_not_available(monkeypatch):
    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs: [],
    )

    loader = (
        _load_market_result_function()
    )

    frame, status, detail = loader(
        pd.Timestamp(
            "2026-09-07"
        ).date()
    )

    assert frame is None
    assert status == "not_available"
    assert detail is None


def test_market_result_bad_payload_is_malformed(monkeypatch):
    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs: {
            "unexpected": "payload"
        },
    )

    loader = (
        _load_market_result_function()
    )

    frame, status, detail = loader(
        pd.Timestamp(
            "2026-09-07"
        ).date()
    )

    assert frame is None
    assert status == "malformed_response"
    assert detail is None


def _complete_market_payload(delivery_date):
    start_local = pd.Timestamp(delivery_date).tz_localize("Europe/Paris")
    end_local = pd.Timestamp(
        pd.Timestamp(delivery_date) + pd.Timedelta(days=1)
    ).tz_localize("Europe/Paris")
    timestamps = pd.date_range(
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    return [{
        "values": [
            {
                "start_date": ts.isoformat(),
                "price": 50.0 + i / 100.0,
            }
            for i, ts in enumerate(timestamps)
        ]
    }]


def test_rte_http_200_invalid_json_is_malformed_when_requested(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid JSON")

    monkeypatch.setattr(
        rte_client,
        "_get_access_token",
        lambda raise_errors=False: "token",
    )
    monkeypatch.setattr(
        rte_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(rte_client.RTEMalformedResponse):
        rte_client._get(
            "test/endpoint",
            raise_errors=True,
        )

    assert rte_client._get(
        "test/endpoint",
        raise_errors=False,
    ) is None


def test_market_result_rejects_one_bad_point_among_valid_points(monkeypatch):
    day = pd.Timestamp("2026-09-07").date()
    payload = _complete_market_payload(day)
    payload[0]["values"][12]["start_date"] = "bad"

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs: payload,
    )

    frame, status, detail = _load_market_result_function()(day)
    assert frame is None
    assert status == "malformed_response"
    assert detail is None


def test_market_result_rejects_incomplete_15_minute_curve(monkeypatch):
    day = pd.Timestamp("2026-09-07").date()
    payload = _complete_market_payload(day)
    payload[0]["values"].pop()

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs: payload,
    )

    frame, status, detail = _load_market_result_function()(day)
    assert frame is None
    assert status == "malformed_response"
    assert detail is None


@pytest.mark.parametrize(
    "day,expected_points",
    [
        ("2026-03-29", 92),
        ("2026-09-07", 96),
        ("2026-10-25", 100),
    ],
)
def test_market_result_accepts_complete_dst_safe_15_minute_curve(
    monkeypatch,
    day,
    expected_points,
):
    delivery = pd.Timestamp(day).date()
    payload = _complete_market_payload(delivery)
    assert len(payload[0]["values"]) == expected_points

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs: payload,
    )

    frame, status, detail = _load_market_result_function()(delivery)
    assert status == "ok"
    assert detail is None
    assert len(frame) == expected_points



def test_market_result_persists_and_reuses_snapshot(
    monkeypatch,
):
    day = pd.Timestamp(
        "2026-09-07"
    ).date()

    payload = _complete_market_payload(
        day
    )

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs:
            payload,
    )

    loader = (
        _load_market_result_function()
    )

    first, status, detail = loader(
        day
    )

    assert status == "ok"
    assert detail is None

    snapshot = Path(
        "data/live/market_result_2026-09-07.csv"
    )

    assert snapshot.is_file()

    wrong_day = pd.Timestamp(
        "2026-09-08"
    ).date()

    wrong_payload = (
        _complete_market_payload(
            wrong_day
        )
    )

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs:
            wrong_payload,
    )

    loader = (
        _load_market_result_function()
    )

    second, status, detail = loader(
        day
    )

    assert status == "ok"
    assert detail == "local_snapshot"

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
    )


def test_market_result_rejects_incomplete_local_snapshot(
    monkeypatch,
):
    day = pd.Timestamp(
        "2026-09-07"
    ).date()

    path = Path(
        "data/live/market_result_2026-09-07.csv"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame({
        "datetime": [
            "2026-09-06T22:00:00Z"
        ],
        "market_price_eur_mwh": [
            50.0
        ],
    }).to_csv(
        path,
        index=False,
    )

    monkeypatch.setattr(
        rte_client,
        "get_day_ahead_prices",
        lambda *args, **kwargs:
            [],
    )

    frame, status, detail = (
        _load_market_result_function()(
            day
        )
    )

    assert frame is None
    assert status == "not_available"
    assert detail is None
