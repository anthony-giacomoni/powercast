import datetime
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_dataset_strict
import entsoe_client
import entsoe_unavailability_client
import forecast_live
import macro_features
import odre_client


class _DBResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "series": {
                "docs": [
                    {
                        "period": ["2026-09-04", "2026-09-07", "2026-09-08"],
                        "value": [30.0, 40.0, 99.0],
                    }
                ]
            }
        }


def test_ttf_refresh_replaces_stale_cache_and_cutoff_stays_strict(tmp_path, monkeypatch):
    cache = tmp_path / "macro"
    cache.mkdir()
    path = cache / "ttf_D_TEST.csv"
    pd.DataFrame({
        "date": ["2026-09-04"],
        "price": [10.0],
    }).to_csv(path, index=False)

    monkeypatch.setattr(
        macro_features.requests,
        "get",
        lambda *args, **kwargs: _DBResponse(),
    )

    refreshed = macro_features.fetch_ttf_contract(
        "D.TEST",
        str(cache),
        refresh=True,
    )

    assert refreshed.iloc[-1]["date"] == datetime.date(2026, 9, 8)
    cached = pd.read_csv(path)
    assert float(cached.iloc[-1]["price"]) == 99.0

    price, source_date, _ = macro_features.lookup_ttf(
        datetime.date(2026, 9, 9),
        pd.Timestamp("2026-09-08T09:00:00Z"),
        {"2026-10": refreshed},
    )
    assert source_date == datetime.date(2026, 9, 7)
    assert price == 40.0


def test_eua_refresh_replaces_stale_current_file_and_excludes_post_cutoff(tmp_path, monkeypatch):
    cache = tmp_path / "macro"
    cache.mkdir()
    history = cache / "eua_2012_2025.zip"
    with zipfile.ZipFile(history, "w") as zf:
        zf.writestr("history.xlsx", b"history")

    current = cache / "eua_current.xlsx"
    current.write_bytes(b"stale")

    downloads = []

    def fake_download(url, path, timeout=120):
        downloads.append((url, path))
        Path(path).write_bytes(b"fresh")
        return path

    def fake_parse(blob, source_name):
        if blob == b"history":
            return pd.DataFrame({
                "auction_ts_utc": [pd.Timestamp("2025-12-30T10:00:00Z")],
                "eua_price": [70.0],
                "source": [source_name],
            })
        if blob == b"fresh":
            return pd.DataFrame({
                "auction_ts_utc": [
                    pd.Timestamp("2026-09-08T08:00:00Z"),
                    pd.Timestamp("2026-09-08T10:30:00Z"),
                ],
                "eua_price": [80.0, 999.0],
                "source": [source_name, source_name],
            })
        raise AssertionError("stale current file should not be parsed during refresh")

    monkeypatch.setattr(macro_features, "_download", fake_download)
    monkeypatch.setattr(macro_features, "parse_eua_excel_bytes", fake_parse)

    eua = macro_features.load_eua(
        str(cache),
        str(current),
        refresh_current=True,
    )

    assert downloads == [(macro_features.EUA_CURRENT_URL, str(current))]
    assert current.read_bytes() == b"fresh"

    price, source_ts, _ = macro_features.lookup_eua(
        pd.Timestamp("2026-09-08T09:00:00Z"),
        eua,
    )
    assert price == 80.0
    assert source_ts == pd.Timestamp("2026-09-08T08:00:00Z")


def test_live_explicitly_requests_current_macro_refresh():
    source = (ROOT / "src" / "forecast_live.py").read_text(encoding="utf-8")
    assert "refresh_current=True" in source


def test_odre_partial_pagination_fails_closed(monkeypatch):
    calls = {"count": 0}

    class PageOne:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"date_heure": "2026-01-01T00:00:00+00:00", "prevision_j1": i}
                    for i in range(100)
                ]
            }

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return PageOne()
        raise requests.exceptions.ConnectionError("simulated page-2 failure")

    monkeypatch.setattr(odre_client.requests, "get", fake_get)

    with pytest.raises(odre_client.ODREAPIError, match="partial results were discarded"):
        odre_client.get_eco2mix_data(
            dataset=odre_client.DATASET_HISTORICAL,
            start_date="2026-01-01",
            end_date="2026-02-01",
            limit=200,
        )


def test_odre_end_boundary_is_exclusive(monkeypatch):
    captured = {}

    class EmptyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    def fake_get(url, params=None, timeout=None):
        captured.update(params or {})
        return EmptyResponse()

    monkeypatch.setattr(odre_client.requests, "get", fake_get)
    assert odre_client.get_eco2mix_data(
        dataset=odre_client.DATASET_HISTORICAL,
        start_date="2026-01-01",
        end_date="2026-02-01",
        limit=1,
    ) is None

    assert "date_heure < '2026-02-01'" in captured["where"]


def _token_http_error(token):
    response = requests.Response()
    response.status_code = 401
    response.url = "https://example.test/api?securityToken=%s" % token
    return requests.exceptions.HTTPError(
        "401 Client Error for url: %s" % response.url,
        response=response,
    )


def test_entsoe_price_error_never_logs_security_token(monkeypatch, capsys):
    token = "TOPSECRET_PRICE"
    monkeypatch.setattr(entsoe_client, "API_KEY", token)

    def fail(*args, **kwargs):
        raise _token_http_error(token)

    monkeypatch.setattr(entsoe_client.requests, "get", fail)
    assert entsoe_client.get_historical_prices("2026-01-01", "2026-01-02") is None
    output = capsys.readouterr().out
    assert token not in output
    assert "securityToken" not in output


def test_entsoe_a65_error_never_exposes_security_token(monkeypatch):
    token = "TOPSECRET_A65"
    monkeypatch.setattr(entsoe_client, "API_KEY", token)

    def fail(*args, **kwargs):
        raise _token_http_error(token)

    monkeypatch.setattr(entsoe_client.requests, "get", fail)

    with pytest.raises(entsoe_client.ENTSOEAPIError) as exc:
        entsoe_client.get_day_ahead_load_forecast("2026-09-07")

    assert token not in str(exc.value)
    assert "securityToken" not in str(exc.value)


def test_entsoe_a80_error_never_exposes_security_token(monkeypatch):
    token = "TOPSECRET_A80"

    def fail(*args, **kwargs):
        raise _token_http_error(token)

    monkeypatch.setattr(entsoe_unavailability_client.requests, "get", fail)
    response, error = entsoe_unavailability_client._get({
        "securityToken": token,
        "documentType": "A80",
    })
    assert response is None
    assert token not in error
    assert "securityToken" not in error


def _complete_d1_prices(delivery):
    previous_day = delivery.date() - pd.Timedelta(days=1)
    start = pd.Timestamp(previous_day, tz=forecast_live.PARIS_TZ).tz_convert("UTC")
    end = pd.Timestamp(previous_day + pd.Timedelta(days=1), tz=forecast_live.PARIS_TZ).tz_convert("UTC")
    return pd.DataFrame({
        "datetime": pd.date_range(start, end, freq="1h", inclusive="left"),
        "price_eur_mwh": 50.0,
    })


def test_rte_fallback_is_not_called_when_d1_prices_are_complete(monkeypatch):
    delivery = pd.Timestamp("2026-09-07", tz=forecast_live.PARIS_TZ)
    prices = _complete_d1_prices(delivery)

    def forbidden(*args, **kwargs):
        raise AssertionError("RTE must not be called for a complete D-1")

    monkeypatch.setattr(forecast_live, "get_day_ahead_prices", forbidden)
    out = forecast_live.fill_previous_day_price_gap_from_rte(prices, delivery)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), prices.reset_index(drop=True))


def test_rte_unavailable_keeps_missing_d1_timestamp_as_nan(monkeypatch):
    delivery = pd.Timestamp("2026-09-07", tz=forecast_live.PARIS_TZ)
    prices = _complete_d1_prices(delivery)
    missing_dt = prices.iloc[5]["datetime"]
    prices = prices[prices["datetime"] != missing_dt].copy()

    monkeypatch.setattr(
        forecast_live,
        "get_day_ahead_prices",
        lambda *args, **kwargs: None,
    )

    out = forecast_live.fill_previous_day_price_gap_from_rte(prices, delivery)
    row = out[out["datetime"] == missing_dt]
    assert len(row) == 1
    assert row["price_eur_mwh"].isna().all()


def test_a80_pagination_cap_fails_closed(monkeypatch):
    monkeypatch.setattr(entsoe_unavailability_client, "API_KEY", "dummy")
    monkeypatch.setattr(
        entsoe_unavailability_client,
        "_request_outage_page",
        lambda **kwargs: ([b"x"] * entsoe_unavailability_client.PAGE_SIZE, None),
    )
    monkeypatch.setattr(
        entsoe_unavailability_client,
        "_parse_unavailability_xml",
        lambda xml: [{"dummy": 1}],
    )

    with pytest.raises(RuntimeError, match="pagination safety cap"):
        entsoe_unavailability_client.get_nuclear_unavailability_events(
            "2026-01-01",
            "2026-02-01",
            max_documents=entsoe_unavailability_client.PAGE_SIZE * 2,
        )


def test_live_sha256_helper_hashes_exact_model_bytes(tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"exact-production-model-bytes")
    expected = __import__("hashlib").sha256(model.read_bytes()).hexdigest()
    assert forecast_live.sha256_file(str(model)) == expected


def _write_fake_eua_sources(tmp_path):
    cache = tmp_path / "macro"
    cache.mkdir()
    history = cache / "eua_2012_2025.zip"
    with zipfile.ZipFile(history, "w") as zf:
        zf.writestr("history.xlsx", b"history")
    current = cache / "eua_current.xlsx"
    current.write_bytes(b"current")
    return cache, current


def test_eua_current_year_empty_parse_fails_closed(tmp_path, monkeypatch):
    cache, current = _write_fake_eua_sources(tmp_path)

    def fake_parse(blob, source_name):
        if blob == b"history":
            return pd.DataFrame({
                "auction_ts_utc": [pd.Timestamp("2025-12-30T10:00:00Z")],
                "eua_price": [70.0],
                "source": [source_name],
            })
        if blob == b"current":
            return pd.DataFrame(
                columns=["auction_ts_utc", "eua_price", "source"]
            )
        raise AssertionError("unexpected EUA fixture")

    monkeypatch.setattr(macro_features, "parse_eua_excel_bytes", fake_parse)

    with pytest.raises(RuntimeError, match="no parseable auction rows"):
        macro_features.load_eua(
            str(cache),
            str(current),
            require_current_year=True,
        )


def test_eua_current_report_without_current_year_rows_fails_closed(tmp_path, monkeypatch):
    cache, current = _write_fake_eua_sources(tmp_path)

    def fake_parse(blob, source_name):
        if blob in (b"history", b"current"):
            return pd.DataFrame({
                "auction_ts_utc": [pd.Timestamp("2025-12-30T10:00:00Z")],
                "eua_price": [70.0],
                "source": [source_name],
            })
        raise AssertionError("unexpected EUA fixture")

    monkeypatch.setattr(macro_features, "parse_eua_excel_bytes", fake_parse)

    with pytest.raises(RuntimeError, match="no parseable 2026 auction rows"):
        macro_features.load_eua(
            str(cache),
            str(current),
            require_current_year=True,
        )


def test_builder_forwards_explicit_current_macro_refresh(tmp_path, monkeypatch):
    captured = {}
    prices = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-09-01T00:00:00Z")],
        "price_eur_mwh": [50.0],
    })

    monkeypatch.setattr(
        build_dataset_strict,
        "fetch_prices_strict",
        lambda *args, **kwargs: prices.copy(),
    )
    monkeypatch.setattr(
        build_dataset_strict,
        "fetch_consumption_j1",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        build_dataset_strict,
        "fetch_nuclear_preauction",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        build_dataset_strict,
        "get_national_preauction_weather",
        lambda *args, **kwargs: None,
    )

    def fake_macro(df, refresh_current=False, **kwargs):
        captured["refresh_current"] = refresh_current
        return df

    monkeypatch.setattr(
        build_dataset_strict,
        "add_macro_features",
        fake_macro,
    )

    build_dataset_strict.build_strict_dataset(
        "2026-09-01",
        "2026-09-02",
        output_path=str(tmp_path / "dataset.csv"),
        refresh_current=True,
    )

    assert captured["refresh_current"] is True


def test_a80_malformed_xml_fails_closed(monkeypatch):
    monkeypatch.setattr(entsoe_unavailability_client, "API_KEY", "dummy")
    monkeypatch.setattr(
        entsoe_unavailability_client,
        "_request_outage_page",
        lambda **kwargs: ([b"malformed"], None),
    )

    def malformed(_xml):
        raise entsoe_unavailability_client.ET.ParseError("bad XML")

    monkeypatch.setattr(
        entsoe_unavailability_client,
        "_parse_unavailability_xml",
        malformed,
    )

    with pytest.raises(RuntimeError, match="Malformed ENTSO-E A80 XML"):
        entsoe_unavailability_client.get_nuclear_unavailability_events(
            "2026-01-01",
            "2026-02-01",
        )

def test_a80_all_months_empty_fails_closed(tmp_path, monkeypatch):
    catalog = pd.DataFrame({
        "catalog_eic": ["TEST_UNIT"],
    })

    monkeypatch.setattr(
        build_dataset_strict,
        "get_nuclear_unavailability_events",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(
        RuntimeError,
        match="No A80 nuclear data available",
    ):
        build_dataset_strict.fetch_nuclear_events(
            "2026-01-01",
            "2026-02-01",
            {2026: catalog},
            cache_dir=str(tmp_path),
            refresh=True,
        )
