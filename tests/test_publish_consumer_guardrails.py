import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_dataset_strict
import entsoe_unavailability_client as a80




def _load_app_functions(*names):
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = []
    name_set = set(names)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in name_set:
            node.decorator_list = []
            wanted.append(node)

    found = {node.name for node in wanted}
    if found != name_set:
        raise AssertionError("Could not extract app functions: %s" % sorted(name_set - found))

    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)

    st = SimpleNamespace(session_state={})
    namespace = {
        "pd": pd,
        "st": st,
    }
    exec(compile(module, str(ROOT / "app.py"), "exec"), namespace)
    return namespace


def _write_valid_forecast(path):
    pd.DataFrame({
        "datetime": ["2099-01-01T00:00:00Z"],
        "predicted_price_eur_mwh": [50.0],
        "forecast_generated_at": ["2098-12-31T11:30:00Z"],
    }).to_csv(path, index=False)


@pytest.mark.parametrize(
    "present",
    [
        {"forecast"},
        {"metadata"},
        {"inputs"},
        {"forecast", "metadata"},
        {"forecast", "inputs"},
        {"inputs", "metadata"},
    ],
)
def test_load_price_demo_rejects_partial_frozen_bundle(tmp_path, monkeypatch, present):
    functions = _load_app_functions(
        "frozen_price_bundle_complete",
        "load_price_demo",
    )
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)

    if "forecast" in present:
        _write_valid_forecast(live / "forecast_2099-01-01.csv")
    if "inputs" in present:
        (live / "inputs_2099-01-01.csv").write_text("x\\n1\\n", encoding="utf-8")
    if "metadata" in present:
        (live / "forecast_2099-01-01.json").write_text("{}\\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    df, delivery_date, note = functions["load_price_demo"]()

    assert df is None
    assert delivery_date is None
    assert note is not None
    assert "generated yet" in note.lower() or "valid frozen" in note.lower()


def test_load_price_demo_accepts_complete_frozen_bundle(tmp_path, monkeypatch):
    functions = _load_app_functions(
        "frozen_price_bundle_complete",
        "load_price_demo",
    )
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)

    _write_valid_forecast(live / "forecast_2099-01-01.csv")
    (live / "inputs_2099-01-01.csv").write_text("x\\n1\\n", encoding="utf-8")
    (live / "forecast_2099-01-01.json").write_text("{}\\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    df, delivery_date, note = functions["load_price_demo"]()

    assert df is not None
    assert delivery_date.isoformat() == "2099-01-01"
    assert "frozen" in note.lower()


def test_a80_transport_error_raises_instead_of_becoming_no_data(monkeypatch):
    monkeypatch.setattr(a80, "API_KEY", "dummy")
    monkeypatch.setattr(
        a80,
        "_request_outage_page",
        lambda *args, **kwargs: (None, "network request failed (ConnectionError)"),
    )

    with pytest.raises(RuntimeError, match="A80 request failed"):
        a80.get_nuclear_unavailability_events(
            "2026-01-01",
            "2026-02-01",
            catalog=pd.DataFrame({"resource_eic": ["TEST"]}),
        )


def test_a80_true_no_data_remains_legitimate_empty_month(monkeypatch):
    monkeypatch.setattr(a80, "API_KEY", "dummy")
    monkeypatch.setattr(
        a80,
        "_request_outage_page",
        lambda *args, **kwargs: ([], None),
    )

    result = a80.get_nuclear_unavailability_events(
        "2026-01-01",
        "2026-02-01",
        catalog=pd.DataFrame({"resource_eic": ["TEST"]}),
    )

    assert result is None


def test_multimonth_a80_rebuild_refuses_partial_success_after_source_error(
    tmp_path,
    monkeypatch,
):
    catalogs = {
        2026: pd.DataFrame({
            "resource_eic": ["TEST_UNIT"],
            "resource_name": ["Test Unit"],
            "installed_capacity_mw": [1000.0],
        })
    }
    calls = []

    def fake_fetch(start, end, catalog=None):
        calls.append(start)
        if start == "2026-01-01":
            raise RuntimeError("simulated A80 source failure")
        return pd.DataFrame({
            "created_doc_time": [pd.Timestamp("2026-02-01T00:00:00Z")],
            "start": [pd.Timestamp("2026-02-01T00:00:00Z")],
            "end": [pd.Timestamp("2026-02-02T00:00:00Z")],
        })

    monkeypatch.setattr(
        build_dataset_strict,
        "get_nuclear_unavailability_events",
        fake_fetch,
    )

    with pytest.raises(RuntimeError, match="simulated A80 source failure"):
        build_dataset_strict.fetch_nuclear_events(
            "2026-01-01",
            "2026-03-01",
            catalogs,
            cache_dir=str(tmp_path),
            refresh=True,
        )

    assert calls == ["2026-01-01"]
