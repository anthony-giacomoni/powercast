import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from entsoe_unavailability_client import (
    build_nuclear_availability_forecast_j1,
)


def catalog():
    return {
        2026: pd.DataFrame({
            "resource_eic": ["UNIT1"],
            "installed_capacity_mw": [1000.0],
        })
    }


def test_revision_published_after_cutoff_is_invisible():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC1",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T10:00:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 500.0,
        },
        {
            # D-1 11:00 Paris = 10:00 UTC in January.
            # This revision is therefore too late.
            "document_mrid": "DOC1",
            "revision": "2",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T12:00:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 1000.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-01-10",
        "2026-01-11",
        cutoff_hour_local=11,
    )

    assert len(out) == 24

    assert np.allclose(
        out["nuclear_nominal_capacity_mw"],
        1000.0,
    )

    # Only revision 1 was known at cutoff:
    # 1000 nominal - 500 available = 500 unavailable.
    assert np.allclose(
        out["nuclear_unavailable_forecast_mw"],
        500.0,
    )

    assert np.allclose(
        out["nuclear_available_forecast_mw"],
        500.0,
    )


def test_latest_cancelled_revision_removes_outage():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC1",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T09:00:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 500.0,
        },
        {
            # Latest revision known before cutoff cancels the document.
            "document_mrid": "DOC1",
            "revision": "2",
            "docstatus": "A09",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T09:30:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 500.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-01-10",
        "2026-01-11",
        cutoff_hour_local=11,
    )

    assert len(out) == 24

    assert np.allclose(
        out["nuclear_unavailable_forecast_mw"],
        0.0,
    )

    assert np.allclose(
        out["nuclear_available_forecast_mw"],
        1000.0,
    )



def test_withdrawn_a13_revision_removes_outage():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC-A13",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T08:00:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 500.0,
        },
        {
            "document_mrid": "DOC-A13",
            "revision": "2",
            "docstatus": "A13",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T09:30:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 500.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-01-10",
        "2026-01-11",
    )

    assert np.allclose(
        out["nuclear_unavailable_forecast_mw"],
        0.0,
    )


def test_same_unit_overlap_uses_most_restrictive_outage():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC-A",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T08:00:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 700.0,
        },
        {
            "document_mrid": "DOC-B",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T08:30:00Z",
            "start": "2026-01-09T23:00:00Z",
            "end": "2026-01-10T23:00:00Z",
            "available_mw": 400.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-01-10",
        "2026-01-11",
    )

    # 1000 nominal - min(700, 400) = 600 MW unavailable.
    # The two documents must not be double-counted.
    assert np.allclose(
        out["nuclear_unavailable_forecast_mw"],
        600.0,
    )


def test_intra_hour_outage_is_time_weighted():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC-HALF",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-01-09T08:00:00Z",
            "start": "2026-01-09T23:15:00Z",
            "end": "2026-01-09T23:45:00Z",
            "available_mw": 0.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-01-10",
        "2026-01-11",
    )

    # First local hour is unavailable for exactly 30 minutes.
    assert np.isclose(
        out.iloc[0][
            "nuclear_unavailable_forecast_mw"
        ],
        500.0,
    )

    assert np.allclose(
        out.iloc[1:][
            "nuclear_unavailable_forecast_mw"
        ],
        0.0,
    )


def test_default_11_cutoff_is_dst_safe():
    events = pd.DataFrame([
        {
            "document_mrid": "DOC-DST",
            "revision": "1",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            # D-1 is 29 Mar 2026: 11:00 Paris = 09:00 UTC.
            "created_doc_time": "2026-03-29T08:59:00Z",
            "start": "2026-03-29T22:00:00Z",
            "end": "2026-03-30T22:00:00Z",
            "available_mw": 500.0,
        },
        {
            "document_mrid": "DOC-DST",
            "revision": "2",
            "docstatus": "A05",
            "catalog_eic": "UNIT1",
            "created_doc_time": "2026-03-29T09:01:00Z",
            "start": "2026-03-29T22:00:00Z",
            "end": "2026-03-30T22:00:00Z",
            "available_mw": 1000.0,
        },
    ])

    out = build_nuclear_availability_forecast_j1(
        events,
        catalog(),
        "2026-03-30",
        "2026-03-31",
    )

    assert len(out) == 24

    assert np.allclose(
        out["nuclear_unavailable_forecast_mw"],
        500.0,
    )
