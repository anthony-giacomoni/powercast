import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "src"
    ),
)

from macro_features import (
    TTF_CONTRACTS,
    _next_month_key,
    validate_ttf_contract_keys,
    validate_eua_delivery_dates,
)


def test_next_month_contract_selection():
    assert _next_month_key(
        datetime.date(2026, 10, 15)
    ) == "2026-11"


def test_year_boundary_contract_selection():
    assert _next_month_key(
        datetime.date(2026, 12, 15)
    ) == "2027-01"


def test_verified_extended_horizon_present():
    assert TTF_CONTRACTS[
        "2026-11"
    ] == "D.6277389"

    assert TTF_CONTRACTS[
        "2026-12"
    ] == "D.6277390"

    assert TTF_CONTRACTS[
        "2027-01"
    ] == "D.6277391"

    assert TTF_CONTRACTS[
        "2027-02"
    ] == "D.6277392"

    assert TTF_CONTRACTS[
        "2027-03"
    ] == "D.6277393"


def test_known_contracts_pass():
    required = validate_ttf_contract_keys({
        "2026-10",
        "2026-11",
        "2027-03",
    })

    assert required == [
        "2026-10",
        "2026-11",
        "2027-03",
    ]


def test_unknown_contract_fails():
    with pytest.raises(
        RuntimeError,
        match="2027-04",
    ):
        validate_ttf_contract_keys({
            "2027-04",
        })


def test_all_missing_contracts_are_reported():
    with pytest.raises(
        RuntimeError,
    ) as exc:
        validate_ttf_contract_keys({
            "2027-04",
            "2027-05",
        })

    message = str(exc.value)

    assert "2027-04" in message
    assert "2027-05" in message



def test_eua_2026_delivery_is_supported():
    dates = [
        datetime.date(
            2026,
            12,
            31,
        )
    ]

    assert validate_eua_delivery_dates(
        dates
    ) == dates


def test_eua_2027_delivery_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="2027",
    ):
        validate_eua_delivery_dates([
            datetime.date(
                2027,
                1,
                2,
            )
        ])
