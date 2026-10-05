import sys
from pathlib import Path

import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from forecast_live import (  # noqa: E402
    choose_delivery_date,
    make_target_grid,
)


PARIS_TZ = "Europe/Paris"


@pytest.mark.parametrize(
    "delivery_day, expected_hours",
    [
        ("2026-03-28", 24),
        ("2026-03-29", 23),
        ("2026-03-30", 24),
        ("2026-10-24", 24),
        ("2026-10-25", 25),
        ("2026-10-26", 24),
    ],
)
def test_delivery_grid_is_dst_safe(
    delivery_day,
    expected_hours,
):
    delivery = pd.Timestamp(
        delivery_day,
        tz=PARIS_TZ,
    )

    grid = make_target_grid(delivery)

    local = (
        pd.to_datetime(
            grid["datetime"],
            utc=True,
        )
        .dt.tz_convert(PARIS_TZ)
    )

    assert len(grid) == expected_hours

    assert set(local.dt.date) == {
        pd.Timestamp(delivery_day).date()
    }

    # Every delivery period must represent a unique instant.
    assert grid["datetime"].is_unique


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-03-29 10:59:59",
        "2026-10-25 10:59:59",
    ],
)
def test_live_forecast_refused_before_cutoff(stamp):
    now = pd.Timestamp(
        stamp,
        tz=PARIS_TZ,
    )

    with pytest.raises(RuntimeError):
        choose_delivery_date(now)


@pytest.mark.parametrize(
    "stamp, expected_delivery",
    [
        (
            "2026-03-29 11:00:00",
            "2026-03-30",
        ),
        (
            "2026-03-29 11:30:00",
            "2026-03-30",
        ),
        (
            "2026-10-25 11:00:00",
            "2026-10-26",
        ),
        (
            "2026-10-25 11:30:00",
            "2026-10-26",
        ),
    ],
)
def test_live_cutoff_is_dst_safe(
    stamp,
    expected_delivery,
):
    now = pd.Timestamp(
        stamp,
        tz=PARIS_TZ,
    )

    delivery = choose_delivery_date(now)

    assert delivery.date() == (
        pd.Timestamp(expected_delivery).date()
    )


def test_autumn_dst_keeps_both_0200_hours():
    delivery = pd.Timestamp(
        "2026-10-25",
        tz=PARIS_TZ,
    )

    grid = make_target_grid(delivery)

    local = (
        pd.to_datetime(
            grid["datetime"],
            utc=True,
        )
        .dt.tz_convert(PARIS_TZ)
    )

    two_am = local[
        local.dt.hour == 2
    ]

    # Europe/Paris has two distinct 02:00 delivery periods
    # when clocks move backward.
    assert len(two_am) == 2

    utc_two_am = (
        two_am
        .dt.tz_convert("UTC")
    )

    assert utc_two_am.iloc[0] != utc_two_am.iloc[1]


def test_spring_dst_has_no_0200_hour():
    delivery = pd.Timestamp(
        "2026-03-29",
        tz=PARIS_TZ,
    )

    grid = make_target_grid(delivery)

    local = (
        pd.to_datetime(
            grid["datetime"],
            utc=True,
        )
        .dt.tz_convert(PARIS_TZ)
    )

    assert not (local.dt.hour == 2).any()



import forecast_live


def _complete_a65_frame(delivery):
    frame = forecast_live.make_target_grid(
        delivery
    )
    frame["consumption_forecast_j1"] = (
        50000.0
    )
    return frame


@pytest.mark.parametrize(
    "delivery_day,expected_hours",
    [
        ("2026-03-29", 23),
        ("2026-09-07", 24),
        ("2026-10-25", 25),
    ],
)
def test_build_live_inputs_discards_target_day_prices(
    monkeypatch,
    tmp_path,
    delivery_day,
    expected_hours,
):
    delivery = pd.Timestamp(
        delivery_day,
        tz=PARIS_TZ,
    )

    delivery_start_utc = (
        pd.Timestamp(
            delivery.date()
        )
        .tz_localize(PARIS_TZ)
        .tz_convert("UTC")
    )

    history_index = pd.date_range(
        delivery_start_utc
        - pd.Timedelta(hours=192),
        delivery_start_utc
        + pd.Timedelta(
            hours=expected_hours - 1
        ),
        freq="1h",
    )

    prices = pd.DataFrame({
        "datetime": history_index,
        "price_eur_mwh": [
            float(i)
            for i in range(
                len(history_index)
            )
        ],
    })

    monkeypatch.setattr(
        forecast_live,
        "fetch_prices_strict",
        lambda *args, **kwargs:
            prices.copy(),
    )

    monkeypatch.setattr(
        forecast_live,
        "fill_previous_day_price_gap_from_rte",
        lambda frame, delivery:
            frame,
    )

    demand = _complete_a65_frame(
        delivery
    )

    monkeypatch.setattr(
        forecast_live,
        "load_a65_snapshot",
        lambda *args, **kwargs: (
            demand.copy(),
            {
                "snapshot_csv_sha256":
                    "test",
            },
        ),
    )

    empty_source = (
        forecast_live
        .make_target_grid(
            delivery
        )
    )

    monkeypatch.setattr(
        forecast_live,
        "fetch_nuclear_preauction",
        lambda *args, **kwargs:
            empty_source.copy(),
    )

    monkeypatch.setattr(
        forecast_live,
        "get_national_preauction_weather",
        lambda *args, **kwargs:
            empty_source.copy(),
    )

    monkeypatch.setattr(
        forecast_live,
        "add_macro_features",
        lambda frame, **kwargs:
            frame,
    )

    raw, target, _ = (
        forecast_live
        .build_live_inputs(
            delivery,
            output_dir=str(tmp_path),
        )
    )

    local = (
        pd.to_datetime(
            raw["datetime"],
            utc=True,
        )
        .dt.tz_convert(PARIS_TZ)
    )

    delivery_rows = raw[
        local.dt.date
        == delivery.date()
    ]

    assert len(
        delivery_rows
    ) == expected_hours

    assert (
        delivery_rows[
            "price_eur_mwh"
        ]
        .isna()
        .all()
    )

    assert len(
        target
    ) == expected_hours

    historical = raw[
        pd.to_datetime(
            raw["datetime"],
            utc=True,
        )
        < delivery_start_utc
    ]

    assert (
        historical[
            "price_eur_mwh"
        ]
        .notna()
        .any()
    )


@pytest.mark.parametrize(
    "bad_prices",
    [
        pd.DataFrame(),
        pd.DataFrame({
            "price_eur_mwh":
                [50.0],
        }),
        pd.DataFrame({
            "datetime": [
                "2026-09-01T00:00:00Z"
            ],
        }),
    ],
)
def test_historical_price_source_failure_is_explicit(
    monkeypatch,
    tmp_path,
    bad_prices,
):
    delivery = pd.Timestamp(
        "2026-09-07",
        tz=PARIS_TZ,
    )

    monkeypatch.setattr(
        forecast_live,
        "fetch_prices_strict",
        lambda *args, **kwargs:
            bad_prices.copy(),
    )

    with pytest.raises(
        RuntimeError,
        match="Historical day-ahead prices",
    ):
        forecast_live.build_live_inputs(
            delivery,
            output_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    "delivery_day",
    [
        "2026-03-29",
        "2026-10-25",
    ],
)
def test_a65_snapshot_capture_accepts_pre_cutoff(
    monkeypatch,
    tmp_path,
    delivery_day,
):
    delivery = pd.Timestamp(
        delivery_day,
        tz=PARIS_TZ,
    )

    cutoff = (
        forecast_live
        .delivery_cutoff(
            delivery
        )
    )

    now = (
        cutoff
        - pd.Timedelta(minutes=1)
    )

    def fake_a65(*args, **kwargs):
        frame = _complete_a65_frame(
            delivery
        )

        frame.attrs = {
            "source":
                "ENTSO-E A65/A01",
            "retrieved_at_utc":
                now
                .tz_convert("UTC")
                .isoformat(),
            "document_created_at_utc":
                now
                .tz_convert("UTC")
                .isoformat(),
            "document_mrid":
                "test-mrid",
            "document_revision":
                1,
        }

        return frame

    monkeypatch.setattr(
        forecast_live,
        "get_day_ahead_load_forecast",
        fake_a65,
    )

    frame, metadata = (
        forecast_live
        .capture_a65_snapshot(
            delivery,
            str(tmp_path),
            now_local=now,
        )
    )

    csv_path, json_path = (
        forecast_live
        .a65_snapshot_paths(
            str(tmp_path),
            delivery,
        )
    )

    assert Path(
        csv_path
    ).is_file()

    assert Path(
        json_path
    ).is_file()

    assert len(frame) in (
        23,
        25,
    )

    assert (
        metadata[
            "snapshot_csv_sha256"
        ]
        == forecast_live.sha256_file(
            csv_path
        )
    )


@pytest.mark.parametrize(
    "offset_seconds",
    [
        0,
        1,
    ],
)
def test_a65_snapshot_capture_rejects_at_or_after_cutoff(
    monkeypatch,
    tmp_path,
    offset_seconds,
):
    delivery = pd.Timestamp(
        "2026-09-07",
        tz=PARIS_TZ,
    )

    cutoff = (
        forecast_live
        .delivery_cutoff(
            delivery
        )
    )

    called = {
        "value": False
    }

    def should_not_call(*args, **kwargs):
        called["value"] = True
        raise AssertionError(
            "A65 API should not be called"
        )

    monkeypatch.setattr(
        forecast_live,
        "get_day_ahead_load_forecast",
        should_not_call,
    )

    with pytest.raises(
        RuntimeError,
        match="only allowed before",
    ):
        forecast_live.capture_a65_snapshot(
            delivery,
            str(tmp_path),
            now_local=(
                cutoff
                + pd.Timedelta(
                    seconds=offset_seconds
                )
            ),
        )

    assert not called["value"]


def test_a65_snapshot_rejects_post_cutoff_document(
    monkeypatch,
    tmp_path,
):
    delivery = pd.Timestamp(
        "2026-09-07",
        tz=PARIS_TZ,
    )

    cutoff = (
        forecast_live
        .delivery_cutoff(
            delivery
        )
    )

    now = (
        cutoff
        - pd.Timedelta(minutes=10)
    )

    def fake_a65(*args, **kwargs):
        frame = _complete_a65_frame(
            delivery
        )

        frame.attrs = {
            "retrieved_at_utc":
                now
                .tz_convert("UTC")
                .isoformat(),
            "document_created_at_utc":
                (
                    cutoff
                    + pd.Timedelta(
                        minutes=1
                    )
                )
                .tz_convert("UTC")
                .isoformat(),
        }

        return frame

    monkeypatch.setattr(
        forecast_live,
        "get_day_ahead_load_forecast",
        fake_a65,
    )

    with pytest.raises(
        RuntimeError,
        match="created after",
    ):
        forecast_live.capture_a65_snapshot(
            delivery,
            str(tmp_path),
            now_local=now,
        )


def test_a65_snapshot_missing_fails_closed(
    tmp_path,
):
    delivery = pd.Timestamp(
        "2026-09-07",
        tz=PARIS_TZ,
    )

    with pytest.raises(
        RuntimeError,
        match="No frozen pre-cutoff",
    ):
        forecast_live.load_a65_snapshot(
            delivery,
            str(tmp_path),
        )


def test_a65_snapshot_hash_mismatch_fails_closed(
    monkeypatch,
    tmp_path,
):
    delivery = pd.Timestamp(
        "2026-09-07",
        tz=PARIS_TZ,
    )

    cutoff = (
        forecast_live
        .delivery_cutoff(
            delivery
        )
    )

    now = (
        cutoff
        - pd.Timedelta(minutes=5)
    )

    def fake_a65(*args, **kwargs):
        frame = _complete_a65_frame(
            delivery
        )

        frame.attrs = {
            "retrieved_at_utc":
                now
                .tz_convert("UTC")
                .isoformat(),
            "document_created_at_utc":
                now
                .tz_convert("UTC")
                .isoformat(),
        }

        return frame

    monkeypatch.setattr(
        forecast_live,
        "get_day_ahead_load_forecast",
        fake_a65,
    )

    forecast_live.capture_a65_snapshot(
        delivery,
        str(tmp_path),
        now_local=now,
    )

    csv_path, _ = (
        forecast_live
        .a65_snapshot_paths(
            str(tmp_path),
            delivery,
        )
    )

    with open(
        csv_path,
        "a",
    ) as f:
        f.write("\n")

    with pytest.raises(
        RuntimeError,
        match="hash does not match",
    ):
        forecast_live.load_a65_snapshot(
            delivery,
            str(tmp_path),
        )


@pytest.mark.parametrize(
    "delivery_day,expected_hours",
    [
        ("2026-03-29", 23),
        ("2026-09-07", 24),
        ("2026-10-25", 25),
    ],
)
def test_a65_snapshot_grid_is_dst_safe(
    delivery_day,
    expected_hours,
):
    delivery = pd.Timestamp(
        delivery_day,
        tz=PARIS_TZ,
    )

    frame = _complete_a65_frame(
        delivery
    )

    out = (
        forecast_live
        .validate_a65_snapshot_frame(
            frame,
            delivery,
        )
    )

    assert len(
        out
    ) == expected_hours
