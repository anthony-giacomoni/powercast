# ============================================================
# PowerCast — strict pre-auction benchmark trainer
#
# Strict benchmark model:
#   - D-1 consumption forecast
#   - fixed pre-auction ECMWF weather
#   - clean nuclear representation
#   - known historical price lags
#   - TTF gas features
#   - EUA carbon features
#
# Excludes A69, A09, A61 and experimental neighbour prices.
# ============================================================

import argparse

import pandas as pd

from strict_model import (
    engineer_strict_features,
    fit_variant,
    save_variant,
)


MACRO_FEATURES = [
    "ttf_front_month_eur_mwh",
    "ttf_change_5obs_pct",
    "eua_auction_eur_tco2",
    "eua_change_5auctions_pct",
    "ccgt_cost_proxy_eur_mwh",
    "ocgt_cost_proxy_eur_mwh",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/dataset_strict.csv",
    )

    parser.add_argument(
        "--test-start",
        default="2025-12-12T23:00:00Z",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save model, predictions and metrics.",
    )

    args = parser.parse_args()

    print("PowerCast strict pre-auction model")
    print("Loading %s..." % args.data)

    raw = pd.read_csv(args.data)

    print("Loaded %d rows" % len(raw))

    missing = [
        c for c in MACRO_FEATURES
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            "Missing strict macro columns: %s"
            % missing
        )

    df = engineer_strict_features(raw)

    clean_macro = df.dropna(
        subset=MACRO_FEATURES
    ).copy()

    print(
        "Rows with complete macro features: %d/%d"
        % (
            len(clean_macro),
            len(df),
        )
    )

    result = fit_variant(
        clean_macro,
        args.test_start,
        "STRICT CLEAN PRE-AUCTION + TTF/EUA",
    )

    if args.save:
        save_variant(
            result,
            "strict",
            (
                "Strict clean pre-auction PowerCast model: "
                "fixed ECMWF D-1 weather, J-1 demand, "
                "clean nuclear split, historical price lags, "
                "TTF gas and EUA carbon features."
            ),
        )

        print(
            "\n✅ Saved strict benchmark artifacts"
        )


if __name__ == "__main__":
    main()
