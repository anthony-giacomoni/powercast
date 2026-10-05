# PowerCast

**Strict pre-auction forecasting of French day-ahead electricity prices.**

PowerCast is a machine-learning project that forecasts French day-ahead electricity prices while enforcing a simple rule: **a feature is only admissible if it could realistically have been known before the D-1 market auction**.

The project focuses as much on information timing, leakage prevention and reproducibility as on predictive performance.

## Why PowerCast

Electricity-price forecasting is unusually vulnerable to target leakage. Realized generation, realized weather, late outage revisions, market-clearing outputs or very recent price information can make a historical model look excellent while being impossible to reproduce in real operation.

PowerCast therefore keeps two clearly separated branches:

- **Historical / explanatory model** — uses contemporaneous realized information and reaches roughly **14.1 €/MWh RMSE**. It is retained for analysis, not presented as an operational day-ahead forecast.
- **Strict pre-auction benchmark model** — the leakage-controlled validation pipeline, restricted to information available before the auction.
- **Strict production model** — a separate model trained on all currently admissible historical rows for frozen live forecasting.

The Streamlit application reads **frozen strict production forecasts** when available. Benchmark and production artifacts are intentionally kept separate.

## Current strict model

The final strict model uses 32 engineered features built from:

- historical French J-1 load forecasts from ODRÉ, with ENTSO-E A65 used separately for live inference
- fixed pre-auction ECMWF weather forecasts
- historical French day-ahead price lags
- audited ENTSO-E A80 nuclear availability
- TTF natural-gas futures
- official EEX EUA carbon auctions; the current verified EUA source is intentionally fail-closed beyond 2026 until a new annual source is configured
- calendar and weather-derived features

The model uses a frozen LightGBMRegressor configuration chosen during the historical development phase. The public repository preserves the benchmark implementation, frozen model, aggregated metrics and canonical reference hashes. The reference dataset and prediction CSV are not redistributed because they contain third-party market data.

### Features deliberately excluded

The strict baseline excludes several variables that were predictive but not sufficiently defensible from a point-in-time perspective:

- ENTSO-E A69 renewable forecasts
- ENTSO-E A09 scheduled commercial exchanges
- ENTSO-E A61 transfer capacity
- realized generation
- realized weather
- one-hour price lag

See [`METHODOLOGY.md`](METHODOLOGY.md) for the full rationale.

## Results

PowerCast reports **two different benchmarks**. They measure different protocols and should not be compared as if they were the same test.

| Protocol | RMSE | MAE | Interpretation |
|---|---:|---:|---|
| Static strict split | **30.11 €/MWh** | **20.27 €/MWh** | Fixed chronological holdout aligned to a full Europe/Paris delivery day |
| Monthly expanding retrain, Jun-Aug 2026 | **27.185 €/MWh** | **20.924 €/MWh** | Development benchmark closer to operational monthly retraining |

### Static strict split

Current saved strict artifacts:

- Train: **13,923 hourly observations**
- Test: **3,862 hourly observations**
- Test start: **13 December 2025 00:00 Europe/Paris**
- RMSE: **30.11 €/MWh**
- MAE: **20.27 €/MWh**
- 24-hour lag baseline RMSE: **39.96 €/MWh**

With the local reference dataset available, the benchmark training procedure is reproduced by:

```bash
python src/train_model_strict.py --data data/dataset_strict.csv
```

The reference dataset itself is not redistributed in the public repository. Its canonical SHA-256 remains documented for provenance.

### Monthly retraining development benchmark

Expanding-window retraining by month produced:

| Month | RMSE |
|---|---:|
| June 2026 | 23.687 €/MWh |
| July 2026 | 27.734 €/MWh |
| August 2026 | 29.287 €/MWh |
| **Combined** | **27.185 €/MWh** |

Combined MAE: **20.924 €/MWh** across **1,957 hourly observations**.

**Important:** June-August 2026 was inspected repeatedly during later model and data audits. It is therefore reported as a **development benchmark, not an untouched final out-of-sample test set**. A future unseen period is required for a genuine final OOS evaluation.

## Data pipeline

The strict branch is intentionally small and auditable:

```text
src/build_dataset_strict.py
        ↓
data/dataset_strict.csv
        ↓
src/strict_model.py
        ↓
src/train_model_strict.py
        ↓
data/model_strict.pkl
data/predictions_strict.csv
data/metrics_strict.json
```

The historical / explanatory branch remains separate:

```text
src/build_dataset.py
        ↓
data/dataset.csv
        ↓
src/train_model.py
        ↓
data/model.pkl
```

Generated datasets, prediction CSVs, live artifacts, production artifacts and API caches are excluded from Git. The public release retains `model_strict.pkl` and `metrics_strict.json`; the reference dataset and prediction CSV remain local-only because they contain third-party market data.

## Repository structure

```text
powercast/
├── app.py
├── README.md
├── METHODOLOGY.md
├── requirements.txt
├── .env.example
├── src/
│   ├── build_dataset.py
│   ├── build_dataset_strict.py
│   ├── data_utils.py
│   ├── entsoe_client.py
│   ├── entsoe_unavailability_client.py
│   ├── macro_features.py
│   ├── odre_client.py
│   ├── preauction_weather_client.py
│   ├── rte_client.py
│   ├── strict_model.py
│   ├── train_model.py
│   ├── train_model_strict.py
│   ├── train_model_production.py
│   ├── forecast_live.py
│   └── weather_client.py
└── tests/
    ├── test_benchmark_regression.py
    ├── test_final_data_integrity.py
    ├── test_live_dst.py
    ├── test_macro_features.py
    ├── test_nuclear_cutoff.py
    ├── test_release_guardrails.py
    ├── test_rte_market_result.py
    └── test_strict_features.py
```

## Quick start

### 1. Install dependencies

Tested release environment: **Python 3.8.8**, **pandas 2.0.3** and **LightGBM 4.3.0**. Direct dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API credentials

Copy the example environment file:

```bash
cp .env.example .env
```

Then provide:

```text
RTE_CLIENT_ID=
RTE_CLIENT_SECRET=
ENTSOE_API_KEY=
```

Never commit `.env`.

### 3. Build the strict dataset

```bash
python src/build_dataset_strict.py \
  --start 2024-03-15 \
  --end 2026-06-01 \
  --output data/dataset_strict.csv
```

The builder also keeps a short price-history warm-up so the 24/48/72/168-hour lags can be created without contaminating the actual training sample.

### 4. Train and save the strict model

```bash
python src/train_model_strict.py \
  --data data/dataset_strict.csv \
  --save
```

This creates the local **benchmark** artifacts. Live forecasting uses the separate production trainer and frozen forecast workflow.

### 4b. Production model and frozen live forecast

The production model is trained separately from the static benchmark and requires an explicitly refreshed strict dataset snapshot. Refresh current TTF/EUA inputs and the A80 cache when rebuilding that snapshot:

```bash
python src/build_dataset_strict.py \
  --start 2024-03-15 \
  --end YYYY-MM-DD \
  --output /path/to/refreshed_dataset_strict.csv \
  --refresh-current \
  --refresh-nuclear

python src/train_model_production.py \
  --data /path/to/refreshed_dataset_strict.csv
```

The local `data/dataset_strict.csv` reference snapshot is not redistributed in the public repository and is not implicitly treated as the latest production-training dataset. Historical caches remain reusable for benchmark and research rebuilds; `--refresh-current` is the explicit production-refresh switch for current TTF/EUA sources.

The generated production model and its metadata are intentionally local and Git-ignored. After cloning the public repository, regenerate the production artifacts with `src/train_model_production.py` using an explicitly refreshed strict dataset before running the live forecaster.

Before the D-1 11:00 Europe/Paris operational cutoff, capture the ENTSO-E A65 demand snapshot:

    python src/forecast_live.py --snapshot-a65

After the cutoff, generate the forecast from that frozen snapshot:

    python src/forecast_live.py

The live forecaster does not refetch the current A65 document after the cutoff. It validates delivery-day completeness, provenance timing and snapshot SHA-256, and fails closed if no valid pre-cutoff snapshot exists.

Live forecasts are frozen once written unless an explicit force option is used. Inputs and metadata are written atomically first; the forecast CSV is written atomically last as the bundle completion marker. A partial bundle is therefore rebuilt instead of being mistaken for a complete frozen forecast. Production artifacts and live outputs are local/generated files and are not the benchmark artifacts reported above.

### 5. Run the dashboard

```bash
streamlit run app.py
```

## Tests

Run:

```bash
python -m pytest -q
```

The current test suite protects the highest-risk methodological details:

- exact 32-feature strict allowlist and canonical LightGBM order
- exclusion of accidental or forbidden numeric inputs
- point-in-time A80 revision and cancellation handling
- DST-safe 23-hour and 25-hour live delivery grids
- DST-safe 11:00 Europe/Paris live cutoff handling
- explicit TTF horizon failure when a verified contract ID is unavailable

The current release test suite includes benchmark regression, live-data integrity, RTE market-result validation, A80/DST guardrails and strict feature-contract tests. Run `pytest -q` for the authoritative current count.

Direct project dependencies are pinned to explicit versions in `requirements.txt`; compatibility is checked in an isolated Python environment before release.

## Key engineering findings

### Nuclear A80 had to be treated as point-in-time data

PowerCast reconstructs nuclear availability from ENTSO-E A80 documents using document revisions, publication timestamps, cancellations, withdrawals, overlapping outages and unit capacities.

A manual development audit over 96 hourly observations produced:

- MAE: **0 MW**
- maximum error: **0 MW**
- mismatches: **0 / 96**

These empirical figures come from a development-time audit and are **not reproduced by the public release snapshot**; the repository tests reproduce the A80 business rules themselves.

However, current ENTSO-E historical extracts do not expose reliable pre-migration point-in-time A80 publication history for the older period. PowerCast therefore treats historical A80 information as **missing**, rather than falsely interpreting missing documents as zero outages.

### Feature order is part of reproducibility

The LightGBM model uses `colsample_bytree=0.85`. Because column sampling depends on feature position, reordering otherwise identical columns can change the fitted trees even with a fixed random seed.

PowerCast therefore defines an explicit `STRICT_FEATURES` allowlist containing all 32 feature names in canonical order. Extra numeric columns cannot silently enter the model, and missing canonical features fail loudly.

## Limitations

- No untouched final OOS period has yet been evaluated after the full research cycle.
- Extreme negative and high-price hours remain much harder than ordinary positive-price hours.
- The model tends to compress extreme prices toward the mean.
- European system conditions are only partially represented.
- Historical pre-migration A80 point-in-time information cannot be fully reconstructed from the current ENTSO-E API.
- TTF monthly contract identifiers are explicitly mapped; the pipeline fails fast when a delivery date requires an unconfigured verified contract.
- Historical ODRÉ demand values are J-1 forecasts, but the exact revision that existed at 11:00 cannot be reconstructed from the current historical source.
- Live demand uses ENTSO-E A65, creating a documented train/live source shift. PowerCast records the returned A65 document ID, revision, document `createdDateTime` and retrieval time in the frozen forecast metadata, while explicitly not claiming that the API reconstructs the exact historical revision available at 11:00.

## Project philosophy

PowerCast repeatedly faced a trade-off between a better backtest score and a more defensible forecasting methodology.

When those objectives conflicted, the project kept the **defensible methodology**.

The main result is therefore not a leaderboard number. It is a forecasting pipeline whose timing assumptions, leakage controls, rejected experiments, validation protocol and known limitations are explicit and auditable.

For the detailed methodology and experiment log, see [`METHODOLOGY.md`](METHODOLOGY.md).
