# PowerCast Methodology

This document describes the strict pre-auction branch of PowerCast: its information set, feature engineering, validation protocol, nuclear-availability audit, rejected experiments and known limitations.

## 1. Forecasting objective

PowerCast forecasts French day-ahead electricity prices for delivery day **D** using only information that could realistically have been available on **D-1 before the market auction**.

The central constraint is point-in-time admissibility:

> A variable is allowed only if its value, publication or forecast version could have been observed before the forecasting cutoff.

This rule is applied to weather forecasts, demand forecasts, nuclear availability, fuel and carbon prices, historical electricity prices and all candidate market-system variables.

The strict pipeline is intentionally different from the historical / explanatory model, which uses contemporaneous realized information and is not considered operational.

## 2. Final strict dataset

The final strict dataset contains 20 raw columns before feature engineering. It is produced by `src/build_dataset_strict.py` and then transformed by `src/strict_model.py` into 32 model features.

### 2.1 French day-ahead price

Historical French day-ahead prices are the model target and are also used to create only genuinely historical price features:

- `price_lag_24h`
- `price_lag_48h`
- `price_lag_72h`
- `price_lag_168h`
- `previous_day_price_mean`

The strict branch does **not** use `price_lag_1h`, because such recent information is inconsistent with the intended pre-auction forecasting horizon.

The dataset builder retains an eight-day price-only warm-up before the strict feature period so the longest 168-hour lag can be constructed without adding earlier observations to the usable training sample.

### 2.2 D-1 demand forecast

The main demand input is:

- `consumption_forecast_j1`

The historical benchmark uses the ODRÉ J-1 forecast field and therefore avoids realized consumption. However, the current historical source does not expose enough revision metadata to prove that each stored value is exactly the revision that existed at 11:00 on D-1.

Live inference uses ENTSO-E A65 total-load forecasts. This creates a documented train/live source shift. Operationally, A65 is captured before the D-1 11:00 Europe/Paris cutoff and frozen locally; inference after the cutoff loads only that validated snapshot.

A UTC/local-time boundary bug was explicitly audited during the final refactor. The builder now loads the UTC month required by the Europe/Paris delivery interval, preventing the first local hours of a month from disappearing when they fall in the previous UTC month.

### 2.3 Pre-auction ECMWF weather

Weather features are based on a fixed pre-auction ECMWF forecast run rather than realized meteorological data.

Raw weather inputs include:

- `preauction_temperature_c`
- `preauction_wind_speed_10m_kmh`
- `preauction_wind_speed_100m_kmh`
- `preauction_cloud_cover_pct`
- `preauction_shortwave_radiation_wm2`

Weather is aggregated across weighted French locations.

Derived features include:

- heating degree days (`preauction_hdd`)
- cooling degree days (`preauction_cdd`)
- `solar_weather_pressure`
- `wind100_cubic_proxy`
- `wind_weather_pressure`
- `preauction_hdd_x_weekday`

Realized weather is explicitly excluded from the strict model.

## 3. Nuclear availability — ENTSO-E A80

Nuclear availability required the most extensive point-in-time audit in the project.

### 3.1 Reconstruction logic

PowerCast uses ENTSO-E A80 unavailability documents together with the nuclear unit catalogue.

The production reconstruction handles:

- document publication timestamps
- multiple revisions of the same document
- cancelled documents (`A09`)
- withdrawn documents (`A13`)
- overlapping outages
- unit-level nominal capacity
- intra-hour outage changes through time weighting

The latest admissible revision known by the historical cutoff is used.

### 3.2 Audit result

A manual development audit checked the reconstruction on four dates between November 2025 and July 2026, covering 96 hourly observations. These empirical figures are retained as development evidence but are **not reproduced by the public release snapshot**; the release test suite instead reproduces the underlying A80 revision/cancellation/overlap/time-weighting rules.

Results:

- MAE: **0 MW**
- maximum absolute error: **0 MW**
- mismatches: **0 / 96**

An apparent 3,161 MW discrepancy on 15 August 2026 was traced entirely to cancelled A09 rows and confirmed that the production exclusion logic was correct.

### 3.3 Historical ENTSO-E migration problem

A major limitation was discovered in the currently exposed historical A80 data.

Although delivery periods can extend further back, useful `created_doc_time` values for historical point-in-time reconstruction appear only around the later 2025 migration period. Earlier delivery periods cannot be reliably reconstructed as they would have appeared at the original D-1 cutoff.

Therefore, historical absence of an A80 document cannot honestly be interpreted as:

> 0 MW unavailable.

It may instead mean that the original historical point-in-time document is no longer exposed by the current API.

### 3.4 Clean nuclear representation

The final model therefore separates nuclear information into always-valid nominal context and A80-specific point-in-time information.

Always available:

- `nuclear_nominal_capacity_safe`
- `nuclear_nominal_to_demand_ratio_safe`
- `nuclear_nominal_demand_margin_safe`

A80-specific:

- `nuclear_a80_unavailable_safe`

The `_safe` suffix is an internal modelling convention meaning that a feature is admitted by the strict information-timing policy. It does not mean that every value is a perfect historical point-in-time reconstruction. In particular, the three nominal-capacity features should be interpreted as **fleet nominal context**, not as reconstructed historical available capacity.

`nuclear_a80_unavailable_safe` is intentionally missing before **12 November 2025 (Europe/Paris)**, then uses the reconstructed point-in-time outage value after that boundary.

LightGBM handles this missing value natively.

This treatment slightly worsened some headline historical scores compared with the previous assumption of zero outages, but removed a methodologically invalid historical fiction.

### 3.5 Cutoff convention

The strict builder reconstructs A80 at **D-1 11:00 Europe/Paris**.

During manual development QA, the previous 12:00 implementation was rebuilt at 11:00 across the complete strict dataset. The two cutoffs differed for **42 hourly observations**, with a maximum A80 unavailable-capacity difference of **873 MW**. These figures are historical development-audit results and are **not independently reproduced by the public release snapshot**.

The benchmark was therefore rebuilt at 11:00 rather than retaining the later snapshot. The D-1 11:00 cutoff is now the canonical historical A80 convention used by the strict benchmark.

## 4. TTF natural gas

Dutch TTF monthly futures are used as a proxy for European gas costs and thermal marginal economics.

PowerCast uses the next-calendar-month contract and only observations dated strictly before the local cutoff date, preventing same-day settlement leakage.

Features include:

- `ttf_front_month_eur_mwh`
- `ttf_change_5obs_pct`
- `ccgt_cost_proxy_eur_mwh`
- `ocgt_cost_proxy_eur_mwh`

Contract identifiers are explicitly mapped in `src/macro_features.py`. They are never guessed or generated automatically. A fail-fast horizon guard stops dataset construction or live inference when a required next-calendar-month contract does not have a verified mapping.

## 5. EUA carbon prices

European carbon prices are built from official EEX primary-auction reports.

For each delivery date, the pipeline uses the latest EUA auction whose timestamp is strictly before the forecasting cutoff.

Features include:

- `eua_auction_eur_tco2`
- `eua_change_5auctions_pct`
- the CCGT and OCGT cost proxies listed above

The code uses a conservative fallback when an auction report does not expose a usable time: it does not assume that same-day information was available early.

## 6. Calendar features

The final strict model also includes calendar information that is known in advance:

- hour
- day of week
- weekend flag
- month
- French public-holiday flag

These are combined with weather and demand context where useful.

## 7. Final feature set

The current strict model uses **32 features**.

`src/strict_model.py` defines an explicit `STRICT_FEATURES` allowlist containing the exact feature names and canonical order. Extra numeric dataset columns are ignored rather than silently entering the model, and missing canonical features raise an error.

The four clean nuclear features occupy the final four positions in the feature vector:

1. `nuclear_nominal_capacity_safe`
2. `nuclear_nominal_to_demand_ratio_safe`
3. `nuclear_nominal_demand_margin_safe`
4. `nuclear_a80_unavailable_safe`

This ordering is not cosmetic; it is part of model reproducibility.

## 8. Model

The final model is LightGBM with frozen hyperparameters:

```python
lgb.LGBMRegressor(
    objective="regression",
    n_estimators=324,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)
```

The frozen configuration was chosen during the 2025 historical development phase. The public release preserves the benchmark implementation, frozen model, aggregated metrics and canonical reference hashes. The raw reference dataset and prediction CSV remain local-only because they contain third-party market data.

## 9. Reproducibility and feature ordering

The research sections below are retained as a record of the development process. They are not presented as a fully reproducible experiment suite. The public release preserves the frozen 32-feature benchmark specification, fixed configuration, model, aggregated metrics, canonical hashes and tests; the reference dataset and prediction CSV remain local-only.

A subtle LightGBM issue was discovered during final integration.

Because the model uses:

```text
colsample_bytree = 0.85
```

LightGBM samples a subset of columns while fitting trees. Changing the input-column order can therefore alter the sampled feature set even when `random_state=42` is unchanged.

During integration, an otherwise identical model moved from approximately 27.185 to 27.350 €/MWh RMSE only because feature order changed.

PowerCast now enforces a canonical order and protects it with automated tests.

## 10. Validation protocols

PowerCast reports two separate protocols.

### 10.1 Static strict split

This is the fixed static validation protocol used by the strict benchmark. The public release retains the frozen model, aggregated metrics and canonical hashes, while the raw reference dataset and prediction CSV remain local-only. The Streamlit live forecast path uses a separate production model.

Current result:

- Features: **32**
- Train rows: **13,923**
- Test rows: **3,862**
- RMSE: **30.11 €/MWh**
- MAE: **20.27 €/MWh**
- RMSE / mean price: **46.1%**
- 24-hour lag baseline RMSE: **39.96 €/MWh**
- 24-hour lag baseline MAE: **26.45 €/MWh**
- 168-hour lag baseline RMSE: **52.58 €/MWh**
- 168-hour lag baseline MAE: **37.80 €/MWh**

The test interval begins at **13 December 2025 00:00 Europe/Paris** (`2025-12-12T23:00:00Z`) and runs to the end of May 2026. The split is deliberately aligned to a local delivery-day boundary rather than splitting a delivery day.

For a fixed dataset snapshot, the training procedure is deterministic and reproducible from `train_model_strict.py`.

Reference benchmark snapshot SHA-256 values:

- `dataset_strict.csv`: `4723efa3d456a96a4f0f61b2b1b1f9179065ff40c0a25f3494614697a7476673`
- `model_strict.pkl`: `6b51327fca4414a76102eeeb6d2bcd700dc9917fbef0180f2193b660d9a0722a`
- `predictions_strict.csv`: `c0e97085dcd6cd6cf5f12a655ab2dcac1ea2ea4ca46ab2847e1ba71eedf0dce2`
- `metrics_strict.json`: `50471ceb98719efbc58066a3cc0798fa487ae0cd97eb6ffa86bc9cb6da65005a`

These hashes identify the exact local snapshot used for the reported result. Dataset and prediction hashes are retained for provenance even though those CSVs are not redistributed publicly. Because upstream APIs may revise historical data, rebuilding all raw inputs later is not claimed to produce byte-identical source data.

### 10.2 Monthly expanding retraining

A second protocol was used to approximate operational deployment more closely.

For each evaluation month:

1. all clean historical observations before that month were used for training;
2. LightGBM was retrained;
3. the following month was predicted;
4. future observations were never included in that month's training set.

Results:

| Month | RMSE | MAE |
|---|---:|---:|
| June 2026 | 23.687 | 18.236 |
| July 2026 | 27.734 | 22.515 |
| August 2026 | 29.287 | 21.636 |
| **Combined** | **27.185** | **20.924** |

Combined observations: **1,957 hours**.

This is the official **strict clean development baseline**.

However, June-August 2026 was consulted repeatedly during later feature, audit and integration work. It is therefore **not an untouched final OOS test set** and must not be presented as one.

## 11. Error analysis

On the June-August 2026 development benchmark:

- global bias (`actual - prediction`): **+9.074 €/MWh**
- June bias: **+6.25 €/MWh**
- July bias: **+7.53 €/MWh**
- August bias: **+12.88 €/MWh**

The model compresses prices toward the mean:

- actual 0-25 €/MWh: predictions are too high on average
- actual 100-150 €/MWh: predictions are too low
- actual 150+ €/MWh: predictions are too low

The evening peak is particularly difficult, especially around 19:00-21:00. High-demand and extreme-wind regimes also remain harder than ordinary conditions.

This pattern suggests that remaining errors are more structural than a simple global calibration problem.

## 12. Features and experiments deliberately rejected

The research phase tested a large number of additions. The final strict model kept only changes that generalized and passed information-timing requirements.

### Residual Ridge correction

A residual linear model improved one static benchmark but performed worse in out-of-fold validation and later evaluation.

**Rejected.**

### European lagged neighboring prices

Small RMSE improvement, worse MAE and insufficient gain relative to complexity.

**Rejected.**

### European pre-auction weather

Belgium, Germany, Netherlands, UK, Spain and Italy weather produced only about 0.16 €/MWh RMSE improvement in one development experiment.

**Not integrated.**

### Rolling training windows

Windows of 6, 9, 12, 15 and 18 months were tested.

The expanding window remained best.

**Rejected.**

### Full previous-day price curve

Additional curve information worsened validation.

**Rejected.**

### Day-ahead curve-shape features

Profile-shape features degraded the development benchmark.

**Rejected.**

### Recency weighting

Multiple weighting schemes failed to improve the historical development checks consistently.

**Rejected.**

### ENTSO-E A72 hydro

Historical gain was negligible and 2026 development performance deteriorated materially.

**Rejected.**

### Affine post-calibration

Attempted to correct price compression but strongly worsened the historical development checks.

**Rejected.**

### Hour-specific bias correction

Best gain was only about 0.03 €/MWh RMSE and insufficiently consistent across folds.

**Rejected.**

### Hour-regime specialist LightGBMs

Separate regime models for selected hours degraded overall performance.

**Rejected.**

### Tail-weighted training

Extra emphasis on extreme-price observations produced only a very small and unstable improvement.

**Rejected.**

## 13. Inputs explicitly excluded from the strict baseline

### ENTSO-E A69 renewable forecast

A69 improved one static experiment, but publication timing could not be guaranteed to precede the strict cutoff across the full historical sample.

**Excluded.**

### ENTSO-E A09 scheduled commercial exchanges

These are outputs of the day-ahead market-clearing process. Using them to predict the same market-clearing price would constitute target leakage.

**Excluded.**

### ENTSO-E A61 transfer capacity

A small gain was observed, but historical revision timing was not audited strongly enough for strict point-in-time use.

**Excluded.**

### Realized generation and realized weather

Highly informative historically, but unavailable at the actual forecasting decision point.

**Excluded.**

## 14. Automated tests

The final backend includes automated regression tests for the highest-risk methodological details.

Current suite:

- `tests/test_strict_features.py`
  - verifies the clean nuclear split boundary
  - verifies the exact 32-feature allowlist and canonical ordering
  - verifies that accidental numeric inputs cannot silently enter the model
  - verifies that missing canonical features fail loudly

- `tests/test_nuclear_cutoff.py`
  - verifies that revisions published after the cutoff are invisible
  - verifies that a cancelled latest revision removes the outage

- `tests/test_live_dst.py`
  - verifies 23-hour spring and 25-hour autumn delivery grids
  - verifies the 11:00 Europe/Paris live cutoff across DST transitions
  - verifies that repeated autumn 02:00 hours remain distinct instants

- `tests/test_macro_features.py`
  - verifies next-calendar-month TTF selection
  - verifies explicit failure when a verified TTF mapping is missing

- `tests/test_release_guardrails.py`
  - verifies the different training/live missing-value contracts
  - verifies that required live features cannot be silently missing
  - verifies atomic CSV and JSON artifact writes
  - verifies SHA-256 provenance helpers

Current status: the full release test suite passes in the documented environment.

## 15. Known limitations

### No untouched final test set

June-August 2026 was originally future data relative to earlier training work, but it was later inspected repeatedly. A future period that has never been consulted is required for the final OOS claim.

### Extreme prices remain difficult

Negative and very high prices dominate the tail RMSE. The model still tends to overpredict some very low prices and underpredict high-price events.

### European system information remains incomplete

French day-ahead prices depend heavily on neighboring markets, renewable production, interconnector conditions and continental scarcity. Some candidate European signals were tested but were either too weak or insufficiently auditable for the strict baseline.

### Historical A80 cannot be perfectly reconstructed before migration

The current ENTSO-E API does not expose enough historical publication metadata to recreate all older point-in-time outage states. PowerCast treats this uncertainty explicitly as missing information.

### TTF contract maintenance

TTF monthly futures are mapped to explicit DBnomics / ICE series identifiers. The verified mapping now reaches the March 2027 contract, supporting February 2027 deliveries under the next-calendar-month convention. A delivery requiring an unmapped contract fails explicitly before any fetch. Future contract IDs must be independently verified before being added.

### Demand-forecast point-in-time limitation

The historical ODRÉ input is a J-1 demand forecast rather than realized load, but the source does not prove the exact historical revision that existed at 11:00 for every delivery date.

Live inference uses ENTSO-E A65. PowerCast captures the operational A65 snapshot before D-1 11:00 Europe/Paris, freezes its provenance and hash, and refuses strict inference when no valid pre-cutoff snapshot exists. This does not claim reconstruction of exact historical A65 revisions for past dates.

### Production-model provenance

The production model is deliberately separate from the static benchmark.

The final pre-release production snapshot contains **21,791 raw hourly rows** covering `2024-03-06 23:00 UTC` through `2026-08-31 21:00 UTC`. Its SHA-256 is:

`478935b53e02615ec3f5b54e97b371fb6d1e8caac472b741f4c9303db91466f6`

ODRÉ provides no J-1 demand observations for July-August 2026 in the current source, so those later source rows are not admissible for training. The production model therefore uses **18,368 rows** and is trained through **2026-06-30 21:00 UTC**.

The generated production model SHA-256 is:

`b3294445a6e8ef5b5dd68224465ac456377b1bf5b2a62fb3d057ba15c90140a5`

`train_model_production.py` now records the dataset hash, byte size, raw row count, source period, actual training period, exact feature list and Python/pandas/LightGBM/joblib versions in production metadata.

Generated production artifacts remain local and are not confused with the static benchmark artifacts.

### Missing historical price lags in live inference

The contracts are now centralized in `strict_model.py`: training allows intentional missingness only for the historical A80 feature, whereas live inference additionally permits genuine upstream gaps in the 24h/48h/72h/168h price lags. Other missing model inputs fail explicitly, and this difference is covered by automated tests.

Benchmark training uses rows with the required historical feature inputs available, while live inference may occasionally retain missing long-horizon price lags when an upstream A44 historical gap cannot be repaired defensibly. LightGBM handles these missing values natively; PowerCast does not interpolate or fabricate the missing wholesale prices.

### Dependency reproducibility

Direct project dependencies are pinned to explicit versions in `requirements.txt`. A clean isolated environment is used as the final compatibility check rather than relying only on the developer's existing Anaconda installation.

### Frozen-artifact durability

Live CSV/JSON artifacts are written to temporary files and moved into place with `os.replace()`. This does not create an industrial transactional storage system, but prevents a normal interrupted write from leaving a partially written forecast artifact.


### Final release guardrails

The final release adds several fail-closed behaviours around reproducibility and live auditability:

- `holidays==0.58` is a mandatory canonical feature dependency. If it is unavailable, strict feature engineering fails instead of silently replacing French holidays with zeros.
- the public regression suite validates the retained benchmark artifacts and feature contract; when the private reference CSVs are locally available, it also verifies their canonical hashes, the 13,923/3,862 chronological split, RMSE/MAE and both lag baselines;
- RTE market calls used by the dashboard can propagate a typed API/network error, so API failure is distinct from a genuinely unpublished market result;
- a live frozen bundle is complete only when forecast, inference snapshot and metadata all exist; before any rebuild or `--force` recalculation, an existing forecast marker is removed, then inputs and metadata are written first and the forecast CSV is written last as the completion marker;
- the low-level A80 reconstruction helper now defaults to D-1 **11:00 Europe/Paris** and constructs that cutoff from the civil previous date, including across DST;
- the currently verified EEX EUA annual source covers 2026. A run requiring 2026 fails if the current workbook contains no parseable 2026 auction rows, and delivery dates beyond 2026 fail until a verified newer annual source is configured; stale prior-year auctions are never silently carried forward;
- live ENTSO-E A65 demand remains a documented train/live source shift. The operational workflow captures A65 before D-1 11:00 Europe/Paris, freezes its provenance and snapshot hash, and fails closed if no valid pre-cutoff snapshot exists.

## 16. Research status

The feature-research phase is currently frozen.

The focus is now on:

- maintaining the strict data pipeline
- keeping tests and reproducibility intact
- documenting the model honestly
- evaluating on a future untouched period

The project deliberately avoids continued optimization against the already-inspected June-August 2026 benchmark.

## 17. Core methodological lesson

PowerCast repeatedly encountered cases where a feature or assumption improved backtest performance but weakened the credibility of the forecast.

The final rule was simple:

> When a stronger score conflicts with defensible information timing, keep the defensible methodology.

That decision is why the strict model is intentionally separated from the much stronger historical/explanatory model.
