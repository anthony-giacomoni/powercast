# PowerCast

PowerCast forecasts French day-ahead electricity prices with LightGBM. Most of the work went into one question: which inputs could someone really have had on the morning before the auction?

The repo has a strict benchmark model with 32 features, a separate model and workflow for live forecasts, and a Streamlit dashboard. An older model that uses realised data is kept in `legacy/`. It reaches about 14.1 €/MWh RMSE, which is exactly why I don't treat it as a forecast.

## Results

Evaluation uses a fixed chronological split: 13 December 2025 to the end of May 2026, 3,862 hourly observations. The 13,923 rows before that are the training set.

PowerCast reaches 30.11 €/MWh RMSE and 20.27 MAE. The 24h lag baseline, which simply repeats the price from 24 hours earlier, reaches 39.96 and 26.45. RMSE is 24.6% lower.

I looked at this split while comparing features and model variants. It never went into `fit()`, but I wouldn't call 30.11 an unbiased out-of-sample number. It is a development benchmark.

## Inputs

- demand: the J-1 forecast, from ODRÉ for the history and from ENTSO-E (A65) for live runs
- weather: ECMWF IFS forecasts from the 00Z run of D-1 via Open-Meteo, or the D-2 12Z run when the archived 00Z run is missing. Temperature, wind at 10 m and 100 m, cloud cover and shortwave radiation, plus degree-day, wind and solar proxies
- prices: lags at 24, 48, 72 and 168 hours, and the previous day's mean
- fuel and carbon: TTF futures, EUA auction prices and gas-plant cost proxies built from them
- calendar variables
- nuclear fleet capacity and ENTSO-E A80 outage data

Realised generation, realised weather, the one-hour price lag, A69 renewable forecasts, scheduled exchanges and transfer capacity are not used. [METHODOLOGY.md](METHODOLOGY.md) explains why for each one.

Column order matters. LightGBM samples columns (`colsample_bytree = 0.85`), so reordering them changes the trees even with a fixed seed. The 32 features are an explicit ordered list in `strict_model.py`. Extra columns are ignored and a missing one raises an error.

## Timing limits

The historical demand forecast comes from ODRÉ, and that source has no revision metadata. I can't prove which version was visible at 11:00 on a given day. The strict dataset follows a pre-auction information policy, and it is not a perfect point-in-time reconstruction.

Live runs use ENTSO-E A65 instead. The forecast is captured before the 11:00 Europe/Paris cutoff, saved with its document ID, revision, creation and retrieval times and a file hash, and the forecaster refuses to run on a missing, late or mismatched snapshot. Training and live inference do use different demand sources.

## Nuclear data

A80 outage documents get updated, cancelled, withdrawn and overlapped, so rebuilding what was known at the cutoff takes more than downloading the latest file. I checked 96 hourly observations over four dates by hand during development and found no mismatch. The public release doesn't reproduce that check, only the revision and cancellation rules in the tests.

The current API doesn't expose old publication history. Before 12 November 2025 the A80 feature is therefore missing, not 0 MW. In the saved model it accounts for about 0.002% of the total gain. The other nuclear features are built from nominal fleet capacity, which is constant in the training data, so in practice they are transformations of the demand forecast.

## Live forecasting

The live forecaster doesn't use the benchmark model. It loads a separate production model trained on all admissible rows, and that model isn't in the repo. Build a refreshed dataset and train it first:

    python src/build_dataset_strict.py --start 2024-03-15 --end YYYY-MM-DD --output /path/to/refreshed_dataset_strict.csv --refresh-current --refresh-nuclear
    python src/train_model_production.py --data /path/to/refreshed_dataset_strict.csv

Then capture the A65 snapshot before the cutoff and forecast after it:

    python src/forecast_live.py --snapshot-a65   # before 11:00 Europe/Paris on D-1
    python src/forecast_live.py                  # after the cutoff, uses the frozen snapshot

23-hour and 25-hour days around daylight-saving changes are handled. The dashboard shows frozen forecasts next to the published RTE day-ahead market results.

## Running it

Reference environment: Python 3.8.8, pandas 2.0.3, LightGBM 4.3.0.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env    # then fill RTE_CLIENT_ID, RTE_CLIENT_SECRET, ENTSOE_API_KEY
    python src/build_dataset_strict.py --start 2024-03-15 --end 2026-06-01 --output data/dataset_strict.csv
    python src/train_model_strict.py --data data/dataset_strict.csv --save
    streamlit run app.py
    python -m pytest -q

The tests are aimed at the mistakes I wanted the pipeline to fail loudly on: stray model inputs, missing features, A80 revisions and cancellations, target-day information, daylight-saving days, the 11:00 cutoff, bad A65 snapshots, malformed RTE results and unsupported TTF contracts.

The reference dataset and prediction CSV aren't published because they contain third-party market data. The repo keeps the frozen model (`data/model_strict.pkl`) and the aggregated metrics, and [METHODOLOGY.md](METHODOLOGY.md) lists the SHA-256 hashes of the reference artifacts.

## Data sources

ODRÉ (historical J-1 demand forecasts), the ENTSO-E Transparency Platform (day-ahead prices, A65 demand forecasts, A80 outage documents), the Open-Meteo Single Runs API (archived ECMWF IFS forecasts; weather data by Open-Meteo.com), ICE via DBnomics (Dutch TTF futures), EEX (EUA primary auctions) and RTE (live French market data in the dashboard). All of them stay under their providers' terms.

## Limitations

- No untouched final out-of-sample period exists after the full research cycle.
- The 11:00 revision of the historical demand forecast can't be reconstructed, and older A80 states can't always be either.
- Training uses ODRÉ demand, live inference uses ENTSO-E A65.
- Extreme prices are the weak spot. The model tends to predict too high when prices collapse and too low during large spikes.
- Cross-border conditions are only partly represented.
- TTF contracts are mapped by hand, so a delivery date that needs an unknown contract fails instead of guessing.
- The verified EUA source covers 2026 and will need an update for later years.

Experiment history and the timing assumption behind each feature are in [METHODOLOGY.md](METHODOLOGY.md).