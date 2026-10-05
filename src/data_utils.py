import pandas as pd


def resample_hourly(df, agg="mean"):
    """
    Resample numeric data to hourly frequency.

    Infra-hourly values are converted to numeric first and then
    aggregated, avoiding silent loss of columns inferred as object.
    """
    if df.empty or "datetime" not in df.columns:
        return df

    working = df.set_index(
        "datetime"
    ).copy()

    for col in working.columns:
        working[col] = pd.to_numeric(
            working[col],
            errors="coerce",
        )

    numeric_cols = working.select_dtypes(
        include="number"
    ).columns

    return (
        working[numeric_cols]
        .resample("h")
        .agg(agg)
        .reset_index()
    )
