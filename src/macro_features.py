# ============================================================
# PowerCast — strict pre-auction TTF + EUA macro features
#
# TTF:
#   - ICE Dutch TTF monthly futures via DBnomics
#   - use next-calendar-month contract (front-month during delivery month)
#   - only observations dated STRICTLY BEFORE the local cutoff date
#     so the same-day settlement can never leak past 11:00 D-1
#
# EUA:
#   - official EEX primary-auction reports
#   - use latest auction whose UTC timestamp is strictly before cutoff
#
# Output is separate; input is never overwritten.
# Python 3.8 compatible.
# ============================================================

import io
import os
import re
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

DBNOMICS_BASE = "https://api.db.nomics.world/v22/series/ICE/DUTCH_TTF_GAS_FUTURES"
EUA_HISTORY_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
    "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
    "emission-spot-primary-market-auction-report-2012-2025-data.zip"
)
EUA_CURRENT_URL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-2026-data.xlsx"
)

EUA_CURRENT_YEAR = 2026

# Exact monthly-contract codes confirmed from the ICE/DBnomics catalogue.
TTF_CONTRACTS = {
    "2024-02": "D.5714606", "2024-03": "D.5733529",
    "2024-04": "D.5756699", "2024-05": "D.5776658", "2024-06": "D.5786631",
    "2024-07": "D.5786630", "2024-08": "D.5786628", "2024-09": "D.5786634",
    "2024-10": "D.5786633", "2024-11": "D.5786632", "2024-12": "D.5786629",
    "2025-01": "D.5798617", "2025-02": "D.5815810", "2025-03": "D.5844634",
    "2025-04": "D.5863238", "2025-05": "D.5878892", "2025-06": "D.5899671",
    "2025-07": "D.5927115", "2025-08": "D.5959809", "2025-09": "D.5980688",
    "2025-10": "D.6006994", "2025-11": "D.6036837", "2025-12": "D.6058601",
    "2026-01": "D.6089679", "2026-02": "D.6117262", "2026-03": "D.6142642",
    "2026-04": "D.6164787", "2026-05": "D.6187520", "2026-06": "D.6214891",
    "2026-07": "D.6243134",
    "2026-08": "D.6277386",
    "2026-09": "D.6277387",
    "2026-10": "D.6277388",
    "2026-11": "D.6277389",
    "2026-12": "D.6277390",
    "2027-01": "D.6277391",
    "2027-02": "D.6277392",
    "2027-03": "D.6277393",
}


def _next_month_key(d):
    y, m = d.year, d.month
    if m == 12:
        return "%04d-01" % (y + 1)
    return "%04d-%02d" % (y, m + 1)


def _download(url, path, timeout=120):
    headers = {"User-Agent": "Mozilla/5.0 PowerCast/1.0"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def fetch_ttf_contract(code, cache_dir, refresh=False):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "ttf_%s.csv" % code.replace(".", "_"))
    if os.path.exists(path) and not refresh:
        out = pd.read_csv(path)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        return out.dropna(subset=["date", "price"]).sort_values("date")

    url = "%s/%s" % (DBNOMICS_BASE, code)
    r = requests.get(url, params={"observations": 1}, timeout=60)
    r.raise_for_status()
    j = r.json()
    docs = j.get("series", {}).get("docs", [])
    if not docs:
        raise RuntimeError("DBnomics returned no series doc for %s" % code)
    doc = docs[0]
    periods = doc.get("period", [])
    values = doc.get("value", [])
    if not periods or not values or len(periods) != len(values):
        raise RuntimeError("DBnomics observations malformed for %s" % code)

    out = pd.DataFrame({"date": pd.to_datetime(periods, errors="coerce"), "price": values})
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["date", "price"]).copy()
    out["date"] = out["date"].dt.date
    out = out.sort_values("date")
    out.to_csv(path, index=False)
    return out



def validate_ttf_contract_keys(required_keys):
    """
    Fail before any network/cache work if PowerCast needs a TTF
    monthly contract whose verified ICE/DBnomics ID is not configured.

    Contract IDs must never be guessed or generated automatically.
    """
    required = sorted(set(required_keys))

    missing = [
        key
        for key in required
        if key not in TTF_CONTRACTS
    ]

    if missing:
        raise RuntimeError(
            "Missing verified TTF contract mapping for: %s. "
            "PowerCast cannot safely build delivery dates requiring "
            "these contracts. Add verified ICE/DBnomics contract IDs "
            "to TTF_CONTRACTS before continuing."
            % ", ".join(missing)
        )

    return required

def fetch_all_ttf(required_keys, cache_dir, refresh_current=False):
    required_keys = validate_ttf_contract_keys(required_keys)
    series = {}
    for i, key in enumerate(sorted(required_keys)):
        code = TTF_CONTRACTS.get(key)
        if not code:
            raise RuntimeError("No TTF contract code configured for %s" % key)
        try:
            series[key] = fetch_ttf_contract(
                code,
                cache_dir,
                refresh=refresh_current,
            )
            print("  TTF %s -> %d observations" % (key, len(series[key])))
        except Exception as e:
            raise RuntimeError("TTF fetch failed for %s (%s): %s" % (key, code, e))
        if i < len(required_keys) - 1:
            time.sleep(0.15)
    return series


def _norm_col(x):
    s = str(x).strip().lower()
    s = s.replace("€", "eur").replace("₂", "2")
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_time_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "hour") and hasattr(v, "minute"):
        return pd.Timedelta(hours=int(v.hour), minutes=int(v.minute), seconds=int(getattr(v, "second", 0)))
    if isinstance(v, (int, float)) and 0 <= float(v) < 1:
        return pd.Timedelta(days=float(v))
    s = str(v).strip()
    if not s:
        return None
    try:
        td = pd.to_timedelta(s)
        return td
    except Exception:
        return None


def parse_eua_excel_bytes(blob, source_name):
    rows = []
    xls = pd.ExcelFile(io.BytesIO(blob))

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)

        header_row = None
        for i in range(min(40, len(raw))):
            norms = [_norm_col(v) for v in raw.iloc[i].tolist()]

            has_date = any(
                n == "date" or "auction date" in n
                for n in norms
            )

            has_price = any(
                "auction price" in n or
                "auction clearing price" in n
                for n in norms
            )

            if has_date and has_price:
                header_row = i
                break

        if header_row is None:
            continue

        df = pd.read_excel(
            xls,
            sheet_name=sheet,
            header=header_row
        )

        cmap = {_norm_col(c): c for c in df.columns}

        date_col = next(
            (
                orig for norm, orig in cmap.items()
                if norm == "date" or "auction date" in norm
            ),
            None
        )

        price_col = next(
            (
                orig for norm, orig in cmap.items()
                if "auction price" in norm
                or "auction clearing price" in norm
            ),
            None
        )

        time_col = next(
            (
                orig for norm, orig in cmap.items()
                if norm == "time"
                or ("time" in norm and "utc" in norm)
            ),
            None
        )

        status_col = next(
            (
                orig for norm, orig in cmap.items()
                if norm == "status"
            ),
            None
        )

        if date_col is None or price_col is None:
            continue

        for _, r in df.iterrows():

            if status_col is not None:
                status = str(r.get(status_col, "")).strip().lower()
                if status and status != "nan" and "successful" not in status:
                    continue

            d = pd.to_datetime(
                r.get(date_col),
                errors="coerce"
            )

            p = pd.to_numeric(
                r.get(price_col),
                errors="coerce"
            )

            if pd.isna(d) or pd.isna(p):
                continue

            td = (
                _parse_time_value(r.get(time_col))
                if time_col is not None
                else None
            )

            if td is None:
                # Conservative fallback:
                # never assume a same-day auction was known early.
                td = pd.Timedelta(hours=23, minutes=59)

            ts = pd.Timestamp(d.date()) + td
            ts = ts.tz_localize("UTC")

            rows.append({
                "auction_ts_utc": ts,
                "eua_price": float(p),
                "source": "%s:%s" % (source_name, sheet),
            })

    return pd.DataFrame(rows)



def validate_eua_delivery_dates(
    delivery_dates,
):
    """
    Fail closed when delivery dates require an EUA auction
    source newer than the verified current-year EEX report.

    PowerCast must never silently carry the final 2026 EUA
    auction forward into an unsupported 2027 pipeline.
    """
    unsupported = sorted({
        d.year
        for d in delivery_dates
        if d.year > EUA_CURRENT_YEAR
    })

    if unsupported:
        raise RuntimeError(
            "Missing verified EEX EUA auction source for "
            "delivery year(s): %s. "
            "PowerCast currently supports EUA data through %d "
            "and will not silently carry stale auctions forward."
            % (
                ", ".join(
                    str(y)
                    for y in unsupported
                ),
                EUA_CURRENT_YEAR,
            )
        )

    return delivery_dates

def load_eua(
    cache_dir,
    current_path,
    refresh_current=False,
    require_current_year=True,
):
    os.makedirs(cache_dir, exist_ok=True)
    history_zip = os.path.join(cache_dir, "eua_2012_2025.zip")
    if not os.path.exists(history_zip):
        print("Downloading official EEX EUA history 2012-2025...")
        _download(EUA_HISTORY_URL, history_zip)

    current_xlsx = (
        current_path
        if current_path
        else os.path.join(cache_dir, "eua_2026.xlsx")
    )

    frames = []
    with zipfile.ZipFile(history_zip, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith((".xlsx", ".xls")):
                continue
            try:
                blob = z.read(name)
                part = parse_eua_excel_bytes(blob, name)
                if not part.empty:
                    frames.append(part)
            except Exception as e:
                print("  ⚠️ EUA archive file skipped %s: %s" % (name, e))

    # Historical-only builds through 2025 are fully covered by the
    # immutable archive and should not depend on the current report.
    # Any dataset/live run requiring 2026, however, must fail closed if
    # the current workbook cannot produce genuine 2026 auction rows.
    if require_current_year:
        if refresh_current or not os.path.exists(current_xlsx):
            print("Downloading official EEX EUA 2026 report...")
            _download(EUA_CURRENT_URL, current_xlsx)

        with open(current_xlsx, "rb") as f:
            current = parse_eua_excel_bytes(
                f.read(),
                os.path.basename(current_xlsx),
            )

        if current.empty:
            raise RuntimeError(
                "Current %d EEX EUA report contains no parseable auction rows"
                % EUA_CURRENT_YEAR
            )

        current = current.copy()
        current["auction_ts_utc"] = pd.to_datetime(
            current["auction_ts_utc"],
            utc=True,
            errors="coerce",
        )
        current["eua_price"] = pd.to_numeric(
            current["eua_price"],
            errors="coerce",
        )
        current = current.dropna(
            subset=["auction_ts_utc", "eua_price"]
        )

        current_year_rows = current[
            current["auction_ts_utc"].dt.year
            == EUA_CURRENT_YEAR
        ]

        if current_year_rows.empty:
            raise RuntimeError(
                "Current %d EEX EUA report contains no parseable %d auction rows"
                % (
                    EUA_CURRENT_YEAR,
                    EUA_CURRENT_YEAR,
                )
            )

        frames.append(current)

    if not frames:
        raise RuntimeError("No EUA auction rows could be parsed")

    eua = pd.concat(frames, ignore_index=True)
    eua = eua.dropna(subset=["auction_ts_utc", "eua_price"])
    eua = eua.sort_values("auction_ts_utc").drop_duplicates("auction_ts_utc", keep="last")
    eua = eua.reset_index(drop=True)
    print("Parsed EUA auctions: %d (%s -> %s)" % (
        len(eua), eua["auction_ts_utc"].min(), eua["auction_ts_utc"].max()
    ))
    return eua


def cutoff_for_delivery_date(delivery_date):
    prev = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
    naive = pd.Timestamp(prev.date()) + pd.Timedelta(hours=11)
    return naive.tz_localize("Europe/Paris").tz_convert("UTC")


def lookup_ttf(delivery_date, cutoff_utc, ttf_series):
    key = _next_month_key(delivery_date)
    s = ttf_series.get(key)
    if s is None or s.empty:
        return (np.nan, None, np.nan)
    # Strict conservative rule: ignore any settlement carrying the local cutoff date.
    cutoff_local_date = cutoff_utc.tz_convert("Europe/Paris").date()
    eligible = s[s["date"] < cutoff_local_date]
    if eligible.empty:
        return (np.nan, None, np.nan)
    last = eligible.iloc[-1]
    price = float(last["price"])
    change5 = np.nan
    if len(eligible) >= 6:
        old = float(eligible.iloc[-6]["price"])
        if old != 0:
            change5 = 100.0 * (price / old - 1.0)
    return (price, last["date"], change5)


def lookup_eua(cutoff_utc, eua):
    eligible = eua[eua["auction_ts_utc"] < cutoff_utc]
    if eligible.empty:
        return (np.nan, None, np.nan)
    last = eligible.iloc[-1]
    price = float(last["eua_price"])
    change5 = np.nan
    if len(eligible) >= 6:
        old = float(eligible.iloc[-6]["eua_price"])
        if old != 0:
            change5 = 100.0 * (price / old - 1.0)
    return (price, last["auction_ts_utc"], change5)



def add_macro_features(
    df,
    eua_current="data/cache/macro/eua_auction_2026.xlsx",
    cache_dir="data/cache/macro",
    refresh_current=False,
):
    """
    Add strict pre-auction TTF and EUA features to an existing dataframe.

    The input dataframe must contain a UTC-compatible ``datetime`` column.
    No target-day settlement or post-cutoff EUA auction is used.
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    local = df["datetime"].dt.tz_convert("Europe/Paris")
    delivery_dates = sorted(
        set(local.dt.date)
    )

    validate_eua_delivery_dates(
        delivery_dates
    )

    required_keys = set(
        _next_month_key(d)
        for d in delivery_dates
    )

    print(
        "Fetching/caching %d TTF monthly contracts..."
        % len(required_keys)
    )

    ttf = fetch_all_ttf(
        required_keys,
        cache_dir,
        refresh_current=refresh_current,
    )

    require_current_eua = any(
        d.year == EUA_CURRENT_YEAR
        for d in delivery_dates
    )

    eua = load_eua(
        cache_dir,
        eua_current,
        refresh_current=refresh_current,
        require_current_year=require_current_eua,
    )

    daily = []

    for d in delivery_dates:

        cutoff = cutoff_for_delivery_date(d)

        (
            ttf_price,
            ttf_date,
            ttf_change5,
        ) = lookup_ttf(
            d,
            cutoff,
            ttf,
        )

        (
            eua_price,
            eua_ts,
            eua_change5,
        ) = lookup_eua(
            cutoff,
            eua,
        )

        daily.append({
            "delivery_date": d,
            "ttf_front_month_eur_mwh": ttf_price,
            "ttf_source_date": (
                str(ttf_date)
                if ttf_date
                else None
            ),
            "ttf_change_5obs_pct": ttf_change5,
            "eua_auction_eur_tco2": eua_price,
            "eua_source_ts_utc": (
                str(eua_ts)
                if eua_ts is not None
                else None
            ),
            "eua_change_5auctions_pct": eua_change5,
        })

    macro = pd.DataFrame(daily)

    macro["ccgt_cost_proxy_eur_mwh"] = (
        macro["ttf_front_month_eur_mwh"] / 0.55
        + 0.35 * macro["eua_auction_eur_tco2"]
    )

    macro["ocgt_cost_proxy_eur_mwh"] = (
        macro["ttf_front_month_eur_mwh"] / 0.38
        + 0.55 * macro["eua_auction_eur_tco2"]
    )

    out = df.copy()
    out["delivery_date"] = local.dt.date

    out = out.merge(
        macro,
        on="delivery_date",
        how="left",
    )

    out = out.drop(
        columns=["delivery_date"]
    )

    return out
