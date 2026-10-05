# ============================================================
#  PowerCast — ENTSO-E Historical Prices Client (native XML parser)
#
#  Official pan-European transmission system operator platform — the
#  authoritative source for day-ahead electricity prices, with a full
#  multi-year history (unlike RTE's own API, which only serves
#  today/tomorrow — see rte_client.py).
#
#  IMPORTANT: this does NOT use the entsoe-py library. A raw HTTP test
#  confirmed the ENTSO-E API itself works fine and returns real XML
#  data (verified: France, January 2026, real prices like 95.95,
#  82.47, 64.96 EUR/MWh) — but entsoe-py 0.6.2 (the only version
#  installable on Python 3.8) raises NoMatchingDataError even when the
#  raw API clearly has data. Rather than fight a buggy old dependency,
#  this parses the confirmed-real XML structure directly with the
#  standard library (no extra dependency beyond requests).
#
#  Setup: put your token in .env as ENTSOE_API_KEY (obtained via
#  transparency.entsoe.eu -> My Account -> generate token, after
#  requesting API access by contacting transparency@entsoe.eu with
#  subject "Restful API access").
# ============================================================

import os
from typing import Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ENTSOE_API_KEY")
BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Code EIC de la zone de dépôt France (bidding zone), confirmé par un
# appel réel réussi.
FRANCE_EIC = "10YFR-RTE------C"

# Namespace XML confirmé dans la vraie réponse
NS = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}


class ENTSOEAPIError(RuntimeError):
    """ENTSO-E request failure with credentials/query parameters redacted."""


def _safe_request_failure(label, exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return "%s failed (HTTP %s)" % (label, status)
    return "%s failed (%s)" % (label, type(exc).__name__)



def _parse_price_xml(xml_text: str) -> list:
    """
    Parse le XML de prix day-ahead ENTSO-E (structure confirmée par un
    appel réel) en une liste de dicts {datetime, price_eur_mwh}.

    Structure : un ou plusieurs <TimeSeries>, chacun avec un <Period>
    ayant un <timeInterval><start> et une <resolution> (ex: PT15M ou
    PT60M), et une liste de <Point> avec <position> (index séquentiel
    depuis le début de la période) et <price.amount>.
    """
    root = ET.fromstring(xml_text)
    rows = []

    for timeseries in root.findall("ns:TimeSeries", NS):
        period = timeseries.find("ns:Period", NS)
        if period is None:
            continue

        start_str = period.find("ns:timeInterval/ns:start", NS).text
        resolution = period.find("ns:resolution", NS).text

        # PT15M = 15 minutes, PT60M/PT1H = 1 heure — on convertit en minutes
        if "M" in resolution and "T" in resolution:
            minutes = int(resolution.replace("PT", "").replace("M", ""))
        else:
            minutes = 60  # fallback raisonnable

        period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")

        for point in period.findall("ns:Point", NS):
            position = int(point.find("ns:position", NS).text)
            price = float(point.find("ns:price.amount", NS).text)
            point_time = period_start + timedelta(minutes=minutes * (position - 1))
            rows.append({"datetime": point_time, "price_eur_mwh": price})

    return rows


def get_historical_prices(start: str, end: str, zone_eic: str = FRANCE_EIC) -> Optional[pd.DataFrame]:
    """
    Récupère l'historique de prix day-ahead pour une zone (France par
    défaut) sur une plage de dates. Contrairement à l'API RTE, ENTSO-E
    sert bien du vrai historique (pas seulement demain).

    start / end au format 'YYYY-MM-DD'. Retourne un DataFrame avec
    colonnes datetime (UTC), price_eur_mwh — ou None en cas d'échec.

    NOTE: ENTSO-E limite ce type de requête à ~1 an de plage par appel
    (documenté) — pour plusieurs années, voir get_historical_prices_range.
    """
    if not API_KEY:
        print("⚠️  ENTSOE_API_KEY manquant dans .env.")
        return None

    start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d0000")
    end_fmt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d0000")

    params = {
        "securityToken": API_KEY,
        "documentType": "A44",  # Price Document
        "in_Domain": zone_eic,
        "out_Domain": zone_eic,
        "periodStart": start_fmt,
        "periodEnd": end_fmt,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=90)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(
            _safe_request_failure(
                "ENTSO-E historical-price request",
                e,
            )
        )
        return None

    try:
        rows = _parse_price_xml(resp.text)
    except ET.ParseError as e:
        print(f"ENTSO-E XML parse error: {e}")
        print(f"Response start: {resp.text[:500]}")
        return None

    if not rows:
        print("⚠️  No price points found in the response (empty TimeSeries).")
        return None

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df


def get_historical_prices_range(start_year: int, end_year: int, zone_eic: str = FRANCE_EIC) -> Optional[pd.DataFrame]:
    """
    Récupère plusieurs années d'historique en enchaînant des appels
    annuels (par sécurité vis-à-vis de la limite ~1 an/appel).
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        df = get_historical_prices(f"{year}-01-01", f"{year}-12-31", zone_eic)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("❌ No data retrieved for the requested range.")
        return None

    return pd.concat(dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _xml_local_name(tag):
    return tag.split("}")[-1]


def _load_resolution_minutes(value):
    if value in ("PT60M", "PT1H"):
        return 60

    if value == "PT30M":
        return 30

    if value == "PT15M":
        return 15

    raise ValueError(
        "Unsupported ENTSO-E load resolution: %s"
        % value
    )



def _extract_load_document_metadata(xml_text):
    """
    Extract document-level A65 publication/version metadata.

    createdDateTime is the timestamp published in the ENTSO-E
    MarketDocument itself, not the local HTTP retrieval time.
    """
    root = ET.fromstring(xml_text)

    metadata = {
        "document_mrid": None,
        "revision_number": None,
        "created_at_utc": None,
    }

    for node in list(root):
        name = _xml_local_name(node.tag)

        if name == "mRID":
            metadata["document_mrid"] = node.text

        elif name == "revisionNumber":
            try:
                metadata["revision_number"] = int(
                    node.text
                )
            except (TypeError, ValueError):
                metadata["revision_number"] = node.text

        elif name == "createdDateTime":
            try:
                created = pd.Timestamp(
                    node.text
                )

                if created.tzinfo is None:
                    created = created.tz_localize(
                        "UTC"
                    )
                else:
                    created = created.tz_convert(
                        "UTC"
                    )

                metadata[
                    "created_at_utc"
                ] = created

            except Exception:
                metadata[
                    "created_at_utc"
                ] = None

    return metadata


def _parse_load_forecast_xml(xml_text):
    """
    Parse ENTSO-E A65/A01 day-ahead total-load forecast.

    Returned values are resampled to hourly means so the live
    consumption feature has the same hourly granularity as the
    historical PowerCast training data.
    """
    root = ET.fromstring(xml_text)
    rows = []

    for timeseries in root.iter():

        if _xml_local_name(timeseries.tag) != "TimeSeries":
            continue

        for period in timeseries.iter():

            if _xml_local_name(period.tag) != "Period":
                continue

            start_text = None
            resolution = None

            for node in period.iter():
                name = _xml_local_name(node.tag)

                if name == "start" and start_text is None:
                    start_text = node.text

                elif name == "resolution":
                    resolution = node.text

            if start_text is None or resolution is None:
                continue

            start = pd.Timestamp(start_text)

            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")

            minutes = _load_resolution_minutes(
                resolution
            )

            for point in period.iter():

                if _xml_local_name(point.tag) != "Point":
                    continue

                position = None
                quantity = None

                for node in point:
                    name = _xml_local_name(node.tag)

                    if name == "position":
                        position = int(node.text)

                    elif name == "quantity":
                        quantity = float(node.text)

                if position is None or quantity is None:
                    continue

                dt = (
                    start
                    + pd.Timedelta(
                        minutes=minutes * (position - 1)
                    )
                )

                rows.append({
                    "datetime": dt,
                    "consumption_forecast_j1": quantity,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = (
        df.sort_values("datetime")
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .set_index("datetime")
        .resample("1h")
        .mean()
        .reset_index()
    )

    return df


def get_day_ahead_load_forecast(
    delivery_date,
    zone_eic=FRANCE_EIC,
):
    """
    Return the official ENTSO-E day-ahead total-load forecast
    (A65 / A01) for one local Europe/Paris delivery day.

    The result is hourly UTC and uses the PowerCast feature name
    ``consumption_forecast_j1``.

    The returned ENTSO-E document metadata is preserved in
    DataFrame.attrs for live auditability. PowerCast does not
    claim that this reconstructs the exact historical revision
    that existed at D-1 11:00.
    """
    if not API_KEY:
        raise RuntimeError(
            "ENTSOE_API_KEY missing from .env"
        )

    day = pd.Timestamp(
        delivery_date
    ).date()

    start_naive = pd.Timestamp(day)

    end_naive = (
        start_naive
        + pd.Timedelta(days=1)
    )

    start_local = start_naive.tz_localize(
        "Europe/Paris"
    )

    end_local = end_naive.tz_localize(
        "Europe/Paris"
    )

    start_utc = start_local.tz_convert(
        "UTC"
    )

    end_utc = end_local.tz_convert(
        "UTC"
    )

    params = {
        "securityToken": API_KEY,
        "documentType": "A65",
        "processType": "A01",
        "outBiddingZone_Domain": zone_eic,
        "periodStart": start_utc.strftime(
            "%Y%m%d%H%M"
        ),
        "periodEnd": end_utc.strftime(
            "%Y%m%d%H%M"
        ),
    }

    try:
        resp = requests.get(
            BASE_URL,
            params=params,
            timeout=90,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ENTSOEAPIError(
            _safe_request_failure(
                "ENTSO-E A65 request",
                exc,
            )
        ) from None

    document_metadata = (
        _extract_load_document_metadata(
            resp.text
        )
    )

    df = _parse_load_forecast_xml(
        resp.text
    )

    if df.empty:
        raise RuntimeError(
            "ENTSO-E returned no day-ahead load forecast "
            "for %s"
            % delivery_date
        )

    df.attrs[
        "source"
    ] = "ENTSO-E A65/A01"

    df.attrs[
        "retrieved_at_utc"
    ] = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    df.attrs[
        "revision_policy"
    ] = "returned_document_metadata_recorded"

    df.attrs[
        "document_mrid"
    ] = document_metadata.get(
        "document_mrid"
    )

    df.attrs[
        "document_revision"
    ] = document_metadata.get(
        "revision_number"
    )

    created = document_metadata.get(
        "created_at_utc"
    )

    df.attrs[
        "document_created_at_utc"
    ] = (
        created.isoformat()
        if created is not None
        else None
    )

    # These fields improve traceability of the live input.
    # They are NOT treated as proof that ENTSO-E has returned
    # the exact revision historically available at 11:00.
    df.attrs[
        "exact_asof_11_reconstructed"
    ] = False

    return df


if __name__ == "__main__":
    print("Testing ENTSO-E historical prices (January 2026, native XML parser)...")
    df = get_historical_prices("2026-01-01", "2026-01-07")
    if df is not None:
        print(f"  -> Got {len(df)} rows")
        print(df.head())
        print(f"  -> Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    else:
        print("  -> Failed — see error above.")
