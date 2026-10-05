# ============================================================
#  PowerCast — ENTSO-E Generation Unit Unavailability Client (A80)
#
#  Retrieves ENTSO-E outage documents and keeps publication metadata
#  needed to reconstruct nuclear availability AS KNOWN at a historical
#  J-1 forecast cutoff.
#
#  Key points:
#    - A80 outage documents are limited to 200 documents per page.
#    - A80 returns generation-unit outages across technologies. We retrieve the
#      French nuclear unit catalogue separately with A71 / A33 / psrType=B14,
#      then filter A80 conservatively by normalized unit name (EIC first when possible).
#    - Available_Period Point.quantity is AVAILABLE capacity, not unavailable
#      capacity. unavailable_mw = nominal_power_mw - available_mw when the
#      nominal capacity is known.
#    - createdDateTime + revisionNumber are preserved so a later pipeline can
#      reconstruct only what was known before each historical J-1 cutoff.
# ============================================================

import io
import os
import re
import unicodedata
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ENTSOE_API_KEY")
BASE_URL = "https://web-api.tp.entsoe.eu/api"
FRANCE_EIC = "10YFR-RTE------C"
PAGE_SIZE = 200
NUCLEAR_PSR_TYPE = "B14"

BUSINESS_TYPES = {
    "A53": "planned maintenance",
    "A54": "forced unavailability",
}
CANCELLED_DOC_STATUSES = {"A09", "A13"}  # cancelled / withdrawn


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_by_local(root: ET.Element, local_name: str):
    for el in root.iter():
        if _local_name(el.tag) == local_name:
            yield el


def _first_text(root: Optional[ET.Element], *local_names: str) -> Optional[str]:
    if root is None:
        return None
    wanted = set(local_names)
    for el in root.iter():
        if _local_name(el.tag) in wanted and el.text is not None:
            text = el.text.strip()
            if text:
                return text
    return None


def _child_text(root: ET.Element, *local_names: str) -> Optional[str]:
    wanted = set(local_names)
    for el in list(root):
        if _local_name(el.tag) in wanted and el.text is not None:
            text = el.text.strip()
            if text:
                return text
    return None




def _normalize_unit_name(value: Optional[str]) -> Optional[str]:
    """Normalize ENTSO-E unit names for conservative A80 <-> A71 matching."""
    if not value:
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    # Common descriptive words vary between A80 and A71; keep plant + unit number.
    replacements = {
        "CENTRALE NUCLEAIRE": " ",
        "NUCLEAR POWER PLANT": " ",
        "POWER PLANT": " ",
        "PRODUCTION UNIT": " ",
        "GENERATION UNIT": " ",
        "TRANCHE": " ",
        "UNIT": " ",
        "UNITE": " ",
        "CNPE": " ",
        "EDF": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _name_match_key(row: pd.Series) -> Optional[str]:
    for col in ("generation_unit_name", "production_unit_name", "location"):
        key = _normalize_unit_name(row.get(col))
        if key:
            return key
    return None

def _to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _iso_duration_to_timedelta(value: Optional[str]) -> Optional[pd.Timedelta]:
    if not value:
        return None
    try:
        return pd.Timedelta(value)
    except (ValueError, TypeError):
        match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
        if not match:
            return None
        hours, minutes, seconds = (int(x or 0) for x in match.groups())
        return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _ack_reason(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text[:500]
    texts = []
    for reason in _iter_by_local(root, "Reason"):
        code = _first_text(reason, "code")
        text = _first_text(reason, "text")
        if code or text:
            texts.append(" ".join(x for x in (code, text) if x))
    return " | ".join(texts) or xml_text[:500]


def _xml_documents(content: bytes) -> List[bytes]:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return [
                zf.read(name)
                for name in zf.namelist()
                if name.lower().endswith(".xml")
            ]
    return [content] if content.strip() else []


def _get(params: Dict[str, object]) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        resp = requests.get(BASE_URL, params=params, timeout=60)
    except requests.exceptions.RequestException as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            return None, "HTTP request failed (status=%s)" % status
        return None, "network request failed (%s)" % type(exc).__name__

    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}: {_ack_reason(resp.text)}"

    # ENTSO-E sometimes returns acknowledgement XML with HTTP 200.
    ctype = resp.headers.get("content-type", "").lower()
    if "xml" in ctype and "Acknowledgement_MarketDocument" in resp.text:
        reason = _ack_reason(resp.text)
        if "no matching data" in reason.lower():
            return None, "NO_DATA"
        return None, reason

    return resp, None


# ---------------------------------------------------------------------------
# Nuclear master-data catalogue (A71 / A33 / B14)
# ---------------------------------------------------------------------------

def _parse_nuclear_catalog_xml(xml_content) -> List[dict]:
    if isinstance(xml_content, bytes):
        root = ET.fromstring(xml_content)
    else:
        root = ET.fromstring(xml_content.encode("utf-8"))

    rows = []
    for ts in _iter_by_local(root, "TimeSeries"):
        psr_type = _first_text(ts, "psrType")
        if psr_type and psr_type != NUCLEAR_PSR_TYPE:
            continue

        resource_eic = _first_text(ts, "registeredResource.mRID")
        resource_name = _first_text(ts, "registeredResource.name")
        bidding_zone = _first_text(ts, "inBiddingZone_Domain.mRID")

        period = next(_iter_by_local(ts, "Period"), None)
        start = None
        installed_capacity_mw = None
        if period is not None:
            interval = next(_iter_by_local(period, "timeInterval"), None)
            start = _first_text(interval, "start")
            point = next(_iter_by_local(period, "Point"), None)
            installed_capacity_mw = _to_float(_first_text(point, "quantity")) if point is not None else None

        if resource_eic:
            rows.append({
                "resource_eic": resource_eic,
                "resource_name": resource_name,
                "psr_type": psr_type or NUCLEAR_PSR_TYPE,
                "bidding_zone": bidding_zone,
                "catalog_start": start,
                "installed_capacity_mw": installed_capacity_mw,
            })
    return rows


def get_nuclear_unit_catalog(year: int, zone_eic: str = FRANCE_EIC) -> Optional[pd.DataFrame]:
    """Return ENTSO-E French nuclear production units for one calendar year."""
    if not API_KEY:
        print("⚠️  ENTSOE_API_KEY manquant dans .env.")
        return None

    params = {
        "securityToken": API_KEY,
        "documentType": "A71",
        "processType": "A33",  # year-ahead / installed capacity per unit
        "in_Domain": zone_eic,
        "periodStart": f"{year}01010000",
        "periodEnd": f"{year + 1}01010000",
    }

    resp, error = _get(params)
    if error:
        if error == "NO_DATA":
            print(f"⚠️  No A71 nuclear catalogue data for {year}.")
        else:
            print(f"ENTSO-E nuclear catalogue error ({year}): {error}")
        return None

    rows = []
    for xml_doc in _xml_documents(resp.content):
        try:
            rows.extend(_parse_nuclear_catalog_xml(xml_doc))
        except ET.ParseError as exc:
            print(f"⚠️  Skipping malformed A71 XML document: {exc}")

    if not rows:
        print(f"⚠️  A71 returned data for {year}, but no B14 nuclear units were parsed.")
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["resource_eic"], keep="last")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# A80 outage retrieval + pagination
# ---------------------------------------------------------------------------

def _request_outage_page(
    start_fmt: str,
    end_fmt: str,
    zone_eic: str,
    offset: int,
    business_type: Optional[str],
) -> Tuple[Optional[List[bytes]], Optional[str]]:
    params = {
        "securityToken": API_KEY,
        "documentType": "A80",
        "biddingZone_domain": zone_eic,
        "periodStart": start_fmt,
        "periodEnd": end_fmt,
        "offset": offset,
    }
    if business_type:
        params["businessType"] = business_type

    resp, error = _get(params)
    if error:
        if error == "NO_DATA":
            return [], None
        return None, error

    try:
        return _xml_documents(resp.content), None
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        return None, f"response decoding error: {exc}"


def _parse_unavailability_xml(xml_content) -> list:
    if isinstance(xml_content, bytes):
        root = ET.fromstring(xml_content)
    else:
        root = ET.fromstring(xml_content.encode("utf-8"))

    document_mrid = _child_text(root, "mRID") or _first_text(root, "mRID")
    revision_text = _child_text(root, "revisionNumber") or _first_text(root, "revisionNumber")
    try:
        revision = int(revision_text) if revision_text is not None else None
    except ValueError:
        revision = None
    created_doc_time = _child_text(root, "createdDateTime") or _first_text(root, "createdDateTime")

    docstatus = None
    for status in _iter_by_local(root, "docStatus"):
        docstatus = _first_text(status, "value") or (status.text.strip() if status.text else None)
        break

    rows = []
    for ts in _iter_by_local(root, "TimeSeries"):
        business_type = _first_text(ts, "businessType")
        production_unit_eic = _first_text(ts, "production_RegisteredResource.mRID")
        production_unit_name = _first_text(ts, "production_RegisteredResource.name")
        generation_unit_eic = _first_text(ts, "production_RegisteredResource.pSRType.powerSystemResources.mRID")
        generation_unit_name = _first_text(ts, "production_RegisteredResource.pSRType.powerSystemResources.name")
        location = _first_text(ts, "production_RegisteredResource.location.name")
        psr_type = _first_text(ts, "production_RegisteredResource.pSRType.psrType")
        nominal_power = _to_float(
            _first_text(ts, "production_RegisteredResource.pSRType.powerSystemResources.nominalP")
        )
        quantity_uom = _first_text(ts, "quantity_Measure_Unit.name")
        curve_type = _first_text(ts, "curveType")
        bidding_zone = _first_text(ts, "biddingZone_Domain.mRID", "biddingZone_domain.mRID")

        for period in _iter_by_local(ts, "Available_Period"):
            interval = next(_iter_by_local(period, "timeInterval"), None)
            period_start_text = _first_text(interval, "start") if interval is not None else None
            period_end_text = _first_text(interval, "end") if interval is not None else None
            resolution_text = _first_text(period, "resolution")
            delta = _iso_duration_to_timedelta(resolution_text)

            try:
                period_start = pd.Timestamp(period_start_text) if period_start_text else None
                period_end = pd.Timestamp(period_end_text) if period_end_text else None
            except Exception:
                period_start = period_end = None

            parsed_points = []
            for point in _iter_by_local(period, "Point"):
                pos_text = _first_text(point, "position")
                qty_text = _first_text(point, "quantity")
                try:
                    position = int(pos_text) if pos_text else None
                except ValueError:
                    position = None
                parsed_points.append((position, _to_float(qty_text)))

            for idx, (position, available_mw) in enumerate(parsed_points):
                start = period_start
                end = period_end
                if period_start is not None and delta is not None and position is not None:
                    start = period_start + delta * (position - 1)
                    if idx + 1 < len(parsed_points) and parsed_points[idx + 1][0] is not None:
                        end = period_start + delta * (parsed_points[idx + 1][0] - 1)

                unavailable_mw = None
                if nominal_power is not None and available_mw is not None:
                    unavailable_mw = max(nominal_power - available_mw, 0.0)

                rows.append({
                    "created_doc_time": created_doc_time,
                    "document_mrid": document_mrid,
                    "revision": revision,
                    "docstatus": docstatus,
                    "business_type": business_type,
                    "business_type_label": BUSINESS_TYPES.get(business_type, business_type),
                    "bidding_zone": bidding_zone,
                    "production_unit_eic": production_unit_eic,
                    "production_unit_name": production_unit_name,
                    "generation_unit_eic": generation_unit_eic,
                    "generation_unit_name": generation_unit_name,
                    "location": location,
                    "psr_type": psr_type,
                    "nominal_power_mw": nominal_power,
                    "quantity_uom": quantity_uom,
                    "curve_type": curve_type,
                    "resolution": resolution_text,
                    "position": position,
                    "start": start,
                    "end": end,
                    "available_mw": available_mw,
                    "unavailable_mw": unavailable_mw,
                })

    return rows


def _catalog_for_period(start: str, end: str, zone_eic: str) -> Optional[pd.DataFrame]:
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    frames = []
    for year in range(start_year, end_year + 1):
        cat = get_nuclear_unit_catalog(year, zone_eic=zone_eic)
        if cat is not None and not cat.empty:
            cat = cat.copy()
            cat["catalog_year"] = year
            frames.append(cat)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["resource_eic"], keep="last"
    )


def get_nuclear_unavailability_events(
    start: str,
    end: str,
    zone_eic: str = FRANCE_EIC,
    business_type: Optional[str] = None,
    max_documents: int = 5000,
    catalog: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Retrieve French nuclear A80 outages and retain revision metadata."""
    if not API_KEY:
        print("⚠️  ENTSOE_API_KEY manquant dans .env.")
        return None

    if business_type is not None and business_type not in BUSINESS_TYPES:
        raise ValueError("business_type must be None, 'A53' (planned) or 'A54' (forced)")

    start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d0000")
    end_fmt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d0000")

    all_rows = []
    fetched_docs = 0

    for offset in range(0, max_documents, PAGE_SIZE):
        docs, error = _request_outage_page(
            start_fmt=start_fmt,
            end_fmt=end_fmt,
            zone_eic=zone_eic,
            offset=offset,
            business_type=business_type,
        )
        if error:
            raise RuntimeError(
                "ENTSO-E A80 request failed at offset=%d: %s"
                % (offset, error)
            )
        if docs is None:
            raise RuntimeError(
                "ENTSO-E A80 request returned no document container "
                "at offset=%d" % offset
            )
        if not docs:
            break

        print(f"  ENTSO-E A80 offset={offset}: {len(docs)} document(s)")
        fetched_docs += len(docs)

        for xml_doc in docs:
            try:
                all_rows.extend(_parse_unavailability_xml(xml_doc))
            except ET.ParseError as exc:
                raise RuntimeError(
                    "Malformed ENTSO-E A80 XML document; refusing an incomplete outage dataset"
                ) from exc

        if len(docs) < PAGE_SIZE:
            break
    else:
        raise RuntimeError(
            "ENTSO-E A80 pagination safety cap reached (%d documents); "
            "refusing a potentially truncated outage dataset"
            % max_documents
        )

    if fetched_docs == 0:
        print("⚠️  No A80 outage documents returned for this period.")
        return None
    if not all_rows:
        print("⚠️  A80 documents were downloaded but no Available_Period rows were parsed.")
        return None

    outages = pd.DataFrame(all_rows)

    # A80 returns outages for multiple technologies. Filter them against the
    # authoritative A71 B14 nuclear catalogue. A80/A71 EIC levels can differ,
    # so exact EIC is attempted first and normalized unit-name matching is the
    # conservative fallback.
    print(f"  A80 parsed rows: {len(outages)}")
    print("  A80 production-name sample:", outages["production_unit_name"].dropna().unique()[:12])

    # Match against the authoritative A71 installed-capacity-per-unit catalogue
    # requested specifically for B14 (Nuclear). A pre-fetched catalogue may be
    # injected by the day-ahead pipeline to avoid re-querying A71 for every month.
    if catalog is None:
        print("  Fetching ENTSO-E A71 nuclear unit catalogue (B14)...")
        catalog = _catalog_for_period(start, end, zone_eic)
    if catalog is None or catalog.empty:
        print("⚠️  Could not obtain the A71 B14 nuclear catalogue; cannot safely filter A80 outages.")
        return None

    nuclear_ids = set(catalog["resource_eic"].dropna().astype(str))
    cap_map = dict(zip(catalog["resource_eic"], catalog["installed_capacity_mw"]))
    name_map = dict(zip(catalog["resource_eic"], catalog["resource_name"]))

    # 1) Preferred match: exact EIC when A80 and A71 happen to expose the same level.
    def _matched_catalog_eic(row):
        for candidate in (row.get("production_unit_eic"), row.get("generation_unit_eic")):
            if candidate in nuclear_ids:
                return candidate
        return None

    outages["catalog_eic"] = outages.apply(_matched_catalog_eic, axis=1)
    outages["match_method"] = outages["catalog_eic"].notna().map({True: "eic", False: None})

    # 2) French A80 can expose a different EIC level from A71. Fall back
    # conservatively to normalized registered-resource names from the B14-only catalogue.
    catalog = catalog.copy()
    catalog["name_key"] = catalog["resource_name"].map(_normalize_unit_name)
    valid_name_rows = catalog.dropna(subset=["name_key"]).copy()
    # Only use unique names: ambiguous keys are deliberately excluded.
    counts = valid_name_rows["name_key"].value_counts()
    unique_keys = set(counts[counts == 1].index)
    name_to_eic = dict(
        zip(
            valid_name_rows.loc[valid_name_rows["name_key"].isin(unique_keys), "name_key"],
            valid_name_rows.loc[valid_name_rows["name_key"].isin(unique_keys), "resource_eic"],
        )
    )

    outages["a80_name_key"] = outages.apply(_name_match_key, axis=1)
    missing = outages["catalog_eic"].isna()
    outages.loc[missing, "catalog_eic"] = outages.loc[missing, "a80_name_key"].map(name_to_eic)
    outages.loc[missing & outages["catalog_eic"].notna(), "match_method"] = "name"

    # Keep an explicit B14 tag if ENTSO-E happens to provide one directly as a third route.
    direct_b14 = outages["psr_type"].eq(NUCLEAR_PSR_TYPE) & outages["catalog_eic"].isna()
    # Direct B14 without a catalogue match is not enough to compute nominal capacity safely,
    # so retain it only for diagnostics rather than silently including it.

    nuclear = outages[outages["catalog_eic"].notna()].copy()
    if nuclear.empty:
        print(
            f"⚠️  Parsed {len(outages)} A80 outage row(s) and {len(catalog)} B14 catalogue unit(s), "
            "but neither EIC nor conservative name matching succeeded."
        )
        print("  A80 production-name sample:", outages["production_unit_name"].dropna().unique()[:12])
        print("  A80 generation-name sample:", outages["generation_unit_name"].dropna().unique()[:12])
        print("  A80 location sample:", outages["location"].dropna().unique()[:12])
        print("  A71 B14 name sample:", catalog["resource_name"].dropna().unique()[:12])
        print("  A80 production EIC sample:", outages["production_unit_eic"].dropna().unique()[:8])
        print("  A71 B14 EIC sample:", catalog["resource_eic"].dropna().unique()[:8])
        return None

    nuclear["catalog_name"] = nuclear["catalog_eic"].map(name_map)
    nuclear["catalog_nominal_power_mw"] = nuclear["catalog_eic"].map(cap_map)
    nuclear["nominal_power_mw"] = nuclear["nominal_power_mw"].fillna(
        nuclear["catalog_nominal_power_mw"]
    )
    nuclear["psr_type"] = NUCLEAR_PSR_TYPE

    # ENTSO-E occasionally reports a tiny available capacity above the A71
    # nominal value. Keep the safe clip, but never mask it silently.
    capacity_inconsistency = (
        nuclear["available_mw"].notna()
        & nuclear["nominal_power_mw"].notna()
        & (nuclear["available_mw"] > nuclear["nominal_power_mw"])
    )
    if capacity_inconsistency.any():
        bad = nuclear.loc[capacity_inconsistency].copy()
        excess = bad["available_mw"] - bad["nominal_power_mw"]
        print(
            f"⚠️  {len(bad)} nuclear outage row(s) have available_mw > nominal_power_mw "
            f"(max excess {excess.max():.1f} MW); unavailable_mw is clipped at 0 for those rows."
        )

    nuclear["unavailable_mw"] = (
        nuclear["nominal_power_mw"] - nuclear["available_mw"]
    ).clip(lower=0)

    for col in ("created_doc_time", "start", "end"):
        nuclear[col] = pd.to_datetime(nuclear[col], utc=True, errors="coerce")

    nuclear = nuclear.sort_values(
        ["created_doc_time", "document_mrid", "revision", "catalog_eic", "start"],
        na_position="last",
    ).reset_index(drop=True)

    print(
        f"  Matched {len(nuclear)} nuclear outage row(s) across "
        f"{nuclear['catalog_eic'].nunique()} nuclear unit(s) "
        f"(methods: {nuclear['match_method'].value_counts().to_dict()})."
    )
    return nuclear


def _latest_document_rows_as_of(events: pd.DataFrame, cutoff_utc: pd.Timestamp) -> pd.DataFrame:
    """Return only the latest revision of each outage document known at cutoff_utc."""
    if events.empty:
        return events.copy()

    known = events[
        events["created_doc_time"].notna()
        & (events["created_doc_time"] <= cutoff_utc)
    ].copy()
    known = known[known["document_mrid"].notna()].copy()
    if known.empty:
        return known

    known["_revision_sort"] = pd.to_numeric(
        known["revision"], errors="coerce"
    ).fillna(-1)
    meta = known[
        ["document_mrid", "_revision_sort", "created_doc_time", "docstatus"]
    ].drop_duplicates()
    latest = (
        meta.sort_values(
            ["document_mrid", "_revision_sort", "created_doc_time"]
        )
        .drop_duplicates(subset=["document_mrid"], keep="last")
    )

    current = known.merge(
        latest[["document_mrid", "_revision_sort", "created_doc_time"]],
        on=["document_mrid", "_revision_sort", "created_doc_time"],
        how="inner",
    )
    current = current[
        ~current["docstatus"].isin(CANCELLED_DOC_STATUSES)
    ].copy()
    return current.drop(columns=["_revision_sort"], errors="ignore")


def build_nuclear_availability_forecast_j1(
    events: pd.DataFrame,
    catalogs_by_year: Dict[int, pd.DataFrame],
    start: str,
    end: str,
    cutoff_hour_local: int = 11,
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """
    Reconstruct hourly French nuclear capacity AS KNOWN at a D-1 cutoff.

    For each delivery day D, only A80 document revisions published by the
    D-1 cutoff are visible. The latest known revision of each document wins;
    cancelled/withdrawn documents are ignored. Overlapping documents for the
    same unit use the most restrictive available capacity, and intra-hour
    changes are time-weighted.
    """
    if not 0 <= cutoff_hour_local <= 23:
        raise ValueError("cutoff_hour_local must be between 0 and 23")

    events = events.copy()
    for col in ("created_doc_time", "start", "end"):
        events[col] = pd.to_datetime(events[col], utc=True, errors="coerce")
    events = events.dropna(
        subset=["created_doc_time", "start", "end", "catalog_eic"]
    )

    start_local = pd.Timestamp(start).tz_localize(timezone)
    end_local = pd.Timestamp(end).tz_localize(timezone)
    hours_local = pd.date_range(
        start_local, end_local, freq="h", inclusive="left"
    )

    output_rows = []
    delivery_dates = sorted(set(hours_local.date))
    for delivery_date in delivery_dates:
        day_hours = hours_local[hours_local.date == delivery_date]
        year = delivery_date.year
        catalog = catalogs_by_year.get(year)
        if catalog is None or catalog.empty:
            raise ValueError(f"Missing A71 B14 nuclear catalogue for {year}")

        catalog = catalog.drop_duplicates(
            subset=["resource_eic"], keep="last"
        ).copy()
        catalog["installed_capacity_mw"] = pd.to_numeric(
            catalog["installed_capacity_mw"], errors="coerce"
        )
        catalog = catalog.dropna(
            subset=["resource_eic", "installed_capacity_mw"]
        )
        nominal_map = dict(
            zip(catalog["resource_eic"], catalog["installed_capacity_mw"])
        )
        total_nominal = float(catalog["installed_capacity_mw"].sum())

        previous_date = (
            pd.Timestamp(delivery_date)
            - pd.Timedelta(days=1)
        ).date()

        cutoff_local = pd.Timestamp(
            "%s %02d:00:00"
            % (
                previous_date.isoformat(),
                cutoff_hour_local,
            ),
            tz=timezone,
        )

        cutoff_utc = cutoff_local.tz_convert(
            "UTC"
        )
        current = _latest_document_rows_as_of(events, cutoff_utc)
        current = current[current["catalog_eic"].isin(nominal_map)].copy()

        day_start_utc = day_hours[0].tz_convert("UTC")
        day_end_utc = (day_hours[-1] + pd.Timedelta(hours=1)).tz_convert("UTC")
        current = current[
            (current["start"] < day_end_utc)
            & (current["end"] > day_start_utc)
        ].copy()

        for hour_local in day_hours:
            hour_start = hour_local.tz_convert("UTC")
            hour_end = (hour_local + pd.Timedelta(hours=1)).tz_convert("UTC")
            active = current[
                (current["start"] < hour_end)
                & (current["end"] > hour_start)
            ].copy()

            if active.empty:
                avg_unavailable = 0.0
            else:
                points = {hour_start, hour_end}
                for value in active["start"]:
                    if hour_start < value < hour_end:
                        points.add(value)
                for value in active["end"]:
                    if hour_start < value < hour_end:
                        points.add(value)
                points = sorted(points)

                weighted_unavailable = 0.0
                for left, right in zip(points[:-1], points[1:]):
                    midpoint = left + (right - left) / 2
                    sub = active[
                        (active["start"] <= midpoint)
                        & (active["end"] > midpoint)
                    ].copy()
                    if sub.empty:
                        continue

                    sub["_nominal"] = sub["catalog_eic"].map(nominal_map)
                    sub["_available"] = pd.to_numeric(
                        sub["available_mw"], errors="coerce"
                    )
                    sub = sub.dropna(subset=["_nominal", "_available"])
                    if sub.empty:
                        continue

                    sub["_available"] = sub[
                        ["_available", "_nominal"]
                    ].min(axis=1).clip(lower=0)
                    sub["_unavailable"] = (
                        sub["_nominal"] - sub["_available"]
                    )
                    unavailable_now = float(
                        sub.groupby("catalog_eic")["_unavailable"].max().sum()
                    )
                    interval_hours = (
                        right - left
                    ).total_seconds() / 3600.0
                    weighted_unavailable += unavailable_now * interval_hours

                avg_unavailable = weighted_unavailable

            available = max(total_nominal - avg_unavailable, 0.0)
            output_rows.append({
                "datetime": hour_local,
                "nuclear_available_forecast_mw": available,
                "nuclear_unavailable_forecast_mw": avg_unavailable,
                "nuclear_nominal_capacity_mw": total_nominal,
                "nuclear_forecast_cutoff": cutoff_local,
            })

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    from datetime import timedelta

    print("PowerCast ENTSO-E client v7 — A80/A71 + historical J-1 reconstruction")
    print("Testing ENTSO-E nuclear A80 outage retrieval (last 30 days)...")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    df = get_nuclear_unavailability_events(start, end)

    if df is not None and not df.empty:
        print(f"  -> SUCCESS: {len(df)} parsed nuclear outage row(s)")
        print(f"  -> Nuclear units matched: {df['catalog_eic'].nunique()}")
        show_cols = [
            "created_doc_time", "business_type", "catalog_eic", "catalog_name",
            "nominal_power_mw", "available_mw", "unavailable_mw", "start", "end"
        ]
        print(df[show_cols].head(20).to_string(index=False))
    else:
        print("  -> Failed or empty — see messages above.")
