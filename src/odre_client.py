# ============================================================
#  France Power Price Predictor — ODRÉ Client
#  (Open Data Réseaux Énergies — éCO2mix production/consumption)
#
#  IMPORTANT DISCOVERY: Eco2mix data (production by source including
#  nuclear, consumption, CO2 emissions) is NOT hosted on
#  data.rte-france.com (which only has Wholesale Market / prices,
#  requiring OAuth2). Eco2mix lives on a SEPARATE, fully public
#  platform: ODRÉ (odre.opendatasoft.com) — free, no account, no
#  authentication needed at all.
#
#  This replaces the get_eco2mix_actual_generation() function that
#  was originally (incorrectly) attempted via data.rte-france.com in
#  rte_client.py. rte_client.py should now ONLY be used for prices
#  (Wholesale Market), which does require the OAuth2 flow.
#
#  Dataset used: eco2mix-national-tr (real-time national, updated
#  hourly, 15-min granularity, ~1 month of rolling history — good
#  enough for a first pass; eco2mix-national-cons-def has the full
#  historical archive back to 2012 if more history is needed later).
# ============================================================

import requests
from typing import Optional, Dict, List
import pandas as pd

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.0/catalog/datasets"


class ODREAPIError(RuntimeError):
    """ODRÉ request/pagination failure; partial datasets are not returned."""

# Dataset "temps réel" : ~1 mois d'historique glissant, rafraîchi à l'heure.
# Pour un historique plus long, voir eco2mix-national-cons-def (données
# consolidées/définitives, mises à jour une fois par jour, depuis 2012).
DATASET_REALTIME = "eco2mix-national-tr"
DATASET_HISTORICAL = "eco2mix-national-cons-def"

# Colonnes utiles dans ce dataset (noms de champs ODRÉ) :
#   date_heure       -> timestamp
#   consommation     -> demande réalisée (MW)
#   nucleaire        -> production nucléaire (MW)
#   eolien           -> production éolienne (MW)
#   solaire          -> production solaire (MW)
#   hydraulique      -> production hydraulique (MW)
#   gaz, charbon, fioul, bioenergies -> autres filières
#   taux_co2         -> intensité carbone (g CO2/kWh)


def get_eco2mix_data(dataset: str = DATASET_REALTIME,
                      start_date: str = None, end_date: str = None,
                      limit: int = 100, max_pages: int = None,
                      order: str = "asc") -> Optional[pd.DataFrame]:
    """
    Récupère les données éCO2mix (production par filière, consommation,
    CO2) depuis ODRÉ. Aucune authentification requise.

    order: "asc" (défaut, chronologique — nécessaire pour construire un
    historique cohérent dans build_dataset.py) ou "desc" (le plus récent
    en premier — utile pour ne récupérer QUE les derniers points, comme
    dans app.py qui veut "maintenant", pas le début de la fenêtre
    glissante de DATASET_REALTIME).

    start_date / end_date au format 'YYYY-MM-DD' — définissent une fenêtre
    [start_date, end_date), donc la borne de fin est exclusive. L'API OpenDataSoft Explore v2 plafonne généralement à 100
    résultats par appel (un limit plus élevé déclenche un 400 Bad Request
    plutôt qu'une troncature silencieuse) — donc on pagine avec offset
    plutôt que de demander un gros limit d'un coup.

    max_pages est calculé automatiquement à partir de `limit` si non
    fourni (limit // 100, minimum 1) — avant, un max_pages fixe à 50
    plafonnait silencieusement à 5000 lignes quel que soit le `limit`
    demandé, ce qui coupait l'historique bien avant la fin de la plage
    de dates réellement demandée par l'appelant.
    """
    where_clauses = []
    if start_date:
        where_clauses.append(f"date_heure >= '{start_date}'")
    if end_date:
        where_clauses.append(f"date_heure < '{end_date}'")

    page_size = min(limit, 100) if limit < 100 else 100  # 100 = plafond connu de l'API Explore v2
    if max_pages is None:
        max_pages = max(1, -(-limit // page_size))  # ceil(limit / page_size)

    all_rows = []
    url = f"{BASE_URL}/{dataset}/records"

    for page in range(max_pages):
        offset = page * page_size
        params = {
            "limit": page_size,
            "offset": offset,
            "order_by": f"date_heure {order}",
        }
        if where_clauses:
            params["where"] = " and ".join(where_clauses)

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(
                getattr(e, "response", None),
                "status_code",
                None,
            )
            raise ODREAPIError(
                "ODRÉ HTTP failure on page %d (offset=%d, status=%s); "
                "partial results were discarded"
                % (page, offset, status)
            ) from None
        except requests.exceptions.RequestException as e:
            raise ODREAPIError(
                "ODRÉ network failure on page %d (offset=%d, %s); "
                "partial results were discarded"
                % (page, offset, type(e).__name__)
            ) from None
        except ValueError:
            raise ODREAPIError(
                "ODRÉ returned invalid JSON on page %d (offset=%d); "
                "partial results were discarded"
                % (page, offset)
            ) from None

        records = data.get("results", data.get("records"))
        if not records:
            print(f"  (page {page}: 0 records — stopping, either end of data or end of range)")
            break  # plus de résultats, on a tout récupéré

        if isinstance(records[0], dict) and "record" in records[0]:
            rows = [r["record"].get("fields", {}) for r in records]
        else:
            rows = records

        all_rows.extend(rows)

        if len(records) < page_size:
            print(f"  (page {page}: {len(records)} records, less than page_size={page_size} — reached the end)")
            break  # dernière page atteinte (moins de résultats que demandé)

    if not all_rows:
        print("⚠️  No records retrieved from ODRÉ — check dataset ID / field names / date range.")
        return None

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    print(f"Testing ODRÉ eco2mix ({DATASET_REALTIME}), last 100 records...")
    df = get_eco2mix_data(limit=10)
    if df is not None:
        print(f"  -> Got {len(df)} rows")
        print(f"  -> Columns: {list(df.columns)}")
        print(df.head())
    else:
        print("  -> Failed — see error above.")
