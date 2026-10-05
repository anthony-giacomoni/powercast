# ============================================================
#  PowerCast — France Power Price Predictor — RTE API Client
#
#  Wraps two RTE APIs (both free, both require an account on
#  data.rte-france.com):
#    - Wholesale Market API  -> EPEX Spot day-ahead prices
#    - Eco2mix API           -> consumption, production by source
#                               (incl. nuclear availability), CO2
#
#  Setup (one-time, done by the user, not by this script):
#    1. Create an account at https://data.rte-france.com
#    2. Subscribe to "Wholesale Market" and "Eco2mix" APIs
#    3. Create a "Web/Server" application -> get client_id + client_secret
#    4. Put them in .env as RTE_CLIENT_ID and RTE_CLIENT_SECRET
#
#  Transport/auth failures can be propagated as RTEAPIError when
#  callers need to distinguish API failure from data unavailability.
# ============================================================

import os
import base64
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
BASE_URL = "https://digital.iservices.rte-france.com/open_api"

CLIENT_ID = os.getenv("RTE_CLIENT_ID")
CLIENT_SECRET = os.getenv("RTE_CLIENT_SECRET")

# Cache simple du token en mémoire process (valable ~2h côté RTE)
_token_cache: Dict = {"access_token": None, "expires_at": 0}


class RTEAPIError(RuntimeError):
    """Authenticated RTE API/network request failure."""


class RTEMalformedResponse(RuntimeError):
    """RTE returned a successful HTTP response with invalid payload data."""



def _get_access_token(raise_errors: bool = False) -> Optional[str]:
    """
    Récupère un token OAuth2 (client_credentials flow), avec cache en
    mémoire pour éviter de redemander un token à chaque appel.
    """
    global _token_cache

    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    if not CLIENT_ID or not CLIENT_SECRET:
        if raise_errors:
            raise RTEAPIError(
                "RTE credentials are not configured"
            )

        print(
            "RTE fallback unavailable: credentials are not configured"
        )
        return None

    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()

    try:
        resp = requests.post(
            TOKEN_URL,
            headers={"Authorization": f"Basic {encoded}"},
            timeout=15,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            if raise_errors:
                raise RTEAPIError(
                    "RTE authentication response was not valid JSON"
                ) from e

            print(
                "RTE auth error: malformed JSON response"
            )
            return None
    except requests.exceptions.RequestException as e:
        if raise_errors:
            raise RTEAPIError(
                "RTE authentication request failed"
            ) from e

        print(
            "RTE auth error: %s"
            % type(e).__name__
        )
        return None

    token = data.get("access_token")

    if not token:
        if raise_errors:
            raise RTEAPIError(
                "RTE authentication response did not contain an access token"
            )

        print(
            "RTE auth error: access token missing from response"
        )
        return None

    expires_in = data.get("expires_in", 7200)  # secondes, ~2h par défaut

    _token_cache["access_token"] = token
    _token_cache["expires_at"] = time.time() + expires_in - 60  # marge de sécurité

    return token


def _get(
    endpoint: str,
    params: Dict = None,
    raise_errors: bool = False,
) -> Optional[dict]:
    """Appel générique authentifié à l'API RTE."""
    token = _get_access_token(
        raise_errors=raise_errors
    )
    if not token:
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
            timeout=20,
        )
        resp.raise_for_status()

        try:
            return resp.json()
        except ValueError as e:
            if raise_errors:
                raise RTEMalformedResponse(
                    "RTE API response was not valid JSON"
                ) from e

            print(
                "RTE malformed response on '%s': invalid JSON"
                % endpoint
            )
            return None
    except requests.exceptions.HTTPError as e:
        status_code = getattr(
            e.response,
            "status_code",
            None,
        )

        if raise_errors:
            raise RTEAPIError(
                "RTE API request failed on %s (HTTP %s)"
                % (
                    endpoint,
                    status_code,
                )
            ) from e

        print(
            "RTE API error on '%s': HTTP %s"
            % (
                endpoint,
                status_code,
            )
        )
        return None
    except requests.exceptions.RequestException as e:
        if raise_errors:
            raise RTEAPIError(
                "RTE network request failed on %s"
                % endpoint
            ) from e

        print(
            "RTE network error on '%s': %s"
            % (
                endpoint,
                type(e).__name__,
            )
        )
        return None


def get_day_ahead_prices(
    start_date: str,
    end_date: str,
    raise_errors: bool = False,
) -> Optional[List[Dict]]:
    """
    Prix spot EPEX day-ahead (marché France), sur une plage de dates.
    start_date / end_date au format 'YYYY-MM-DDTHH:MM:SS+01:00'.

    Retourne la collection ``france_power_exchanges`` lorsqu'elle
    est présente. Si RTE renvoie une autre structure, le payload brut est
    transmis au caller afin qu'il puisse valider la réponse explicitement.
    """
    data = _get(
        "wholesale_market/v3/france_power_exchanges",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        raise_errors=raise_errors,
    )
    if not data:
        return None

    if isinstance(data, dict):
        return data.get(
            "france_power_exchanges",
            data,
        )

    # Preserve a non-dict JSON payload so the dashboard can
    # classify it explicitly as malformed rather than as an API
    # transport failure.
    return data


def get_eco2mix_actual_generation(start_date: str, end_date: str) -> Optional[List[Dict]]:
    """
    Production réalisée par filière (nucléaire, gaz, éolien, solaire, etc.)
    et consommation, sur une plage de dates.
    """
    data = _get(
        "actual_generation/v1/actual_generations_per_production_type",
        params={"start_date": start_date, "end_date": end_date},
    )
    if not data:
        return None
    return data.get("actual_generations_per_production_type", data)


def get_consumption(start_date: str, end_date: str) -> Optional[List[Dict]]:
    """Consommation réalisée et prévisionnelle (J-1, J)."""
    data = _get(
        "consumption/v1/short_term",
        params={"start_date": start_date, "end_date": end_date},
    )
    if not data:
        return None
    return data.get("short_term", data)


if __name__ == "__main__":
    # Test manuel rapide — à lancer soi-même une fois les identifiants en place
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00+01:00")
    today = datetime.now().strftime("%Y-%m-%dT00:00:00+01:00")

    print("Testing RTE auth...")
    token = _get_access_token()
    print(f"Token obtained: {'yes' if token else 'no'}")

    if token:
        print("\nTesting day-ahead prices...")
        prices = get_day_ahead_prices(yesterday, today)
        print(prices)
