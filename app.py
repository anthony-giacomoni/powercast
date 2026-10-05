# ============================================================
#  PowerCast — Interface Streamlit
#  Point d'entrée : streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

st.set_page_config(
    page_title="PowerCast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette & styles ────────────────────────────────────────
# Tons évoquant le réseau électrique et l'énergie : bleu nuit (fond),
# bleu-pétrole (panneaux), ambre chaud (accent prix), vert sourd
# (nucléaire/stable) — pas le combo noir+vert acide générique.
BG_DEEP     = "#0D1B2A"
PANEL       = "#15293B"
ACCENT_AMBER = "#E8A33D"
ACCENT_GREEN = "#4A9782"
TEXT_WARM   = "#F2E9DC"
TEXT_MUTED  = "#8FA3B0"
GRID_LINE   = "rgba(143,163,176,0.15)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {BG_DEEP} !important;
        color: {TEXT_WARM} !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .main .block-container {{ padding: 2rem 3rem !important; max-width: 1400px !important; }}

    .hero-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT_WARM};
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }}
    .hero-subtitle {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        margin-bottom: 1.5rem;
    }}
    .panel {{
        background: {PANEL};
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(143,163,176,0.12);
    }}
    .panel-label {{
        font-size: 0.7rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.3rem;
    }}
    .panel-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {TEXT_WARM};
        font-variant-numeric: tabular-nums;
    }}
    .panel-delta-up {{ color: {ACCENT_AMBER}; font-size: 0.8rem; }}
    .panel-delta-down {{ color: {ACCENT_GREEN}; font-size: 0.8rem; }}

    [data-testid="stDataFrame"] {{ border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


# ── Chargement des données ───────────────────────────────────
def frozen_price_bundle_complete(forecast_path):
    """Return True only for a complete frozen live forecast bundle."""
    from pathlib import Path

    forecast_path = Path(forecast_path)
    prefix = "forecast_"

    if not forecast_path.name.startswith(prefix):
        return False

    date_string = forecast_path.stem[len(prefix):]
    if not date_string:
        return False

    inputs_path = forecast_path.with_name(
        "inputs_%s.csv" % date_string
    )
    metadata_path = forecast_path.with_name(
        "forecast_%s.json" % date_string
    )

    return all(
        path.is_file()
        for path in (
            forecast_path,
            inputs_path,
            metadata_path,
        )
    )


@st.cache_data(ttl=60)
def load_price_demo():
    """
    Load the latest frozen PowerCast strict pre-auction forecast.

    This function NEVER runs the model and NEVER recalculates a
    forecast. It only reads an already-generated artifact from
    data/live/.
    """
    try:
        from pathlib import Path

        live_dir = Path("data/live")

        files = sorted(
            path
            for path in live_dir.glob("forecast_*.csv")
            if frozen_price_bundle_complete(path)
        )

        if not files:
            return None, None, (
                "No frozen PowerCast forecast has been generated yet."
            )

        now_paris = pd.Timestamp.now(
            tz="Europe/Paris"
        )

        today = now_paris.date()

        candidates = []

        for path in files:
            try:
                date_string = (
                    path.stem
                    .replace("forecast_", "")
                )

                delivery_date = pd.Timestamp(
                    date_string
                ).date()

                candidates.append(
                    (
                        delivery_date,
                        path,
                    )
                )

            except Exception:
                continue

        if not candidates:
            return None, None, (
                "No valid frozen PowerCast forecast artifact found."
            )

        # Prefer tomorrow's frozen forecast when it exists.
        tomorrow = (
            pd.Timestamp(today)
            + pd.Timedelta(days=1)
        ).date()

        chosen = None

        for delivery_date, path in candidates:
            if delivery_date == tomorrow:
                chosen = (
                    delivery_date,
                    path,
                )
                break

        # Before a new D+1 forecast exists, use today's forecast
        # generated the previous day.
        if chosen is None:
            for delivery_date, path in candidates:
                if delivery_date == today:
                    chosen = (
                        delivery_date,
                        path,
                    )
                    break

        # Final fallback: latest available frozen forecast.
        if chosen is None:
            chosen = max(
                candidates,
                key=lambda x: x[0],
            )

        delivery_date, forecast_path = chosen

        df = pd.read_csv(
            forecast_path
        )

        required = {
            "datetime",
            "predicted_price_eur_mwh",
        }

        if not required.issubset(
            set(df.columns)
        ):
            raise ValueError(
                "Frozen forecast artifact has invalid columns."
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce",
        )

        df[
            "predicted_price_eur_mwh"
        ] = pd.to_numeric(
            df[
                "predicted_price_eur_mwh"
            ],
            errors="coerce",
        )

        df = (
            df.dropna(
                subset=[
                    "datetime",
                    "predicted_price_eur_mwh",
                ]
            )
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        # Display delivery hours in Europe/Paris local time.
        # Forecast artifacts remain stored in UTC.
        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("Europe/Paris")
        )

        # Keep the column name expected by the existing chart code.
        df["price_eur_mwh"] = df[
            "predicted_price_eur_mwh"
        ]

        note_parts = []

        if "forecast_generated_at" in df.columns:
            generated = pd.to_datetime(
                df[
                    "forecast_generated_at"
                ].iloc[0],
                errors="coerce",
            )

            if pd.notna(generated):
                note_parts.append(
                    "Frozen forecast generated %s"
                    % generated.strftime(
                        "%d %b %Y · %H:%M"
                    )
                )

        note_parts.append(
            "PowerCast strict pre-auction model"
        )

        note_parts.append(
            "Forecast is frozen and is not recalculated on refresh."
        )

        note = " · ".join(
            note_parts
        )

        return (
            df,
            delivery_date,
            note,
        )

    except Exception as exc:
        st.session_state[
            "_price_error"
        ] = str(exc)

        return None, None, None

@st.cache_data(ttl=3600)
def load_generation_demo():
    """
    Production/consommation ODRÉ. Combine deux sources :
    - DATASET_REALTIME (~200 derniers points, ~1 mois glissant) pour la
      valeur actuelle et le sparkline récent.
    - DATASET_HISTORICAL sur une petite fenêtre autour de "il y a 1 an"
      pour la comparaison année sur année (le flux temps réel ne remonte
      pas jusque-là).
    """
    try:
        from odre_client import get_eco2mix_data, DATASET_REALTIME, DATASET_HISTORICAL
        # order="desc" : sans ça, la pagination par défaut (croissant)
        # renvoie les 200 points les plus ANCIENS de la fenêtre glissante
        # d'~1 mois de DATASET_REALTIME, pas les plus récents — d'où le
        # panneau qui affichait "2 juillet" au lieu d'aujourd'hui alors
        # que la météo (chargée différemment) était bien à jour.
        df = get_eco2mix_data(dataset=DATASET_REALTIME, limit=200, order="desc")
        if df is None or df.empty:
            return None, None
        keep = ["date_heure", "consommation", "nucleaire", "eolien", "solaire", "taux_co2"]
        available = [c for c in keep if c in df.columns]
        df = df[available].rename(columns={"date_heure": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")  # remet en ordre chronologique croissant pour le sparkline

        # Fenêtre de quelques jours autour d'il y a 1 an, pour trouver un
        # point de comparaison à l'heure la plus proche possible.
        latest_time = df["datetime"].iloc[-1]
        year_ago_start = (latest_time - pd.Timedelta(days=366)).strftime("%Y-%m-%d")
        year_ago_end = (latest_time - pd.Timedelta(days=363)).strftime("%Y-%m-%d")
        df_year_ago = get_eco2mix_data(
            dataset=DATASET_HISTORICAL, start_date=year_ago_start, end_date=year_ago_end, limit=500
        )
        if df_year_ago is not None and not df_year_ago.empty:
            df_year_ago = df_year_ago[available].rename(columns={"date_heure": "datetime"})
            df_year_ago["datetime"] = pd.to_datetime(df_year_ago["datetime"])

        return df, df_year_ago
    except Exception:
        return None, None


@st.cache_data(ttl=3600)
def load_weather_demo():
    """
    Météo nationale pondérée — fonctionne sans authentification.
    Utilise l'API FORECAST (get_national_weather_now) pour les données
    récentes/sparkline — l'API archive (get_national_weather) a un délai
    de traitement de quelques heures à un jour, donc "aujourd'hui,
    maintenant" y apparaît vide et le dernier point reste celui de la
    veille au soir.
    Récupère aussi une petite fenêtre autour d'il y a 1 an (archive,
    seule option pour une donnée aussi ancienne) pour la comparaison
    année sur année.
    """
    try:
        from weather_client import get_national_weather, get_national_weather_now
        result = get_national_weather_now()
        if result is None or result.empty:
            st.session_state["_weather_error"] = "get_national_weather_now returned None/empty (see terminal for details)"
            return None, None

        year_ago_end = (datetime.now() - timedelta(days=363)).strftime("%Y-%m-%d")
        year_ago_start = (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d")
        result_year_ago = get_national_weather(year_ago_start, year_ago_end)

        return result, result_year_ago
    except Exception as e:
        st.session_state["_weather_error"] = str(e)
        return None, None


price_df, price_delivery_date, price_availability_note = load_price_demo()
generation_df, generation_df_year_ago = load_generation_demo()
weather_df, weather_df_year_ago = load_weather_demo()



@st.cache_data(ttl=300)
def load_market_result(delivery_date):
    from pathlib import Path
    import os

    from rte_client import (
        RTEMalformedResponse,
        get_day_ahead_prices,
    )

    start_local = pd.Timestamp(
        delivery_date
    ).tz_localize(
        "Europe/Paris"
    )

    next_date = (
        pd.Timestamp(delivery_date)
        + pd.Timedelta(days=1)
    ).date()

    end_local = pd.Timestamp(
        next_date
    ).tz_localize(
        "Europe/Paris"
    )

    expected_utc = pd.date_range(
        start=start_local.tz_convert("UTC"),
        end=end_local.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )

    snapshot_path = Path(
        "data/live/market_result_%s.csv"
        % delivery_date.isoformat()
    )

    def validated(frame):
        required = {
            "datetime",
            "market_price_eur_mwh",
        }

        if (
            not isinstance(frame, pd.DataFrame)
            or frame.empty
            or not required.issubset(
                frame.columns
            )
        ):
            return None

        out = frame[
            [
                "datetime",
                "market_price_eur_mwh",
            ]
        ].copy()

        out["datetime"] = pd.to_datetime(
            out["datetime"],
            utc=True,
            errors="coerce",
        )

        out["market_price_eur_mwh"] = (
            pd.to_numeric(
                out["market_price_eur_mwh"],
                errors="coerce",
            )
        )

        if out.isna().any().any():
            return None

        out = (
            out.sort_values("datetime")
            .reset_index(drop=True)
        )

        if out[
            "datetime"
        ].duplicated().any():
            return None

        if not pd.DatetimeIndex(
            out["datetime"]
        ).equals(expected_utc):
            return None

        out["datetime"] = (
            out["datetime"]
            .dt.tz_convert(
                "Europe/Paris"
            )
        )

        return out

    def load_snapshot():
        if not snapshot_path.is_file():
            return None

        try:
            return validated(
                pd.read_csv(
                    snapshot_path
                )
            )
        except Exception:
            return None

    def save_snapshot(frame):
        snapshot_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = snapshot_path.with_name(
            "%s.tmp.%d"
            % (
                snapshot_path.name,
                os.getpid(),
            )
        )

        stored = frame.copy()

        stored["datetime"] = (
            pd.to_datetime(
                stored["datetime"],
                utc=True,
            )
        )

        try:
            stored.to_csv(
                tmp,
                index=False,
            )
            os.replace(
                tmp,
                snapshot_path,
            )
        finally:
            if tmp.exists():
                tmp.unlink()

    snapshot = load_snapshot()

    def fallback(
        status,
        detail=None,
    ):
        if snapshot is not None:
            return (
                snapshot,
                "ok",
                "local_snapshot",
            )

        return (
            None,
            status,
            detail,
        )

    try:
        try:
            raw = get_day_ahead_prices(
                start_local.isoformat(),
                end_local.isoformat(),
                raise_errors=True,
            )

        except RTEMalformedResponse:
            return fallback(
                "malformed_response",
                None,
            )

        except Exception as exc:
            if snapshot is not None:
                return (
                    snapshot,
                    "ok",
                    "local_snapshot",
                )

            return (
                None,
                "api_error",
                type(exc).__name__,
            )

        if not raw:
            return fallback(
                "not_available",
                None,
            )

        if not isinstance(
            raw,
            list,
        ):
            return fallback(
                "malformed_response",
                None,
            )

        rows = []

        for block in raw:
            if not isinstance(
                block,
                dict,
            ):
                return fallback(
                    "malformed_response",
                    None,
                )

            values = block.get(
                "values",
                [],
            )

            if not isinstance(
                values,
                list,
            ):
                return fallback(
                    "malformed_response",
                    None,
                )

            for point in values:
                if not isinstance(
                    point,
                    dict,
                ):
                    return fallback(
                        "malformed_response",
                        None,
                    )

                dt = pd.to_datetime(
                    point.get(
                        "start_date"
                    ),
                    utc=True,
                    errors="coerce",
                )

                price = pd.to_numeric(
                    point.get("price"),
                    errors="coerce",
                )

                if (
                    pd.isna(dt)
                    or pd.isna(price)
                ):
                    return fallback(
                        "malformed_response",
                        None,
                    )

                rows.append({
                    "datetime":
                        dt,
                    "market_price_eur_mwh":
                        float(price),
                })

        if not rows:
            return fallback(
                "malformed_response",
                None,
            )

        raw_frame = pd.DataFrame(
            rows
        )

        frame = validated(
            raw_frame
        )

        if frame is None:
            local_dates = set(
                pd.to_datetime(
                    raw_frame["datetime"],
                    utc=True,
                    errors="coerce",
                )
                .dt.tz_convert(
                    "Europe/Paris"
                )
                .dt.date
                .dropna()
            )

            if (
                local_dates
                and delivery_date
                not in local_dates
            ):
                return fallback(
                    "not_available",
                    None,
                )

            return fallback(
                "malformed_response",
                None,
            )

        try:
            save_snapshot(
                frame
            )
        except OSError:
            pass

        return (
            frame,
            "ok",
            None,
        )

    except Exception as exc:
        if snapshot is not None:
            return (
                snapshot,
                "ok",
                "local_snapshot",
            )

        return (
            None,
            "api_error",
            type(exc).__name__,
        )

# ── En-tête ───────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">PowerCast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">French electricity price forecasting — '
        'RTE market data · ODRÉ generation mix · national weather</div>',
        unsafe_allow_html=True,
    )
with col_status:
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ── Hero : graphique de prix ─────────────────────────────────
if price_df is not None and not price_df.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_df["datetime"], y=price_df["price_eur_mwh"],
        mode="lines", line=dict(color=ACCENT_AMBER, width=2.5),
        fill="tozeroy", fillcolor="rgba(232,163,61,0.08)",
        hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f} €/MWh<extra></extra>",
        name="PowerCast forecast",
    ))
    today_paris = pd.Timestamp.now(tz="Europe/Paris").date()
    tomorrow_paris = today_paris + pd.Timedelta(days=1)

    if price_delivery_date == tomorrow_paris:
        day_label = "Tomorrow"
    elif price_delivery_date == today_paris:
        day_label = "Today"
    else:
        day_label = price_delivery_date.strftime("%d %b %Y")

    # Published market result, when available.
    market_df, market_status, market_detail = load_market_result(
        price_delivery_date
    )

    if market_status == "not_available":
        st.caption(
            "Market Result not published yet for this delivery day."
        )
    elif market_status == "api_error":
        st.caption(
            "⚠️ Market Result unavailable because the RTE API "
            "request failed (%s)." % market_detail
        )
    elif market_status == "malformed_response":
        st.caption(
            "⚠️ Market Result unavailable because RTE returned "
            "an unusable response."
        )

    live_rmse = None
    live_mae = None
    live_hours = 0

    if (
        market_df is not None
        and not market_df.empty
    ):
        import plotly.graph_objects as _go

        # Keep native RTE/EPEX granularity for the visual curve.
        # Since the European day-ahead market can contain 15-min
        # values, this gives 92/96/100 market points on 23/24/25-hour civil days.
        fig_price.add_trace(
            _go.Scatter(
                x=market_df["datetime"],
                y=market_df[
                    "market_price_eur_mwh"
                ],
                mode="lines",
                name="Market Result",
                line=dict(
                    color=TEXT_WARM,
                    width=1.8,
                ),
                opacity=0.9,
                hovertemplate=(
                    "%{x|%H:%M}<br>"
                    "%{y:.2f} €/MWh"
                    "<extra>Market Result</extra>"
                ),
            )
        )

        # For model evaluation, compare like with like:
        # 15-min market prices -> hourly averages.
        market_hourly = (
            market_df
            .set_index("datetime")[
                "market_price_eur_mwh"
            ]
            .resample("1h")
            .mean()
            .rename("market_hourly")
            .reset_index()
        )

        forecast_hourly = price_df[
            [
                "datetime",
                "price_eur_mwh",
            ]
        ].copy()

        forecast_hourly = (
            forecast_hourly
            .rename(
                columns={
                    "price_eur_mwh":
                        "forecast_hourly"
                }
            )
        )

        comparison = (
            forecast_hourly
            .merge(
                market_hourly,
                on="datetime",
                how="inner",
            )
            .dropna()
        )

        live_hours = len(
            comparison
        )

        if live_hours:
            errors = (
                comparison["forecast_hourly"]
                - comparison["market_hourly"]
            )

            live_mae = float(
                errors.abs().mean()
            )

            live_rmse = float(
                (
                    errors.pow(2).mean()
                ) ** 0.5
            )

        chart_title = (
            "PowerCast Day-Ahead — Forecast vs Market Result · %s"
            % price_delivery_date.strftime(
                "%d %b %Y"
            )
        )

    else:
        chart_title = (
            "PowerCast Day-Ahead Forecast — %s · %s"
            % (
                day_label,
                price_delivery_date.strftime(
                    "%d %b %Y"
                ),
            )
        )

    fig_price.update_layout(
        title=dict(text=chart_title, font=dict(size=15, color=TEXT_WARM), x=0),
        xaxis=dict(title="", showgrid=False, tickformat="%H:%M", color=TEXT_MUTED),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor=GRID_LINE, color=TEXT_MUTED),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=50, b=30),
        height=340,
        hovermode="x unified",
        font=dict(color=TEXT_WARM),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    if (
        market_df is not None
        and live_rmse is not None
    ):
        live_m1, live_m2 = st.columns(2)

        live_m1.metric(
            "Live RMSE",
            "%.2f €/MWh" % live_rmse,
        )

        live_m2.metric(
            "Live MAE",
            "%.2f €/MWh" % live_mae,
        )

        st.caption(
            "Market Result is displayed at its published "
            "granularity; live error metrics compare PowerCast "
            "with hourly averages over %d matched delivery hours."
            % live_hours
        )

    if price_availability_note:
        st.caption(price_availability_note)
else:
    price_error = st.session_state.get("_price_error")
    message = (
        "No frozen PowerCast forecast is available right now."
        if not price_error
        else "PowerCast forecast unavailable: %s" % price_error
    )
    st.markdown(
        '<div class="panel">'
        '<div class="panel-label">POWERCAST DAY-AHEAD FORECAST</div>'
        '<div class="panel-value" style="font-size:1rem; color:#8FA3B0;">'
        + message +
        '</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Bande de contexte : nucléaire / conso / météo ────────────
col1, col2, col3 = st.columns(3)

def _sparkline(df, y_col, color, fill_rgba):
    fig = go.Figure(go.Scatter(
        x=df["datetime"], y=df[y_col],
        mode="lines", line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=fill_rgba,
    ))
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=5, b=0), height=120,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _latest_with_context(df, col, unit, df_year_ago=None, decimals=0):
    """
    Retourne (valeur la plus récente, horodatage, delta vs ~24h avant,
    delta vs ~1 an avant) pour donner du contexte à un chiffre brut
    affiché seul — sans ça, "40,674 MW" ne dit rien sur si c'est
    haut/bas/normal ni de quand ça date.

    df_year_ago (optionnel) : DataFrame séparé couvrant une fenêtre
    autour d'il y a 1 an, nécessaire car le flux temps réel (df) ne
    remonte généralement que sur ~1 mois glissant.
    """
    series = df.dropna(subset=[col])
    if series.empty:
        return None, None, None, None
    latest_row = series.iloc[-1]
    latest_val = latest_row[col]
    latest_time = latest_row["datetime"]

    # Delta vs ~24h avant
    target_time = latest_time - pd.Timedelta(hours=24)
    prior = series.iloc[(series["datetime"] - target_time).abs().argsort()[:1]]
    delta_24h = None
    if not prior.empty:
        delta_24h = latest_val - prior[col].iloc[0]

    # Delta vs ~1 an avant, à la même heure du jour si possible
    delta_1y = None
    if df_year_ago is not None and not df_year_ago.empty and col in df_year_ago.columns:
        year_ago_series = df_year_ago.dropna(subset=[col])
        if not year_ago_series.empty:
            target_time_1y = latest_time - pd.Timedelta(days=365)
            prior_1y = year_ago_series.iloc[
                (year_ago_series["datetime"] - target_time_1y).abs().argsort()[:1]
            ]
            if not prior_1y.empty:
                delta_1y = latest_val - prior_1y[col].iloc[0]

    return latest_val, latest_time, delta_24h, delta_1y


def _fun_facts_mw(mw_value: float) -> list:
    """
    Traduit une puissance instantanée (MW) en équivalents concrets, pour
    rendre un chiffre abstrait ("40,674 MW") tangible. Ratios basés sur
    des sources réelles, pas inventés :
    - Toast : ~0.03 kWh par tranche (grille-pain 1200W, ~1.5 min)
    - Foyer français : ~4 255 kWh/an (moyenne Enedis/ADEME 2024)
    - Tesla Model 3 (batterie ~57.5 kWh) : ordre de grandeur standard
    - Ampoule LED 10W : usage courant
    On traite le MW instantané comme s'il était soutenu pendant 1 heure
    (= MWh) pour ces comparaisons — approximation d'ordre de grandeur,
    pas une vraie mesure d'énergie cumulée sur la durée.
    """
    mwh = mw_value  # 1h à cette puissance = ce nombre de MWh
    kwh = mwh * 1000

    toasts_billions = (kwh / 0.03) / 1_000_000_000
    homes_per_year = (mwh * 8760) / 4255
    tesla_charges = kwh / 57.5
    led_bulbs_1h = kwh / 0.01
    tgv_km = kwh / 20  # un TGV consomme environ 20 kWh/km à vitesse de croisière

    return [
        ("toast", f"{toasts_billions:.3f} billion", "slices of bread"),
        ("power", f"{homes_per_year:,.0f}", "French homes for a year"),
        ("fully charge", f"{tesla_charges:,.0f}", "Tesla Model 3 batteries"),
        ("power", f"{led_bulbs_1h:,.0f}", "LED bulbs for an hour"),
        ("run a TGV for", f"{tgv_km:,.0f}", "km"),
    ]


def _render_fun_facts(val, mw_value_for_intro):
    """Affiche 'X MW Could:' une seule fois, puis une vraie liste à puces
    avec les nombres en gras et en couleur ambre pour ressortir."""
    items_html = "".join(
        f"<li>{verb} <b style='color:{ACCENT_AMBER};'>{number}</b> {rest}</li>"
        for verb, number, rest in _fun_facts_mw(val)
    )
    st.markdown(
        f"<div style='font-size:0.8rem; color:#8FA3B0; margin-top:8px;'>"
        f"{mw_value_for_intro:,.0f} MW could:</div>"
        f"<ul style='font-size:0.8rem; color:#8FA3B0; margin:2px 0 0 -18px; padding-left:18px;'>"
        f"{items_html}</ul>",
        unsafe_allow_html=True,
    )


with col1:
    with st.container(border=True):
        st.caption("NUCLEAR OUTPUT", help="Total electricity currently being generated by France's nuclear power plants, "
                                           "which typically supply 60-70% of the country's electricity.")
        val, ts, delta_24h, delta_1y = _latest_with_context(
            generation_df, "nucleaire", "MW", df_year_ago=generation_df_year_ago
        ) if generation_df is not None and not generation_df.empty else (None, None, None, None)
        if val is not None:
            st.markdown(f"### {val:,.0f} MW")
            lines = [f"As of {ts.strftime('%d %b, %H:%M')}"]
            if delta_24h is not None:
                lines.append(f"{delta_24h:+,.0f} MW vs 24h ago")
            if delta_1y is not None:
                lines.append(f"{delta_1y:+,.0f} MW vs 1 year ago")
            st.caption("  \n".join(lines))
            st.plotly_chart(
                _sparkline(generation_df, "nucleaire", ACCENT_GREEN, "rgba(74,151,130,0.1)"),
                use_container_width=True, config={"displayModeBar": False},
            )
            _render_fun_facts(val, val)
        else:
            st.markdown("### No data")

with col2:
    with st.container(border=True):
        st.caption("NATIONAL CONSUMPTION")
        val, ts, delta_24h, delta_1y = _latest_with_context(
            generation_df, "consommation", "MW", df_year_ago=generation_df_year_ago
        ) if generation_df is not None and not generation_df.empty else (None, None, None, None)
        if val is not None:
            st.markdown(f"### {val:,.0f} MW")
            lines = [f"As of {ts.strftime('%d %b, %H:%M')}"]
            if delta_24h is not None:
                lines.append(f"{delta_24h:+,.0f} MW vs 24h ago")
            if delta_1y is not None:
                lines.append(f"{delta_1y:+,.0f} MW vs 1 year ago")
            st.caption("  \n".join(lines))
            st.plotly_chart(
                _sparkline(generation_df, "consommation", TEXT_WARM, "rgba(242,233,220,0.08)"),
                use_container_width=True, config={"displayModeBar": False},
            )
            _render_fun_facts(val, val)
        else:
            st.markdown("### No data")

with col3:
    with st.container(border=True):
        st.caption("NATIONAL TEMPERATURE")
        val, ts, delta_24h, delta_1y = _latest_with_context(
            weather_df, "temperature_c", "°C", df_year_ago=weather_df_year_ago
        ) if weather_df is not None and not weather_df.empty else (None, None, None, None)
        if val is not None:
            st.markdown(f"### {val:.1f}°C")
            lines = [f"As of {ts.strftime('%d %b, %H:%M')}"]
            if delta_24h is not None:
                lines.append(f"{delta_24h:+.1f}°C vs 24h ago")
            if delta_1y is not None:
                lines.append(f"{delta_1y:+.1f}°C vs 1 year ago")
            st.caption("  \n".join(lines))
            st.plotly_chart(
                _sparkline(weather_df, "temperature_c", ACCENT_AMBER, "rgba(232,163,61,0.08)"),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            error_msg = st.session_state.get("_weather_error", "no error captured")
            st.markdown("### No data")
            st.caption(f"Debug: {error_msg}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Prédictions du modèle : réel vs prédit ───────────────────
@st.cache_data(ttl=3600)
def load_model_results():
    """
    Charge les prédictions et métriques du modèle strict pré-auction
    déjà calculées par train_model_strict.py. Le modèle n'est jamais
    ré-entraîné au chargement de la page : Streamlit ne fait ici que
    visualiser les artefacts de validation sauvegardés.
    """
    try:
        preds_df = pd.read_csv("data/predictions_strict.csv")
        preds_df["datetime"] = pd.to_datetime(preds_df["datetime"])
        with open("data/metrics_strict.json") as f:
            metrics = json.load(f)
        return preds_df, metrics
    except FileNotFoundError:
        return None, None


preds_df, metrics = load_model_results()

if preds_df is not None:
    st.subheader("Strict Pre-Auction Model — Real vs Predicted (test period)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} €/MWh")
    m2.metric("Mean Absolute Error", f"{metrics['mae']:.2f} €/MWh")
    m3.metric("Root Mean Squared Error as % of mean price", f"{metrics['rmse_pct']:.1f}%")

    # Le test set couvre potentiellement des années de données horaires —
    # tout afficher d'un coup écrase visuellement les courbes sur un axe
    # X trop compressé, rendant réel/prédit indiscernables. On limite la
    # vue par défaut aux N derniers jours (lisible), avec un slider pour
    # explorer une autre fenêtre si besoin.
    max_date = preds_df["datetime"].max()
    min_date = preds_df["datetime"].min()
    default_window_days = 30

    date_range = st.date_input(
        "Date range",
        value=(max_date.date() - pd.Timedelta(days=default_window_days), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="pred_date_range",
    )
    # st.date_input renvoie un tuple (start, end) une fois les deux bornes
    # choisies, mais un tuple à UN SEUL élément (date_range[0],) tant que
    # l'utilisateur n'a pas encore cliqué la seconde date — le bug
    # précédent traitait par erreur ce tuple à 1 élément comme la valeur
    # de range_start elle-même (au lieu d'en extraire l'élément), d'où un
    # TypeError en comparant un tuple à une date. On distingue maintenant
    # explicitement les 3 cas possibles.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        range_start, range_end = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        st.info("👉 Please select the end of the range.")
        st.stop()
    else:
        range_start, range_end = date_range, max_date.date()

    plot_df = preds_df[
        (preds_df["datetime"].dt.date >= range_start) &
        (preds_df["datetime"].dt.date <= range_end)
    ]
    window_days = (range_end - range_start).days

    # Marge de 10% sur l'axe Y pour que l'écart entre les deux courbes
    # soit visuellement clair, plutôt que de laisser Plotly ajuster
    # automatiquement une échelle qui peut être trop resserrée.
    if plot_df.empty:
        st.warning("No data in the selected date range.")
    else:
        y_min = min(plot_df["price_eur_mwh"].min(), plot_df["predicted_price_eur_mwh"].min())
        y_max = max(plot_df["price_eur_mwh"].max(), plot_df["predicted_price_eur_mwh"].max())
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 5

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=plot_df["datetime"], y=plot_df["price_eur_mwh"],
            mode="lines", line=dict(color=TEXT_WARM, width=2),
            name="Actual price",
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>Actual: %{y:.1f} €/MWh<extra></extra>",
        ))
        fig_pred.add_trace(go.Scatter(
            x=plot_df["datetime"], y=plot_df["predicted_price_eur_mwh"],
            mode="lines", line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
            name="Predicted",
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>Predicted: %{y:.1f} €/MWh<extra></extra>",
        ))
        fig_pred.update_layout(
            xaxis=dict(title="", showgrid=False, color=TEXT_MUTED),
            yaxis=dict(
                title="€/MWh", showgrid=True, gridcolor=GRID_LINE, color=TEXT_MUTED,
                range=[y_min - y_margin, y_max + y_margin],
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=20, t=20, b=30),
            height=420,
            hovermode="x unified",
            font=dict(color=TEXT_WARM),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption(
            f"Showing {window_days} days ({range_start} to {range_end}) · "
            "Strict pre-auction validation: every model input is restricted to information "
            "available before the D-1 market cutoff. The test period is chronologically "
            "after training (no random split)."
        )
else:
    st.info(
        "No strict model artifacts found yet. Run "
        "`python src/train_model_strict.py --data data/dataset_strict.csv --save` "
        "to train the model and generate predictions."
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Statut du projet (transparence sur ce qui est réel vs en attente) ──
with st.expander("ℹ️ About this data", expanded=False):
    st.markdown("""
    **PowerCast day-ahead forecast panel** — The top chart displays the frozen hourly forecast
    produced by the PowerCast LightGBM model for the selected 23/24/25-hour civil delivery day.
    A forecast is generated from pre-auction information and saved as an immutable local artifact;
    refreshing the dashboard does not retrain the model or recalculate the prediction. Once the
    French EPEX/RTE market result is published, its native market curve is overlaid automatically.
    Live RMSE and MAE are calculated against hourly averages so they remain comparable with the
    hourly PowerCast model.

    **Live nuclear / consumption panels** — ODRÉ (Open Data Réseaux Énergies). These panels
    provide current system context; the strict forecasting model uses its own pre-auction inputs.

    **Live temperature panel** — Open-Meteo national weighted temperature. The strict model
    instead uses a fixed ECMWF 00Z D-1 weather snapshot, so future information cannot leak in.

    **Strict model inputs** — J-1 consumption forecast, fixed pre-auction ECMWF weather,
    clean nuclear availability/capacity information, historical price lags, calendar effects,
    TTF gas futures and EUA carbon auction data. Realised generation, A69 renewable forecasts,
    A09 commercial exchanges and A61 transfer-capacity experiments are excluded.

    **Model** — LightGBM gradient boosting with a chronological train/test split. The displayed
    metrics come from the saved strict pre-auction test period, not from the live market chart.
    """)
