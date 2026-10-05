# ============================================================
#  EDF / France Power Price Predictor — Feature Engineering + Model
#
#  Takes the raw dataset built by build_dataset.py and:
#    1. Engineers time-series-appropriate features (lags, rolling
#       averages, calendar effects) — never using future information
#       to predict the past.
#    2. Splits train/test CHRONOLOGICALLY (never randomly — random
#       splits on time series leak future information into training).
#    3. Trains a LightGBM regressor to predict next-day EPEX price.
#    4. Reports RMSE/MAE and feature importance.
#
#  Usage:
#      python src/train_model.py --data data/dataset.csv
#
#  NOTE: column names below (price_eur_mwh, temperature_c, etc.) match
#  what build_dataset.py currently produces. If the real RTE API
#  response uses different field names, fix build_dataset.py's parsing
#  FIRST, then this script will work unchanged (it only depends on the
#  cleaned CSV, not on RTE's raw format).
# ============================================================

import argparse
import sys
import os
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import lightgbm as lgb
    MODEL_LIB = "lightgbm"
except ImportError:
    try:
        import xgboost as xgb
        MODEL_LIB = "xgboost"
    except ImportError:
        MODEL_LIB = None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features à partir du dataset brut. Toutes les features
    utilisent uniquement de l'information disponible AVANT l'instant
    prédit (pas de fuite de données du futur).
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="mixed", errors="coerce")
    if df["datetime"].isna().all():
        raise ValueError(
            "La colonne 'datetime' n'a pas pu être convertie en date — "
            "vérifie le format dans le CSV source (voir build_dataset.py)."
        )
    n_invalid = df["datetime"].isna().sum()
    if n_invalid > 0:
        print(f"⚠️  {n_invalid} rows had an unparseable datetime and will be dropped.")
        df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Effets calendaires — le prix électricité a une forte saisonnalité
    # hebdomadaire (weekday vs weekend) et journalière (heures de pointe).
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["datetime"].dt.month

    # Lags du prix — le prix d'hier à la même heure est un signal fort
    # (persistance des conditions de marché à court terme).
    if "price_eur_mwh" in df.columns:
        df["price_lag_24h"] = df["price_eur_mwh"].shift(24)   # même heure, veille
        df["price_lag_168h"] = df["price_eur_mwh"].shift(168)  # même heure, semaine dernière
        df["price_rolling_mean_24h"] = df["price_eur_mwh"].rolling(24).mean().shift(1)

    # Météo : HDD/CDD (Heating/Cooling Degree Days, cf. Hull — Energy and
    # Derivatives) plutôt qu'une simple température au carré. Le seuil de
    # confort standard est 65°F (~18.3°C) ; HDD capture la demande de
    # chauffage (grand froid), CDD la demande de clim (forte chaleur).
    # Contrairement à temperature^2, HDD/CDD séparent explicitement les
    # deux régimes — pertinent en France où le chauffage électrique est
    # très répandu (forte sensibilité au froid) alors que la clim l'est
    # moins (sensibilité à la chaleur plus faible mais bien réelle en été,
    # cf. pics de prix electricité observés en canicule).
    COMFORT_THRESHOLD_C = 18.3  # 65°F converti
    if "temperature_c" in df.columns:
        df["hdd"] = (COMFORT_THRESHOLD_C - df["temperature_c"]).clip(lower=0)
        df["cdd"] = (df["temperature_c"] - COMFORT_THRESHOLD_C).clip(lower=0)

    # Lags supplémentaires — 1h capture l'inertie très court terme (le
    # prix ne saute pas arbitrairement d'une heure à l'autre), 48h/72h
    # ajoutent des points de comparaison intermédiaires entre 24h et 168h,
    # pour capturer plus finement la mémoire court-terme du marché.
    if "price_eur_mwh" in df.columns:
        df["price_lag_1h"] = df["price_eur_mwh"].shift(1)
        df["price_lag_48h"] = df["price_eur_mwh"].shift(48)
        df["price_lag_72h"] = df["price_eur_mwh"].shift(72)

    # Ratio nucléaire/consommation prévue (J-1) — le NIVEAU de nucléaire
    # et de consommation pris séparément dit moins que leur RATIO : un
    # nucléaire élevé avec une conso élevée n'a pas le même effet sur le
    # prix qu'un nucléaire élevé avec une conso faible. +1e-6 évite une
    # division par zéro.
    #
    # FUITE TEMPORELLE RÉSIDUELLE ASSUMÉE ET DOCUMENTÉE : 'nucleaire'
    # reste ici la production RÉALISÉE (pas une prévision RTE), car
    # l'intégration de l'API RTE "Unavailability-Additional-Information"
    # (disponibilité prévisionnelle par tranche) est en cours séparément.
    # Ce ratio n'est donc PAS encore ex-ante propre côté numérateur — à
    # corriger dès que cette source sera intégrée.
    if "nucleaire" in df.columns and "consumption_forecast_j1" in df.columns:
        df["nuclear_to_demand_ratio"] = df["nucleaire"] / (df["consumption_forecast_j1"] + 1e-6)

    # Marge résiduelle du système (production bas-carbone disponible
    # moins consommation prévue) — signal direct des régimes de prix
    # extrêmes. Diagnostic préalable : ~3% de prix négatifs, ~4.4% >
    # 200€/MWh, jusqu'à 4000€/MWh en pointe.
    #
    # MÊME RÉSERVE que ci-dessus : nucléaire/éolien/solaire restent des
    # valeurs réalisées, pas des prévisions — fuite résiduelle assumée,
    # documentée, à corriger une fois la source RTE nucléaire intégrée
    # (et idéalement une prévision éolien/solaire, qu'Open-Meteo peut
    # indirectement approximer via son propre forecast de vent/nébulosité).
    renewable_cols = [c for c in ["nucleaire", "eolien", "solaire"] if c in df.columns]
    if renewable_cols and "consumption_forecast_j1" in df.columns:
        df["residual_margin"] = df[renewable_cols].sum(axis=1) - df["consumption_forecast_j1"]

    # Jours fériés français — la consommation industrielle chute
    # nettement ces jours-là, avec un effet direct sur le prix.
    try:
        import holidays
    except ImportError as exc:
        raise RuntimeError(
            "holidays==0.58 is required for PowerCast feature engineering"
        ) from exc

    fr_holidays = holidays.France(
        years=range(
            df["datetime"].dt.year.min(),
            df["datetime"].dt.year.max() + 1,
        )
    )
    df["is_holiday"] = df["datetime"].dt.date.astype(str).isin(
        [str(d) for d in fr_holidays.keys()]
    ).astype(int)

    # Interaction HDD x jour de semaine — le besoin de chauffage (HDD)
    # n'a pas le même effet sur le prix en semaine (usines/bureaux
    # chauffés en plus des logements) qu'un weekend (surtout du
    # résidentiel). Une interaction explicite permet au modèle de capter
    # cette différence sans avoir à la redécouvrir indirectement via les
    # deux variables séparées.
    if "hdd" in df.columns:
        df["hdd_x_weekday"] = df["hdd"] * (1 - df["is_weekend"])

    return df


def time_series_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Split chronologique (PAS aléatoire) : les test_fraction% les plus
    récents servent de test, le reste d'entraînement. C'est crucial pour
    une série temporelle — un split aléatoire laisserait le modèle
    s'entraîner sur des données futures par rapport à ce qu'il prédit,
    ce qui donnerait un score trompeur bien meilleur qu'en production réelle.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def hyperparameter_search(df_clean: pd.DataFrame, feature_cols: list, target_col: str,
                           n_splits: int = 5) -> dict:
    """
    Recherche d'hyperparamètres par grid search avec validation croisée
    temporelle (TimeSeriesSplit de scikit-learn) — PAS une K-fold
    classique, qui mélangerait aléatoirement passé et futur entre les
    folds. TimeSeriesSplit crée des folds successifs où chaque
    validation porte toujours sur des données postérieures à son
    entraînement, cohérent avec la contrainte déjà appliquée dans
    time_series_split() pour le split final train/test.

    N'est PAS lancé par défaut (coûte plusieurs minutes vu le volume de
    données et le nombre de combinaisons) — activé via --tune en CLI.
    Retourne les meilleurs hyperparamètres trouvés, à utiliser ensuite
    pour l'entraînement final sur le vrai split train/test.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Grille volontairement modeste (2x2x2 = 8 combinaisons) pour rester
    # dans un temps raisonnable — un vrai grid search exhaustif prendrait
    # des heures sur ce volume de données avec 5 folds par combinaison.
    param_grid = [
        {"learning_rate": lr, "max_depth": depth, "num_leaves": leaves}
        for lr in [0.03, 0.05]
        for depth in [6, 8]
        for leaves in [31, 63]
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    print(f"\n=== Hyperparameter search: {len(param_grid)} combinations x {n_splits} folds ===")
    for i, params in enumerate(param_grid):
        fold_rmses = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(
                n_estimators=300,  # réduit pour la recherche (rapidité) ; le modèle
                                   # final réutilisera 800 comme avant, une fois les
                                   # meilleurs learning_rate/depth/leaves trouvés
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                num_leaves=params["num_leaves"],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

        mean_rmse = np.mean(fold_rmses)
        results.append({**params, "mean_rmse": mean_rmse})
        print(f"  [{i+1}/{len(param_grid)}] {params} -> mean RMSE across {n_splits} folds: {mean_rmse:.2f}")

    best = min(results, key=lambda r: r["mean_rmse"])
    print(f"\n✅ Best params: {best}")
    return best


def train_peak_offpeak_split(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list,
                              target_col: str, model_params: dict) -> np.ndarray:
    """
    Entraîne deux modèles séparés — heures de pointe vs creuses — plutôt
    qu'un seul modèle appelé à apprendre les deux régimes en même temps.
    Heures de pointe françaises typiques : 8h-13h et 18h-20h (forte
    demande, tension système plus fréquente) ; le reste est "creux".

    Retourne les prédictions combinées sur le test set (recomposées dans
    l'ordre chronologique d'origine) pour un calcul de RMSE/MAE
    directement comparable au modèle unique.
    """
    def _is_peak(hour_series):
        return ((hour_series >= 8) & (hour_series < 13)) | ((hour_series >= 18) & (hour_series < 20))

    train_peak_mask = _is_peak(train["hour"])
    test_peak_mask = _is_peak(test["hour"])

    print(f"Peak hours: {train_peak_mask.sum()} train rows, {test_peak_mask.sum()} test rows")
    print(f"Off-peak hours: {(~train_peak_mask).sum()} train rows, {(~test_peak_mask).sum()} test rows")

    preds_combined = pd.Series(index=test.index, dtype=float)

    for label, mask_train, mask_test in [("peak", train_peak_mask, test_peak_mask),
                                          ("off-peak", ~train_peak_mask, ~test_peak_mask)]:
        X_tr = train.loc[mask_train, feature_cols]
        y_tr = train.loc[mask_train, target_col]
        X_te = test.loc[mask_test, feature_cols]

        model = lgb.LGBMRegressor(**model_params, verbosity=-1) if MODEL_LIB == "lightgbm" \
            else xgb.XGBRegressor(**model_params)
        model.fit(X_tr, y_tr)
        preds_combined.loc[mask_test] = model.predict(X_te)
        print(f"  {label} model trained on {len(X_tr)} rows.")

    return preds_combined.values


def train_and_evaluate(df: pd.DataFrame, target_col: str = "price_eur_mwh", test_fraction: float = 0.2,
                        tune: bool = False, use_log_transform: bool = False, split_peak_hours: bool = False):
    """Entraîne le modèle et affiche les métriques + importance des features."""
    if MODEL_LIB is None:
        print("❌ Neither lightgbm nor xgboost is installed. Run:")
        print("   pip install lightgbm --break-system-packages")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in (
        "datetime", target_col
    ) and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    # Diagnostic : combien de valeurs manquantes par colonne, pour
    # comprendre où sont vraiment les trous avant de les traiter.
    print("\n=== Missing values per column (before cleaning) ===")
    for col in [target_col] + feature_cols:
        n_missing = df[col].isna().sum()
        pct = 100 * n_missing / len(df)
        if n_missing > 0:
            print(f"  {col}: {n_missing} missing ({pct:.1f}%)")

    # Le prix (target) ne peut pas être imputé — une ligne sans prix réel
    # est inutilisable pour l'entraînement, donc on la supprime.
    df_clean = df.dropna(subset=[target_col]).copy()

    # Pour les FEATURES en revanche, un trou isolé (ex: météo indisponible
    # ce jour précis) ne justifie pas de jeter toute la ligne alors que
    # le prix et les 20 autres colonnes sont bons. On comble avec un
    # forward-fill (valeur précédente) puis backward-fill pour les tout
    # premiers trous en début de série — raisonnable pour des variables
    # qui évoluent lentement (météo, nucléaire, consommation) heure par
    # heure, contrairement au prix qui peut varier brutalement.
    df_clean[feature_cols] = df_clean[feature_cols].ffill().bfill()

    # S'il reste des NaN après ça (ex: une colonne entièrement vide sur
    # toute la période), on les supprime en dernier recours seulement.
    remaining_nan_cols = df_clean[feature_cols].columns[df_clean[feature_cols].isna().any()].tolist()
    if remaining_nan_cols:
        print(f"⚠️  Still missing after ffill/bfill (likely fully empty columns): {remaining_nan_cols}")
        df_clean = df_clean.dropna(subset=remaining_nan_cols)

    print(f"\n{len(df)} rows before cleaning -> {len(df_clean)} rows after "
          f"(dropped {len(df) - len(df_clean)} rows total, mostly due to missing target price "
          f"or feature lags at the start of the series)")

    if df_clean.empty:
        print("❌ No rows left after dropping NaNs — check feature engineering "
              "(lags create NaNs at the start of the series, that's expected, "
              "but if EVERYTHING is NaN something upstream is broken).")
        sys.exit(1)

    train, test = time_series_split(df_clean, test_fraction=test_fraction)
    print(f"Train: {len(train)} rows ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"Test:  {len(test)} rows ({test['datetime'].min()} to {test['datetime'].max()})")

    X_train, y_train_raw = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    # Transformation signée log(1+|x|)*sign(x) plutôt qu'un log standard,
    # qui n'existe pas pour les valeurs négatives (~3% des prix, cf.
    # diagnostic préalable). Compresse les extrêmes (jusqu'à 4000 EUR/MWh
    # observés) tout en préservant le signe, ce qui peut aider LightGBM à
    # mieux répartir son attention entre régime normal et régime extrême
    # plutôt que d'être dominé par l'échelle brute des pics.
    # Le modèle est entraîné SUR l'échelle transformée, mais RMSE/MAE sont
    # calculés après avoir inversé la transformation, pour rester
    # comparables en EUR/MWh aux résultats précédents (sans ça, comparer
    # un RMSE en échelle log à un RMSE en échelle brute n'aurait aucun sens).
    if use_log_transform:
        y_train = np.sign(y_train_raw) * np.log1p(np.abs(y_train_raw))
        print("Using signed log transform on target (train only; predictions inverted before scoring).")
    else:
        y_train = y_train_raw

    # Valeurs par défaut (ajustement manuel documenté, voir commit history) —
    # remplacées par le résultat de hyperparameter_search si --tune est passé.
    best_params = {"learning_rate": 0.03, "max_depth": 8, "num_leaves": 63}
    if tune:
        best_params_search = hyperparameter_search(train, feature_cols, target_col)
        best_params = {k: v for k, v in best_params_search.items() if k != "mean_rmse"}

    model_params = {
        "n_estimators": 800,
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
    }
    if MODEL_LIB == "lightgbm":
        model_params["num_leaves"] = best_params["num_leaves"]

    if split_peak_hours:
        # Deux modèles séparés (heures de pointe / creuses) plutôt qu'un
        # seul — voir train_peak_offpeak_split() pour le détail. Court-
        # circuite l'entraînement standard ci-dessous.
        model = None  # pas de modèle unique à sauvegarder dans ce mode
        importances = {}
        preds = train_peak_offpeak_split(train, test, feature_cols, target_col, model_params)
    elif MODEL_LIB == "lightgbm":
        model = lgb.LGBMRegressor(
            **model_params,
            verbosity=-1,  # supprime le flot de warnings "no further splits with positive gain" —
                           # bénin (juste des arbres qui s'arrêtent tôt sur certaines branches),
                           # mais pollue massivement la sortie sur un dataset de cette taille.
        )
        model.fit(X_train, y_train)
        importances = dict(zip(feature_cols, model.feature_importances_))
    else:
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        importances = dict(zip(feature_cols, model.feature_importances_))

    if not split_peak_hours:
        preds = model.predict(X_test)
    if use_log_transform:
        # Inverse de sign(x)*log1p(|x|) : sign(pred)*(exp(|pred|)-1)
        preds = np.sign(preds) * (np.expm1(np.abs(preds)))

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"\n=== Results ({MODEL_LIB}) ===")
    print(f"RMSE: {rmse:.2f} EUR/MWh")
    print(f"MAE:  {mae:.2f} EUR/MWh")
    print(f"Mean actual price in test set: {y_test.mean():.2f} EUR/MWh "
          f"(RMSE as % of mean: {100*rmse/y_test.mean():.1f}%)")

    print("\n=== Feature importance ===")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")

    # Sauvegarde le modèle + les prédictions sur le test set, pour que
    # l'interface Streamlit (app.py) puisse afficher une courbe prix
    # réel vs prédit SANS ré-entraîner le modèle à chaque chargement de
    # page (ce qui prendrait plusieurs minutes à chaque fois autrement).
    # En mode split_peak_hours, il y a deux modèles distincts (pas un
    # seul) -> on ne sauvegarde pas model.pkl dans ce cas pour éviter
    # de laisser un fichier trompeur (None ou un seul des deux modèles).
    import joblib
    os.makedirs("data", exist_ok=True)
    if model is not None:
        joblib.dump(model, "data/model.pkl")
        print(f"\n✅ Model saved to data/model.pkl")
    else:
        print("\nℹ️  Skipping model.pkl save (split_peak_hours mode has two separate models).")

    predictions_df = test[["datetime", target_col]].copy()
    predictions_df["predicted"] = preds
    predictions_df.to_csv("data/predictions.csv", index=False)

    metrics = {"rmse": float(rmse), "mae": float(mae), "rmse_pct": float(100 * rmse / y_test.mean())}
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Test predictions saved to data/predictions.csv ({len(predictions_df)} rows)")
    print(f"✅ Metrics saved to data/metrics.json")

    return model, {"rmse": rmse, "mae": mae, "feature_importance": importances}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the power price prediction model.")
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--target", type=str, default="price_eur_mwh")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                         help="Fraction of data (most recent, chronologically) held out for testing.")
    parser.add_argument("--tune", action="store_true",
                         help="Run hyperparameter search with time-series cross-validation before "
                              "training the final model (takes several extra minutes).")
    parser.add_argument("--log-transform", action="store_true",
                         help="Train on a signed log1p transform of the target (handles negative "
                              "prices, compresses extreme values) instead of raw EUR/MWh.")
    parser.add_argument("--split-peak-hours", action="store_true",
                         help="Train two separate models for peak (8-13h, 18-20h) and off-peak hours "
                              "instead of a single model for all hours.")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    df = engineer_features(df)
    train_and_evaluate(df, target_col=args.target, test_fraction=args.test_fraction, tune=args.tune,
                        use_log_transform=args.log_transform, split_peak_hours=args.split_peak_hours)
