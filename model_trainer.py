"""
Enhanced Model Trainer — rebuilt with:
- TEMPORAL ISOLATION: feature creation only uses games BEFORE the target game
- Consistent feature schema via a canonical feature list
- Integration with new config.py and data_collector.py
- Clean separation between feature engineering and model fitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from config import TRAINING, LEAGUE, PREDICTION
from data_collector import DataCollector

logger = logging.getLogger(__name__)


# ── Canonical Feature Lists ──────────────────────────────────────────────────
# These define the EXACT feature columns used during training.
# Prediction must produce the same columns in the same order.

TEAM_FEATURE_COLS = [
    "PTS_AVG", "PTS_STD", "FG_PCT", "FG3_PCT", "FT_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "OFF_RATING", "AST_RATE", "TOV_RATE",
    "REB_AVG", "STL_AVG", "BLK_AVG",
    "PTS_L3", "PTS_L5", "PTS_L10",
    "FG_PCT_L5", "FG3_PCT_L5",
    "WIN_PCT_L5", "WIN_PCT_L10",
    "PTS_TREND", "FG_PCT_TREND",
    "HOME_PTS_AVG", "AWAY_PTS_AVG",
    "HOME_GAME", "REST_DAYS", "SEASON_WEIGHT",
]

PLAYER_FEATURE_COLS = [
    "PTS_AVG", "PTS_STD", "MIN_AVG", "MIN_STD",
    "FG_PCT", "FG3_PCT", "FT_PCT", "USAGE",
    "PTS_L3", "PTS_L5", "PTS_L10",
    "MIN_L3", "MIN_L5", "MIN_L10",
    "PTS_PER_MIN", "PTS_PER_MIN_L5",
    "HOT_COLD_FACTOR", "MINUTES_TREND",
    "HOME_PTS_AVG", "AWAY_PTS_AVG",
    "TEAM_PACE", "TEAM_PTS_AVG",
    "OPP_DEF_RATING", "OPP_PACE",
    "HOME_GAME", "REST_DAYS", "GAMES_PLAYED", "SEASON_WEIGHT",
]


class ModelTrainer:
    """
    Model trainer with temporal isolation and consistent feature schemas.
    """

    def __init__(self, collector: DataCollector):
        self.collector = collector

        # Models and preprocessing
        self.team_models: Dict = {}
        self.player_models: Dict = {"points": {}, "minutes": {}}
        self.scalers: Dict = {}
        self.feature_selectors: Dict = {}
        self.team_ensemble_weights: Dict = {}
        self.player_ensemble_weights: Dict = {"points": {}, "minutes": {}}

        # Model configs
        self.team_model_configs = {
            "xgb": XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=TRAINING.RANDOM_STATE,
            ),
            "lgb": lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=TRAINING.RANDOM_STATE, verbosity=-1,
            ),
            "rf": RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=TRAINING.RANDOM_STATE,
            ),
        }

        self.player_model_configs = {
            "xgb": XGBRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=6,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=0.5,
                random_state=TRAINING.RANDOM_STATE,
            ),
            "lgb": lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=6,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=0.5,
                random_state=TRAINING.RANDOM_STATE, verbosity=-1,
            ),
            "rf": RandomForestRegressor(
                n_estimators=150, max_depth=8,
                min_samples_split=3, min_samples_leaf=1,
                random_state=TRAINING.RANDOM_STATE,
            ),
        }

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING — TEMPORAL ISOLATION
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def create_team_features_from_history(
        history_df: pd.DataFrame,
        is_home: bool = True,
        rest_days: int = 2,
        season_weight: float = 1.0,
    ) -> Optional[Dict]:
        """
        Create team features from a DataFrame of ONLY historical games
        (games that occurred BEFORE the target game).

        This is the key fix for training data leakage: the caller is
        responsible for slicing the DataFrame to exclude the target game
        and all future games.
        """
        if history_df is None or len(history_df) < TRAINING.MIN_GAMES_FOR_FEATURES:
            return None

        df = history_df.copy()

        # Ensure numeric
        num_cols = ["PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM",
                    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        f = {}

        # Season-long aggregates (from history only)
        f["PTS_AVG"] = df["PTS"].mean()
        f["PTS_STD"] = df["PTS"].std() if len(df) > 1 else 12.0
        f["FG_PCT"] = df["FGM"].sum() / max(1, df["FGA"].sum())
        f["FG3_PCT"] = df["FG3M"].sum() / max(1, df["FG3A"].sum())
        f["FT_PCT"] = df["FTM"].sum() / max(1, df["FTA"].sum())
        f["EFG_PCT"] = (df["FGM"].sum() + 0.5 * df["FG3M"].sum()) / max(1, df["FGA"].sum())

        total_pts = df["PTS"].sum()
        total_fga = df["FGA"].sum()
        total_fta = df["FTA"].sum()
        f["TS_PCT"] = total_pts / max(1, 2 * (total_fga + 0.44 * total_fta))

        poss = df["FGA"] + 0.44 * df["FTA"] + df["TOV"] - df["OREB"]
        f["PACE"] = poss.mean()
        f["OFF_RATING"] = (total_pts / max(1, poss.sum())) * 100
        f["AST_RATE"] = df["AST"].sum() / max(1, df["FGM"].sum())
        f["TOV_RATE"] = df["TOV"].sum() / max(1, poss.sum()) * 100
        f["REB_AVG"] = df["REB"].mean()
        f["STL_AVG"] = df["STL"].mean()
        f["BLK_AVG"] = df["BLK"].mean()

        # Rolling windows (most recent N games from history)
        for w in [3, 5, 10]:
            if len(df) >= w:
                window = df.head(w)
                f[f"PTS_L{w}"] = window["PTS"].mean()
                if w == 5:
                    f["FG_PCT_L5"] = window["FGM"].sum() / max(1, window["FGA"].sum())
                    f["FG3_PCT_L5"] = window["FG3M"].sum() / max(1, window["FG3A"].sum())
                if "WL" in window.columns:
                    f[f"WIN_PCT_L{w}"] = (window["WL"] == "W").mean()
            else:
                f[f"PTS_L{w}"] = f["PTS_AVG"]
                if w == 5:
                    f["FG_PCT_L5"] = f["FG_PCT"]
                    f["FG3_PCT_L5"] = f["FG3_PCT"]
                f[f"WIN_PCT_L{w}"] = 0.5

        # Trends (slope over last 10 historical games)
        trend_n = min(10, len(df))
        if trend_n >= 3:
            x = np.arange(trend_n)
            f["PTS_TREND"] = float(np.polyfit(x, df["PTS"].head(trend_n).values, 1)[0])
            fg_vals = (df["FGM"].head(trend_n) / df["FGA"].head(trend_n).replace(0, 1)).values
            f["FG_PCT_TREND"] = float(np.polyfit(x, fg_vals, 1)[0])
        else:
            f["PTS_TREND"] = 0.0
            f["FG_PCT_TREND"] = 0.0

        # Home/away splits from history
        if "MATCHUP" in df.columns:
            home_g = df[df["MATCHUP"].str.contains("vs.", na=False)]
            away_g = df[df["MATCHUP"].str.contains("@", na=False)]
            f["HOME_PTS_AVG"] = home_g["PTS"].mean() if not home_g.empty else f["PTS_AVG"]
            f["AWAY_PTS_AVG"] = away_g["PTS"].mean() if not away_g.empty else f["PTS_AVG"]
        else:
            f["HOME_PTS_AVG"] = f["PTS_AVG"]
            f["AWAY_PTS_AVG"] = f["PTS_AVG"]

        # Context features (set by caller)
        f["HOME_GAME"] = 1.0 if is_home else 0.0
        f["REST_DAYS"] = float(rest_days)
        f["SEASON_WEIGHT"] = season_weight

        return f

    @staticmethod
    def create_player_features_from_history(
        player_history: pd.DataFrame,
        team_stats: Dict,
        opp_stats: Dict,
        is_home: bool = True,
        rest_days: int = 2,
        season_weight: float = 1.0,
        convert_minutes_fn=None,
    ) -> Optional[Dict]:
        """
        Create player features from ONLY historical games.
        Same temporal isolation principle as team features.
        """
        if player_history is None or len(player_history) < TRAINING.MIN_GAMES_FOR_FEATURES:
            return None

        df = player_history.copy()

        # Convert minutes
        if convert_minutes_fn:
            df["MIN"] = df["MIN"].apply(convert_minutes_fn)
        else:
            df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)

        num_cols = ["PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM",
                    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "MIN"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        active = df[df["MIN"] > 0]
        if len(active) < 3:
            return None

        f = {}
        f["PTS_AVG"] = active["PTS"].mean()
        f["PTS_STD"] = active["PTS"].std() if len(active) > 1 else 5.0
        f["MIN_AVG"] = active["MIN"].mean()
        f["MIN_STD"] = active["MIN"].std() if len(active) > 1 else 5.0
        f["FG_PCT"] = active["FGM"].sum() / max(1, active["FGA"].sum())
        f["FG3_PCT"] = active["FG3M"].sum() / max(1, active["FG3A"].sum())
        f["FT_PCT"] = active["FTM"].sum() / max(1, active["FTA"].sum())

        # Usage estimate
        poss_per_min = 100 / 48
        usages = []
        for _, g in active.iterrows():
            if g["MIN"] > 0:
                player_poss = g["FGA"] + 0.44 * g["FTA"] + g["TOV"]
                team_poss = poss_per_min * g["MIN"]
                usages.append(min(1.0, player_poss / max(1, team_poss)))
        f["USAGE"] = np.mean(usages) if usages else 0.2

        # Rolling windows
        for w in [3, 5, 10]:
            if len(active) >= w:
                window = active.head(w)
                f[f"PTS_L{w}"] = window["PTS"].mean()
                f[f"MIN_L{w}"] = window["MIN"].mean()
            else:
                f[f"PTS_L{w}"] = f["PTS_AVG"]
                f[f"MIN_L{w}"] = f["MIN_AVG"]

        f["PTS_PER_MIN"] = active["PTS"].sum() / max(1, active["MIN"].sum())

        if len(active) >= 5:
            r5 = active.head(5)
            f["PTS_PER_MIN_L5"] = r5["PTS"].sum() / max(1, r5["MIN"].sum())
            f["HOT_COLD_FACTOR"] = (r5["PTS"].mean() - f["PTS_AVG"]) / max(1, f["PTS_AVG"])
            f["MINUTES_TREND"] = (r5["MIN"].mean() - f["MIN_AVG"]) / max(1, f["MIN_AVG"])
        else:
            f["PTS_PER_MIN_L5"] = f["PTS_PER_MIN"]
            f["HOT_COLD_FACTOR"] = 0.0
            f["MINUTES_TREND"] = 0.0

        # Home/away splits
        if "MATCHUP" in active.columns:
            home_g = active[active["MATCHUP"].str.contains("vs.", na=False)]
            away_g = active[active["MATCHUP"].str.contains("@", na=False)]
            f["HOME_PTS_AVG"] = home_g["PTS"].mean() if not home_g.empty else f["PTS_AVG"]
            f["AWAY_PTS_AVG"] = away_g["PTS"].mean() if not away_g.empty else f["PTS_AVG"]
        else:
            f["HOME_PTS_AVG"] = f["PTS_AVG"]
            f["AWAY_PTS_AVG"] = f["PTS_AVG"]

        # Team and opponent context
        f["TEAM_PACE"] = team_stats.get("pace", LEAGUE.AVG_PACE) if team_stats else LEAGUE.AVG_PACE
        f["TEAM_PTS_AVG"] = team_stats.get("pts_avg", LEAGUE.AVG_TEAM_SCORE) if team_stats else LEAGUE.AVG_TEAM_SCORE
        f["OPP_DEF_RATING"] = opp_stats.get("off_rating", LEAGUE.AVG_DEF_RATING) if opp_stats else LEAGUE.AVG_DEF_RATING
        f["OPP_PACE"] = opp_stats.get("pace", LEAGUE.AVG_PACE) if opp_stats else LEAGUE.AVG_PACE

        f["HOME_GAME"] = 1.0 if is_home else 0.0
        f["REST_DAYS"] = float(rest_days)
        f["GAMES_PLAYED"] = float(len(active))
        f["SEASON_WEIGHT"] = season_weight

        return f

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING DATA PREPARATION — WITH TEMPORAL ISOLATION
    # ══════════════════════════════════════════════════════════════════════

    def prepare_training_data(self) -> Tuple[List, List]:
        """
        Build training data with strict temporal isolation.

        For each training sample:
        - Target = actual points scored in game i
        - Features = computed from games i+1, i+2, ... (all games BEFORE game i)

        Game logs are sorted most-recent-first, so iloc[i+1:] gives
        all games that happened before game i.
        """
        from nba_api.stats.static import teams as nba_teams_static

        all_nba_teams = nba_teams_static.get_teams()
        team_data = []
        player_data = []

        logger.info("Preparing team training data with temporal isolation...")

        for season in TRAINING.SEASONS:
            sw = TRAINING.SEASON_WEIGHTS.get(season, 0.7)

            for team_info in all_nba_teams:
                team_id = team_info["id"]
                try:
                    games = self.collector.get_team_recent_games(team_id, n_games=82, season=season)
                    if games is None or len(games) < 15:
                        continue

                    # Ensure numeric PTS
                    games["PTS"] = pd.to_numeric(games["PTS"], errors="coerce").fillna(0)

                    # For each game from index 10 onwards, use games AFTER that
                    # index (i.e., older games) as history
                    for i in range(10, min(70, len(games))):
                        target_game = games.iloc[i]
                        target_pts = float(target_game["PTS"])

                        # TEMPORAL ISOLATION: only use games that happened BEFORE
                        # this one (higher indices = older games in this sort order)
                        history = games.iloc[i + 1:]

                        if len(history) < TRAINING.MIN_GAMES_FOR_FEATURES:
                            continue

                        # Determine context from the target game
                        is_home = "vs." in str(target_game.get("MATCHUP", ""))

                        # Calculate rest days from the game before
                        rest = 2  # default
                        if i + 1 < len(games):
                            try:
                                d1 = pd.to_datetime(target_game["GAME_DATE"])
                                d2 = pd.to_datetime(games.iloc[i + 1]["GAME_DATE"])
                                rest = max(1, (d1 - d2).days)
                            except Exception:
                                pass

                        features = self.create_team_features_from_history(
                            history, is_home=is_home, rest_days=rest, season_weight=sw,
                        )
                        if features:
                            team_data.append({
                                "features": features,
                                "target": target_pts,
                                "weight": sw,
                            })

                except Exception as e:
                    logger.warning(f"Error processing team {team_info['full_name']}: {e}")

        logger.info(f"Collected {len(team_data)} team training samples")

        # ── Player data ──────────────────────────────────────────────────
        logger.info("Preparing player training data with temporal isolation...")

        from nba_api.stats.static import players as nba_players_static
        all_players = nba_players_static.get_players()

        # Sample active players (limit for training speed)
        for player_info in all_players[:300]:
            player_id = player_info["id"]
            try:
                games = self.collector.get_player_recent_games(player_id, n_games=60)
                if games is None or len(games) < 10:
                    continue

                games["PTS"] = pd.to_numeric(games["PTS"], errors="coerce").fillna(0)
                games["MIN"] = games["MIN"].apply(self.collector._convert_minutes)

                for i in range(5, min(45, len(games))):
                    target_game = games.iloc[i]
                    target_pts = float(target_game["PTS"])
                    target_min = float(target_game["MIN"])

                    if target_min < 5:
                        continue  # skip garbage time / DNP-ish

                    # TEMPORAL ISOLATION
                    history = games.iloc[i + 1:]

                    if len(history) < TRAINING.MIN_GAMES_FOR_FEATURES:
                        continue

                    is_home = "vs." in str(target_game.get("MATCHUP", ""))
                    rest = 2
                    if i + 1 < len(games):
                        try:
                            d1 = pd.to_datetime(target_game["GAME_DATE"])
                            d2 = pd.to_datetime(games.iloc[i + 1]["GAME_DATE"])
                            rest = max(1, (d1 - d2).days)
                        except Exception:
                            pass

                    features = self.create_player_features_from_history(
                        history,
                        team_stats={},  # simplified for training
                        opp_stats={},
                        is_home=is_home,
                        rest_days=rest,
                        season_weight=1.0,
                        convert_minutes_fn=self.collector._convert_minutes,
                    )
                    if features:
                        player_data.append({
                            "features": features,
                            "target_points": target_pts,
                            "target_minutes": target_min,
                            "player_id": player_id,
                        })

            except Exception as e:
                logger.warning(f"Error processing player {player_info['full_name']}: {e}")

        logger.info(f"Collected {len(player_data)} player training samples")
        return team_data, player_data

    # ══════════════════════════════════════════════════════════════════════
    # MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════════

    def train_ensemble_models(self) -> bool:
        """Train team and player ensemble models."""
        team_data, player_data = self.prepare_training_data()

        if len(team_data) < TRAINING.MIN_TEAM_TRAINING_SAMPLES:
            logger.error(f"Insufficient team data: {len(team_data)} < {TRAINING.MIN_TEAM_TRAINING_SAMPLES}")
            return False
        if len(player_data) < TRAINING.MIN_PLAYER_TRAINING_SAMPLES:
            logger.error(f"Insufficient player data: {len(player_data)} < {TRAINING.MIN_PLAYER_TRAINING_SAMPLES}")
            return False

        ok = self._train_team_ensemble(team_data)
        if not ok:
            return False
        ok = self._train_player_ensemble(player_data)
        return ok

    def _train_team_ensemble(self, team_data: List[Dict]) -> bool:
        """Train team score prediction ensemble."""
        try:
            X = pd.DataFrame([s["features"] for s in team_data])
            y = np.array([s["target"] for s in team_data])
            weights = np.array([s["weight"] for s in team_data])

            # Enforce canonical column order, fill missing with 0
            X = X.reindex(columns=TEAM_FEATURE_COLS, fill_value=0.0).fillna(0.0)

            # Feature selection
            k = min(TRAINING.FEATURE_SELECTION_K_TEAM, len(TEAM_FEATURE_COLS))
            selector = SelectKBest(score_func=f_regression, k=k)
            X_sel = selector.fit_transform(X, y)
            self.feature_selectors["team"] = selector

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_sel)
            self.scalers["team"] = scaler

            tscv = TimeSeriesSplit(n_splits=TRAINING.CV_SPLITS)
            models = {}
            scores = {}

            for name, model in self.team_model_configs.items():
                logger.info(f"Training team model: {name}")
                cv = cross_val_score(
                    model, X_scaled, y, cv=tscv,
                    scoring="neg_mean_absolute_error",
                    fit_params={"sample_weight": weights} if name in ("xgb", "lgb") else {},
                )
                scores[name] = -cv.mean()
                logger.info(f"  {name} CV MAE: {scores[name]:.2f}")

                if name in ("xgb", "lgb"):
                    model.fit(X_scaled, y, sample_weight=weights)
                else:
                    model.fit(X_scaled, y)
                models[name] = model

            self.team_models = models
            inv_total = sum(1 / s for s in scores.values())
            self.team_ensemble_weights = {n: (1 / s) / inv_total for n, s in scores.items()}
            logger.info(f"Team ensemble weights: {self.team_ensemble_weights}")
            return True

        except Exception as e:
            logger.error(f"Team ensemble training failed: {e}")
            return False

    def _train_player_ensemble(self, player_data: List[Dict]) -> bool:
        """Train player points and minutes prediction ensembles."""
        try:
            X = pd.DataFrame([s["features"] for s in player_data])
            y_pts = np.array([s["target_points"] for s in player_data])
            y_min = np.array([s["target_minutes"] for s in player_data])

            X = X.reindex(columns=PLAYER_FEATURE_COLS, fill_value=0.0).fillna(0.0)

            # Points models
            k = min(TRAINING.FEATURE_SELECTION_K_PLAYER, len(PLAYER_FEATURE_COLS))
            pts_selector = SelectKBest(score_func=f_regression, k=k)
            X_pts = pts_selector.fit_transform(X, y_pts)
            self.feature_selectors["player_points"] = pts_selector

            pts_scaler = RobustScaler()
            X_pts_scaled = pts_scaler.fit_transform(X_pts)
            self.scalers["player_points"] = pts_scaler

            # Minutes models
            min_selector = SelectKBest(score_func=f_regression, k=k)
            X_min = min_selector.fit_transform(X, y_min)
            self.feature_selectors["player_minutes"] = min_selector

            min_scaler = RobustScaler()
            X_min_scaled = min_scaler.fit_transform(X_min)
            self.scalers["player_minutes"] = min_scaler

            tscv = TimeSeriesSplit(n_splits=3)

            # Train points models
            pts_models = {}
            pts_scores = {}
            for name, model in self.player_model_configs.items():
                cv = cross_val_score(model, X_pts_scaled, y_pts, cv=tscv, scoring="neg_mean_absolute_error")
                pts_scores[name] = -cv.mean()
                logger.info(f"Player points {name} CV MAE: {pts_scores[name]:.2f}")
                model.fit(X_pts_scaled, y_pts)
                pts_models[name] = model
            self.player_models["points"] = pts_models

            # Train minutes models
            min_models = {}
            min_scores = {}
            for name, model in self.player_model_configs.items():
                cv = cross_val_score(model, X_min_scaled, y_min, cv=tscv, scoring="neg_mean_absolute_error")
                min_scores[name] = -cv.mean()
                logger.info(f"Player minutes {name} CV MAE: {min_scores[name]:.2f}")
                model.fit(X_min_scaled, y_min)
                min_models[name] = model
            self.player_models["minutes"] = min_models

            # Ensemble weights
            pts_inv = sum(1 / s for s in pts_scores.values())
            min_inv = sum(1 / s for s in min_scores.values())
            self.player_ensemble_weights = {
                "points": {n: (1 / s) / pts_inv for n, s in pts_scores.items()},
                "minutes": {n: (1 / s) / min_inv for n, s in min_scores.items()},
            }
            return True

        except Exception as e:
            logger.error(f"Player ensemble training failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    # PREDICTION (uses same canonical feature schema)
    # ══════════════════════════════════════════════════════════════════════

    def predict_team_score(self, team_id: int, opponent_id: int = None, is_home: bool = True) -> Optional[Dict]:
        """Predict team score using trained ensemble."""
        try:
            games = self.collector.get_team_recent_games(team_id, n_games=30)
            if games is None or len(games) < TRAINING.MIN_GAMES_FOR_FEATURES:
                return None

            # Calculate rest
            rest = 2
            if len(games) >= 2:
                try:
                    d = pd.to_datetime(games["GAME_DATE"])
                    rest = max(1, (d.iloc[0] - d.iloc[1]).days)
                except Exception:
                    pass

            features = self.create_team_features_from_history(
                games, is_home=is_home, rest_days=rest,
            )
            if not features:
                return None

            X = pd.DataFrame([features]).reindex(columns=TEAM_FEATURE_COLS, fill_value=0.0).fillna(0.0)

            selector = self.feature_selectors.get("team")
            scaler = self.scalers.get("team")
            if not selector or not scaler:
                return None

            X_sel = selector.transform(X)
            X_scaled = scaler.transform(X_sel)

            preds = {}
            for name, model in self.team_models.items():
                preds[name] = float(model.predict(X_scaled)[0])

            ensemble = sum(p * self.team_ensemble_weights[n] for n, p in preds.items())

            return {
                "prediction": ensemble,
                "std": float(np.std(list(preds.values()))),
                "individual_predictions": preds,
            }
        except Exception as e:
            logger.error(f"Team score prediction error: {e}")
            return None

    def predict_player_performance(
        self, player_id: int, team_id: int,
        opponent_id: int = None, is_home: bool = True,
    ) -> Optional[Dict]:
        """Predict player points and minutes using trained ensemble."""
        try:
            games = self.collector.get_player_recent_games(player_id, n_games=30)
            if games is None or len(games) < TRAINING.MIN_GAMES_FOR_FEATURES:
                return None

            rest = 2
            if len(games) >= 2:
                try:
                    d = pd.to_datetime(games["GAME_DATE"])
                    rest = max(1, (d.iloc[0] - d.iloc[1]).days)
                except Exception:
                    pass

            team_stats = self.collector.get_team_stats(team_id)
            opp_stats = self.collector.get_team_stats(opponent_id) if opponent_id else {}

            features = self.create_player_features_from_history(
                games, team_stats=team_stats or {}, opp_stats=opp_stats or {},
                is_home=is_home, rest_days=rest,
                convert_minutes_fn=self.collector._convert_minutes,
            )
            if not features:
                return None

            X = pd.DataFrame([features]).reindex(columns=PLAYER_FEATURE_COLS, fill_value=0.0).fillna(0.0)

            result = {}

            # Points prediction
            pts_sel = self.feature_selectors.get("player_points")
            pts_scaler = self.scalers.get("player_points")
            if pts_sel and pts_scaler and self.player_models.get("points"):
                X_pts = pts_scaler.transform(pts_sel.transform(X))
                pts_preds = {n: float(m.predict(X_pts)[0]) for n, m in self.player_models["points"].items()}
                result["points"] = max(0, sum(
                    p * self.player_ensemble_weights["points"][n] for n, p in pts_preds.items()
                ))
                result["points_std"] = max(2.0, float(np.std(list(pts_preds.values()))))
            else:
                result["points"] = features.get("PTS_AVG", 0)
                result["points_std"] = PREDICTION.DEFAULT_PLAYER_POINTS_STD

            # Minutes prediction
            min_sel = self.feature_selectors.get("player_minutes")
            min_scaler = self.scalers.get("player_minutes")
            if min_sel and min_scaler and self.player_models.get("minutes"):
                X_min = min_scaler.transform(min_sel.transform(X))
                min_preds = {n: float(m.predict(X_min)[0]) for n, m in self.player_models["minutes"].items()}
                result["minutes"] = max(0, sum(
                    p * self.player_ensemble_weights["minutes"][n] for n, p in min_preds.items()
                ))
                result["minutes_std"] = max(2.0, float(np.std(list(min_preds.values()))))
            else:
                result["minutes"] = features.get("MIN_AVG", 20)
                result["minutes_std"] = PREDICTION.DEFAULT_PLAYER_MINUTES_STD

            return result
        except Exception as e:
            logger.error(f"Player prediction error: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════════

    def save_models(self, path: str = None):
        path = path or TRAINING.MODELS_DIR
        os.makedirs(path, exist_ok=True)

        for name, model in self.team_models.items():
            joblib.dump(model, f"{path}/team_{name}.joblib")
        for target_type, models in self.player_models.items():
            for name, model in models.items():
                joblib.dump(model, f"{path}/player_{target_type}_{name}.joblib")
        for name, s in self.scalers.items():
            joblib.dump(s, f"{path}/scaler_{name}.joblib")
        for name, s in self.feature_selectors.items():
            joblib.dump(s, f"{path}/selector_{name}.joblib")
        joblib.dump(self.team_ensemble_weights, f"{path}/team_weights.joblib")
        joblib.dump(self.player_ensemble_weights, f"{path}/player_weights.joblib")
        logger.info(f"Models saved to {path}/")

    def load_models(self, path: str = None) -> bool:
        path = path or TRAINING.MODELS_DIR
        try:
            self.team_models = {}
            for name in self.team_model_configs:
                p = f"{path}/team_{name}.joblib"
                if os.path.exists(p):
                    self.team_models[name] = joblib.load(p)

            self.player_models = {"points": {}, "minutes": {}}
            for target in ("points", "minutes"):
                for name in self.player_model_configs:
                    p = f"{path}/player_{target}_{name}.joblib"
                    if os.path.exists(p):
                        self.player_models[target][name] = joblib.load(p)

            for f in os.listdir(path):
                if f.startswith("scaler_"):
                    name = f.replace("scaler_", "").replace(".joblib", "")
                    self.scalers[name] = joblib.load(f"{path}/{f}")
                elif f.startswith("selector_"):
                    name = f.replace("selector_", "").replace(".joblib", "")
                    self.feature_selectors[name] = joblib.load(f"{path}/{f}")

            w = f"{path}/team_weights.joblib"
            if os.path.exists(w):
                self.team_ensemble_weights = joblib.load(w)
            w = f"{path}/player_weights.joblib"
            if os.path.exists(w):
                self.player_ensemble_weights = joblib.load(w)

            logger.info("Models loaded successfully")
            return bool(self.team_models)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False


# Backward compat alias
EnhancedModelTrainer = ModelTrainer
