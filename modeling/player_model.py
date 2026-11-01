"""
Player points model (props).

  * Features are strictly pre-game: rolling/EWMA stats are computed on a player's *previous*
    games only (shift(1)), through one code path used for both training and live inference,
    so there is no train/serve skew and no lookahead.
  * Game-script features (expected team points, pace, blowout risk) come from the team model.
  * `miss_share` captures how much of the team's normal rotation minutes are absent (injuries,
    rest, trades) - the main driver of prop value that season averages cannot see.
  * Points are modelled with Poisson-loss gradient boosting, and the *distribution* with a
    negative binomial whose dispersion is fitted on out-of-fold residuals. That handles integer
    lines (pushes) and skew properly, unlike the old normal approximation with a heuristic std.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from config import MODEL, PROPS

logger = logging.getLogger(__name__)

FEATURES = [
    "ewm_pts", "ewm_minutes", "ewm_fga", "ewm_fta", "ewm_fg3a", "ewm_tov", "ppm_ewm",
    "last3_pts", "last5_pts", "last10_pts", "std10_pts", "season_avg_pts", "season_avg_min",
    "starter_rate10", "n_prior", "is_home", "rest", "b2b",
    "team_pts_hat", "opp_pts_hat", "poss_hat", "abs_margin_hat", "team_margin_hat",
    "miss_share", "top_missing",
]
BASE_COLS = ["pts", "minutes", "fga", "fta", "fg3a", "tov"]
OUT_STATUSES = ("out", "doubtful")


# ── rolling features (single code path for training + inference) ─────────────
def compute_player_features(rows: pd.DataFrame) -> pd.DataFrame:
    """
    rows: one row per PLAYED player-game (history + any placeholder rows for upcoming games, whose
    stat columns are NaN). Required columns: player_key, game_date, game_key, season, starter and BASE_COLS.
    Returns `rows` (sorted) with feature columns added; each row's features use only earlier rows.
    """
    pp = rows.sort_values(["player_key", "game_date", "game_key"]).reset_index(drop=True).copy()
    H = MODEL.PLAYER_EWM_HALFLIFE_GAMES
    g = pp.groupby("player_key", sort=False)
    for c in BASE_COLS:
        pp[f"ewm_{c}"] = g[c].transform(lambda s: s.shift(1).ewm(halflife=H).mean())
    pp["ppm_ewm"] = pp["ewm_pts"] / pp["ewm_minutes"].replace(0, np.nan)
    for n in (3, 5, 10):
        pp[f"last{n}_pts"] = g["pts"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
    pp["std10_pts"] = g["pts"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    pp["starter_rate10"] = g["starter"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    pp["n_prior"] = g.cumcount()
    gs = pp.groupby(["player_key", "season"], sort=False)
    pp["season_avg_pts"] = gs["pts"].transform(lambda s: s.shift(1).expanding().mean())
    pp["season_avg_min"] = gs["minutes"].transform(lambda s: s.shift(1).expanding().mean())
    return pp


# ── rotation / missing-teammates ─────────────────────────────────────────────
def rotation_from_games(last_games: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Players in the rotation (>=3 of last 10 games, >= ROTATION_MIN_MINUTES avg) -> avg minutes."""
    tot: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    for gm in last_games:
        for p, m in gm.items():
            tot[p] = tot.get(p, 0.0) + m
            cnt[p] = cnt.get(p, 0) + 1
    return {p: tot[p] / cnt[p] for p in tot if cnt[p] >= 3 and tot[p] / cnt[p] >= MODEL.ROTATION_MIN_MINUTES}


def missing_features(rotation: Dict[str, float], absent: Set[str]) -> Dict[str, float]:
    if not rotation:
        return {"miss_share": 0.0, "top_missing": 0.0}
    total = sum(rotation.values())
    miss = sum(m for p, m in rotation.items() if p in absent)
    top = max(rotation, key=rotation.get)
    return {"miss_share": miss / total if total else 0.0, "top_missing": 1.0 if top in absent else 0.0}


def historical_missing(pb: pd.DataFrame) -> pd.DataFrame:
    """miss_share / top_missing for every team-game from actual box scores (absent = rotation player who didn't play)."""
    out = []
    played = pb[pb["played"] == 1]
    for team_id, tg in played.groupby("team_id"):
        window: deque = deque(maxlen=10)
        for (date, key), gdf in tg.sort_values(["game_date", "game_key"]).groupby(["game_date", "game_key"], sort=True):
            here = dict(zip(gdf["player_key"], gdf["minutes"]))
            rot = rotation_from_games(list(window))
            feats = missing_features(rot, set(rot) - set(here))
            out.append({"game_key": key, "team_id": team_id, **feats})
            window.append(here)
    return pd.DataFrame(out, columns=["game_key", "team_id", "miss_share", "top_missing"])


# ── training frame ───────────────────────────────────────────────────────────
def build_training_frame(player_box: pd.DataFrame, tg: pd.DataFrame, game_preds: pd.DataFrame) -> pd.DataFrame:
    """
    player_box: warehouse player_box_df (all rows). tg: team-games (frames.build_team_games).
    game_preds: walk-forward ratings frame (GameModel/RatingEngine.walk_forward output) with r_* columns.
    """
    if player_box.empty:
        return pd.DataFrame()
    pb = player_box.copy()
    pb["game_date"] = pd.to_datetime(pb["game_date"])
    played = pb[(pb["played"] == 1) & (pb["minutes"] > 0)].copy()
    feats = compute_player_features(played)
    ctx = tg[["game_key", "team_id", "opp_id", "is_home", "rest", "b2b"]]
    feats = feats.merge(ctx, on=["game_key", "team_id"], how="left")
    gp = game_preds[["game_key", "r_home_pts", "r_away_pts", "r_poss", "r_margin"]].copy()
    feats = feats.merge(gp, on="game_key", how="left")
    home = feats["is_home"] == 1
    feats["team_pts_hat"] = np.where(home, feats["r_home_pts"], feats["r_away_pts"])
    feats["opp_pts_hat"] = np.where(home, feats["r_away_pts"], feats["r_home_pts"])
    feats["team_margin_hat"] = np.where(home, feats["r_margin"], -feats["r_margin"])
    feats["abs_margin_hat"] = feats["team_margin_hat"].abs()
    feats["poss_hat"] = feats["r_poss"]
    miss = historical_missing(pb)
    feats = feats.merge(miss, on=["game_key", "team_id"], how="left")
    feats[["miss_share", "top_missing"]] = feats[["miss_share", "top_missing"]].fillna(0.0)
    return feats.sort_values(["game_date", "game_key"]).reset_index(drop=True)


def training_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows with enough history and a real expected role. Selection uses PRE-game info only."""
    return ((frame["n_prior"] >= MODEL.MIN_PRIOR_GAMES) & (frame["ewm_minutes"] >= MODEL.MIN_TRAIN_MINUTES)
            & frame["team_pts_hat"].notna())


# ── inference frame ──────────────────────────────────────────────────────────
def roster_for_team(player_box: pd.DataFrame, team_id: int, game_date, injuries: Optional[pd.DataFrame] = None):
    """Players who've played for the team in its last 10 games, with their injury status (if any)."""
    pb = player_box[(player_box["team_id"] == team_id) & (player_box["game_date"] < pd.Timestamp(game_date))]
    if pb.empty:
        return [], {}, set()
    keys = sorted(pb["game_key"].unique(), key=lambda k: pb.loc[pb["game_key"] == k, "game_date"].iloc[0])[-10:]
    last = pb[pb["game_key"].isin(keys) & (pb["played"] == 1)]
    games = [dict(zip(g["player_key"], g["minutes"])) for _, g in last.groupby("game_key", sort=False)]
    status: Dict[str, str] = {}
    if injuries is not None and not injuries.empty:
        inj = injuries[injuries["team_id"] == team_id]
        status = {r["player_key"]: str(r["status"]).lower() for _, r in inj.iterrows()}
    absent = {p for p, s in status.items() if any(s.startswith(o) for o in OUT_STATUSES)}
    return games, status, absent


def build_inference_frame(player_box: pd.DataFrame, tg: pd.DataFrame, team_id: int, opp_id: int, is_home: bool,
                          game_date, team_pts_hat: float, opp_pts_hat: float, poss_hat: float,
                          injuries: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per candidate player on `team_id` for an upcoming game, using the SAME feature code as training."""
    from modeling.frames import context_for
    pb = player_box.copy()
    pb["game_date"] = pd.to_datetime(pb["game_date"])
    gd = pd.Timestamp(game_date)
    games, status, absent = roster_for_team(pb, team_id, gd, injuries)
    rot = rotation_from_games(games)
    miss = missing_features(rot, absent)
    candidates = set(p for g in games for p in g)
    if not candidates:
        return pd.DataFrame()
    hist = pb[(pb["played"] == 1) & (pb["minutes"] > 0) & (pb["player_key"].isin(candidates)) & (pb["game_date"] < gd)].copy()
    hist = hist[["player_key", "game_date", "game_key", "season", "starter"] + BASE_COLS]
    from config import season_for_date
    ph = pd.DataFrame({"player_key": sorted(candidates), "game_date": gd, "game_key": "__upcoming__",
                       "season": season_for_date(gd.date()), "starter": np.nan})
    for c in BASE_COLS:
        ph[c] = np.nan
    feats = compute_player_features(pd.concat([hist, ph], ignore_index=True))
    feats = feats[feats["game_key"] == "__upcoming__"].copy()
    ctx = context_for(tg, team_id, gd)
    margin_hat = team_pts_hat - opp_pts_hat
    feats = feats.assign(
        is_home=1.0 if is_home else 0.0, rest=ctx["rest"], b2b=ctx["b2b"], team_pts_hat=team_pts_hat,
        opp_pts_hat=opp_pts_hat, poss_hat=poss_hat, team_margin_hat=margin_hat, abs_margin_hat=abs(margin_hat),
        miss_share=miss["miss_share"], top_missing=miss["top_missing"], team_id=team_id, opp_id=opp_id)
    names = pb.drop_duplicates("player_key", keep="last").set_index("player_key")["player_name"]
    feats["player_name"] = feats["player_key"].map(names).fillna(feats["player_key"].str.title())
    feats["status"] = feats["player_key"].map(status).fillna("")
    feats["out"] = feats["player_key"].isin(absent)
    return feats.reset_index(drop=True)


# ── negative-binomial helpers ────────────────────────────────────────────────
def _nb_dist(mu, r):
    mu = np.maximum(np.asarray(mu, dtype=float), 0.05)
    return nbinom(r, r / (r + mu))


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """MLE of the NB dispersion r (variance = mu + mu^2 / r) on out-of-fold predictions."""
    y = np.asarray(y, dtype=float); mu = np.maximum(np.asarray(mu, dtype=float), 0.05)
    def nll(logr):
        return -_nb_dist(mu, np.exp(logr)).logpmf(y).sum()
    res = minimize_scalar(nll, bounds=(np.log(1.0), np.log(500.0)), method="bounded")
    return float(np.exp(res.x))


def over_under_probs(mu: float, line: float, r: float) -> Dict[str, float]:
    """P(over), P(under), P(push) for points ~ NB(mu, r). Integer lines can push; x.5 lines cannot."""
    d = _nb_dist(mu, r)
    if abs(line - round(line)) < 1e-9:                 # integer line
        k = int(round(line))
        p_push = float(d.pmf(k))
        p_under = float(d.cdf(k - 1))
        p_over = 1.0 - p_under - p_push
    else:
        k = int(np.floor(line))
        p_under = float(d.cdf(k))
        p_over, p_push = 1.0 - p_under, 0.0
    return {"over": p_over, "under": p_under, "push": p_push}


# ── model ────────────────────────────────────────────────────────────────────
def _hgb(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=60,
        l2_regularization=2.0, early_stopping=False, random_state=MODEL.RANDOM_STATE)


@dataclass
class PlayerModel:
    pts_model: Optional[object] = None
    min_model: Optional[object] = None
    nb_r: float = PROPS.DEFAULT_NB_R
    metrics: Dict = field(default_factory=dict)
    trained_through: Optional[str] = None

    def fit(self, frame: pd.DataFrame) -> "PlayerModel":
        df = frame[training_mask(frame)].sort_values(["game_date", "game_key"]).reset_index(drop=True)
        if len(df) < 500:
            raise ValueError(f"Only {len(df)} usable player-games; run `python cli.py backfill` first (need ~500+).")
        X, y, ym = df[FEATURES], df["pts"].values.astype(float), df["minutes"].values.astype(float)

        oof = np.full(len(df), np.nan); oof_min = np.full(len(df), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=min(MODEL.CV_FOLDS, max(2, len(df) // 2000))).split(X):
            oof[te] = _hgb("poisson").fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            oof_min[te] = _hgb("squared_error").fit(X.iloc[tr], ym[tr]).predict(X.iloc[te])
        ev = ~np.isnan(oof)
        r = fit_dispersion(y[ev], oof[ev])

        d = _nb_dist(oof[ev], r)
        lo, hi = d.ppf(0.10), d.ppf(0.90)
        cover80 = float(np.mean((y[ev] >= lo) & (y[ev] <= hi)))
        base_ewm = df.loc[ev, "ewm_pts"].values
        base_season = df.loc[ev, "season_avg_pts"].fillna(df.loc[ev, "ewm_pts"]).values
        tiers = {}
        for name, lo_m, hi_m in (("bench(<20 exp min)", 0, 20), ("rotation(20-30)", 20, 30), ("stars(30+)", 30, 99)):
            mk = (df.loc[ev, "ewm_minutes"].values >= lo_m) & (df.loc[ev, "ewm_minutes"].values < hi_m)
            if mk.sum() > 50:
                tiers[name] = {"n": int(mk.sum()), "mae": float(np.abs(y[ev][mk] - oof[ev][mk]).mean()),
                               "mae_ewma": float(np.abs(y[ev][mk] - base_ewm[mk]).mean())}
        self.metrics = {
            "n_rows": int(len(df)), "n_scored": int(ev.sum()),
            "pts_mae": float(np.abs(y[ev] - oof[ev]).mean()),
            "pts_mae_ewma_baseline": float(np.nanmean(np.abs(y[ev] - base_ewm))),
            "pts_mae_season_avg_baseline": float(np.nanmean(np.abs(y[ev] - base_season))),
            "pts_bias": float(np.mean(oof[ev] - y[ev])),
            "minutes_mae": float(np.abs(ym[ev] - oof_min[ev]).mean()),
            "nb_r": r, "coverage_80": cover80, "by_tier": tiers,
        }
        self.nb_r = r
        self.pts_model = _hgb("poisson").fit(X, y)
        self.min_model = _hgb("squared_error").fit(X, ym)
        self.trained_through = str(df["game_date"].max().date())
        return self

    def predict_frame(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = feats[FEATURES]
        out = feats[["player_key"]].copy()
        out["proj_pts"] = self.pts_model.predict(X)
        out["proj_min"] = np.clip(self.min_model.predict(X), 0, 48)
        return out

    def prop_probs(self, mu: float, line: float) -> Dict[str, float]:
        return over_under_probs(mu, line, self.nb_r)

    def save(self, directory: str = None) -> str:
        directory = directory or MODEL.MODELS_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "player_model.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(directory: str = None) -> Optional["PlayerModel"]:
        path = os.path.join(directory or MODEL.MODELS_DIR, "player_model.joblib")
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            logger.warning("Could not load %s (%s); retrain with `python cli.py train`.", path, e)
            return None
