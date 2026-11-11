"""
Game model: predicts home margin and total.

  1. Walk-forward team ratings (modeling/ratings.py) give a base margin/total.
  2. A small ridge model learns residual effects of rest / back-to-backs / schedule density.
     It is only kept for a target if it improves out-of-fold error (otherwise it is skipped).
  3. Out-of-fold residuals give the *fitted* margin and total sigmas used by the EV engine.
     The old code used the spread of ensemble members as "std", which is ~1-2 points and made
     every probability wildly over-confident.
"""

from __future__ import annotations

<<<<<<< HEAD
import copy
=======
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL
from modeling.frames import context_for
from modeling.ratings import RatingEngine, RatingState

logger = logging.getLogger(__name__)

MIN_RESID_GAIN = 0.004
RESID_FEATURES = ["rest_h", "rest_a", "b2b_h", "b2b_a", "g4_h", "g4_a", "rest_diff", "playoff"]


def _resid_X(g: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=g.index)
    X["rest_h"] = g["rest_h"].clip(upper=5)
    X["rest_a"] = g["rest_a"].clip(upper=5)
    X["b2b_h"], X["b2b_a"] = g["b2b_h"], g["b2b_a"]
    X["g4_h"], X["g4_a"] = g["g4_h"], g["g4_a"]
    X["rest_diff"] = X["rest_h"] - X["rest_a"]
    X["playoff"] = (g["season_type"] == 3).astype(float) if "season_type" in g else 0.0
    return X[RESID_FEATURES].fillna(0.0)


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 12)))


@dataclass
class GameModel:
    state: Optional[RatingState] = None
    resid_margin: Optional[object] = None
    resid_total: Optional[object] = None
    margin_sigma: float = MODEL.DEFAULT_MARGIN_SIGMA
    total_sigma: float = MODEL.DEFAULT_TOTAL_SIGMA
    metrics: Dict = field(default_factory=dict)
    trained_through: Optional[str] = None
<<<<<<< HEAD
    oof: Optional[pd.DataFrame] = None        # out-of-sample predictions per game (used by the backtest; not saved)
    last_wf: Optional[pd.DataFrame] = None    # walk-forward ratings frame from the last fit (not saved)
=======
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b

    # ── training / evaluation ────────────────────────────────────────────
    def fit(self, g: pd.DataFrame, engine: Optional[RatingEngine] = None) -> "GameModel":
        engine = engine or RatingEngine()
        wf, final_state, _ = engine.walk_forward(g)
        wf = wf.dropna(subset=["r_margin"]).copy()
        for c in ("r_margin", "r_total", "r_home_pts", "r_away_pts", "r_poss"):
            wf[c] = wf[c].astype(float)
        wf["res_margin"] = wf["margin"] - wf["r_margin"]
        wf["res_total"] = wf["total"] - wf["r_total"]
        ok = (wf["gno_h"] >= MODEL.MIN_TEAM_GAMES_FOR_EVAL) & (wf["gno_a"] >= MODEL.MIN_TEAM_GAMES_FOR_EVAL)
        X = _resid_X(wf)

        # out-of-fold residual-model predictions (time-ordered folds, no shuffling)
        oof_m = np.zeros(len(wf)); oof_t = np.zeros(len(wf)); has_oof = np.zeros(len(wf), bool)
        n_splits = min(MODEL.CV_FOLDS, max(2, len(wf) // 300))
        if len(wf) >= 400:
            for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
                mm = _ridge().fit(X.iloc[tr][ok.iloc[tr].values], wf["res_margin"].iloc[tr][ok.iloc[tr].values])
                mt = _ridge().fit(X.iloc[tr][ok.iloc[tr].values], wf["res_total"].iloc[tr][ok.iloc[tr].values])
                oof_m[te], oof_t[te] = mm.predict(X.iloc[te]), mt.predict(X.iloc[te])
                has_oof[te] = True

        ev = ok.values & (has_oof if has_oof.any() else ok.values)
        base_mae_m = float(wf.loc[ev, "res_margin"].abs().mean()) if ev.any() else float("nan")
        base_mae_t = float(wf.loc[ev, "res_total"].abs().mean()) if ev.any() else float("nan")
        use_m = use_t = False
        if has_oof.any() and ev.any():
            mae_m = float(np.abs(wf.loc[ev, "res_margin"].values - oof_m[ev]).mean())
            mae_t = float(np.abs(wf.loc[ev, "res_total"].values - oof_t[ev]).mean())
            # Only keep a residual model if it clearly helps out-of-fold (>0.4% lower MAE); noise-level gains don't count.
            use_m = mae_m < base_mae_m * (1 - MIN_RESID_GAIN)
            use_t = mae_t < base_mae_t * (1 - MIN_RESID_GAIN)
        else:
            mae_m, mae_t = base_mae_m, base_mae_t

        final_res_m = final_res_t = None
        if use_m or use_t:
            fit_rows = ok.values
            if use_m:
                final_res_m = _ridge().fit(X[fit_rows], wf.loc[fit_rows, "res_margin"])
            if use_t:
                final_res_t = _ridge().fit(X[fit_rows], wf.loc[fit_rows, "res_total"])

        pred_m = wf["r_margin"].values + (oof_m if use_m else 0.0)
        pred_t = wf["r_total"].values + (oof_t if use_t else 0.0)
        err_m, err_t = (wf["margin"].values - pred_m)[ev], (wf["total"].values - pred_t)[ev]

        # naive baseline = what the old app did: each team's season-to-date points average + fixed home court
        naive_m, naive_t = self._naive_baseline(wf)
        nm = (wf["margin"].values - naive_m)[ev]
        nt = (wf["total"].values - naive_t)[ev]

        sig_m = float(np.std(err_m)) if len(err_m) > 50 else MODEL.DEFAULT_MARGIN_SIGMA
        sig_t = float(np.std(err_t)) if len(err_t) > 50 else MODEL.DEFAULT_TOTAL_SIGMA
        home_win = (wf["margin"].values > 0)[ev]
        p_home = norm.cdf(pred_m[ev] / sig_m)
        ll = float(-np.mean(np.where(home_win, np.log(np.clip(p_home, 1e-6, 1)), np.log(np.clip(1 - p_home, 1e-6, 1)))))
        p0 = home_win.mean() if len(home_win) else 0.5
        ll0 = float(-np.mean(np.where(home_win, np.log(p0), np.log(1 - p0)))) if len(home_win) else float("nan")

<<<<<<< HEAD
        self.oof = pd.DataFrame({
            "game_key": wf["game_key"].values, "game_date": wf["game_date"].values, "season": wf["season"].values,
            "margin": wf["margin"].values, "total": wf["total"].values, "pred_margin": pred_m, "pred_total": pred_t,
            "naive_margin": naive_m, "naive_total": naive_t, "scored": ev})
        self.last_wf = wf
=======
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
        self.state = final_state
        self.resid_margin, self.resid_total = final_res_m, final_res_t
        self.margin_sigma, self.total_sigma = sig_m, sig_t
        self.trained_through = str(g["game_date"].max().date())
        self.metrics = {
            "n_games_scored": int(ev.sum()),
            "margin_mae": float(np.abs(err_m).mean()) if len(err_m) else float("nan"),
            "total_mae": float(np.abs(err_t).mean()) if len(err_t) else float("nan"),
            "margin_mae_ratings_only": base_mae_m, "total_mae_ratings_only": base_mae_t,
            "margin_mae_naive": float(np.abs(nm).mean()) if len(nm) else float("nan"),
            "total_mae_naive": float(np.abs(nt).mean()) if len(nt) else float("nan"),
            "margin_sigma": sig_m, "total_sigma": sig_t,
            "margin_bias": float(np.mean(err_m)) if len(err_m) else float("nan"),
            "total_bias": float(np.mean(err_t)) if len(err_t) else float("nan"),
            "home_win_accuracy": float(np.mean((p_home > 0.5) == home_win)) if len(home_win) else float("nan"),
            "home_win_logloss": ll, "home_win_logloss_baseline": ll0,
            "residual_model_margin": bool(use_m), "residual_model_total": bool(use_t),
            "hca_fitted": float(final_state.hca) if final_state else None,
        }
        return self

    @staticmethod
    def _naive_baseline(wf: pd.DataFrame):
        """Season-to-date scoring average per team (prior games only) + 3 pts home court."""
        long = pd.concat([
            wf[["game_key", "game_date", "season", "home_id", "home_pts"]].rename(columns={"home_id": "t", "home_pts": "p"}).assign(side="h"),
            wf[["game_key", "game_date", "season", "away_id", "away_pts"]].rename(columns={"away_id": "t", "away_pts": "p"}).assign(side="a"),
        ]).sort_values(["t", "season", "game_date", "game_key"])
        long["avg"] = long.groupby(["t", "season"])["p"].transform(lambda s: s.shift(1).expanding().mean())
        league = float(wf[["home_pts", "away_pts"]].values.mean())
        long["avg"] = long["avg"].fillna(league)
        h = long[long["side"] == "h"].set_index("game_key")["avg"].reindex(wf["game_key"]).values
        a = long[long["side"] == "a"].set_index("game_key")["avg"].reindex(wf["game_key"]).values
        return (h - a) + 3.0, h + a

    # ── inference ────────────────────────────────────────────────────────
    def predict(self, home_id: int, away_id: int, tg: Optional[pd.DataFrame] = None, game_date=None,
                state: Optional[RatingState] = None) -> Dict[str, float]:
        st = state or self.state
        if st is None:
            raise RuntimeError("GameModel is not fitted")
        base = st.predict(int(home_id), int(away_id))
        margin, total = base["margin"], base["total"]
        ctx_h = ctx_a = None
        if tg is not None and game_date is not None and (self.resid_margin is not None or self.resid_total is not None):
            ctx_h, ctx_a = context_for(tg, int(home_id), game_date), context_for(tg, int(away_id), game_date)
            row = pd.DataFrame([{
                "rest_h": ctx_h["rest"], "rest_a": ctx_a["rest"], "b2b_h": ctx_h["b2b"], "b2b_a": ctx_a["b2b"],
                "g4_h": ctx_h["g_last4"], "g4_a": ctx_a["g_last4"], "season_type": 2,
            }])
            X = _resid_X(row)
            if self.resid_margin is not None:
                margin += float(self.resid_margin.predict(X)[0])
            if self.resid_total is not None:
                total += float(self.resid_total.predict(X)[0])
        return {
            "home_pts": (total + margin) / 2, "away_pts": (total - margin) / 2, "margin": margin, "total": total,
            "poss": base["poss"], "margin_sigma": self.margin_sigma, "total_sigma": self.total_sigma,
            "home_win_prob": float(norm.cdf(margin / self.margin_sigma)),
            "rest_home": ctx_h["rest"] if ctx_h else None, "rest_away": ctx_a["rest"] if ctx_a else None,
        }

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, directory: str = None) -> str:
        directory = directory or MODEL.MODELS_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "game_model.joblib")
<<<<<<< HEAD
        slim = copy.copy(self)
        slim.oof, slim.last_wf = None, None
        joblib.dump(slim, path)
=======
        joblib.dump(self, path)
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
        return path

    @staticmethod
    def load(directory: str = None) -> Optional["GameModel"]:
        path = os.path.join(directory or MODEL.MODELS_DIR, "game_model.joblib")
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:  # incompatible library versions etc.
            logger.warning("Could not load %s (%s); retrain with `python cli.py train`.", path, e)
            return None
