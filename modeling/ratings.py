"""
Opponent-adjusted team ratings: a walk-forward, time-decayed ridge regression.

For every game, each team's scoring efficiency (points per 100 possessions) is modelled as

    eff(team scoring vs opp) = mu  +/- hca  + off[team] + def[opp]

and the number of possessions as  poss = mu_p + pace[home] + pace[away].
`def` is "points allowed relative to average" (higher = worse defence).

Why this instead of season scoring averages (what the old app used): a team that scored 120
against bad defences is not the same as one that scored 120 against good ones, and averages
ignore pace, home court and schedule. Weights decay exponentially with age so form matters,
and the ridge penalty pulls small samples toward a prior (last season's ratings, shrunk).

Walk-forward means the prediction for a game on date D is made with ONLY games before D, so
backtests are leakage-free. (tests/test_modeling.py verifies this explicitly.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import MODEL
from nbadata.teams import all_team_ids


@dataclass
class RatingState:
    mu: float = 115.0                                        # league avg points per 100 possessions
    hca: float = 2.5                                         # home edge in efficiency points
    off: Dict[int, float] = field(default_factory=dict)
    dfn: Dict[int, float] = field(default_factory=dict)
    mu_p: float = 99.0
    pace: Dict[int, float] = field(default_factory=dict)
    asof: Optional[str] = None
    n_games: int = 0

    def predict(self, home_id: int, away_id: int) -> Dict[str, float]:
        poss = self.mu_p + self.pace.get(home_id, 0.0) + self.pace.get(away_id, 0.0)
        eff_h = self.mu + self.hca + self.off.get(home_id, 0.0) + self.dfn.get(away_id, 0.0)
        eff_a = self.mu - self.hca + self.off.get(away_id, 0.0) + self.dfn.get(home_id, 0.0)
        hp, ap = eff_h * poss / 100.0, eff_a * poss / 100.0
        return {"home_pts": hp, "away_pts": ap, "margin": hp - ap, "total": hp + ap, "poss": poss}

    def net_rating(self, team_id: int) -> float:
        """Off minus def (higher = better)."""
        return self.off.get(team_id, 0.0) - self.dfn.get(team_id, 0.0)


class RatingEngine:
    def __init__(self, half_life: float = None, ridge: float = None, pace_ridge: float = None, carry: float = None):
        self.half_life = half_life or MODEL.RATING_HALF_LIFE_DAYS
        self.ridge = MODEL.RATING_RIDGE if ridge is None else ridge
        self.pace_ridge = MODEL.PACE_RIDGE if pace_ridge is None else pace_ridge
        self.carry = MODEL.PRIOR_SEASON_CARRY if carry is None else carry
        self.teams = all_team_ids()
        self.tix = {t: i for i, t in enumerate(self.teams)}
        self.T = len(self.teams)

    # ── design matrices ──────────────────────────────────────────────────
    def _prepare(self, g: pd.DataFrame):
        """g: games_frame(). Builds efficiency (2 rows/game) and possession (1 row/game) systems."""
        G, T = len(g), self.T
        hi = g["home_id"].map(self.tix).values
        ai = g["away_id"].map(self.tix).values
        poss = g["poss"].values.astype(float)
        eff_h = 100.0 * g["home_pts"].values / poss
        eff_a = 100.0 * g["away_pts"].values / poss

        X = np.zeros((2 * G, 2 + 2 * T))
        r = np.arange(G)
        X[r, 0] = 1; X[r, 1] = 1; X[r, 2 + hi] = 1; X[r, 2 + T + ai] = 1          # home scoring
        X[G + r, 0] = 1; X[G + r, 1] = -1; X[G + r, 2 + ai] = 1; X[G + r, 2 + T + hi] = 1   # away scoring
        y = np.concatenate([eff_h, eff_a])

        P = np.zeros((G, 1 + T))
        P[r, 0] = 1; P[r, 1 + hi] += 1; P[r, 1 + ai] += 1
        dates = g["game_date"].values.astype("datetime64[D]").astype(np.int64)
        return X, y, P, poss, np.concatenate([dates, dates]), dates

    def _solve(self, X, y, P, poss, d2, d1, asof_day: int, prior: Optional[np.ndarray], prior_p: Optional[np.ndarray]):
        T = self.T
        m2 = d2 < asof_day
        m1 = d1 < asof_day
        if m1.sum() < 10:
            return None
        w2 = 0.5 ** ((asof_day - d2[m2]) / self.half_life)
        w1 = 0.5 ** ((asof_day - d1[m1]) / self.half_life)
        Xm, ym = X[m2], y[m2]
        lam = np.zeros(2 + 2 * T); lam[2:] = self.ridge
        A = (Xm.T * w2) @ Xm + np.diag(lam)
        b = (Xm.T * w2) @ ym + lam * (prior if prior is not None else 0.0)
        # tiny jitter on mu/hca keeps A well-conditioned when data is thin
        A[0, 0] += 1e-6; A[1, 1] += 1e-6
        beta = np.linalg.solve(A, b)

        Pm, pm = P[m1], poss[m1]
        lam_p = np.zeros(1 + T); lam_p[1:] = self.pace_ridge
        Ap = (Pm.T * w1) @ Pm + np.diag(lam_p)
        bp = (Pm.T * w1) @ pm + lam_p * (prior_p if prior_p is not None else 0.0)
        Ap[0, 0] += 1e-6
        bet_p = np.linalg.solve(Ap, bp)
        return beta, bet_p, int(m1.sum())

    def _state(self, beta, bet_p, n, asof_day) -> RatingState:
        T = self.T
        off = {t: float(beta[2 + i]) for t, i in self.tix.items()}
        dfn = {t: float(beta[2 + T + i]) for t, i in self.tix.items()}
        # centre ratings so league average is 0 (the ridge nearly does this already)
        mo, md = np.mean(list(off.values())), np.mean(list(dfn.values()))
        off = {k: v - mo for k, v in off.items()}
        dfn = {k: v - md for k, v in dfn.items()}
        mu = float(beta[0] + mo + md)
        pace = {t: float(bet_p[1 + i]) for t, i in self.tix.items()}
        mp = np.mean(list(pace.values()))
        pace = {k: v - mp for k, v in pace.items()}
        return RatingState(mu=mu, hca=float(beta[1]), off=off, dfn=dfn, mu_p=float(bet_p[0] + 2 * mp), pace=pace,
                           asof=str(np.datetime64(int(asof_day), "D")), n_games=n)

    def _prior_vectors(self, state: Optional[RatingState]):
        if state is None:
            return None, None
        T = self.T
        pr = np.zeros(2 + 2 * T)
        for t, i in self.tix.items():
            pr[2 + i] = self.carry * state.off.get(t, 0.0)
            pr[2 + T + i] = self.carry * state.dfn.get(t, 0.0)
        pp = np.zeros(1 + T)
        for t, i in self.tix.items():
            pp[1 + i] = self.carry * state.pace.get(t, 0.0)
        return pr, pp

    # ── public API ───────────────────────────────────────────────────────
    def walk_forward(self, g: pd.DataFrame):
        """
        As-of predictions for every game in `g` (a games_frame), plus the final state.
        Returns (preds_df indexed like g, final_state, season_end_states).
        """
        g = g.sort_values(["game_date", "game_key"]).reset_index(drop=True)
        X, y, P, poss, d2, d1 = self._prepare(g)
        out = pd.DataFrame(index=g.index, columns=["r_home_pts", "r_away_pts", "r_margin", "r_total", "r_poss", "r_hca"], dtype=float)
        dates = d1
        seasons = g["season"].values
        season_order = list(dict.fromkeys(seasons))
        prior_state: Optional[RatingState] = None
        season_end: Dict[str, RatingState] = {}
        for season in season_order:
            idx = np.where(seasons == season)[0]
            prior, prior_p = self._prior_vectors(prior_state)
            last_state = None
            for day in np.unique(dates[idx]):
                sol = self._solve(X, y, P, poss, d2, d1, int(day), prior, prior_p)
                rows = idx[dates[idx] == day]
                if sol is None:
                    # no history yet: fall back to the prior/league average
                    st = prior_state if prior_state is not None else RatingState()
                else:
                    st = self._state(*sol, int(day))
                    last_state = st
                for i in rows:
                    p = st.predict(int(g.at[i, "home_id"]), int(g.at[i, "away_id"]))
                    out.loc[i, :] = [p["home_pts"], p["away_pts"], p["margin"], p["total"], p["poss"], st.hca]
            # state after the final game of the season becomes next season's prior
            end_day = int(dates[idx].max()) + 1
            sol = self._solve(X, y, P, poss, d2, d1, end_day, prior, prior_p)
            end_state = self._state(*sol, end_day) if sol else (last_state or prior_state)
            if end_state is not None:
                season_end[season] = end_state
                prior_state = end_state
        final = self.fit_asof(g, None, prior_state=None)
        return pd.concat([g, out], axis=1), final, season_end

    def fit_asof(self, g: pd.DataFrame, asof: Optional[pd.Timestamp] = None, prior_state: Optional[RatingState] = None,
                 prior_from_previous_season: bool = True) -> RatingState:
        """Ratings using all games strictly before `asof` (default: after the last game)."""
        g = g.sort_values(["game_date", "game_key"]).reset_index(drop=True)
        if g.empty:
            return RatingState()
        X, y, P, poss, d2, d1 = self._prepare(g)
        last_day = int(d1.max()) + 1
        asof_day = last_day if asof is None else int(np.datetime64(pd.Timestamp(asof).date(), "D").astype(np.int64))
        if prior_state is None and prior_from_previous_season:
            prior_state = self._previous_season_end(g, asof_day, X, y, P, poss, d2, d1)
        prior, prior_p = self._prior_vectors(prior_state)
        sol = self._solve(X, y, P, poss, d2, d1, asof_day, prior, prior_p)
        if sol is None:
            return prior_state or RatingState()
        return self._state(*sol, asof_day)

    def _previous_season_end(self, g, asof_day, X, y, P, poss, d2, d1) -> Optional[RatingState]:
        """Fit the state at the end of the season before the one containing `asof_day`."""
        seasons = list(dict.fromkeys(g["season"].values))
        cur_idx = np.where(d1 < asof_day)[0]
        if len(cur_idx) == 0:
            return None
        cur_season = g["season"].values[cur_idx.max()]
        k = seasons.index(cur_season)
        if k == 0:
            return None
        prev_idx = np.where(g["season"].values == seasons[k - 1])[0]
        end_day = int(d1[prev_idx].max()) + 1
        sol = self._solve(X, y, P, poss, d2, d1, end_day, None, None)
        return self._state(*sol, end_day) if sol else None
