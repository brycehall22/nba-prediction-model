"""
Application service layer: wires warehouse + models + odds into the calls the CLI uses.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BETTING, MODEL, PREDICTION
from ev_engine import GameEV, evaluate_prop
from modeling.frames import build_team_games, games_frame
from modeling.game_model import GameModel
from modeling.player_model import PlayerModel, build_inference_frame, build_training_frame
from modeling.ratings import RatingEngine
from nbadata.players import match_player
from nbadata.teams import TEAMS_BY_ID, resolve_team, team_abbr, team_name
from nbadata.timeutil import today_eastern
from nbadata.warehouse import Warehouse, make_game_key
from oddsfeed.service import OddsService
from predictions import PredictionLog

logger = logging.getLogger(__name__)


class NotReady(RuntimeError):
    """Raised with an actionable message when data or models are missing."""


class Engine:
    def __init__(self, warehouse: Optional[Warehouse] = None, odds: Optional[OddsService] = None,
                 models_dir: Optional[str] = None):
        self.wh = warehouse or Warehouse()
        self.odds = odds or OddsService(self.wh)
        self.models_dir = models_dir or MODEL.MODELS_DIR
        self.game_model: Optional[GameModel] = GameModel.load(self.models_dir)
        self.player_model: Optional[PlayerModel] = PlayerModel.load(self.models_dir)
        self.log = PredictionLog(self.wh)
        self._cache_key = None
        self._tg: Optional[pd.DataFrame] = None
        self._g: Optional[pd.DataFrame] = None
        self._pb: Optional[pd.DataFrame] = None
        self._rating_engine = RatingEngine()

    # ── data frames (cached until the warehouse changes) ─────────────────
    def _frames(self):
        c = self.wh.counts()
        key = (c["games_with_box"], c["player_rows"], c["last_date"])
        if key != self._cache_key:
            # Preseason (season_type 1) is excluded: starters rest and rotations are unrepresentative.
            games = self.wh.games_df()
            games = games[games["season_type"].fillna(2) != 1]
            tb = self.wh.team_box_df()
            tb = tb[tb["game_key"].isin(games["game_key"])]
            if tb.empty:
                raise NotReady("The warehouse has no box scores yet. Run `python cli.py backfill` first.")
            self._tg = build_team_games(tb, games)
            self._g = games_frame(self._tg)
            pb = self.wh.player_box_df()
            self._pb = pb[pb["game_key"].isin(games["game_key"])].copy()
            self._pb["game_date"] = pd.to_datetime(self._pb["game_date"])
            self._cache_key = key
        return self._tg, self._g, self._pb

    # ── training ─────────────────────────────────────────────────────────
    def train(self, progress=None) -> Dict:
        tg, g, pb = self._frames()
        if progress: progress("fitting team ratings + game model (walk-forward)")
        gm = GameModel().fit(g)
        wf, _, _ = self._rating_engine.walk_forward(g)
        if progress: progress("building player features")
        frame = build_training_frame(pb, tg, wf)
        if progress: progress("fitting player model (time-ordered CV)")
        pm = PlayerModel().fit(frame)
        gm.save(self.models_dir); pm.save(self.models_dir)
        self.game_model, self.player_model = gm, pm
        return {"game": gm.metrics, "player": pm.metrics, "trained_through": gm.trained_through}

    def require_models(self):
        if self.game_model is None or self.player_model is None:
            raise NotReady("No trained models found. Run `python cli.py backfill` then `python cli.py train`.")

    # ── status ───────────────────────────────────────────────────────────
    def status(self) -> Dict:
        c = self.wh.counts()
        return {**c, "odds": self.odds.status(),
                "models": {"game": self.game_model.trained_through if self.game_model else None,
                           "player": self.player_model.trained_through if self.player_model else None}}

    # ── predictions ──────────────────────────────────────────────────────
    def predict_game(self, home_id: int, away_id: int, game_date=None, with_props: bool = True,
                     game_weight: Optional[float] = None, prop_weight: Optional[float] = None) -> Dict:
        self.require_models()
        tg, g, pb = self._frames()
        gd = pd.Timestamp(game_date or today_eastern())
        # Ratings are refreshed from all games played so far (cheap); the residual model & sigmas come from training.
        state = self._rating_engine.fit_asof(g, asof=gd)
        pred = self.game_model.predict(home_id, away_id, tg, gd, state=state)
        ds = gd.date().isoformat()
        market = self.odds.get_game_market(home_id, away_id, ds)
        if not market.get("available"):
            market = self.odds.get_game_market(home_id, away_id)      # nearest upcoming snapshot
        bets = GameEV().evaluate(pred, market, weight=game_weight)
        out = {"game_date": ds, "game_key": make_game_key(ds, away_id, home_id), "home_id": home_id, "away_id": away_id,
               "home": team_name(home_id), "away": team_name(away_id), "pred": pred, "market": market, "game_bets": bets,
               "home_players": [], "away_players": [], "prop_bets": []}
        if with_props:
            self._add_players(out, tg, pb, pred, market, gd, prop_weight)
        return out

    def _add_players(self, out: Dict, tg, pb, pred, market, gd, prop_weight: Optional[float] = None):
        inj = self.wh.injuries_df()
        # Game script: blend our expected team points with the market-implied ones when a line exists.
        hp, ap, poss = pred["home_pts"], pred["away_pts"], pred["poss"]
        if market.get("available") and market.get("home_spread") is not None and market.get("total") is not None:
            mh = (market["total"] - market["home_spread"]) / 2.0
            ma = (market["total"] + market["home_spread"]) / 2.0
            hp, ap = 0.5 * hp + 0.5 * mh, 0.5 * ap + 0.5 * ma
        rosters = {}
        for side, tid, oid, mine, theirs in (("home", out["home_id"], out["away_id"], hp, ap), ("away", out["away_id"], out["home_id"], ap, hp)):
            feats = build_inference_frame(pb, tg, tid, oid, side == "home", gd, mine, theirs, poss, injuries=inj)
            if feats.empty:
                continue
            proj = self.player_model.predict_frame(feats).set_index("player_key")
            feats = feats.set_index("player_key").join(proj)
            live = ~feats["out"]
            raw_sum = feats.loc[live, "proj_pts"].sum()
            scale = float(np.clip(mine / raw_sum, PREDICTION.MIN_PLAYER_SCALE, PREDICTION.MAX_PLAYER_SCALE)) if raw_sum > 0 else 1.0
            feats["proj_pts_adj"] = np.where(live, feats["proj_pts"] * scale, 0.0)
            feats = feats.sort_values("proj_min", ascending=False)
            rosters[side] = feats
            rows = []
            for pk, r in feats.iterrows():
                rows.append({"player_key": pk, "player": r["player_name"], "proj_pts": float(r["proj_pts_adj"]), "proj_min": float(r["proj_min"]),
                             "ewm_pts": float(r["ewm_pts"]), "last5_pts": float(r["last5_pts"]), "status": r["status"] or ("OUT" if r["out"] else ""),
                             "out": bool(r["out"]), "miss_share": float(r["miss_share"])})
            out[f"{side}_players"] = rows
        # props: match sportsbook names to roster players (restricted to the two teams in this game)
        props = self.odds.get_props(out["home_id"], out["away_id"], out["game_date"] if out["market"].get("available") else None)
        if not props:
            return
        cand = {}
        for side, feats in rosters.items():
            for pk in feats.index:
                cand[pk] = (side, feats)
        names = list(cand)
        for p in props:
            m = match_player(p["player"], names)
            if m is None:
                continue
            side, feats = cand[m]
            row = feats.loc[m]
            if bool(row["out"]):
                continue
            mu = float(row["proj_pts_adj"])
            if mu <= 0:
                continue
            probs = self.player_model.prop_probs(mu, p["line"])
            extra = {"player": p["player"], "player_key": m, "projection": mu, "proj_min": float(row["proj_min"]),
                     "team": team_abbr(out["home_id"] if side == "home" else out["away_id"]), "book": p["book"],
                     "n_books": p["n_books"], "consensus_line": p["consensus_line"], "status": row["status"],
                     "miss_share": float(row["miss_share"]), "others": p["others"]}
            out["prop_bets"].extend(evaluate_prop(probs, p["line"], p["over_odds"], p["under_odds"], p["fair_over"], weight=prop_weight, extra=extra))

    def slate(self, game_date=None, with_props: bool = True, log: bool = True,
              game_weight: Optional[float] = None, prop_weight: Optional[float] = None) -> List[Dict]:
        """Predict every game on the odds board (or warehouse schedule) for a date."""
        ds = (game_date or today_eastern())
        ds = ds.isoformat() if hasattr(ds, "isoformat") else str(ds)
        sched = self.odds.slate(ds)
        games: List[Dict] = []
        if not sched.empty:
            games = [(int(r["home_team_id"]), int(r["away_team_id"])) for _, r in sched.iterrows()]
        else:
            wg = self.wh.games_df(start=ds, end=ds)
            games = [(int(r["home_team_id"]), int(r["away_team_id"])) for _, r in wg.iterrows()]
        results = []
        for h, a in games:
            try:
                res = self.predict_game(h, a, ds, with_props=with_props, game_weight=game_weight, prop_weight=prop_weight)
            except NotReady:
                raise
            except Exception as e:
                logger.exception("prediction failed for %s @ %s", team_abbr(a), team_abbr(h))
                res = {"error": str(e), "home_id": h, "away_id": a, "game_date": ds, "home": team_name(h), "away": team_name(a)}
            results.append(res)
            if log and "error" not in res and res["market"].get("available"):
                self.log.log(res["game_bets"] + res["prop_bets"], res["game_key"], ds, res["market"].get("book", ""))
        return results
