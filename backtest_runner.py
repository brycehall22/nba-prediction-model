"""
High-level backtest entry points used by `cli.py backtest`.

  run_backtest(engine)          whole history: diagnostics, verified corrections, optionally applied + saved
  replay_week(engine, end)      one week replayed day by day as if live: baseline model vs corrected model

Both use only information that was available at the time: walk-forward ratings, models fitted before the
period being scored, and (for the week replay) corrections that were themselves derived from earlier data only.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from config import BETTING
from modeling.backtest import (GRID, Derived, apply_platt, derive_tuning, logloss)
from modeling.player_model import roster_for_team, rotation_from_games
from modeling.ratings import RatingEngine
from modeling.tuning import Tuning
from predictions import PredictionLog
from oddsfeed.schema import american_to_decimal
from services import Engine


# ── full-history backtest ────────────────────────────────────────────────────
def run_backtest(engine: Engine, apply: bool = True, tune_rating: bool = True, seasons: Optional[int] = None,
                 grid: Optional[Dict] = None, progress=None) -> Dict:
    tg, g, pb = engine._frames()
    if seasons:
        keep = list(dict.fromkeys(g.sort_values("game_date")["season"]))[-seasons:]
        g, tg, pb = g[g["season"].isin(keep)], tg[tg["season"].isin(keep)], pb[pb["season"].isin(keep)]
    d = derive_tuning(g, tg, pb, tune_rating=tune_rating, grid=grid or GRID, progress=progress)
    adopted = [f for f in d.findings if f.adopted]
    result = {"derived": d, "adopted": adopted, "applied": False, "saved_to": None, "previous": engine.tuning.describe(),
              "n_games": len(g), "seasons": sorted(g["season"].unique())}
    if apply and adopted:
        d.tuning.save(engine.models_dir)
        engine.tuning = d.tuning
        engine._rating_engine = RatingEngine(**d.tuning.rating)
        engine._state_cache.clear()
        if seasons is None:
            d.gm1.save(engine.models_dir)
            d.pm1.save(engine.models_dir)
            engine.game_model, engine.player_model = d.gm1, d.pm1
        else:
            if progress:
                progress("retraining on the full history with the new corrections")
            engine.train()
        result["applied"], result["saved_to"] = True, engine.models_dir
    return result


# ── week replay ──────────────────────────────────────────────────────────────
def _scratch(engine: Engine, gm, pm, tuning: Tuning) -> Engine:
    """An Engine that shares data with `engine` but uses in-memory models; nothing is saved or logged."""
    e = Engine(engine.wh, engine.odds, models_dir=engine.models_dir)
    e.game_model, e.player_model, e.tuning = gm, pm, tuning
    e._rating_engine = RatingEngine(**tuning.rating)
    e._cache_key, e._tg, e._g, e._pb = engine._cache_key, engine._tg, engine._g, engine._pb
    e._state_cache = {}
    return e


def _actual_absentees(pb: pd.DataFrame, game_key: str, home: int, away: int, gd: pd.Timestamp) -> pd.DataFrame:
    """
    Replay assumption: the injury report matched who actually sat out. Rotation players (last 10 games) who did not
    play are treated as 'Out'. This is optimistic (late scratches are included) and is stated in the report.
    """
    rows = []
    played = pb[(pb["game_key"] == game_key) & (pb["played"] == 1)]
    for team in (home, away):
        games, _, _ = roster_for_team(pb, team, gd, None)
        rot = rotation_from_games(games)
        here = set(played[played["team_id"] == team]["player_key"])
        rows += [{"team_id": team, "player_key": p, "status": "Out"} for p in rot if p not in here]
    return pd.DataFrame(rows, columns=["team_id", "player_key", "status"])


def _naive(tg: pd.DataFrame, home: int, away: int, gd: pd.Timestamp, season: str, league: float):
    """The old app's approach: each team's season-to-date points average + 3 home-court points."""
    def avg(t):
        d = tg[(tg["team_id"] == t) & (tg["game_date"] < gd) & (tg["season"] == season)]
        return d["pts"].mean() if len(d) else league
    h, a = avg(home), avg(away)
    return h - a + 3.0, h + a


def _units(result: str, price: int) -> float:
    return american_to_decimal(price) - 1.0 if result == "win" else (-1.0 if result == "loss" else 0.0)


def _grade_variant(res: Dict, game_row: pd.Series, pts_lookup: Dict) -> List[Dict]:
    out = []
    for b in res["game_bets"] + res["prop_bets"]:
        if not b.get("is_pick"):          # score only the side the model actually chose for each line
            continue
        pts = pts_lookup.get((res["game_key"], b.get("player_key"))) if b["kind"] == "prop" else None
        r = PredictionLog._grade_row(pd.Series({"kind": b["kind"], "side": b["side"], "line": b["line"]}), game_row, pts)
        if r is None or r[0] == "void":
            continue
        out.append({"kind": b["kind"], "side": b["side"], "line": b["line"], "price": b["price"], "ev_pct": b["ev_pct"],
                    "result": r[0], "units": _units(r[0], b["price"]), "player": b.get("player"), "game_key": res["game_key"],
                    "flagged": b["ev_pct"] >= BETTING.EV_THRESHOLD_PCT})
    return out


def _roi(rows: List[Dict], flagged_only: bool) -> Dict:
    r = [x for x in rows if (x["flagged"] or not flagged_only)]
    if not r:
        return {"bets": 0, "units": 0.0, "roi_pct": None, "wins": 0, "losses": 0}
    u = sum(x["units"] for x in r)
    return {"bets": len(r), "units": u, "roi_pct": 100 * u / len(r), "wins": sum(x["result"] == "win" for x in r), "losses": sum(x["result"] == "loss" for x in r)}


def replay_week(engine: Engine, end=None, tune_rating: bool = True, grid: Optional[Dict] = None, progress=None) -> Dict:
    tg, g, pb = engine._frames()
    end = pd.Timestamp(end) if end else g["game_date"].max()
    start = end - pd.Timedelta(days=6)
    week = g[(g["game_date"] >= start) & (g["game_date"] <= end)].sort_values(["game_date", "game_key"])
    if week.empty:
        return {"error": f"No completed games between {start.date()} and {end.date()} in the warehouse (try `python cli.py update`, or another --ending date)."}
    pre_g, pre_tg, pre_pb = g[g["game_date"] < start], tg[tg["game_date"] < start], pb[pb["game_date"] < start]
    if len(pre_g) < 300:
        return {"error": f"Only {len(pre_g)} games exist before {start.date()}; the replay needs about 300+ to fit models honestly. Run `python cli.py backfill`."}
    d = derive_tuning(pre_g, pre_tg, pre_pb, tune_rating=tune_rating, grid=grid or GRID, progress=progress)
    base, corr = _scratch(engine, d.gm0, d.pm0, Tuning()), _scratch(engine, d.gm1, d.pm1, d.tuning)
    league = float(pre_tg["pts"].mean())
    if progress:
        progress(f"replaying {len(week)} games day by day (each uses only data from before that day)")

    game_rows, player_rows, bets_b, bets_c = [], [], [], []
    pts_lookup = {(r.game_key, r.player_key): float(r.pts) for r in pb[pb["played"] == 1].itertuples()}
    for _, row in week.iterrows():
        h, a, gd = int(row["home_id"]), int(row["away_id"]), row["game_date"]
        ds = gd.date().isoformat()
        inj = _actual_absentees(pb, row["game_key"], h, a, gd)
        rb, rc = base.predict_game(h, a, ds, injuries=inj), corr.predict_game(h, a, ds, injuries=inj)
        nm, nt = _naive(tg, h, a, gd, row["season"], league)
        am, at = float(row["margin"]), float(row["total"])
        game_rows.append({
            "date": ds, "game_key": row["game_key"], "matchup": "@".join(row["game_key"].split("_")[1:3]),
            "actual_margin": am, "actual_total": at, "naive_margin": nm, "naive_total": nt,
            "base_margin": rb["pred"]["margin"], "base_total": rb["pred"]["total"], "base_win_p": rb["pred"]["home_win_prob"],
            "corr_margin": rc["pred"]["margin"], "corr_total": rc["pred"]["total"], "corr_win_p": rc["pred"]["home_win_prob"],
            "has_market": bool(rb["market"].get("available")), "n_props": len(rb["prop_bets"]) // 2,
        })
        gp = pd.Series({"home_pts": row["home_pts"], "away_pts": row["away_pts"]})
        bets_b += _grade_variant(rb, gp, pts_lookup)
        bets_c += _grade_variant(rc, gp, pts_lookup)
        actual = pb[(pb["game_key"] == row["game_key"]) & (pb["played"] == 1)].set_index("player_key")
        cp = {p["player_key"]: p for p in rc["home_players"] + rc["away_players"]}
        for p in rb["home_players"] + rb["away_players"]:
            k = p["player_key"]
            if p["out"] or k not in actual.index or p["proj_min"] < 12:
                continue
            player_rows.append({"date": ds, "game": game_rows[-1]["matchup"], "player": p["player"], "player_key": k,
                                "proj_base": p["proj_pts"], "proj_corr": cp[k]["proj_pts"] if k in cp else p["proj_pts"], "ewm": p["ewm_pts"],
                                "proj_min": p["proj_min"], "actual_pts": float(actual.loc[k, "pts"]), "actual_min": float(actual.loc[k, "minutes"]),
                                "miss_share": p["miss_share"], "abs_margin": abs(am)})

    G, P = pd.DataFrame(game_rows), pd.DataFrame(player_rows)
    out: Dict = {"start": start.date().isoformat(), "end": end.date().isoformat(), "games": G, "players": P, "derived": d,
                 "n_pre_games": len(pre_g), "bets_base": bets_b, "bets_corr": bets_c}
    out["game_summary"] = _game_summary(G)
    out["player_summary"] = _player_summary(P, d)
    out["betting"] = {"base_flagged": _roi(bets_b, True), "corr_flagged": _roi(bets_c, True),
                      "base_all": _roi(bets_b, False), "corr_all": _roi(bets_c, False), "games_with_lines": int(G["has_market"].sum()) if len(G) else 0}
    out["misses"] = _misses(P)
    return out


def _game_summary(G: pd.DataFrame) -> Dict:
    if G.empty:
        return {}
    s = {"n": len(G)}
    for name, m, t in (("naive", "naive_margin", "naive_total"), ("base", "base_margin", "base_total"), ("corr", "corr_margin", "corr_total")):
        s[name] = {"margin_mae": float((G["actual_margin"] - G[m]).abs().mean()), "total_mae": float((G["actual_total"] - G[t]).abs().mean()),
                   "margin_bias": float((G["actual_margin"] - G[m]).mean()), "total_bias": float((G["actual_total"] - G[t]).mean()),
                   "winner_correct": float((np.sign(G[m]) == np.sign(G["actual_margin"])).mean())}
    y = (G["actual_margin"] > 0).astype(float).values
    s["base"]["win_logloss"], s["corr"]["win_logloss"] = logloss(G["base_win_p"].values, y), logloss(G["corr_win_p"].values, y)
    return s


def _player_summary(P: pd.DataFrame, d: Derived) -> Dict:
    if P.empty:
        return {}
    line = np.floor(P["ewm"].values) + 0.5
    over = P["actual_pts"].values > line

    def block(col):
        return {"mae": float((P["actual_pts"] - P[col]).abs().mean()), "bias": float((P["actual_pts"] - P[col]).mean()),
                "form_line_accuracy": float(((P[col].values > line) == over).mean())}
    pm0, pm1 = d.pm0, d.pm1
    res = {"n": len(P), "mae_form": float((P["actual_pts"] - P["ewm"]).abs().mean()), "base": block("proj_base"), "corr": block("proj_corr")}
    for name, col, pm, cal in (("base", "proj_base", pm0, None), ("corr", "proj_corr", pm1, d.tuning.prop_cal)):
        ps, ys = [], []
        for off in (-4, -2, 0, 2, 4):
            ln = np.floor(P[col].values + off) + 0.5
            p = 1.0 - nbinom.cdf(np.floor(ln), pm.nb_r, pm.nb_r / (pm.nb_r + np.maximum(P[col].values, 0.05)))
            ps.append(apply_platt(p, *cal) if cal else p)
            ys.append((P["actual_pts"].values > ln).astype(float))
        res[name]["calibration_logloss"] = logloss(np.concatenate(ps), np.concatenate(ys))
    return res


def _misses(P: pd.DataFrame, top: int = 10) -> Dict:
    if P.empty:
        return {"top": [], "tag_counts": {}}
    P = P.assign(err=P["actual_pts"] - P["proj_corr"])
    P = P.assign(abs_err=P["err"].abs())

    def tag(r):
        tags = []
        if abs(r["actual_min"] - r["proj_min"]) >= 8:
            tags.append("minutes surprise")
        if r["miss_share"] >= 0.2:
            tags.append("teammates out")
        if r["abs_margin"] >= 20:
            tags.append("blowout")
        return tags or ["scoring variance"]
    big = P[P["abs_err"] >= 8].copy()
    counts: Dict[str, int] = {}
    for _, r in big.iterrows():
        for t in tag(r):
            counts[t] = counts.get(t, 0) + 1
    top_rows = P.sort_values("abs_err", ascending=False).head(top)
    return {"top": [{**r.to_dict(), "tags": tag(r)} for _, r in top_rows.iterrows()], "tag_counts": counts, "n_big": int(len(big)), "n": int(len(P))}
