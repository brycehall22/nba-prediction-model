"""
Prediction log, grading and reporting.

Historical betting lines (especially player props) aren't available for free, so a backtest
against real prices isn't possible up front. Instead the app logs every evaluated bet with the
price it was shown at, grades it after the game from the warehouse box scores, and reports
ROI, calibration and closing-line value (CLV) as real data accumulates. After a few hundred
graded bets, `report` also suggests the market-blend weight that minimises log loss.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BETTING
from nbadata.timeutil import iso_utc, now_utc
from nbadata.warehouse import Warehouse
from oddsfeed.schema import american_to_decimal


class PredictionLog:
    def __init__(self, wh: Warehouse):
        self.wh = wh

    def log(self, bets: List[Dict], game_key: str, game_date: str, book: str,
            min_ev: Optional[float] = None) -> int:
        """Insert bets once per (game, kind, side, player, line, book): the first logged price is the 'shown' price."""
        n = 0
        created = iso_utc(now_utc())
        with self.wh.connect() as c:
            for b in bets:
                if min_ev is not None and b["ev_pct"] < min_ev:
                    continue
                pk = b.get("player_key")
                bk = b.get("book") or book          # props carry their own book
                exists = c.execute(
                    "SELECT 1 FROM predictions WHERE game_key=? AND kind=? AND side=? AND IFNULL(player_key,'')=? "
                    "AND IFNULL(line,-999)=? AND book=?",
                    (game_key, b["kind"], b["side"], pk or "", b["line"] if b["line"] is not None else -999, bk)).fetchone()
                if exists:
                    continue
                c.execute(
                    """INSERT INTO predictions(created_at, game_key, game_date, kind, side, player_key, player_name, line, price,
                           book, model_prob, market_prob, blended_prob, ev_pct, kelly, projection)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (created, game_key, game_date, b["kind"], b["side"], pk, b.get("player"), b["line"], b["price"], bk,
                     b["model_prob"], b.get("market_fair_prob"), b["blended_prob"], b["ev_pct"], b["kelly_pct"],
                     b.get("projection")))
                n += 1
        return n

    # ── grading ──────────────────────────────────────────────────────────
    @staticmethod
    def _grade_row(r: pd.Series, game: pd.Series, player_pts: Optional[float]) -> Optional[tuple]:
        """-> (result, actual) or None if it can't be graded yet."""
        hp, ap = game["home_pts"], game["away_pts"]
        if pd.isna(hp) or pd.isna(ap):
            return None
        kind, side, line = r["kind"], r["side"], r["line"]
        if kind == "moneyline":
            home_won = hp > ap
            return ("win" if (side == "HOME") == home_won else "loss", float(hp - ap))
        if kind == "spread":
            team_margin = (hp - ap) if side == "HOME" else (ap - hp)
            v = team_margin + line
            return ("push" if abs(v) < 1e-9 else ("win" if v > 0 else "loss"), float(team_margin))
        if kind == "total":
            tot = hp + ap
            v = tot - line
            if abs(v) < 1e-9:
                return ("push", float(tot))
            return ("win" if (v > 0) == (side == "OVER") else "loss", float(tot))
        if kind == "prop":
            if player_pts is None:
                return ("void", None)          # did not play -> bet voided
            v = player_pts - line
            if abs(v) < 1e-9:
                return ("push", float(player_pts))
            return ("win" if (v > 0) == (side == "OVER") else "loss", float(player_pts))
        return None

    def grade(self) -> int:
        pending = self.wh.query_df("SELECT * FROM predictions WHERE result IS NULL")
        if pending.empty:
            return 0
        games = self.wh.games_df(status="final").set_index("game_key")
        pb = self.wh.query_df("SELECT game_key, player_key, pts, played, minutes FROM player_box")
        pbi = pb.set_index(["game_key", "player_key"])
        n = 0
        with self.wh.connect() as c:
            for _, r in pending.iterrows():
                if r["game_key"] not in games.index:
                    continue
                game = games.loc[r["game_key"]]
                pts = None
                if r["kind"] == "prop":
                    key = (r["game_key"], r["player_key"])
                    if key in pbi.index and pbi.loc[key, "played"] == 1 and pbi.loc[key, "minutes"] > 0:
                        pts = float(pbi.loc[key, "pts"])
                res = self._grade_row(r, game, pts)
                if res is None:
                    continue
                closing_line, closing_price = self._closing(c, r, game)
                c.execute("UPDATE predictions SET result=?, actual=?, closing_line=?, closing_price=?, graded_at=? WHERE id=?",
                          (res[0], res[1], closing_line, closing_price, iso_utc(now_utc()), int(r["id"])))
                n += 1
        return n

    @staticmethod
    def _closing(c, r, game):
        """Last odds snapshot we have for this bet's book (our best proxy for the closing line)."""
        h, a, d = int(game["home_team_id"]), int(game["away_team_id"]), game["game_date"]
        if r["kind"] == "prop":
            row = c.execute("SELECT line, over_price, under_price FROM prop_odds WHERE game_date=? AND home_team_id=? AND away_team_id=? "
                            "AND book=? AND player_key=? AND market='player_points' ORDER BY fetched_at DESC LIMIT 1",
                            (d, h, a, r["book"], r["player_key"])).fetchone()
            if row:
                return row["line"], (row["over_price"] if r["side"] == "OVER" else row["under_price"])
        else:
            row = c.execute("SELECT * FROM game_odds WHERE game_date=? AND home_team_id=? AND away_team_id=? AND book=? "
                            "ORDER BY fetched_at DESC LIMIT 1", (d, h, a, r["book"])).fetchone()
            if row:
                if r["kind"] == "spread":
                    return ((row["spread_home"] if r["side"] == "HOME" else -row["spread_home"]) if row["spread_home"] is not None else None,
                            row["spread_home_price"] if r["side"] == "HOME" else row["spread_away_price"])
                if r["kind"] == "total":
                    return row["total"], (row["over_price"] if r["side"] == "OVER" else row["under_price"])
                if r["kind"] == "moneyline":
                    return None, (row["ml_home"] if r["side"] == "HOME" else row["ml_away"])
        return None, None

    # ── reporting ────────────────────────────────────────────────────────
    def graded(self) -> pd.DataFrame:
        return self.wh.query_df("SELECT * FROM predictions WHERE result IN ('win','loss','push')")

    @staticmethod
    def _units(r) -> float:
        if r["result"] == "win":
            return american_to_decimal(r["price"]) - 1.0
        return -1.0 if r["result"] == "loss" else 0.0

    def report(self, min_ev: float = None) -> Dict:
        df = self.graded()
        if df.empty:
            return {"n": 0}
        min_ev = BETTING.EV_THRESHOLD_PCT if min_ev is None else min_ev
        df = df.assign(units=df.apply(self._units, axis=1))
        out: Dict = {"n": int(len(df)), "by_kind": {}, "flagged": {}}

        def summarize(d: pd.DataFrame) -> Dict:
            decided = d[d["result"] != "push"]
            return {
                "bets": int(len(d)), "win_rate": float((decided["result"] == "win").mean()) if len(decided) else None,
                "units": float(d["units"].sum()), "roi_pct": float(100 * d["units"].sum() / len(d)),
                "avg_ev_pct": float(d["ev_pct"].mean()), "avg_model_prob": float(d["model_prob"].mean()),
            }
        for kind, d in df.groupby("kind"):
            out["by_kind"][kind] = summarize(d)
        flagged = df[df["ev_pct"] >= min_ev]
        if len(flagged):
            out["flagged"] = summarize(flagged)
            clv = flagged.dropna(subset=["closing_price"])
            if len(clv):
                shown = clv["price"].map(lambda p: 1 / american_to_decimal(p))
                close = clv["closing_price"].map(lambda p: 1 / american_to_decimal(p))
                out["flagged"]["clv_implied_prob_pts"] = float(100 * (close - shown).mean())   # >0: we beat the close
                out["flagged"]["clv_n"] = int(len(clv))

        # calibration: model vs market vs blends, on decided bets where we have a market probability
        cal = df[(df["result"] != "push")].dropna(subset=["market_prob"]).copy()
        if len(cal) >= 30:
            y = (cal["result"] == "win").astype(float).values
            def ll(p):
                p = np.clip(p, 1e-4, 1 - 1e-4)
                return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            pm, pk = cal["model_prob"].values, cal["market_prob"].values
            grid = {round(w, 2): ll((1 - w) * pm + w * pk) for w in np.linspace(0, 1, 11)}
            best = min(grid, key=grid.get)
            out["calibration"] = {"n": int(len(cal)), "logloss_model": grid[0.0], "logloss_market": grid[1.0],
                                  "best_market_weight": best, "logloss_best": grid[best],
                                  "enough_data": bool(len(cal) >= 300)}
        return out


    # ── detailed reporting (used by `cli.py report`) ─────────────────────
    @staticmethod
    def _summ(d: pd.DataFrame) -> Dict:
        n = len(d)
        wins, losses, pushes = int((d["result"] == "win").sum()), int((d["result"] == "loss").sum()), int((d["result"] == "push").sum())
        decided = wins + losses
        se = float(100 * d["units"].std(ddof=1) / sqrt(n)) if n >= 3 else None
        nopush = d[d["result"] != "push"]
        return {
            "bets": n, "wins": wins, "losses": losses, "pushes": pushes,
            "win_rate": wins / decided if decided else None,
            "expected_win_rate": float(nopush["blended_prob"].mean()) if decided else None,
            "units": float(d["units"].sum()), "roi_pct": float(100 * d["units"].sum() / n) if n else None,
            "roi_se_pct": se, "avg_ev_pct": float(d["ev_pct"].mean()) if n else None,
        }

    @staticmethod
    def _clv(d: pd.DataFrame) -> Dict:
        out: Dict = {}
        c = d.dropna(subset=["closing_price"])
        if len(c):
            shown = c["price"].map(lambda p: 1 / american_to_decimal(p))
            close = c["closing_price"].map(lambda p: 1 / american_to_decimal(p))
            out["price_clv_pts"], out["price_clv_n"] = float(100 * (close - shown).mean()), int(len(c))
        ln = d.dropna(subset=["closing_line", "line"])
        ln = ln[ln["kind"].isin(["spread", "total", "prop"])]
        if len(ln):
            under_like = ln["side"].isin(["UNDER"])
            moved = (ln["closing_line"] - ln["line"])           # >0 means the line rose after we bet
            fav = pd.Series(0.0, index=ln.index)
            fav[ln["side"].isin(["OVER"])] = (moved > 0).astype(float)[ln["side"].isin(["OVER"])]       # got a lower number to go over
            fav[under_like] = (moved < 0).astype(float)[under_like]                                     # got a higher number to go under
            sp = ln["kind"] == "spread"
            fav[sp] = (moved < 0).astype(float)[sp]                                                      # our side's spread got worse
            neutral = moved.abs() < 1e-9
            out["line_n"] = int(len(ln)); out["line_unchanged_pct"] = float(100 * neutral.mean())
            out["line_beat_pct"] = float(100 * fav.mean())
        return out

    def detailed_report(self, min_ev: float = None, since: Optional[str] = None, kind: Optional[str] = None) -> Dict:
        min_ev = BETTING.EV_THRESHOLD_PCT if min_ev is None else min_ev
        allp = self.wh.query_df("SELECT * FROM predictions")
        if since:
            allp = allp[allp["game_date"] >= since]
        if kind:
            allp = allp[allp["kind"] == kind]
        out: Dict = {"logged": int(len(allp)), "graded": 0, "pending": int(allp["result"].isna().sum()),
                     "void": int((allp["result"] == "void").sum()), "min_ev": min_ev, "since": since, "kind": kind,
                     "first_game": allp["game_date"].min() if len(allp) else None, "last_game": allp["game_date"].max() if len(allp) else None}
        df = allp[allp["result"].isin(["win", "loss", "push"])].copy()
        out["graded"] = int(len(df))
        if df.empty:
            return out
        df["units"] = df.apply(self._units, axis=1)
        out["overall"] = self._summ(df)
        out["overall"].update(self._clv(df))
        out["by_kind"] = {k: self._summ(d) for k, d in df.groupby("kind")}
        out["by_side"] = {f"{k} {sd}": self._summ(d) for (k, sd), d in df.groupby(["kind", "side"])}
        buckets = [("below 0%", -1e9, 0), ("0% to 3%", 0, 3), ("3% to 8%", 3, 8), ("8% and above", 8, 1e9)]
        out["by_ev_bucket"] = {name: self._summ(df[(df["ev_pct"] >= lo) & (df["ev_pct"] < hi)])
                               for name, lo, hi in buckets if ((df["ev_pct"] >= lo) & (df["ev_pct"] < hi)).any()}
        flagged = df[df["ev_pct"] >= min_ev]
        if len(flagged):
            out["flagged"] = self._summ(flagged)
            out["flagged"].update(self._clv(flagged))
        out["calibration"] = self.report(min_ev=min_ev).get("calibration") if not kind and not since else None
        recent = df.sort_values(["game_date", "id"], ascending=[False, False])
        out["recent"] = recent[["game_date", "game_key", "kind", "side", "player_name", "line", "price", "model_prob",
                                "blended_prob", "ev_pct", "result", "actual", "closing_line", "closing_price"]].to_dict("records")
        return out

    def grade_detail(self, since_iso: str) -> Dict:
        """Breakdown of what the most recent grading pass did and what is still waiting."""
        done = self.wh.query_df("SELECT result, COUNT(*) n FROM predictions WHERE graded_at >= ? GROUP BY result", [since_iso])
        pend = self.wh.query_df(
            """SELECT p.game_date, COUNT(*) n, MAX(CASE WHEN g.status='final' THEN 1 ELSE 0 END) final_known
               FROM predictions p LEFT JOIN games g ON g.game_key = p.game_key
               WHERE p.result IS NULL GROUP BY p.game_date ORDER BY p.game_date""")
        return {"done": {r["result"]: int(r["n"]) for _, r in done.iterrows()},
                "pending": [{"date": r["game_date"], "n": int(r["n"]), "final": bool(r["final_known"])} for _, r in pend.iterrows()]}
