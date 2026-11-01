"""
Odds service: refresh (with quota-aware throttling and fallbacks), snapshot storage,
and market lookups for the model/UI (DraftKings preferred, other books as consensus).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from statistics import median
from typing import Dict, List, Optional, Sequence

import pandas as pd

from config import BETTING, ODDS
from nbadata.http import ProviderError
from nbadata.teams import team_abbr
from nbadata.timeutil import iso_utc, now_utc, today_eastern, utcnow_naive
from nbadata.warehouse import Warehouse
from oddsfeed.schema import GameOdds, PropOdds, RefreshReport, devig_two_way
from oddsfeed.the_odds_api import OddsAPIError, TheOddsAPI

logger = logging.getLogger(__name__)


class OddsStore:
    def __init__(self, wh: Warehouse):
        self.wh = wh

    def save(self, games: Sequence[GameOdds], props: Sequence[PropOdds]) -> None:
        with self.wh.connect() as c:
            gcols = ["fetched_at", "game_date", "home_team_id", "away_team_id", "commence_utc", "book", "source",
                     "spread_home", "spread_home_price", "spread_away_price", "total", "over_price", "under_price",
                     "ml_home", "ml_away"]
            c.executemany(f"INSERT INTO game_odds({','.join(gcols)}) VALUES({','.join('?' * len(gcols))})",
                          [tuple(g.as_row()[k] for k in gcols) for g in games])
            pcols = ["fetched_at", "game_date", "home_team_id", "away_team_id", "commence_utc", "book", "source",
                     "market", "player_name", "player_key", "line", "over_price", "under_price"]
            c.executemany(f"INSERT INTO prop_odds({','.join(pcols)}) VALUES({','.join('?' * len(pcols))})",
                          [tuple(p.as_row()[k] for k in pcols) for p in props])

    def latest_game_odds(self, game_date: Optional[str] = None, home: Optional[int] = None,
                         away: Optional[int] = None, min_date: Optional[str] = None) -> pd.DataFrame:
        sql = """SELECT g.* FROM game_odds g
                 JOIN (SELECT game_date, home_team_id, away_team_id, book, MAX(fetched_at) mf
                       FROM game_odds GROUP BY game_date, home_team_id, away_team_id, book) m
                   ON g.game_date=m.game_date AND g.home_team_id=m.home_team_id AND g.away_team_id=m.away_team_id
                  AND g.book=m.book AND g.fetched_at=m.mf WHERE 1=1"""
        params: List = []
        if game_date:
            sql += " AND g.game_date=?"; params.append(game_date)
        if min_date:
            sql += " AND g.game_date>=?"; params.append(min_date)
        if home is not None:
            sql += " AND g.home_team_id=?"; params.append(int(home))
        if away is not None:
            sql += " AND g.away_team_id=?"; params.append(int(away))
        return self.wh.query_df(sql + " ORDER BY g.game_date, g.commence_utc, g.book", params)

    def latest_props(self, home: int, away: int, game_date: Optional[str] = None,
                     market: str = "player_points") -> pd.DataFrame:
        sql = """SELECT p.* FROM prop_odds p
                 JOIN (SELECT game_date, home_team_id, away_team_id, book, market, MAX(fetched_at) mf
                       FROM prop_odds GROUP BY game_date, home_team_id, away_team_id, book, market) m
                   ON p.game_date=m.game_date AND p.home_team_id=m.home_team_id AND p.away_team_id=m.away_team_id
                  AND p.book=m.book AND p.market=m.market AND p.fetched_at=m.mf
                 WHERE p.home_team_id=? AND p.away_team_id=? AND p.market=?"""
        params: List = [int(home), int(away), market]
        if game_date:
            sql += " AND p.game_date=?"; params.append(game_date)
        else:
            sql += " AND p.game_date>=?"; params.append((today_eastern() - timedelta(days=1)).isoformat())
        return self.wh.query_df(sql + " ORDER BY p.player_key, p.book", params)


def _opt(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


class OddsService:
    def __init__(self, wh: Warehouse, client: Optional[TheOddsAPI] = None, espn=None):
        self.wh = wh
        self.store = OddsStore(wh)
        self.client = client if client is not None else TheOddsAPI()
        self._espn = espn

    # ── refresh ──────────────────────────────────────────────────────────
    def minutes_since_refresh(self) -> Optional[float]:
        t = self.wh.get_meta_time("odds_last_refresh")
        return (utcnow_naive() - t).total_seconds() / 60.0 if t else None

    def refresh(self, force: bool = False, include_props: bool = True) -> RefreshReport:
        rep = RefreshReport()
        age = self.minutes_since_refresh()
        if not force and age is not None and age < ODDS.MIN_REFRESH_MINUTES:
            rep.ok, rep.skipped = True, True
            rep.note(f"Odds are {age:.0f} min old (refresh interval {ODDS.MIN_REFRESH_MINUTES} min); not re-pulling to save API credits.")
            return rep

        if self.client.configured:
            try:
                games, props, notes = self.client.pull(include_props=include_props)
                self.store.save(games, props)
                rep.ok, rep.source, rep.games, rep.props = True, "the_odds_api", len(games), len(props)
                rep.credits_remaining, rep.credits_used = self.client.credits_remaining, self.client.credits_used
                for n in notes:
                    rep.note(n)
                self.wh.set_meta("odds_last_refresh", iso_utc(now_utc()))
                self.wh.set_meta("odds_source", "the_odds_api")
                if rep.credits_remaining is not None:
                    self.wh.set_meta("odds_credits_remaining", rep.credits_remaining)
                return rep
            except (OddsAPIError, ProviderError) as e:
                rep.note(f"The Odds API failed: {e}")
        else:
            rep.note("No ODDS_API_KEY configured, so only ESPN's embedded game lines are available (no player props). "
                     "A free key from https://the-odds-api.com enables DraftKings game lines and props.")

        if ODDS.ENABLE_ESPN_FALLBACK:
            try:
                games = self._espn_game_lines()
                if games:
                    self.store.save(games, [])
                    rep.ok, rep.source, rep.games = True, "espn", len(games)
                    rep.note(f"Used ESPN fallback: {len(games)} game lines (no props).")
                    self.wh.set_meta("odds_last_refresh", iso_utc(now_utc()))
                    self.wh.set_meta("odds_source", "espn")
                    return rep
                rep.note("ESPN fallback returned no lines (no games scheduled, or ESPN has not posted odds yet).")
            except ProviderError as e:
                rep.note(f"ESPN fallback failed: {e}")
        return rep

    def _espn_game_lines(self) -> List[GameOdds]:
        from nbadata.providers.espn import ESPNProvider
        espn = self._espn or ESPNProvider()
        fetched = iso_utc(now_utc())
        out: List[GameOdds] = []
        for i in range(0, 3):
            for r in espn.fetch_scoreboard_odds(today_eastern() + timedelta(days=i)):
                out.append(GameOdds(fetched_at=fetched, source="espn", **{
                    k: r.get(k) for k in ("game_date", "home_team_id", "away_team_id", "commence_utc", "book",
                                          "spread_home", "spread_home_price", "total", "ml_home", "ml_away")}))
        return out

    def import_game_lines(self, games: Sequence[GameOdds]):
        self.store.save(games, [])
        self.wh.set_meta("odds_last_refresh", iso_utc(now_utc()))
        self.wh.set_meta("odds_source", "csv")

    def import_props(self, props: Sequence[PropOdds]):
        self.store.save([], props)

    # ── lookups for the model / UI ───────────────────────────────────────
    def get_game_market(self, home_team_id: int, away_team_id: int, game_date: Optional[str] = None) -> Dict:
        df = self.store.latest_game_odds(game_date, home_team_id, away_team_id,
                                         min_date=None if game_date else (today_eastern() - timedelta(days=1)).isoformat())
        if df.empty:
            return {"available": False}
        df = df[df["game_date"] == df["game_date"].min()] if not game_date else df
        target = df[df["book"] == ODDS.TARGET_BOOK]
        row = (target.iloc[0] if not target.empty else df.iloc[0])
        spread = _opt(row["spread_home"])
        total = _opt(row["total"])

        def price(v):
            v = _opt(v)
            return int(v) if v is not None else BETTING.DEFAULT_ODDS

        # Fair (no-vig) probabilities: use the other books quoting the same number, else this book alone.
        def fair_from(rows: pd.DataFrame, a: str, b: str, line_col: Optional[str], line_val):
            vals = []
            for _, r in rows.iterrows():
                if line_col and (_opt(r[line_col]) is None or line_val is None or abs(r[line_col] - line_val) > 1e-9):
                    continue
                d = devig_two_way(_opt(r[a]), _opt(r[b]))
                if d:
                    vals.append(d[0])
            return (float(median(vals)), len(vals)) if vals else (None, 0)

        fair_win, n_ml = fair_from(df, "ml_home", "ml_away", None, None)
        fair_cover, n_sp = fair_from(df, "spread_home_price", "spread_away_price", "spread_home", spread)
        fair_over, n_tot = fair_from(df, "over_price", "under_price", "total", total)
        spreads = [x for x in df["spread_home"].dropna()]
        totals = [x for x in df["total"].dropna()]
        return {
            "available": spread is not None or total is not None or _opt(row["ml_home"]) is not None,
            "book": row["book"], "source": row["source"], "n_books": int(df["book"].nunique()),
            "fetched_at": row["fetched_at"], "commence_utc": _opt(row["commence_utc"]), "game_date": row["game_date"],
            "home_spread": spread, "away_spread": (-spread if spread is not None else None),
            "home_spread_odds": price(row["spread_home_price"]), "away_spread_odds": price(row["spread_away_price"]),
            "total": total, "total_over_odds": price(row["over_price"]), "total_under_odds": price(row["under_price"]),
            "home_moneyline": _opt(row["ml_home"]), "away_moneyline": _opt(row["ml_away"]),
            "consensus_spread": float(median(spreads)) if spreads else None,
            "consensus_total": float(median(totals)) if totals else None,
            "fair": {"home_win": fair_win, "home_cover": fair_cover, "over": fair_over,
                     "n_ml": n_ml, "n_spread": n_sp, "n_total": n_tot},
            "prices_assumed": bool(_opt(row["spread_home_price"]) is None),
        }

    def get_props(self, home_team_id: int, away_team_id: int, game_date: Optional[str] = None,
                  market: str = "player_points") -> List[Dict]:
        df = self.store.latest_props(home_team_id, away_team_id, game_date, market)
        if df.empty:
            return []
        if not game_date:
            df = df[df["game_date"] == df["game_date"].min()]
        out = []
        for pkey, grp in df.groupby("player_key"):
            tgt = grp[grp["book"] == ODDS.TARGET_BOOK]
            row = tgt.iloc[0] if not tgt.empty else grp.iloc[0]
            line = float(row["line"])
            same_line = grp[(grp["line"] - line).abs() < 1e-9]
            fairs = []
            for _, r in same_line.iterrows():
                d = devig_two_way(_opt(r["over_price"]), _opt(r["under_price"]))
                if d:
                    fairs.append(d[0])
            over_p = _opt(row["over_price"]); under_p = _opt(row["under_price"])
            out.append({
                "player_key": pkey, "player": row["player_name"], "line": line,
                "over_odds": int(over_p) if over_p is not None else BETTING.DEFAULT_ODDS,
                "under_odds": int(under_p) if under_p is not None else BETTING.DEFAULT_ODDS,
                "book": row["book"], "n_books": int(grp["book"].nunique()),
                "fair_over": float(median(fairs)) if fairs else None,
                "consensus_line": float(median(grp["line"])),
                "others": [{"book": r["book"], "line": float(r["line"]), "over": _opt(r["over_price"]),
                            "under": _opt(r["under_price"])} for _, r in grp.iterrows()],
                "fetched_at": row["fetched_at"], "game_date": row["game_date"],
            })
        return out

    def slate(self, game_date: Optional[str] = None) -> pd.DataFrame:
        """One row per upcoming game (target book preferred) for the app's slate view."""
        df = self.store.latest_game_odds(game_date, min_date=None if game_date else today_eastern().isoformat())
        if df.empty:
            return df
        df = df.assign(_pref=(df["book"] != ODDS.TARGET_BOOK).astype(int))
        df = df.sort_values(["game_date", "commence_utc", "home_team_id", "_pref"]).drop_duplicates(
            ["game_date", "home_team_id", "away_team_id"], keep="first")
        df["matchup"] = df.apply(lambda r: f"{team_abbr(r['away_team_id'])} @ {team_abbr(r['home_team_id'])}", axis=1)
        return df.drop(columns="_pref").reset_index(drop=True)

    def board(self, game_date: Optional[str] = None) -> List[Dict]:
        """One summary row per upcoming game with lines: what is actually loaded (for CLI display)."""
        sl = self.slate(game_date)
        rows = []
        for _, r in sl.iterrows():
            h, a, d = int(r["home_team_id"]), int(r["away_team_id"]), r["game_date"]
            allb = self.store.latest_game_odds(d, h, a)
            props = self.store.latest_props(h, a, d)
            rows.append({
                "game_date": d, "matchup": r["matchup"], "commence_utc": r["commence_utc"], "book": r["book"],
                "spread_home": _opt(r["spread_home"]), "total": _opt(r["total"]), "ml_home": _opt(r["ml_home"]), "ml_away": _opt(r["ml_away"]),
                "n_books": int(allb["book"].nunique()), "fetched_at": r["fetched_at"],
                "props_target_book": int(props[props["book"] == ODDS.TARGET_BOOK]["player_key"].nunique()) if not props.empty else 0,
                "props_any_book": int(props["player_key"].nunique()) if not props.empty else 0,
            })
        return rows

    def status(self) -> Dict:
        age = self.minutes_since_refresh()
        return {
            "last_refresh_minutes": age, "source": self.wh.get_meta("odds_source"),
            "credits_remaining": self.wh.get_meta("odds_credits_remaining"),
            "api_key_configured": self.client.configured,
        }
