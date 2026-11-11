"""
CSV import: the escape hatch. Lets you paste lines from ANY source (or a spreadsheet
you maintain by hand) so the app keeps working if every automated source is down.

Game-lines CSV (canonical):
    game_date, away_team, home_team, book, spread_home, spread_home_price, spread_away_price,
    total, over_price, under_price, ml_home, ml_away
Player-props CSV (canonical):
    game_date, away_team, home_team, book, player, line, over_price, under_price   [market optional]

The legacy `draftkings_player_points_props.csv` format (Game, Player, Points Line, Over Odds, Under Odds)
is also accepted.
"""

from __future__ import annotations

import re
from datetime import date
from typing import List, Tuple

import pandas as pd

from nbadata.players import normalize_name
from nbadata.teams import resolve_team
from nbadata.timeutil import iso_utc, now_utc, today_eastern
from oddsfeed.schema import GameOdds, PropOdds


def _num(x, as_int=False):
    if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() in ("", "nan", "None"):
        return None
    s = str(x).replace("\u2212", "-").replace("+", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return int(round(v)) if as_int else v


def import_game_lines(path: str, default_book: str = "manual") -> List[GameOdds]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    fetched = iso_utc(now_utc())
    out = []
    for _, r in df.iterrows():
        home, away = resolve_team(r.get("home_team")), resolve_team(r.get("away_team"))
        if not (home and away):
            continue
        out.append(GameOdds(
            game_date=str(r.get("game_date") or today_eastern().isoformat())[:10],
            home_team_id=home.id, away_team_id=away.id, commence_utc=None,
            book=str(r.get("book") or default_book).lower(), fetched_at=fetched, source="csv",
            spread_home=_num(r.get("spread_home")), spread_home_price=_num(r.get("spread_home_price"), True),
            spread_away_price=_num(r.get("spread_away_price"), True), total=_num(r.get("total")),
            over_price=_num(r.get("over_price"), True), under_price=_num(r.get("under_price"), True),
            ml_home=_num(r.get("ml_home"), True), ml_away=_num(r.get("ml_away"), True),
        ))
    return out


def import_props(path: str, default_book: str = "manual") -> List[PropOdds]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    fetched = iso_utc(now_utc())
    legacy = "points line" in df.columns
    out = []
    for _, r in df.iterrows():
        if legacy:
            m = re.match(r"^\s*(.+?)\s+(?:at|@)\s+(.+?)\s*$", str(r.get("game", "")))
            away, home = (resolve_team(m.group(1)), resolve_team(m.group(2))) if m else (None, None)
            player, line, over, under = r.get("player"), _num(r.get("points line")), _num(r.get("over odds"), True), _num(r.get("under odds"), True)
            gdate = today_eastern().isoformat()
        else:
            home, away = resolve_team(r.get("home_team")), resolve_team(r.get("away_team"))
            player, line = r.get("player"), _num(r.get("line"))
            over, under = _num(r.get("over_price"), True), _num(r.get("under_price"), True)
            gdate = str(r.get("game_date") or today_eastern().isoformat())[:10]
        if not (home and away and player and line is not None):
            continue
        out.append(PropOdds(
            game_date=gdate, home_team_id=home.id, away_team_id=away.id, commence_utc=None,
            book=str(r.get("book") or default_book).lower(), fetched_at=fetched, source="csv",
            market=str(r.get("market") or "player_points"), player_name=str(player),
            player_key=normalize_name(player), line=line, over_price=over, under_price=under,
        ))
    return out
