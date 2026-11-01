"""
NBA.com static-CDN provider (no key required). Used as the fallback to ESPN.

  Schedule : https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json   (current season)
  Box score: https://cdn.nba.com/static/json/liveData/boxscore/boxscore_<gameId>.json

Unlike stats.nba.com (which throttles / times out aggressively, the cause of the old
timeouts), this is a plain CDN. Parsing is tolerant; `health_check` reports what broke.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, List, Optional

from nbadata.http import HttpClient, ProviderError
from nbadata.players import normalize_name
from nbadata.providers.base import BoxScore, GameDataProvider, GameStub, PlayerLine, build_stub
from nbadata.timeutil import eastern_date, parse_iso_utc, today_eastern

logger = logging.getLogger(__name__)

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


def _int(x, default=0):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def parse_iso_minutes(x) -> float:
    """'PT34M12.00S' -> 34.2 ; '34:12' -> 34.2 ; '' -> 0."""
    s = str(x or "")
    m = re.match(r"^PT(?:(\d+)M)?(?:([\d.]+)S)?$", s)
    if m:
        return float(m.group(1) or 0) + float(m.group(2) or 0) / 60.0
    if ":" in s:
        a, b = s.split(":", 1)
        return _int(a) + _int(b) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _season_type_from_id(gid: str) -> Optional[int]:
    return {"001": 1, "002": 2, "003": 3, "004": 3, "005": 5}.get(str(gid)[:3])


class NBACDNProvider(GameDataProvider):
    name = "nba_cdn"

    BROWSER_HEADERS = {
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Accept": "application/json, text/plain, */*",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }

    def __init__(self, http: Optional[HttpClient] = None):
        # cdn.nba.com answers 403 "Access Denied" to requests that don't look like they came from nba.com.
        self.http = http or HttpClient(headers=self.BROWSER_HEADERS)
        self._schedule: Optional[List[GameStub]] = None

    def _load_schedule(self) -> List[GameStub]:
        if self._schedule is None:
            self._schedule = self.parse_schedule(self.http.get_json(SCHEDULE_URL))
        return self._schedule

    @staticmethod
    def parse_schedule(data: dict) -> List[GameStub]:
        out: List[GameStub] = []
        for gd in ((data or {}).get("leagueSchedule") or {}).get("gameDates", []) or []:
            for g in gd.get("games", []) or []:
                try:
                    tip = parse_iso_utc(g.get("gameDateTimeUTC"))
                    if tip is None:
                        continue
                    status_code = _int(g.get("gameStatus"), 1)
                    status = {1: "scheduled", 2: "in_progress", 3: "final"}.get(status_code, "scheduled")
                    home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
                    stub = build_stub(
                        game_date=eastern_date(tip), season_type=_season_type_from_id(g.get("gameId", "")),
                        home_text=home.get("teamTricode") or home.get("teamName"),
                        away_text=away.get("teamTricode") or away.get("teamName"),
                        tipoff_utc=tip.strftime("%Y-%m-%dT%H:%M:%SZ"), status=status,
                        home_pts=_int(home.get("score"), None) if status == "final" else None,
                        away_pts=_int(away.get("score"), None) if status == "final" else None,
                        nba_id=str(g.get("gameId")), source="nba_cdn",
                    )
                    if stub:
                        out.append(stub)
                except Exception as e:
                    logger.debug("CDN schedule game skipped: %s", e)
        return out

    def list_games(self, start: date, end: date) -> List[GameStub]:
        s, e = start.isoformat(), end.isoformat()
        return [g for g in self._load_schedule() if s <= g.game_date <= e]

    def supports(self, stub: GameStub) -> bool:
        return bool(stub.nba_id)

    def fetch_box(self, stub: GameStub) -> Optional[BoxScore]:
        if not stub.nba_id:
            return None
        data = self.http.get_json(BOX_URL.format(gid=stub.nba_id))
        return self.parse_boxscore(data, stub)

    @staticmethod
    def parse_boxscore(data: dict, stub: GameStub) -> Optional[BoxScore]:
        game = (data or {}).get("game") or {}
        if not game:
            return None
        players: List[PlayerLine] = []
        scores: Dict[int, int] = {}
        for side in ("homeTeam", "awayTeam"):
            tb = game.get(side) or {}
            tid = stub.home_team_id if side == "homeTeam" else stub.away_team_id
            if tb.get("score") is not None:
                scores[tid] = _int(tb.get("score"))
            for p in tb.get("players") or []:
                st = p.get("statistics") or {}
                name = p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
                if not name:
                    continue
                mins = parse_iso_minutes(st.get("minutes"))
                inactive = str(p.get("status", "")).upper() == "INACTIVE" or (p.get("played") in ("0", 0) and mins == 0)
                players.append(PlayerLine(
                    team_id=tid, player_key=normalize_name(name), player_name=name,
                    player_id=str(p.get("personId")) if p.get("personId") is not None else None,
                    starter=1 if str(p.get("starter")) in ("1", "True", "true") else 0,
                    played=1 if mins > 0 else 0, minutes=mins,
                    pts=_int(st.get("points")), reb=_int(st.get("reboundsTotal")), ast=_int(st.get("assists")),
                    fgm=_int(st.get("fieldGoalsMade")), fga=_int(st.get("fieldGoalsAttempted")),
                    fg3m=_int(st.get("threePointersMade")), fg3a=_int(st.get("threePointersAttempted")),
                    ftm=_int(st.get("freeThrowsMade")), fta=_int(st.get("freeThrowsAttempted")),
                    oreb=_int(st.get("reboundsOffensive")), dreb=_int(st.get("reboundsDefensive")),
                    tov=_int(st.get("turnovers")), stl=_int(st.get("steals")), blk=_int(st.get("blocks")),
                    pf=_int(st.get("foulsPersonal")), plus_minus=_int(st.get("plusMinusPoints"), None),
                    dnp_reason=(p.get("notPlayingReason") or p.get("notPlayingDescription") or None) if inactive or mins == 0 else None,
                ))
        if not players:
            return None
        hp = scores.get(stub.home_team_id, stub.home_pts)
        ap = scores.get(stub.away_team_id, stub.away_pts)
        stub = GameStub(**{**stub.__dict__, "home_pts": hp, "away_pts": ap, "status": "final"})
        return BoxScore(game=stub, players=players)

    def health_check(self) -> Dict:
        res = {"provider": self.name, "ok": False, "steps": []}

        def step(name, ok, detail=""):
            res["steps"].append({"step": name, "ok": ok, "detail": detail})
            return ok

        try:
            sched = self._load_schedule()
            if not step("schedule", bool(sched), f"{len(sched)} games in current-season schedule"):
                return res
            finals = [g for g in sched if g.status == "final"]
            if not finals:
                res["ok"] = True
                res["note"] = "schedule reachable, but no finished games yet to verify box-score parsing"
                return res
            box = self.fetch_box(finals[-1])
            if not step("box score", box is not None, f"{len(box.players) if box else 0} player lines for {finals[-1].game_key}"):
                return res
            problems = box.validate()
            step("box score sanity", not problems, "; ".join(problems) or "player points match team scores")
            res["ok"] = not problems
        except ProviderError as e:
            step("request", False, str(e))
        return res
