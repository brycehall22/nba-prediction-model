"""
ESPN public site API provider (no key required).

  Scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
  Box score : https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>
  Injuries  : https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

These are undocumented endpoints, so parsing is deliberately schema-tolerant (stat
columns are located by label rather than position) and `health_check` reports exactly
which part of the response shape stopped matching if ESPN ever changes it.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Dict, List, Optional

from nbadata.http import HttpClient, ProviderError
from nbadata.players import normalize_name
from nbadata.providers.base import BoxScore, GameDataProvider, GameStub, PlayerLine, build_stub
from nbadata.teams import resolve_team
from nbadata.timeutil import eastern_date, parse_iso_utc, today_eastern

logger = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def _int(x, default=0) -> int:
    try:
        if x in (None, "", "--"):
            return default
        return int(float(str(x).replace("+", "")))
    except (ValueError, TypeError):
        return default


def _made_att(x) -> tuple:
    """'10-20' -> (10, 20)."""
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", str(x or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _minutes(x) -> float:
    s = str(x or "").strip()
    if not s or s in ("--", "DNP"):
        return 0.0
    if ":" in s:
        m, sec = s.split(":", 1)
        return _int(m) + _int(sec) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _status(event: dict) -> str:
    t = ((event.get("status") or {}).get("type")) or {}
    name = str(t.get("name", "")).upper()
    state = str(t.get("state", "")).lower()
    if t.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final"
    if "POSTPONED" in name or "CANCELED" in name:
        return "postponed"
    if state == "in":
        return "in_progress"
    return "scheduled"


class ESPNProvider(GameDataProvider):
    name = "espn"

    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient()
        self._scoreboard_cache: Dict[str, List[GameStub]] = {}

    # ── schedule / scores ────────────────────────────────────────────────
    def _scoreboard(self, d: date) -> List[GameStub]:
        key = d.strftime("%Y%m%d")
        if key in self._scoreboard_cache and d < today_eastern():
            return self._scoreboard_cache[key]
        data = self.http.get_json(f"{BASE}/scoreboard", params={"dates": key, "limit": 100})
        stubs = self.parse_scoreboard(data)
        self._scoreboard_cache[key] = stubs
        return stubs

    @staticmethod
    def parse_scoreboard(data: dict) -> List[GameStub]:
        out: List[GameStub] = []
        for ev in (data or {}).get("events", []) or []:
            try:
                comp = (ev.get("competitions") or [{}])[0]
                comps = comp.get("competitors") or []
                home = next((c for c in comps if c.get("homeAway") == "home"), None)
                away = next((c for c in comps if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                tip = parse_iso_utc(ev.get("date") or comp.get("date"))
                gdate = eastern_date(tip) if tip else None
                if gdate is None:
                    continue
                status = _status(ev)
                hp = _int(home.get("score"), None) if status == "final" else None
                ap = _int(away.get("score"), None) if status == "final" else None
                stub = build_stub(
                    game_date=gdate,
                    season_type=_int((ev.get("season") or {}).get("type"), None),
                    home_text=(home.get("team") or {}).get("abbreviation") or (home.get("team") or {}).get("displayName"),
                    away_text=(away.get("team") or {}).get("abbreviation") or (away.get("team") or {}).get("displayName"),
                    tipoff_utc=tip.strftime("%Y-%m-%dT%H:%M:%SZ"), status=status,
                    home_pts=hp, away_pts=ap, espn_id=str(ev.get("id")), source="espn",
                )
                if stub:
                    out.append(stub)
            except Exception as e:  # one bad event must not poison the day
                logger.debug("ESPN scoreboard event skipped: %s", e)
        return out

    def list_games(self, start: date, end: date) -> List[GameStub]:
        out: Dict[str, GameStub] = {}
        d = start
        while d <= end:
            for stub in self._scoreboard(d):
                out[stub.game_key] = stub
            d += timedelta(days=1)
        return sorted(out.values(), key=lambda s: (s.game_date, s.game_key))

    def supports(self, stub: GameStub) -> bool:
        return bool(stub.espn_id)

    # ── box score ────────────────────────────────────────────────────────
    def fetch_box(self, stub: GameStub) -> Optional[BoxScore]:
        if not stub.espn_id:
            return None
        data = self.http.get_json(f"{BASE}/summary", params={"event": stub.espn_id})
        return self.parse_summary(data, stub)

    @staticmethod
    def parse_summary(data: dict, stub: GameStub) -> Optional[BoxScore]:
        box = (data or {}).get("boxscore") or {}
        groups = box.get("players") or []
        if not groups:
            return None
        players: List[PlayerLine] = []
        for grp in groups:
            team = resolve_team((grp.get("team") or {}).get("abbreviation") or (grp.get("team") or {}).get("displayName"))
            if team is None or team.id not in (stub.home_team_id, stub.away_team_id):
                continue
            for stat_block in grp.get("statistics") or []:
                labels = [str(x).upper() for x in (stat_block.get("labels") or stat_block.get("names") or [])]
                if not labels:
                    continue
                idx = {lab: i for i, lab in enumerate(labels)}
                for a in stat_block.get("athletes") or []:
                    ath = a.get("athlete") or {}
                    name = ath.get("displayName") or ath.get("shortName")
                    if not name:
                        continue
                    stats = a.get("stats") or []

                    def val(*labs):
                        for lab in labs:
                            i = idx.get(lab)
                            if i is not None and i < len(stats):
                                return stats[i]
                        return None

                    dnp = bool(a.get("didNotPlay")) or not stats
                    mins = 0.0 if dnp else _minutes(val("MIN"))
                    fgm, fga = _made_att(val("FG"))
                    tpm, tpa = _made_att(val("3PT", "3P"))
                    ftm, fta = _made_att(val("FT"))
                    players.append(PlayerLine(
                        team_id=team.id, player_key=normalize_name(name), player_name=name,
                        player_id=str(ath.get("id")) if ath.get("id") is not None else None,
                        starter=1 if a.get("starter") else 0, played=1 if mins > 0 else 0, minutes=mins,
                        pts=_int(val("PTS")), reb=_int(val("REB")), ast=_int(val("AST")),
                        fgm=fgm, fga=fga, fg3m=tpm, fg3a=tpa, ftm=ftm, fta=fta,
                        oreb=_int(val("OREB")), dreb=_int(val("DREB")), tov=_int(val("TO", "TOV")),
                        stl=_int(val("STL")), blk=_int(val("BLK")), pf=_int(val("PF")),
                        plus_minus=_int(val("+/-"), None), dnp_reason=(a.get("reason") or None) if dnp else None,
                    ))
        if not players:
            return None
        # final score: header is the most reliable; fall back to summing players
        hp, ap = stub.home_pts, stub.away_pts
        try:
            comp = ((data.get("header") or {}).get("competitions") or [{}])[0]
            for c in comp.get("competitors") or []:
                t = resolve_team((c.get("team") or {}).get("abbreviation") or (c.get("team") or {}).get("displayName"))
                if t and c.get("score") not in (None, ""):
                    if t.id == stub.home_team_id:
                        hp = _int(c.get("score"))
                    elif t.id == stub.away_team_id:
                        ap = _int(c.get("score"))
        except Exception:
            pass
        if hp is None:
            hp = sum(p.pts for p in players if p.team_id == stub.home_team_id)
        if ap is None:
            ap = sum(p.pts for p in players if p.team_id == stub.away_team_id)
        stub = GameStub(**{**stub.__dict__, "home_pts": hp, "away_pts": ap, "status": "final"})
        return BoxScore(game=stub, players=players)

    # ── injuries ─────────────────────────────────────────────────────────
    def fetch_injuries(self) -> List[Dict]:
        data = self.http.get_json(f"{BASE}/injuries")
        return self.parse_injuries(data)

    @staticmethod
    def parse_injuries(data: dict) -> List[Dict]:
        rows = []
        for team_block in (data or {}).get("injuries", []) or []:
            team = resolve_team(team_block.get("displayName") or (team_block.get("team") or {}).get("abbreviation"))
            if team is None:
                continue
            for inj in team_block.get("injuries") or []:
                ath = inj.get("athlete") or {}
                name = ath.get("displayName")
                if not name:
                    continue
                rows.append({
                    "team_id": team.id, "player_key": normalize_name(name), "player_name": name,
                    "status": str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip(),
                    "detail": str(inj.get("shortComment") or inj.get("longComment") or "")[:200],
                })
        return rows

    # ── embedded game odds (fallback source) ─────────────────────────────
    def fetch_scoreboard_odds(self, d: date) -> List[Dict]:
        data = self.http.get_json(f"{BASE}/scoreboard", params={"dates": d.strftime("%Y%m%d"), "limit": 100})
        return self.parse_scoreboard_odds(data)

    @staticmethod
    def parse_scoreboard_odds(data: dict) -> List[Dict]:
        """Extract the single book's game lines ESPN embeds in the scoreboard (best effort)."""
        out = []
        for ev in (data or {}).get("events", []) or []:
            comp = (ev.get("competitions") or [{}])[0]
            odds_list = comp.get("odds") or []
            if not odds_list:
                continue
            o = odds_list[0]
            comps = comp.get("competitors") or []
            home = next((c for c in comps if c.get("homeAway") == "home"), {})
            away = next((c for c in comps if c.get("homeAway") == "away"), {})
            ht, at = resolve_team((home.get("team") or {}).get("abbreviation")), resolve_team((away.get("team") or {}).get("abbreviation"))
            tip = parse_iso_utc(ev.get("date"))
            if not (ht and at and tip):
                continue
            home_odds, away_odds = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
            spread = o.get("spread")
            # ESPN's `spread` is from the HOME team's perspective when homeTeamOdds.favorite is set;
            # derive it from `details` ("BOS -5.5") when ambiguous.
            spread_home = None
            details = str(o.get("details") or "")
            m = re.match(r"^\s*([A-Za-z.]+)\s+([+-]?\d+(?:\.\d+)?)\s*$", details)
            if m:
                fav = resolve_team(m.group(1))
                val = float(m.group(2))
                if fav is not None:
                    spread_home = val if fav.id == ht.id else -val
            if spread_home is None and spread is not None:
                try:
                    spread_home = float(spread)
                except (TypeError, ValueError):
                    pass
            out.append({
                "game_date": eastern_date(tip).isoformat(), "home_team_id": ht.id, "away_team_id": at.id,
                "commence_utc": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "book": str((o.get("provider") or {}).get("name") or "espn").lower().replace(" ", ""),
                "spread_home": spread_home,
                # ESPN's embedded odds do not reliably expose juice; leave prices empty (callers assume -110)
                "spread_home_price": None,
                "total": float(o["overUnder"]) if o.get("overUnder") is not None else None,
                "ml_home": _int(home_odds.get("moneyLine"), None), "ml_away": _int(away_odds.get("moneyLine"), None),
            })
        return out

    # ── health ───────────────────────────────────────────────────────────
    def health_check(self) -> Dict:
        res = {"provider": self.name, "ok": False, "steps": []}

        def step(name, ok, detail=""):
            res["steps"].append({"step": name, "ok": ok, "detail": detail})
            return ok

        try:
            # Look back for a day with a finished game (handles off-season / All-Star break)
            d, found = today_eastern() - timedelta(days=1), None
            for _ in range(10):
                stubs = [s for s in self._scoreboard(d) if s.status == "final"]
                if stubs:
                    found = stubs[0]
                    break
                d -= timedelta(days=1)
            if not step("scoreboard", found is not None,
                        f"{len(stubs) if found else 0} final games on {d}" if found else "no finished games in last 10 days (off-season?)"):
                res["ok"] = True  # endpoint answered; just nothing to fetch
                res["note"] = "reachable, but no recent games to verify box-score parsing"
                return res
            box = self.fetch_box(found)
            if not step("box score", box is not None, f"{len(box.players) if box else 0} player lines for {found.game_key}"):
                return res
            problems = box.validate()
            step("box score sanity", not problems, "; ".join(problems) or "player points match team scores")
            res["ok"] = not problems
        except ProviderError as e:
            step("request", False, str(e))
        return res
