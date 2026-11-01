"""
Response builders that mirror the *shape* of the ESPN / NBA-CDN / The Odds API payloads.

NOTE: these are hand-built reconstructions of the documented/known schemas, not captured
live responses. They prove the parsers are correct for that shape and fail loudly when
fields are missing; `python cli.py doctor` is what verifies the live endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

ESPN_LABELS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]


def espn_scoreboard(events: List[Dict]) -> Dict:
    out = []
    for e in events:
        odds = e.get("odds")
        out.append({
            "id": str(e["id"]), "date": e["date"], "season": {"year": 2025, "type": e.get("type", 2)},
            "status": {"type": {"name": "STATUS_FINAL" if e.get("final", True) else "STATUS_SCHEDULED",
                                "completed": e.get("final", True), "state": "post" if e.get("final", True) else "pre"}},
            "competitions": [{
                "competitors": [
                    {"homeAway": "home", "score": str(e.get("home_pts", 0)), "team": {"id": "1", "abbreviation": e["home"], "displayName": e.get("home_name", e["home"])}},
                    {"homeAway": "away", "score": str(e.get("away_pts", 0)), "team": {"id": "2", "abbreviation": e["away"], "displayName": e.get("away_name", e["away"])}},
                ],
                **({"odds": [odds]} if odds else {}),
            }],
        })
    return {"events": out}


def _espn_athlete(name: str, pid: int, mins: str, pts: int, starter=True, fg="5-10", tp="1-3", ft="2-2",
                  reb=4, ast=3, to=1, dnp=False):
    stats = [] if dnp else [mins, str(pts), fg, tp, ft, str(reb), str(ast), str(to), "1", "0", "1", str(reb - 1), "2", "+3"]
    return {"athlete": {"id": str(pid), "displayName": name}, "starter": starter, "didNotPlay": dnp,
            "reason": "COACH'S DECISION" if dnp else "", "stats": stats}


def espn_summary(home: str, away: str, home_players: List[tuple], away_players: List[tuple],
                 home_pts: int, away_pts: int) -> Dict:
    def block(abbr, players):
        return {"team": {"abbreviation": abbr, "displayName": abbr},
                "statistics": [{"labels": ESPN_LABELS, "athletes": [
                    _espn_athlete(n, i, m, p, starter=s, dnp=(m == "DNP")) for i, (n, m, p, s) in enumerate(players, start=1)]}]}
    return {
        "boxscore": {"players": [block(home, home_players), block(away, away_players)]},
        "header": {"competitions": [{"competitors": [
            {"homeAway": "home", "score": str(home_pts), "team": {"abbreviation": home}},
            {"homeAway": "away", "score": str(away_pts), "team": {"abbreviation": away}}]}]},
    }


def roster(prefix: str, total_pts: int, n: int = 9) -> List[tuple]:
    """n players whose points sum exactly to total_pts."""
    base, extra = divmod(total_pts, n)
    out = []
    for i in range(n):
        pts = base + (1 if i < extra else 0)
        mins = "34:30" if i < 5 else "14:15"
        out.append((f"{prefix} Player{i + 1}", mins, pts, i < 5))
    out.append((f"{prefix} Benchwarmer", "DNP", 0, False))
    return out


def cdn_schedule(games: List[Dict]) -> Dict:
    by_date: Dict[str, list] = {}
    for g in games:
        by_date.setdefault(g["date"], []).append({
            "gameId": g["id"], "gameStatus": g.get("status", 3), "gameDateTimeUTC": g["utc"],
            "homeTeam": {"teamTricode": g["home"], "score": g.get("home_pts", 0)},
            "awayTeam": {"teamTricode": g["away"], "score": g.get("away_pts", 0)},
        })
    return {"leagueSchedule": {"gameDates": [{"gameDate": d, "games": gs} for d, gs in by_date.items()]}}


def cdn_boxscore(home_players: List[tuple], away_players: List[tuple], home_pts: int, away_pts: int) -> Dict:
    def team(players, score):
        plist = []
        for i, (n, m, p, s) in enumerate(players, start=1):
            dnp = m == "DNP"
            mm, ss = (m.split(":") + ["0"])[:2] if not dnp else ("0", "0")
            plist.append({
                "personId": 1000 + i, "name": n, "starter": "1" if s else "0", "status": "INACTIVE" if dnp else "ACTIVE",
                "notPlayingReason": "INACTIVE_COACH" if dnp else "",
                "statistics": {"minutes": "" if dnp else f"PT{int(mm):02d}M{int(ss):02d}.00S", "points": 0 if dnp else p,
                               "reboundsTotal": 4, "assists": 3, "fieldGoalsMade": 5, "fieldGoalsAttempted": 10,
                               "threePointersMade": 1, "threePointersAttempted": 3, "freeThrowsMade": 2, "freeThrowsAttempted": 2,
                               "reboundsOffensive": 1, "reboundsDefensive": 3, "turnovers": 1, "steals": 1, "blocks": 0,
                               "foulsPersonal": 2, "plusMinusPoints": 3.0},
            })
        return {"score": score, "players": plist}
    return {"game": {"homeTeam": team(home_players, home_pts), "awayTeam": team(away_players, away_pts)}}


def odds_api_event(eid: str, home: str, away: str, commence: str, books: Dict[str, Dict]) -> Dict:
    """books: {book_key: {'spread': (home_point, home_price, away_price), 'total': (pt, over, under), 'ml': (home, away)}}"""
    bks = []
    for key, m in books.items():
        markets = []
        if "ml" in m:
            markets.append({"key": "h2h", "outcomes": [{"name": home, "price": m["ml"][0]}, {"name": away, "price": m["ml"][1]}]})
        if "spread" in m:
            sp, hp, ap = m["spread"]
            markets.append({"key": "spreads", "outcomes": [{"name": home, "price": hp, "point": sp}, {"name": away, "price": ap, "point": -sp}]})
        if "total" in m:
            t, o, u = m["total"]
            markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": o, "point": t}, {"name": "Under", "price": u, "point": t}]})
        bks.append({"key": key, "title": key.title(), "markets": markets})
    return {"id": eid, "sport_key": "basketball_nba", "commence_time": commence, "home_team": home, "away_team": away, "bookmakers": bks}


def odds_api_props(eid: str, home: str, away: str, commence: str, books: Dict[str, List[tuple]]) -> Dict:
    """books: {book_key: [(player, line, over_price, under_price), ...]}"""
    bks = []
    for key, rows in books.items():
        outs = []
        for player, line, o, u in rows:
            outs.append({"name": "Over", "description": player, "price": o, "point": line})
            outs.append({"name": "Under", "description": player, "price": u, "point": line})
        bks.append({"key": key, "title": key.title(), "markets": [{"key": "player_points", "outcomes": outs}]})
    return {"id": eid, "sport_key": "basketball_nba", "commence_time": commence, "home_team": home, "away_team": away, "bookmakers": bks}
