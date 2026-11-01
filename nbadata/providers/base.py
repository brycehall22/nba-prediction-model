"""Provider-neutral records and the interface every game-data provider implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from config import season_for_date
from nbadata.teams import resolve_team
from nbadata.warehouse import make_game_key


@dataclass
class GameStub:
    game_key: str
    game_date: str                 # US/Eastern YYYY-MM-DD
    season: str
    season_type: Optional[int]     # 1 pre, 2 regular, 3 playoffs, 5 play-in
    home_team_id: int
    away_team_id: int
    tipoff_utc: Optional[str]
    status: str                    # scheduled | in_progress | final | postponed
    home_pts: Optional[int] = None
    away_pts: Optional[int] = None
    espn_id: Optional[str] = None
    nba_id: Optional[str] = None
    source: str = ""

    def as_row(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class PlayerLine:
    team_id: int
    player_key: str
    player_name: str
    player_id: Optional[str] = None
    starter: int = 0
    played: int = 0
    minutes: float = 0.0
    pts: int = 0
    reb: int = 0
    ast: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    tov: int = 0
    stl: int = 0
    blk: int = 0
    pf: int = 0
    plus_minus: Optional[int] = None
    dnp_reason: Optional[str] = None


@dataclass
class BoxScore:
    game: GameStub
    players: List[PlayerLine] = field(default_factory=list)

    def team_rows(self) -> List[Dict]:
        """Team totals = sum of player lines (robust to provider quirks); points from the final score."""
        rows = []
        for tid, is_home, pts in (
            (self.game.home_team_id, 1, self.game.home_pts),
            (self.game.away_team_id, 0, self.game.away_pts),
        ):
            ps = [p for p in self.players if p.team_id == tid]
            agg = {k: sum(getattr(p, k) for p in ps) for k in
                   ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pf")}
            player_pts = sum(p.pts for p in ps)
            rows.append({"game_key": self.game.game_key, "team_id": tid, "is_home": is_home,
                         "pts": pts if pts is not None else player_pts, **agg})
        return rows

    def player_rows(self) -> List[Dict]:
        return [{"game_key": self.game.game_key, **p.__dict__} for p in self.players]

    def validate(self) -> List[str]:
        """Sanity checks. Returns a list of problems (empty = looks right)."""
        problems = []
        if self.game.home_pts is None or self.game.away_pts is None:
            problems.append("missing final score")
        for tid, pts in ((self.game.home_team_id, self.game.home_pts), (self.game.away_team_id, self.game.away_pts)):
            ps = [p for p in self.players if p.team_id == tid]
            if len([p for p in ps if p.played]) < 5:
                problems.append(f"team {tid}: fewer than 5 players with minutes")
            tot = sum(p.pts for p in ps)
            if pts is not None and ps and abs(tot - pts) > 2:
                problems.append(f"team {tid}: player points {tot} != team score {pts}")
        return problems


def build_stub(*, game_date: date, season_type: Optional[int], home_text, away_text, tipoff_utc, status,
               home_pts=None, away_pts=None, espn_id=None, nba_id=None, source="") -> Optional[GameStub]:
    """Create a GameStub, or None if either team isn't an NBA team (All-Star, exhibitions...)."""
    home, away = resolve_team(home_text), resolve_team(away_text)
    if home is None or away is None:
        return None
    ds = game_date.isoformat()
    return GameStub(
        game_key=make_game_key(ds, away.id, home.id), game_date=ds, season=season_for_date(game_date),
        season_type=season_type, home_team_id=home.id, away_team_id=away.id, tipoff_utc=tipoff_utc,
        status=status, home_pts=home_pts, away_pts=away_pts, espn_id=espn_id, nba_id=nba_id, source=source,
    )


class GameDataProvider:
    """Interface. `list_games` returns stubs for a date range; `fetch_box` returns a full box score."""

    name = "base"

    def list_games(self, start: date, end: date) -> List[GameStub]:
        raise NotImplementedError

    def fetch_box(self, stub: GameStub) -> Optional[BoxScore]:
        raise NotImplementedError

    def supports(self, stub: GameStub) -> bool:
        """Whether this provider already has an id for the game (otherwise ingest looks it up by key)."""
        return True

    def health_check(self) -> Dict:
        raise NotImplementedError
