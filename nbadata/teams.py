"""
Static NBA team table and tolerant name resolution.

Every data source spells teams differently ("LA Clippers", "LAC", "L.A. Clippers",
"GS Warriors", "Golden State", "WSH"...). All of them are resolved to one canonical
`Team`, keyed by the official NBA team id, so odds, box scores and schedules can
be joined without string heuristics scattered through the codebase.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Team:
    id: int            # official NBA team id (same as nba_api)
    abbr: str          # NBA tricode, our canonical short code
    city: str
    nickname: str
    espn_abbr: str
    extra_aliases: Tuple[str, ...] = ()

    @property
    def full_name(self) -> str:
        return f"{self.city} {self.nickname}"


# id, abbr, city, nickname, espn_abbr, extra aliases
_RAW: List[Tuple] = [
    (1610612737, "ATL", "Atlanta", "Hawks", "ATL", ()),
    (1610612738, "BOS", "Boston", "Celtics", "BOS", ()),
    (1610612751, "BKN", "Brooklyn", "Nets", "BKN", ("BRK",)),
    (1610612766, "CHA", "Charlotte", "Hornets", "CHA", ("CHO",)),
    (1610612741, "CHI", "Chicago", "Bulls", "CHI", ()),
    (1610612739, "CLE", "Cleveland", "Cavaliers", "CLE", ("Cavs",)),
    (1610612742, "DAL", "Dallas", "Mavericks", "DAL", ("Mavs",)),
    (1610612743, "DEN", "Denver", "Nuggets", "DEN", ()),
    (1610612765, "DET", "Detroit", "Pistons", "DET", ()),
    (1610612744, "GSW", "Golden State", "Warriors", "GS", ("GS", "Golden St")),
    (1610612745, "HOU", "Houston", "Rockets", "HOU", ()),
    (1610612754, "IND", "Indiana", "Pacers", "IND", ()),
    (1610612746, "LAC", "Los Angeles", "Clippers", "LAC", ("LA Clippers", "L.A. Clippers")),
    (1610612747, "LAL", "Los Angeles", "Lakers", "LAL", ("LA Lakers", "L.A. Lakers")),
    (1610612763, "MEM", "Memphis", "Grizzlies", "MEM", ()),
    (1610612748, "MIA", "Miami", "Heat", "MIA", ()),
    (1610612749, "MIL", "Milwaukee", "Bucks", "MIL", ()),
    (1610612750, "MIN", "Minnesota", "Timberwolves", "MIN", ("Wolves", "T-Wolves")),
    (1610612740, "NOP", "New Orleans", "Pelicans", "NO", ("NO", "NOH", "NOLA")),
    (1610612752, "NYK", "New York", "Knicks", "NY", ("NY",)),
    (1610612760, "OKC", "Oklahoma City", "Thunder", "OKC", ()),
    (1610612753, "ORL", "Orlando", "Magic", "ORL", ()),
    (1610612755, "PHI", "Philadelphia", "76ers", "PHI", ("Sixers", "PHL")),
    (1610612756, "PHX", "Phoenix", "Suns", "PHX", ("PHO",)),
    (1610612757, "POR", "Portland", "Trail Blazers", "POR", ("Blazers", "Trailblazers")),
    (1610612758, "SAC", "Sacramento", "Kings", "SAC", ()),
    (1610612759, "SAS", "San Antonio", "Spurs", "SA", ("SA",)),
    (1610612761, "TOR", "Toronto", "Raptors", "TOR", ()),
    (1610612762, "UTA", "Utah", "Jazz", "UTAH", ("UTAH", "UTH")),
    (1610612764, "WAS", "Washington", "Wizards", "WSH", ("WSH",)),
]

TEAMS: List[Team] = [Team(*r) for r in _RAW]
TEAMS_BY_ID: Dict[int, Team] = {t.id: t for t in TEAMS}
TEAMS_BY_ABBR: Dict[str, Team] = {t.abbr: t for t in TEAMS}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_index() -> Dict[str, Team]:
    idx: Dict[str, Team] = {}
    ambiguous: set = set()

    def add(key: str, team: Team, allow_override: bool = False):
        key = _norm(key)
        if not key:
            return
        if key in idx and idx[key] is not team and not allow_override:
            ambiguous.add(key)
        else:
            idx[key] = team

    for t in TEAMS:
        add(t.full_name, t)
        add(t.abbr, t)
        add(t.espn_abbr, t)
        add(t.nickname, t)                       # "Lakers" (nicknames are unique)
        for a in t.extra_aliases:
            add(a, t)
        # Legacy DraftKings style "BKN Nets" / "PHO Suns" / "GS Warriors": any short code + nickname
        short_codes = {t.abbr, t.espn_abbr} | {a for a in t.extra_aliases if len(a) <= 4}
        for code in short_codes:
            add(f"{code} {t.nickname}", t)
    # City-only names are only unambiguous where a single team plays there
    for t in TEAMS:
        if t.city not in ("Los Angeles",):
            add(t.city, t)
    # "LA Lakers" / "LA Clippers" are explicit aliases above; bare "LA"/"Los Angeles" is ambiguous.
    for k in ambiguous:
        idx.pop(k, None)
    return idx


_INDEX = _build_index()


def resolve_team(text) -> Optional[Team]:
    """Resolve any reasonable spelling of an NBA team to a `Team`, or None."""
    if text is None:
        return None
    if isinstance(text, (int,)) and text in TEAMS_BY_ID:
        return TEAMS_BY_ID[text]
    key = _norm(text)
    if not key:
        return None
    if key in _INDEX:
        return _INDEX[key]
    # Try dropping a leading token, e.g. "the boston celtics"
    parts = key.split()
    if len(parts) > 1 and parts[0] in ("the",):
        return _INDEX.get(" ".join(parts[1:]))
    return None


def team_id(text) -> Optional[int]:
    t = resolve_team(text)
    return t.id if t else None


def team_abbr(team_id_: int) -> str:
    t = TEAMS_BY_ID.get(int(team_id_))
    return t.abbr if t else str(team_id_)


def team_name(team_id_: int) -> str:
    t = TEAMS_BY_ID.get(int(team_id_))
    return t.full_name if t else str(team_id_)


def all_team_ids() -> List[int]:
    return [t.id for t in TEAMS]
