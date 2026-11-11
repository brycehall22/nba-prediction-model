"""Canonical odds records, American-odds helpers and de-vigging. No network, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return odds / 100.0 + 1.0 if odds > 0 else 100.0 / abs(odds) + 1.0


def american_to_prob(odds: float) -> float:
    """Raw implied probability (includes the book's vig)."""
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def devig_two_way(price_a: Optional[float], price_b: Optional[float]) -> Optional[Tuple[float, float]]:
    """Proportional de-vig of a two-outcome market -> (fair_prob_a, fair_prob_b)."""
    if price_a is None or price_b is None:
        return None
    pa, pb = american_to_prob(price_a), american_to_prob(price_b)
    s = pa + pb
    return (pa / s, pb / s) if s > 0 else None


@dataclass
class GameOdds:
    game_date: str
    home_team_id: int
    away_team_id: int
    commence_utc: Optional[str]
    book: str
    fetched_at: str
    source: str
    spread_home: Optional[float] = None
    spread_home_price: Optional[int] = None
    spread_away_price: Optional[int] = None
    total: Optional[float] = None
    over_price: Optional[int] = None
    under_price: Optional[int] = None
    ml_home: Optional[int] = None
    ml_away: Optional[int] = None

    def as_row(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class PropOdds:
    game_date: str
    home_team_id: int
    away_team_id: int
    commence_utc: Optional[str]
    book: str
    fetched_at: str
    source: str
    market: str
    player_name: str
    player_key: str
    line: float
    over_price: Optional[int] = None
    under_price: Optional[int] = None

    def as_row(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class RefreshReport:
    ok: bool = False
    source: str = ""
    games: int = 0
    props: int = 0
    credits_remaining: Optional[int] = None
    credits_used: Optional[int] = None
    skipped: bool = False
    messages: List[str] = field(default_factory=list)

    def note(self, msg: str):
        self.messages.append(msg)
