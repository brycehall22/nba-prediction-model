"""
Centralised configuration.

Everything tunable lives here. Nothing in this file touches the network.
Secrets (the odds API key) are NOT stored here: set the ODDS_API_KEY environment
variable or put it in a local `.env` file.
"""

from __future__ import annotations

import os

# joblib/loky shells out to `wmic` to count physical cores; modern Windows no longer ships it, which prints a
# scary (harmless) traceback. Supplying the count up front silences it on every platform.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional, Tuple


# ── Season helpers ───────────────────────────────────────────────────────────
def season_label(start_year: int) -> str:
    """2025 -> '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_for_date(d: Optional[date] = None) -> str:
    """NBA season label a calendar date belongs to (season rolls over in July)."""
    d = d or date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return season_label(start)


def recent_seasons(n: int, as_of: Optional[date] = None) -> Tuple[str, ...]:
    """The last `n` season labels ending with the one containing `as_of`, oldest first."""
    as_of = as_of or date.today()
    end = as_of.year if as_of.month >= 7 else as_of.year - 1
    return tuple(season_label(y) for y in range(end - n + 1, end + 1))


# ── League baselines (only used as priors when there is no data yet) ─────────
@dataclass(frozen=True)
class LeagueDefaults:
    SEASON: str = field(default_factory=season_for_date)
    AVG_TEAM_SCORE: float = 114.0
    AVG_PACE: float = 99.0                 # possessions per team per game
    AVG_OFF_RATING: float = 115.0
    HOME_COURT_ADVANTAGE: float = 2.5      # points of margin (prior; the model fits its own)
    SEASON_START_MONTH: int = 10


# ── Data warehouse & providers ───────────────────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    DB_PATH: str = "data/nba_warehouse.db"
    REQUEST_TIMEOUT: float = 20.0
    MIN_REQUEST_INTERVAL: float = 0.35     # polite delay between calls to the same host
    MAX_RETRIES: int = 3
    RETRY_BACKOFF: float = 1.5
    # Order matters: the first provider that answers wins, the rest are fallbacks.
    PROVIDERS: Tuple[str, ...] = ("espn", "nba_cdn")
    BACKFILL_SEASONS: int = 3
    UPDATE_DAYS_BACK: int = 4              # re-check recent days on every update
    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


# ── Odds ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OddsConfig:
    API_KEY_ENV: str = "ODDS_API_KEY"
    SPORT_KEY: str = "basketball_nba"
    TARGET_BOOK: str = "draftkings"        # the book you actually bet at
    REGIONS: str = "us"                    # all US books -> consensus at no extra credit cost
    GAME_MARKETS: Tuple[str, ...] = ("h2h", "spreads", "totals")
    PROP_MARKETS: Tuple[str, ...] = ("player_points",)
    PROP_LOOKAHEAD_HOURS: int = 36         # only pull props for games starting soon
    MAX_PROP_EVENTS: int = 15              # hard cap on per-event prop calls per refresh
    MIN_REFRESH_MINUTES: int = 60          # don't re-pull more often than this (saves quota)
    # ESPN's scoreboard embeds one book's game lines; used only if no API key is set.
    ENABLE_ESPN_FALLBACK: bool = True


# ── Modelling ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    MODELS_DIR: str = "models"
    TRAIN_SEASONS: int = 3
<<<<<<< HEAD
    # Team ratings (walk-forward, time-decayed ridge).
    # Defaults match values adopted by `cli.py backtest` on holdout data
    # (half_life 50→100, ridge 25→10, carry 0.55→0.3). models/tuning.json can still override.
    RATING_HALF_LIFE_DAYS: float = 100.0
    RATING_RIDGE: float = 10.0             # shrinkage of off/def ratings toward the prior
    PACE_RIDGE: float = 10.0
    PRIOR_SEASON_CARRY: float = 0.3        # fraction of last season's ratings carried over
=======
    # Team ratings (walk-forward, time-decayed ridge)
    RATING_HALF_LIFE_DAYS: float = 50.0
    RATING_RIDGE: float = 25.0             # shrinkage of off/def ratings toward the prior
    PACE_RIDGE: float = 25.0
    PRIOR_SEASON_CARRY: float = 0.55       # fraction of last season's ratings carried over
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
    # Game-level uncertainty. These are priors; training replaces them with fitted values.
    DEFAULT_MARGIN_SIGMA: float = 12.5
    DEFAULT_TOTAL_SIGMA: float = 18.0
    MIN_TEAM_GAMES_FOR_EVAL: int = 15      # ignore early-season games when scoring accuracy
    # Players
<<<<<<< HEAD
    PLAYER_EWM_HALFLIFE_GAMES: float = 12.0  # was 8; longer window stabilizes rates
=======
    PLAYER_EWM_HALFLIFE_GAMES: float = 8.0
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
    MIN_PRIOR_GAMES: int = 8
    MIN_TRAIN_MINUTES: float = 8.0
    ROTATION_MIN_MINUTES: float = 12.0
    CV_FOLDS: int = 4
    RANDOM_STATE: int = 42


# ── Betting math ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BettingConfig:
    DEFAULT_ODDS: int = -110
<<<<<<< HEAD
    EV_THRESHOLD_PCT: float = 2.5          # flag as value above this EV% (slightly lower after heavy market blend)
=======
    EV_THRESHOLD_PCT: float = 3.0          # flag as value above this EV%
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b
    HIGH_EV_THRESHOLD_PCT: float = 8.0
    KELLY_FRACTION: float = 0.25           # quarter Kelly
    MAX_KELLY_BET_PCT: float = 5.0
    # Props & games: blend final probability toward the de-vigged market probability.
    # 0 = trust the model fully, 1 = just echo the market. Calibrate with `cli.py report`.
<<<<<<< HEAD
    # Defaults raised after graded-pick calibration (props best weight ≈90%, games ≈70%).
    # models/tuning.json overrides these when present.
    PROP_MARKET_WEIGHT: float = 0.90
    GAME_MARKET_WEIGHT: float = 0.70
    # Do not recommend a moneyline/spread side the model itself gives < this probability.
    # Stops longshot "value" bets (e.g. 26% dog at +450) that are +EV on paper but historically lost.
    # Pure EV on every side is still visible with `slate --all`.
    MIN_MODEL_PROB_TO_FLAG: float = 0.42
    # When a prop has no two-way market fair probability, still shrink the model toward 50%
    # so we do not claim 30%+ EV on a pure-model number.
    PROP_NO_MARKET_SHRINK: float = 0.50
=======
    PROP_MARKET_WEIGHT: float = 0.50
    GAME_MARKET_WEIGHT: float = 0.35
>>>>>>> 70c7a466080363221144ffdb48ce1198fc17722b


@dataclass(frozen=True)
class PropsConfig:
    DEFAULT_NB_R: float = 18.0             # negative-binomial dispersion prior (fitted at training)


@dataclass(frozen=True)
class PredictionConfig:
    MAX_PLAYER_SCALE: float = 1.12         # clamp when reconciling player sums to the team total
    MIN_PLAYER_SCALE: float = 0.88


LEAGUE = LeagueDefaults()
DATA = DataConfig()
ODDS = OddsConfig()
MODEL = ModelConfig()
BETTING = BettingConfig()
PROPS = PropsConfig()
PREDICTION = PredictionConfig()
