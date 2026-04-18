"""
Centralized configuration for the NBA Prediction System.

All tunable parameters, thresholds, and constants live here.
Import from this module instead of hardcoding values across files.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ── League Baseline Constants ────────────────────────────────────────────────
@dataclass(frozen=True)
class LeagueDefaults:
    """League-wide baseline statistics (updated each season)."""
    SEASON: str = "2024-25"
    AVG_TEAM_SCORE: float = 113.0
    AVG_PACE: float = 100.0
    AVG_OFF_RATING: float = 115.0
    AVG_DEF_RATING: float = 115.0
    AVG_EFG_PCT: float = 0.54
    AVG_TS_PCT: float = 0.57
    AVG_AST_RATIO: float = 24.0
    AVG_REB_RATE: float = 50.0
    AVG_FG3_PCT: float = 0.363
    HOME_COURT_ADVANTAGE: float = 3.5  # points
    GAMES_PER_SEASON: int = 82
    SEASON_START_MONTH: int = 10  # October
    SEASON_END_MONTH: int = 4    # April


# ── API & Networking ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class APIConfig:
    """NBA API request configuration."""
    MIN_REQUEST_INTERVAL: float = 2.5   # seconds between requests
    MAX_JITTER: float = 0.5             # random jitter added to interval
    MAX_RETRIES: int = 3                # retries on transient failures
    RETRY_BACKOFF_BASE: float = 2.0     # exponential backoff base
    REQUEST_TIMEOUT: int = 30           # seconds
    USER_AGENTS: tuple = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    )


# ── Caching ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CacheConfig:
    """SQLite cache TTLs and cleanup rules."""
    DB_PATH: str = "nba_cache.db"
    TEAM_GAMES_TTL_HOURS: int = 6       # re-fetch after game nights
    PLAYER_GAMES_TTL_HOURS: int = 6
    ADVANCED_STATS_TTL_HOURS: int = 12
    CLEANUP_AFTER_DAYS: int = 7


# ── Model Training ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainingConfig:
    """Model training hyperparameters and thresholds."""
    SEASONS: tuple = ("2022-23", "2023-24", "2024-25")
    SEASON_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "2022-23": 0.5,
        "2023-24": 0.8,
        "2024-25": 1.0,
    })
    MIN_TEAM_TRAINING_SAMPLES: int = 80
    MIN_PLAYER_TRAINING_SAMPLES: int = 50
    MIN_GAMES_FOR_FEATURES: int = 5
    ROLLING_WINDOWS: tuple = (3, 5, 10, 15)
    FEATURE_SELECTION_K_TEAM: int = 50
    FEATURE_SELECTION_K_PLAYER: int = 30
    CV_SPLITS: int = 5
    RANDOM_STATE: int = 42
    MODELS_DIR: str = "enhanced_models"


# ── Prediction ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PredictionConfig:
    """Prediction pipeline settings."""
    MAX_ROSTER_PLAYERS: int = 12
    DEFAULT_TEAM_STD: float = 12.0
    DEFAULT_PLAYER_POINTS_STD: float = 5.0
    DEFAULT_PLAYER_MINUTES_STD: float = 5.0
    BACK_TO_BACK_PENALTY: float = 3.0     # points
    LONG_REST_BONUS: float = 1.5          # points
    LONG_REST_THRESHOLD_DAYS: int = 3
    LATE_SEASON_THRESHOLD: float = 0.8    # fraction of season
    LATE_SEASON_PENALTY: float = 2.0
    MAX_HOME_ADVANTAGE_CAP: float = 8.0
    MAX_TREND_ADJUSTMENT: float = 5.0
    MIN_CONFIDENCE: float = 0.1
    MAX_CONFIDENCE: float = 0.95


# ── Betting & EV ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BettingConfig:
    """Betting analysis parameters."""
    DEFAULT_ODDS: int = -110
    EV_THRESHOLD_PCT: float = 3.0        # minimum EV% to flag as value
    HIGH_EV_THRESHOLD_PCT: float = 8.0
    CONFIDENCE_THRESHOLD: float = 0.6
    KELLY_FRACTION: float = 0.25         # quarter Kelly for safety
    MAX_KELLY_BET_PCT: float = 5.0       # max 5% of bankroll
    ODDS_REGRESSION_STRENGTH: float = 0.35  # blend toward market
    # Dynamic regression: confident predictions get less regression
    MIN_REGRESSION: float = 0.15
    MAX_REGRESSION: float = 0.55
    SIMULATION_SAMPLES: int = 10_000


# ── Player Props ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PropsConfig:
    """Player props prediction settings."""
    MIN_RECENT_GAMES: int = 5
    RECENT_WEIGHT: float = 0.7
    HOME_BOOST: float = 1.03
    PTS_PER_MIN_BOUNDS: tuple = (0.15, 1.2)
    # Relative std by minutes tier
    STD_LOW_MINUTES: float = 0.45    # < 15 min
    STD_MID_MINUTES: float = 0.37    # 15-25 min
    STD_HIGH_MINUTES: float = 0.28   # 25-32 min
    STD_STAR_MINUTES: float = 0.24   # > 32 min
    MIN_POINTS_STD: float = 3.0


# ── Frontend ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UIConfig:
    """Frontend display settings."""
    ODDS_FILE: str = "draftkings_nba_odds.csv"
    PROPS_FILE: str = "draftkings_player_points_props.csv"
    UPDATE_INTERVAL_HOURS: int = 4
    MAX_PLAYERS_DISPLAY: int = 10
    CACHE_TTL_SECONDS: int = 300  # Streamlit cache TTL


# ── Instantiate singletons ──────────────────────────────────────────────────
LEAGUE = LeagueDefaults()
API = APIConfig()
CACHE = CacheConfig()
TRAINING = TrainingConfig()
PREDICTION = PredictionConfig()
BETTING = BettingConfig()
PROPS = PropsConfig()
UI = UIConfig()
