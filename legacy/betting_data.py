"""
Betting data service — loads DraftKings CSV odds and player props,
provides clean lookup methods.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Optional
from pathlib import Path

from config import UI

logger = logging.getLogger(__name__)

# ── Team Name Mapping ────────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    "Atlanta Hawks": "ATL Hawks",
    "Boston Celtics": "BOS Celtics",
    "Brooklyn Nets": "BKN Nets",
    "Charlotte Hornets": "CHA Hornets",
    "Chicago Bulls": "CHI Bulls",
    "Cleveland Cavaliers": "CLE Cavaliers",
    "Dallas Mavericks": "DAL Mavericks",
    "Denver Nuggets": "DEN Nuggets",
    "Detroit Pistons": "DET Pistons",
    "Golden State Warriors": "GS Warriors",
    "Houston Rockets": "HOU Rockets",
    "Indiana Pacers": "IND Pacers",
    "Los Angeles Clippers": "LA Clippers",
    "Los Angeles Lakers": "LA Lakers",
    "Memphis Grizzlies": "MEM Grizzlies",
    "Miami Heat": "MIA Heat",
    "Milwaukee Bucks": "MIL Bucks",
    "Minnesota Timberwolves": "MIN Timberwolves",
    "New Orleans Pelicans": "NO Pelicans",
    "New York Knicks": "NY Knicks",
    "Oklahoma City Thunder": "OKC Thunder",
    "Orlando Magic": "ORL Magic",
    "Philadelphia 76ers": "PHI 76ers",
    "Phoenix Suns": "PHO Suns",
    "Portland Trail Blazers": "POR Trail Blazers",
    "Sacramento Kings": "SAC Kings",
    "San Antonio Spurs": "SA Spurs",
    "Toronto Raptors": "TOR Raptors",
    "Utah Jazz": "UTA Jazz",
    "Washington Wizards": "WAS Wizards",
}

# Build reverse map too (abbreviated → full)
_REVERSE_MAP = {v: k for k, v in TEAM_NAME_MAP.items()}


def _to_abbrev(name: str) -> str:
    """Normalize a team name to the abbreviated DraftKings format."""
    return TEAM_NAME_MAP.get(name, name)


def _parse_odds(s) -> int:
    """Parse an odds string like '-110' or '+250' to int."""
    if pd.isna(s):
        return -110
    s = str(s).replace("\u2212", "-").replace("−", "-").replace("+", "").strip()
    try:
        return int(s)
    except ValueError:
        return -110


def _parse_spread(s) -> Optional[float]:
    """Parse a spread string like '+7.5 (-110)' → 7.5"""
    if pd.isna(s):
        return None
    m = re.search(r"([+-]?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else None


def _parse_spread_odds(s) -> int:
    """Parse odds from a spread string like '+7.5 (-110)' → -110"""
    if pd.isna(s):
        return -110
    m = re.search(r"\(([+-]?\d+)\)", str(s).replace("\u2212", "-").replace("−", "-"))
    return int(m.group(1)) if m else -110


def _parse_total(s) -> Optional[float]:
    """Parse total from string like '206.5 (O: -110, U: -110)' → 206.5"""
    if pd.isna(s):
        return None
    m = re.search(r"(\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else None


def _parse_total_odds(s, side: str) -> int:
    """Parse over or under odds from total string."""
    if pd.isna(s):
        return -110
    s = str(s).replace("\u2212", "-").replace("−", "-")
    pattern = f"{side[0].upper()}:\\s*([+-]?\\d+)"
    m = re.search(pattern, s)
    return int(m.group(1)) if m else -110


def _parse_moneyline(s) -> Optional[int]:
    if pd.isna(s):
        return None
    s = str(s).replace("\u2212", "-").replace("−", "-").replace("+", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


class BettingDataService:
    """Load and query DraftKings game odds."""

    def __init__(self, odds_path: str = None):
        self.odds_path = odds_path or UI.ODDS_FILE
        self._odds_df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        path = Path(self.odds_path)
        if not path.exists():
            logger.warning(f"Odds file not found: {path}")
            return
        try:
            self._odds_df = pd.read_csv(path)
            logger.info(f"Loaded {len(self._odds_df)} game odds from {path}")
        except Exception as e:
            logger.error(f"Failed to load odds: {e}")

    def get_betting_lines(self, home_team: str, away_team: str) -> Dict:
        """
        Look up betting lines for a matchup.

        Returns a dict with keys: available, home_spread, away_spread,
        home_spread_odds, away_spread_odds, total, total_over_odds,
        total_under_odds, home_moneyline, away_moneyline.
        """
        if self._odds_df is None or self._odds_df.empty:
            return {"available": False}

        home_abbrev = _to_abbrev(home_team)
        away_abbrev = _to_abbrev(away_team)

        # Try exact match
        mask = (
            self._odds_df["Home Team"].str.contains(home_abbrev.split()[0], case=False, na=False)
            & self._odds_df["Away Team"].str.contains(away_abbrev.split()[0], case=False, na=False)
        )
        rows = self._odds_df[mask]
        if rows.empty:
            return {"available": False}

        row = rows.iloc[0]
        return {
            "available": True,
            "home_spread": _parse_spread(row.get("Home Spread")),
            "away_spread": _parse_spread(row.get("Away Spread")),
            "home_spread_odds": _parse_spread_odds(row.get("Home Spread")),
            "away_spread_odds": _parse_spread_odds(row.get("Away Spread")),
            "total": _parse_total(row.get("Total")),
            "total_over_odds": _parse_total_odds(row.get("Total"), "O"),
            "total_under_odds": _parse_total_odds(row.get("Total"), "U"),
            "home_moneyline": _parse_moneyline(row.get("Home Moneyline")),
            "away_moneyline": _parse_moneyline(row.get("Away Moneyline")),
        }


class PlayerPropsService:
    """Load and query DraftKings player points props."""

    def __init__(self, props_path: str = None):
        self.props_path = props_path or UI.PROPS_FILE
        self._props_df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        path = Path(self.props_path)
        if not path.exists():
            logger.warning(f"Props file not found: {path}")
            return
        try:
            df = pd.read_csv(path)
            df["Points Line"] = pd.to_numeric(df["Points Line"], errors="coerce")
            df["Over Odds"] = df["Over Odds"].apply(_parse_odds)
            df["Under Odds"] = df["Under Odds"].apply(_parse_odds)
            self._props_df = df
            logger.info(f"Loaded {len(df)} player props from {path}")
        except Exception as e:
            logger.error(f"Failed to load props: {e}")

    def get_player_props(self, home_team: str, away_team: str) -> Dict:
        """
        Get all player props for a matchup.

        Returns {player_name: {line, over_odds, under_odds}} or empty dict.
        """
        if self._props_df is None or self._props_df.empty:
            return {}

        home_abbrev = _to_abbrev(home_team)
        away_abbrev = _to_abbrev(away_team)

        # Match game column
        mask = self._props_df["Game"].apply(
            lambda g: (away_abbrev.split()[0] in str(g)) and (home_abbrev.split()[0] in str(g))
        )
        game_props = self._props_df[mask]
        if game_props.empty:
            return {}

        result = {}
        for _, row in game_props.iterrows():
            name = row["Player"]
            result[name] = {
                "line": float(row["Points Line"]),
                "over_odds": int(row["Over Odds"]),
                "under_odds": int(row["Under Odds"]),
            }
        return result
