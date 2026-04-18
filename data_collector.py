import pandas as pd
import numpy as np
import logging
import time
import random
import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from nba_api.stats.endpoints import (
    TeamGameLogs, PlayerGameLogs, CommonPlayerInfo,
    TeamPlayerDashboard,
)
from nba_api.stats.static import teams, players

from config import LEAGUE, API, CACHE

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Context information for a game."""
    rest_days: int
    is_back_to_back: bool
    home_game: bool
    opponent_strength: float
    season_progress: float
    injury_impact: float


class DataCollector:
    """
    Robust NBA data collector with retry logic and smart caching.
    """

    def __init__(self, cache_db_path: str = None):
        self.cache_db_path = cache_db_path or CACHE.DB_PATH
        self._setup_database()
        self._last_request_time = 0.0

    # ── Database Setup ───────────────────────────────────────────────────

    def _setup_database(self):
        """Create cache tables with UNIQUE constraints to prevent duplicates."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS team_games_cache (
                    team_id    INTEGER NOT NULL,
                    season     TEXT    NOT NULL,
                    data       TEXT    NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_id, season)
                );
                CREATE TABLE IF NOT EXISTS player_games_cache (
                    player_id  INTEGER NOT NULL,
                    season     TEXT    NOT NULL,
                    data       TEXT    NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season)
                );
                CREATE TABLE IF NOT EXISTS stats_cache (
                    entity_id   INTEGER NOT NULL,
                    entity_type TEXT    NOT NULL,
                    stat_type   TEXT    NOT NULL,
                    data        TEXT    NOT NULL,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entity_id, entity_type, stat_type)
                );
                CREATE TABLE IF NOT EXISTS roster_cache (
                    team_id    INTEGER NOT NULL,
                    data       TEXT    NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_id)
                );
            """)

    # ── Rate Limiting & Retries ──────────────────────────────────────────

    def _rate_limit(self):
        """Enforce minimum interval between API requests with jitter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < API.MIN_REQUEST_INTERVAL:
            jitter = random.uniform(0, API.MAX_JITTER)
            time.sleep(API.MIN_REQUEST_INTERVAL - elapsed + jitter)
        self._last_request_time = time.time()

    def _get_headers(self) -> dict:
        return {
            "User-Agent": random.choice(API.USER_AGENTS),
            "Referer": "https://stats.nba.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _api_call_with_retry(self, endpoint_factory, description: str = "API call"):
        """
        Call an NBA API endpoint with exponential backoff retries.

        Args:
            endpoint_factory: callable that returns the endpoint object
            description: human-readable label for logging

        Returns:
            First non-empty DataFrame from the endpoint, or None.
        """
        last_error = None
        for attempt in range(1, API.MAX_RETRIES + 1):
            try:
                self._rate_limit()
                endpoint = endpoint_factory()
                dfs = self._safe_get_data_frames(endpoint)
                if dfs and len(dfs) > 0 and not dfs[0].empty:
                    return dfs[0]
                logger.warning(f"{description}: empty response on attempt {attempt}")
                return None  # empty but no error — don't retry
            except Exception as e:
                last_error = e
                if attempt < API.MAX_RETRIES:
                    wait = API.RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1)
                    logger.warning(
                        f"{description}: attempt {attempt} failed ({e}), "
                        f"retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"{description}: all {API.MAX_RETRIES} attempts failed: {last_error}")
        return None

    @staticmethod
    def _safe_get_data_frames(endpoint) -> Optional[List[pd.DataFrame]]:
        """Extract DataFrames from an endpoint, handling both response formats."""
        try:
            dfs = endpoint.get_data_frames()
            if dfs and len(dfs) > 0:
                return dfs
        except Exception:
            pass

        # Fallback: parse raw dict
        try:
            raw = endpoint.get_dict()
            for key in ("resultSets", "resultSet"):
                result = raw.get(key)
                if result is None:
                    continue
                # resultSets is a list, resultSet is a single dict
                items = result if isinstance(result, list) else [result]
                frames = []
                for item in items:
                    if "rowSet" in item and "headers" in item:
                        frames.append(pd.DataFrame(item["rowSet"], columns=item["headers"]))
                if frames:
                    return frames
        except Exception as e:
            logger.warning(f"Raw data extraction failed: {e}")

        return None

    # ── JSON Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _json_default(obj):
        """Serialize numpy/pandas types to JSON-safe Python types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    # ── Team Games ───────────────────────────────────────────────────────

    def get_team_recent_games(
        self, team_id: int, n_games: int = 15, season: str = None
    ) -> Optional[pd.DataFrame]:
        """Fetch team game logs with caching."""
        season = season or LEAGUE.SEASON

        # Check cache
        cached = self._read_cache("team_games_cache", {"team_id": team_id, "season": season},
                                   ttl_hours=CACHE.TEAM_GAMES_TTL_HOURS)
        if cached is not None:
            return cached.head(n_games)

        # Fetch from API
        def factory():
            return TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                headers=self._get_headers(),
            )

        df = self._api_call_with_retry(factory, f"TeamGameLogs(team={team_id}, season={season})")
        if df is not None:
            self._write_cache("team_games_cache", {"team_id": team_id, "season": season}, df)
            return df.head(n_games)
        return None

    # ── Player Games ─────────────────────────────────────────────────────

    def get_player_recent_games(
        self, player_id: int, n_games: int = 15, season: str = None
    ) -> Optional[pd.DataFrame]:
        """Fetch player game logs with caching."""
        season = season or LEAGUE.SEASON

        cached = self._read_cache("player_games_cache", {"player_id": player_id, "season": season},
                                   ttl_hours=CACHE.PLAYER_GAMES_TTL_HOURS)
        if cached is not None:
            return cached.head(n_games)

        def factory():
            return PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=season,
                headers=self._get_headers(),
            )

        df = self._api_call_with_retry(factory, f"PlayerGameLogs(player={player_id})")
        if df is not None:
            self._write_cache("player_games_cache", {"player_id": player_id, "season": season}, df)
            return df.head(n_games)
        return None

    # ── Team Stats (Aggregated) ──────────────────────────────────────────

    def get_team_stats(self, team_id: int, season: str = None) -> Dict:
        """Get comprehensive team statistics with advanced metrics."""
        season = season or LEAGUE.SEASON

        cached = self._read_stats_cache(team_id, "team", "advanced")
        if cached:
            return cached

        games = self.get_team_recent_games(team_id, n_games=82, season=season)
        if games is None or games.empty:
            return {}

        stats = self._calculate_team_metrics(games)
        if stats:
            self._write_stats_cache(team_id, "team", "advanced", stats)
        return stats

    def _calculate_team_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced team metrics from game logs."""
        try:
            num_cols = ["PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM",
                        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            s = {}
            s["games_played"] = len(df)
            s["pts_avg"] = df["PTS"].mean()
            s["pts_std"] = df["PTS"].std()
            s["fg_pct"] = df["FGM"].sum() / max(1, df["FGA"].sum())
            s["fg3_pct"] = df["FG3M"].sum() / max(1, df["FG3A"].sum())
            s["ft_pct"] = df["FTM"].sum() / max(1, df["FTA"].sum())

            # Advanced metrics
            s["efg_pct"] = (df["FGM"].sum() + 0.5 * df["FG3M"].sum()) / max(1, df["FGA"].sum())
            total_pts = df["PTS"].sum()
            total_fga = df["FGA"].sum()
            total_fta = df["FTA"].sum()
            s["ts_pct"] = total_pts / max(1, 2 * (total_fga + 0.44 * total_fta))

            poss = df["FGA"] + 0.44 * df["FTA"] + df["TOV"] - df["OREB"]
            s["pace"] = poss.mean()
            s["off_rating"] = (total_pts / max(1, poss.sum())) * 100
            s["ast_rate"] = df["AST"].sum() / max(1, df["FGM"].sum())
            s["tov_rate"] = df["TOV"].sum() / max(1, poss.sum()) * 100
            s["reb_rate"] = df["REB"].mean()
            s["stl_avg"] = df["STL"].mean()
            s["blk_avg"] = df["BLK"].mean()

            # Recent form (last 10)
            if len(df) >= 10:
                recent = df.head(10)
                s["recent_form"] = {
                    "pts_avg_l10": recent["PTS"].mean(),
                    "fg_pct_l10": recent["FGM"].sum() / max(1, recent["FGA"].sum()),
                    "wins_l10": int((recent.get("WL", pd.Series()) == "W").sum()),
                    "pts_trend": float(np.polyfit(np.arange(10), recent["PTS"].values, 1)[0]),
                }

            # Home/road splits
            if "MATCHUP" in df.columns:
                home = df[df["MATCHUP"].str.contains("vs.", na=False)]
                road = df[df["MATCHUP"].str.contains("@", na=False)]
                if not home.empty and not road.empty:
                    s["home_road_splits"] = {
                        "home_pts_avg": home["PTS"].mean(),
                        "road_pts_avg": road["PTS"].mean(),
                        "home_wins": int((home.get("WL", pd.Series()) == "W").sum()),
                        "road_wins": int((road.get("WL", pd.Series()) == "W").sum()),
                        "home_games": len(home),
                        "road_games": len(road),
                    }

            return s
        except Exception as e:
            logger.error(f"Team metrics calculation error: {e}")
            return {}

    # ── Player Stats ─────────────────────────────────────────────────────

    def get_player_stats(self, player_id: int, season: str = None) -> Dict:
        """Get comprehensive player statistics."""
        season = season or LEAGUE.SEASON

        cached = self._read_stats_cache(player_id, "player", "enhanced")
        if cached:
            return cached

        games = self.get_player_recent_games(player_id, n_games=50, season=season)
        if games is None or games.empty:
            return {}

        stats = self._calculate_player_metrics(games)
        if stats:
            # Attach player context
            stats.update(self._get_player_context(player_id, games))
            self._write_stats_cache(player_id, "player", "enhanced", stats)
        return stats

    def _calculate_player_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced player metrics."""
        try:
            df = df.copy()
            df["MIN"] = df["MIN"].apply(self._convert_minutes)
            num_cols = ["PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM",
                        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "MIN"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            active = df[df["MIN"] > 0]
            if active.empty:
                return {"status": "inactive"}

            s = {
                "games_played": len(active),
                "minutes_avg": active["MIN"].mean(),
                "minutes_std": active["MIN"].std(),
                "pts_avg": active["PTS"].mean(),
                "pts_std": active["PTS"].std(),
                "pts_per_min": active["PTS"].sum() / max(1, active["MIN"].sum()),
                "usage_estimate": self._estimate_usage(active),
            }

            total_fgm = active["FGM"].sum()
            total_fga = active["FGA"].sum()
            total_fg3m = active["FG3M"].sum()
            total_fg3a = active["FG3A"].sum()
            total_ftm = active["FTM"].sum()
            total_fta = active["FTA"].sum()

            s["shooting"] = {
                "fg_pct": total_fgm / max(1, total_fga),
                "fg3_pct": total_fg3m / max(1, total_fg3a),
                "ft_pct": total_ftm / max(1, total_fta),
                "efg_pct": (total_fgm + 0.5 * total_fg3m) / max(1, total_fga),
                "ts_pct": active["PTS"].sum() / max(1, 2 * (total_fga + 0.44 * total_fta)),
            }

            total_min = active["MIN"].sum()
            if total_min > 0:
                s["per_36"] = {
                    "pts": active["PTS"].sum() * 36 / total_min,
                    "reb": active["REB"].sum() * 36 / total_min,
                    "ast": active["AST"].sum() * 36 / total_min,
                    "stl": active["STL"].sum() * 36 / total_min,
                    "blk": active["BLK"].sum() * 36 / total_min,
                    "tov": active["TOV"].sum() * 36 / total_min,
                }

            if len(active) > 1:
                s["consistency"] = {
                    "pts_cv": s["pts_std"] / max(1, s["pts_avg"]),
                    "double_digit_games": int((active["PTS"] >= 10).sum()),
                    "big_games": int((active["PTS"] >= s["pts_avg"] * 1.5).sum()),
                }

            # Recent form
            if len(active) >= 5:
                r5 = active.head(5)
                r10 = active.head(min(10, len(active)))
                s["recent_form"] = {
                    "pts_l5": r5["PTS"].mean(),
                    "pts_l10": r10["PTS"].mean(),
                    "min_l5": r5["MIN"].mean(),
                    "min_l10": r10["MIN"].mean(),
                    "trend_l10": float(np.polyfit(np.arange(len(r10)), r10["PTS"].values, 1)[0])
                    if len(r10) >= 2 else 0.0,
                }

            # Situational: home/away
            if "MATCHUP" in active.columns:
                home_g = active[active["MATCHUP"].str.contains("vs.", na=False)]
                away_g = active[active["MATCHUP"].str.contains("@", na=False)]
                if not home_g.empty and not away_g.empty:
                    s["situational"] = {
                        "home_away": {
                            "home_pts_avg": home_g["PTS"].mean(),
                            "away_pts_avg": away_g["PTS"].mean(),
                            "home_advantage": home_g["PTS"].mean() - away_g["PTS"].mean(),
                        }
                    }

            return s
        except Exception as e:
            logger.error(f"Player metrics error: {e}")
            return {}

    def _estimate_usage(self, df: pd.DataFrame) -> float:
        """Estimate player usage rate."""
        try:
            poss_per_min = 100 / 48
            usages = []
            for _, g in df.iterrows():
                if g["MIN"] > 0:
                    player_poss = g["FGA"] + 0.44 * g["FTA"] + g["TOV"]
                    team_poss_est = poss_per_min * g["MIN"]
                    usages.append(min(1.0, player_poss / max(1, team_poss_est)))
            return float(np.mean(usages)) if usages else 0.2
        except Exception:
            return 0.2

    def _get_player_context(self, player_id: int, games_df: pd.DataFrame) -> Dict:
        """Get player metadata (position, role, etc.)."""
        ctx = {}
        try:
            def factory():
                return CommonPlayerInfo(player_id=player_id, headers=self._get_headers())

            self._rate_limit()
            endpoint = factory()
            dfs = self._safe_get_data_frames(endpoint)
            if dfs and not dfs[0].empty:
                info = dfs[0]
                ctx["position"] = info.get("POSITION", pd.Series()).iloc[0] if "POSITION" in info.columns else None
                ctx["experience"] = info.get("SEASON_EXP", pd.Series()).iloc[0] if "SEASON_EXP" in info.columns else None
        except Exception as e:
            logger.warning(f"Player info lookup failed for {player_id}: {e}")

        avg_min = self._convert_minutes_series(games_df.get("MIN", pd.Series())).mean()
        if avg_min >= 32:
            ctx["role"] = "star"
        elif avg_min >= 24:
            ctx["role"] = "starter"
        elif avg_min >= 15:
            ctx["role"] = "rotation"
        else:
            ctx["role"] = "bench"
        return ctx

    # ── Real Roster Fetching ─────────────────────────────────────────────

    def get_team_players(self, team_id: int, season: str = None) -> List[Dict]:
        """
        Get the ACTUAL current roster for a team via TeamPlayerDashboard.
        Falls back to static player list only as a last resort.
        """
        season = season or LEAGUE.SEASON

        # Check roster cache (short TTL since rosters change with trades)
        cached = self._read_roster_cache(team_id)
        if cached:
            return cached

        try:
            def factory():
                return TeamPlayerDashboard(
                    team_id=team_id,
                    season=season,
                    headers=self._get_headers(),
                )

            self._rate_limit()
            endpoint = factory()
            dfs = self._safe_get_data_frames(endpoint)

            # TeamPlayerDashboard returns [OverallTeamDashboard, PlayersSeasonTotals]
            if dfs and len(dfs) >= 2:
                players_df = dfs[1]
                if not players_df.empty:
                    # Filter to players with actual games played
                    if "GP" in players_df.columns:
                        players_df = players_df[players_df["GP"] > 0]

                    roster = []
                    for _, row in players_df.iterrows():
                        roster.append({
                            "id": int(row.get("PLAYER_ID", 0)),
                            "full_name": str(row.get("PLAYER_NAME", "Unknown")),
                            "gp": int(row.get("GP", 0)),
                            "min_avg": float(row.get("MIN", 0)) if "MIN" in row.index else 0.0,
                            "pts_avg": float(row.get("PTS", 0)) if "PTS" in row.index else 0.0,
                        })

                    # Sort by minutes (most important players first)
                    roster.sort(key=lambda x: x["min_avg"], reverse=True)
                    self._write_roster_cache(team_id, roster)
                    return roster
        except Exception as e:
            logger.warning(f"TeamPlayerDashboard failed for team {team_id}: {e}")

        # Fallback: return empty list rather than random players
        logger.warning(f"Could not fetch roster for team {team_id}")
        return []

    # ── Game Context ─────────────────────────────────────────────────────

    def get_game_context(self, team_id: int, game_date: str = None) -> GameContext:
        """Get contextual information for a team's next game."""
        try:
            recent = self.get_team_recent_games(team_id, n_games=5)
            if recent is None or len(recent) < 2:
                return GameContext(1, False, True, 0.5, self._season_progress(), 0.0)

            dates = pd.to_datetime(recent["GAME_DATE"])
            rest = (dates.iloc[0] - dates.iloc[1]).days
            is_b2b = rest <= 1
            home = "vs." in str(recent["MATCHUP"].iloc[0])

            return GameContext(
                rest_days=rest,
                is_back_to_back=is_b2b,
                home_game=home,
                opponent_strength=0.5,
                season_progress=self._season_progress(),
                injury_impact=0.0,
            )
        except Exception as e:
            logger.error(f"Game context error: {e}")
            return GameContext(1, False, True, 0.5, 0.5, 0.0)

    @staticmethod
    def _season_progress() -> float:
        """Calculate fraction of NBA season completed (0–1)."""
        now = datetime.now()
        if now.month >= 10:
            start = datetime(now.year, 10, 15)
            end = datetime(now.year + 1, 4, 15)
        else:
            start = datetime(now.year - 1, 10, 15)
            end = datetime(now.year, 4, 15)
        return max(0.0, min(1.0, (now - start).days / max(1, (end - start).days)))

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _convert_minutes(val) -> float:
        """Convert minute strings ('32:15') or numbers to float."""
        if pd.isna(val) or val == "":
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and ":" in val:
            parts = val.split(":")
            return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @classmethod
    def _convert_minutes_series(cls, series: pd.Series) -> pd.Series:
        return series.apply(cls._convert_minutes)

    @staticmethod
    def get_team_abbreviation(team_id: int) -> Optional[str]:
        info = next((t for t in teams.get_teams() if t["id"] == team_id), None)
        return info["abbreviation"] if info else None

    @staticmethod
    def get_team_name(team_id: int) -> str:
        info = next((t for t in teams.get_teams() if t["id"] == team_id), None)
        return info["full_name"] if info else f"Team {team_id}"

    # ── Generic Cache Helpers ────────────────────────────────────────────

    def _read_cache(self, table: str, keys: dict, ttl_hours: int) -> Optional[pd.DataFrame]:
        """Read a cached DataFrame if it exists and is fresh."""
        try:
            where = " AND ".join(f"{k} = ?" for k in keys)
            params = list(keys.values())
            with sqlite3.connect(self.cache_db_path) as conn:
                row = conn.execute(
                    f"SELECT data, updated_at FROM {table} WHERE {where} "
                    f"ORDER BY updated_at DESC LIMIT 1",
                    params,
                ).fetchone()
            if row:
                updated = datetime.fromisoformat(row[1])
                if datetime.now() - updated < timedelta(hours=ttl_hours):
                    return pd.DataFrame(json.loads(row[0]))
        except Exception as e:
            logger.debug(f"Cache read miss ({table}): {e}")
        return None

    def _write_cache(self, table: str, keys: dict, df: pd.DataFrame):
        """Upsert a DataFrame into the cache."""
        try:
            cols = list(keys.keys()) + ["data", "updated_at"]
            placeholders = ", ".join("?" for _ in cols)
            conflict_cols = ", ".join(keys.keys())
            upsert_sql = (
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT({conflict_cols}) DO UPDATE SET "
                f"data=excluded.data, updated_at=excluded.updated_at"
            )
            data_json = json.dumps(df.to_dict("records"), default=self._json_default)
            vals = list(keys.values()) + [data_json, datetime.now().isoformat()]
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(upsert_sql, vals)
        except Exception as e:
            logger.warning(f"Cache write failed ({table}): {e}")

    def _read_stats_cache(self, entity_id: int, entity_type: str, stat_type: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                row = conn.execute(
                    "SELECT data, updated_at FROM stats_cache "
                    "WHERE entity_id=? AND entity_type=? AND stat_type=? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (entity_id, entity_type, stat_type),
                ).fetchone()
            if row:
                updated = datetime.fromisoformat(row[1])
                if datetime.now() - updated < timedelta(hours=CACHE.ADVANCED_STATS_TTL_HOURS):
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def _write_stats_cache(self, entity_id: int, entity_type: str, stat_type: str, stats: Dict):
        try:
            data_json = json.dumps(stats, default=self._json_default)
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    "INSERT INTO stats_cache (entity_id, entity_type, stat_type, data, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(entity_id, entity_type, stat_type) DO UPDATE SET "
                    "data=excluded.data, updated_at=excluded.updated_at",
                    (entity_id, entity_type, stat_type, data_json, datetime.now().isoformat()),
                )
        except Exception as e:
            logger.warning(f"Stats cache write failed: {e}")

    def _read_roster_cache(self, team_id: int) -> Optional[List[Dict]]:
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                row = conn.execute(
                    "SELECT data, updated_at FROM roster_cache WHERE team_id=?",
                    (team_id,),
                ).fetchone()
            if row:
                updated = datetime.fromisoformat(row[1])
                # Rosters change infrequently — cache for 24h
                if datetime.now() - updated < timedelta(hours=24):
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def _write_roster_cache(self, team_id: int, roster: List[Dict]):
        try:
            data_json = json.dumps(roster, default=self._json_default)
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    "INSERT INTO roster_cache (team_id, data, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(team_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                    (team_id, data_json, datetime.now().isoformat()),
                )
        except Exception as e:
            logger.warning(f"Roster cache write failed: {e}")

    def cleanup_cache(self, days_old: int = None):
        """Remove stale cache entries."""
        days_old = days_old or CACHE.CLEANUP_AFTER_DAYS
        try:
            cutoff = f"-{days_old} days"
            with sqlite3.connect(self.cache_db_path) as conn:
                for table in ("team_games_cache", "player_games_cache", "stats_cache", "roster_cache"):
                    conn.execute(f"DELETE FROM {table} WHERE updated_at < datetime('now', ?)", (cutoff,))
            logger.info(f"Cleaned cache entries older than {days_old} days")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
