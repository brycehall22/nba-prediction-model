"""
SQLite warehouse: games, team/player box scores, injuries, odds snapshots, prediction log.

One game = one row keyed by `game_key` = '<YYYYMMDD>_<AWAY>_<HOME>' (US/Eastern date,
NBA tricodes). The key is provider-independent, so ESPN and NBA-CDN data for the same
game collapse into one row and odds can be joined without fragile id mapping.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config import DATA
from nbadata.teams import team_abbr
from nbadata.timeutil import utcnow_naive

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);

CREATE TABLE IF NOT EXISTS games (
    game_key     TEXT PRIMARY KEY,
    game_date    TEXT NOT NULL,
    season       TEXT NOT NULL,
    season_type  INTEGER,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    tipoff_utc   TEXT,
    status       TEXT NOT NULL,
    home_pts     INTEGER,
    away_pts     INTEGER,
    espn_id      TEXT,
    nba_id       TEXT,
    has_box      INTEGER NOT NULL DEFAULT 0,
    source       TEXT,
    updated_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);

CREATE TABLE IF NOT EXISTS team_box (
    game_key TEXT NOT NULL, team_id INTEGER NOT NULL, is_home INTEGER NOT NULL,
    pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER, fta INTEGER,
    oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, tov INTEGER, stl INTEGER, blk INTEGER, pf INTEGER,
    PRIMARY KEY (game_key, team_id)
);

CREATE TABLE IF NOT EXISTS player_box (
    game_key TEXT NOT NULL, team_id INTEGER NOT NULL,
    player_key TEXT NOT NULL,          -- normalised name; stable across providers
    player_name TEXT, player_id TEXT,  -- provider id, informational only
    starter INTEGER, played INTEGER, minutes REAL,
    pts INTEGER, reb INTEGER, ast INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, tov INTEGER, stl INTEGER, blk INTEGER,
    pf INTEGER, plus_minus INTEGER, dnp_reason TEXT,
    PRIMARY KEY (game_key, player_key)
);
CREATE INDEX IF NOT EXISTS idx_pbox_player ON player_box(player_key);

CREATE TABLE IF NOT EXISTS injuries (
    team_id INTEGER NOT NULL, player_key TEXT NOT NULL, player_name TEXT,
    status TEXT, detail TEXT, fetched_at TEXT,
    PRIMARY KEY (team_id, player_key)
);

CREATE TABLE IF NOT EXISTS game_odds (
    fetched_at TEXT NOT NULL, game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL, away_team_id INTEGER NOT NULL, commence_utc TEXT,
    book TEXT NOT NULL, source TEXT,
    spread_home REAL, spread_home_price INTEGER, spread_away_price INTEGER,
    total REAL, over_price INTEGER, under_price INTEGER,
    ml_home INTEGER, ml_away INTEGER
);
CREATE INDEX IF NOT EXISTS idx_godds ON game_odds(game_date, home_team_id, away_team_id, book, fetched_at);

CREATE TABLE IF NOT EXISTS prop_odds (
    fetched_at TEXT NOT NULL, game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL, away_team_id INTEGER NOT NULL, commence_utc TEXT,
    book TEXT NOT NULL, source TEXT, market TEXT NOT NULL,
    player_name TEXT, player_key TEXT NOT NULL,
    line REAL, over_price INTEGER, under_price INTEGER
);
CREATE INDEX IF NOT EXISTS idx_podds ON prop_odds(game_date, home_team_id, away_team_id, market, player_key, book, fetched_at);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, game_key TEXT, game_date TEXT,
    kind TEXT NOT NULL,                 -- spread | total | moneyline | prop
    side TEXT NOT NULL,                 -- HOME/AWAY/OVER/UNDER
    player_key TEXT, player_name TEXT,
    line REAL, price INTEGER, book TEXT,
    model_prob REAL, market_prob REAL, blended_prob REAL, ev_pct REAL, kelly REAL,
    projection REAL,
    result TEXT,                        -- win | loss | push | NULL (pending)
    actual REAL, closing_line REAL, closing_price INTEGER, graded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(game_date, result);
"""


class Warehouse:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DATA.DB_PATH
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._mem_conn: Optional[sqlite3.Connection] = None
        self.init_schema()

    # ── connection handling ──────────────────────────────────────────────
    @contextmanager
    def connect(self):
        if self.path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.row_factory = sqlite3.Row
            yield self._mem_conn
            self._mem_conn.commit()
            return
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self):
        with self.connect() as c:
            c.executescript(SCHEMA)

    # ── meta ─────────────────────────────────────────────────────────────
    def set_meta(self, key: str, value) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT INTO meta(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, str(value), utcnow_naive().isoformat(timespec="seconds")),
            )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.connect() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def get_meta_time(self, key: str) -> Optional[datetime]:
        with self.connect() as c:
            row = c.execute("SELECT updated_at FROM meta WHERE key=?", (key,)).fetchone()
        return datetime.fromisoformat(row["updated_at"]) if row and row["updated_at"] else None

    # ── writes ───────────────────────────────────────────────────────────
    def upsert_games(self, games: Sequence[dict]) -> int:
        """Insert/update games. Never downgrade a final game with box data to a scheduled stub."""
        n = 0
        now = utcnow_naive().isoformat(timespec="seconds")
        with self.connect() as c:
            for g in games:
                existing = c.execute(
                    "SELECT status, has_box, espn_id, nba_id FROM games WHERE game_key=?", (g["game_key"],)
                ).fetchone()
                if existing and existing["status"] == "final" and g["status"] != "final":
                    # keep the final record, but we may still learn a provider id
                    c.execute(
                        "UPDATE games SET espn_id=COALESCE(?,espn_id), nba_id=COALESCE(?,nba_id) WHERE game_key=?",
                        (g.get("espn_id"), g.get("nba_id"), g["game_key"]),
                    )
                    continue
                c.execute(
                    """INSERT INTO games(game_key, game_date, season, season_type, home_team_id, away_team_id,
                           tipoff_utc, status, home_pts, away_pts, espn_id, nba_id, has_box, source, updated_at)
                       VALUES(:game_key,:game_date,:season,:season_type,:home_team_id,:away_team_id,
                           :tipoff_utc,:status,:home_pts,:away_pts,:espn_id,:nba_id,0,:source,:now)
                       ON CONFLICT(game_key) DO UPDATE SET
                           season_type=COALESCE(excluded.season_type, games.season_type),
                           tipoff_utc=COALESCE(excluded.tipoff_utc, games.tipoff_utc),
                           status=excluded.status,
                           home_pts=COALESCE(excluded.home_pts, games.home_pts),
                           away_pts=COALESCE(excluded.away_pts, games.away_pts),
                           espn_id=COALESCE(excluded.espn_id, games.espn_id),
                           nba_id=COALESCE(excluded.nba_id, games.nba_id),
                           source=excluded.source, updated_at=excluded.updated_at""",
                    {**{k: g.get(k) for k in (
                        "game_key", "game_date", "season", "season_type", "home_team_id", "away_team_id",
                        "tipoff_utc", "status", "home_pts", "away_pts", "espn_id", "nba_id", "source")},
                     "now": now},
                )
                n += 1
        return n

    def save_boxscore(self, game: dict, team_rows: Sequence[dict], player_rows: Sequence[dict]) -> None:
        """Atomically replace a game's box score and mark it complete."""
        with self.connect() as c:
            c.execute("DELETE FROM team_box WHERE game_key=?", (game["game_key"],))
            c.execute("DELETE FROM player_box WHERE game_key=?", (game["game_key"],))
            tcols = ["game_key", "team_id", "is_home", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                     "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pf"]
            c.executemany(
                f"INSERT INTO team_box({','.join(tcols)}) VALUES({','.join('?' * len(tcols))})",
                [tuple(r.get(k) for k in tcols) for r in team_rows],
            )
            pcols = ["game_key", "team_id", "player_key", "player_name", "player_id", "starter", "played", "minutes",
                     "pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "stl",
                     "blk", "pf", "plus_minus", "dnp_reason"]
            c.executemany(
                f"INSERT OR REPLACE INTO player_box({','.join(pcols)}) VALUES({','.join('?' * len(pcols))})",
                [tuple(r.get(k) for k in pcols) for r in player_rows],
            )
            c.execute(
                "UPDATE games SET has_box=1, status='final', home_pts=COALESCE(?,home_pts), away_pts=COALESCE(?,away_pts), "
                "updated_at=? WHERE game_key=?",
                (game.get("home_pts"), game.get("away_pts"), utcnow_naive().isoformat(timespec="seconds"),
                 game["game_key"]),
            )

    def replace_injuries(self, rows: Sequence[dict]) -> None:
        now = utcnow_naive().isoformat(timespec="seconds")
        with self.connect() as c:
            c.execute("DELETE FROM injuries")
            c.executemany(
                "INSERT OR REPLACE INTO injuries(team_id, player_key, player_name, status, detail, fetched_at) "
                "VALUES(?,?,?,?,?,?)",
                [(r["team_id"], r["player_key"], r.get("player_name"), r.get("status"), r.get("detail"), now)
                 for r in rows],
            )

    # ── reads ────────────────────────────────────────────────────────────
    def query_df(self, sql: str, params: Sequence = ()) -> pd.DataFrame:
        with self.connect() as c:
            return pd.read_sql_query(sql, c, params=list(params))

    def games_df(self, start: Optional[str] = None, end: Optional[str] = None,
                 status: Optional[str] = None, with_box: Optional[bool] = None) -> pd.DataFrame:
        sql, cond, params = "SELECT * FROM games", [], []
        if start:
            cond.append("game_date >= ?"); params.append(str(start))
        if end:
            cond.append("game_date <= ?"); params.append(str(end))
        if status:
            cond.append("status = ?"); params.append(status)
        if with_box is not None:
            cond.append("has_box = ?"); params.append(1 if with_box else 0)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY game_date, game_key"
        return self.query_df(sql, params)

    def team_box_df(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        sql = ("SELECT t.*, g.game_date, g.season, g.season_type FROM team_box t "
               "JOIN games g ON g.game_key=t.game_key WHERE g.has_box=1")
        params: List = []
        if start:
            sql += " AND g.game_date >= ?"; params.append(str(start))
        if end:
            sql += " AND g.game_date <= ?"; params.append(str(end))
        return self.query_df(sql + " ORDER BY g.game_date, t.game_key", params)

    def player_box_df(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        sql = ("SELECT p.*, g.game_date, g.season, g.season_type, g.home_team_id, g.away_team_id FROM player_box p "
               "JOIN games g ON g.game_key=p.game_key WHERE g.has_box=1")
        params: List = []
        if start:
            sql += " AND g.game_date >= ?"; params.append(str(start))
        if end:
            sql += " AND g.game_date <= ?"; params.append(str(end))
        return self.query_df(sql + " ORDER BY g.game_date, p.game_key", params)

    def games_missing_box(self, before: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        sql = "SELECT * FROM games WHERE status='final' AND has_box=0"
        params: List = []
        if before:
            sql += " AND game_date <= ?"; params.append(before)
        sql += " ORDER BY game_date DESC"      # newest first: most useful data arrives first
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query_df(sql, params)

    def injuries_df(self) -> pd.DataFrame:
        return self.query_df("SELECT * FROM injuries")

    def season_counts(self) -> pd.DataFrame:
        return self.query_df(
            "SELECT season, COUNT(*) games, SUM(CASE WHEN status='final' THEN 1 ELSE 0 END) final, "
            "SUM(has_box) with_box, MIN(game_date) first_date, MAX(CASE WHEN has_box=1 THEN game_date END) last_box "
            "FROM games WHERE IFNULL(season_type,2) != 1 GROUP BY season ORDER BY season")

    def counts(self) -> Dict[str, object]:
        with self.connect() as c:
            out = {
                "games": c.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                "games_final": c.execute("SELECT COUNT(*) FROM games WHERE status='final'").fetchone()[0],
                "games_with_box": c.execute("SELECT COUNT(*) FROM games WHERE has_box=1").fetchone()[0],
                "player_rows": c.execute("SELECT COUNT(*) FROM player_box").fetchone()[0],
                "first_date": c.execute("SELECT MIN(game_date) FROM games WHERE has_box=1").fetchone()[0],
                "last_date": c.execute("SELECT MAX(game_date) FROM games WHERE has_box=1").fetchone()[0],
                "odds_snapshots": c.execute("SELECT COUNT(DISTINCT fetched_at) FROM game_odds").fetchone()[0],
                "predictions_logged": c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            }
        return out


def make_game_key(game_date, away_team_id: int, home_team_id: int) -> str:
    d = game_date if isinstance(game_date, str) else game_date.isoformat()
    return f"{d.replace('-', '')}_{team_abbr(away_team_id)}_{team_abbr(home_team_id)}"
