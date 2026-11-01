"""
Ingestion: keep the warehouse in sync with the providers, with fallback and resumability.

  backfill(seasons)  one-time historical load (safe to interrupt and re-run: it only fetches what's missing)
  update()           daily refresh: recent results + upcoming schedule + injury report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Sequence

from config import DATA, recent_seasons
from nbadata.http import HttpClient, ProviderError
from nbadata.providers.base import BoxScore, GameDataProvider, GameStub
from nbadata.timeutil import today_eastern
from nbadata.warehouse import Warehouse

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    games_seen: int = 0
    boxes_saved: int = 0
    boxes_failed: int = 0
    boxes_remaining: int = 0
    provider_errors: Dict[str, int] = field(default_factory=dict)
    empty_providers: List[str] = field(default_factory=list)
    injuries: int = 0
    messages: List[str] = field(default_factory=list)

    def note(self, msg: str):
        logger.info(msg)
        self.messages.append(msg)


def default_providers(http: Optional[HttpClient] = None) -> List[GameDataProvider]:
    from nbadata.providers.espn import ESPNProvider
    from nbadata.providers.nba_cdn import NBACDNProvider
    registry = {"espn": ESPNProvider, "nba_cdn": NBACDNProvider}
    return [registry[n](http) if http else registry[n]() for n in DATA.PROVIDERS if n in registry]


def season_bounds(season: str) -> tuple:
    start_year = int(season[:4])
    return date(start_year, 10, 1), date(start_year + 1, 6, 30)


class Ingestor:
    def __init__(self, warehouse: Warehouse, providers: Optional[Sequence[GameDataProvider]] = None):
        self.wh = warehouse
        self.providers = list(providers) if providers is not None else default_providers()

    # ── schedule / scores ────────────────────────────────────────────────
    def sync_games(self, start: date, end: date, report: Optional[IngestReport] = None) -> int:
        report = report or IngestReport()
        for prov in self.providers:
            try:
                stubs = prov.list_games(start, end)
            except ProviderError as e:
                report.provider_errors[prov.name] = report.provider_errors.get(prov.name, 0) + 1
                report.note(f"{prov.name}: could not list games {start}..{end} ({e})")
                continue
            if stubs:
                n = self.wh.upsert_games([s.as_row() for s in stubs])
                report.games_seen += n
                return n
            # Empty can be legitimate (off-season) or a silent schema change; asking the next source is cheap either way.
            if prov.name not in report.empty_providers:
                report.empty_providers.append(prov.name)
        return 0

    # ── box scores ───────────────────────────────────────────────────────
    def _fetch_box_with_fallback(self, row: dict, report: IngestReport) -> Optional[BoxScore]:
        stub = GameStub(**{k: row[k] for k in GameStub.__dataclass_fields__ if k in row})
        for prov in self.providers:
            try:
                s = stub
                if not prov.supports(s):
                    # We only know another provider's id: look this game up by date to get ours.
                    d = date.fromisoformat(stub.game_date)
                    match = [g for g in prov.list_games(d, d) if g.game_key == stub.game_key]
                    if not match:
                        continue
                    s = GameStub(**{**match[0].__dict__, "status": "final",
                                    "home_pts": stub.home_pts, "away_pts": stub.away_pts})
                    self.wh.upsert_games([match[0].as_row()])   # remember the id for next time
                box = prov.fetch_box(s)
                if box is None:
                    continue
                problems = box.validate()
                if problems:
                    report.note(f"{prov.name}: {stub.game_key} box score failed validation ({'; '.join(problems)}); trying next source")
                    continue
                return box
            except ProviderError as e:
                report.provider_errors[prov.name] = report.provider_errors.get(prov.name, 0) + 1
                logger.warning("%s: box score for %s failed: %s", prov.name, stub.game_key, e)
        return None

    def sync_boxscores(self, limit: Optional[int] = None, before: Optional[str] = None,
                       progress: Optional[Callable[[int, int], None]] = None,
                       report: Optional[IngestReport] = None) -> IngestReport:
        report = report or IngestReport()
        todo = self.wh.games_missing_box(before=before, limit=limit)
        total = len(todo)
        consecutive_failures = 0
        for i, (_, row) in enumerate(todo.iterrows(), start=1):
            box = self._fetch_box_with_fallback(row.to_dict(), report)
            if box is None:
                report.boxes_failed += 1
                consecutive_failures += 1
                if consecutive_failures >= 12:
                    report.note("Stopping early: 12 consecutive failures. Run `python cli.py doctor` to see which source is down; "
                                "re-running later resumes where this left off.")
                    break
            else:
                self.wh.save_boxscore(box.game.as_row(), box.team_rows(), box.player_rows())
                report.boxes_saved += 1
                consecutive_failures = 0
            if progress:
                progress(i, total)
        report.boxes_remaining = len(self.wh.games_missing_box())
        return report

    # ── injuries ─────────────────────────────────────────────────────────
    def sync_injuries(self, report: Optional[IngestReport] = None) -> int:
        report = report or IngestReport()
        for prov in self.providers:
            fetch = getattr(prov, "fetch_injuries", None)
            if not fetch:
                continue
            try:
                rows = fetch()
                self.wh.replace_injuries(rows)
                report.note(f"injuries: {len(rows)} entries from {prov.name}")
                return len(rows)
            except ProviderError as e:
                report.note(f"{prov.name}: injury report unavailable ({e})")
        return 0

    # ── high-level jobs ──────────────────────────────────────────────────
    def backfill(self, seasons: Optional[Sequence[str]] = None, progress=None) -> IngestReport:
        report = IngestReport()
        today = today_eastern()
        seasons = seasons or recent_seasons(DATA.BACKFILL_SEASONS, today)
        for season in seasons:
            start, end = season_bounds(season)
            if start > today:
                continue
            end = min(end, today)
            # Chunk by month so an interrupted run still leaves usable data behind
            d = start
            while d <= end:
                chunk_end = min(d + timedelta(days=30), end)
                self.sync_games(d, chunk_end, report)
                d = chunk_end + timedelta(days=1)
            report.note(f"{season}: schedule synced")
        self.sync_boxscores(progress=progress, report=report)
        return report

    def update(self, days_back: Optional[int] = None, days_ahead: int = 7, progress=None) -> IngestReport:
        report = IngestReport()
        today = today_eastern()
        days_back = DATA.UPDATE_DAYS_BACK if days_back is None else days_back
        last = self.wh.counts().get("last_date")
        start = today - timedelta(days=days_back)
        if last:
            start = min(start, date.fromisoformat(last) - timedelta(days=1))
        self.sync_games(start, today + timedelta(days=days_ahead), report)
        self.sync_boxscores(progress=progress, report=report)
        report.injuries = self.sync_injuries(report)
        self.wh.set_meta("last_update", today_eastern().isoformat())
        return report
