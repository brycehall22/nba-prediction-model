"""
The Odds API client (https://the-odds-api.com): the primary, supported odds source.

It replaces the Selenium/HTML DraftKings scraper. DraftKings' page markup changes
constantly and is bot-protected; this is a documented JSON API that returns
DraftKings (and every other US book) in one stable schema.

Quota: one call to /odds with 3 markets costs 3 credits; each player-props call costs
1 credit per market per event. The free plan's 500 credits/month covers roughly one
refresh a day. The client reads the `x-requests-remaining` header so the app can show
what's left, and the service refuses to re-pull more often than OddsConfig.MIN_REFRESH_MINUTES.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from config import ODDS
from nbadata.http import HttpClient, ProviderError
from nbadata.players import normalize_name
from nbadata.teams import resolve_team
from nbadata.timeutil import eastern_date, iso_utc, now_utc, parse_iso_utc
from oddsfeed.schema import GameOdds, PropOdds

logger = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"


def load_api_key() -> Optional[str]:
    """Key lookup order: environment variable, then ./.env."""
    key = os.environ.get(ODDS.API_KEY_ENV)
    if key:
        return key.strip()
    try:
        if os.path.exists(".env"):
            for line in open(".env", encoding="utf-8"):
                line = line.strip()
                if line.startswith(f"{ODDS.API_KEY_ENV}="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return None


class OddsAPIError(ProviderError):
    pass


class TheOddsAPI:
    name = "the_odds_api"

    def __init__(self, api_key: Optional[str] = None, http: Optional[HttpClient] = None):
        self.api_key = api_key or load_api_key()
        self.http = http or HttpClient(min_interval=0.2)
        self.credits_remaining: Optional[int] = None
        self.credits_used: Optional[int] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ── low level ────────────────────────────────────────────────────────
    def _get(self, path: str, params: Dict) -> object:
        if not self.api_key:
            raise OddsAPIError(
                f"No odds API key. Get a free key at https://the-odds-api.com and set {ODDS.API_KEY_ENV} "
                "(environment variable or .env file)."
            )
        try:
            data = self.http.get_json(f"{BASE}{path}", params={"apiKey": self.api_key, **params})
        except ProviderError as e:
            if e.status == 401:
                raise OddsAPIError("Odds API rejected the key (401). Check ODDS_API_KEY.", 401, e.url) from e
            if e.status == 429:
                raise OddsAPIError("Odds API rate limit or monthly credits exhausted (429).", 429, e.url) from e
            if e.status == 422:
                raise OddsAPIError(f"Odds API rejected the request parameters (422): {e}", 422, e.url) from e
            raise
        h = {k.lower(): v for k, v in self.http.last_headers.items()}
        try:
            self.credits_remaining = int(float(h["x-requests-remaining"])) if "x-requests-remaining" in h else self.credits_remaining
            self.credits_used = int(float(h["x-requests-used"])) if "x-requests-used" in h else self.credits_used
        except ValueError:
            pass
        return data

    # ── endpoints ────────────────────────────────────────────────────────
    def fetch_game_odds(self, markets: Tuple[str, ...] = None, regions: str = None) -> List[dict]:
        return self._get(f"/sports/{ODDS.SPORT_KEY}/odds", {
            "regions": regions or ODDS.REGIONS,
            "markets": ",".join(markets or ODDS.GAME_MARKETS),
            "oddsFormat": "american", "dateFormat": "iso",
        })

    def fetch_event_props(self, event_id: str, markets: Tuple[str, ...] = None, regions: str = None) -> dict:
        return self._get(f"/sports/{ODDS.SPORT_KEY}/events/{event_id}/odds", {
            "regions": regions or ODDS.REGIONS,
            "markets": ",".join(markets or ODDS.PROP_MARKETS),
            "oddsFormat": "american", "dateFormat": "iso",
        })

    def check_key(self) -> Dict:
        """Free call (events list costs no credits). Used by `doctor`."""
        events = self._get(f"/sports/{ODDS.SPORT_KEY}/events", {"dateFormat": "iso"})
        return {"events": len(events), "credits_remaining": self.credits_remaining, "credits_used": self.credits_used}

    # ── parsing ──────────────────────────────────────────────────────────
    @staticmethod
    def _int(x) -> Optional[int]:
        try:
            return int(round(float(x)))
        except (TypeError, ValueError):
            return None

    @classmethod
    def parse_game_odds(cls, events: List[dict], fetched_at: Optional[str] = None) -> List[GameOdds]:
        fetched_at = fetched_at or iso_utc(now_utc())
        out: List[GameOdds] = []
        for ev in events or []:
            home, away = resolve_team(ev.get("home_team")), resolve_team(ev.get("away_team"))
            tip = parse_iso_utc(ev.get("commence_time"))
            if not (home and away and tip):
                logger.debug("Odds API event skipped (unrecognised teams): %s @ %s", ev.get("away_team"), ev.get("home_team"))
                continue
            gdate = eastern_date(tip).isoformat()
            for bk in ev.get("bookmakers") or []:
                rec = GameOdds(game_date=gdate, home_team_id=home.id, away_team_id=away.id,
                               commence_utc=iso_utc(tip), book=str(bk.get("key")), fetched_at=fetched_at, source="the_odds_api")
                for mk in bk.get("markets") or []:
                    outs = mk.get("outcomes") or []
                    key = mk.get("key")
                    if key == "h2h":
                        for o in outs:
                            t = resolve_team(o.get("name"))
                            if t and t.id == home.id:
                                rec.ml_home = cls._int(o.get("price"))
                            elif t and t.id == away.id:
                                rec.ml_away = cls._int(o.get("price"))
                    elif key == "spreads":
                        for o in outs:
                            t = resolve_team(o.get("name"))
                            if t and t.id == home.id:
                                rec.spread_home = float(o["point"]) if o.get("point") is not None else None
                                rec.spread_home_price = cls._int(o.get("price"))
                            elif t and t.id == away.id:
                                rec.spread_away_price = cls._int(o.get("price"))
                    elif key == "totals":
                        for o in outs:
                            nm = str(o.get("name", "")).lower()
                            if nm == "over":
                                rec.total = float(o["point"]) if o.get("point") is not None else rec.total
                                rec.over_price = cls._int(o.get("price"))
                            elif nm == "under":
                                rec.under_price = cls._int(o.get("price"))
                                if rec.total is None and o.get("point") is not None:
                                    rec.total = float(o["point"])
                out.append(rec)
        return out

    @classmethod
    def parse_props(cls, event: dict, fetched_at: Optional[str] = None) -> List[PropOdds]:
        """One per-event odds response -> PropOdds rows (one per book/market/player/line)."""
        fetched_at = fetched_at or iso_utc(now_utc())
        home, away = resolve_team(event.get("home_team")), resolve_team(event.get("away_team"))
        tip = parse_iso_utc(event.get("commence_time"))
        if not (home and away and tip):
            return []
        gdate = eastern_date(tip).isoformat()
        out: List[PropOdds] = []
        for bk in event.get("bookmakers") or []:
            for mk in bk.get("markets") or []:
                market = mk.get("key")
                if not str(market).startswith("player_"):
                    continue
                grouped: Dict[Tuple[str, float], Dict[str, Optional[int]]] = {}
                names: Dict[str, str] = {}
                for o in mk.get("outcomes") or []:
                    player = o.get("description") or o.get("participant")
                    side = str(o.get("name", "")).lower()
                    if not player or side not in ("over", "under") or o.get("point") is None:
                        continue
                    key = (normalize_name(player), float(o["point"]))
                    names[key[0]] = player
                    grouped.setdefault(key, {})[side] = cls._int(o.get("price"))
                for (pkey, line), sides in grouped.items():
                    out.append(PropOdds(
                        game_date=gdate, home_team_id=home.id, away_team_id=away.id, commence_utc=iso_utc(tip),
                        book=str(bk.get("key")), fetched_at=fetched_at, source="the_odds_api", market=str(market),
                        player_name=names[pkey], player_key=pkey, line=line,
                        over_price=sides.get("over"), under_price=sides.get("under"),
                    ))
        return out

    # ── high level ───────────────────────────────────────────────────────
    def pull(self, include_props: bool = True, now: Optional[datetime] = None) -> Tuple[List[GameOdds], List[PropOdds], List[str]]:
        """Fetch game lines and (optionally) player props for games starting soon."""
        notes: List[str] = []
        fetched_at = iso_utc(now_utc())
        events = self.fetch_game_odds()
        games = self.parse_game_odds(events, fetched_at)
        notes.append(f"{len({(g.home_team_id, g.away_team_id, g.game_date) for g in games})} games, {len(games)} book lines")
        props: List[PropOdds] = []
        if include_props:
            horizon = (now or now_utc()) + timedelta(hours=ODDS.PROP_LOOKAHEAD_HOURS)
            upcoming = []
            for ev in events:
                tip = parse_iso_utc(ev.get("commence_time"))
                if tip and tip <= horizon and tip >= (now or now_utc()) - timedelta(hours=1):
                    upcoming.append(ev)
            upcoming.sort(key=lambda e: e.get("commence_time", ""))
            if len(upcoming) > ODDS.MAX_PROP_EVENTS:
                notes.append(f"props limited to the next {ODDS.MAX_PROP_EVENTS} of {len(upcoming)} games (credit cap)")
                upcoming = upcoming[:ODDS.MAX_PROP_EVENTS]
            for ev in upcoming:
                try:
                    props.extend(self.parse_props(self.fetch_event_props(ev["id"]), fetched_at))
                except OddsAPIError as e:
                    if e.status in (401, 429):
                        notes.append(f"props stopped: {e}")
                        break
                    notes.append(f"props unavailable for {ev.get('away_team')} @ {ev.get('home_team')}: {e}")
                except ProviderError as e:
                    notes.append(f"props failed for {ev.get('away_team')} @ {ev.get('home_team')}: {e}")
            notes.append(f"{len(props)} prop lines")
        return games, props, notes
