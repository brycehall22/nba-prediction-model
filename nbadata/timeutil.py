"""Time helpers. NBA 'game dates' are US/Eastern calendar dates of tip-off."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union

try:  # zoneinfo needs the `tzdata` package on Windows
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback if tz database is missing
    _ET = None


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _fallback_eastern(dt_utc: datetime) -> datetime:
    y = dt_utc.year
    dst_start = datetime.combine(_nth_weekday(y, 3, 6, 2), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=7)
    dst_end = datetime.combine(_nth_weekday(y, 11, 6, 1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=6)
    offset = -4 if dst_start <= dt_utc < dst_end else -5
    return (dt_utc + timedelta(hours=offset)).replace(tzinfo=None)


def parse_iso_utc(s: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse '2025-01-15T00:30:00Z', '2025-01-15T00:30Z', or offset-aware ISO strings to aware UTC."""
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        txt = str(s).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(txt)
        except ValueError:
            # '2025-01-15T00:30+00:00' style is fine; handle 'YYYY-MM-DDTHH:MM' without seconds
            try:
                dt = datetime.strptime(txt[:16], "%Y-%m-%dT%H:%M")
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def eastern_date(dt_utc: Union[str, datetime]) -> Optional[date]:
    """US/Eastern calendar date for a UTC timestamp."""
    dt = parse_iso_utc(dt_utc)
    if dt is None:
        return None
    if _ET is not None:
        return dt.astimezone(_ET).date()
    return _fallback_eastern(dt).date()


def iso_utc(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_eastern() -> date:
    return eastern_date(now_utc())


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def utcnow_naive() -> datetime:
    """Naive UTC 'now' (what we store in SQLite); replaces the deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
