"""A small, polite HTTP client: retries with backoff, per-host throttling, clear errors."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

from config import DATA

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """A data provider failed in a way the caller may want to report or fall back from."""

    def __init__(self, message: str, status: Optional[int] = None, url: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.url = url


class HttpClient:
    def __init__(
        self,
        min_interval: float = None,
        timeout: float = None,
        retries: int = None,
        backoff: float = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.min_interval = DATA.MIN_REQUEST_INTERVAL if min_interval is None else min_interval
        self.timeout = timeout or DATA.REQUEST_TIMEOUT
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": DATA.USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        if headers:
            self.session.headers.update(headers)
        if Retry is not None and session is None:
            total = retries if retries is not None else DATA.MAX_RETRIES
            retry = Retry(
                total=total, connect=min(total, 2), read=min(total, 2),
                backoff_factor=backoff if backoff is not None else DATA.RETRY_BACKOFF,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        self._last: Dict[str, float] = {}
        self.last_headers: Dict[str, str] = {}

    def _throttle(self, host: str):
        wait = self.min_interval - (time.time() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        host = urlparse(url).netloc
        self._throttle(host)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise ProviderError(f"{host}: request failed ({type(e).__name__}: {e})", url=url) from e
        self.last_headers = dict(resp.headers)
        if resp.status_code != 200:
            text = re.sub(r"<[^>]+>", " ", resp.text or "")            # strip HTML error pages
            snippet = re.sub(r"\s+", " ", text).strip()[:100]
            hint = ""
            if resp.status_code in (401, 403):
                hint = " (blocked by the site: bot protection, VPN/proxy or network filter; not a parsing problem)"
            raise ProviderError(f"{host}: HTTP {resp.status_code} {snippet}{hint}", status=resp.status_code, url=url)
        try:
            return resp.json()
        except ValueError as e:
            ctype = resp.headers.get("Content-Type", "?")
            raise ProviderError(
                f"{host}: response was not JSON (Content-Type {ctype}); the endpoint may have changed or blocked this IP",
                status=resp.status_code, url=url,
            ) from e
