# net.py — the one place the app talks HTTP (card lists, prices, images).
#
# A single User-Agent (some CDNs, e.g. Ravensburger's image host, 403 an
# empty urllib agent), explicit timeouts, and one retry policy: transient
# failures (timeouts, connection drops, 408/429, 5xx) are retried with
# back-off; other 4xx answers are final.

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "Lore-Book/1.0 (personal card-collection tool)"


def _request(url: str, headers: dict | None = None) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def fetch_json(url: str, headers: dict | None = None, timeout: float = 120) -> Any:
    """GET url and decode JSON. Raises on any network/HTTP/JSON error."""
    with urllib.request.urlopen(_request(url, headers), timeout=timeout) as resp:
        return json.load(resp)


def is_retryable(err: Exception) -> bool:
    """True for failures worth retrying; a 4xx other than 408/429 is final."""
    if isinstance(err, urllib.error.HTTPError):
        return not (400 <= err.code < 500) or err.code in (408, 429)
    return isinstance(err, (urllib.error.URLError, TimeoutError, ConnectionError))


def fetch_bytes(url: str, retries: int = 3, backoff: float = 1.5, timeout: float = 60) -> bytes:
    """GET url's body, retrying transient failures with linear back-off."""
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if not is_retryable(e) or attempt == attempts - 1:
                raise
            logger.debug("Retrying %s after %s", url, e)
            time.sleep(backoff * (attempt + 1))
    raise AssertionError("unreachable")  # the loop always returns or raises
