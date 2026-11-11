"""Player-name normalisation and matching across data sources."""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, Iterable, Optional

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Known differences between sportsbook / box-score spellings. Keys and values are
# already-normalised names. Extend freely; the fuzzy matcher covers most other cases.
ALIASES: Dict[str, str] = {
    "nicolas claxton": "nic claxton",
    "moe wagner": "moritz wagner",
    "cam johnson": "cameron johnson",
    "cam thomas": "cameron thomas",
    "herb jones": "herbert jones",
    "pj washington": "p j washington",
    "ron holland": "ronald holland",
    "alex sarr": "alexandre sarr",
    "nah shon hyland": "bones hyland",
    "kj martin": "kenyon martin",
    "ajg": "a j green",
}


def normalize_name(name) -> str:
    """'Luka Dončić' -> 'luka doncic'; 'Gary Trent Jr.' -> 'gary trent'; 'P.J. Tucker' -> 'p j tucker'."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("’", "'")
    s = re.sub(r"[.']", " ", s)           # P.J. -> p j ; O'Neale -> o neale
    s = re.sub(r"[^a-z0-9 ]+", " ", s)    # hyphens -> spaces
    toks = [t for t in s.split() if t]
    while len(toks) > 1 and toks[-1] in _SUFFIXES:
        toks.pop()
    out = " ".join(toks)
    return ALIASES.get(out, out)


def _squash(norm: str) -> str:
    return norm.replace(" ", "")


def match_player(name: str, candidates: Iterable[str], cutoff: float = 0.86) -> Optional[str]:
    """
    Match a (sportsbook) player name to one of `candidates` (already normalised names,
    typically the active roster of one team). Returns the matched candidate or None.
    Conservative on purpose: a wrong match on a prop is worse than no match.
    """
    n = normalize_name(name)
    cands = list(dict.fromkeys(candidates))
    if not n or not cands:
        return None
    if n in cands:
        return n
    squashed = {_squash(c): c for c in cands}
    if _squash(n) in squashed:
        return squashed[_squash(n)]

    # Same last name + same first initial, unique
    toks = n.split()
    first, last = toks[0], toks[-1]
    same_last = [c for c in cands if c.split()[-1] == last]
    initial = [c for c in same_last if c.split()[0][:1] == first[:1]]
    if len(initial) == 1:
        return initial[0]

    close = difflib.get_close_matches(n, cands, n=2, cutoff=cutoff)
    if len(close) == 1:
        return close[0]
    if len(close) == 2:
        r0 = difflib.SequenceMatcher(None, n, close[0]).ratio()
        r1 = difflib.SequenceMatcher(None, n, close[1]).ratio()
        return close[0] if r0 - r1 > 0.05 else None
    if len(same_last) == 1 and len(last) >= 5:
        return same_last[0]
    return None
