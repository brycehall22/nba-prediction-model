"""
Betting math: probabilities, de-vig comparison, expected value and Kelly sizing.

Fixes vs. the previous engine
  * Spread cover probability had its sign wrong (a fair 50/50 favourite showed ~72%), which
    systematically inflated EV on favourites. Covered by tests/test_ev_engine.py.
  * Uncertainty now comes from fitted out-of-sample residuals, not ensemble spread.
  * Pushes (integer lines) are handled in probability, EV and Kelly.
  * Final probabilities are blended toward the de-vigged market (BETTING.*_MARKET_WEIGHT), because
    a book's number contains information (lineups, injuries) the model can't see.
  * Large model-vs-market disagreements are flagged instead of presented as free money.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

from config import BETTING
from oddsfeed.schema import american_to_decimal, american_to_prob, devig_two_way

BIG_DISAGREEMENT = 0.12     # |blended prob - fair market prob| above this gets a warning


# ── primitives ───────────────────────────────────────────────────────────────
def ev_per_unit(p_win: float, p_loss: float, american: float) -> float:
    """Expected profit per 1 unit staked; a push (1 - p_win - p_loss) returns the stake."""
    return p_win * (american_to_decimal(american) - 1.0) - p_loss


def kelly_fraction(p_win: float, p_loss: float, american: float,
                   fraction: float = None, cap_pct: float = None) -> float:
    """Fractional Kelly for a win/loss/push bet, as a fraction of bankroll. 0 when there is no edge."""
    fraction = BETTING.KELLY_FRACTION if fraction is None else fraction
    cap = (BETTING.MAX_KELLY_BET_PCT if cap_pct is None else cap_pct) / 100.0
    b = american_to_decimal(american) - 1.0
    if b <= 0 or (p_win + p_loss) <= 0:
        return 0.0
    f = (p_win * b - p_loss) / (b * (p_win + p_loss))
    return float(min(max(f * fraction, 0.0), cap))


def blend_win_prob(p_win: float, p_push: float, market_fair: Optional[float], weight: float) -> float:
    """Blend the (push-conditional) model win probability toward the de-vigged market probability."""
    if market_fair is None or weight <= 0:
        return p_win
    cond = p_win / max(1.0 - p_push, 1e-9)
    cond = (1.0 - weight) * cond + weight * market_fair
    return cond * (1.0 - p_push)


def value_label(ev_pct: float) -> str:
    if ev_pct >= BETTING.HIGH_EV_THRESHOLD_PCT:
        return "high"
    if ev_pct >= BETTING.EV_THRESHOLD_PCT:
        return "value"
    return "none"


def _bet(kind: str, side: str, line: Optional[float], price: int, p_win: float, p_push: float,
         market_fair: Optional[float], weight: float, extra: Optional[Dict] = None) -> Dict:
    p_win = float(np.clip(p_win, 0.0, 1.0 - p_push))
    blended = blend_win_prob(p_win, p_push, market_fair, weight)
    p_loss = max(1.0 - blended - p_push, 0.0)
    ev = ev_per_unit(blended, p_loss, price) * 100.0
    raw_ev = ev_per_unit(p_win, max(1.0 - p_win - p_push, 0.0), price) * 100.0
    out = {
        "kind": kind, "side": side, "line": line, "price": int(price),
        "model_prob": p_win, "push_prob": p_push, "market_fair_prob": market_fair, "blended_prob": blended,
        "implied_prob": american_to_prob(price), "ev_pct": ev, "raw_ev_pct": raw_ev,
        "kelly_pct": kelly_fraction(blended, p_loss, price) * 100.0, "label": value_label(ev),
        "warning": None,
    }
    if market_fair is not None and abs(p_win / max(1 - p_push, 1e-9) - market_fair) > BIG_DISAGREEMENT:
        out["warning"] = ("Model and market disagree strongly; check lineups/injuries before trusting this edge.")
    if extra:
        out.update(extra)
    return out


# ── games ────────────────────────────────────────────────────────────────────
class GameEV:
    """Probabilities for spread / total / moneyline from a model margin & total with fitted sigmas."""

    @staticmethod
    def _cover_probs(mean: float, sigma: float, threshold: float) -> Dict[str, float]:
        """
        P(X > threshold), P(X < threshold), P(X == threshold) for an integer-valued X (final margin or total)
        approximated by a normal with a continuity correction. Half-point thresholds cannot push.
        """
        if abs(threshold - round(threshold)) < 1e-9:
            hi, lo = threshold + 0.5, threshold - 0.5
            over = 1.0 - norm.cdf(hi, mean, sigma)
            under = norm.cdf(lo, mean, sigma)
            return {"over": float(over), "under": float(under), "push": float(1.0 - over - under)}
        over = 1.0 - norm.cdf(threshold, mean, sigma)
        return {"over": float(over), "under": float(1.0 - over), "push": 0.0}

    def spread_probs(self, margin: float, sigma: float, home_spread: float) -> Dict[str, float]:
        """
        home_spread is the home team's line (negative = home favoured, e.g. -5.5).
        Home covers when  final_margin + home_spread > 0  <=>  margin > -home_spread.
        (The previous implementation used `margin > home_spread`, which is the sign bug.)
        """
        c = self._cover_probs(margin, sigma, -home_spread)
        return {"home": c["over"], "away": c["under"], "push": c["push"]}

    def total_probs(self, total: float, sigma: float, line: float) -> Dict[str, float]:
        return self._cover_probs(total, sigma, line)

    @staticmethod
    def win_prob(margin: float, sigma: float) -> float:
        return float(norm.cdf(margin / sigma))

    def evaluate(self, pred: Dict, market: Dict, weight: float = None) -> List[Dict]:
        """All available game bets with EV. `pred` from GameModel.predict, `market` from OddsService.get_game_market."""
        w = BETTING.GAME_MARKET_WEIGHT if weight is None else weight
        bets: List[Dict] = []
        if not market or not market.get("available"):
            return bets
        fair = market.get("fair") or {}
        ms, ts = pred["margin_sigma"], pred["total_sigma"]
        hs = market.get("home_spread")
        if hs is not None:
            p = self.spread_probs(pred["margin"], ms, hs)
            fh = fair.get("home_cover")
            edge = pred["margin"] - (-hs)
            for side, key, line, price, mf in (
                ("HOME", "home", hs, market["home_spread_odds"], fh),
                ("AWAY", "away", -hs, market["away_spread_odds"], (1 - fh) if fh is not None else None)):
                bets.append(_bet("spread", side, line, price, p[key], p["push"], mf, w,
                                 {"edge_pts": edge if side == "HOME" else -edge}))
        tl = market.get("total")
        if tl is not None:
            p = self.total_probs(pred["total"], ts, tl)
            fo = fair.get("over")
            edge = pred["total"] - tl
            for side, key, price, mf in (
                ("OVER", "over", market["total_over_odds"], fo),
                ("UNDER", "under", market["total_under_odds"], (1 - fo) if fo is not None else None)):
                bets.append(_bet("total", side, tl, price, p[key], p["push"], mf, w,
                                 {"edge_pts": edge if side == "OVER" else -edge}))
        hml, aml = market.get("home_moneyline"), market.get("away_moneyline")
        if hml is not None and aml is not None:
            ph = self.win_prob(pred["margin"], ms)
            fw = fair.get("home_win")
            bets.append(_bet("moneyline", "HOME", None, int(hml), ph, 0.0, fw, w))
            bets.append(_bet("moneyline", "AWAY", None, int(aml), 1 - ph, 0.0, (1 - fw) if fw is not None else None, w))
        return bets


# ── props ────────────────────────────────────────────────────────────────────
def evaluate_prop(probs: Dict[str, float], line: float, over_odds: int, under_odds: int,
                  fair_over: Optional[float] = None, weight: float = None, extra: Optional[Dict] = None) -> List[Dict]:
    """probs = {'over','under','push'} from player_model.over_under_probs."""
    w = BETTING.PROP_MARKET_WEIGHT if weight is None else weight
    push = probs["push"]
    return [
        _bet("prop", "OVER", line, over_odds, probs["over"], push, fair_over, w, extra),
        _bet("prop", "UNDER", line, under_odds, probs["under"], push, (1 - fair_over) if fair_over is not None else None, w, extra),
    ]
