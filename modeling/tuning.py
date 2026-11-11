"""
Corrections found by the backtest, stored in models/tuning.json and applied at prediction time.

Everything here is optional: with no file the model runs exactly as trained. A correction is only
ever written after it has been verified on held-out later games (see modeling/backtest.py).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from scipy.stats import norm

from config import MODEL

FILENAME = "tuning.json"


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def platt(p: float, a: float, b: float) -> float:
    """Platt scaling: sigmoid(a * logit(p) + b). a=1, b=0 is the identity."""
    z = a * logit(p) + b
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Tuning:
    rating: Dict[str, float] = field(default_factory=dict)        # half_life, ridge, pace_ridge, carry
    margin_scale: float = 1.0                                      # stretches projected margins (1.0 = unchanged)
    margin_bias: float = 0.0                                       # added to the (scaled) projected home margin
    margin_sigma: Optional[float] = None                           # replaces the fitted margin sigma when the scale is used
    total_bias: float = 0.0                                        # added to the projected total
    pts_bias: float = 0.0                                          # additive correction to player point projections
    win_cal: Optional[Tuple[float, float]] = None                  # Platt (a, b) for P(home win)
    prop_cal: Optional[Tuple[float, float]] = None                 # Platt (a, b) for P(over | no push)
    prop_market_weight: Optional[float] = None                    # learned from graded bets (`report --apply-weights`)
    game_market_weight: Optional[float] = None
    created: Optional[str] = None
    evidence: Dict = field(default_factory=dict)                   # what was measured when this was fitted

    # ── state ────────────────────────────────────────────────────────────
    def is_default(self) -> bool:
        return (not self.rating and self.margin_scale == 1.0 and self.margin_bias == 0 and self.total_bias == 0
                and self.pts_bias == 0 and self.win_cal is None and self.prop_cal is None
                and self.prop_market_weight is None and self.game_market_weight is None)

    def describe(self) -> list:
        """Human-readable list of the corrections in effect."""
        out = []
        if self.rating:
            out.append("rating parameters: " + ", ".join(f"{k}={v:g}" for k, v in sorted(self.rating.items())))
        if self.margin_scale != 1.0:
            out.append(f"margin stretch x{self.margin_scale:.2f} (projected margins were too close to zero)" if self.margin_scale > 1
                       else f"margin shrink x{self.margin_scale:.2f} (projected margins were too extreme)")
        if self.margin_bias:
            out.append(f"margin bias correction {self.margin_bias:+.2f} pts")
        if self.total_bias:
            out.append(f"total bias correction {self.total_bias:+.2f} pts")
        if self.pts_bias:
            out.append(f"player-points bias correction {self.pts_bias:+.2f} pts (offsets systematic under-projection)")
        if self.win_cal:
            out.append(f"win-probability recalibration (slope {self.win_cal[0]:.2f}, shift {self.win_cal[1]:+.2f})")
        if self.prop_cal:
            out.append(f"over/under probability recalibration (slope {self.prop_cal[0]:.2f}, shift {self.prop_cal[1]:+.2f})")
        if self.prop_market_weight is not None:
            out.append(f"player-prop market blend {self.prop_market_weight:.0%} (learned from graded bets)")
        if self.game_market_weight is not None:
            out.append(f"game-bet market blend {self.game_market_weight:.0%} (learned from graded bets)")
        return out

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, directory: str = None) -> str:
        directory = directory or MODEL.MODELS_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, FILENAME)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @staticmethod
    def load(directory: str = None) -> "Tuning":
        path = os.path.join(directory or MODEL.MODELS_DIR, FILENAME)
        if not os.path.exists(path):
            return Tuning()
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            t = Tuning(rating={k: float(v) for k, v in (d.get("rating") or {}).items()},
                       margin_scale=float(d.get("margin_scale") or 1.0), margin_sigma=float(d["margin_sigma"]) if d.get("margin_sigma") else None,
                       prop_market_weight=float(d["prop_market_weight"]) if d.get("prop_market_weight") is not None else None,
                       game_market_weight=float(d["game_market_weight"]) if d.get("game_market_weight") is not None else None,
                       margin_bias=float(d.get("margin_bias") or 0.0), total_bias=float(d.get("total_bias") or 0.0),
                       pts_bias=float(d.get("pts_bias") or 0.0),
                       win_cal=tuple(d["win_cal"]) if d.get("win_cal") else None,
                       prop_cal=tuple(d["prop_cal"]) if d.get("prop_cal") else None,
                       created=d.get("created"), evidence=d.get("evidence") or {})
            return t
        except (OSError, ValueError, KeyError, TypeError):
            return Tuning()

    @staticmethod
    def reset(directory: str = None) -> bool:
        path = os.path.join(directory or MODEL.MODELS_DIR, FILENAME)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ── application ──────────────────────────────────────────────────────
    def apply_game(self, pred: Dict) -> Dict:
        """Apply bias and win-probability corrections to a GameModel.predict() result."""
        if (self.margin_scale == 1.0 and self.margin_bias == 0 and self.total_bias == 0 and self.win_cal is None
                and self.margin_sigma is None):
            return pred
        out = dict(pred)
        margin = pred["margin"] * self.margin_scale + self.margin_bias
        total = pred["total"] + self.total_bias
        sigma = self.margin_sigma or pred["margin_sigma"]
        out.update(margin=margin, total=total, margin_sigma=sigma, home_pts=(total + margin) / 2, away_pts=(total - margin) / 2)
        p = float(norm.cdf(margin / sigma))
        out["home_win_prob"] = platt(p, *self.win_cal) if self.win_cal else p
        return out

    def apply_mu(self, mu: float) -> float:
        """Apply additive player-points bias (model was systematically low vs actuals)."""
        if not self.pts_bias:
            return mu
        return max(0.0, float(mu) + self.pts_bias)

    def apply_prop(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Recalibrate P(over) (conditional on no push); push probability is unchanged."""
        if self.prop_cal is None:
            return probs
        push = probs["push"]
        denom = max(1.0 - push, 1e-9)
        cond = platt(min(max(probs["over"] / denom, 1e-6), 1 - 1e-6), *self.prop_cal)
        return {"over": cond * denom, "under": (1.0 - cond) * denom, "push": push}
