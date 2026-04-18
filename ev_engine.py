"""
Unified Expected Value Engine.

Single source of truth for ALL probability, EV, and betting math.
No other module should implement these calculations independently.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

from config import BETTING, PROPS

logger = logging.getLogger(__name__)


class EVEngine:
    """
    Probability-based betting analysis engine.

    Consolidates:
    - Score distribution generation
    - Spread / total / moneyline probabilities
    - Expected value calculations
    - Kelly criterion sizing
    - Player prop evaluation
    - Dynamic odds-to-probability conversion
    """

    def __init__(self, n_samples: int = None):
        self.n_samples = n_samples or BETTING.SIMULATION_SAMPLES

    # ── Odds Conversion ──────────────────────────────────────────────────

    @staticmethod
    def american_to_probability(odds: int) -> float:
        """Convert American odds to implied probability (no-vig)."""
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def american_to_decimal(odds: int) -> float:
        """Convert American odds to decimal odds."""
        if odds > 0:
            return odds / 100.0 + 1.0
        return 100.0 / abs(odds) + 1.0

    @staticmethod
    def remove_vig(prob_a: float, prob_b: float) -> Tuple[float, float]:
        """Remove vig from a two-outcome market to get fair probabilities."""
        total = prob_a + prob_b
        if total == 0:
            return 0.5, 0.5
        return prob_a / total, prob_b / total

    # ── Score Distributions ──────────────────────────────────────────────

    def generate_score_distribution(self, mean: float, std: float) -> np.ndarray:
        """
        Generate truncated-normal score distribution.
        Scores bounded to [0, 200].
        """
        if std <= 0:
            return np.full(self.n_samples, mean)
        dist = stats.truncnorm(
            (0 - mean) / std,
            (200 - mean) / std,
            loc=mean,
            scale=std,
        )
        return dist.rvs(size=self.n_samples)

    # ── Game-Level Probabilities ─────────────────────────────────────────

    def spread_probability(
        self, home_mean: float, home_std: float,
        away_mean: float, away_std: float,
        spread: float,
    ) -> float:
        """
        P(home team covers the spread).
        spread is from the home team's perspective (negative = home favored).
        """
        margin_mean = home_mean - away_mean
        margin_std = np.sqrt(home_std**2 + away_std**2)
        if margin_std <= 0:
            return 1.0 if margin_mean > spread else 0.0
        return float(1 - stats.norm.cdf(spread, loc=margin_mean, scale=margin_std))

    def total_probability(
        self, home_mean: float, home_std: float,
        away_mean: float, away_std: float,
        total_line: float,
    ) -> Tuple[float, float]:
        """P(over), P(under) for the game total."""
        total_mean = home_mean + away_mean
        total_std = np.sqrt(home_std**2 + away_std**2)
        if total_std <= 0:
            over = 1.0 if total_mean > total_line else 0.0
            return over, 1.0 - over
        over = float(1 - stats.norm.cdf(total_line, loc=total_mean, scale=total_std))
        return over, 1.0 - over

    def win_probability(
        self, home_mean: float, home_std: float,
        away_mean: float, away_std: float,
    ) -> float:
        """P(home team wins)."""
        return self.spread_probability(home_mean, home_std, away_mean, away_std, spread=0.0)

    # ── Expected Value ───────────────────────────────────────────────────

    @staticmethod
    def expected_value(model_prob: float, odds: int) -> float:
        """
        Calculate EV% for a bet.

        Returns:
            EV as a percentage. Positive = +EV.
        """
        if odds == 0:
            return 0.0
        if odds > 0:
            decimal = odds / 100.0 + 1.0
        else:
            decimal = 100.0 / abs(odds) + 1.0

        ev = model_prob * (decimal - 1.0) - (1.0 - model_prob)
        return round(ev * 100.0, 2)

    # ── Kelly Criterion ──────────────────────────────────────────────────

    @staticmethod
    def kelly_fraction(
        model_prob: float,
        odds: int,
        fraction: float = None,
        max_bet: float = None,
    ) -> float:
        """
        Fractional Kelly bet sizing.

        Returns:
            Recommended bet as fraction of bankroll [0, max_bet].
        """
        fraction = fraction or BETTING.KELLY_FRACTION
        max_bet = max_bet or (BETTING.MAX_KELLY_BET_PCT / 100.0)

        if odds > 0:
            b = odds / 100.0
        else:
            b = 100.0 / abs(odds)

        edge = model_prob * b - (1.0 - model_prob)
        if edge <= 0:
            return 0.0

        kelly = (edge / b) * fraction
        return min(max_bet, max(0.0, kelly))

    # ── Player Prop Evaluation ───────────────────────────────────────────

    def evaluate_player_prop(
        self,
        predicted_pts: float,
        pts_std: float,
        line: float,
        over_odds: int,
        under_odds: int,
    ) -> Dict:
        """
        Full evaluation of a player points prop.

        Returns dict with over/under probabilities, EVs, best bet, Kelly size,
        confidence, and value rating.
        """
        if pts_std <= 0:
            pts_std = max(PROPS.MIN_POINTS_STD, predicted_pts * 0.3)

        # Z-score and probabilities via normal CDF
        z = (line - predicted_pts) / pts_std
        over_prob = float(1 - stats.norm.cdf(z))
        under_prob = float(stats.norm.cdf(z))

        over_ev = self.expected_value(over_prob, over_odds)
        under_ev = self.expected_value(under_prob, under_odds)

        if over_ev >= under_ev and over_ev > 0:
            best_bet, best_ev, best_prob, best_odds = "OVER", over_ev, over_prob, over_odds
        elif under_ev > 0:
            best_bet, best_ev, best_prob, best_odds = "UNDER", under_ev, under_prob, under_odds
        else:
            best_bet = "NONE"
            best_ev = max(over_ev, under_ev)
            best_prob = over_prob if over_ev > under_ev else under_prob
            best_odds = over_odds if over_ev > under_ev else under_odds

        kelly = self.kelly_fraction(best_prob, best_odds) if best_bet != "NONE" else 0.0

        # Confidence based on distance from the line in std deviations
        dist_std = abs(z)
        if dist_std < 0.25:
            confidence = 0.3
        elif dist_std < 0.6:
            confidence = 0.5
        elif dist_std < 1.0:
            confidence = 0.7
        else:
            confidence = 0.85

        # Value rating 1-10
        if best_ev > 0:
            value_rating = min(10, max(1, round(best_ev / 3 + 4)))
        else:
            value_rating = 1

        return {
            "valid": True,
            "line": line,
            "predicted_pts": predicted_pts,
            "pts_std": pts_std,
            "over": {"prob": round(over_prob, 4), "odds": over_odds, "ev": over_ev},
            "under": {"prob": round(under_prob, 4), "odds": under_odds, "ev": under_ev},
            "best_bet": best_bet,
            "best_ev": best_ev,
            "best_prob": round(best_prob, 4),
            "kelly_fraction": round(kelly, 4),
            "confidence": round(confidence, 3),
            "value_rating": value_rating,
            "edge_pts": round(abs(predicted_pts - line), 1),
            "z_score": round(z, 3),
        }

    # ── Dynamic Odds Regression ──────────────────────────────────────────

    @staticmethod
    def dynamic_regression_strength(model_confidence: float) -> float:
        """
        Calculate how much to regress toward market based on model confidence.
        High-confidence predictions get less regression.
        """
        # Linear interpolation: confidence 0 → MAX_REGRESSION, confidence 1 → MIN_REGRESSION
        return BETTING.MAX_REGRESSION - (
            (BETTING.MAX_REGRESSION - BETTING.MIN_REGRESSION) * model_confidence
        )

    def calibrate_with_market(
        self,
        home_score: float,
        away_score: float,
        home_std: float,
        away_std: float,
        model_confidence: float,
        market_data: Dict,
    ) -> Dict:
        """
        Blend model prediction with market expectations using dynamic regression.

        Returns calibrated scores, stds, and full EV metrics.
        """
        if not market_data or not market_data.get("available"):
            return {
                "home_score": home_score,
                "away_score": away_score,
                "home_std": home_std,
                "away_std": away_std,
                "regression_applied": False,
                "ev_metrics": {"available": False},
            }

        spread = market_data.get("home_spread", 0)
        total = market_data.get("total", 0)

        # Market-implied scores
        market_home = (total - spread) / 2.0
        market_away = (total + spread) / 2.0

        # Dynamic regression
        rs = self.dynamic_regression_strength(model_confidence)
        cal_home = (1 - rs) * home_score + rs * market_home
        cal_away = (1 - rs) * away_score + rs * market_away

        # Compute EV metrics with calibrated scores
        spread_cover = self.spread_probability(cal_home, home_std, cal_away, away_std, spread)
        over_prob, under_prob = self.total_probability(cal_home, home_std, cal_away, away_std, total)
        win_prob = self.win_probability(cal_home, home_std, cal_away, away_std)

        home_spread_odds = market_data.get("home_spread_odds", BETTING.DEFAULT_ODDS)
        away_spread_odds = market_data.get("away_spread_odds", BETTING.DEFAULT_ODDS)
        over_odds = market_data.get("total_over_odds", BETTING.DEFAULT_ODDS)
        under_odds = market_data.get("total_under_odds", BETTING.DEFAULT_ODDS)
        home_ml = market_data.get("home_moneyline", 0)
        away_ml = market_data.get("away_moneyline", 0)

        spread_home_ev = self.expected_value(spread_cover, home_spread_odds)
        spread_away_ev = self.expected_value(1 - spread_cover, away_spread_odds)
        total_over_ev = self.expected_value(over_prob, over_odds)
        total_under_ev = self.expected_value(under_prob, under_odds)
        ml_home_ev = self.expected_value(win_prob, home_ml) if home_ml else 0
        ml_away_ev = self.expected_value(1 - win_prob, away_ml) if away_ml else 0

        ev_metrics = {
            "available": True,
            "win_probability": {"home": round(win_prob, 4), "away": round(1 - win_prob, 4)},
            "spread": {
                "line": spread,
                "home_cover_prob": round(spread_cover, 4),
                "home_ev": spread_home_ev,
                "away_ev": spread_away_ev,
                "best_bet": "HOME" if spread_home_ev > spread_away_ev else "AWAY",
                "best_ev": max(spread_home_ev, spread_away_ev),
            },
            "total": {
                "line": total,
                "over_prob": round(over_prob, 4),
                "under_prob": round(under_prob, 4),
                "over_ev": total_over_ev,
                "under_ev": total_under_ev,
                "best_bet": "OVER" if total_over_ev > total_under_ev else "UNDER",
                "best_ev": max(total_over_ev, total_under_ev),
            },
            "moneyline": {
                "home_odds": home_ml,
                "away_odds": away_ml,
                "home_ev": ml_home_ev,
                "away_ev": ml_away_ev,
                "best_bet": "HOME" if ml_home_ev > ml_away_ev else "AWAY",
                "best_ev": max(ml_home_ev, ml_away_ev),
            },
        }

        # Find single best bet across all markets
        all_bets = [
            ("SPREAD HOME", spread_home_ev),
            ("SPREAD AWAY", spread_away_ev),
            ("OVER", total_over_ev),
            ("UNDER", total_under_ev),
            ("ML HOME", ml_home_ev),
            ("ML AWAY", ml_away_ev),
        ]
        best = max(all_bets, key=lambda x: x[1])
        ev_metrics["best_overall"] = {"bet": best[0], "ev": best[1]}

        return {
            "home_score": round(cal_home, 1),
            "away_score": round(cal_away, 1),
            "home_std": home_std,
            "away_std": away_std,
            "regression_applied": True,
            "regression_strength": round(rs, 3),
            "market_home": round(market_home, 1),
            "market_away": round(market_away, 1),
            "ev_metrics": ev_metrics,
        }

    # ── Estimate Points Std ──────────────────────────────────────────────

    @staticmethod
    def estimate_points_std(pts: float, minutes: float) -> float:
        """Heuristic std when we don't have historical data."""
        if minutes < 15:
            rel = PROPS.STD_LOW_MINUTES
        elif minutes < 25:
            rel = PROPS.STD_MID_MINUTES
        elif minutes <= 32:
            rel = PROPS.STD_HIGH_MINUTES
        else:
            rel = PROPS.STD_STAR_MINUTES
        return max(PROPS.MIN_POINTS_STD, pts * rel)

    # ══════════════════════════════════════════════════════════════════════
    # PLAYER-TO-TEAM SUM CONSTRAINT
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def normalize_player_projections(
        players: list,
        team_total: float,
        pts_key: str = "points",
        std_key: str = "points_std",
    ) -> list:
        """
        Rescale player point projections so they sum to the team total.

        This ensures internal consistency: if the team is predicted to score
        112, the individual player predictions add up to 112.

        The rescaling is proportional — each player's share of the total
        is preserved, and standard deviations are scaled by the same factor
        to maintain relative uncertainty.

        Args:
            players: list of dicts, each with pts_key and std_key
            team_total: predicted team total points
            pts_key: key for the points value in each dict
            std_key: key for the std value in each dict

        Returns:
            Same list with pts_key and std_key rescaled.
        """
        if not players or team_total <= 0:
            return players

        raw_sum = sum(p.get(pts_key, 0) for p in players)
        if raw_sum <= 0:
            return players

        scale_factor = team_total / raw_sum

        for p in players:
            raw_pts = p.get(pts_key, 0)
            raw_std = p.get(std_key, 0)
            p[pts_key] = round(raw_pts * scale_factor, 1)
            p[std_key] = round(raw_std * scale_factor, 2)

        return players

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATED BET DETECTION & PORTFOLIO KELLY
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def detect_correlated_bets(game_bets: list, prop_bets: list) -> list:
        """
        Identify groups of correlated bets within the same game.

        Correlation rules:
        - A team OVER total + player OVER from the SAME team are positively
          correlated (if the game goes over, high-scoring players benefit).
        - A team UNDER total + player UNDER from the same team are correlated.
        - A team spread bet + player props from the favored team are correlated
          (favored team winning big usually means their stars scored a lot).
        - Multiple player OVERs from the same team are correlated.

        Args:
            game_bets: list of dicts with keys: type (SPREAD/TOTAL/MONEYLINE),
                       bet (HOME/AWAY/OVER/UNDER), ev, kelly_fraction
            prop_bets: list of dicts with keys: player, team (home/away),
                       best_bet (OVER/UNDER), ev, kelly_fraction

        Returns:
            list of correlation warnings, each a dict with:
                group: list of bet descriptions
                correlation: estimated correlation coefficient (0-1)
                adjusted_kelly: recommended combined Kelly after haircut
                reason: human-readable explanation
        """
        warnings = []

        # Group prop bets by team and direction
        home_overs = [b for b in prop_bets if b.get("team") == "home" and b.get("best_bet") == "OVER"]
        away_overs = [b for b in prop_bets if b.get("team") == "away" and b.get("best_bet") == "OVER"]
        home_unders = [b for b in prop_bets if b.get("team") == "home" and b.get("best_bet") == "UNDER"]
        away_unders = [b for b in prop_bets if b.get("team") == "away" and b.get("best_bet") == "UNDER"]

        # Check: Total OVER + same-team player OVERs
        total_bets = [b for b in game_bets if b.get("type") == "TOTAL"]
        for tb in total_bets:
            if tb.get("bet") == "OVER":
                for group, label in [(home_overs, "home"), (away_overs, "away")]:
                    if len(group) >= 1:
                        warnings.append(_build_correlation_warning(
                            anchor=f"Game OVER total",
                            correlated=[f"{b['player']} OVER" for b in group],
                            correlation=0.45,
                            kellys=[tb.get("kelly_fraction", 0)] + [b.get("kelly_fraction", 0) for b in group],
                            reason=(
                                f"Game OVER and {label}-team player OVERs are positively correlated — "
                                f"a high-scoring game lifts all scorers on both teams."
                            ),
                        ))
            elif tb.get("bet") == "UNDER":
                for group, label in [(home_unders, "home"), (away_unders, "away")]:
                    if len(group) >= 1:
                        warnings.append(_build_correlation_warning(
                            anchor=f"Game UNDER total",
                            correlated=[f"{b['player']} UNDER" for b in group],
                            correlation=0.40,
                            kellys=[tb.get("kelly_fraction", 0)] + [b.get("kelly_fraction", 0) for b in group],
                            reason=(
                                f"Game UNDER and {label}-team player UNDERs are positively correlated — "
                                f"a low-scoring game suppresses individual scorers."
                            ),
                        ))

        # Check: Multiple player OVERs on the same team
        for group, label in [(home_overs, "home"), (away_overs, "away")]:
            if len(group) >= 2:
                warnings.append(_build_correlation_warning(
                    anchor=f"{group[0]['player']} OVER",
                    correlated=[f"{b['player']} OVER" for b in group[1:]],
                    correlation=0.30,
                    kellys=[b.get("kelly_fraction", 0) for b in group],
                    reason=(
                        f"Multiple player OVERs on the {label} team are correlated — "
                        f"they share the same game pace and scoring environment."
                    ),
                ))

        # Check: Spread + player props on favored team
        spread_bets = [b for b in game_bets if b.get("type") == "SPREAD"]
        for sb in spread_bets:
            favored_side = sb.get("bet", "").upper()  # HOME or AWAY
            if favored_side == "HOME" and home_overs:
                warnings.append(_build_correlation_warning(
                    anchor=f"Spread {favored_side}",
                    correlated=[f"{b['player']} OVER" for b in home_overs],
                    correlation=0.35,
                    kellys=[sb.get("kelly_fraction", 0)] + [b.get("kelly_fraction", 0) for b in home_overs],
                    reason=(
                        f"Spread on home team + their player OVERs are correlated — "
                        f"the team covering usually means their stars performed well."
                    ),
                ))
            elif favored_side == "AWAY" and away_overs:
                warnings.append(_build_correlation_warning(
                    anchor=f"Spread {favored_side}",
                    correlated=[f"{b['player']} OVER" for b in away_overs],
                    correlation=0.35,
                    kellys=[sb.get("kelly_fraction", 0)] + [b.get("kelly_fraction", 0) for b in away_overs],
                    reason=(
                        f"Spread on away team + their player OVERs are correlated — "
                        f"the team covering usually means their stars performed well."
                    ),
                ))

        return warnings

    @staticmethod
    def portfolio_kelly(
        individual_kellys: list,
        correlation: float,
        max_total_exposure: float = None,
    ) -> list:
        """
        Reduce Kelly fractions for a group of correlated bets.

        For perfectly independent bets, total Kelly = sum of individual Kellys.
        For correlated bets, we apply a haircut:

            adjusted_total = original_total × (1 - correlation × overlap_factor)

        where overlap_factor increases with the number of correlated bets.

        Args:
            individual_kellys: list of Kelly fractions for each bet
            correlation: estimated correlation (0–1)
            max_total_exposure: maximum combined Kelly (default: 2× single max)

        Returns:
            list of adjusted Kelly fractions (same order as input)
        """
        max_total = max_total_exposure or (BETTING.MAX_KELLY_BET_PCT / 100.0 * 2)

        if not individual_kellys:
            return []

        original_total = sum(individual_kellys)
        if original_total <= 0:
            return individual_kellys

        n = len(individual_kellys)
        # Overlap factor: scales with number of bets (more bets = more overlap risk)
        overlap_factor = min(1.0, (n - 1) / 4.0)  # caps at 1.0 for 5+ bets
        discount = 1.0 - correlation * overlap_factor

        adjusted_total = original_total * discount
        adjusted_total = min(adjusted_total, max_total)

        # Scale each bet proportionally
        scale = adjusted_total / original_total if original_total > 0 else 1.0
        return [round(k * scale, 4) for k in individual_kellys]


def _build_correlation_warning(
    anchor: str,
    correlated: list,
    correlation: float,
    kellys: list,
    reason: str,
) -> dict:
    """Helper to build a structured correlation warning."""
    adjusted = EVEngine.portfolio_kelly(kellys, correlation)
    original_total = sum(kellys)
    adjusted_total = sum(adjusted)

    return {
        "group": [anchor] + correlated,
        "correlation": round(correlation, 2),
        "original_kelly_total": round(original_total, 4),
        "adjusted_kelly_total": round(adjusted_total, 4),
        "reduction_pct": round((1 - adjusted_total / max(0.0001, original_total)) * 100, 1),
        "adjusted_kellys": adjusted,
        "reason": reason,
    }
