import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class OddsCalibrator:
    """
    A class for calibrating model predictions using betting market odds data
    to better align predictions and identify potential value.
    """
    
    def __init__(self, regression_strength: float = 0.5):
        """
        Initialize the OddsCalibrator
        
        Args:
            regression_strength: Default strength of regression toward market (0-1)
                                0 = no regression, 1 = full regression to market
        """
        self.regression_strength = regression_strength
        
    def implied_total_from_odds(self, total_line: float, 
                              over_odds: int, under_odds: int) -> float:
        """
        Calculate the implied total from over/under odds
        
        Args:
            total_line: The posted total line
            over_odds: American odds for the over
            under_odds: American odds for the under
            
        Returns:
            Implied total score based on odds skew
        """
        # Convert odds to probabilities
        over_prob = self._american_odds_to_probability(over_odds)
        under_prob = self._american_odds_to_probability(under_odds)
        
        # Normalize probabilities
        sum_prob = over_prob + under_prob
        over_prob_norm = over_prob / sum_prob
        under_prob_norm = under_prob / sum_prob
        
        # Calculate how much the line should adjust based on probability skew
        # More extreme skews should move the line more
        skew = over_prob_norm - 0.5  # Positive if over is more likely
        
        # Calculate adjustment - larger when odds are more skewed
        # Typically 2-4 points for significant odds differences
        max_adjustment = 3.0
        adjustment = skew * max_adjustment * 2 # Multiply by 2 to get stronger effect
        
        return total_line + adjustment
    
    def implied_spread_from_odds(self, spread_line: float, 
                               favorite_odds: int, underdog_odds: int) -> float:
        """
        Calculate the implied spread from spread betting odds
        
        Args:
            spread_line: The posted spread line (positive = home team is underdog)
            favorite_odds: American odds for the favorite
            underdog_odds: American odds for the underdog
            
        Returns:
            Implied spread based on odds skew
        """
        # Convert odds to probabilities
        fav_prob = self._american_odds_to_probability(favorite_odds)
        dog_prob = self._american_odds_to_probability(underdog_odds)
        
        # Normalize probabilities
        sum_prob = fav_prob + dog_prob
        fav_prob_norm = fav_prob / sum_prob
        dog_prob_norm = dog_prob / sum_prob
        
        # Calculate how much the line should adjust based on probability skew
        skew = fav_prob_norm - 0.5  # Positive if favorite is more likely
        
        # Calculate adjustment - larger when odds are more skewed
        max_adjustment = 2.0
        adjustment = skew * max_adjustment * 2  # Multiply by 2 to get stronger effect
        
        # Adjust spread (negative adjustment means spread should be larger)
        # If spread_line is positive (home is underdog), adjustment sign flips
        return spread_line - adjustment * np.sign(spread_line)
    
    def _american_odds_to_probability(self, odds: int) -> float:
        """Convert American odds to probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def calculate_team_std_from_odds(self, spread_line: float, 
                                  spread_odds: Tuple[int, int], 
                                  total_line: float,
                                  total_odds: Tuple[int, int]) -> Tuple[float, float]:
        """
        Estimate team score standard deviations from betting market odds
        
        Args:
            spread_line: Point spread (positive means home team is underdog)
            spread_odds: Tuple of (favorite odds, underdog odds)
            total_line: Over/under line
            total_odds: Tuple of (over odds, under odds)
            
        Returns:
            Tuple of (home_std, away_std) standard deviations
        """
        # NBA games typically have standard deviations around 10-12 points
        # Use odds to adjust this baseline
        base_std = 11.0
        
        # Tighter spreads or totals with balanced odds suggest higher variance
        fav_prob = self._american_odds_to_probability(spread_odds[0])
        dog_prob = self._american_odds_to_probability(spread_odds[1])
        
        # Measure how even the odds are (closer to 0 means more balanced)
        spread_balance = abs(fav_prob - dog_prob) 
        
        # For more balanced games (odds near even), increase standard deviation
        # For heavily favored teams, decrease standard deviation
        std_adjustment = 1.0 - spread_balance * 2  # Scale to roughly ±15%
        
        # Calculate final standard deviation
        game_std = base_std * (1 + std_adjustment)
        
        # Split between home and away teams - favored teams are typically more consistent
        if spread_line > 0:  # Home team is underdog
            home_std = game_std * 1.1  # Underdogs are more variable
            away_std = game_std * 0.9  # Favorites are more consistent
        else:  # Home team is favorite
            home_std = game_std * 0.9
            away_std = game_std * 1.1
            
        return (home_std, away_std)
    
    def calibrate_prediction(self, prediction: Dict, market_data: Dict) -> Dict:
        """
        Calibrate a prediction using betting market data
        
        Args:
            prediction: Raw prediction dictionary with team scores
            market_data: Dictionary of market odds data
            
        Returns:
            Calibrated prediction with market-based adjustments
        """
        if not market_data or not market_data.get('available', False):
            # If no market data, return prediction with default standard deviations
            home_score = prediction['home_score']
            away_score = prediction['away_score']
            
            # Add default standard deviations
            prediction['home_std'] = 11.0
            prediction['away_std'] = 11.0
            
            return prediction
        
        # Extract market data
        home_spread = market_data.get('home_spread', 0)
        total = market_data.get('total', 0)
        
        # Extract odds for spread and total
        spread_odds = (
            market_data.get('home_spread_odds', -110), 
            market_data.get('away_spread_odds', -110)
        )
        total_odds = (
            market_data.get('total_over_odds', -110),
            market_data.get('total_under_odds', -110)
        )
        
        # Calculate implied market expectations
        implied_spread = self.implied_spread_from_odds(home_spread, spread_odds[0], spread_odds[1])
        implied_total = self.implied_total_from_odds(total, total_odds[0], total_odds[1])
        
        # Calculate market expectations for team scores
        market_home_score = (implied_total + implied_spread) / 2
        market_away_score = (implied_total - implied_spread) / 2
        
        # Blend model prediction with market expectations based on regression strength
        rs = self.regression_strength
        calibrated_home_score = (1 - rs) * prediction['home_score'] + rs * market_home_score
        calibrated_away_score = (1 - rs) * prediction['away_score'] + rs * market_away_score
        
        # Calculate standard deviations
        home_std, away_std = self.calculate_team_std_from_odds(
            home_spread, spread_odds, total, total_odds
        )
        
        # Update prediction
        calibrated = prediction.copy()
        calibrated['home_score'] = calibrated_home_score
        calibrated['away_score'] = calibrated_away_score
        calibrated['home_std'] = home_std
        calibrated['away_std'] = away_std
        calibrated['market_home_score'] = market_home_score
        calibrated['market_away_score'] = market_away_score
        calibrated['regression_applied'] = True
        calibrated['regression_strength'] = rs
        
        # Add regression details for transparency
        calibrated['regression_details'] = {
            'original_home': prediction['home_score'],
            'original_away': prediction['away_score'],
            'market_home': market_home_score,
            'market_away': market_away_score,
            'implied_spread': implied_spread,
            'implied_total': implied_total
        }
        
        return calibrated
    
    def calculate_ev_metrics(self, calibrated: Dict, market_data: Dict, 
                          probabilistic_model) -> Dict:
        """
        Calculate expected value metrics for betting opportunities
        
        Args:
            calibrated: Calibrated prediction with standard deviations
            market_data: Dictionary of market odds data
            probabilistic_model: Instance of ProbabilisticModel
            
        Returns:
            Dictionary with EV metrics for various bet types
        """
        if not market_data or not market_data.get('available', False):
            return {'available': False}
            
        # Generate score distributions
        home_dist = probabilistic_model.generate_score_distribution(
            calibrated['home_score'], calibrated['home_std']
        )
        away_dist = probabilistic_model.generate_score_distribution(
            calibrated['away_score'], calibrated['away_std']
        )
        
        # Extract market lines and odds
        spread = market_data.get('home_spread', 0)
        total = market_data.get('total', 0)
        home_ml = market_data.get('home_moneyline', 0)
        away_ml = market_data.get('away_moneyline', 0)
        
        # Calculate probabilities
        spread_prob = probabilistic_model.calculate_spread_probability(home_dist, away_dist, spread)
        total_over_prob, total_under_prob = probabilistic_model.calculate_total_probability(
            home_dist, away_dist, total
        )
        win_prob = probabilistic_model.calculate_win_probability(home_dist, away_dist)
        
        # Calculate expected values
        spread_home_ev = probabilistic_model.calculate_expected_value(
            spread_prob, market_data.get('home_spread_odds', -110)
        )
        spread_away_ev = probabilistic_model.calculate_expected_value(
            1 - spread_prob, market_data.get('away_spread_odds', -110)
        )
        total_over_ev = probabilistic_model.calculate_expected_value(
            total_over_prob, market_data.get('total_over_odds', -110)
        )
        total_under_ev = probabilistic_model.calculate_expected_value(
            total_under_prob, market_data.get('total_under_odds', -110)
        )
        ml_home_ev = probabilistic_model.calculate_expected_value(win_prob, home_ml) if home_ml else 0
        ml_away_ev = probabilistic_model.calculate_expected_value(1 - win_prob, away_ml) if away_ml else 0
        
        # Calculate confidence intervals
        home_ci = probabilistic_model.generate_confidence_interval(home_dist)
        away_ci = probabilistic_model.generate_confidence_interval(away_dist)
        total_ci = probabilistic_model.generate_confidence_interval(home_dist + away_dist)
        margin_ci = probabilistic_model.generate_confidence_interval(home_dist - away_dist)
        
        # Compile results
        ev_metrics = {
            'available': True,
            'win_probability': {
                'home': win_prob,
                'away': 1 - win_prob
            },
            'spread': {
                'line': spread,
                'home_cover_prob': spread_prob,
                'away_cover_prob': 1 - spread_prob,
                'home_ev': spread_home_ev,
                'away_ev': spread_away_ev,
                'best_bet': 'HOME' if spread_home_ev > spread_away_ev else 'AWAY',
                'best_ev': max(spread_home_ev, spread_away_ev)
            },
            'total': {
                'line': total,
                'over_prob': total_over_prob,
                'under_prob': total_under_prob,
                'over_ev': total_over_ev,
                'under_ev': total_under_ev,
                'best_bet': 'OVER' if total_over_ev > total_under_ev else 'UNDER',
                'best_ev': max(total_over_ev, total_under_ev)
            },
            'moneyline': {
                'home_odds': home_ml,
                'away_odds': away_ml,
                'home_ev': ml_home_ev,
                'away_ev': ml_away_ev,
                'best_bet': 'HOME' if ml_home_ev > ml_away_ev else 'AWAY',
                'best_ev': max(ml_home_ev, ml_away_ev)
            },
            'confidence_intervals': {
                'home_score': home_ci,
                'away_score': away_ci,
                'total': total_ci,
                'margin': margin_ci
            }
        }
        
        # Find best overall bet
        best_evs = [
            ('SPREAD HOME', spread_home_ev),
            ('SPREAD AWAY', spread_away_ev),
            ('TOTAL OVER', total_over_ev),
            ('TOTAL UNDER', total_under_ev),
            ('ML HOME', ml_home_ev),
            ('ML AWAY', ml_away_ev)
        ]
        
        best_bet = max(best_evs, key=lambda x: x[1])
        ev_metrics['best_overall_bet'] = {
            'bet': best_bet[0],
            'ev': best_bet[1]
        }
        
        return ev_metrics