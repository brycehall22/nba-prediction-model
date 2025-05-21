import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

class ProbabilisticModel:
    """
    A framework for probability-based predictions that can be integrated with betting markets
    to identify positive expected value opportunities.
    """
    
    def __init__(self):
        self.confidence_interval = 0.95  # Default 95% confidence interval
    
    def generate_score_distribution(self, mean_score: float, std_dev: float, 
                                  samples: int = 10000) -> np.ndarray:
        """
        Generate a probability distribution of possible scores
        
        Args:
            mean_score: Predicted mean score
            std_dev: Standard deviation of the score prediction
            samples: Number of samples to generate
            
        Returns:
            Array of sampled possible scores
        """
        # Use truncated normal distribution to avoid negative scores
        lower_bound = 0  # Scores can't be negative
        upper_bound = 200  # Reasonable upper limit for NBA scores
        
        distribution = stats.truncnorm(
            (lower_bound - mean_score) / std_dev,
            (upper_bound - mean_score) / std_dev,
            loc=mean_score,
            scale=std_dev
        )
        
        return distribution.rvs(size=samples)
    
    def calculate_spread_probability(self, home_dist: np.ndarray, away_dist: np.ndarray, 
                                   spread: float) -> float:
        """
        Calculate probability of covering a spread
        
        Args:
            home_dist: Distribution of home team scores
            away_dist: Distribution of away team scores
            spread: Point spread (positive for home team as underdog)
            
        Returns:
            Probability of home team covering the spread
        """
        # Calculate margin distribution (home - away)
        margin_dist = home_dist - away_dist
        
        # Calculate probability of margin > spread
        prob_cover = np.mean(margin_dist > spread)
        
        return prob_cover
    
    def calculate_total_probability(self, home_dist: np.ndarray, away_dist: np.ndarray,
                                  total_line: float) -> Tuple[float, float]:
        """
        Calculate probability of total going over/under
        
        Args:
            home_dist: Distribution of home team scores
            away_dist: Distribution of away team scores
            total_line: Over/under line
            
        Returns:
            Tuple of (over_probability, under_probability)
        """
        # Calculate total points distribution
        total_dist = home_dist + away_dist
        
        # Calculate probabilities
        prob_over = np.mean(total_dist > total_line)
        prob_under = 1 - prob_over
        
        return (prob_over, prob_under)
    
    def calculate_win_probability(self, home_dist: np.ndarray, away_dist: np.ndarray) -> float:
        """
        Calculate probability of home team winning
        
        Args:
            home_dist: Distribution of home team scores
            away_dist: Distribution of away team scores
            
        Returns:
            Probability of home team winning
        """
        # Calculate margin distribution (home - away)
        margin_dist = home_dist - away_dist
        
        # Calculate probability of margin > 0 (home win)
        prob_home_win = np.mean(margin_dist > 0)
        
        return prob_home_win
    
    def calculate_expected_value(self, probability: float, odds: int) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            probability: Model's probability of the outcome
            odds: American odds for the outcome
            
        Returns:
            Expected value as a percentage (positive is +EV)
        """
        if odds == 0:
            return 0
            
        # Convert odds to implied probability
        if odds > 0:
            implied_prob = 100 / (odds + 100)
        else:
            implied_prob = abs(odds) / (abs(odds) + 100)
            
        # Calculate edge (difference between model probability and implied probability)
        edge = probability - implied_prob
        
        # Calculate expected value (EV)
        if odds > 0:
            ev = edge * odds / 100
        else:
            ev = edge * 100 / abs(odds)
            
        return ev * 100  # Return as percentage
    
    def generate_confidence_interval(self, distribution: np.ndarray) -> Tuple[float, float]:
        """
        Generate confidence interval for a distribution
        
        Args:
            distribution: Array of sampled values
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower_percentile = (1 - self.confidence_interval) / 2 * 100
        upper_percentile = (1 + self.confidence_interval) / 2 * 100
        
        lower_bound = np.percentile(distribution, lower_percentile)
        upper_bound = np.percentile(distribution, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def calculate_player_prop_probability(self, predicted_points: float, std_dev: float, 
                                        line: float) -> Tuple[float, float]:
        """
        Calculate probability of a player going over/under their points prop
        
        Args:
            predicted_points: Model's predicted mean points
            std_dev: Standard deviation of the prediction
            line: Points prop line
            
        Returns:
            Tuple of (over_probability, under_probability)
        """
        # Generate distribution of player points
        points_dist = self.generate_score_distribution(predicted_points, std_dev, samples=10000)
        
        # Calculate probabilities
        prob_over = np.mean(points_dist > line)
        prob_under = 1 - prob_over
        
        return (prob_over, prob_under)
    
    def find_prop_edges(self, player_predictions: Dict, props_data: Dict) -> Dict:
        """
        Find edges in player props based on model predictions vs. market odds
        
        Args:
            player_predictions: Dictionary of player predictions with standard deviations
            props_data: Dictionary of player props with lines and odds
            
        Returns:
            Dictionary of players with positive expected value bets
        """
        prop_edges = {}
        
        for player_name, prediction in player_predictions.items():
            if player_name not in props_data:
                continue
                
            prop = props_data[player_name]
            line = prop['line']
            
            # Calculate over/under probabilities
            over_prob, under_prob = self.calculate_player_prop_probability(
                prediction['points'], prediction['std_dev'], line
            )
            
            # Calculate expected values
            over_ev = self.calculate_expected_value(over_prob, prop['over_odds'])
            under_ev = self.calculate_expected_value(under_prob, prop['under_odds'])
            
            # If either bet has positive EV, add to results
            if over_ev > 0 or under_ev > 0:
                prop_edges[player_name] = {
                    'line': line,
                    'predicted_points': prediction['points'],
                    'over_probability': over_prob,
                    'under_probability': under_prob,
                    'over_odds': prop['over_odds'],
                    'over_ev': over_ev,
                    'under_odds': prop['under_odds'],
                    'under_ev': under_ev,
                    'best_bet': 'OVER' if over_ev > under_ev else 'UNDER',
                    'best_ev': max(over_ev, under_ev),
                    'raw_edge': abs(prediction['points'] - line)
                }
                
        return prop_edges