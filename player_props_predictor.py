import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Tuple, Union, Optional
import random

logger = logging.getLogger(__name__)

class PlayerPropsPredictor:
    """
    Enhanced system for predicting player props with probabilities and expected value calculations.
    Focuses on improved minutes projections and point predictions.
    """
    
    def __init__(self, collector, model_trainer=None):
        """
        Initialize the player props predictor
        
        Args:
            collector: DataCollector instance for fetching player data
            model_trainer: Optional ModelTrainer instance for using trained models
        """
        self.collector = collector
        self.model_trainer = model_trainer
        self.use_model = model_trainer is not None
        
        # Constants and settings
        self.min_player_games = 5  # Minimum recent games to use for player analysis
        self.recent_games_weight = 0.7  # Weight for recent games vs season averages
        self.home_boost_factor = 1.03  # Home players score about 3% more
        self.pts_per_minute_bounds = (0.15, 1.2)  # Reasonable bounds for pts/min
        
        # Standard deviation parameters
        self.base_std_per_minute = 0.4  # Base standard deviation per minute played
        self.std_reduction_with_sample = 0.4  # How much std decreases with more samples
        
    def predict_player_points(self, player_id: int, 
                            is_home: bool = True, 
                            opp_team_id: int = None, 
                            minutes_override: float = None) -> Dict:
        """
        Generate probabilistic points prediction for a player
        
        Args:
            player_id: NBA API player ID
            is_home: Whether player is on home team
            opp_team_id: Opponent team ID for matchup analysis
            minutes_override: Override projected minutes (if None, will be predicted)
            
        Returns:
            Dictionary with points prediction and confidence metrics
        """
        try:
            # Get player's recent game data
            recent_games = self.collector.get_player_recent_games(player_id, n_games=15)
            if recent_games is None or len(recent_games) < self.min_player_games:
                return self._generate_fallback_prediction(player_id, is_home, minutes_override)
            
            # Get player stats from collector
            player_stats = self.collector.get_player_stats(player_id)
            if player_stats is None:
                return self._generate_fallback_prediction(player_id, is_home, minutes_override)
            
            # Check if player is inactive
            if player_stats.get('INACTIVE', False) or player_stats.get('GAMES_PLAYED', 0) == 0:
                return {
                    'points': 0.0,
                    'points_std': 0.0,
                    'minutes': 0.0,
                    'minutes_std': 0.0,
                    'status': 'Inactive',
                    'projection_quality': 0.0
                }
            
            # Compute projected minutes with enhanced algorithm
            minutes_projection, minutes_std = self._project_minutes(
                recent_games, player_stats, minutes_override, is_home
            )
            
            # Calculate recent points per minute
            recent_games['PTS_PER_MIN'] = recent_games.apply(
                lambda x: x['PTS'] / max(1, x['MIN']), axis=1
            )
            
            # Get average points per minute (weighted toward recent games)
            pts_per_min_list = recent_games['PTS_PER_MIN'].tolist()
            if len(pts_per_min_list) > 5:
                # Weight recent games more heavily
                weights = np.exp(np.linspace(0, -1, len(pts_per_min_list)))
                weights = weights / weights.sum()
                pts_per_min = np.average(pts_per_min_list, weights=weights)
            else:
                pts_per_min = np.mean(pts_per_min_list)
            
            # Apply bounds to points per minute
            pts_per_min = max(self.pts_per_minute_bounds[0], 
                             min(self.pts_per_minute_bounds[1], pts_per_min))
            
            # Calculate standard deviation of points per minute
            pts_per_min_std = max(0.1, np.std(pts_per_min_list))
            
            # Adjust for sample size - smaller samples have higher uncertainty
            sample_size_factor = min(1.0, len(recent_games) / 20)
            pts_per_min_std = pts_per_min_std * (1 + (1 - sample_size_factor))
            
            # Apply adjustments for matchup and home court
            if is_home:
                pts_per_min *= self.home_boost_factor
            
            # Apply opponent defense adjustment
            if opp_team_id is not None:
                pts_per_min = self._apply_opponent_adjustment(
                    pts_per_min, opp_team_id, player_stats
                )
            
            # Calculate projected points
            points_projection = pts_per_min * minutes_projection
            
            # Calculate standard deviation of points
            # Uncertainty comes from both minutes and scoring rate variability
            # Use propagation of uncertainty formula
            points_std = np.sqrt(
                (minutes_projection * pts_per_min_std)**2 + 
                (minutes_std * pts_per_min)**2
            )
            
            # Reduce standard deviation for players with more games (more predictable)
            games_played_factor = min(1.0, self.min_player_games / max(1, player_stats.get('GAMES_PLAYED', 0)))
            points_std *= (0.8 + 0.2 * games_played_factor)
            
            # Calculate projection quality score (0-1)
            projection_quality = self._calculate_projection_quality(
                recent_games, player_stats, points_std / max(1, points_projection)
            )
            
            # Build result dictionary
            result = {
                'points': round(points_projection, 1),
                'points_std': round(points_std, 2),
                'minutes': round(minutes_projection, 1),
                'minutes_std': round(minutes_std, 2),
                'pts_per_min': round(pts_per_min, 3),
                'status': 'Active',
                'projection_quality': round(projection_quality, 2)
            }
            
            # Add confidence intervals
            ci_lower = max(0, points_projection - 1.96 * points_std)
            ci_upper = points_projection + 1.96 * points_std
            result['points_95ci'] = (round(ci_lower, 1), round(ci_upper, 1))
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting player points: {str(e)}")
            return self._generate_fallback_prediction(player_id, is_home, minutes_override)
    
    def _project_minutes(self, recent_games: pd.DataFrame, 
                       player_stats: Dict, 
                       minutes_override: float = None,
                       is_home: bool = True) -> Tuple[float, float]:
        """
        Project minutes for a player with improved algorithm
        
        Args:
            recent_games: DataFrame of player's recent games
            player_stats: Dictionary of player statistics
            minutes_override: Override minutes projection
            is_home: Whether player is on home team
            
        Returns:
            Tuple of (projected_minutes, minutes_std)
        """
        if minutes_override is not None:
            # Use override with reduced standard deviation
            return minutes_override, max(2.0, minutes_override * 0.15)
        
        # Filter out DNPs for better minutes projection
        if 'DNP' in recent_games.columns:
            active_games = recent_games[~recent_games['DNP']]
        else:
            active_games = recent_games[recent_games['MIN'] > 0]
        
        if len(active_games) == 0:
            return 0.0, 0.0
        
        # Check for minutes trend over last 5 games
        last_5_games = active_games.head(5)
        if len(last_5_games) >= 3:
            # Calculate trend
            x = np.arange(len(last_5_games))
            y = last_5_games['MIN'].values
            
            # Use linear regression to detect trend
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Apply trend adjustment (max ±5 minutes)
            trend_adjustment = max(-5, min(5, slope * 3))
        else:
            trend_adjustment = 0
        
        # Calculate minutes using weighted average of recent games
        recent_min = active_games['MIN'].mean()
        
        # Check if player has reduced minutes in very recent games (last 3)
        if len(active_games) >= 5:
            very_recent = active_games.head(3)['MIN'].mean()
            season_avg = active_games['MIN'].mean()
            
            # If recent minutes are significantly lower, might indicate 
            # injury or reduced role
            if very_recent < 0.7 * season_avg and season_avg > 15:
                # Weight recent games more heavily
                minutes_projection = very_recent * 0.7 + season_avg * 0.3
            else:
                # Normal weighting with trend adjustment
                minutes_projection = recent_min + trend_adjustment
        else:
            minutes_projection = recent_min
            
        # Slight boost for home games
        if is_home:
            minutes_projection *= 1.02  # 2% more minutes at home
        
        # Calculate standard deviation of minutes
        minutes_std = max(2.0, active_games['MIN'].std())
        
        # Reduce standard deviation if player has consistent minutes
        # Calculate coefficient of variation (CV) - lower is more consistent
        cv = minutes_std / max(1.0, recent_min)
        consistency_factor = min(1.0, cv / 0.3)  # Normalize: CV of 0.3 is typical
        
        # Adjust std based on consistency (more consistent = lower std)
        adjusted_std = minutes_std * (0.7 + 0.3 * consistency_factor)
        
        return minutes_projection, adjusted_std
    
    def _apply_opponent_adjustment(self, pts_per_min: float, 
                                opp_team_id: int, 
                                player_stats: Dict) -> float:
        """
        Adjust points per minute based on opponent defensive strength
        
        Args:
            pts_per_min: Base points per minute
            opp_team_id: Opponent team ID
            player_stats: Player statistics dictionary
            
        Returns:
            Adjusted points per minute
        """
        try:
            # Get opponent team stats
            opp_stats = self.collector.get_team_stats(opp_team_id)
            if opp_stats is None:
                return pts_per_min
                
            # Get opponent defensive rating and points allowed
            opp_def_rating = opp_stats.get('DEF_RATING', 110.0)
            opp_pts_allowed = opp_stats.get('OPP_PTS_ALLOWED', 110.0)
            
            # Calculate adjustment factor
            # Higher defensive rating = worse defense = more points
            league_avg_def_rating = 110.0
            defense_factor = opp_def_rating / league_avg_def_rating
            
            # Soften the adjustment (only use 60% of the raw adjustment)
            defense_adjustment = ((defense_factor - 1.0) * 0.6) + 1.0
            
            # Apply player-specific position adjustment
            # Some positions are more affected by certain defensive metrics
            position = player_stats.get('POSITION', '')
            
            if position == 'G':  # Guards
                # Guards more affected by perimeter defense
                # Adjust based on 3PT% allowed
                if 'OPP_FG3_PCT' in opp_stats:
                    perimeter_factor = opp_stats['OPP_FG3_PCT'] / 0.36  # League avg ~36%
                    defense_adjustment *= (perimeter_factor * 0.3 + 0.7)  # 30% weight
            elif position == 'C':  # Centers
                # Centers more affected by interior defense
                # Adjust based on blocks and FG% allowed
                if 'BLK' in opp_stats:
                    interior_factor = 1.0 - ((opp_stats['BLK'] / 5.0) * 0.1)  # 5 blocks is good
                    defense_adjustment *= interior_factor
            
            return pts_per_min * defense_adjustment
            
        except Exception as e:
            logger.warning(f"Opponent adjustment error: {str(e)}")
            return pts_per_min
    
    def _calculate_projection_quality(self, recent_games: pd.DataFrame, 
                                   player_stats: Dict,
                                   relative_std: float) -> float:
        """
        Calculate the quality of the projection (0-1 scale)
        
        Args:
            recent_games: DataFrame of player's recent games
            player_stats: Dictionary of player statistics
            relative_std: Standard deviation relative to the prediction
            
        Returns:
            Quality score between 0 and 1
        """
        # Start with base quality
        quality = 0.7
        
        # Adjust based on number of games (more games = better projection)
        games_factor = min(1.0, len(recent_games) / 20)
        quality += games_factor * 0.1
        
        # Adjust based on consistency (lower relative std = better projection)
        consistency_factor = max(0, 1.0 - relative_std * 2)
        quality += consistency_factor * 0.1
        
        # Adjust based on minutes consistency
        if 'CONSISTENCY' in player_stats:
            quality += player_stats['CONSISTENCY'] * 0.1
        
        # Cap at 0-1 range
        return max(0.0, min(1.0, quality))
    
    def _generate_fallback_prediction(self, player_id: int,
                                   is_home: bool,
                                   minutes_override: float = None) -> Dict:
        """
        Generate a fallback prediction when insufficient data is available
        
        Args:
            player_id: Player ID
            is_home: Whether player is on home team
            minutes_override: Override for projected minutes
            
        Returns:
            Dictionary with prediction values
        """
        try:
            # Try to get minimal player info
            player_info = self.collector.get_player_stats(player_id)
            
            if player_info and player_info.get('PTS_AVG') > 0:
                # Use available averages with higher uncertainty
                points = player_info.get('PTS_AVG', 8.0)
                minutes = minutes_override or player_info.get('MIN', 20.0)
                
                # High standard deviations due to limited data
                minutes_std = max(5.0, minutes * 0.3)
                points_std = max(5.0, points * 0.4)
                
                # Apply home court adjustment
                if is_home:
                    points *= self.home_boost_factor
                
                return {
                    'points': round(points, 1),
                    'points_std': round(points_std, 2),
                    'minutes': round(minutes, 1),
                    'minutes_std': round(minutes_std, 2),
                    'pts_per_min': round(points / max(1, minutes), 3),
                    'status': 'Active (limited data)',
                    'projection_quality': 0.4,
                    'points_95ci': (round(max(0, points - 1.96 * points_std), 1), 
                                  round(points + 1.96 * points_std, 1))
                }
                
            else:
                # Generic fallback values
                minutes = minutes_override or 15.0
                points = minutes * 0.4  # Conservative estimate
                
                return {
                    'points': round(points, 1),
                    'points_std': round(points * 0.5, 2),
                    'minutes': round(minutes, 1),
                    'minutes_std': round(minutes * 0.4, 2),
                    'pts_per_min': 0.4,
                    'status': 'Unknown (fallback)',
                    'projection_quality': 0.2,
                    'points_95ci': (round(max(0, points * 0.5), 1), round(points * 1.5, 1))
                }
                
        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            
            # Last resort values
            minutes = minutes_override or 10.0
            points = minutes * 0.35
            
            return {
                'points': round(points, 1),
                'points_std': round(5.0, 2),
                'minutes': round(minutes, 1),
                'minutes_std': round(5.0, 2),
                'pts_per_min': 0.35,
                'status': 'Error (generic fallback)',
                'projection_quality': 0.1,
                'points_95ci': (round(max(0, points - 7.5), 1), round(points + 7.5, 1))
            }
    
    def calculate_prop_probabilities(self, prediction: Dict, line: float) -> Dict:
        """
        Calculate over/under probabilities for a points prop
        
        Args:
            prediction: Player prediction dictionary
            line: Prop line (over/under threshold)
            
        Returns:
            Dictionary with probabilities and confidence metrics
        """
        if 'points' not in prediction or 'points_std' not in prediction:
            return {
                'over_probability': 0.5,
                'under_probability': 0.5,
                'confidence': 0.0,
                'edge': 0.0,
                'distance_in_std': 0.0
            }
        
        # Extract prediction values
        predicted_points = prediction['points']
        points_std = prediction['points_std']
        
        # Calculate z-score for the line
        z_score = (line - predicted_points) / points_std if points_std > 0 else 0
        
        # Calculate probabilities using normal distribution
        over_probability = 1 - stats.norm.cdf(z_score)
        under_probability = stats.norm.cdf(z_score)
        
        # Calculate edge (difference from line)
        edge = abs(predicted_points - line)
        
        # Calculate confidence based on standard deviation and projection quality
        # Distance in standard deviations (higher = more confidence)
        distance_in_std = abs(z_score)
        
        # Base confidence on how far the line is from our projection
        if distance_in_std < 0.25:
            # Very close to the line - low confidence
            confidence = 0.3
        elif distance_in_std < 0.6:
            # Within 0.6 standard deviations - moderate confidence
            confidence = 0.5
        elif distance_in_std < 1.0:
            # Within 1 standard deviation - good confidence
            confidence = 0.7
        else:
            # Beyond 1 standard deviation - high confidence
            confidence = 0.8
        
        # Adjust confidence based on projection quality
        if 'projection_quality' in prediction:
            confidence *= prediction['projection_quality']
        
        return {
            'over_probability': round(over_probability, 3),
            'under_probability': round(under_probability, 3),
            'confidence': round(confidence, 2),
            'edge': round(edge, 1),
            'distance_in_std': round(distance_in_std, 2)
        }
    
    def calculate_expected_value(self, probability: float, odds: int) -> float:
        """
        Calculate the expected value of a bet
        
        Args:
            probability: Model's probability of outcome
            odds: American odds
            
        Returns:
            Expected value as percentage
        """
        if odds == 0:
            return 0.0
            
        # Convert American odds to implied probability
        if odds > 0:
            implied_prob = 100 / (odds + 100)
            decimal_odds = odds/100 + 1
        else:
            implied_prob = abs(odds) / (abs(odds) + 100)
            decimal_odds = 100/abs(odds) + 1
            
        # Calculate edge
        edge = probability - implied_prob
        
        # Calculate EV
        ev = (probability * (decimal_odds - 1)) - (1 - probability)
        
        return round(ev * 100, 2)  # Convert to percentage
    
    def evaluate_prop_bet(self, prediction: Dict, prop_data: Dict) -> Dict:
        """
        Evaluate a player prop bet for expected value
        
        Args:
            prediction: Player prediction dictionary
            prop_data: Prop data dictionary with line and odds
            
        Returns:
            Dictionary with evaluation metrics
        """
        if 'points' not in prediction or 'line' not in prop_data:
            return {'valid': False}
            
        # Extract prop data
        line = prop_data['line']
        over_odds = prop_data.get('over_odds', -110)
        under_odds = prop_data.get('under_odds', -110)
        
        # Calculate probabilities
        prob_result = self.calculate_prop_probabilities(prediction, line)
        over_probability = prob_result['over_probability']
        under_probability = prob_result['under_probability']
        
        # Calculate expected values
        over_ev = self.calculate_expected_value(over_probability, over_odds)
        under_ev = self.calculate_expected_value(under_probability, under_odds)
        
        # Determine best bet
        if over_ev > under_ev and over_ev > 0:
            best_bet = 'OVER'
            best_ev = over_ev
            best_prob = over_probability
            best_odds = over_odds
        elif under_ev > 0:
            best_bet = 'UNDER'
            best_ev = under_ev
            best_prob = under_probability
            best_odds = under_odds
        else:
            best_bet = 'NONE'
            best_ev = max(over_ev, under_ev)
            best_prob = over_probability if over_ev > under_ev else under_probability
            best_odds = over_odds if over_ev > under_ev else under_odds
        
        # Kelly criterion calculation
        kelly_multiplier = 0.5  # Fractional Kelly for risk management
        if best_ev > 0:
            if best_odds > 0:
                b = best_odds / 100
            else:
                b = 100 / abs(best_odds)
                
            kelly_fraction = max(0, (best_prob * b - (1 - best_prob)) / b) * kelly_multiplier
        else:
            kelly_fraction = 0
        
        # Create evaluation result
        evaluation = {
            'valid': True,
            'line': line,
            'predicted_points': prediction['points'],
            'points_std': prediction['points_std'],
            'over': {
                'probability': over_probability,
                'odds': over_odds,
                'ev': over_ev
            },
            'under': {
                'probability': under_probability,
                'odds': under_odds,
                'ev': under_ev
            },
            'best_bet': best_bet,
            'best_ev': best_ev,
            'kelly_fraction': round(kelly_fraction, 3),
            'confidence': prob_result['confidence'],
            'value_rating': self._calculate_value_rating(best_ev, prob_result['confidence'])
        }
        
        # Add additional context metrics
        evaluation['edge'] = abs(prediction['points'] - line)
        evaluation['edge_percentage'] = round(evaluation['edge'] / line * 100, 1) if line > 0 else 0
        
        return evaluation
    
    def _calculate_value_rating(self, ev: float, confidence: float) -> int:
        """
        Calculate an overall value rating (1-10 scale)
        
        Args:
            ev: Expected value percentage
            confidence: Confidence in prediction (0-1)
            
        Returns:
            Value rating from 1-10
        """
        # Start with EV-based score
        # EV of 10% -> rating 5, EV of 20% -> rating 8, etc.
        ev_score = min(10, max(1, ev / 4 + 5)) if ev > 0 else 0
        
        # Weight by confidence
        weighted_score = ev_score * confidence
        
        # Floor at 1, ceiling at 10
        return round(max(1, min(10, weighted_score)))