import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import re
import subprocess
import time
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from player_prop_service import PlayerPropsService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('betting_integration')

class BettingTeamMapper:
    """
    Simple and direct mapping between NBA API team names and betting data team names
    """
    
    def __init__(self):
        # Map from full NBA team names to betting CSV format
        self.nba_to_betting = {
            # Eastern Conference
            "Boston Celtics": "BOS Celtics",
            "Brooklyn Nets": "BKN Nets",
            "New York Knicks": "NY Knicks",
            "Philadelphia 76ers": "PHI 76ers",
            "Toronto Raptors": "TOR Raptors",
            "Chicago Bulls": "CHI Bulls",
            "Cleveland Cavaliers": "CLE Cavaliers",
            "Detroit Pistons": "DET Pistons",
            "Indiana Pacers": "IND Pacers",
            "Milwaukee Bucks": "MIL Bucks",
            "Atlanta Hawks": "ATL Hawks",
            "Charlotte Hornets": "CHA Hornets",
            "Miami Heat": "MIA Heat",
            "Orlando Magic": "ORL Magic",
            "Washington Wizards": "WAS Wizards",
            
            # Western Conference
            "Denver Nuggets": "DEN Nuggets",
            "Minnesota Timberwolves": "MIN Timberwolves",
            "Oklahoma City Thunder": "OKC Thunder",
            "Portland Trail Blazers": "POR Trail Blazers",
            "Utah Jazz": "UTA Jazz",
            "Golden State Warriors": "GS Warriors",
            "Los Angeles Clippers": "LA Clippers",
            "Los Angeles Lakers": "LA Lakers",
            "Phoenix Suns": "PHO Suns",
            "Sacramento Kings": "SAC Kings",
            "Dallas Mavericks": "DAL Mavericks",
            "Houston Rockets": "HOU Rockets",
            "Memphis Grizzlies": "MEM Grizzlies",
            "New Orleans Pelicans": "NO Pelicans",
            "San Antonio Spurs": "SA Spurs"
        }
        
        # Create reverse mapping (betting to NBA)
        self.betting_to_nba = {v: k for k, v in self.nba_to_betting.items()}

    def get_betting_name(self, nba_name: str) -> Optional[str]:
        """
        Convert NBA API team name to betting data team name
        
        Args:
            nba_name: Team name from NBA API (e.g., "Boston Celtics")
            
        Returns:
            Betting data team name (e.g., "BOS Celtics") or None if not found
        """
        if nba_name in self.nba_to_betting:
            return self.nba_to_betting[nba_name]
        
        # Log the miss
        logger.warning(f"No betting name match found for NBA name: {nba_name}")
        return None
    
    def get_nba_name(self, betting_name: str) -> Optional[str]:
        """
        Convert betting data team name to NBA API team name
        
        Args:
            betting_name: Team name from betting data (e.g., "BOS Celtics")
            
        Returns:
            NBA API team name (e.g., "Boston Celtics") or None if not found
        """
        if betting_name in self.betting_to_nba:
            return self.betting_to_nba[betting_name]
        
        # Log the miss
        logger.warning(f"No NBA name match found for betting name: {betting_name}")
        return None

class BettingDataService:
    """Service to fetch, parse, and integrate betting data"""
    
    def __init__(self, odds_file_path='draftkings_nba_odds.csv', update_interval_hours=6):
        self.odds_file_path = odds_file_path
        self.update_interval_hours = update_interval_hours
        self.last_update = None
        self.betting_data = None
        self.player_props_service = PlayerPropsService()
    
    def update_betting_data(self, force=False) -> bool:
        """
        Update betting data if needed or if forced
        
        Args:
            force: Force update regardless of last update time
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        current_time = datetime.now()
        
        # Check if update is needed
        if not force and self.last_update and \
           (current_time - self.last_update).total_seconds() < self.update_interval_hours * 3600:
            logger.info("Betting data is still fresh, skipping update")
            return True
        
        try:
            logger.info("Updating betting data...")
            
            # Use subprocess to run the scraper script
            # Note: You may need to adjust this based on your environment
            subprocess.run(["python", "nba_odds_scraper.py"], check=True)
            
            # Check if file exists
            if not os.path.exists(self.odds_file_path):
                logger.error(f"Odds file not found at {self.odds_file_path}")
                return False
            
            # Load the data
            self.betting_data = pd.read_csv(self.odds_file_path)
            self.last_update = current_time
            
            logger.info(f"Successfully updated betting data, found {len(self.betting_data)} games")
            
            # Also update player props data
            self.player_props_service.update_props_data(force=force)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating betting data: {str(e)}")
            return False
    
    def get_betting_lines(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get betting lines for a specific matchup using direct team name mapping
        
        Args:
            home_team: Home team name from NBA API
            away_team: Away team name from NBA API
            
        Returns:
            Dict with betting lines or None if not found
        """
        if self.betting_data is None:
            self.update_betting_data()
            
        if self.betting_data is None:
            return None
        
        try:
            # Initialize the team mapper
            mapper = BettingTeamMapper()
            
            # Convert NBA API team names to betting data format
            betting_home_team = mapper.get_betting_name(home_team)
            betting_away_team = mapper.get_betting_name(away_team)
            
            # Debug logging
            logger.info(f"Looking for: {away_team} @ {home_team}")
            logger.info(f"Converted to betting names: {betting_away_team} @ {betting_home_team}")
            
            # Skip if we couldn't map the team names
            if not betting_home_team or not betting_away_team:
                logger.warning(f"Could not map one or both team names to betting format")
                return None
            
            # Find the game in the betting data
            game_data = self.betting_data[
                (self.betting_data['Home Team'] == betting_home_team) & 
                (self.betting_data['Away Team'] == betting_away_team)
            ]
            
            # If not found, try with case-insensitive comparison
            if game_data.empty:
                logger.info("No exact match, trying case-insensitive and partial comparison...")
                home_betting_name_lower = betting_home_team.lower() if betting_home_team else ""
                away_betting_name_lower = betting_away_team.lower() if betting_away_team else ""
                
                for _, row in self.betting_data.iterrows():
                    row_home_lower = row['Home Team'].lower()
                    row_away_lower = row['Away Team'].lower()
                    
                    # Try partial matching on team abbreviations (first part of name)
                    if (home_betting_name_lower.split()[0] in row_home_lower and 
                        away_betting_name_lower.split()[0] in row_away_lower):
                        game_data = pd.DataFrame([row])
                        break
            
            if game_data.empty:
                logger.warning(f"No betting data found for {betting_away_team} @ {betting_home_team}")
                return None
            
            # Extract the first matching game
            game = game_data.iloc[0]
            
            # Parse spread values
            home_spread = self._parse_spread(game['Home Spread'])
            away_spread = self._parse_spread(game['Away Spread'])
            
            # Parse total
            total = self._parse_total(game['Total'])
            
            # Parse moneylines
            try:
                home_ml = float(game['Home Moneyline'])
            except (ValueError, TypeError):
                home_ml = None
                
            try:
                away_ml = float(game['Away Moneyline'])
            except (ValueError, TypeError):
                away_ml = None
            
            # Calculate implied probabilities from moneylines
            home_prob = self._moneyline_to_probability(home_ml)
            away_prob = self._moneyline_to_probability(away_ml)
            
            # Return structured betting data
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_spread': home_spread['line'] if home_spread else None,
                'home_spread_odds': home_spread['odds'] if home_spread else None,
                'away_spread': away_spread['line'] if away_spread else None,
                'away_spread_odds': away_spread['odds'] if away_spread else None,
                'total': total['line'] if total else None,
                'total_over_odds': total['over_odds'] if total else None,
                'total_under_odds': total['under_odds'] if total else None,
                'home_moneyline': home_ml,
                'away_moneyline': away_ml,
                'home_implied_probability': home_prob,
                'away_implied_probability': away_prob
            }
            
        except Exception as e:
            logger.error(f"Error retrieving betting lines: {str(e)}")
            return None
    
    def _parse_spread(self, spread_text: str) -> Optional[Dict]:
        """
        Parse spread text into components
        
        Args:
            spread_text: Text containing spread and odds (e.g., "-5.5 (-110)")
            
        Returns:
            Dict with 'line' and 'odds' or None if parsing fails
        """
        if not spread_text or pd.isna(spread_text):
            return None
        
        try:
            # Extract spread line and odds
            match = re.search(r'([+-]?\d+\.?\d*)\s*\(([+-]?\d+)\)', spread_text)
            if match:
                line = float(match.group(1))
                odds = int(match.group(2))
                return {'line': line, 'odds': odds}
            return None
        except Exception:
            return None
    
    def _parse_total(self, total_text: str) -> Optional[Dict]:
        """
        Parse total text into components
        
        Args:
            total_text: Text containing total line and odds (e.g., "220.5 (O: -110, U: -110)")
            
        Returns:
            Dict with 'line', 'over_odds', and 'under_odds' or None if parsing fails
        """
        if not total_text or pd.isna(total_text):
            return None
        
        try:
            # Extract total line and odds
            match = re.search(r'(\d+\.?\d*)\s*\(O:\s*([+-]?\d+),\s*U:\s*([+-]?\d+)\)', total_text)
            if match:
                line = float(match.group(1))
                over_odds = int(match.group(2))
                under_odds = int(match.group(3))
                return {'line': line, 'over_odds': over_odds, 'under_odds': under_odds}
            return None
        except Exception:
            return None
    
    def _moneyline_to_probability(self, moneyline: Optional[float]) -> Optional[float]:
        """
        Convert moneyline odds to implied probability
        
        Args:
            moneyline: Moneyline odds (e.g., -110, +150)
            
        Returns:
            Implied probability as a decimal (0-1) or None if conversion fails
        """
        if moneyline is None or pd.isna(moneyline):
            return None
        
        try:
            if moneyline > 0:
                # Positive moneyline (underdog)
                return 100 / (moneyline + 100)
            else:
                # Negative moneyline (favorite)
                return abs(moneyline) / (abs(moneyline) + 100)
        except Exception:
            return None
            
    def get_player_props(self, home_team: str, away_team: str) -> Dict:
        """
        Get player props for a specific matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with player props data
        """
        return self.player_props_service.get_player_props(home_team, away_team)
        
    def evaluate_player_props(self, prediction: Dict, home_team: str, away_team: str) -> Dict:
        """
        Evaluate player props for expected value based on model predictions
        
        Args:
            prediction: Prediction dictionary from predictor
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with player props evaluation
        """
        return self.player_props_service.compare_prediction_with_market(prediction, home_team, away_team)
        
    def get_best_props_bets(self, prediction: Dict, home_team: str, away_team: str) -> List[Dict]:
        """
        Get the best player props bets ranked by expected value
        
        Args:
            prediction: Prediction dictionary from predictor
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            List of best prop bets with EV and confidence metrics
        """
        return self.player_props_service.get_best_props(home_team, away_team, prediction)
                

class PredictionCalibrator:
    """
    Calibrates model predictions using betting market data
    """
    
    def __init__(self, betting_service: BettingDataService):
        self.betting_service = betting_service
        # For probabilistic modeling
        self.use_probabilities = True
        self.base_team_std = 11.0  # Base standard deviation for team scoring
    
    def calibrate_prediction(self, prediction: Dict, home_team: str, away_team: str) -> Dict:
        """
        Calibrate a prediction using betting market data
        
        Args:
            prediction: Prediction dictionary from the model
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Calibrated prediction with market comparisons
        """
        # Get betting lines
        betting_lines = self.betting_service.get_betting_lines(home_team, away_team)
        
        if not betting_lines:
            logger.warning(f"No betting lines found for {away_team} @ {home_team}, using uncalibrated prediction")
            # Add empty market data to prediction
            prediction['market_data'] = {
                'available': False
            }
            
            # Add standard deviations if not present
            if 'home_std' not in prediction:
                prediction['home_std'] = self._estimate_team_std(prediction.get('home_score', 0), home=True)
            if 'away_std' not in prediction:
                prediction['away_std'] = self._estimate_team_std(prediction.get('away_score', 0), home=False)
            
            return prediction
        
        # Extract prediction values
        predicted_home_score = prediction.get('home_score', 0)
        predicted_away_score = prediction.get('away_score', 0)
        predicted_total = predicted_home_score + predicted_away_score
        predicted_margin = predicted_home_score - predicted_away_score
        
        # Extract market values
        market_home_spread = betting_lines.get('home_spread', 0)
        market_total = betting_lines.get('total', 0)
        
        # Calculate market-implied scores
        market_home_score = (market_total + (-market_home_spread)) / 2
        market_away_score = (market_total - (-market_home_spread)) / 2
        
        # Calculate differences
        spread_diff = abs(predicted_margin - (-market_home_spread)) if market_home_spread else None
        total_diff = abs(predicted_total - market_total) if market_total else None
        
        # Calculate value ratings (0-10 scale)
        spread_value = self._calculate_value_rating(spread_diff, 10) if spread_diff is not None else None
        total_value = self._calculate_value_rating(total_diff, 15) if total_diff is not None else None
        
        # Adjust confidence based on market alignment
        market_confidence_factor = 1.0
        if spread_diff is not None and total_diff is not None:
            # Reduce confidence for predictions that differ greatly from the market
            market_confidence_factor = max(0.7, 1.0 - (spread_diff / 20) - (total_diff / 30))
            
            # Apply confidence adjustment
            if 'confidence' in prediction:
                prediction['confidence'] *= market_confidence_factor
                prediction['confidence'] = min(1.0, max(0.2, prediction['confidence']))
        
        # Determine if prediction should be regressed toward market
        regression_strength = 0.0
        
        # Apply stronger regression for extreme predictions
        if spread_diff is not None and spread_diff > 7:
            # Calculate regression strength based on difference (max 50%)
            regression_strength = min(0.5, (spread_diff - 7) / 20)
            
            # Apply regression to margin
            regressed_margin = predicted_margin * (1 - regression_strength) + (-market_home_spread) * regression_strength
            
            # Recalculate scores
            avg_score = predicted_total / 2
            prediction['home_score'] = avg_score + regressed_margin / 2
            prediction['away_score'] = avg_score - regressed_margin / 2
            
            # Round scores
            prediction['home_score'] = round(prediction['home_score'], 1)
            prediction['away_score'] = round(prediction['away_score'], 1)
            
            logger.info(f"Applied market regression: {predicted_margin:.1f} → {regressed_margin:.1f} (strength: {regression_strength:.2f})")
        
        # Add standard deviations for probabilistic modeling
        home_std = self._calculate_team_std_from_market(
            betting_lines.get('home_spread', 0),
            betting_lines.get('home_spread_odds', -110),
            betting_lines.get('total', 220),
            True
        )
        
        away_std = self._calculate_team_std_from_market(
            betting_lines.get('away_spread', 0),
            betting_lines.get('away_spread_odds', -110),
            betting_lines.get('total', 220),
            False
        )
        
        prediction['home_std'] = home_std
        prediction['away_std'] = away_std
        
        # Add market data to prediction
        prediction['market_data'] = {
            'available': True,
            'home_spread': market_home_spread,
            'away_spread': -market_home_spread if market_home_spread else None,
            'total': market_total,
            'home_moneyline': betting_lines.get('home_moneyline'),
            'away_moneyline': betting_lines.get('away_moneyline'),
            'home_implied_win_probability': betting_lines.get('home_implied_probability'),
            'away_implied_win_probability': betting_lines.get('away_implied_probability'),
            'spread_difference': spread_diff,
            'total_difference': total_diff,
            'spread_value_rating': spread_value,
            'total_value_rating': total_value,
            'market_confidence_factor': market_confidence_factor,
            'regression_applied': regression_strength > 0,
            'regression_strength': regression_strength,
            'market_home_score': market_home_score,
            'market_away_score': market_away_score
        }
        
        # Calculate probabilistic metrics
        if self.use_probabilities:
            prediction = self._add_probability_metrics(prediction, betting_lines)
        
        return prediction
    
    def _calculate_value_rating(self, difference: float, threshold: float) -> float:
        """
        Calculate value rating on a 0-10 scale based on difference from market
        
        Args:
            difference: Absolute difference between prediction and market
            threshold: Threshold for maximum difference (10 for spread, 15 for total)
            
        Returns:
            Value rating on a 0-10 scale
        """
        if difference is None:
            return 0
            
        # Lower difference = higher value
        value = max(0, 10 - (difference / threshold) * 10)
        return round(value, 1)
    
    def _estimate_team_std(self, points: float, home: bool = True) -> float:
        """Estimate team score standard deviation based on predicted points"""
        # Base standard deviation for NBA team scoring is around 10-12 points
        base_std = self.base_team_std
        
        # Home teams are slightly more consistent
        if home:
            base_std *= 0.95
        else:
            base_std *= 1.05
            
        # Higher scoring teams tend to have higher variance
        points_factor = points / 110.0  # normalize to average NBA team score
        
        # Calculate adjusted standard deviation
        return max(8.0, base_std * points_factor)
    
    def _calculate_team_std_from_market(self, spread: float, odds: int, total: float, is_home: bool) -> float:
        """
        Calculate team standard deviation from market odds
        
        Args:
            spread: Point spread
            odds: Spread odds
            total: Over/under total
            is_home: Whether this is the home team
            
        Returns:
            Estimated standard deviation for team scoring
        """
        # Base standard deviation
        base_std = self.base_team_std
        
        # Adjust based on spread size (larger spreads = more variance)
        spread_factor = 1.0 + (abs(spread) / 50.0)  # Small adjustment
        
        # Adjust based on total (higher totals = more variance)
        total_factor = total / 220.0  # Normalized to average NBA total
        
        # Adjust based on odds (more extreme odds = more certainty)
        odds_factor = 1.0
        if abs(odds) > 130:  # More extreme than -130/+130
            # Calculate how far from -110 (standard)
            deviation = (abs(odds) - 110) / 100.0
            odds_factor = max(0.85, 1.0 - deviation * 0.1)  # Reduce std by up to 15%
        
        # Home teams are slightly more consistent
        home_factor = 0.95 if is_home else 1.05
        
        # Calculate adjusted standard deviation
        return max(8.0, base_std * spread_factor * total_factor * odds_factor * home_factor)
    
    def _add_probability_metrics(self, prediction: Dict, betting_lines: Dict) -> Dict:
        """
        Add probability-based metrics to the prediction
        
        Args:
            prediction: Prediction dictionary
            betting_lines: Betting lines dictionary
            
        Returns:
            Updated prediction with probability metrics
        """
        # Extract needed values
        home_score = prediction['home_score']
        away_score = prediction['away_score']
        home_std = prediction['home_std']
        away_std = prediction['away_std']
        
        # Calculate win probability using normal distribution
        margin_std = np.sqrt(home_std**2 + away_std**2)  # Combined standard deviation
        z_score = (home_score - away_score) / margin_std
        win_probability = stats.norm.cdf(z_score)
        
        # Calculate spread probability
        spread = betting_lines.get('home_spread', 0)
        spread_z_score = (home_score - away_score - spread) / margin_std
        cover_probability = stats.norm.cdf(spread_z_score)
        
        # Calculate total probability
        total = betting_lines.get('total', 0)
        total_std = np.sqrt(home_std**2 + away_std**2 + 2 * 0.15 * home_std * away_std)  # Add small correlation
        total_z_score = (home_score + away_score - total) / total_std
        over_probability = stats.norm.cdf(total_z_score)
        
        # Calculate expected values
        home_ml_ev = self._calculate_ev(win_probability, betting_lines.get('home_moneyline', 0))
        away_ml_ev = self._calculate_ev(1 - win_probability, betting_lines.get('away_moneyline', 0))
        spread_home_ev = self._calculate_ev(cover_probability, betting_lines.get('home_spread_odds', -110))
        spread_away_ev = self._calculate_ev(1 - cover_probability, betting_lines.get('away_spread_odds', -110))
        total_over_ev = self._calculate_ev(over_probability, betting_lines.get('total_over_odds', -110))
        total_under_ev = self._calculate_ev(1 - over_probability, betting_lines.get('total_under_odds', -110))
        
        # Add to prediction
        prediction['probability_metrics'] = {
            'win_probability': win_probability,
            'spread_cover_probability': cover_probability, 
            'total_over_probability': over_probability,
            'expected_values': {
                'home_ml': home_ml_ev,
                'away_ml': away_ml_ev,
                'home_spread': spread_home_ev,
                'away_spread': spread_away_ev,
                'over': total_over_ev,
                'under': total_under_ev
            },
            'best_bet': self._determine_best_bet(
                [home_ml_ev, away_ml_ev, spread_home_ev, spread_away_ev, total_over_ev, total_under_ev]
            )
        }
        
        return prediction
    
    def _calculate_ev(self, probability: float, odds: int) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            probability: Probability of outcome
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
        
        return round(ev * 100, 2)  # Return as percentage
    
    def _determine_best_bet(self, ev_list: List[float]) -> Dict:
        """Determine the best bet from a list of EVs"""
        bet_types = ['HOME_ML', 'AWAY_ML', 'HOME_SPREAD', 'AWAY_SPREAD', 'OVER', 'UNDER']
        
        # Find maximum EV
        max_ev = max(ev_list)
        max_index = ev_list.index(max_ev)
        
        return {
            'type': bet_types[max_index],
            'ev': max_ev,
            'strength': 'HIGH' if max_ev > 8 else 'MEDIUM' if max_ev > 4 else 'LOW'
        }
    
    def calculate_kelly_criterion(self, probability: float, odds: int, fraction: float = 0.5) -> float:
        """
        Calculate Kelly Criterion for optimal bet sizing
        
        Args:
            probability: Probability of winning
            odds: American odds
            fraction: Fraction of full Kelly to use (0.5 = Half Kelly, safer)
            
        Returns:
            Recommended bet size as fraction of bankroll
        """
        # Convert odds to decimal
        if odds > 0:
            decimal_odds = odds/100 + 1
        else:
            decimal_odds = 100/abs(odds) + 1
        
        # Calculate Kelly stake
        edge = probability * decimal_odds - 1
        if edge <= 0:
            return 0
            
        kelly = edge / (decimal_odds - 1)
        
        # Apply fractional Kelly (safer)
        kelly = kelly * fraction
        
        # Cap at reasonable maximum
        return min(0.05, kelly)  # Cap at 5% of bankroll