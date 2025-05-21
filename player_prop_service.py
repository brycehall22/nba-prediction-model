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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('player_props_service')

class PlayerPropsService:
    """Service to integrate player point props with prediction system"""
    
    def __init__(self, props_file_path='draftkings_player_points_props.csv', update_interval_hours=6):
        self.props_file_path = props_file_path
        self.update_interval_hours = update_interval_hours
        self.last_update = None
        self.props_data = None
        self.team_mapping = self._create_team_mapping()
    
    def _create_team_mapping(self) -> Dict[str, str]:
        """Create mapping between different team name formats"""
        return {
            # Full name to abbreviated format (for matching with props data)
            "Atlanta Hawks": "ATL Hawks",
            "Boston Celtics": "BOS Celtics",
            "Brooklyn Nets": "BKN Nets",
            "Charlotte Hornets": "CHA Hornets",
            "Chicago Bulls": "CHI Bulls",
            "Cleveland Cavaliers": "CLE Cavaliers",
            "Dallas Mavericks": "DAL Mavericks",
            "Denver Nuggets": "DEN Nuggets",
            "Detroit Pistons": "DET Pistons",
            "Golden State Warriors": "GS Warriors",
            "Houston Rockets": "HOU Rockets",
            "Indiana Pacers": "IND Pacers",
            "Los Angeles Clippers": "LA Clippers",
            "Los Angeles Lakers": "LA Lakers",
            "Memphis Grizzlies": "MEM Grizzlies",
            "Miami Heat": "MIA Heat",
            "Milwaukee Bucks": "MIL Bucks",
            "Minnesota Timberwolves": "MIN Timberwolves",
            "New Orleans Pelicans": "NO Pelicans",
            "New York Knicks": "NY Knicks",
            "Oklahoma City Thunder": "OKC Thunder",
            "Orlando Magic": "ORL Magic",
            "Philadelphia 76ers": "PHI 76ers",
            "Phoenix Suns": "PHX Suns",
            "Portland Trail Blazers": "POR Trail Blazers",
            "Sacramento Kings": "SAC Kings",
            "San Antonio Spurs": "SA Spurs",
            "Toronto Raptors": "TOR Raptors",
            "Utah Jazz": "UTA Jazz",
            "Washington Wizards": "WAS Wizards"
        }
    
    def update_props_data(self, force=False) -> bool:
        """
        Update player props data if needed or if forced
        
        Args:
            force: Force update regardless of last update time
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        current_time = datetime.now()
        
        # Check if update is needed
        if not force and self.last_update and \
           (current_time - self.last_update).total_seconds() < self.update_interval_hours * 3600:
            logger.info("Player props data is still fresh, skipping update")
            return True
        
        try:
            logger.info("Updating player props data...")
            
            # Use subprocess to run the scraper script
            subprocess.run(["python", "nba_points_scraper.py"], check=True)
            
            # Check if file exists after running scraper
            if not os.path.exists(self.props_file_path):
                logger.error(f"Props file not found at {self.props_file_path}")
                return False
            
            # Load the data
            self.props_data = pd.read_csv(self.props_file_path)
            self.last_update = current_time
            
            # Clean up data
            self._clean_props_data()
            
            logger.info(f"Successfully updated player props data, found {len(self.props_data)} props")
            return True
            
        except Exception as e:
            logger.error(f"Error updating player props data: {str(e)}")
            return False
    
    def _clean_props_data(self):
        """Clean and standardize props data"""
        if self.props_data is None:
            return
        
        try:
            # Convert odds strings to numeric
            self.props_data['Over Odds'] = self.props_data['Over Odds'].apply(self._convert_odds_to_numeric)
            self.props_data['Under Odds'] = self.props_data['Under Odds'].apply(self._convert_odds_to_numeric)
            
            # Convert points line to float
            self.props_data['Points Line'] = pd.to_numeric(self.props_data['Points Line'], errors='coerce')
            
            # Fill any missing values
            self.props_data = self.props_data.fillna({
                'Over Odds': -110,
                'Under Odds': -110,
                'Points Line': 0.0
            })
            
            # Add additional calculated fields
            self.props_data['Over Probability'] = self.props_data['Over Odds'].apply(self._odds_to_probability)
            self.props_data['Under Probability'] = self.props_data['Under Odds'].apply(self._odds_to_probability)
            
            # Add team identifiers
            self.props_data['Teams'] = self.props_data['Game'].apply(lambda x: x.split(' at '))
            self.props_data['Away Team'] = self.props_data['Teams'].apply(lambda x: x[0] if len(x) > 0 else '')
            self.props_data['Home Team'] = self.props_data['Teams'].apply(lambda x: x[1] if len(x) > 1 else '')
            
            # Clean up
            if 'Teams' in self.props_data.columns:
                self.props_data = self.props_data.drop(columns=['Teams'])
            
        except Exception as e:
            logger.error(f"Error cleaning props data: {str(e)}")
    
    def _convert_odds_to_numeric(self, odds_str: str) -> int:
        """Convert odds string to numeric value"""
        if pd.isna(odds_str):
            return -110  # Default value
        
        try:
            # Handle special characters like '−' (not a standard minus sign)
            odds_str = odds_str.replace('−', '-').replace('+', '')
            return int(odds_str)
        except:
            return -110  # Default value if conversion fails
    
    def _odds_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds == 0:
            return 0.5
        
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def get_player_props(self, home_team: str, away_team: str) -> Dict:
        """
        Get player props for a specific matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with player props data
        """
        if self.props_data is None:
            self.update_props_data()
            
        if self.props_data is None or self.props_data.empty:
            logger.warning("No props data available")
            return {}
        
        try:
            # Convert team names to props format
            home_team_props = self.team_mapping.get(home_team, home_team)
            away_team_props = self.team_mapping.get(away_team, away_team)
            
            # Look for the game in different formats
            game_filters = [
                # Format: "Away at Home"
                (self.props_data['Away Team'] == away_team_props) & 
                (self.props_data['Home Team'] == home_team_props),
                
                # Look for partial matches with team abbreviations
                self.props_data['Game'].str.contains(f"{away_team_props.split()[0]} at {home_team_props.split()[0]}", 
                                                 case=False, regex=False),
                
                # Even more flexible matching
                (self.props_data['Away Team'].str.contains(away_team_props.split()[0], case=False)) &
                (self.props_data['Home Team'].str.contains(home_team_props.split()[0], case=False))
            ]
            
            # Try each filter until we find a match
            game_props = None
            for game_filter in game_filters:
                filtered_props = self.props_data[game_filter]
                if not filtered_props.empty:
                    game_props = filtered_props
                    break
            
            if game_props is None or game_props.empty:
                logger.warning(f"No props found for {away_team} at {home_team}")
                return {}
            
            # Format the props data for return
            results = {}
            for _, prop in game_props.iterrows():
                player_name = prop['Player']
                results[player_name] = {
                    'line': float(prop['Points Line']),
                    'over_odds': int(prop['Over Odds']),
                    'under_odds': int(prop['Under Odds']),
                    'over_probability': float(prop['Over Probability']),
                    'under_probability': float(prop['Under Probability']),
                    'player': player_name,
                    'team': self._determine_team(player_name, prop['Home Team'], prop['Away Team'], home_team, away_team)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting player props: {str(e)}")
            return {}
    
    def _determine_team(self, player_name, prop_home, prop_away, home_team, away_team):
        """Determine which team a player is on based on available data"""
        # This is a simplistic approach - a more robust solution would use a player-team database
        # For now, we return the teams from the props for display purposes
        return {'home': prop_home, 'away': prop_away}
    
    def compare_prediction_with_market(self, prediction, home_team: str, away_team: str) -> Dict:
        """
        Compare player points predictions with market lines
        
        Args:
            prediction: Prediction dictionary from predictor
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with comparison metrics
        """
        props = self.get_player_props(home_team, away_team)
        if not props:
            return {'available': False}
        
        # Extract player predictions from prediction object
        home_players = prediction.get('home_player_predictions', [])
        away_players = prediction.get('away_player_predictions', [])
        
        # Combine all players
        all_player_predictions = {}
        for player in home_players:
            all_player_predictions[player['name']] = {
                'predicted_points': player['points'],
                'team': 'home',
                'status': player['status'],
                'minutes': player.get('minutes', 0),
                # Add standard deviation if available, otherwise estimate it
                'points_std': player.get('points_std', self._estimate_points_std(player['points'], player.get('minutes', 0)))
            }
        
        for player in away_players:
            all_player_predictions[player['name']] = {
                'predicted_points': player['points'],
                'team': 'away',
                'status': player['status'],
                'minutes': player.get('minutes', 0),
                # Add standard deviation if available, otherwise estimate it
                'points_std': player.get('points_std', self._estimate_points_std(player['points'], player.get('minutes', 0)))
            }
        
        # Compare with props data
        comparison = {
            'available': True,
            'players': {},
            'summary': {
                'total_props': len(props),
                'matched_players': 0,
                'value_opportunities': 0
            }
        }
        
        for player_name, prop_data in props.items():
            # Try to find matching player in predictions
            matched_player = self._find_matching_player(player_name, all_player_predictions)
            
            if matched_player:
                pred = all_player_predictions[matched_player]
                predicted_points = pred['predicted_points']
                points_std = pred['points_std']
                prop_line = prop_data['line']
                
                # Calculate the difference and edge percentage
                difference = predicted_points - prop_line
                edge_pct = (abs(difference) / prop_line) * 100 if prop_line > 0 else 0
                
                # Calculate probabilities using normal distribution
                z_score = (prop_line - predicted_points) / points_std if points_std > 0 else 0
                over_probability = 1 - stats.norm.cdf(z_score)
                under_probability = stats.norm.cdf(z_score)
                
                # Calculate expected values
                over_ev = self._calculate_expected_value(over_probability, prop_data['over_odds'])
                under_ev = self._calculate_expected_value(under_probability, prop_data['under_odds'])
                
                # Determine if this is a value opportunity
                is_over_value = over_ev > 5.0  # 5% EV threshold
                is_under_value = under_ev > 5.0
                is_value = is_over_value or is_under_value
                
                # Determine the best bet
                if over_ev > under_ev and over_ev > 0:
                    best_bet = 'OVER'
                    best_ev = over_ev
                elif under_ev > 0:
                    best_bet = 'UNDER'
                    best_ev = under_ev
                else:
                    best_bet = 'NONE'
                    best_ev = 0
                
                # Calculate confidence based on z-score and sample size
                confidence = self._calculate_confidence(z_score, pred.get('minutes', 25))
                
                comparison['players'][player_name] = {
                    'prediction': {
                        'points': predicted_points,
                        'std_dev': points_std,
                        'minutes': pred.get('minutes', 0),
                        'status': pred.get('status', 'Active')
                    },
                    'market': {
                        'line': prop_line,
                        'over_odds': prop_data['over_odds'],
                        'under_odds': prop_data['under_odds'],
                        'implied_over_prob': prop_data['over_probability'],
                        'implied_under_prob': prop_data['under_probability']
                    },
                    'analysis': {
                        'difference': difference,
                        'edge_percentage': edge_pct,
                        'over_probability': over_probability,
                        'under_probability': under_probability,
                        'over_ev': over_ev,
                        'under_ev': under_ev,
                        'best_bet': best_bet,
                        'best_ev': best_ev,
                        'confidence': confidence,
                        'is_value': is_value,
                        'z_score': z_score
                    }
                }
                
                comparison['summary']['matched_players'] += 1
                if is_value:
                    comparison['summary']['value_opportunities'] += 1
        
        # Calculate overall value rating for the game
        if comparison['summary']['matched_players'] > 0:
            comparison['summary']['match_rate'] = comparison['summary']['matched_players'] / len(props)
            comparison['summary']['value_rate'] = comparison['summary']['value_opportunities'] / comparison['summary']['matched_players'] if comparison['summary']['matched_players'] > 0 else 0
        else:
            comparison['summary']['match_rate'] = 0
            comparison['summary']['value_rate'] = 0
        
        # Sort value opportunities by expected value
        value_props = [
            {
                'player': player,
                'line': data['market']['line'],
                'predicted_points': data['prediction']['points'],
                'best_bet': data['analysis']['best_bet'],
                'best_ev': data['analysis']['best_ev'],
                'confidence': data['analysis']['confidence']
            }
            for player, data in comparison['players'].items()
            if data['analysis']['is_value']
        ]
        
        comparison['value_props'] = sorted(value_props, key=lambda x: x['best_ev'], reverse=True)
        
        return comparison
    
    def _find_matching_player(self, prop_player_name, predictions_dict):
        """Find a matching player in the predictions dictionary using fuzzy matching"""
        # Try direct match
        if prop_player_name in predictions_dict:
            return prop_player_name
        
        # Try case-insensitive match
        for pred_name in predictions_dict:
            if pred_name.lower() == prop_player_name.lower():
                return pred_name
        
        # Try last name matching
        prop_last_name = prop_player_name.split()[-1].lower()
        
        for pred_name in predictions_dict:
            pred_last_name = pred_name.split()[-1].lower()
            if prop_last_name == pred_last_name:
                return pred_name
                
        # No match found
        return None
    
    def _estimate_points_std(self, points, minutes):
        """Estimate standard deviation for player points prediction if not provided"""
        # NBA players have variance proportional to their scoring and minutes
        # A decent heuristic is around 30-40% of their points total
        relative_std = 0.35  # baseline
        
        # Adjust based on minutes (fewer minutes = more variance relatively)
        if minutes < 15:
            relative_std = 0.45  # 45% relative std for low-minute players
        elif minutes < 25:
            relative_std = 0.40  # 40% relative std for rotation players
        elif minutes > 32:
            relative_std = 0.30  # 30% relative std for high-minute stars
        
        # Calculate std with a minimum value
        return max(3.0, points * relative_std)
    
    def _calculate_expected_value(self, probability, odds):
        """Calculate expected value of a bet given probability and odds"""
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
    
    def _calculate_confidence(self, z_score, minutes):
        """Calculate confidence in the prediction based on z-score and minutes"""
        # How far the line is from our predicted mean affects confidence
        z_factor = min(1.0, max(0.1, 1.0 - abs(z_score) * 0.2))
        
        # Minutes played affects confidence (more minutes = more predictable)
        minutes_factor = min(1.0, max(0.6, minutes / 36))
        
        # Calculate overall confidence (0-1 scale)
        confidence = z_factor * 0.7 + minutes_factor * 0.3
        
        return round(confidence, 2)
    
    def calculate_kelly_criterion(self, probability, odds, fraction=0.5):
        """
        Calculate Kelly Criterion bet size
        
        Args:
            probability: Probability of winning the bet
            odds: American odds
            fraction: Fraction of full Kelly to use (default 0.5 for Half Kelly)
            
        Returns:
            Recommended bet size as a fraction of bankroll
        """
        # Convert odds to decimal format
        if odds > 0:
            decimal_odds = odds/100 + 1
        else:
            decimal_odds = 100/abs(odds) + 1
        
        # Calculate Kelly stake
        edge = (probability * decimal_odds) - 1
        
        # No edge = no bet
        if edge <= 0:
            return 0
            
        # Calculate full Kelly stake
        kelly = edge / (decimal_odds - 1)
        
        # Apply fractional Kelly (safer)
        fractional_kelly = kelly * fraction
        
        # Cap at reasonable maximum
        return min(0.05, fractional_kelly)  # Cap at 5% of bankroll
    
    def get_best_props(self, home_team: str, away_team: str, prediction) -> List[Dict]:
        """
        Get the best player props bets for a game
        
        Args:
            home_team: Home team name
            away_team: Away team name
            prediction: Game prediction dictionary
            
        Returns:
            List of best prop bets with EV and confidence
        """
        # Get comprehensive comparison
        comparison = self.compare_prediction_with_market(prediction, home_team, away_team)
        
        if not comparison.get('available', False):
            return []
            
        # Extract value props
        value_props = comparison.get('value_props', [])
        
        # Add Kelly criterion recommendations
        for prop in value_props:
            if prop['best_bet'] == 'OVER':
                player_data = comparison['players'][prop['player']]
                probability = player_data['analysis']['over_probability']
                odds = player_data['market']['over_odds']
            else:
                player_data = comparison['players'][prop['player']]
                probability = player_data['analysis']['under_probability']
                odds = player_data['market']['under_odds']
                
            # Calculate Kelly stake
            kelly = self.calculate_kelly_criterion(probability, odds)
            prop['kelly_stake'] = kelly
            prop['recommended_units'] = round(kelly * 100, 1)
            
            # Add a 1-10 value rating
            prop['value_rating'] = min(10, max(1, round(prop['best_ev'] / 2 + 4)))
        
        # Sort by EV
        best_props = sorted(value_props, key=lambda x: x['best_ev'], reverse=True)
        
        return best_props