import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from nba_api.stats.static import teams

# Import custom modules (adjust imports to match your file structure)
from probability_model import ProbabilisticModel
from odds_calibrator import OddsCalibrator
from player_props_predictor import PlayerPropsPredictor
from test import get_team_name

logger = logging.getLogger(__name__)

class EVPredictionSystem:
    """
    Integrated prediction system focused on expected value and probability-based betting insights.
    This system wraps the existing NBA prediction components and adds EV calculations.
    """
    
    def __init__(self, predictor, betting_service, collector):
        """
        Initialize the EV prediction system
        
        Args:
            predictor: Existing NBAPredictor instance
            betting_service: BettingDataService instance
            collector: DataCollector instance
        """
        self.predictor = predictor
        self.betting_service = betting_service
        self.collector = collector
        
        # Initialize the enhanced components
        self.prob_model = ProbabilisticModel()
        self.odds_calibrator = OddsCalibrator(regression_strength=0.5)  # Adjust strength as needed
        self.props_predictor = PlayerPropsPredictor(collector, predictor.trainer if hasattr(predictor, 'trainer') else None)
        
        # Settings
        self.ev_threshold = 3.0  # Minimum EV% to consider a bet valuable
        self.confidence_threshold = 0.6  # Minimum confidence for recommendations

    def _get_team_name(team_id):
        nba_teams = teams.get_teams()
        team = [team for team in nba_teams if team['id'] == team_id][0]
        return team['full_name']
    
    def predict_game_with_ev(self, home_team_id, away_team_id) -> Dict:
        """
        Generate game prediction with expected value metrics
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            
        Returns:
            Dictionary with prediction results and EV metrics
        """
        try:
            # Step 1: Get base prediction from existing system
            base_prediction = self.predictor.predict_game(home_team_id, away_team_id)
            
            if 'error' in base_prediction and base_prediction['error']:
                return base_prediction
            
            # Step 2: Get team names for betting data lookup
            home_team_name = get_team_name(home_team_id)
            away_team_name = get_team_name(away_team_id)
            
            # Step 3: Get betting lines
            betting_lines = self.betting_service.get_betting_lines(home_team_name, away_team_name)
            if betting_lines:
                market_data = betting_lines
            else:
                market_data = {'available': False}
            
            # Step 4: Calibrate prediction with market data
            calibrated = self.odds_calibrator.calibrate_prediction(base_prediction, market_data)
            
            # Step 5: Generate probabilistic metrics
            ev_metrics = self.odds_calibrator.calculate_ev_metrics(
                calibrated, market_data, self.prob_model
            )
            
            # Step 6: Evaluate player props for this game
            player_ev = self.evaluate_player_props(
                calibrated['home_player_predictions'],
                calibrated['away_player_predictions'],
                home_team_name,
                away_team_name
            )
            
            # Step 7: Combine everything into a comprehensive result
            result = {
                'prediction': calibrated,
                'ev_metrics': ev_metrics,
                'player_props_ev': player_ev,
                'betting_recommendations': self._generate_betting_recommendations(
                    ev_metrics, player_ev
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in EV prediction: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_player_props(self, home_players, away_players, 
                           home_team_name, away_team_name) -> Dict:
        """
        Evaluate player props for expected value
        
        Args:
            home_players: List of home player predictions
            away_players: List of away player predictions
            home_team_name: Home team name
            away_team_name: Away team name
            
        Returns:
            Dictionary with player props EV analysis
        """
        # Initialize result
        player_ev = {
            'available': False,
            'props_analyzed': 0,
            'props_with_edge': 0,
            'best_props': []
        }
        
        try:
            # Get props from PlayerPropsService
            # Assuming you have a method to get player props in your betting_service
            player_props = self.betting_service.get_player_props(home_team_name, away_team_name)
            
            if not player_props:
                return player_ev
            
            player_ev['available'] = True
            player_ev['props_analyzed'] = len(player_props)
            
            # Process each player prop
            evaluations = []
            
            for player_name, prop_data in player_props.items():
                # Try to find matching player in predictions
                player_info = self._find_player_in_predictions(
                    player_name, home_players, away_players
                )
                
                if not player_info:
                    continue
                
                # Extract prediction
                base_prediction = {
                    'name': player_info['name'],
                    'team': player_info['team'],
                    'points': player_info['points'],
                    'minutes': player_info['minutes']
                }
                
                # Get enhanced prediction with uncertainty
                enhanced_prediction = self._get_enhanced_player_prediction(
                    player_info, player_name, prop_data
                )
                
                # Evaluate the prop bet
                evaluation = self.props_predictor.evaluate_prop_bet(
                    enhanced_prediction, prop_data
                )
                
                if evaluation['valid'] and evaluation['best_bet'] != 'NONE' and evaluation['best_ev'] > 0:
                    evaluation['player'] = player_name
                    evaluation['team'] = player_info['team']
                    evaluations.append(evaluation)
            
            # Sort by EV and filter for positive value
            positive_ev_props = [e for e in evaluations if e['best_ev'] > 0]
            player_ev['props_with_edge'] = len(positive_ev_props)
            
            # Sort by EV and take top 5
            best_props = sorted(positive_ev_props, key=lambda x: x['best_ev'], reverse=True)[:5]
            player_ev['best_props'] = best_props
            
            return player_ev
            
        except Exception as e:
            logger.error(f"Error evaluating player props: {str(e)}")
            return player_ev
    
    def _find_player_in_predictions(self, prop_player_name, home_players, away_players):
        """Find a player in predictions by name with fuzzy matching"""
        
        # Try direct match
        for player_list in [home_players, away_players]:
            for player in player_list:
                if player['name'].lower() == prop_player_name.lower():
                    player['team'] = 'home' if player_list == home_players else 'away'
                    return player
        
        # Try matching last name
        prop_last_name = prop_player_name.split()[-1].lower()
        
        for player_list in [home_players, away_players]:
            for player in player_list:
                player_last_name = player['name'].split()[-1].lower()
                
                if prop_last_name == player_last_name:
                    player['team'] = 'home' if player_list == home_players else 'away'
                    return player
        
        # No match found
        return None
    
    def _get_enhanced_player_prediction(self, player_info, prop_player_name, prop_data):
        """Get enhanced player prediction with standard deviation"""
        
        # Start with base prediction
        enhanced = {
            'name': player_info['name'],
            'team': player_info['team'],
            'points': player_info['points'],
            'minutes': player_info['minutes'],
            'status': player_info.get('status', 'Active')
        }
        
        # Add standard deviation if not present
        if 'points_std' not in player_info:
            # Estimate standard deviation based on points
            # More points = higher std, but with diminishing returns
            points = player_info['points']
            base_std = min(10, max(3, points * 0.3))
            
            # Low minutes usually means higher variability relative to output
            if player_info['minutes'] < 15:
                rel_std_factor = 0.45  # 45% relative std
            elif player_info['minutes'] < 25:
                rel_std_factor = 0.35  # 35% relative std
            else:
                rel_std_factor = 0.27  # 27% relative std
                
            enhanced['points_std'] = max(base_std, points * rel_std_factor)
        else:
            enhanced['points_std'] = player_info['points_std']
        
        return enhanced
    
    def _generate_betting_recommendations(self, ev_metrics, player_ev):
        """Generate betting recommendations based on EV metrics"""
        recommendations = {
            'game_bets': [],
            'player_props': []
        }
        
        # Game bets
        if ev_metrics.get('available', False):
            # Check spread
            spread = ev_metrics['spread']
            if spread['best_ev'] > self.ev_threshold:
                recommendations['game_bets'].append({
                    'type': 'SPREAD',
                    'bet': spread['best_bet'],
                    'line': spread['line'],
                    'ev': spread['best_ev'],
                    'probability': spread['home_cover_prob'] if spread['best_bet'] == 'HOME' else spread['away_cover_prob'],
                    'confidence': 'HIGH' if spread['best_ev'] > 8 else 'MEDIUM' if spread['best_ev'] > 5 else 'LOW'
                })
            
            # Check total
            total = ev_metrics['total']
            if total['best_ev'] > self.ev_threshold:
                recommendations['game_bets'].append({
                    'type': 'TOTAL',
                    'bet': total['best_bet'],
                    'line': total['line'],
                    'ev': total['best_ev'],
                    'probability': total['over_prob'] if total['best_bet'] == 'OVER' else total['under_prob'],
                    'confidence': 'HIGH' if total['best_ev'] > 8 else 'MEDIUM' if total['best_ev'] > 5 else 'LOW'
                })
            
            # Check moneyline
            ml = ev_metrics['moneyline']
            if ml['best_ev'] > self.ev_threshold:
                recommendations['game_bets'].append({
                    'type': 'MONEYLINE',
                    'bet': ml['best_bet'],
                    'odds': ml['home_odds'] if ml['best_bet'] == 'HOME' else ml['away_odds'],
                    'ev': ml['best_ev'],
                    'probability': ev_metrics['win_probability']['home'] if ml['best_bet'] == 'HOME' else ev_metrics['win_probability']['away'],
                    'confidence': 'HIGH' if ml['best_ev'] > 8 else 'MEDIUM' if ml['best_ev'] > 5 else 'LOW'
                })
        
        # Player props
        if player_ev.get('available', False) and player_ev.get('best_props'):
            for prop in player_ev['best_props']:
                if prop['best_ev'] > self.ev_threshold and prop['confidence'] > self.confidence_threshold:
                    recommendations['player_props'].append({
                        'player': prop['player'],
                        'team': prop['team'],
                        'type': 'POINTS',
                        'bet': prop['best_bet'],
                        'line': prop['line'],
                        'ev': prop['best_ev'],
                        'probability': prop['over']['probability'] if prop['best_bet'] == 'OVER' else prop['under']['probability'],
                        'confidence': prop['confidence'],
                        'value_rating': prop['value_rating']
                    })
        
        # Sort by EV
        recommendations['game_bets'] = sorted(recommendations['game_bets'], key=lambda x: x['ev'], reverse=True)
        recommendations['player_props'] = sorted(recommendations['player_props'], key=lambda x: x['ev'], reverse=True)
        
        # Add best overall bet
        all_bets = recommendations['game_bets'] + recommendations['player_props']
        if all_bets:
            recommendations['best_overall'] = max(all_bets, key=lambda x: x['ev'])
        
        return recommendations