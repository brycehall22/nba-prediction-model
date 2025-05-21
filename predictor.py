from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs, CommonPlayerInfo, TeamPlayerDashboard
from nba_api.stats.endpoints import LeagueGameLog, BoxScoreTraditionalV2
from nba_api.stats.static import teams, players
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime, timedelta
import time
import os
import re
import requests
from bs4 import BeautifulSoup
import random
from data_collector import DataCollector
from model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NBAPredictor:
    def __init__(self):
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.models_loaded = False
        self.fallback_mode = False
        
        # Define league averages for baseline predictions
        self.league_avg_points = 110.0  # League average team points
        self.league_avg_player_mins = 24.0  # Average minutes for a starter
        self.league_avg_player_pts = 12.0  # Average points for a rotation player
        
        # Try to load models, train if not available
        try:
            if not self.trainer.load_models():
                logger.info("Training new models...")
                success = self.trainer.train_models()
                if success:
                    self.trainer.save_models()
                    self.models_loaded = True
                else:
                    logger.error("Failed to train models")
                    self.fallback_mode = True
            else:
                self.models_loaded = True
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            self.fallback_mode = True

    def predict_game(self, home_team_id, away_team_id):
        """
        Predict the outcome of an NBA game with comprehensive team and player predictions
        """
        try:
            # Update injury data before making prediction
            self.collector.update_injury_data()
            
            # Get detailed team stats with opponent analysis
            home_stats = self.collector.get_team_stats(home_team_id, away_team_id)
            away_stats = self.collector.get_team_stats(away_team_id, home_team_id)
            
            if home_stats is None or away_stats is None:
                return {'error': 'Could not get team stats'}

            # Set home/away indicators
            home_stats['HOME_GAME'] = 1
            away_stats['HOME_GAME'] = 0
            
            # Get injury information
            home_injuries = self.collector.get_team_injuries(home_team_id)
            away_injuries = self.collector.get_team_injuries(away_team_id)
            
            # Apply injury adjustments to team stats
            if home_injuries:
                home_injury_impact = home_injuries['impact_score']
                # Reduce scoring based on injury impact (up to 12% reduction)
                home_stats['INJURY_ADJUSTED_PTS'] = home_stats['PTS_AVG'] * (1 - min(0.12, home_injury_impact * 0.04))
            else:
                home_stats['INJURY_ADJUSTED_PTS'] = home_stats['PTS_AVG']
                
            if away_injuries:
                away_injury_impact = away_injuries['impact_score']
                # Reduce scoring based on injury impact (up to 12% reduction)
                away_stats['INJURY_ADJUSTED_PTS'] = away_stats['PTS_AVG'] * (1 - min(0.12, away_injury_impact * 0.04))
            else:
                away_stats['INJURY_ADJUSTED_PTS'] = away_stats['PTS_AVG']

            # Use fallback approach if models aren't loaded
            if self.fallback_mode or not self.models_loaded:
                logger.warning("Using fallback prediction approach")
                # Statistical prediction based on recent averages with injury adjustment
                home_team_pred = home_stats['INJURY_ADJUSTED_PTS'] + 3.0  # Add home court advantage
                away_team_pred = away_stats['INJURY_ADJUSTED_PTS']
                prediction_method = "statistical"
            else:
                # Model-based team prediction
                try:
                    # Prepare feature dataframes
                    home_features = pd.DataFrame([home_stats])
                    away_features = pd.DataFrame([away_stats])
                    
                    # Handle missing features that might be in the model but not in current data
                    for col in self.trainer.team_model.feature_names_in_:
                        if col not in home_features.columns:
                            home_features[col] = 0
                        if col not in away_features.columns:
                            away_features[col] = 0
                    
                    # Keep only features used in the model
                    home_features = home_features[self.trainer.team_model.feature_names_in_]
                    away_features = away_features[self.trainer.team_model.feature_names_in_]
                    
                    # Scale features
                    home_scaled = self.trainer.team_scaler.transform(home_features)
                    away_scaled = self.trainer.team_scaler.transform(away_features)
                    
                    # Generate predictions
                    home_team_pred = self.trainer.team_model.predict(home_scaled)[0]
                    away_team_pred = self.trainer.team_model.predict(away_scaled)[0]
                    
                    # Apply injury adjustments post-prediction
                    if home_injuries:
                        home_team_pred *= (1 - min(0.12, home_injuries['impact_score'] * 0.04))
                    if away_injuries:
                        away_team_pred *= (1 - min(0.12, away_injuries['impact_score'] * 0.04))
                    
                    prediction_method = "model"
                except Exception as e:
                    logger.error(f"Error in team prediction: {str(e)}")
                    # Fallback to recent average if model fails
                    home_team_pred = home_stats['INJURY_ADJUSTED_PTS'] + 3.0  # Add home court advantage
                    away_team_pred = away_stats['INJURY_ADJUSTED_PTS']
                    prediction_method = "fallback"

            # Get more detailed player predictions
            home_players = self.collector.get_team_players(home_team_id)
            away_players = self.collector.get_team_players(away_team_id)
            
            # Track which players are predicted
            home_player_preds = []
            away_player_preds = []
            
            # Set reasonable timeouts for API calls
            player_timeout = 1.5  # 1.5 seconds per player
            total_timeout = 15.0  # 15 seconds total for all players
            
            # Get predictions for home team players with improved logic
            start_time = time.time()
            for player in home_players[:12]:  # Get top 12 players by minutes
                if time.time() - start_time > total_timeout:
                    break
                    
                player_id = player['id']
                player_name = player['full_name']
                avg_minutes = player.get('avg_minutes', 0)
                
                # Skip players with very limited minutes
                if avg_minutes < 5:
                    continue
                
                # Check if player is injured
                is_injured = False
                if home_injuries and 'injured_players' in home_injuries:
                    is_injured = any(p['name'] == player_name for p in home_injuries['injured_players'])
                
                if is_injured:
                    # Add injured player with 0 points
                    home_player_preds.append({
                        'name': player_name,
                        'points': 0.0,
                        'minutes': 0.0,
                        'status': 'Injured'
                    })
                    continue
                
                # Try to get player prediction with timeout
                player_start_time = time.time()
                try:
                    while time.time() - player_start_time < player_timeout:
                        pred = self.predict_player_performance(
                            player_id, 
                            is_home=True,
                            opp_team_id=away_team_id,
                            minutes=avg_minutes
                        )
                        if pred is not None:
                            home_player_preds.append({
                                'name': player_name,
                                'points': pred['points'],
                                'minutes': pred['minutes'],
                                'status': 'Active'
                            })
                            break
                        time.sleep(0.2)
                except Exception as e:
                    logger.error(f"Error predicting for {player_name}: {str(e)}")
            
            # Get predictions for away team players with improved logic
            start_time = time.time()
            for player in away_players[:12]:  # Get top 12 players by minutes
                if time.time() - start_time > total_timeout:
                    break
                    
                player_id = player['id']
                player_name = player['full_name']
                avg_minutes = player.get('avg_minutes', 0)
                
                # Skip players with very limited minutes
                if avg_minutes < 5:
                    continue
                
                # Check if player is injured
                is_injured = False
                if away_injuries and 'injured_players' in away_injuries:
                    is_injured = any(p['name'] == player_name for p in away_injuries['injured_players'])
                
                if is_injured:
                    # Add injured player with 0 points
                    away_player_preds.append({
                        'name': player_name,
                        'points': 0.0,
                        'minutes': 0.0,
                        'status': 'Injured'
                    })
                    continue
                
                # Try to get player prediction with timeout
                player_start_time = time.time()
                try:
                    while time.time() - player_start_time < player_timeout:
                        pred = self.predict_player_performance(
                            player_id, 
                            is_home=False,
                            opp_team_id=home_team_id,
                            minutes=avg_minutes
                        )
                        if pred is not None:
                            away_player_preds.append({
                                'name': player_name,
                                'points': pred['points'],
                                'minutes': pred['minutes'],
                                'status': 'Active'
                            })
                            break
                        time.sleep(0.2)
                except Exception as e:
                    logger.error(f"Error predicting for {player_name}: {str(e)}")
            
            # If we have few player predictions, try backup approach
            if len(home_player_preds) < 5:
                logger.warning(f"Few home player predictions ({len(home_player_preds)}), using backup approach")
                home_player_preds = self._generate_backup_player_predictions(
                    home_players, home_team_pred, is_home=True, existing_preds=home_player_preds)
                    
            if len(away_player_preds) < 5:
                logger.warning(f"Few away player predictions ({len(away_player_preds)}), using backup approach")
                away_player_preds = self._generate_backup_player_predictions(
                    away_players, away_team_pred, is_home=False, existing_preds=away_player_preds)
            
            # Calculate team totals from player predictions
            home_player_sum = sum(p['points'] for p in home_player_preds)
            away_player_sum = sum(p['points'] for p in away_player_preds)
            
            # Apply score normalization if player predictions are way off
            if home_player_preds and abs(home_player_sum - home_team_pred) > 20:
                # Scale factor for normalization
                scale_factor = home_team_pred / max(home_player_sum, 1)
                # Apply scaling to each player
                for i in range(len(home_player_preds)):
                    home_player_preds[i]['points'] = round(home_player_preds[i]['points'] * scale_factor, 1)
                home_player_sum = sum(p['points'] for p in home_player_preds)
            
            if away_player_preds and abs(away_player_sum - away_team_pred) > 20:
                # Scale factor for normalization
                scale_factor = away_team_pred / max(away_player_sum, 1)
                # Apply scaling to each player
                for i in range(len(away_player_preds)):
                    away_player_preds[i]['points'] = round(away_player_preds[i]['points'] * scale_factor, 1)
                away_player_sum = sum(p['points'] for p in away_player_preds)

            # Calculate coverage and confidence metrics
            home_coverage = min(1.0, len(home_player_preds) / 8)
            away_coverage = min(1.0, len(away_player_preds) / 8)
            
            # Blend team and player predictions based on coverage
            if home_coverage > 0.7 and away_coverage > 0.7:
                # Good player coverage, give more weight to player-based predictions
                home_score = 0.7 * home_team_pred + 0.3 * home_player_sum
                away_score = 0.7 * away_team_pred + 0.3 * away_player_sum
            else:
                # Limited player data, rely more on team-based predictions
                home_player_weight = max(0.1, home_coverage * 0.3)
                away_player_weight = max(0.1, away_coverage * 0.3)
                home_score = (1 - home_player_weight) * home_team_pred + home_player_weight * home_player_sum
                away_score = (1 - away_player_weight) * away_team_pred + away_player_weight * away_player_sum

            # Apply home court advantage adjustment if not already in model
            if prediction_method != "model":
                home_court_advantage = 3.0
                home_score += home_court_advantage
            
            # Pace adjustment based on teams' playing styles
            if 'PACE' in home_stats and 'PACE' in away_stats:
                # Average the two teams' paces
                avg_pace = (home_stats['PACE'] + away_stats['PACE']) / 2
                # Compare to league average (approx 100)
                pace_factor = avg_pace / 100.0
                # Adjust scores proportionally (faster pace = more points)
                pace_adjustment = (pace_factor - 1.0) * 5  # Up to +/- 5 points
                home_score += pace_adjustment
                away_score += pace_adjustment

            # Calculate prediction confidence
            confidence_factors = {
                'home_consistency': 1.0 - min(0.5, home_stats.get('PTS_STD', 10) / 20),
                'away_consistency': 1.0 - min(0.5, away_stats.get('PTS_STD', 10) / 20),
                'player_coverage': (home_coverage + away_coverage) / 2,
                'model_reliability': 0.8 if prediction_method == "model" else 0.5,
                'injury_impact': 1.0 - 0.1 * (
                    (home_injuries.get('impact_score', 0) if home_injuries else 0) +
                    (away_injuries.get('impact_score', 0) if away_injuries else 0)
                ) / 2
            }
            
            # Point margin affects confidence (closer games are harder to predict)
            margin = abs(home_score - away_score)
            margin_factor = min(0.95, max(0.5, 0.7 + margin / 30))  # 0.5-0.95 based on margin
            
            # Calculate weighted confidence
            confidence = (
                0.25 * confidence_factors['home_consistency'] +
                0.25 * confidence_factors['away_consistency'] +
                0.20 * confidence_factors['player_coverage'] +
                0.15 * confidence_factors['model_reliability'] +
                0.15 * confidence_factors['injury_impact']
            ) * margin_factor

            # Sort player predictions by points
            home_player_preds_sorted = sorted(home_player_preds, key=lambda x: x['points'], reverse=True)
            away_player_preds_sorted = sorted(away_player_preds, key=lambda x: x['points'], reverse=True)

            # Prepare injury information
            injury_info = {}
            if home_injuries:
                injury_info['home'] = {
                    'players_out': [p['name'] for p in home_injuries.get('injured_players', []) 
                                   if p.get('status', '').lower() in ('out', 'doubtful')],
                    'impact': home_injuries.get('impact_score', 0)
                }
            if away_injuries:
                injury_info['away'] = {
                    'players_out': [p['name'] for p in away_injuries.get('injured_players', []) 
                                   if p.get('status', '').lower() in ('out', 'doubtful')],
                    'impact': away_injuries.get('impact_score', 0)
                }

            return {
                'home_score': round(home_score, 1),
                'away_score': round(away_score, 1),
                'home_player_predictions': home_player_preds_sorted,
                'away_player_predictions': away_player_preds_sorted,
                'confidence': round(confidence, 3),
                'prediction_method': prediction_method,
                'injuries': injury_info,
                'error': None
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}
            
    def _generate_backup_player_predictions(self, players_list, team_total, is_home, existing_preds=None):
        """
        Generate backup player predictions when API data is limited
        """
        if existing_preds is None:
            existing_preds = []
        
        # Create a new list to avoid modifying the original
        backup_preds = [p for p in existing_preds]
        
        # Get players who don't already have predictions
        existing_names = {p['name'] for p in backup_preds}
        remaining_players = [p for p in players_list if p['full_name'] not in existing_names]
        
        # Skip if no players to predict
        if not remaining_players:
            return backup_preds
            
        # Calculate how many points we need to distribute
        existing_points = sum(p['points'] for p in backup_preds)
        points_to_distribute = max(0, team_total - existing_points)
        
        # Get total minutes to distribute
        total_minutes = sum(p.get('avg_minutes', 10) for p in remaining_players)
        
        # If no minutes data, use default estimates
        if total_minutes == 0:
            # Estimate about 240 total minutes per team (48 min x 5 players)
            used_minutes = sum(p.get('minutes', 0) for p in backup_preds)
            remaining_minutes = max(0, 240 - used_minutes)
            
            # Distribute minutes somewhat realistically for remaining players
            if len(remaining_players) > 0:
                # Minutes distribution that roughly matches NBA rotation patterns
                minute_weights = [1.0] * 5 + [0.6] * 3 + [0.3] * (len(remaining_players) - 8)
                minute_weights = minute_weights[:len(remaining_players)]
                
                # Normalize weights
                total_weight = sum(minute_weights)
                minute_weights = [w / total_weight for w in minute_weights]
                
                # Assign minutes
                for i, player in enumerate(remaining_players):
                    player['avg_minutes'] = remaining_minutes * minute_weights[i]
                    
                total_minutes = remaining_minutes
        
        # Distribute points proportional to minutes
        if total_minutes > 0:
            # Home advantage factor
            home_factor = 1.05 if is_home else 0.95
            
            for player in remaining_players:
                # Get minutes and calculate points
                minutes = player.get('avg_minutes', 10)
                minutes_share = minutes / total_minutes
                
                # Calculate expected points based on minutes share and team total
                expected_points = points_to_distribute * minutes_share * home_factor
                
                # Reasonable points per minute rates based on role
                if minutes >= 30:  # Star player
                    points_per_minute = random.uniform(0.7, 0.9)
                elif minutes >= 20:  # Starter
                    points_per_minute = random.uniform(0.5, 0.7)
                elif minutes >= 10:  # Rotation player
                    points_per_minute = random.uniform(0.3, 0.5)
                else:  # Bench player
                    points_per_minute = random.uniform(0.2, 0.4)
                
                # Alternative calculation based on realistic points per minute
                alt_points = minutes * points_per_minute
                
                # Blend the two approaches
                points = (expected_points * 0.7) + (alt_points * 0.3)
                
                # Add some variability (±15%)
                variance = random.uniform(0.85, 1.15)
                final_points = points * variance
                
                # Ensure no negative points
                final_points = max(0, round(final_points, 1))
                
                # Add prediction
                backup_preds.append({
                    'name': player['full_name'],
                    'points': final_points,
                    'minutes': round(minutes, 1),
                    'status': 'Active (est)'
                })
        
        # Sort by points and return
        return sorted(backup_preds, key=lambda x: x['points'], reverse=True)

    def predict_player_performance(self, player_id, is_home=True, opp_team_id=None, minutes=None):
        """
        Predict player performance with more realistic point estimation
        """
        try:
            # Get player stats
            stats = self.collector.get_player_stats(player_id)
            if stats is None:
                return None
                
            # Check if player is inactive or not playing
            if stats.get('INACTIVE', False) or stats.get('GAMES_PLAYED', 0) == 0:
                return {
                    'points': 0.0,
                    'minutes': 0.0,
                    'status': 'Inactive'
                }
                
            # Add home/away context
            stats['HOME_GAME'] = 1 if is_home else 0
            
            # Add opponent defensive strength if available
            if opp_team_id is not None:
                opp_stats = self.collector.get_team_stats(opp_team_id)
                if opp_stats is not None:
                    # Adjust for opponent's defensive rating - higher means worse defense
                    def_rating = opp_stats.get('DEF_RATING', 110)
                    # League average is around 110, so >110 means easier to score against
                    stats['OPP_DEF_RATING'] = def_rating
                    
                    # Get opponent's points allowed
                    pts_allowed = opp_stats.get('OPP_PTS_ALLOWED', 110)
                    stats['OPP_PTS_ALLOWED'] = pts_allowed
            
            # If using model-based prediction
            if self.models_loaded and not self.fallback_mode:
                try:
                    # Create features dataframe
                    features = pd.DataFrame([stats])
                    
                    # Handle missing features that might be in the model
                    for col in self.trainer.player_model.feature_names_in_:
                        if col not in features.columns:
                            features[col] = 0
                            
                    # Keep only features used in the model
                    features = features[self.trainer.player_model.feature_names_in_]
                    
                    # Scale features
                    scaled_features = self.trainer.player_scaler.transform(features)
                    
                    # Get position for position-specific model if available
                    player_position = None
                    if 'POSITION' in stats:
                        player_position = stats['POSITION']
                    
                    # Use position-specific model if available, otherwise use general model
                    if player_position and player_position in self.trainer.position_models:
                        # Need to create features for position model (might have different columns)
                        pos_model = self.trainer.position_models[player_position]
                        pos_scaler = self.trainer.position_scalers[player_position]
                        
                        # Handle missing features for position model
                        pos_features = pd.DataFrame([stats])
                        for col in pos_model.feature_names_in_:
                            if col not in pos_features.columns:
                                pos_features[col] = 0
                                
                        # Keep only features used in position model
                        pos_features = pos_features[pos_model.feature_names_in_]
                        
                        # Scale features
                        pos_scaled_features = pos_scaler.transform(pos_features)
                        
                        # Blend predictions from general and position-specific models
                        general_pred = self.trainer.player_model.predict(scaled_features)[0]
                        position_pred = pos_model.predict(pos_scaled_features)[0]
                        
                        # Weight position model more heavily (60/40 split)
                        predicted_points = (0.4 * general_pred) + (0.6 * position_pred)
                    else:
                        # Use general model only
                        predicted_points = self.trainer.player_model.predict(scaled_features)[0]
                    
                    # Apply adjustments
                    # Home court boost
                    if is_home:
                        predicted_points *= 1.03
                    else:
                        predicted_points *= 0.97
                    
                    # Apply opponent adjustment if available
                    if 'OPP_DEF_RATING' in stats:
                        # Scale based on league average (110)
                        def_factor = stats['OPP_DEF_RATING'] / 110.0
                        # Reduce the impact for more stability (80% effect)
                        def_adjustment = ((def_factor - 1.0) * 0.8) + 1.0
                        predicted_points *= def_adjustment
                    
                    # Apply minutes adjustment if provided minutes differ from historical
                    if minutes is not None and 'MIN' in stats and stats['MIN'] > 0:
                        min_factor = minutes / stats['MIN']
                        # Limit the adjustment to a reasonable range
                        min_factor = min(1.5, max(0.5, min_factor))
                        # Apply with reduced impact for stability
                        min_adjustment = ((min_factor - 1.0) * 0.7) + 1.0
                        predicted_points *= min_adjustment
                        
                        # Update minutes to the provided value
                        predicted_minutes = minutes
                    else:
                        predicted_minutes = stats.get('MIN', 24)
                    
                    # Apply recent form adjustment
                    if 'STREAK' in stats:
                        streak = stats['STREAK']
                        # Limit the impact of hot/cold streaks
                        streak_adjustment = 1.0 + (min(2.0, max(-2.0, streak)) * 0.03)
                        predicted_points *= streak_adjustment
                    
                    # Apply final sanity check
                    points_per_minute = predicted_points / max(1, predicted_minutes)
                    
                    # If points per minute looks unreasonable, adjust it
                    if points_per_minute > 1.2:  # Extremely high (>1.2 pts/min)
                        predicted_points = predicted_minutes * 1.2
                    elif points_per_minute < 0.1 and predicted_minutes > 10:  # Very low for a rotation player
                        predicted_points = predicted_minutes * 0.1
                    
                    # Add randomness for realism (±5%)
                    variance = random.uniform(0.95, 1.05)
                    final_points = predicted_points * variance
                    
                    return {
                        'points': round(max(0, final_points), 1),
                        'minutes': round(predicted_minutes, 1),
                        'status': 'Active'
                    }
                except Exception as e:
                    logger.error(f"Error in player model prediction: {str(e)}")
                    # Fall through to stats-based approach on error
            
            # Fallback to statistical approach
            # Base prediction on recent average with adjustments
            base_pts = stats.get('PTS_AVG', 0)
            base_min = stats.get('MIN', 24)
            
            # Update minutes if provided
            if minutes is not None:
                # Calculate points per minute from historical data
                pts_per_min = base_pts / max(1, base_min)
                # Apply minutes adjustment with reasonable scaling
                min_factor = minutes / max(1, base_min)
                min_factor = min(1.5, max(0.5, min_factor))  # Limit to reasonable range
                # Scale points by minutes change (not 1:1)
                base_pts = pts_per_min * minutes * (0.7 + 0.3 * min_factor)
                # Update minutes
                base_min = minutes
            
            # Apply home court adjustment
            if is_home:
                base_pts *= 1.03  # Players tend to perform better at home
            else:
                base_pts *= 0.97  # And worse on the road
                
            # Apply opponent defensive adjustment if available
            if 'OPP_DEF_RATING' in stats:
                def_factor = stats['OPP_DEF_RATING'] / 110.0
                def_adjustment = ((def_factor - 1.0) * 0.8) + 1.0  # Reduced impact
                base_pts *= def_adjustment
                
            # Apply recent form adjustment
            trend = stats.get('TREND', 0)
            if abs(trend) > 1:  # Only if there's a significant trend
                trend_factor = 1 + (trend / (base_pts * 5)) if base_pts > 0 else 1
                trend_factor = min(1.15, max(0.85, trend_factor))  # Limit adjustment
                base_pts *= trend_factor
                
            # Apply consistency adjustment
            consistency = stats.get('CONSISTENCY', 0.5)
            # More consistent players have less variance
            variance = random.uniform(
                1.0 - (0.1 * (1 - consistency)),
                1.0 + (0.1 * (1 - consistency))
            )
            base_pts *= variance
            
            # Final sanity check on points per minute
            points_per_minute = base_pts / max(1, base_min)
            if points_per_minute > 1.2:  # Extremely high
                base_pts = base_min * 1.2
            elif points_per_minute < 0.1 and base_min > 10:  # Very low for a rotation player
                base_pts = base_min * 0.1
                
            return {
                'points': round(max(0, base_pts), 1),
                'minutes': round(base_min, 1),
                'status': 'Active'
            }
                
        except Exception as e:
            logger.error(f"Error predicting player performance: {str(e)}")
            return None

    def get_team_players(self, team_id):
        """Get active players for a team with their stats"""
        try:
            players = self.collector.get_team_players(team_id)
            return [p for p in players if p.get('avg_minutes', 0) >= 2]
        except Exception as e:
            logger.error(f"Error getting team players: {str(e)}")
            return []

    def predict_player_performance(self, player_id, is_home=True, opp_team_id=None, minutes=None):
        """
        Predict player performance with more realistic point estimation and variance
        
        Args:
            player_id: NBA API player ID
            is_home: Whether player is on home team
            opp_team_id: Opponent team ID for matchup analysis
            minutes: Override projected minutes
            
        Returns:
            Dictionary with points prediction and confidence metrics
        """
        try:
            # Get player's recent game data
            recent_games = self.collector.get_player_recent_games(player_id, n_games=15)
            if recent_games is None or len(recent_games) < 5:
                return self._generate_fallback_prediction(player_id, is_home, minutes)
            
            # Get player stats from collector
            player_stats = self.collector.get_player_stats(player_id)
            if player_stats is None:
                return self._generate_fallback_prediction(player_id, is_home, minutes)
            
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
                recent_games, player_stats, minutes, is_home
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
            
            # Apply bounds to points per minute (0.15-1.2 is a reasonable range)
            pts_per_min = max(0.15, min(1.2, pts_per_min))
            
            # Calculate standard deviation of points per minute
            pts_per_min_std = max(0.1, np.std(pts_per_min_list))
            
            # Adjust for sample size - smaller samples have higher uncertainty
            sample_size_factor = min(1.0, len(recent_games) / 20)
            pts_per_min_std = pts_per_min_std * (1 + (1 - sample_size_factor))
            
            # Apply adjustments for matchup and home court
            if is_home:
                pts_per_min *= 1.03  # 3% boost for home games
            
            # Apply opponent defense adjustment
            if opp_team_id is not None:
                pts_per_min = self._apply_opponent_adjustment(
                    pts_per_min, opp_team_id, player_stats
                )
            
            # Calculate projected points
            points_projection = pts_per_min * minutes_projection
            
            # Calculate standard deviation of points
            # Uncertainty comes from both minutes and scoring rate variability
            points_std = np.sqrt(
                (minutes_projection * pts_per_min_std)**2 + 
                (minutes_std * pts_per_min)**2
            )
            
            # Reduce standard deviation for players with more games (more predictable)
            games_played_factor = min(1.0, 5 / max(1, player_stats.get('GAMES_PLAYED', 0)))
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
            logger.error(f"Error predicting player performance: {str(e)}")
            return self._generate_fallback_prediction(player_id, is_home, minutes)

def _project_minutes(self, recent_games, player_stats, minutes_override=None, is_home=True):
    """
    Project minutes for a player with improved algorithm
    
    Args:
        recent_games: DataFrame of player's recent games
        player_stats: Dictionary of player statistics
        minutes_override: Override projected minutes
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

def _apply_opponent_adjustment(self, pts_per_min, opp_team_id, player_stats):
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

def _calculate_projection_quality(self, recent_games, player_stats, relative_std):
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

def _generate_fallback_prediction(self, player_id, is_home, minutes_override=None):
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
                points *= 1.03
            
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

def predict_game(self, home_team_id, away_team_id):
    """
    Enhanced predict_game method that adds standard deviations for probabilistic modeling
    """
    try:
        # Get base prediction using existing method
        base_prediction = super().predict_game(home_team_id, away_team_id)
        
        if 'error' in base_prediction and base_prediction['error']:
            return base_prediction

        # Add standard deviations to team scores if not already present
        if 'home_std' not in base_prediction:
            base_prediction['home_std'] = self._estimate_team_std(base_prediction['home_score'], home_team_id)
        if 'away_std' not in base_prediction:
            base_prediction['away_std'] = self._estimate_team_std(base_prediction['away_score'], away_team_id)

        # Enhance player predictions with standard deviations
        self._enhance_player_predictions(base_prediction)
        
        return base_prediction
    except Exception as e:
        logger.error(f"Error in enhanced predict_game: {str(e)}")
        return {'error': str(e)}

def _estimate_team_std(self, score, team_id):
    """
    Estimate standard deviation for team score
    
    Args:
        score: Predicted score
        team_id: Team ID
        
    Returns:
        Estimated standard deviation
    """
    # Get team stats if available
    team_stats = self.collector.get_team_stats(team_id)
    
    # If we have historical standard deviation, use it
    if team_stats and 'PTS_STD' in team_stats:
        # Use historical std with a reasonable floor
        return max(8.0, team_stats['PTS_STD'])
    
    # If no historical data, use a reasonable estimate
    # NBA teams typically have standard deviations around 10-12 points
    # Higher scoring teams tend to have higher variance
    base_std = 11.0
    
    # Adjust based on predicted score (higher scores = higher variance)
    score_factor = score / 110.0  # Normalized to average NBA team score
    
    return max(8.0, base_std * score_factor)

def _enhance_player_predictions(self, prediction):
    """
    Enhance player predictions with standard deviations
    
    Args:
        prediction: Prediction dictionary to enhance
        
    Returns:
        None (modifies prediction in place)
    """
    # Process home players
    for i, player in enumerate(prediction.get('home_player_predictions', [])):
        if 'points_std' not in player:
            # Calculate standard deviation if not present
            points = player['points']
            minutes = player.get('minutes', 0)
            
            # Different variance for different roles
            if minutes >= 30:  # Starters/stars
                rel_std = 0.30  # 30% relative std
            elif minutes >= 20:  # Rotation players
                rel_std = 0.35  # 35% relative std
            else:  # Bench players
                rel_std = 0.45  # 45% relative std
                
            # Calculate and add std
            points_std = max(3.0, points * rel_std)
            prediction['home_player_predictions'][i]['points_std'] = points_std
            
            # Add confidence interval
            ci_lower = max(0, points - 1.96 * points_std)
            ci_upper = points + 1.96 * points_std
            prediction['home_player_predictions'][i]['points_95ci'] = (round(ci_lower, 1), round(ci_upper, 1))
    
    # Process away players (same logic)
    for i, player in enumerate(prediction.get('away_player_predictions', [])):
        if 'points_std' not in player:
            points = player['points']
            minutes = player.get('minutes', 0)
            
            if minutes >= 30:
                rel_std = 0.30
            elif minutes >= 20:
                rel_std = 0.35
            else:
                rel_std = 0.45
                
            points_std = max(3.0, points * rel_std)
            prediction['away_player_predictions'][i]['points_std'] = points_std
            
            ci_lower = max(0, points - 1.96 * points_std)
            ci_upper = points + 1.96 * points_std
            prediction['away_player_predictions'][i]['points_95ci'] = (round(ci_lower, 1), round(ci_upper, 1))
        