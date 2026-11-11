import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
import random

# Import our modules
from data_collector import DataCollector, GameContext
from model_trainer import EnhancedModelTrainer

@dataclass
class PredictionResult:
    """Structured prediction result"""
    team_score: float
    team_score_std: float
    confidence: float
    method: str
    context: GameContext
    features_used: int
    model_agreement: float

@dataclass
class PlayerPrediction:
    """Structured player prediction result"""
    points: float
    points_std: float
    minutes: float
    minutes_std: float
    confidence: float
    usage_rate: float
    matchup_factor: float

class EnhancedNBAPredictor:
    """Enhanced NBA Predictor with advanced ML pipeline and feature engineering"""
    
    def _robust_fallback_prediction(self, team_id: int, is_home: bool = True) -> Dict:
        """Ultra-robust fallback prediction when all else fails"""
        try:
            # Use league averages with basic adjustments
            base_score = 110.0  # League average
            
            # Home court advantage
            if is_home:
                base_score += 3.5
            
            # Add some randomness to avoid identical predictions
            import random
            random_factor = random.uniform(-2, 2)
            final_score = base_score + random_factor
            
            return {
                'prediction': final_score,
                'std': 12.0,
                'individual_predictions': {'fallback': final_score},
                'features_used': []
            }
            
        except Exception as e:
            logging.error(f"Even fallback prediction failed: {e}")
            return {
                'prediction': 110.0,
                'std': 15.0,
                'individual_predictions': {'emergency': 110.0},
                'features_used': []
            }
    
    def __init__(self, use_models: bool = True):
        self.collector = EnhancedDataCollector()
        self.trainer = EnhancedModelTrainer(self.collector)
        self.use_models = use_models
        self.models_loaded = False
        
        # Model performance tracking
        self.model_performance = {
            'team_mae': None,
            'player_points_mae': None,
            'player_minutes_mae': None,
            'last_evaluation': None
        }
        
        # Load models if available
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models (load if available, train if needed)"""
        try:
            if self.use_models:
                # Try to load enhanced models
                success = self.trainer.load_models()
                if success:
                    self.models_loaded = True
                    logging.info("Enhanced models loaded successfully")
                else:
                    logging.info("Enhanced models not found, will train new ones")
                    self.models_loaded = False
            else:
                # Use simpler fallback models
                self.models_loaded = False
                logging.info("Using statistical prediction methods")
                
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            self.models_loaded = False

    def train_models(self, force_retrain: bool = False):
        """Train or retrain models"""
        try:
            if force_retrain or not self.models_loaded:
                logging.info("Training enhanced models...")
                
                success = self.trainer.train_ensemble_models()
                if success:
                    self.trainer.save_models()
                    self.models_loaded = True
                    logging.info("Model training completed successfully")
                    return True
                else:
                    logging.error("Model training failed")
                    return False
            else:
                logging.info("Models already loaded, use force_retrain=True to retrain")
                return True
                
        except Exception as e:
            logging.error(f"Error training models: {e}")
            return False

    def predict_game(self, home_team_id: int, away_team_id: int, 
                    game_date: str = None, detailed: bool = True) -> Dict:
        """
        Predict game outcome with enhanced features and confidence metrics
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date (optional, for future games)
            detailed: Whether to include detailed player predictions
            
        Returns:
            Comprehensive prediction dictionary
        """
        try:
            # Get game contexts
            home_context = self.collector.get_game_context(home_team_id, game_date)
            away_context = self.collector.get_game_context(away_team_id, game_date)
            
            # Get team predictions
            home_prediction = self._predict_team_performance(
                home_team_id, away_team_id, is_home=True, context=home_context
            )
            away_prediction = self._predict_team_performance(
                away_team_id, home_team_id, is_home=False, context=away_context
            )
            
            if not home_prediction or not away_prediction:
                return {'error': 'Failed to generate team predictions'}
            
            # Get player predictions if detailed
            player_predictions = {}
            if detailed:
                player_predictions = self._predict_all_players(
                    home_team_id, away_team_id, home_context, away_context
                )
            
            # Calculate game-level metrics
            total_score = home_prediction.team_score + away_prediction.team_score
            margin = home_prediction.team_score - away_prediction.team_score
            
            # Calculate confidence based on multiple factors
            overall_confidence = self._calculate_game_confidence(
                home_prediction, away_prediction, home_context, away_context
            )
            
            # Determine winner and probability
            home_win_prob = self._calculate_win_probability(
                home_prediction.team_score, away_prediction.team_score,
                home_prediction.team_score_std, away_prediction.team_score_std
            )
            
            # Calculate pace and style metrics
            pace_analysis = self._analyze_game_pace(home_team_id, away_team_id)
            
            # Build comprehensive result
            result = {
                'game_info': {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'game_date': game_date,
                    'prediction_time': datetime.now().isoformat()
                },
                'predictions': {
                    'home_score': round(home_prediction.team_score, 1),
                    'away_score': round(away_prediction.team_score, 1),
                    'total_score': round(total_score, 1),
                    'margin': round(margin, 1),
                    'home_win_probability': round(home_win_prob, 3),
                    'away_win_probability': round(1 - home_win_prob, 3)
                },
                'uncertainty': {
                    'home_score_std': round(home_prediction.team_score_std, 2),
                    'away_score_std': round(away_prediction.team_score_std, 2),
                    'total_score_range': (
                        round(total_score - 2 * np.sqrt(home_prediction.team_score_std**2 + away_prediction.team_score_std**2), 1),
                        round(total_score + 2 * np.sqrt(home_prediction.team_score_std**2 + away_prediction.team_score_std**2), 1)
                    ),
                    'overall_confidence': round(overall_confidence, 3)
                },
                'context': {
                    'home_rest_days': home_context.rest_days,
                    'away_rest_days': away_context.rest_days,
                    'home_back_to_back': home_context.is_back_to_back,
                    'away_back_to_back': away_context.is_back_to_back,
                    'season_progress': round(home_context.season_progress, 3),
                    'pace_analysis': pace_analysis
                },
                'methodology': {
                    'home_method': home_prediction.method,
                    'away_method': away_prediction.method,
                    'features_used': home_prediction.features_used + away_prediction.features_used,
                    'model_agreement': round((home_prediction.model_agreement + away_prediction.model_agreement) / 2, 3),
                    'enhanced_models': self.models_loaded
                },
                'player_predictions': player_predictions if detailed else {},
                'error': None
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error in game prediction: {e}")
            return {'error': str(e)}

    def _predict_team_performance(self, team_id: int, opponent_id: int, 
                                is_home: bool, context: GameContext) -> PredictionResult:
        """Predict team performance using enhanced models or fallback methods"""
        try:
            if self.models_loaded:
                # Use enhanced models
                prediction_result = self.trainer.predict_team_score(
                    team_id, opponent_id, is_home
                )
                
                if prediction_result:
                    # Calculate model agreement (how close are individual model predictions)
                    individual_preds = list(prediction_result['individual_predictions'].values())
                    if len(individual_preds) > 1:
                        pred_std = np.std(individual_preds)
                        model_agreement = max(0, 1 - (pred_std / np.mean(individual_preds)))
                    else:
                        model_agreement = 0.8
                    
                    return PredictionResult(
                        team_score=prediction_result['prediction'],
                        team_score_std=prediction_result['std'],
                        confidence=model_agreement,
                        method='enhanced_ensemble',
                        context=context,
                        features_used=len(prediction_result.get('features_used', [])),
                        model_agreement=model_agreement
                    )
            
            # Fallback to statistical method
            stat_result = self._statistical_team_prediction(team_id, opponent_id, is_home, context)
            if stat_result:
                return stat_result
            
            # If statistical method also fails, use robust fallback
            fallback_result = self._robust_fallback_prediction(team_id, is_home)
            return PredictionResult(
                team_score=fallback_result['prediction'],
                team_score_std=fallback_result['std'],
                confidence=0.2,
                method='robust_fallback',
                context=context,
                features_used=0,
                model_agreement=0.2
            )
            
        except Exception as e:
            logging.error(f"Error predicting team performance: {e}")
            stat_result = self._statistical_team_prediction(team_id, opponent_id, is_home, context)
            if stat_result:
                return stat_result
            
            # Last resort fallback
            fallback_result = self._robust_fallback_prediction(team_id, is_home)
            return PredictionResult(
                team_score=fallback_result['prediction'],
                team_score_std=fallback_result['std'],
                confidence=0.1,
                method='emergency_fallback',
                context=context,
                features_used=0,
                model_agreement=0.1
            )

    def _statistical_team_prediction(self, team_id: int, opponent_id: int, 
                                   is_home: bool, context: GameContext) -> PredictionResult:
        """Fallback statistical prediction method"""
        try:
            # Get advanced team stats
            team_stats = self.collector.get_team_stats(team_id)
            opp_stats = self.collector.get_team_stats(opponent_id)
            
            if not team_stats or not opp_stats:
                # Use basic averages as last resort
                base_score = 110.0 + (5.0 if is_home else 0.0)
                return PredictionResult(
                    team_score=base_score,
                    team_score_std=12.0,
                    confidence=0.3,
                    method='basic_average',
                    context=context,
                    features_used=0,
                    model_agreement=0.3
                )
            
            # Base prediction from recent performance
            base_score = team_stats.get('pts_avg', 110)
            
            # Apply adjustments
            adjustments = 0.0
            
            # Home court advantage
            if is_home:
                home_advantage = team_stats.get('home_road_splits', {}).get('home_pts_avg', base_score) - \
                               team_stats.get('home_road_splits', {}).get('road_pts_avg', base_score)
                adjustments += min(8.0, max(2.0, home_advantage))  # Cap home advantage
            
            # Opponent defense adjustment
            opp_def_rating = opp_stats.get('def_rating', 115)
            league_avg_def = 115.0
            def_factor = opp_def_rating / league_avg_def
            adjustments += (def_factor - 1.0) * base_score * 0.1  # 10% max adjustment
            
            # Rest adjustment
            if context.is_back_to_back:
                adjustments -= 3.0  # Back-to-back penalty
            elif context.rest_days >= 3:
                adjustments += 1.5  # Rest bonus
            
            # Season timing adjustment
            if context.season_progress > 0.8:  # Late season
                # Teams might rest players or play with different intensity
                adjustments -= 2.0
            
            # Recent form adjustment
            recent_form = team_stats.get('recent_form', {})
            if recent_form:
                trend = recent_form.get('pts_trend', 0)
                adjustments += min(5.0, max(-5.0, trend * 2))  # Cap trend adjustment
            
            final_score = base_score + adjustments
            
            # Calculate uncertainty
            score_std = team_stats.get('pts_std', 12.0)
            # Increase uncertainty for more adjustments
            uncertainty_factor = 1.0 + abs(adjustments) / base_score
            final_std = score_std * uncertainty_factor
            
            # Calculate confidence based on available data
            confidence = 0.7
            if team_stats.get('games_played', 0) < 20:
                confidence *= 0.8  # Reduce confidence for small sample
            if recent_form:
                confidence *= 1.1  # Boost for recent form data
            
            confidence = min(0.9, confidence)
            
            return PredictionResult(
                team_score=final_score,
                team_score_std=final_std,
                confidence=confidence,
                method='statistical_enhanced',
                context=context,
                features_used=len([k for k in team_stats.keys() if k != 'games_played']),
                model_agreement=confidence
            )
            
        except Exception as e:
            logging.error(f"Error in statistical prediction: {e}")
            # Last resort prediction
            base_score = 110.0 + (4.0 if is_home else 0.0)
            return PredictionResult(
                team_score=base_score,
                team_score_std=15.0,
                confidence=0.2,
                method='fallback',
                context=context,
                features_used=0,
                model_agreement=0.2
            )

    def _predict_all_players(self, home_team_id: int, away_team_id: int,
                           home_context: GameContext, away_context: GameContext) -> Dict:
        """Predict performance for all players in the game"""
        try:
            home_players = self._predict_team_players(home_team_id, away_team_id, True, home_context)
            away_players = self._predict_team_players(away_team_id, home_team_id, False, away_context)
            
            return {
                'home_players': home_players,
                'away_players': away_players,
                'total_home_predicted': sum(p['points'] for p in home_players),
                'total_away_predicted': sum(p['points'] for p in away_players)
            }
            
        except Exception as e:
            logging.error(f"Error predicting all players: {e}")
            return {}

    def _predict_team_players(self, team_id: int, opponent_id: int, 
                            is_home: bool, context: GameContext) -> List[Dict]:
        """Predict performance for all players on a team"""
        try:
            # Get team roster
            roster = self.collector.get_team_players(team_id)
            if not roster:
                return []
            
            player_predictions = []
            
            for player in roster[:12]:  # Top 12 players
                player_id = player['id']
                player_name = player['full_name']
                
                # Predict player performance
                if self.models_loaded:
                    pred_result = self.trainer.predict_player_performance(
                        player_id, team_id, opponent_id, is_home
                    )
                else:
                    pred_result = self._statistical_player_prediction(
                        player_id, opponent_id, is_home, context
                    )
                
                if pred_result:
                    player_predictions.append({
                        'name': player_name,
                        'points': round(pred_result['points'], 1),
                        'points_std': round(pred_result['points_std'], 2),
                        'minutes': round(pred_result['minutes'], 1),
                        'minutes_std': round(pred_result['minutes_std'], 2),
                        'confidence': round(pred_result.get('confidence', 0.5), 3),
                        'status': 'Active'
                    })
            
            # Sort by predicted points
            player_predictions.sort(key=lambda x: x['points'], reverse=True)
            
            return player_predictions
            
        except Exception as e:
            logging.error(f"Error predicting team players: {e}")
            return []

    def _statistical_player_prediction(self, player_id: int, opponent_id: int,
                                     is_home: bool, context: GameContext) -> Dict:
        """Fallback statistical player prediction"""
        try:
            # Get enhanced player stats
            player_stats = self.collector.get_player_stats(player_id)
            if not player_stats or player_stats.get('status') == 'inactive':
                return {
                    'points': 0.0,
                    'points_std': 0.0,
                    'minutes': 0.0,
                    'minutes_std': 0.0,
                    'confidence': 0.0
                }
            
            # Base predictions from stats
            base_points = player_stats.get('pts_avg', 0)
            base_minutes = player_stats.get('minutes_avg', 0)
            
            # Apply adjustments
            points_adjustments = 0.0
            minutes_adjustments = 0.0
            
            # Home/away adjustment
            if is_home:
                home_road = player_stats.get('situational', {}).get('home_away', {})
                if home_road:
                    home_advantage = home_road.get('home_advantage', 0)
                    points_adjustments += min(3.0, max(-3.0, home_advantage))
            
            # Recent form adjustment
            recent_form = player_stats.get('recent_form', {})
            if recent_form:
                hot_streak = recent_form.get('hot_streak', {})
                if hot_streak.get('type') == 'hot':
                    points_adjustments += 2.0
                elif hot_streak.get('type') == 'cold':
                    points_adjustments -= 2.0
                
                # Trend adjustment
                trend = recent_form.get('trend_l10', 0)
                points_adjustments += min(2.0, max(-2.0, trend))
            
            # Rest adjustment
            rest_analysis = player_stats.get('situational', {}).get('rest', {})
            if context.is_back_to_back and rest_analysis:
                b2b_performance = rest_analysis.get('back_to_back', {})
                if b2b_performance:
                    b2b_pts = b2b_performance.get('pts_avg', base_points)
                    b2b_min = b2b_performance.get('min_avg', base_minutes)
                    points_adjustments += (b2b_pts - base_points) * 0.5
                    minutes_adjustments += (b2b_min - base_minutes) * 0.5
                else:
                    # General back-to-back penalty
                    points_adjustments -= 1.5
                    minutes_adjustments -= 2.0
            
            # Usage and role adjustments
            usage = player_stats.get('usage_estimate', 0.2)
            role = player_stats.get('role', 'bench')
            
            if role == 'star' and usage > 0.25:
                # Star players might see increased usage against tough opponents
                points_adjustments += 1.0
            elif role == 'bench' and context.is_back_to_back:
                # Bench players might get more opportunities on back-to-backs
                minutes_adjustments += 3.0
                points_adjustments += 1.0
            
            # Apply adjustments
            final_points = max(0, base_points + points_adjustments)
            final_minutes = max(0, base_minutes + minutes_adjustments)
            
            # Calculate uncertainty
            points_std = player_stats.get('pts_std', final_points * 0.35)
            minutes_std = player_stats.get('minutes_std', final_minutes * 0.25)
            
            # Adjust uncertainty based on consistency
            consistency = player_stats.get('consistency', {})
            if consistency:
                cv = consistency.get('pts_cv', 0.35)
                points_std = max(2.0, final_points * cv)
            
            # Calculate confidence
            confidence = 0.6
            games_played = player_stats.get('games_played', 0)
            if games_played >= 20:
                confidence += 0.1
            if consistency and consistency.get('pts_cv', 0.5) < 0.3:
                confidence += 0.1  # More consistent player
            if recent_form:
                confidence += 0.05  # Have recent form data
            
            confidence = min(0.85, confidence)
            
            return {
                'points': final_points,
                'points_std': points_std,
                'minutes': final_minutes,
                'minutes_std': minutes_std,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error in statistical player prediction: {e}")
            return {
                'points': 8.0,
                'points_std': 5.0,
                'minutes': 20.0,
                'minutes_std': 8.0,
                'confidence': 0.3
            }

    def _calculate_game_confidence(self, home_pred: PredictionResult, away_pred: PredictionResult,
                                 home_context: GameContext, away_context: GameContext) -> float:
        """Calculate overall game prediction confidence"""
        try:
            # Base confidence from individual team predictions
            base_confidence = (home_pred.confidence + away_pred.confidence) / 2
            
            # Adjust based on context factors
            context_adjustments = 0.0
            
            # Both teams well-rested increases confidence
            if not home_context.is_back_to_back and not away_context.is_back_to_back:
                context_adjustments += 0.05
            
            # Similar rest levels increase confidence
            rest_diff = abs(home_context.rest_days - away_context.rest_days)
            if rest_diff <= 1:
                context_adjustments += 0.03
            elif rest_diff >= 3:
                context_adjustments -= 0.05
            
            # Model agreement factor
            model_agreement = (home_pred.model_agreement + away_pred.model_agreement) / 2
            agreement_bonus = (model_agreement - 0.5) * 0.2  # Up to 0.1 bonus/penalty
            
            # Feature richness factor
            total_features = home_pred.features_used + away_pred.features_used
            if total_features > 50:
                context_adjustments += 0.05
            elif total_features < 20:
                context_adjustments -= 0.05
            
            # Season timing factor
            if home_context.season_progress > 0.8:
                # Late season can be less predictable due to rest/playoff positioning
                context_adjustments -= 0.03
            
            final_confidence = base_confidence + context_adjustments + agreement_bonus
            return max(0.1, min(0.95, final_confidence))
            
        except Exception as e:
            logging.error(f"Error calculating game confidence: {e}")
            return 0.5

    def _calculate_win_probability(self, home_score: float, away_score: float,
                                 home_std: float, away_std: float) -> float:
        """Calculate home team win probability using score distributions"""
        try:
            # Calculate the margin distribution (home - away)
            margin_mean = home_score - away_score
            margin_std = np.sqrt(home_std**2 + away_std**2)
            
            # Use normal distribution to calculate P(margin > 0)
            from scipy import stats
            win_prob = 1 - stats.norm.cdf(0, loc=margin_mean, scale=margin_std)
            
            return max(0.01, min(0.99, win_prob))
            
        except Exception as e:
            logging.error(f"Error calculating win probability: {e}")
            # Fallback to simple logistic function
            margin = home_score - away_score
            return 1 / (1 + np.exp(-margin / 10))

    def _analyze_game_pace(self, home_team_id: int, away_team_id: int) -> Dict:
        """Analyze expected game pace and style"""
        try:
            home_stats = self.collector.get_team_stats(home_team_id)
            away_stats = self.collector.get_team_stats(away_team_id)
            
            if not home_stats or not away_stats:
                return {'pace': 100.0, 'style': 'average'}
            
            # Average pace
            home_pace = home_stats.get('pace', 100)
            away_pace = away_stats.get('pace', 100)
            expected_pace = (home_pace + away_pace) / 2
            
            # Style analysis
            if expected_pace > 105:
                style = 'fast'
            elif expected_pace < 95:
                style = 'slow'
            else:
                style = 'average'
            
            # Efficiency analysis
            home_eff = home_stats.get('off_rating', 115)
            away_eff = away_stats.get('off_rating', 115)
            
            return {
                'expected_pace': round(expected_pace, 1),
                'home_pace': round(home_pace, 1),
                'away_pace': round(away_pace, 1),
                'style': style,
                'pace_differential': round(abs(home_pace - away_pace), 1),
                'efficiency_matchup': {
                    'home_off_rating': round(home_eff, 1),
                    'away_off_rating': round(away_eff, 1),
                    'efficiency_advantage': 'home' if home_eff > away_eff else 'away'
                }
            }
            
        except Exception as e:
            logging.error(f"Error analyzing game pace: {e}")
            return {'pace': 100.0, 'style': 'average'}

    def get_team_players(self, team_id: int) -> List[Dict]:
        """Get team players with enhanced stats"""
        try:
            return self.collector.get_team_players(team_id)
        except Exception as e:
            logging.error(f"Error getting team players: {e}")
            return []

    def evaluate_predictions(self, n_games: int = 20) -> Dict:
        """Evaluate model performance on recent games"""
        try:
            if not self.models_loaded:
                return {'error': 'Models not loaded for evaluation'}
            
            from nba_api.stats.endpoints import LeagueGameLog
            
            # Get recent completed games
            recent_games = LeagueGameLog(
                season='2023-24',
                season_type_all_star='Regular Season',
                headers=self.collector._get_headers()
            ).get_data_frames()[0]
            
            if recent_games.empty:
                return {'error': 'No recent games found'}
            
            # Group by GAME_ID to get pairs of teams
            game_groups = recent_games.groupby('GAME_ID')
            
            evaluation_results = []
            team_errors = []
            games_processed = 0
            
            for game_id, game_data in game_groups:
                if games_processed >= n_games:
                    break
                    
                # Skip if we don't have exactly 2 teams (which should be the case)
                if len(game_data) != 2:
                    continue
                    
                try:
                    # Sort by MATCHUP to identify home/away teams
                    # Home team MATCHUP contains "vs.", away team contains "@"
                    home_team_data = game_data[game_data['MATCHUP'].str.contains('vs.', na=False)]
                    away_team_data = game_data[game_data['MATCHUP'].str.contains('@', na=False)]
                    
                    if home_team_data.empty or away_team_data.empty:
                        continue
                    
                    home_team_id = home_team_data.iloc[0]['TEAM_ID']
                    away_team_id = away_team_data.iloc[0]['TEAM_ID']
                    actual_home_score = float(home_team_data.iloc[0]['PTS'])
                    actual_away_score = float(away_team_data.iloc[0]['PTS'])
                    
                    # Make prediction
                    prediction = self.predict_game(
                        home_team_id, away_team_id, detailed=False
                    )
                    
                    if 'error' in prediction:
                        continue
                    
                    pred_home = prediction['predictions']['home_score']
                    pred_away = prediction['predictions']['away_score']
                    
                    # Calculate errors
                    home_error = abs(pred_home - actual_home_score)
                    away_error = abs(pred_away - actual_away_score)
                    total_error = home_error + away_error
                    
                    # Winner prediction accuracy
                    actual_winner = 'home' if actual_home_score > actual_away_score else 'away'
                    pred_winner = 'home' if pred_home > pred_away else 'away'
                    correct_winner = actual_winner == pred_winner
                    
                    # Margin accuracy
                    actual_margin = actual_home_score - actual_away_score
                    pred_margin = pred_home - pred_away
                    margin_error = abs(actual_margin - pred_margin)
                    
                    evaluation_results.append({
                        'game_id': game_id,
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'actual_home_score': actual_home_score,
                        'actual_away_score': actual_away_score,
                        'pred_home_score': pred_home,
                        'pred_away_score': pred_away,
                        'home_error': home_error,
                        'away_error': away_error,
                        'total_error': total_error,
                        'margin_error': margin_error,
                        'correct_winner': correct_winner,
                        'confidence': prediction['uncertainty']['overall_confidence']
                    })
                    
                    team_errors.extend([home_error, away_error])
                    games_processed += 1
                    
                except Exception as e:
                    logging.error(f"Error evaluating game {game_id}: {e}")
                    continue
            
            if not evaluation_results:
                return {'error': 'No games successfully evaluated'}
            
            # Calculate metrics
            metrics = {
                'games_evaluated': len(evaluation_results),
                'team_score_mae': round(np.mean(team_errors), 2),
                'total_score_mae': round(np.mean([r['total_error'] for r in evaluation_results]), 2),
                'margin_mae': round(np.mean([r['margin_error'] for r in evaluation_results]), 2),
                'winner_accuracy': round(np.mean([r['correct_winner'] for r in evaluation_results]) * 100, 1),
                'avg_confidence': round(np.mean([r['confidence'] for r in evaluation_results]), 3),
                'evaluation_date': datetime.now().isoformat()
            }
            
            # Update performance tracking
            self.model_performance.update({
                'team_mae': metrics['team_score_mae'],
                'last_evaluation': datetime.now()
            })
            
            return {
                'metrics': metrics,
                'detailed_results': evaluation_results[-10:]  # Last 10 for inspection
            }
            
        except Exception as e:
            logging.error(f"Error evaluating predictions: {e}")
            return {'error': str(e)}

    def get_model_info(self) -> Dict:
        """Get information about loaded models and performance"""
        return {
            'models_loaded': self.models_loaded,
            'use_enhanced_models': self.use_models,
            'performance': self.model_performance,
            'model_types': {
                'team_models': list(self.trainer.team_models.keys()) if self.models_loaded else [],
                'player_models': {
                    'points': list(self.trainer.player_models.get('points', {}).keys()),
                    'minutes': list(self.trainer.player_models.get('minutes', {}).keys())
                } if self.models_loaded else {},
                'ensemble_weights': {
                    'team': self.trainer.team_ensemble_weights if hasattr(self.trainer, 'team_ensemble_weights') else {},
                    'player': self.trainer.player_ensemble_weights if hasattr(self.trainer, 'player_ensemble_weights') else {}
                }
            },
            'data_sources': {
                'enhanced_features': True,
                'advanced_stats': True,
                'situational_analysis': True,
                'caching_enabled': True
            }
        }

    def cleanup_data_cache(self, days_old: int = 7):
        """Clean up old cached data"""
        try:
            self.collector.cleanup_cache(days_old)
            logging.info(f"Cleaned up data cache (older than {days_old} days)")
        except Exception as e:
            logging.error(f"Error cleaning up cache: {e}")

    def predict_player_props(self, player_id: int, opponent_id: int = None, 
                           is_home: bool = True, prop_lines: Dict = None) -> Dict:
        """Predict player performance with prop betting analysis"""
        try:
            if self.models_loaded:
                # Use enhanced models for player prediction
                team_id = self._get_player_team_id(player_id)
                if not team_id:
                    return {'error': 'Could not determine player team'}
                
                pred_result = self.trainer.predict_player_performance(
                    player_id, team_id, opponent_id, is_home
                )
            else:
                # Use statistical method
                context = GameContext(1, False, is_home, 0.5, 0.5, 0.0)
                pred_result = self._statistical_player_prediction(
                    player_id, opponent_id, is_home, context
                )
            
            if not pred_result:
                return {'error': 'Could not generate player prediction'}
            
            result = {
                'prediction': {
                    'points': round(pred_result['points'], 1),
                    'points_std': round(pred_result['points_std'], 2),
                    'minutes': round(pred_result['minutes'], 1),
                    'minutes_std': round(pred_result['minutes_std'], 2),
                    'confidence': round(pred_result.get('confidence', 0.5), 3)
                },
                'prop_analysis': {}
            }
            
            # Analyze props if lines provided
            if prop_lines:
                from scipy import stats
                
                for prop_type, line in prop_lines.items():
                    if prop_type == 'points':
                        # Calculate over/under probabilities
                        z_score = (line - pred_result['points']) / pred_result['points_std']
                        over_prob = 1 - stats.norm.cdf(z_score)
                        under_prob = stats.norm.cdf(z_score)
                        
                        result['prop_analysis'][prop_type] = {
                            'line': line,
                            'over_probability': round(over_prob, 3),
                            'under_probability': round(under_prob, 3),
                            'edge': round(abs(pred_result['points'] - line), 1),
                            'recommendation': 'over' if pred_result['points'] > line else 'under',
                            'confidence': round(abs(z_score), 2)
                        }
            
            return result
            
        except Exception as e:
            logging.error(f"Error predicting player props: {e}")
            return {'error': str(e)}

    def _get_player_team_id(self, player_id: int) -> Optional[int]:
        """Get the current team ID for a player"""
        try:
            from nba_api.stats.endpoints import CommonPlayerInfo
            
            player_info = CommonPlayerInfo(
                player_id=player_id,
                headers=self.collector._get_headers()
            ).get_data_frames()[0]
            
            if not player_info.empty and 'TEAM_ID' in player_info.columns:
                return player_info['TEAM_ID'].iloc[0]
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting player team ID: {e}")
            return None
        