import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    def _ensure_feature_consistency(self, X, feature_names=None):
        """Ensure feature consistency between training and prediction"""
        try:
            if feature_names is not None:
                # Reorder columns to match training order
                missing_cols = set(feature_names) - set(X.columns)
                extra_cols = set(X.columns) - set(feature_names)
                
                # Add missing columns with default values
                for col in missing_cols:
                    X[col] = 0.0
                
                # Remove extra columns
                X = X.drop(columns=list(extra_cols), errors='ignore')
                
                # Reorder to match training order
                X = X.reindex(columns=feature_names, fill_value=0.0)
            
            return X
            
        except Exception as e:
            logging.error(f"Error ensuring feature consistency: {e}")
            return X
    
    def _get_feature_names(self, model_type):
        """Get feature names used during training"""
        try:
            selector = self.feature_selectors.get(model_type)
            if selector and hasattr(selector, 'feature_names_in_'):
                return selector.feature_names_in_[selector.get_support()]
            return None
        except:
            return None
    
    def __init__(self, collector):
        self.collector = collector
        self.team_models = {}
        self.player_models = {}
        self.position_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Model configurations
        self.team_model_configs = {
            'xgb': XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        self.player_model_configs = {
            'xgb': XGBRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=42,
                verbosity=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
        }

    def create_advanced_team_features(self, team_id, opponent_id=None, season='2023-24'):
        """Create comprehensive team features including advanced metrics"""
        try:
            # Get basic team stats
            team_stats = self.collector.get_team_stats(team_id, opponent_id)
            if not team_stats:
                return None
            
            # Get recent games for rolling statistics
            recent_games = self.collector.get_team_recent_games(team_id, n_games=20, season=season)
            if recent_games is None or len(recent_games) < 5:
                return None
            
            features = {}
            
            # Basic stats - use consistent naming with data collector
            features.update({
                'PTS_AVG': team_stats.get('pts_avg', 110),
                'FG_PCT': team_stats.get('fg_pct', 0.46),
                'FG3_PCT': team_stats.get('fg3_pct', 0.35),
                'FT_PCT': team_stats.get('ft_pct', 0.78),
                'REB': team_stats.get('reb_rate', 45),
                'AST': team_stats.get('ast_rate', 25),
                'STL': team_stats.get('stl_avg', 8),
                'BLK': team_stats.get('blk_avg', 5),
                'TOV': team_stats.get('tov_rate', 14),
                'PACE': team_stats.get('pace', 100),
                'DEF_RATING': team_stats.get('def_rating', 110),
                'HOME_GAME': 0,  # Will be set during prediction
                'REST_DAYS': 1   # Will be set during prediction
            })

            features['SEASON_WEIGHT'] = 1.0
            
            # Advanced rolling statistics (last 5, 10, 15 games)
            for window in [5, 10, 15]:
                if len(recent_games) >= window:
                    window_games = recent_games.head(window)
                    
                    # Convert to numeric
                    for col in ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']:
                        if col in window_games.columns:
                            window_games[col] = pd.to_numeric(window_games[col], errors='coerce')
                    
                    features.update({
                        f'PTS_L{window}': window_games['PTS'].mean(),
                        f'PTS_STD_L{window}': window_games['PTS'].std(),
                        f'FG_PCT_L{window}': window_games['FG_PCT'].mean(),
                        f'FG3_PCT_L{window}': window_games['FG3_PCT'].mean(),
                        f'REB_L{window}': window_games['REB'].mean(),
                        f'AST_L{window}': window_games['AST'].mean(),
                        f'TOV_L{window}': window_games['TOV'].mean(),
                    })
                    
                    # Win percentage in window
                    if 'WL' in window_games.columns:
                        features[f'WIN_PCT_L{window}'] = (window_games['WL'] == 'W').mean()
            
            # Trend analysis (linear regression slope of last 10 games)
            if len(recent_games) >= 10:
                last_10 = recent_games.head(10)
                x = np.arange(10)
                
                for stat in ['PTS', 'FG_PCT', 'REB', 'AST']:
                    if stat in last_10.columns:
                        y = pd.to_numeric(last_10[stat], errors='coerce').fillna(0)
                        if len(y) == 10:
                            slope = np.polyfit(x, y, 1)[0]
                            features[f'{stat}_TREND'] = slope
            
            # Home/Away splits
            if 'MATCHUP' in recent_games.columns:
                home_games = recent_games[recent_games['MATCHUP'].str.contains('vs.', na=False)]
                away_games = recent_games[recent_games['MATCHUP'].str.contains('@', na=False)]
                
                if not home_games.empty and not away_games.empty:
                    features['HOME_PTS_AVG'] = pd.to_numeric(home_games['PTS'], errors='coerce').mean()
                    features['AWAY_PTS_AVG'] = pd.to_numeric(away_games['PTS'], errors='coerce').mean()
                    features['HOME_AWAY_DIFF'] = features['HOME_PTS_AVG'] - features['AWAY_PTS_AVG']
            
            # Back-to-back and rest factors
            if len(recent_games) >= 2:
                # Calculate rest between recent games
                try:
                    game_dates = pd.to_datetime(recent_games['GAME_DATE'])
                    recent_rest = [(game_dates.iloc[i] - game_dates.iloc[i+1]).days 
                                  for i in range(min(3, len(game_dates)-1))]
                    features['AVG_REST_L3'] = np.mean(recent_rest) if recent_rest else 1
                    features['B2B_GAMES_L10'] = sum(1 for r in recent_rest[:9] if r == 1)
                except:
                    features['AVG_REST_L3'] = 1
                    features['B2B_GAMES_L10'] = 0
            
            # Strength of schedule (opponent quality)
            features['OPP_AVG_PTS'] = team_stats.get('OPP_PTS_ALLOWED', 110)
            features['OPP_DEF_RATING'] = team_stats.get('OPP_DEF_RATING', 110)
            
            # Injury impact
            features['INJURY_IMPACT'] = team_stats.get('INJURY_IMPACT', 0)
            features['KEY_PLAYERS_INJURED'] = team_stats.get('KEY_PLAYERS_INJURED', 0)
            
            # Season timing effects
            features['SEASON_PROGRESS'] = self._calculate_season_progress()
            
            return features
            
        except Exception as e:
            logging.error(f"Error creating team features: {e}")
            return None

    def create_advanced_player_features(self, player_id, team_id, opponent_id=None):
        """Create comprehensive player features with contextual information"""
        try:
            # Get player stats and recent games
            player_stats = self.collector.get_player_stats(player_id)
            recent_games = self.collector.get_player_recent_games(player_id, n_games=25)
            
            if not player_stats or recent_games is None or len(recent_games) < 3:
                return None
            
            features = {}
            
            # Basic player stats - use consistent naming
            features.update({
                'PTS_AVG': player_stats.get('pts_avg', 0),
                'MIN_AVG': player_stats.get('minutes_avg', 0),
                'FG_PCT': player_stats.get('shooting', {}).get('fg_pct', 0),
                'FG3_PCT': player_stats.get('shooting', {}).get('fg3_pct', 0),
                'FT_PCT': player_stats.get('shooting', {}).get('ft_pct', 0),
                'REB': player_stats.get('per_36', {}).get('reb', 0),
                'AST': player_stats.get('per_36', {}).get('ast', 0),
                'USAGE': player_stats.get('usage_estimate', 0),
                'CONSISTENCY': player_stats.get('consistency', {}).get('pts_cv', 0.5),
                'GAMES_PLAYED': player_stats.get('games_played', 0)
            })

            features['SEASON_WEIGHT'] = 1.0
            
            # Convert minutes to numeric
            recent_games['MIN'] = recent_games['MIN'].apply(self.collector._convert_minutes)
            
            # Rolling averages and trends
            for window in [3, 5, 10, 15]:
                if len(recent_games) >= window:
                    window_games = recent_games.head(window)
                    
                    # Ensure numeric columns
                    for col in ['PTS', 'MIN', 'FG_PCT', 'REB', 'AST']:
                        if col in window_games.columns:
                            window_games[col] = pd.to_numeric(window_games[col], errors='coerce')
                    
                    features.update({
                        f'PTS_L{window}': window_games['PTS'].mean(),
                        f'PTS_STD_L{window}': window_games['PTS'].std(),
                        f'MIN_L{window}': window_games['MIN'].mean(),
                        f'MIN_STD_L{window}': window_games['MIN'].std(),
                        f'PTS_PER_MIN_L{window}': (window_games['PTS'] / window_games['MIN'].replace(0, 1)).mean()
                    })
            
            # Recent form and trends
            if len(recent_games) >= 5:
                last_5 = recent_games.head(5)
                season_avg = recent_games['PTS'].mean()
                recent_avg = last_5['PTS'].mean()
                
                features['HOT_COLD_FACTOR'] = (recent_avg - season_avg) / max(1, season_avg)
                features['MINUTES_TREND'] = (last_5['MIN'].mean() - recent_games['MIN'].mean()) / max(1, recent_games['MIN'].mean())
            
            # Home/Away performance splits
            if 'MATCHUP' in recent_games.columns:
                home_games = recent_games[recent_games['MATCHUP'].str.contains('vs.', na=False)]
                away_games = recent_games[recent_games['MATCHUP'].str.contains('@', na=False)]
                
                if not home_games.empty:
                    features['HOME_PTS_AVG'] = home_games['PTS'].mean()
                    features['HOME_MIN_AVG'] = home_games['MIN'].mean()
                if not away_games.empty:
                    features['AWAY_PTS_AVG'] = away_games['PTS'].mean()
                    features['AWAY_MIN_AVG'] = away_games['MIN'].mean()
                
                if not home_games.empty and not away_games.empty:
                    features['HOME_AWAY_PTS_DIFF'] = features['HOME_PTS_AVG'] - features['AWAY_PTS_AVG']
            
            # Rest and fatigue factors
            try:
                if len(recent_games) >= 2:
                    game_dates = pd.to_datetime(recent_games['GAME_DATE'])
                    rest_days = [(game_dates.iloc[i] - game_dates.iloc[i+1]).days 
                                for i in range(min(5, len(game_dates)-1))]
                    features['AVG_REST'] = np.mean(rest_days) if rest_days else 1
                    features['LAST_REST'] = rest_days[0] if rest_days else 1
            except:
                features['AVG_REST'] = 1
                features['LAST_REST'] = 1
            
            # Team context features
            team_stats = self.collector.get_team_stats(team_id)
            if team_stats:
                features['TEAM_PACE'] = team_stats.get('pace', 100)
                features['TEAM_PTS_AVG'] = team_stats.get('pts_avg', 110)
                features['TEAM_OFF_RATING'] = team_stats.get('off_rating', 110)
            
            # Opponent context
            if opponent_id:
                opp_stats = self.collector.get_team_stats(opponent_id)
                if opp_stats:
                    features['OPP_DEF_RATING'] = opp_stats.get('def_rating', 110)
                    features['OPP_PACE'] = opp_stats.get('pace', 100)
                    features['OPP_PTS_ALLOWED'] = opp_stats.get('pts_avg', 110)  # Approximate
            
            # Position-specific features (if available)
            if 'POSITION' in player_stats and player_stats['POSITION']:
                pos = player_stats['POSITION']
                features[f'POS_{pos}'] = 1
                for other_pos in ['G', 'F', 'C']:
                    if other_pos != pos:
                        features[f'POS_{other_pos}'] = 0
            
            # Advanced efficiency metrics
            if features['MIN_AVG'] > 0:
                features['PTS_PER_36'] = features['PTS_AVG'] * 36 / features['MIN_AVG']
                features['EFFICIENCY'] = features['PTS_AVG'] / features['MIN_AVG'] if features['MIN_AVG'] > 0 else 0
            
            # Injury status and minutes reduction flags
            features['REDUCED_MINUTES'] = player_stats.get('REDUCED_MINUTES', False)
            features['MINUTES_REDUCTION'] = player_stats.get('MINUTES_REDUCTION', 0)
            
            return features
            
        except Exception as e:
            logging.error(f"Error creating player features for {player_id}: {e}")
            return None

    def prepare_training_data(self):
        """Prepare comprehensive training data with advanced features"""
        from nba_api.stats.static import teams, players
        
        team_data = []
        player_data = []
        
        # Get all teams
        all_teams = teams.get_teams()
        
        # Use multiple seasons for better training data
        seasons = ['2021-22', '2022-23', '2023-24']
        
        logging.info("Preparing enhanced team training data...")
        
        # Team data preparation
        for season in seasons:
            season_weight = 1.0 if season == '2023-24' else (0.8 if season == '2022-23' else 0.6)
            
            for team in all_teams[:15]:  # Limit for initial testing
                team_id = team['id']
                
                try:
                    games = self.collector.get_team_recent_games(team_id, n_games=82, season=season)
                    if games is None or len(games) < 20:
                        continue
                    
                    # Create features for each game using previous games as context
                    for i in range(10, min(70, len(games))):  # Use games 10-70 for training
                        # Use games before index i for feature creation
                        historical_games = games.iloc[i:]
                        target_game = games.iloc[i]
                        
                        # Create features based on games before target game
                        features = self.create_advanced_team_features(team_id, season=season)
                        if not features:
                            continue
                        
                        # Add target
                        target_points = float(target_game['PTS'])
                        
                        # Add season weight
                        features['SEASON_WEIGHT'] = season_weight
                        
                        team_data.append({
                            'features': features,
                            'target': target_points,
                            'weight': season_weight
                        })
                        
                except Exception as e:
                    logging.error(f"Error processing team {team['full_name']}: {e}")
                    continue
        
        logging.info(f"Collected {len(team_data)} team training samples")
        
        # Player data preparation
        logging.info("Preparing enhanced player training data...")
        
        # Get sample of active players
        all_players = players.get_players()
        sample_players = all_players[:200]  # Limit for testing
        
        for player in sample_players:
            player_id = player['id']
            
            try:
                recent_games = self.collector.get_player_recent_games(player_id, n_games=50)
                if recent_games is None or len(recent_games) < 10:
                    continue
                
                # Create training samples from recent games
                for i in range(5, min(40, len(recent_games))):
                    features = self.create_advanced_player_features(player_id, None)
                    if not features:
                        continue
                    
                    target_game = recent_games.iloc[i]
                    target_points = float(target_game['PTS'])
                    target_minutes = self.collector._convert_minutes(target_game['MIN'])
                    
                    # Skip very low minute games for training
                    if target_minutes < 5:
                        continue
                    
                    # Create separate samples for points and minutes prediction
                    player_data.append({
                        'features': features,
                        'target_points': target_points,
                        'target_minutes': target_minutes,
                        'player_id': player_id
                    })
                    
            except Exception as e:
                logging.error(f"Error processing player {player['full_name']}: {e}")
                continue
        
        logging.info(f"Collected {len(player_data)} player training samples")
        
        return team_data, player_data

    def train_ensemble_models(self):
        """Train ensemble models with advanced features and cross-validation"""
        
        # Prepare training data
        team_data, player_data = self.prepare_training_data()
        
        if len(team_data) < 100:
            logging.error("Insufficient team training data")
            return False
        
        if len(player_data) < 60:
            logging.error("Insufficient player training data")
            return False
        
        # Train team models
        logging.info("Training team ensemble models...")
        success = self._train_team_ensemble(team_data)
        if not success:
            return False
        
        # Train player models
        logging.info("Training player ensemble models...")
        success = self._train_player_ensemble(player_data)
        if not success:
            return False
        
        return True

    def _train_team_ensemble(self, team_data):
        """Train ensemble of team models"""
        try:
            # Prepare data
            features_list = [sample['features'] for sample in team_data]
            targets = [sample['target'] for sample in team_data]
            weights = [sample['weight'] for sample in team_data]
            
            # Create feature dataframe
            X = pd.DataFrame(features_list)
            y = np.array(targets)
            sample_weights = np.array(weights)
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Feature selection
            selector = SelectKBest(score_func=f_regression, k=min(50, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            
            # Store feature selector
            self.feature_selectors['team'] = selector
            selected_features = X.columns[selector.get_support()]
            
            logging.info(f"Selected {len(selected_features)} team features")
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            self.scalers['team'] = scaler
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train multiple models
            team_models = {}
            model_scores = {}
            
            for name, model in self.team_model_configs.items():
                logging.info(f"Training team model: {name}")
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y, cv=tscv, 
                    scoring='neg_mean_absolute_error',
                    fit_params={'sample_weight': sample_weights} if name in ['xgb', 'lgb'] else {}
                )
                
                model_scores[name] = -cv_scores.mean()
                logging.info(f"{name} CV MAE: {model_scores[name]:.2f}")
                
                # Fit on full data
                if name in ['xgb', 'lgb']:
                    model.fit(X_scaled, y, sample_weight=sample_weights)
                else:
                    model.fit(X_scaled, y)
                    
                team_models[name] = model
            
            self.team_models = team_models
            self.team_model_scores = model_scores
            
            # Create ensemble weights based on performance
            total_inverse_error = sum(1/score for score in model_scores.values())
            self.team_ensemble_weights = {
                name: (1/score) / total_inverse_error 
                for name, score in model_scores.items()
            }
            
            logging.info(f"Team ensemble weights: {self.team_ensemble_weights}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training team ensemble: {e}")
            return False

    def _train_player_ensemble(self, player_data):
        """Train ensemble of player models for points and minutes"""
        try:
            # Prepare data
            features_list = [sample['features'] for sample in player_data]
            points_targets = [sample['target_points'] for sample in player_data]
            minutes_targets = [sample['target_minutes'] for sample in player_data]
            
            # Create feature dataframe
            X = pd.DataFrame(features_list)
            y_points = np.array(points_targets)
            y_minutes = np.array(minutes_targets)
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Feature selection for points prediction
            points_selector = SelectKBest(score_func=f_regression, k=min(30, len(X.columns)))
            X_points_selected = points_selector.fit_transform(X, y_points)
            self.feature_selectors['player_points'] = points_selector
            
            # Feature selection for minutes prediction
            minutes_selector = SelectKBest(score_func=f_regression, k=min(30, len(X.columns)))
            X_minutes_selected = minutes_selector.fit_transform(X, y_minutes)
            self.feature_selectors['player_minutes'] = minutes_selector
            
            # Scale features
            points_scaler = RobustScaler()
            X_points_scaled = points_scaler.fit_transform(X_points_selected)
            self.scalers['player_points'] = points_scaler
            
            minutes_scaler = RobustScaler()
            X_minutes_scaled = minutes_scaler.fit_transform(X_minutes_selected)
            self.scalers['player_minutes'] = minutes_scaler
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train points prediction models
            logging.info("Training player points models...")
            points_models = {}
            points_scores = {}
            
            for name, model in self.player_model_configs.items():
                cv_scores = cross_val_score(
                    model, X_points_scaled, y_points, cv=tscv,
                    scoring='neg_mean_absolute_error'
                )
                points_scores[name] = -cv_scores.mean()
                logging.info(f"Points {name} CV MAE: {points_scores[name]:.2f}")
                
                model.fit(X_points_scaled, y_points)
                points_models[name] = model
            
            self.player_models['points'] = points_models
            
            # Train minutes prediction models
            logging.info("Training player minutes models...")
            minutes_models = {}
            minutes_scores = {}
            
            for name, model in self.player_model_configs.items():
                cv_scores = cross_val_score(
                    model, X_minutes_scaled, y_minutes, cv=tscv,
                    scoring='neg_mean_absolute_error'
                )
                minutes_scores[name] = -cv_scores.mean()
                logging.info(f"Minutes {name} CV MAE: {minutes_scores[name]:.2f}")
                
                model.fit(X_minutes_scaled, y_minutes)
                minutes_models[name] = model
            
            self.player_models['minutes'] = minutes_models
            
            # Create ensemble weights
            points_total_inverse = sum(1/score for score in points_scores.values())
            minutes_total_inverse = sum(1/score for score in minutes_scores.values())
            self.player_ensemble_weights = {
                'points': {name: (1/score) / points_total_inverse for name, score in points_scores.items()},
                'minutes': {name: (1/score) / minutes_total_inverse for name, score in minutes_scores.items()}
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Error training player ensemble: {e}")
            return False

    def predict_team_score(self, team_id, opponent_id=None, is_home=True):
        """Make ensemble prediction for team score"""
        try:
            # Create features
            features = self.create_advanced_team_features(team_id, opponent_id)
            if not features:
                return None
            
            features['HOME_GAME'] = 1 if is_home else 0
            
            # Convert to dataframe and prepare
            X = pd.DataFrame([features])
            X = X.fillna(X.mean())
            
            # Ensure feature consistency
            feature_names = self._get_feature_names('team')
            X = self._ensure_feature_consistency(X, feature_names)
            
            # Select features and scale
            selector = self.feature_selectors.get('team')
            scaler = self.scalers.get('team')
            
            if not selector or not scaler:
                return None
            
            try:
                X_selected = selector.transform(X)
                X_scaled = scaler.transform(X_selected)
            except Exception as e:
                logging.error(f"Error transforming features: {e}")
                return None
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.team_models.items():
                pred = model.predict(X_scaled)[0]
                predictions[name] = pred
            
            # Ensemble prediction
            ensemble_pred = sum(
                pred * self.team_ensemble_weights[name] 
                for name, pred in predictions.items()
            )
            
            # Calculate prediction uncertainty
            pred_std = np.std(list(predictions.values()))
            
            return {
                'prediction': ensemble_pred,
                'std': pred_std,
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logging.error(f"Error predicting team score: {e}")
            return None

    def predict_player_performance(self, player_id, team_id, opponent_id=None, is_home=True):
        """Make ensemble prediction for player performance"""
        try:
            # Create features
            features = self.create_advanced_player_features(player_id, team_id, opponent_id)
            if not features:
                return None
            
            # Add game context
            features['HOME_GAME'] = 1 if is_home else 0
            
            # Convert to dataframe
            X = pd.DataFrame([features])
            X = X.fillna(X.mean())
            
            # Predict points
            points_selector = self.feature_selectors.get('player_points')
            points_scaler = self.scalers.get('player_points')
            
            if points_selector and points_scaler:
                # Ensure feature consistency for points
                points_feature_names = self._get_feature_names('player_points')
                X_points_consistent = self._ensure_feature_consistency(X, points_feature_names)
                
                try:
                    X_points = points_selector.transform(X_points_consistent)
                    X_points_scaled = points_scaler.transform(X_points)
                except Exception as e:
                    logging.error(f"Error transforming points features: {e}")
                    ensemble_points = features.get('PTS_AVG', 0)
                    points_std = 5.0
                    X_points = None
                    X_points_scaled = None
                
                points_predictions = {}
                for name, model in self.player_models['points'].items():
                    pred = model.predict(X_points_scaled)[0]
                    points_predictions[name] = pred
                
                ensemble_points = sum(
                    pred * self.player_ensemble_weights['points'][name]
                    for name, pred in points_predictions.items()
                )
                points_std = np.std(list(points_predictions.values()))
            else:
                ensemble_points = features.get('PTS_AVG', 0)
                points_std = 5.0
            
            # Predict minutes
            minutes_selector = self.feature_selectors.get('player_minutes')
            minutes_scaler = self.scalers.get('player_minutes')
            
            if minutes_selector and minutes_scaler:
                # Ensure feature consistency for minutes
                minutes_feature_names = self._get_feature_names('player_minutes')
                X_minutes_consistent = self._ensure_feature_consistency(X, minutes_feature_names)
                
                try:
                    X_minutes = minutes_selector.transform(X_minutes_consistent)
                    X_minutes_scaled = minutes_scaler.transform(X_minutes)
                except Exception as e:
                    logging.error(f"Error transforming minutes features: {e}")
                    ensemble_minutes = features.get('MIN_AVG', 20)
                    minutes_std = 5.0
                    X_minutes = None
                    X_minutes_scaled = None
                
                minutes_predictions = {}
                for name, model in self.player_models['minutes'].items():
                    pred = model.predict(X_minutes_scaled)[0]
                    minutes_predictions[name] = pred
                
                ensemble_minutes = sum(
                    pred * self.player_ensemble_weights['minutes'][name]
                    for name, pred in minutes_predictions.items()
                )
                minutes_std = np.std(list(minutes_predictions.values()))
            else:
                ensemble_minutes = features.get('MIN_AVG', 20)
                minutes_std = 5.0
            
            return {
                'points': max(0, ensemble_points),
                'points_std': max(2.0, points_std),
                'minutes': max(0, ensemble_minutes),
                'minutes_std': max(2.0, minutes_std),
                'individual_predictions': {
                    'points': points_predictions if 'points_predictions' in locals() else {},
                    'minutes': minutes_predictions if 'minutes_predictions' in locals() else {}
                }
            }
            
        except Exception as e:
            logging.error(f"Error predicting player performance: {e}")
            return None

    def _calculate_season_progress(self):
        """Calculate how far into the season we are (0-1)"""
        try:
            # Approximate season progress based on current date
            # NBA season typically runs October-April
            now = datetime.now()
            
            if now.month >= 10:  # October-December
                season_start = datetime(now.year, 10, 1)
                season_end = datetime(now.year + 1, 4, 30)
            else:  # January-April
                season_start = datetime(now.year - 1, 10, 1)
                season_end = datetime(now.year, 4, 30)
            
            total_season_days = (season_end - season_start).days
            current_season_days = (now - season_start).days
            
            return max(0, min(1, current_season_days / total_season_days))
            
        except:
            return 0.5  # Default to mid-season

    def save_models(self, path='enhanced_models/'):
        """Save all enhanced models and preprocessing objects"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save team models
        for name, model in self.team_models.items():
            joblib.dump(model, f'{path}team_{name}_model.joblib')
        
        # Save player models
        for target_type, models in self.player_models.items():
            for name, model in models.items():
                joblib.dump(model, f'{path}player_{target_type}_{name}_model.joblib')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{path}scaler_{name}.joblib')
        
        # Save feature selectors
        for name, selector in self.feature_selectors.items():
            joblib.dump(selector, f'{path}selector_{name}.joblib')
        
        # Save ensemble weights
        joblib.dump(self.team_ensemble_weights, f'{path}team_ensemble_weights.joblib')
        joblib.dump(self.player_ensemble_weights, f'{path}player_ensemble_weights.joblib')
        
        logging.info("Enhanced models saved successfully")

    def load_models(self, path='enhanced_models/'):
        """Load all enhanced models and preprocessing objects"""
        import os
        
        try:
            # Load team models
            self.team_models = {}
            for name in self.team_model_configs.keys():
                model_path = f'{path}team_{name}_model.joblib'
                if os.path.exists(model_path):
                    self.team_models[name] = joblib.load(model_path)
            
            # Load player models
            self.player_models = {'points': {}, 'minutes': {}}
            for target_type in ['points', 'minutes']:
                for name in self.player_model_configs.keys():
                    model_path = f'{path}player_{target_type}_{name}_model.joblib'
                    if os.path.exists(model_path):
                        self.player_models[target_type][name] = joblib.load(model_path)
            
            # Load scalers
            scaler_files = [f for f in os.listdir(path) if f.startswith('scaler_')]
            for scaler_file in scaler_files:
                name = scaler_file.replace('scaler_', '').replace('.joblib', '')
                self.scalers[name] = joblib.load(f'{path}{scaler_file}')
            
            # Load feature selectors
            selector_files = [f for f in os.listdir(path) if f.startswith('selector_')]
            for selector_file in selector_files:
                name = selector_file.replace('selector_', '').replace('.joblib', '')
                self.feature_selectors[name] = joblib.load(f'{path}{selector_file}')
            
            # Load ensemble weights
            weights_path = f'{path}team_ensemble_weights.joblib'
            if os.path.exists(weights_path):
                self.team_ensemble_weights = joblib.load(weights_path)
            
            weights_path = f'{path}player_ensemble_weights.joblib'
            if os.path.exists(weights_path):
                self.player_ensemble_weights = joblib.load(weights_path)
            
            logging.info("Enhanced models loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading enhanced models: {e}")
            return False