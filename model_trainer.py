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
import random
from data_collector import DataCollector 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.collector = DataCollector()
        self.team_scaler = StandardScaler()
        self.player_scaler = StandardScaler()
        self.team_model = None
        self.player_model = None
        self.ensemble_model = None  # Ensemble model for combining predictions
        
        # Additional position-specific player models
        self.position_models = {}
        self.position_scalers = {}
        
        # Track model performance metrics
        self.team_mae = None
        self.player_mae = None

    def prepare_team_training_data(self, n_games=1000):
        """
        Prepare team training data with expanded features and better validation
        """
        all_teams = teams.get_teams()
        training_data = []

        # Add multiple seasons for more robust training
        seasons = ['2021-22', '2022-23', '2023-24']
        
        for team in all_teams:
            team_id = team['id']
            for season in seasons:
                try:
                    games = self.collector.get_team_recent_games(team_id, n_games, season=season)
                    if games is not None and len(games) > 10:
                        for idx in range(len(games) - 10):
                            feature_games = games.iloc[idx:idx+10]
                            target_game = games.iloc[idx+10]
                            
                            # Find opponent ID
                            matchup = target_game['MATCHUP']
                            is_home = 'vs.' in matchup
                            
                            # Extract opponent abbreviation from matchup string
                            if is_home:
                                opponent_abbr = matchup.split('vs. ')[-1]
                            else:
                                opponent_abbr = matchup.split('@ ')[-1]
                            
                            # Clean up any remaining whitespace or special chars
                            opponent_abbr = opponent_abbr.strip()
                            
                            # Find opponent team ID
                            opponent_id = None
                            for opp_team in all_teams:
                                if opp_team['abbreviation'] == opponent_abbr:
                                    opponent_id = opp_team['id']
                                    break
                            
                            # Skip if opponent can't be identified
                            if opponent_id is None:
                                continue
                            
                            # Get stats with opponent information
                            stats = self.collector.get_team_stats(team_id, opponent_id)
                            if stats is None:
                                continue
                                
                            # Add home/away indicator
                            stats['HOME_GAME'] = int(is_home)
                            
                            # Add season information (more recent seasons weighted higher)
                            stats['SEASON'] = 1.0 if season == '2023-24' else (0.7 if season == '2022-23' else 0.4)
                            
                            # Skip games with missing data
                            if any(pd.isna(val) for val in stats.values()):
                                continue
                                
                            training_data.append({
                                'features': stats,
                                'target': float(target_game['PTS'])
                            })
                except Exception as e:
                    logger.error(f"Error preparing team data: {str(e)}")
                    continue

        # Handle case where no data was collected
        if not training_data:
            logger.error("No team training data collected")
            return pd.DataFrame()
            
        return pd.DataFrame(training_data)

    def prepare_player_training_data(self, n_games=200):
        """
        Prepare player training data with improved type checking and conversion
        """
        active_players = players.get_players()
        training_data = []
        
        # Process more players with better sampling
        sample_size = min(len(active_players), 450)  # Increased to cover more of the league
        player_sample = active_players[:sample_size]
        
        # Track player positions for position-specific models
        positions_data = {}
        
        for player in player_sample:
            player_id = player['id']
            try:
                games = self.collector.get_player_recent_games(player_id, n_games)
                if games is None or len(games) < 5:
                    continue
                    
                # Process minutes
                if 'MIN' in games.columns:
                    games['MIN'] = games['MIN'].apply(self.collector._convert_minutes)
                
                # Ensure all numeric columns are properly converted
                for col in games.columns:
                    if col not in ['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                                'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'VIDEO_AVAILABLE']:
                        try:
                            games[col] = pd.to_numeric(games[col], errors='coerce')
                        except:
                            # If conversion fails, set to NaN
                            games[col] = np.nan
                
                # Get player position for position-specific modeling
                position = self._get_player_position(player_id)
                
                for idx in range(len(games) - 5):
                    feature_games = games.iloc[idx:idx+5]
                    target_game = games.iloc[idx+5]
                    
                    # Check that target PTS is numeric
                    try:
                        target_points = float(target_game['PTS'])
                    except (ValueError, TypeError):
                        # Skip if target points is not numeric
                        continue
                    
                    # Include players with at least some playing time
                    min_minutes = 5  # Lowered threshold to include bench players
                    try:
                        if 'MIN' in target_game and pd.to_numeric(target_game['MIN'], errors='coerce') < min_minutes:
                            continue
                    except:
                        # If minutes conversion fails, skip this record
                        continue
                    
                    # Get player stats
                    stats = self.collector.get_player_stats(player_id)
                    if stats is None or pd.isna(stats.get('PTS_AVG')):
                        continue
                    
                    # Ensure all stats are numeric
                    for key in list(stats.keys()):
                        if key not in ['POSITION', 'INJURY_STATUS', 'INJURY_DETAILS']:
                            try:
                                stats[key] = pd.to_numeric(stats[key], errors='coerce')
                            except:
                                # If conversion fails, remove the key
                                del stats[key]
                    
                    # Add matchup information
                    stats['HOME_GAME'] = 1 if 'vs.' in str(target_game['MATCHUP']) else 0
                    
                    # Add team contextual data
                    team_id = None
                    matchup = str(target_game['MATCHUP'])
                    team_abbr = matchup.split()[0]
                    for team in teams.get_teams():
                        if team['abbreviation'] == team_abbr:
                            team_id = team['id']
                            break
                            
                    if team_id:
                        team_stats = self.collector.get_team_stats(team_id)
                        if team_stats:
                            # Add team context
                            try:
                                stats['TEAM_PTS_AVG'] = pd.to_numeric(team_stats['PTS_AVG'], errors='coerce')
                                stats['TEAM_PACE'] = pd.to_numeric(team_stats.get('PACE', 100), errors='coerce')
                            except:
                                # Use defaults if conversion fails
                                stats['TEAM_PTS_AVG'] = 100.0
                                stats['TEAM_PACE'] = 100.0
                    
                    # Add opposition strength
                    opp_team_abbr = matchup.split()[-1] if 'vs.' in matchup else matchup.split('@')[-1]
                    opp_team_id = None
                    for team in teams.get_teams():
                        if team['abbreviation'] == opp_team_abbr:
                            opp_team_id = team['id']
                            break
                            
                    if opp_team_id:
                        opp_stats = self.collector.get_team_stats(opp_team_id)
                        if opp_stats:
                            # Add opposition defensive strength
                            try:
                                stats['OPP_DEF_RATING'] = pd.to_numeric(opp_stats.get('DEF_RATING', 110), errors='coerce')
                            except:
                                stats['OPP_DEF_RATING'] = 110.0
                    
                    # Add position information
                    if position:
                        stats['POSITION'] = position
                    
                    # Check for any remaining string values in stats (except for categorical variables)
                    for key, value in list(stats.items()):
                        if key not in ['POSITION', 'INJURY_STATUS', 'INJURY_DETAILS']:
                            if isinstance(value, str):
                                try:
                                    stats[key] = float(value)
                                except:
                                    # If conversion fails, remove the key
                                    del stats[key]
                    
                    # Skip if any data is missing
                    if any(pd.isna(val) for key, val in stats.items() 
                        if key not in ['POSITION', 'INJURY_STATUS', 'INJURY_DETAILS']):
                        continue
                    
                    # Add to general training data
                    training_data.append({
                        'features': stats,
                        'target': target_points
                    })
                    
                    # Add to position-specific training data
                    if position:
                        if position not in positions_data:
                            positions_data[position] = []
                        positions_data[position].append({
                            'features': stats,
                            'target': target_points
                        })
            except Exception as e:
                logging.error(f"Error preparing player data for {player['full_name']}: {str(e)}")
                continue

        # Handle case where no data was collected
        if not training_data:
            logging.error("No player training data collected")
            return pd.DataFrame(), {}
            
        # Convert position data to DataFrames
        position_dataframes = {}
        for position, data in positions_data.items():
            if len(data) >= 100:  # Only create position models with sufficient data
                position_dataframes[position] = pd.DataFrame(data)
            
        return pd.DataFrame(training_data), position_dataframes
    
    def _get_player_position(self, player_id):
        """Get player position with better error handling"""
        try:
            # Try to get from CommonPlayerInfo
            player_info = CommonPlayerInfo(player_id=player_id, 
                                        headers=self.collector._get_headers()).get_data_frames()[0]
            
            if not player_info.empty and 'POSITION' in player_info.columns:
                position = player_info['POSITION'].iloc[0]
                
                # Check if position is a valid string
                if not isinstance(position, str):
                    return None
                    
                # Simplify to primary position (G, F, C)
                if 'G' in position:
                    return 'G'  # Guard
                elif 'F' in position:
                    return 'F'  # Forward
                elif 'C' in position:
                    return 'C'  # Center
                
            return None
        except Exception as e:
            logging.warning(f"Error getting position for player {player_id}: {str(e)}")
            return None

    def train_models(self):
        """Train team and player prediction models with improved type checking"""
        try:
            team_data = self.prepare_team_training_data()
            if len(team_data) < 100:
                logging.error(f"Insufficient team training data: {len(team_data)} samples")
                return False
                
            X_team = pd.DataFrame([d['features'] for d in team_data.to_dict('records')])
            y_team = team_data['target']
            
            # Ensure all features are numeric
            for col in X_team.columns:
                X_team[col] = pd.to_numeric(X_team[col], errors='coerce')
            
            # Handle any missing data
            X_team = X_team.fillna(X_team.mean())
            
            X_train, X_val, y_train, y_val = train_test_split(X_team, y_team, test_size=0.2, random_state=42)
            X_train_scaled = self.team_scaler.fit_transform(X_train)
            X_val_scaled = self.team_scaler.transform(X_val)
            
            # Optimized XGBoost parameters
            self.team_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mae'
            )
            
            # Train with early stopping
            self.team_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Evaluate on validation set
            val_preds = self.team_model.predict(X_val_scaled)
            self.team_mae = np.mean(np.abs(val_preds - y_val))
            logging.info(f"Team model validation MAE: {self.team_mae:.2f}")
            
            # Feature importance analysis
            importances = self.team_model.feature_importances_
            feature_importance = sorted(list(zip(X_team.columns, importances)), key=lambda x: x[1], reverse=True)
            logging.info("Team model feature importance:")
            for feat, imp in feature_importance[:10]:
                logging.info(f"  {feat}: {imp:.4f}")
        except Exception as e:
            logging.error(f"Error training team model: {str(e)}")
            return False

        # Train player model with improved approach - use our fixed function
        try:
            player_data, position_data = self.prepare_player_training_data(self, n_games=200)
            if len(player_data) < 100:  # Minimum threshold for meaningful model
                logging.error(f"Insufficient player training data: {len(player_data)} samples")
                return False
                
            X_player = pd.DataFrame([d['features'] for d in player_data.to_dict('records')])
            y_player = player_data['target']
            
            # Explicitly ensure all columns are numeric 
            # (except position, which needs to be removed before scaling)
            non_numeric_cols = []
            for col in X_player.columns:
                if col in ['POSITION', 'INJURY_STATUS', 'INJURY_DETAILS']:
                    non_numeric_cols.append(col)
                else:
                    X_player[col] = pd.to_numeric(X_player[col], errors='coerce')
            
            # Store position data before removing columns
            position_data = X_player['POSITION'] if 'POSITION' in X_player.columns else None
            
            # Remove non-numeric columns before scaling
            X_player = X_player.drop(columns=non_numeric_cols, errors='ignore')
                
            # Handle any missing data
            X_player = X_player.fillna(X_player.mean())
            
            # Check for any remaining non-numeric values (debugging)
            for col in X_player.columns:
                non_numeric_count = X_player[col].apply(lambda x: not np.issubdtype(type(x), np.number)).sum()
                if non_numeric_count > 0:
                    logging.warning(f"Column {col} has {non_numeric_count} non-numeric values")
                    # Force conversion one more time
                    X_player[col] = pd.to_numeric(X_player[col], errors='coerce')
            
            X_train, X_val, y_train, y_val = train_test_split(X_player, y_player, test_size=0.2, random_state=42)
            X_train_scaled = self.player_scaler.fit_transform(X_train)
            X_val_scaled = self.player_scaler.transform(X_val)
            
            # Optimized XGBoost parameters for player model
            self.player_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mae'
            )
            
            # Train with early stopping
            self.player_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Evaluate on validation set
            val_preds = self.player_model.predict(X_val_scaled)
            self.player_mae = np.mean(np.abs(val_preds - y_val))
            logging.info(f"Player model validation MAE: {self.player_mae:.2f}")
            
            # Feature importance analysis
            importances = self.player_model.feature_importances_
            feature_importance = sorted(list(zip(X_player.columns, importances)), key=lambda x: x[1], reverse=True)
            logging.info("Player model feature importance:")
            for feat, imp in feature_importance[:10]:
                logging.info(f"  {feat}: {imp:.4f}")
            
            # Train position-specific models if we have enough data
            for position, pos_df in position_data.items():
                try:
                    X_pos = pd.DataFrame([d['features'] for d in pos_df.to_dict('records')])
                    y_pos = pos_df['target']
                    
                    # Skip positions with too little data
                    if len(X_pos) < 100:
                        continue
                    
                    # Remove non-numeric columns
                    X_pos = X_pos.drop(columns=non_numeric_cols, errors='ignore')
                    
                    # Ensure all features are numeric
                    for col in X_pos.columns:
                        X_pos[col] = pd.to_numeric(X_pos[col], errors='coerce')
                    
                    # Handle missing data
                    X_pos = X_pos.fillna(X_pos.mean())
                    
                    X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(
                        X_pos, y_pos, test_size=0.2, random_state=42)
                    
                    # Create position-specific scaler
                    pos_scaler = StandardScaler()
                    X_train_pos_scaled = pos_scaler.fit_transform(X_train_pos)
                    X_val_pos_scaled = pos_scaler.transform(X_val_pos)
                    
                    # Create position-specific model
                    pos_model = XGBRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        eval_metric='mae'
                    )
                    
                    # Train with early stopping
                    pos_model.fit(
                        X_train_pos_scaled, y_train_pos,
                        eval_set=[(X_val_pos_scaled, y_val_pos)],
                        verbose=False
                    )
                    
                    # Save position model and scaler
                    self.position_models[position] = pos_model
                    self.position_scalers[position] = pos_scaler
                    
                    # Evaluate position model
                    pos_val_preds = pos_model.predict(X_val_pos_scaled)
                    pos_mae = np.mean(np.abs(pos_val_preds - y_val_pos))
                    logging.info(f"{position} model validation MAE: {pos_mae:.2f}")
                    
                except Exception as e:
                    logging.error(f"Error training {position} model: {str(e)}")
                    continue
        except Exception as e:
            logging.error(f"Error training player model: {str(e)}")
            return False
        
        return True

    def save_models(self, path='models/'):
        """Save all trained models for future use"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save team model components
        joblib.dump(self.team_model, f'{path}team_model.joblib')
        joblib.dump(self.team_scaler, f'{path}team_scaler.joblib')
        
        # Save player model components
        joblib.dump(self.player_model, f'{path}player_model.joblib')
        joblib.dump(self.player_scaler, f'{path}player_scaler.joblib')
        
        # Save performance metrics
        with open(f'{path}model_metrics.txt', 'w') as f:
            f.write(f"Team MAE: {self.team_mae}\n")
            f.write(f"Player MAE: {self.player_mae}\n")
        
        # Save position-specific models
        if self.position_models:
            os.makedirs(f'{path}position_models/', exist_ok=True)
            for position, model in self.position_models.items():
                joblib.dump(model, f'{path}position_models/{position}_model.joblib')
                joblib.dump(self.position_scalers[position], f'{path}position_models/{position}_scaler.joblib')

    def load_models(self, path='models/'):
        """Load all trained models"""
        try:
            if not os.path.exists(f'{path}team_model.joblib'):
                logger.warning("Model files not found")
                return False
                
            # Load team model components
            self.team_model = joblib.load(f'{path}team_model.joblib')
            self.team_scaler = joblib.load(f'{path}team_scaler.joblib')
            
            # Load player model components
            self.player_model = joblib.load(f'{path}player_model.joblib')
            self.player_scaler = joblib.load(f'{path}player_scaler.joblib')
            
            # Load position-specific models if available
            position_path = f'{path}position_models/'
            if os.path.exists(position_path):
                for pos in ['G', 'F', 'C']:
                    if os.path.exists(f'{position_path}{pos}_model.joblib'):
                        self.position_models[pos] = joblib.load(f'{position_path}{pos}_model.joblib')
                        self.position_scalers[pos] = joblib.load(f'{position_path}{pos}_scaler.joblib')
            
            # Load metrics if available
            metrics_path = f'{path}model_metrics.txt'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        self.team_mae = float(lines[0].split(': ')[1])
                        self.player_mae = float(lines[1].split(': ')[1])
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False