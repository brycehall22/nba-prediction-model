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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache timeout
        self.retries = 5  # Increased retries for API stability
        self.last_request_time = 0  # Track time of last request for better rate limiting
        
        # Initialize injury tracking
        self.injury_data = {}
        self.injury_last_updated = datetime.now() - timedelta(days=1)  # Force initial update
        
        # Configure user agent rotation to avoid API blocks
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]

    def _rate_limit(self):
        """
        Improved rate limiting with jitter to avoid detection patterns
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Base delay is 1.5s, add jitter of 0-0.5s to avoid regular patterns
        min_delay = 1.5
        if elapsed < min_delay:
            jitter = random.uniform(0, 0.5)
            sleep_time = min_delay - elapsed + jitter
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()

    def _get_headers(self):
        """
        Rotate user agents to avoid API blocks
        """
        return {
            'User-Agent': random.choice(self.user_agents),
            'Referer': 'https://stats.nba.com/',
            'Accept-Language': 'en-US,en;q=0.9'
        }

    def get_team_players(self, team_id):
        """Get active roster for a team"""
        try:
            return self.collector.get_team_players(team_id)
        except Exception as e:
            logger.error(f"Error getting team players: {str(e)}")
            return []
            
    def get_league_info(self):
        """Get league-wide information and statistics"""
        try:
            # Get data for several teams to calculate league averages
            all_teams = teams.get_teams()
            sample_teams = all_teams[:10]  # Use 10 teams as a sample
            
            team_stats = []
            for team in sample_teams:
                stats = self.collector.get_team_stats(team['id'])
                if stats:
                    team_stats.append(stats)
            
            if not team_stats:
                return {
                    'avg_team_points': self.league_avg_points,
                    'avg_pace': 100.0,
                    'teams_count': len(all_teams)
                }
            
            # Calculate league averages
            avg_points = sum(s.get('PTS_AVG', 0) for s in team_stats) / len(team_stats)
            avg_pace = sum(s.get('PACE', 100) for s in team_stats) / len(team_stats)
            
            # Update instance variables for fallback predictions
            self.league_avg_points = avg_points
            
            return {
                'avg_team_points': round(avg_points, 1),
                'avg_pace': round(avg_pace, 1),
                'teams_count': len(all_teams)
            }
        except Exception as e:
            logger.error(f"Error getting league info: {str(e)}")
            return None
            
    def get_injury_report(self):
        """Get a league-wide injury report"""
        try:
            # Update injury data
            self.collector.update_injury_data(force=True)
            
            if not self.collector.injury_data:
                return {'error': 'No injury data available'}
                
            # Format injury data by team
            all_teams = teams.get_teams()
            team_injuries = {}
            
            # First get all team players
            team_players = {}
            for team in all_teams:
                team_id = team['id']
                players_list = self.collector.get_team_players(team_id)
                if players_list:
                    team_players[team_id] = {p['full_name']: p for p in players_list}
            
            # Then categorize injuries by team
            for player_name, injury in self.collector.injury_data.items():
                # Find player's team
                player_team = None
                team_id = None
                
                for tid, players in team_players.items():
                    if player_name in players:
                        player_team = next((t['full_name'] for t in all_teams if t['id'] == tid), None)
                        team_id = tid
                        break
                
                if player_team:
                    if player_team not in team_injuries:
                        team_injuries[player_team] = {
                            'team_id': team_id,
                            'injured_players': []
                        }
                        
                    team_injuries[player_team]['injured_players'].append({
                        'name': player_name,
                        'status': injury['status'],
                        'details': injury['details']
                    })
            
            # Calculate impact for each team
            for team_name, data in team_injuries.items():
                # Get total minutes for injured players if available
                team_id = data['team_id']
                if team_id in team_players:
                    players = team_players[team_id]
                    
                    total_injured_minutes = 0
                    key_players_out = 0
                    
                    for injured in data['injured_players']:
                        player_name = injured['name']
                        if player_name in players:
                            minutes = players[player_name].get('avg_minutes', 0)
                            total_injured_minutes += minutes
                            
                            if minutes >= 25:  # Key player threshold
                                key_players_out += 1
                                
                    # Calculate impact score
                    data['total_injured_minutes'] = round(total_injured_minutes, 1)
                    data['key_players_out'] = key_players_out
                    data['impact_score'] = round(min(3.0, total_injured_minutes / 48), 2)  # Cap at 3.0
            
            return {
                'updated': self.collector.injury_last_updated.strftime('%Y-%m-%d %H:%M:%S'),
                'teams': team_injuries
            }
        except Exception as e:
            logger.error(f"Error getting injury report: {str(e)}")
            return {'error': str(e)}
            
    def evaluate_model_performance(self):
        """Evaluate model performance on recent games"""
        try:
            if not self.models_loaded:
                return {'error': 'Models not loaded'}
                
            # Get recent completed games
            from nba_api.stats.endpoints import LeagueGameLog
            recent_games = LeagueGameLog(
                season='2023-24',
                season_type_all_star='Regular Season',
                date_from_nullable=None,  # Last 10 days by default
                date_to_nullable=None,
                headers=self.collector._get_headers()
            ).get_data_frames()[0]
            
            if recent_games.empty:
                return {'error': 'No recent games found'}
                
            # Limit to last 20 games
            recent_games = recent_games.head(20)
            
            # Evaluate predictions
            eval_results = []
            
            for _, game in recent_games.iterrows():
                try:
                    home_team_id = game['HOME_TEAM_ID']
                    away_team_id = game['VISITOR_TEAM_ID']
                    actual_home_score = game['PTS_home']
                    actual_away_score = game['PTS_away']
                    
                    # Generate prediction
                    prediction = self.predict_game(home_team_id, away_team_id)
                    
                    if 'error' in prediction and prediction['error']:
                        continue
                        
                    pred_home_score = prediction['home_score']
                    pred_away_score = prediction['away_score']
                    
                    # Calculate errors
                    home_error = abs(pred_home_score - actual_home_score)
                    away_error = abs(pred_away_score - actual_away_score)
                    total_error = home_error + away_error
                    
                    # Calculate if winner was predicted correctly
                    actual_winner = 'home' if actual_home_score > actual_away_score else 'away'
                    pred_winner = 'home' if pred_home_score > pred_away_score else 'away'
                    correct_winner = actual_winner == pred_winner
                    
                    eval_results.append({
                        'game_id': game['GAME_ID'],
                        'date': game['GAME_DATE'],
                        'matchup': f"{game['VISITOR_TEAM_ABBREVIATION']} @ {game['HOME_TEAM_ABBREVIATION']}",
                        'actual_score': f"{actual_away_score}-{actual_home_score}",
                        'predicted_score': f"{pred_away_score}-{pred_home_score}",
                        'home_error': home_error,
                        'away_error': away_error,
                        'total_error': total_error,
                        'correct_winner': correct_winner
                    })
                except Exception as e:
                    logger.error(f"Error evaluating game: {str(e)}")
                    continue
            
            # Calculate overall metrics
            if not eval_results:
                return {'error': 'No evaluations completed'}
                
            avg_home_error = sum(r['home_error'] for r in eval_results) / len(eval_results)
            avg_away_error = sum(r['away_error'] for r in eval_results) / len(eval_results)
            avg_total_error = sum(r['total_error'] for r in eval_results) / len(eval_results)
            winner_accuracy = sum(1 for r in eval_results if r['correct_winner']) / len(eval_results)
            
            return {
                'games_evaluated': len(eval_results),
                'avg_home_error': round(avg_home_error, 1),
                'avg_away_error': round(avg_away_error, 1),
                'avg_total_error': round(avg_total_error, 1),
                'winner_accuracy': round(winner_accuracy * 100, 1),
                'results': eval_results
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {'error': str(e)}
    def get_team_recent_games(self, team_id, n_games=10, season=None):
        """
        Get recent games for a team with optional season filter and improved error handling
        """
        cache_key = f"team_{team_id}_{n_games}_{season}"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return data

        try:
            self._rate_limit()
            
            # Add retry logic with exponential backoff
            for attempt in range(self.retries):
                try:
                    if season:
                        # Use SeasonYear parameter if season is provided
                        games = TeamGameLogs(team_id_nullable=team_id, 
                                           season_nullable=season,
                                           headers=self._get_headers()).get_data_frames()[0]
                    else:
                        games = TeamGameLogs(team_id_nullable=team_id,
                                          headers=self._get_headers()).get_data_frames()[0]
                    
                    if games.empty:
                        logger.warning(f"No games found for team ID {team_id}")
                        return None
                    
                    break  # Successful API call, exit retry loop
                except Exception as e:
                    if attempt == self.retries - 1:
                        logger.error(f"Failed to get team games after {self.retries} attempts: {str(e)}")
                        
                        # Fallback to LeagueGameLog if TeamGameLogs fails
                        try:
                            # Use LeagueGameLog as alternative data source
                            if season:
                                league_games = LeagueGameLog(season=season, 
                                                          headers=self._get_headers()).get_data_frames()[0]
                            else:
                                league_games = LeagueGameLog(headers=self._get_headers()).get_data_frames()[0]
                                
                            # Filter for this team
                            team_info = next((t for t in teams.get_teams() if t['id'] == team_id), None)
                            if team_info:
                                team_abbr = team_info['abbreviation']
                                home_games = league_games[league_games['HOME_TEAM_ABBREVIATION'] == team_abbr]
                                away_games = league_games[league_games['VISITOR_TEAM_ABBREVIATION'] == team_abbr]
                                
                                if not home_games.empty or not away_games.empty:
                                    # Process games to match expected format
                                    # This is simplified and would need to be expanded
                                    games = pd.concat([home_games, away_games]).head(n_games)
                                    break
                        except:
                            return None
                            
                    # Exponential backoff: wait longer after each retry
                    time.sleep(2 ** attempt)  
            
            # Process the data
            recent_games = games.head(n_games)
            
            # Ensure numeric columns are properly converted
            numeric_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']
            for col in numeric_cols:
                if col in recent_games.columns:
                    recent_games[col] = pd.to_numeric(recent_games[col], errors='coerce').fillna(0)
            
            # Add game pace if not present (estimated from possessions)
            if 'PACE' not in recent_games.columns:
                # Estimate pace from available stats (FGA + TOV + 0.44*FTA - OREB)
                if all(col in recent_games.columns for col in ['FGA', 'TOV', 'FTA', 'OREB']):
                    recent_games['PACE'] = recent_games['FGA'] + recent_games['TOV'] + 0.44*recent_games['FTA'] - recent_games['OREB']
                else:
                    # Use league average if stats not available
                    recent_games['PACE'] = 100
            
            # Calculate defensive rating if not present
            if 'DEF_RATING' not in recent_games.columns:
                # Simple estimate based on opponent points and estimated possessions
                if 'OPP_PTS' in recent_games.columns and 'PACE' in recent_games.columns:
                    recent_games['DEF_RATING'] = recent_games['OPP_PTS'] * 100 / recent_games['PACE']
                else:
                    # Use league average if stats not available
                    recent_games['DEF_RATING'] = 110
            
            self.cache[cache_key] = (datetime.now(), recent_games)
            return recent_games
        except Exception as e:
            logger.error(f"Error getting team games: {str(e)}")
            return None

    def get_player_recent_games(self, player_id, n_games=10):
        """
        Get recent games for a player with improved error handling and data validation
        """
        cache_key = f"player_{player_id}_{n_games}"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return data

        try:
            self._rate_limit()
            
            # Progressive retry strategy with exponential backoff
            for attempt in range(self.retries):
                try:
                    games = PlayerGameLogs(player_id_nullable=player_id, 
                                        headers=self._get_headers()).get_data_frames()[0]
                    break
                except Exception as e:
                    # Track specific error types for better handling
                    error_msg = str(e).lower()
                    
                    # If it's a 429 (too many requests) error, wait longer
                    if "429" in error_msg or "too many requests" in error_msg:
                        wait_time = 5 * (2 ** attempt)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry.")
                        time.sleep(wait_time)
                    elif attempt == self.retries - 1:
                        logger.error(f"Failed to get player games: {str(e)}")
                        
                        # For inactive players, create a minimal data structure
                        try:
                            player_info = CommonPlayerInfo(player_id=player_id, 
                                                         headers=self._get_headers()).get_data_frames()[0]
                            if not player_info.empty and 'ROSTERSTATUS' in player_info.columns:
                                status = player_info['ROSTERSTATUS'].iloc[0]
                                if status == 'Inactive':
                                    logger.info(f"Player ID {player_id} is inactive, returning minimal data")
                                    # Create a minimal DataFrame for inactive players
                                    return pd.DataFrame({
                                        'PLAYER_ID': [player_id],
                                        'MIN': [0],
                                        'PTS': [0],
                                        'INACTIVE': [True]
                                    })
                        except:
                            pass
                            
                        return None
                    else:
                        # For other errors, use standard backoff
                        time.sleep(2 ** attempt)
                    
            if games.empty:
                logger.warning(f"No games found for player ID {player_id}")
                return None
                
            # More lenient with data requirements - include players with at least 1 game
            if len(games) < 1:
                logger.warning(f"Not enough games for player ID {player_id}")
                return None
                
            # Take at most n_games, but be happy with whatever we have
            recent_games = games.head(min(n_games, len(games)))
            
            # Properly handle minute conversions
            if 'MIN' in recent_games.columns:
                recent_games['MIN'] = recent_games['MIN'].apply(self._convert_minutes)
                
            # Additional data cleaning for player games
            numeric_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
            for col in numeric_cols:
                if col in recent_games.columns:
                    recent_games[col] = pd.to_numeric(recent_games[col], errors='coerce').fillna(0)
            
            # Add "did not play" flag
            recent_games['DNP'] = (recent_games['MIN'] == 0)
            
            self.cache[cache_key] = (datetime.now(), recent_games)
            return recent_games
        except Exception as e:
            logger.error(f"Error getting player games: {str(e)}")
            return None
            
    def _convert_minutes(self, min_str):
        """
        Better handling of minutes conversion with multiple formats
        """
        if pd.isna(min_str) or min_str == '':
            return 0
        
        try:
            # Handle numeric values
            if isinstance(min_str, (int, float)):
                return float(min_str)
                
            # Handle string representations of numbers
            if isinstance(min_str, str) and min_str.replace('.', '', 1).isdigit():
                return float(min_str)
                
            # Handle minute:second format
            if isinstance(min_str, str) and ':' in min_str:
                parts = min_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1]) / 60
                    return minutes + seconds
                    
            # Last resort - try direct float conversion
            return float(min_str)
        except:
            # If all else fails, return 0
            return 0
            
    def get_player_stats(self, player_id):
        """
        Get comprehensive player statistics with improved type checking and conversion
        """
        recent_games = self.get_player_recent_games(player_id)
        if recent_games is None or len(recent_games) == 0:
            return None

        # Check for inactive flag
        if 'INACTIVE' in recent_games.columns and recent_games['INACTIVE'].iloc[0]:
            return {
                'PTS_AVG': 0.0,
                'MIN': 0.0,
                'INACTIVE': True
            }

        # Filter out DNP games for more accurate stats
        active_games = recent_games[~recent_games['DNP']] if 'DNP' in recent_games.columns else recent_games
        
        # If no active games, return minimal stats
        if len(active_games) == 0:
            return {
                'PTS_AVG': 0.0,
                'MIN': 0.0,
                'GAMES_PLAYED': 0
            }

        # Ensure proper numeric conversion for all columns
        numeric_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'MIN', 'STL', 'BLK', 'TOV']
        for col in numeric_cols:
            if col in active_games.columns:
                active_games[col] = pd.to_numeric(active_games[col], errors='coerce').fillna(0)

        # Weighting recent games more heavily (exponential decay weights)
        num_games = len(active_games)
        if num_games > 1:
            weights = np.exp(np.linspace(0, -1, num_games))
            weights = weights / weights.sum()
            
            weighted_avg = lambda x: np.average(x, weights=weights[:len(x)])
        else:
            weighted_avg = lambda x: x.mean()

        # Comprehensive stats calculation with recency bias
        stats = {
            'PTS_AVG': float(weighted_avg(active_games['PTS'])),
            'FG_PCT': float(weighted_avg(active_games['FG_PCT'])) if 'FG_PCT' in active_games else 0.0,
            'FG3_PCT': float(weighted_avg(active_games['FG3_PCT'])) if 'FG3_PCT' in active_games else 0.0,
            'FT_PCT': float(weighted_avg(active_games['FT_PCT'])) if 'FT_PCT' in active_games else 0.0,
            'REB': float(weighted_avg(active_games['REB'])) if 'REB' in active_games else 0.0,
            'AST': float(weighted_avg(active_games['AST'])) if 'AST' in active_games else 0.0,
            'MIN': float(weighted_avg(active_games['MIN'])),
            'STL': float(weighted_avg(active_games['STL'])) if 'STL' in active_games else 0.0,
            'BLK': float(weighted_avg(active_games['BLK'])) if 'BLK' in active_games else 0.0,
            'TOV': float(weighted_avg(active_games['TOV'])) if 'TOV' in active_games else 0.0,
            
            # Regular averages and std for reference
            'PTS_MEAN': float(active_games['PTS'].mean()),
            'PTS_STD': float(active_games['PTS'].std()) if len(active_games) > 1 else 5.0,
            
            # Number of games played (for confidence calculation)
            'GAMES_PLAYED': int(len(active_games)),
            
            # Calculate consistency (lower std/mean ratio means more consistent)
            'CONSISTENCY': float(1.0 - min(active_games['PTS'].std() / max(active_games['PTS'].mean(), 1), 0.5)) if len(active_games) > 1 else 0.5,
            
            # Calculate usage percentage based on minutes
            'USAGE': float(active_games['MIN'].mean() / 48.0),
            
            # Scoring trend (difference between consecutive games)
            'TREND': float(active_games['PTS'].diff().dropna().mean()) if len(active_games) > 1 else 0.0,
            
            # Hot/cold streak indicator
            'STREAK': float((active_games['PTS'].iloc[0] - active_games['PTS'].mean()) / max(1, active_games['PTS'].std())) if len(active_games) > 1 else 0.0,
            
            # Add home/road splits if available
            'HOME_ROAD_DIFF': 0.0
        }
        
        # Calculate home/road performance differential
        if 'MATCHUP' in active_games.columns:
            home_games = active_games[active_games['MATCHUP'].str.contains('vs.', na=False)]
            away_games = active_games[active_games['MATCHUP'].str.contains('@', na=False)]
            
            if not home_games.empty and not away_games.empty:
                home_ppg = home_games['PTS'].mean()
                away_ppg = away_games['PTS'].mean()
                stats['HOME_PPG'] = float(home_ppg)
                stats['AWAY_PPG'] = float(away_ppg)
                stats['HOME_ROAD_DIFF'] = float(home_ppg - away_ppg)
        
        # Check if player might be injured but playing limited minutes
        recent_3_games = active_games.head(3)
        if not recent_3_games.empty:
            avg_recent_min = recent_3_games['MIN'].mean()
            avg_all_min = active_games['MIN'].mean()
            
            # If recent minutes are significantly lower, might indicate injury or reduced role
            if avg_all_min > 15 and avg_recent_min < 0.7 * avg_all_min:
                stats['REDUCED_MINUTES'] = True
                stats['MINUTES_REDUCTION'] = float(avg_all_min - avg_recent_min)
            else:
                stats['REDUCED_MINUTES'] = False
                stats['MINUTES_REDUCTION'] = 0.0
                
        # Add injury status if available
        if self.injury_data:
            player_name = self._get_player_name(player_id)
            if player_name in self.injury_data:
                stats['INJURY_STATUS'] = self.injury_data[player_name]['status']
                stats['INJURY_DETAILS'] = self.injury_data[player_name]['details']
        
        return stats

    def _get_player_name(self, player_id):
        """Get player name from ID for injury matching"""
        try:
            player_info = next((p for p in players.get_players() if p['id'] == player_id), None)
            if player_info:
                return player_info['full_name']
            return None
        except:
            return None
                
    def calculate_rest_days(self, games):
        """
        Calculate rest days between games with improved error handling
        """
        try:
            if games is None or len(games) < 2:
                return 3  # Default rest if no prior game
                
            try:
                # Try standard datetime conversion
                game_dates = pd.to_datetime(games['GAME_DATE'])
                rest = (game_dates.iloc[0] - game_dates.iloc[1]).days
                return min(rest, 7)  # Cap at 7 days
            except Exception as e1:
                # If standard conversion fails, try multiple formats
                try:
                    # Try parsing different date formats
                    formats = ['%Y-%m-%d', '%m/%d/%Y', '%b %d, %Y', '%Y%m%d']
                    
                    for fmt in formats:
                        try:
                            if isinstance(games['GAME_DATE'].iloc[0], str):
                                dates = [datetime.strptime(date, fmt) for date in games['GAME_DATE']]
                                rest = (dates[0] - dates[1]).days
                                return min(rest, 7)
                        except:
                            continue
                except Exception as e2:
                    logger.warning(f"Date conversion failed: {str(e2)}")
                    
                # Last resort - check if datetime objects with direct subtraction
                try:
                    if isinstance(games['GAME_DATE'].iloc[0], datetime):
                        rest = (games['GAME_DATE'].iloc[0] - games['GAME_DATE'].iloc[1]).days
                        return min(rest, 7)
                except:
                    pass
                    
                return 3  # Default if all parsing fails
        except Exception as e:
            logger.error(f"Error calculating rest days: {str(e)}")
            return 3  # Default rest days

    def get_team_stats(self, team_id, opponent_id=None):
        """
        Get comprehensive team stats with improved metrics and matchup analysis
        """
        try:
            recent_games = self.get_team_recent_games(team_id)
            if recent_games is None or len(recent_games) == 0:
                return None

            # Convert columns to float to avoid type errors
            numeric_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS',
                            'PACE', 'DEF_RATING']
            for col in numeric_cols:
                if col in recent_games.columns:
                    recent_games[col] = pd.to_numeric(recent_games[col], errors='coerce').fillna(0)

            # Apply recency weighting for better predictions
            num_games = len(recent_games)
            weights = np.exp(np.linspace(0, -1, num_games))
            weights = weights / weights.sum()
            
            weighted_avg = lambda x: np.average(x, weights=weights[:len(x)])

            # Calculate comprehensive team statistics with recency bias
            stats = {
                'PTS_AVG': weighted_avg(recent_games['PTS']),
                'FG_PCT': weighted_avg(recent_games['FG_PCT']),
                'FG3_PCT': weighted_avg(recent_games['FG3_PCT']),
                'FT_PCT': weighted_avg(recent_games['FT_PCT']),
                'REB': weighted_avg(recent_games['REB']),
                'AST': weighted_avg(recent_games['AST']),
                'STL': weighted_avg(recent_games['STL']),
                'BLK': weighted_avg(recent_games['BLK']),
                'TOV': weighted_avg(recent_games['TOV']),
                'PLUS_MINUS': weighted_avg(recent_games['PLUS_MINUS']),
                'WIN_PCT': (recent_games['WL'] == 'W').mean(),
                'REST_DAYS': self.calculate_rest_days(recent_games),
                'TREND_PPG': recent_games['PTS'].diff().fillna(0).mean(),
                'PTS_STD': recent_games['PTS'].std(),
                'PACE': weighted_avg(recent_games['PACE']) if 'PACE' in recent_games else 100.0,
                'DEF_RATING': weighted_avg(recent_games['DEF_RATING']) if 'DEF_RATING' in recent_games else 110.0,
                'OPP_PTS_ALLOWED': 0,
                'OPP_FG_PCT_ALLOWED': 0
            }
            
            # Add more advanced stats if available
            if 'AST_PCT' in recent_games.columns:
                stats['AST_PCT'] = weighted_avg(recent_games['AST_PCT'])
            
            if 'REB_PCT' in recent_games.columns:
                stats['REB_PCT'] = weighted_avg(recent_games['REB_PCT'])
                
            # Calculate home/away performance differential
            if 'MATCHUP' in recent_games.columns:
                home_games = recent_games[recent_games['MATCHUP'].str.contains('vs.', na=False)]
                away_games = recent_games[recent_games['MATCHUP'].str.contains('@', na=False)]
                
                if not home_games.empty and not away_games.empty:
                    stats['HOME_PPG'] = home_games['PTS'].mean()
                    stats['AWAY_PPG'] = away_games['PTS'].mean()
                    stats['HOME_ADV'] = stats['HOME_PPG'] - stats['AWAY_PPG']
            
            # Calculate recent form (last 3 games vs all 10 games)
            if len(recent_games) >= 3:
                recent_3 = recent_games.head(3)
                stats['RECENT_3_PPG'] = recent_3['PTS'].mean()
                stats['RECENT_FORM'] = stats['RECENT_3_PPG'] / stats['PTS_AVG'] if stats['PTS_AVG'] > 0 else 1.0
                
                # Check if significant recent injuries might be affecting performance
                if stats['RECENT_3_PPG'] < 0.85 * stats['PTS_AVG'] and stats['PTS_AVG'] > 100:
                    stats['RECENT_UNDERPERFORMANCE'] = True
                else:
                    stats['RECENT_UNDERPERFORMANCE'] = False

            # Get opponent defensive metrics if opponent_id provided
            if opponent_id:
                opp_games = self.get_team_recent_games(opponent_id)
                if opp_games is not None and len(opp_games) > 0:
                    # Convert columns to float
                    for col in numeric_cols:
                        if col in opp_games.columns:
                            opp_games[col] = pd.to_numeric(opp_games[col], errors='coerce').fillna(0)
                    
                    # Opponent defensive metrics
                    stats['OPP_PTS_ALLOWED'] = opp_games['PTS'].mean()
                    stats['OPP_FG_PCT_ALLOWED'] = opp_games['FG_PCT'].mean()
                    stats['OPP_DEF_RATING'] = opp_games['DEF_RATING'].mean() if 'DEF_RATING' in opp_games else 110.0
                    
                    # Calculate expected scoring against this opponent
                    off_rating = stats['PTS_AVG'] * 100 / stats['PACE'] if stats['PACE'] > 0 else stats['PTS_AVG']
                    expected_pts = (off_rating * stats['OPP_DEF_RATING'] / 110.0) * stats['PACE'] / 100
                    stats['EXPECTED_PTS'] = expected_pts
                    
                    # Calculate matchup-specific adjustments
                    try:
                        # Find previous matchups between these teams
                        matchups = recent_games[recent_games['MATCHUP'].str.contains(
                            self._get_team_abbr(opponent_id), na=False)]
                        
                        if not matchups.empty:
                            stats['H2H_PTS_AVG'] = matchups['PTS'].mean()
                            stats['H2H_DIFF'] = stats['H2H_PTS_AVG'] - stats['PTS_AVG']
                    except:
                        pass
                else:
                    # Fallback to league averages if no opponent data
                    stats['OPP_PTS_ALLOWED'] = 110
                    stats['OPP_FG_PCT_ALLOWED'] = 0.46
                    stats['OPP_DEF_RATING'] = 110.0

            # Add team injury impact
            team_injuries = self.get_team_injuries(team_id)
            if team_injuries:
                stats['KEY_PLAYERS_INJURED'] = team_injuries['key_players_out']
                stats['INJURY_IMPACT'] = team_injuries['impact_score']
            else:
                stats['KEY_PLAYERS_INJURED'] = 0
                stats['INJURY_IMPACT'] = 0
                
            return stats
        except Exception as e:
            logger.error(f"Error getting team stats: {str(e)}")
            return None
            
    def _get_team_abbr(self, team_id):
        """Helper to get team abbreviation from team ID"""
        team_info = next((t for t in teams.get_teams() if t['id'] == team_id), None)
        return team_info['abbreviation'] if team_info else None

    def update_injury_data(self, force=False):
        """
        Update injury data using player game logs and box scores
        """
        # Only update once per day unless forced
        if not force and (datetime.now() - self.injury_last_updated).total_seconds() < 86400:
            return
            
        try:
            # Initialize injury data dictionary
            injury_data = {}
            
            # Approach 1: Web scraping from reliable sources (already implemented)
            try:
                url = "https://www.cbssports.com/nba/injuries/"
                
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Parse injury tables
                    injury_tables = soup.find_all('table', class_='TableBase-table')
                    
                    for table in injury_tables:
                        rows = table.find_all('tr', class_='TableBase-bodyTr')
                        
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 4:  # Player, Position, Injury, Status
                                try:
                                    player_cell = cells[0]
                                    player_name = player_cell.find('span', class_='CellPlayerName--long').text.strip()
                                    injury_details = cells[2].text.strip()
                                    status = cells[3].text.strip()
                                    
                                    # Store in dictionary
                                    injury_data[player_name] = {
                                        'details': injury_details,
                                        'status': status,
                                        'is_out': 'out' in status.lower() or 'doubtful' in status.lower(),
                                        'source': 'web'
                                    }
                                except Exception as e:
                                    logger.warning(f"Error parsing injury row: {str(e)}")
                                    continue
                    
                    logger.info(f"Fetched {len(injury_data)} injuries from web scraping")
                else:
                    logger.warning(f"Failed to get injury data from web: {response.status_code}")
            except Exception as e:
                logger.error(f"Error scraping injury data: {str(e)}")
            
            # Approach 2: Use PlayerGameLogs to detect players who haven't been playing
            if len(injury_data) < 10:  # If web scraping found few injuries, supplement with API data
                logger.info("Supplementing injury data with player game logs analysis")
                
                from nba_api.stats.static import teams as nba_teams
                from nba_api.stats.endpoints import PlayerGameLogs
                
                # Get all active teams
                all_teams = nba_teams.get_teams()
                
                # Sample teams to avoid too many API calls
                sample_teams = random.sample(all_teams, min(10, len(all_teams)))
                
                for team in sample_teams:
                    team_id = team['id']
                    self._rate_limit()
                    
                    # Get team roster
                    try:
                        roster = self.get_team_players(team_id)
                        
                        # Check each player's recent games
                        for player in roster:
                            player_id = player['id']
                            player_name = player['full_name']
                            
                            # Skip if already in injury data
                            if player_name in injury_data:
                                continue
                                
                            self._rate_limit()
                            
                            try:
                                # Get recent games
                                player_logs = PlayerGameLogs(
                                    player_id_nullable=player_id,
                                    headers=self._get_headers()
                                ).get_data_frames()[0]
                                
                                # If no games found or empty dataframe, might be inactive
                                if player_logs.empty:
                                    # Add as potentially injured/inactive
                                    injury_data[player_name] = {
                                        'details': 'No recent games - inactive or injured',
                                        'status': 'Out (inferred)',
                                        'is_out': True,
                                        'source': 'game_logs'
                                    }
                                    continue
                                
                                # Check last 5 games
                                recent_games = player_logs.head(5)
                                
                                # Convert minutes to numeric
                                recent_games['MIN'] = recent_games['MIN'].apply(self._convert_minutes)
                                
                                # Check for DNP pattern
                                consecutive_dnp = 0
                                for _, game in recent_games.iterrows():
                                    if game['MIN'] == 0:
                                        consecutive_dnp += 1
                                    else:
                                        break
                                
                                # Player hasn't played in at least 3 consecutive games
                                if consecutive_dnp >= 3:
                                    injury_data[player_name] = {
                                        'details': f'Has not played in {consecutive_dnp} consecutive games',
                                        'status': 'Out (inferred)',
                                        'is_out': True,
                                        'source': 'game_logs'
                                    }
                                
                                # Check for limited minutes pattern (possible return from injury)
                                elif len(recent_games) >= 3:
                                    avg_mins = recent_games['MIN'].mean()
                                    # Player with significant role but playing limited minutes
                                    if player.get('avg_minutes', 0) > 20 and avg_mins < 10:
                                        injury_data[player_name] = {
                                            'details': 'Limited minutes - possible injury or return from injury',
                                            'status': 'Day-to-Day (inferred)',
                                            'is_out': False,
                                            'source': 'game_logs'
                                        }
                            except Exception as e:
                                # Skip players with errors
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing team {team['full_name']}: {str(e)}")
                        continue
            
            # Approach 3: Look at recent BoxScores for DNP comments
            try:
                from nba_api.stats.endpoints import LeagueGameLog, BoxScoreTraditionalV2
                
                # Get recent games
                self._rate_limit()
                recent_league_games = LeagueGameLog(
                    season='2023-24',  # Using current season
                    date_from_nullable=None,  # Last few days by default
                    date_to_nullable=None,
                    headers=self._get_headers()
                ).get_data_frames()[0]
                
                # Limit to most recent 5 games to avoid too many API calls
                recent_game_ids = recent_league_games['GAME_ID'].head(5).tolist()
                
                for game_id in recent_game_ids:
                    self._rate_limit()
                    
                    try:
                        # Get box score
                        box_score = BoxScoreTraditionalV2(
                            game_id=game_id,
                            headers=self._get_headers()
                        ).get_data_frames()[0]
                        
                        # Check for injury comments
                        for _, player_row in box_score.iterrows():
                            player_name = player_row['PLAYER_NAME']
                            
                            # Skip if already in injury data
                            if player_name in injury_data:
                                continue
                            
                            # Check if COMMENT column exists
                            if 'COMMENT' in box_score.columns:
                                comment = str(player_row['COMMENT']).lower() if not pd.isna(player_row['COMMENT']) else ''
                                
                                # Look for injury-related terms
                                if any(term in comment for term in ['injury', 'injured', 'illness', 'sore', 'sprain']):
                                    injury_data[player_name] = {
                                        'details': player_row['COMMENT'],
                                        'status': 'Out' if player_row['MIN'] == 0 else 'Day-to-Day',
                                        'is_out': player_row['MIN'] == 0,
                                        'source': 'box_score'
                                    }
                    except Exception as e:
                        logger.warning(f"Error processing box score for game {game_id}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error processing box scores for injury data: {str(e)}")
            
            # Update the injury data and timestamp
            self.injury_data = injury_data
            self.injury_last_updated = datetime.now()
            logger.info(f"Updated injury data: {len(injury_data)} players")
                
        except Exception as e:
            logger.error(f"Error updating injury data: {str(e)}")

            
    def get_team_injuries(self, team_id):
        """
        Get injury information for a team's players
        """
        if not self.injury_data:
            self.update_injury_data()
            
        if not self.injury_data:  # If still empty after update attempt
            return None
            
        try:
            # Get team players
            players_list = self.get_team_players(team_id)
            if not players_list:
                return None
                
            # Initialize counters
            key_players_out = 0
            rotation_players_out = 0
            bench_players_out = 0
            impact_score = 0
            injured_players = []
            
            # Check each player's injury status
            for player in players_list:
                player_name = player['full_name']
                avg_minutes = player.get('avg_minutes', 0)
                
                if player_name in self.injury_data and self.injury_data[player_name]['is_out']:
                    injured_players.append({
                        'name': player_name,
                        'status': self.injury_data[player_name]['status'],
                        'minutes': avg_minutes
                    })
                    
                    # Categorize by role based on minutes
                    if avg_minutes >= 30:  # Key player
                        key_players_out += 1
                        impact_score += 1.0  # Full impact
                    elif avg_minutes >= 20:  # Rotation player
                        rotation_players_out += 1
                        impact_score += 0.6  # Moderate impact
                    elif avg_minutes >= 10:  # Bench player
                        bench_players_out += 1
                        impact_score += 0.2  # Low impact
            
            return {
                'team_id': team_id,
                'injured_players': injured_players,
                'key_players_out': key_players_out,
                'rotation_players_out': rotation_players_out,
                'bench_players_out': bench_players_out,
                'total_players_out': key_players_out + rotation_players_out + bench_players_out,
                'impact_score': impact_score  # Higher means more impacted by injuries
            }
        except Exception as e:
            logger.error(f"Error getting team injuries: {str(e)}")
            return None

    def get_team_players(self, team_id):
        """
        Get active roster for a team with better error handling and progressive fallbacks
        """
        cache_key = f"team_players_{team_id}"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return data
                
        try:
            self._rate_limit()
            
            # Progressive fetching strategy with multiple approaches
            dashboard = None
            
            # Approach 1: Try TeamPlayerDashboard
            for attempt in range(self.retries):
                try:
                    dashboard = TeamPlayerDashboard(team_id=team_id, 
                                                 headers=self._get_headers()).get_data_frames()[1]
                    break
                except Exception as e:
                    if attempt == self.retries - 1:
                        logger.warning(f"Failed to get TeamPlayerDashboard: {str(e)}")
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            # Approach 2: Use alternative data source if dashboard failed
            if dashboard is None or dashboard.empty:
                try:
                    # Use CommonTeamRoster as alternative
                    from nba_api.stats.endpoints import CommonTeamRoster
                    roster = CommonTeamRoster(team_id=team_id, 
                                           headers=self._get_headers()).get_data_frames()[0]
                    if not roster.empty:
                        dashboard = roster
                        logger.info(f"Used CommonTeamRoster fallback for team {team_id}")
                except Exception as e2:
                    logger.warning(f"Failed to get CommonTeamRoster: {str(e2)}")
            
            # Approach 3: Last resort - build from recent game logs
            if dashboard is None or dashboard.empty:
                try:
                    # Get recent team games to find players
                    games = self.get_team_recent_games(team_id, 5)
                    if games is not None and not games.empty:
                        # Get box scores for these games to find players
                        from nba_api.stats.endpoints import BoxScoreTraditionalV2
                        
                        players_seen = {}
                        for _, game in games.iterrows():
                            try:
                                game_id = game['GAME_ID']
                                box = BoxScoreTraditionalV2(game_id=game_id, 
                                                         headers=self._get_headers()).get_data_frames()[0]
                                
                                # Filter for players from this team
                                team_box = box[box['TEAM_ID'] == team_id]
                                
                                for _, player in team_box.iterrows():
                                    player_id = player['PLAYER_ID']
                                    player_name = player['PLAYER_NAME']
                                    minutes = self._convert_minutes(player['MIN'])
                                    
                                    if player_id not in players_seen:
                                        players_seen[player_id] = {
                                            'id': player_id,
                                            'full_name': player_name,
                                            'games': 1,
                                            'minutes': [minutes]
                                        }
                                    else:
                                        players_seen[player_id]['games'] += 1
                                        players_seen[player_id]['minutes'].append(minutes)
                            except:
                                continue
                        
                        # Create custom player list from box scores
                        if players_seen:
                            players_list = []
                            for player_id, data in players_seen.items():
                                avg_minutes = sum(data['minutes']) / len(data['minutes']) if data['minutes'] else 0
                                games_played = data['games']
                                
                                # Only include players with meaningful minutes
                                if avg_minutes >= 5 or games_played >= 2:
                                    players_list.append({
                                        'id': player_id,
                                        'full_name': data['full_name'],
                                        'avg_minutes': avg_minutes,
                                        'games_played': games_played
                                    })
                            
                            # Sort by minutes
                            players_list = sorted(players_list, key=lambda x: x['avg_minutes'], reverse=True)
                            
                            if players_list:
                                logger.info(f"Built team roster from box scores for team {team_id}")
                                self.cache[cache_key] = (datetime.now(), players_list)
                                return players_list
                except Exception as e3:
                    logger.warning(f"Failed to build roster from box scores: {str(e3)}")
            
            # If still no data, give up
            if dashboard is None or dashboard.empty:
                logger.error(f"Could not get player data for team {team_id} after all attempts")
                return []
            
            # Process the dashboard/roster data
            players_list = []
            
            for _, row in dashboard.iterrows():
                # Get player ID
                player_id = row['PLAYER_ID'] if 'PLAYER_ID' in row.index else None
                
                if player_id is None:
                    continue
                
                # Try to extract name (different endpoints use different field names)
                player_name = None
                for name_field in ['PLAYER_NAME', 'PLAYER', 'NAME']:
                    if name_field in row.index:
                        player_name = row[name_field]
                        break
                
                if player_name is None:
                    continue
                
                # Try to get minutes information from the dashboard
                avg_minutes = None
                for min_field in ['MIN', 'GP_MIN', 'AVG_MIN']:
                    if min_field in row.index:
                        try:
                            avg_minutes = self._convert_minutes(row[min_field])
                            break
                        except:
                            pass
                
                # Get player games with a timeout
                player_games = None
                
                # Only try fetching games if we don't have minutes yet
                if avg_minutes is None:
                    start_time = time.time()
                    while time.time() - start_time < 2.0:  # 2-second timeout per player
                        try:
                            player_games = self.get_player_recent_games(player_id, 5)
                            break
                        except Exception:
                            time.sleep(0.5)
                    
                    # If we got games, extract minutes
                    if player_games is not None and len(player_games) > 0:
                        if 'MIN' in player_games.columns:
                            avg_minutes = player_games['MIN'].mean()
                
                # If still no minutes, use a default if player appears to be active
                if avg_minutes is None:
                    # Check if there's a games played field
                    for gp_field in ['GP', 'GAMES_PLAYED']:
                        if gp_field in row.index and pd.to_numeric(row[gp_field], errors='coerce') > 0:
                            avg_minutes = 10  # Conservative default
                            break
                    
                    # Last resort default
                    if avg_minutes is None:
                        avg_minutes = 5
                
                # Check if player is actually active (minutes threshold lowered to include more bench players)
                if avg_minutes >= 5:
                    players_list.append({
                        'id': player_id,
                        'full_name': player_name,
                        'avg_minutes': avg_minutes
                    })
                    
            # Sort by minutes played (most important players first)
            players_list = sorted(players_list, key=lambda x: x['avg_minutes'], reverse=True)
            
            # Cache the results
            self.cache[cache_key] = (datetime.now(), players_list)
            
            return players_list
            
        except Exception as e:
            logger.error(f"Error getting team players for {team_id}: {str(e)}")
            return []