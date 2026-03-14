import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    TeamGameLogs, PlayerGameLogs, CommonPlayerInfo, TeamPlayerDashboard,
    LeagueGameLog, BoxScoreTraditionalV2, BoxScoreAdvancedV2,
    TeamGameLog, PlayerGameLog, BoxScoreUsageV2, BoxScoreFourFactorsV2
)
from nba_api.stats.static import teams, players
import logging
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sqlite3
import json

@dataclass
class GameContext:
    """Context information for a game"""
    rest_days: int
    is_back_to_back: bool
    home_game: bool
    opponent_strength: float
    season_progress: float
    injury_impact: float

class EnhancedDataCollector:
    def _safe_get_data_frames(self, endpoint):
        """Safely get data frames from NBA API endpoint"""
        try:
            # Try the standard method first
            data_frames = endpoint.get_data_frames()
            
            if data_frames and len(data_frames) > 0:
                return data_frames
            
            # If that fails, try getting raw data
            try:
                raw_data = endpoint.get_dict()
                if 'resultSets' in raw_data and raw_data['resultSets']:
                    result_set = raw_data['resultSets'][0]
                    if 'rowSet' in result_set and 'headers' in result_set:
                        df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                        return [df]
                elif 'resultSet' in raw_data:
                    result_set = raw_data['resultSet']
                    if 'rowSet' in result_set and 'headers' in result_set:
                        df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                        return [df]
            except Exception as raw_error:
                logging.warning(f"Raw data extraction failed: {raw_error}")
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting data frames: {e}")
            return None
    
    """Enhanced data collector with advanced metrics and caching"""

    def _json_serializer(slef, obj):
        """Custom JSON serializer for NumPy/Pandas types"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # Handle scalar numpy types
            try:
                return obj.item()
            except:
                return str(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def __init__(self, cache_db_path='nba_cache.db'):
        self.cache_db_path = cache_db_path
        self.setup_database()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.5
        
        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        # League averages for context
        self.league_averages = {
            'pace': 100.0,
            'off_rating': 115.0,
            'def_rating': 115.0,
            'efg_pct': 0.54,
            'ts_pct': 0.57,
            'ast_ratio': 24.0,
            'reb_rate': 50.0
        }

    def setup_database(self):
        """Setup SQLite database for caching"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Create tables for caching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_games (
                team_id INTEGER,
                game_id TEXT,
                game_date TEXT,
                season TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_games (
                player_id INTEGER,
                game_id TEXT,
                game_date TEXT,
                season TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_stats (
                entity_id INTEGER,
                entity_type TEXT,
                stat_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def _rate_limit(self):
        """Enhanced rate limiting with jitter"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            jitter = random.uniform(0, 0.5)
            sleep_time = self.min_request_interval - elapsed + jitter
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _get_headers(self):
        """Get rotating headers"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Referer': 'https://stats.nba.com/',
            'Accept-Language': 'en-US,en;q=0.9'
        }

    def get_team_stats(self, team_id: int, opponent_id: int = None, season: str = '2023-24') -> Dict:
        """Get comprehensive team statistics with advanced metrics"""
        try:
            # Check cache first
            cached_data = self._get_cached_stats(team_id, 'team', 'advanced')
            if cached_data:
                return cached_data
            
            self._rate_limit()
            
            # Get basic game logs with error handling
            try:
                team_games_endpoint = TeamGameLogs(
                    team_id_nullable=team_id,
                    season_nullable=season,
                    headers=self._get_headers()
                )
                
                # Get the data frames - handle different response structures
                data_frames = self._safe_get_data_frames(team_games_endpoint)
                
                if not data_frames or len(data_frames) == 0:
                    logging.warning(f"No data frames returned for team {team_id}")
                    return {}
                    
                team_games = data_frames[0]
                
            except Exception as api_error:
                logging.error(f"NBA API error for team {team_id}: {api_error}")
                # Try alternative approach or return empty dict
                return {}
            
            if team_games.empty:
                return {}
            
            # Calculate advanced metrics
            stats = self._calculate_team_metrics(team_games)
            
            # Cache the results
            self._cache_stats(team_id, 'team', 'advanced', stats)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting advanced team stats: {e}")
            return {}

    def _calculate_team_metrics(self, games_df: pd.DataFrame) -> Dict:
        """Calculate advanced team metrics from game logs"""
        try:
            # Ensure numeric columns
            numeric_cols = ['PTS', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 
                           'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF']
            
            for col in numeric_cols:
                if col in games_df.columns:
                    games_df[col] = pd.to_numeric(games_df[col], errors='coerce').fillna(0)
            
            # Basic averages
            stats = {
                'games_played': len(games_df),
                'pts_avg': games_df['PTS'].mean(),
                'pts_std': games_df['PTS'].std(),
                'fg_pct': games_df['FGM'].sum() / max(1, games_df['FGA'].sum()),
                'fg3_pct': games_df['FG3M'].sum() / max(1, games_df['FG3A'].sum()),
                'ft_pct': games_df['FTM'].sum() / max(1, games_df['FTA'].sum()),
            }
            
            # Advanced metrics
            # Effective Field Goal Percentage
            stats['efg_pct'] = (games_df['FGM'].sum() + 0.5 * games_df['FG3M'].sum()) / max(1, games_df['FGA'].sum())
            
            # True Shooting Percentage
            total_pts = games_df['PTS'].sum()
            total_fga = games_df['FGA'].sum()
            total_fta = games_df['FTA'].sum()
            stats['ts_pct'] = total_pts / max(1, 2 * (total_fga + 0.44 * total_fta))
            
            # Pace (possessions per game)
            # Estimate possessions = FGA + 0.44*FTA + TOV - OREB
            possessions = games_df['FGA'] + 0.44 * games_df['FTA'] + games_df['TOV'] - games_df['OREB']
            stats['pace'] = possessions.mean()
            
            # Offensive Rating (points per 100 possessions)
            stats['off_rating'] = (total_pts / max(1, possessions.sum())) * 100
            
            # Assist Rate
            total_ast = games_df['AST'].sum()
            total_fgm = games_df['FGM'].sum()
            stats['ast_rate'] = total_ast / max(1, total_fgm)
            
            # Turnover Rate
            stats['tov_rate'] = games_df['TOV'].sum() / max(1, possessions.sum()) * 100
            
            # Rebound Rate (estimate)
            stats['reb_rate'] = games_df['REB'].mean()
            
            # Four Factors
            stats['four_factors'] = {
                'efg_pct': stats['efg_pct'],
                'tov_rate': stats['tov_rate'],
                'oreb_pct': games_df['OREB'].sum() / max(1, games_df['REB'].sum()),
                'ft_rate': total_fta / max(1, total_fga)
            }
            
            # Recent form (last 10 games)
            if len(games_df) >= 10:
                recent_games = games_df.head(10)
                stats['recent_form'] = {
                    'pts_avg_l10': recent_games['PTS'].mean(),
                    'fg_pct_l10': recent_games['FGM'].sum() / max(1, recent_games['FGA'].sum()),
                    'wins_l10': (recent_games['WL'] == 'W').sum(),
                    'pts_trend': self._calculate_trend(recent_games['PTS'])
                }
            
            # Home/Road splits
            if 'MATCHUP' in games_df.columns:
                home_games = games_df[games_df['MATCHUP'].str.contains('vs.', na=False)]
                road_games = games_df[games_df['MATCHUP'].str.contains('@', na=False)]
                
                if not home_games.empty and not road_games.empty:
                    stats['home_road_splits'] = {
                        'home_pts_avg': home_games['PTS'].mean(),
                        'road_pts_avg': road_games['PTS'].mean(),
                        'home_wins': (home_games['WL'] == 'W').sum(),
                        'road_wins': (road_games['WL'] == 'W').sum(),
                        'home_games': len(home_games),
                        'road_games': len(road_games)
                    }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating advanced team metrics: {e}")
            return {}

    def get_player_stats(self, player_id: int, season: str = '2023-24') -> Dict:
        """Get comprehensive player statistics with advanced metrics"""
        try:
            # Check cache first
            cached_data = self._get_cached_stats(player_id, 'player', 'enhanced')
            if cached_data:
                return cached_data
            
            self._rate_limit()
            
            # Get player game logs with error handling
            try:
                player_games_endpoint = PlayerGameLogs(
                    player_id_nullable=player_id,
                    season_nullable=season,
                    headers=self._get_headers()
                )
                
                # Get the data frames - handle different response structures
                data_frames = self._safe_get_data_frames(player_games_endpoint)
                
                if not data_frames or len(data_frames) == 0:
                    logging.warning(f"No data frames returned for player {player_id}")
                    return {}
                    
                player_games = data_frames[0]
                
            except Exception as api_error:
                logging.error(f"NBA API error for player {player_id}: {api_error}")
                # Return empty dict if API fails
                return {}
            
            if player_games.empty:
                return {}
            
            # Calculate enhanced metrics
            stats = self._calculate_player_metrics(player_games)
            
            # Get additional context
            stats.update(self._get_player_context(player_id, player_games))
            
            # Cache the results
            self._cache_stats(player_id, 'player', 'enhanced', stats)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting enhanced player stats: {e}")
            return {}

    def _calculate_player_metrics(self, games_df: pd.DataFrame) -> Dict:
        """Calculate enhanced player metrics"""
        try:
            # Convert minutes to numeric
            games_df['MIN'] = games_df['MIN'].apply(self._convert_minutes)
            
            # Ensure numeric columns
            numeric_cols = ['PTS', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 
                           'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'MIN']
            
            for col in numeric_cols:
                if col in games_df.columns:
                    games_df[col] = pd.to_numeric(games_df[col], errors='coerce').fillna(0)
            
            # Filter out DNP games
            active_games = games_df[games_df['MIN'] > 0]
            
            if active_games.empty:
                return {'status': 'inactive'}
            
            # Basic stats
            stats = {
                'games_played': len(active_games),
                'minutes_avg': active_games['MIN'].mean(),
                'minutes_std': active_games['MIN'].std(),
                'pts_avg': active_games['PTS'].mean(),
                'pts_std': active_games['PTS'].std(),
                'pts_per_min': active_games['PTS'].sum() / max(1, active_games['MIN'].sum()),
                'usage_estimate': self._estimate_usage(active_games)
            }
            
            # Shooting efficiency
            total_fgm = active_games['FGM'].sum()
            total_fga = active_games['FGA'].sum()
            total_fg3m = active_games['FG3M'].sum()
            total_fg3a = active_games['FG3A'].sum()
            total_ftm = active_games['FTM'].sum()
            total_fta = active_games['FTA'].sum()
            
            stats['shooting'] = {
                'fg_pct': total_fgm / max(1, total_fga),
                'fg3_pct': total_fg3m / max(1, total_fg3a),
                'ft_pct': total_ftm / max(1, total_fta),
                'efg_pct': (total_fgm + 0.5 * total_fg3m) / max(1, total_fga),
                'ts_pct': active_games['PTS'].sum() / max(1, 2 * (total_fga + 0.44 * total_fta))
            }
            
            # Per-36 minute stats
            total_minutes = active_games['MIN'].sum()
            if total_minutes > 0:
                stats['per_36'] = {
                    'pts': active_games['PTS'].sum() * 36 / total_minutes,
                    'reb': active_games['REB'].sum() * 36 / total_minutes,
                    'ast': active_games['AST'].sum() * 36 / total_minutes,
                    'stl': active_games['STL'].sum() * 36 / total_minutes,
                    'blk': active_games['BLK'].sum() * 36 / total_minutes,
                    'tov': active_games['TOV'].sum() * 36 / total_minutes
                }
            
            # Consistency metrics
            if len(active_games) > 1:
                stats['consistency'] = {
                    'pts_cv': stats['pts_std'] / max(1, stats['pts_avg']),  # Coefficient of variation
                    'double_digit_games': (active_games['PTS'] >= 10).sum(),
                    'single_digit_games': (active_games['PTS'] < 10).sum(),
                    'big_games': (active_games['PTS'] >= stats['pts_avg'] * 1.5).sum()
                }
            
            # Recent form analysis
            if len(active_games) >= 5:
                recent_5 = active_games.head(5)
                recent_10 = active_games.head(10) if len(active_games) >= 10 else active_games
                
                stats['recent_form'] = {
                    'pts_l5': recent_5['PTS'].mean(),
                    'pts_l10': recent_10['PTS'].mean(),
                    'min_l5': recent_5['MIN'].mean(),
                    'min_l10': recent_10['MIN'].mean(),
                    'trend_l10': self._calculate_trend(recent_10['PTS']),
                    'hot_streak': self._identify_streak(active_games['PTS'], stats['pts_avg'])
                }
            
            # Situational performance
            stats['situational'] = self._analyze_situational_performance(active_games, stats['pts_avg'])
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating enhanced player metrics: {e}")
            return {}

    def _estimate_usage(self, games_df: pd.DataFrame) -> float:
        """Estimate player usage rate from available stats"""
        try:
            # Usage Rate approximation: (FGA + 0.44*FTA + TOV) / Team possessions * (Team MIN / Player MIN)
            # Simplified version using available data
            
            player_possessions = games_df['FGA'] + 0.44 * games_df['FTA'] + games_df['TOV']
            player_minutes = games_df['MIN']
            
            # Estimate team possessions (rough approximation)
            # Assume team has ~100 possessions per game on average
            estimated_team_poss_per_min = 100 / 48  # ~2.08 per minute
            
            usage_estimates = []
            for _, game in games_df.iterrows():
                if game['MIN'] > 0:
                    game_team_poss = estimated_team_poss_per_min * game['MIN']
                    game_usage = game['FGA'] + 0.44 * game['FTA'] + game['TOV']
                    usage_rate = game_usage / max(1, game_team_poss)
                    usage_estimates.append(min(1.0, usage_rate))  # Cap at 100%
            
            return np.mean(usage_estimates) if usage_estimates else 0.2
            
        except:
            # Fallback estimation based on minutes and points
            if games_df['MIN'].mean() >= 30:
                return 0.25  # High usage
            elif games_df['MIN'].mean() >= 20:
                return 0.20  # Medium usage
            else:
                return 0.15  # Low usage

    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate linear trend (slope) of a series"""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = values.values
            
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
            
        except:
            return 0.0

    def _identify_streak(self, values: pd.Series, average: float) -> Dict:
        """Identify hot/cold streaks"""
        try:
            if len(values) < 3:
                return {'type': 'none', 'length': 0}
            
            # Recent 3 games
            recent_3 = values.head(3)
            above_avg = (recent_3 > average).sum()
            
            if above_avg >= 2:
                return {'type': 'hot', 'length': above_avg, 'avg_over_last_3': recent_3.mean()}
            elif above_avg == 0:
                return {'type': 'cold', 'length': 3, 'avg_over_last_3': recent_3.mean()}
            else:
                return {'type': 'neutral', 'length': 0, 'avg_over_last_3': recent_3.mean()}
                
        except:
            return {'type': 'none', 'length': 0}

    def _analyze_situational_performance(self, games_df: pd.DataFrame, avg_pts: float) -> Dict:
        """Analyze performance in different situations"""
        try:
            situational = {}
            
            # Home vs Away performance
            if 'MATCHUP' in games_df.columns:
                home_games = games_df[games_df['MATCHUP'].str.contains('vs.', na=False)]
                away_games = games_df[games_df['MATCHUP'].str.contains('@', na=False)]
                
                if not home_games.empty and not away_games.empty:
                    situational['home_away'] = {
                        'home_pts_avg': home_games['PTS'].mean(),
                        'away_pts_avg': away_games['PTS'].mean(),
                        'home_advantage': home_games['PTS'].mean() - away_games['PTS'].mean(),
                        'home_games': len(home_games),
                        'away_games': len(away_games)
                    }
            
            # Performance vs rest
            if len(games_df) >= 2:
                rest_performance = self._analyze_rest_performance(games_df)
                situational['rest'] = rest_performance
            
            # Minutes correlation with performance
            if len(games_df) >= 5:
                minutes_corr = np.corrcoef(games_df['MIN'], games_df['PTS'])[0, 1]
                situational['minutes_correlation'] = minutes_corr if not np.isnan(minutes_corr) else 0
            
            return situational
            
        except Exception as e:
            logging.error(f"Error in situational analysis: {e}")
            return {}

    def _analyze_rest_performance(self, games_df: pd.DataFrame) -> Dict:
        """Analyze performance based on rest days"""
        try:
            # Calculate rest days between games
            game_dates = pd.to_datetime(games_df['GAME_DATE'])
            rest_days = []
            
            for i in range(len(game_dates) - 1):
                rest = (game_dates.iloc[i] - game_dates.iloc[i + 1]).days
                rest_days.append(rest)
            
            if not rest_days:
                return {}
            
            # Add rest days to dataframe (shifted by 1 since rest is before the game)
            games_with_rest = games_df.iloc[:-1].copy()
            games_with_rest['REST_DAYS'] = rest_days
            
            # Analyze performance by rest
            b2b_games = games_with_rest[games_with_rest['REST_DAYS'] == 1]  # Back-to-back
            normal_rest = games_with_rest[games_with_rest['REST_DAYS'].between(2, 3)]
            long_rest = games_with_rest[games_with_rest['REST_DAYS'] >= 4]
            
            rest_analysis = {}
            
            if not b2b_games.empty:
                rest_analysis['back_to_back'] = {
                    'games': len(b2b_games),
                    'pts_avg': b2b_games['PTS'].mean(),
                    'min_avg': b2b_games['MIN'].mean()
                }
            
            if not normal_rest.empty:
                rest_analysis['normal_rest'] = {
                    'games': len(normal_rest),
                    'pts_avg': normal_rest['PTS'].mean(),
                    'min_avg': normal_rest['MIN'].mean()
                }
            
            if not long_rest.empty:
                rest_analysis['long_rest'] = {
                    'games': len(long_rest),
                    'pts_avg': long_rest['PTS'].mean(),
                    'min_avg': long_rest['MIN'].mean()
                }
            
            return rest_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing rest performance: {e}")
            return {}

    def _get_player_context(self, player_id: int, games_df: pd.DataFrame) -> Dict:
        """Get additional context about player"""
        try:
            context = {}
            
            # Get player info with error handling
            self._rate_limit()
            try:
                player_info_endpoint = CommonPlayerInfo(
                    player_id=player_id,
                    headers=self._get_headers()
                )
                
                # Get the data frames
                data_frames = self._safe_get_data_frames(player_info_endpoint)
                
                if data_frames and len(data_frames) > 0:
                    player_info = data_frames[0]
                    
                    if not player_info.empty:
                        context['position'] = player_info['POSITION'].iloc[0] if 'POSITION' in player_info.columns else None
                        context['height'] = player_info['HEIGHT'].iloc[0] if 'HEIGHT' in player_info.columns else None
                        context['weight'] = player_info['WEIGHT'].iloc[0] if 'WEIGHT' in player_info.columns else None
                        context['experience'] = player_info['SEASON_EXP'].iloc[0] if 'SEASON_EXP' in player_info.columns else None
                        
            except Exception as e:
                logging.warning(f"Could not get player info for {player_id}: {e}")
            
            # Role analysis based on minutes and usage
            avg_minutes = games_df['MIN'].mean()
            if avg_minutes >= 32:
                context['role'] = 'star'
            elif avg_minutes >= 24:
                context['role'] = 'starter'
            elif avg_minutes >= 15:
                context['role'] = 'rotation'
            else:
                context['role'] = 'bench'
            
            return context
            
        except Exception as e:
            logging.error(f"Error getting player context: {e}")
            return {}

    def _convert_minutes(self, min_str):
        """Convert minutes string to float"""
        if pd.isna(min_str) or min_str == '':
            return 0.0
        
        try:
            if isinstance(min_str, (int, float)):
                return float(min_str)
            
            if isinstance(min_str, str):
                if ':' in min_str:
                    parts = min_str.split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1]) / 60
                        return minutes + seconds
                else:
                    return float(min_str)
            
            return float(min_str)
        except:
            return 0.0

    def get_game_context(self, team_id: int, game_date: str = None) -> GameContext:
        """Get contextual information for a game"""
        try:
            # Get recent games to calculate rest
            recent_games = self.get_team_recent_games(team_id, n_games=5)
            if recent_games is None or len(recent_games) < 2:
                return GameContext(
                    rest_days=1,
                    is_back_to_back=False,
                    home_game=True,
                    opponent_strength=0.5,
                    season_progress=0.5,
                    injury_impact=0.0
                )
            
            # Calculate rest days
            game_dates = pd.to_datetime(recent_games['GAME_DATE'])
            rest_days = (game_dates.iloc[0] - game_dates.iloc[1]).days
            is_back_to_back = rest_days == 1
            
            # Check if most recent game was home
            home_game = 'vs.' in str(recent_games['MATCHUP'].iloc[0])
            
            # Season progress (approximate)
            season_progress = self._calculate_season_progress()
            
            # Injury impact from team
            injury_impact = 0.0  # Will be set by main predictor
            
            return GameContext(
                rest_days=rest_days,
                is_back_to_back=is_back_to_back,
                home_game=home_game,
                opponent_strength=0.5,  # Default, will be calculated with opponent data
                season_progress=season_progress,
                injury_impact=injury_impact
            )
            
        except Exception as e:
            logging.error(f"Error getting game context: {e}")
            return GameContext(1, False, True, 0.5, 0.5, 0.0)

    def _calculate_season_progress(self) -> float:
        """Calculate season progress (0-1)"""
        try:
            now = datetime.now()
            
            # NBA season typically runs October-April
            if now.month >= 10:  # October-December
                season_start = datetime(now.year, 10, 15)  # Season usually starts mid-October
                season_end = datetime(now.year + 1, 4, 15)   # Regular season ends mid-April
            else:  # January-April
                season_start = datetime(now.year - 1, 10, 15)
                season_end = datetime(now.year, 4, 15)
            
            total_days = (season_end - season_start).days
            current_days = (now - season_start).days
            
            return max(0, min(1, current_days / total_days))
            
        except:
            return 0.5

    def get_team_recent_games(self, team_id: int, n_games: int = 10, season: str = None) -> pd.DataFrame:
        """Enhanced team game fetching with caching"""
        try:
            # Check cache first
            cached_games = self._get_cached_team_games(team_id, season)
            if cached_games is not None and len(cached_games) >= n_games:
                return cached_games.head(n_games)
            
            self._rate_limit()
            
            # Fetch from API with error handling
            try:
                if season:
                    games_endpoint = TeamGameLogs(
                        team_id_nullable=team_id,
                        season_nullable=season,
                        headers=self._get_headers()
                    )
                else:
                    games_endpoint = TeamGameLogs(
                        team_id_nullable=team_id,
                        headers=self._get_headers()
                    )
                
                # Get the data frames
                data_frames = self._safe_get_data_frames(games_endpoint)
                
                if not data_frames or len(data_frames) == 0:
                    logging.warning(f"No data frames returned for team games {team_id}")
                    return None
                    
                games = data_frames[0]
                
            except Exception as api_error:
                logging.error(f"NBA API error getting team games for {team_id}: {api_error}")
                return None
            
            if games.empty:
                return None
            
            # Cache the games
            self._cache_team_games(team_id, games, season)
            
            return games.head(n_games)
            
        except Exception as e:
            logging.error(f"Error getting team games: {e}")
            return None

    def get_player_recent_games(self, player_id: int, n_games: int = 15) -> pd.DataFrame:
        """Enhanced player game fetching with caching"""
        try:
            # Check cache first
            cached_games = self._get_cached_player_games(player_id)
            if cached_games is not None and len(cached_games) >= n_games:
                return cached_games.head(n_games)
            
            self._rate_limit()
            
            # Fetch from API with error handling
            try:
                games_endpoint = PlayerGameLogs(
                    player_id_nullable=player_id,
                    headers=self._get_headers()
                )
                
                # Get the data frames
                data_frames = self._safe_get_data_frames(games_endpoint)
                
                if not data_frames or len(data_frames) == 0:
                    logging.warning(f"No data frames returned for player games {player_id}")
                    return None
                    
                games = data_frames[0]
                
            except Exception as api_error:
                logging.error(f"NBA API error getting player games for {player_id}: {api_error}")
                return None
            
            if games.empty:
                return None
            
            # Cache the games
            self._cache_player_games(player_id, games)
            
            return games.head(n_games)
            
        except Exception as e:
            logging.error(f"Error getting player games: {e}")
            return None

    # Caching methods
    def _get_cached_team_games(self, team_id: int, season: str = None) -> pd.DataFrame:
        """Get cached team games"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            query = '''
                SELECT data FROM team_games 
                WHERE team_id = ? AND season = ? 
                AND created_at > datetime('now', '-1 day')
                ORDER BY created_at DESC LIMIT 1
            '''
            
            cursor = conn.execute(query, (team_id, season or '2023-24'))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Load as dict first, then convert to DataFrame
                games_dict = json.loads(row[0])
                return pd.DataFrame(games_dict)
            return None
            
        except Exception as e:
            logging.error(f"Error getting cached team games: {e}")
            return None

    def _cache_team_games(self, team_id: int, games_df: pd.DataFrame, season: str = None):
        """Cache team games"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Convert DataFrame to JSON with proper type handling
            # First convert to dict to handle numpy types
            games_dict = games_df.to_dict('records')
            games_json = json.dumps(games_dict, default=self._json_serializer)
            
            # Insert into cache
            conn.execute('''
                INSERT INTO team_games (team_id, season, data)
                VALUES (?, ?, ?)
            ''', (team_id, season or '2023-24', games_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error caching team games: {e}")

    def _get_cached_player_games(self, player_id: int) -> pd.DataFrame:
        """Get cached player games"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            query = '''
                SELECT data FROM player_games 
                WHERE player_id = ? 
                AND created_at > datetime('now', '-1 day')
                ORDER BY created_at DESC LIMIT 1
            '''
            
            cursor = conn.execute(query, (player_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Load as dict first, then convert to DataFrame
                games_dict = json.loads(row[0])
                return pd.DataFrame(games_dict)
            return None
            
        except Exception as e:
            logging.error(f"Error getting cached player games: {e}")
            return None

    def _cache_player_games(self, player_id: int, games_df: pd.DataFrame):
        """Cache player games"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Convert DataFrame to JSON with proper type handling
            # First convert to dict to handle numpy types
            games_dict = games_df.to_dict('records')
            games_json = json.dumps(games_dict, default=self._json_serializer)
            
            # Insert into cache
            conn.execute('''
                INSERT INTO player_games (player_id, data)
                VALUES (?, ?)
            ''', (player_id, games_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error caching player games: {e}")

    def _get_cached_stats(self, entity_id: int, entity_type: str, stat_type: str) -> Dict:
        """Get cached advanced stats"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            query = '''
                SELECT data FROM advanced_stats 
                WHERE entity_id = ? AND entity_type = ? AND stat_type = ?
                AND created_at > datetime('now', '-12 hours')
                ORDER BY created_at DESC LIMIT 1
            '''
            
            cursor = conn.execute(query, (entity_id, entity_type, stat_type))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
            
        except Exception as e:
            logging.error(f"Error getting cached advanced stats: {e}")
            return None

    def _cache_stats(self, entity_id: int, entity_type: str, stat_type: str, stats: Dict):
        """Cache advanced stats"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Convert stats to JSON with proper type handling
            stats_json = json.dumps(stats, default=self._json_serializer)
            
            # Insert into cache
            conn.execute('''
                INSERT INTO advanced_stats (entity_id, entity_type, stat_type, data)
                VALUES (?, ?, ?, ?)
            ''', (entity_id, entity_type, stat_type, stats_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error caching advanced stats: {e}")

    def get_matchup_history(self, team1_id: int, team2_id: int, n_games: int = 5) -> Dict:
        """Get head-to-head matchup history between two teams"""
        try:
            # Get recent games for both teams
            team1_games = self.get_team_recent_games(team1_id, n_games=20)
            team2_games = self.get_team_recent_games(team2_id, n_games=20)
            
            if team1_games is None or team2_games is None:
                return {}
            
            # Find matchups between these teams
            team1_abbr = self._get_team_abbreviation(team1_id)
            team2_abbr = self._get_team_abbreviation(team2_id)
            
            if not team1_abbr or not team2_abbr:
                return {}
            
            # Find games where team1 played team2
            team1_vs_team2 = team1_games[
                team1_games['MATCHUP'].str.contains(team2_abbr, na=False)
            ].head(n_games)
            
            # Find games where team2 played team1
            team2_vs_team1 = team2_games[
                team2_games['MATCHUP'].str.contains(team1_abbr, na=False)
            ].head(n_games)
            
            # Combine and analyze
            all_matchups = pd.concat([team1_vs_team2, team2_vs_team1])
            
            if all_matchups.empty:
                return {}
            
            # Calculate head-to-head stats
            h2h_stats = {
                'total_games': len(all_matchups),
                'avg_total_points': all_matchups['PTS'].mean(),
                'high_scoring_games': (all_matchups['PTS'] > 120).sum(),
                'low_scoring_games': (all_matchups['PTS'] < 100).sum(),
                'recent_trend': self._calculate_trend(all_matchups['PTS'])
            }
            
            return h2h_stats
            
        except Exception as e:
            logging.error(f"Error getting matchup history: {e}")
            return {}

    def _get_team_abbreviation(self, team_id: int) -> str:
        """Get team abbreviation from team ID"""
        try:
            team_info = next((t for t in teams.get_teams() if t['id'] == team_id), None)
            return team_info['abbreviation'] if team_info else None
        except:
            return None

    def get_team_players(self, team_id: int) -> List[Dict]:
        """Get team players list"""
        try:
            from nba_api.stats.static import players
            
            # This is a simplified implementation
            # In a real scenario, you'd want to get current roster from API
            all_players = players.get_players()
            
            # Return a sample of players (this should be replaced with actual roster API call)
            sample_players = all_players[:12]  # Get first 12 as sample
            
            return [{
                'id': player['id'],
                'full_name': player['full_name']
            } for player in sample_players]
            
        except Exception as e:
            logging.error(f"Error getting team players: {e}")
            return []
    
    def cleanup_cache(self, days_old: int = 7):
        """Clean up old cache entries"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Delete old entries
            conn.execute('''
                DELETE FROM team_games 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days_old))
            
            conn.execute('''
                DELETE FROM player_games 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days_old))
            
            conn.execute('''
                DELETE FROM advanced_stats 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days_old))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up cache entries older than {days_old} days")
            
            
        except Exception as e:
            logging.error(f"Error cleaning up cache: {e}")