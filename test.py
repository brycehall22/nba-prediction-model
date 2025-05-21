from nba_api.stats.endpoints import leaguegamefinder, teamplayerdashboard, commonplayerinfo
from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.static import teams

# Get data from a shorter time period first as a test
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

# Query for games where the Celtics were playing
gamefinder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable='1/8/2025',
                date_to_nullable='2/7/2025',
                league_id_nullable='00'
            )
# The first DataFrame of those returned is what we want.
games = gamefinder.get_data_frames()[0]
games.head()

# print(games)

def get_team_name(team_id):
    nba_teams = teams.get_teams()

    team = [team for team in nba_teams if team['id'] == team_id][0]
    team_name = team['full_name']

    return team_name


# print(get_team_name(1610612738))

def get_active_roster(team_id):
    """Get active roster for a team with recent stats"""
    try:
        # Fetch team players (assuming it returns a list of dicts)
        team_players = teamplayerdashboard.TeamPlayerDashboard(team_id=team_id)

        players = team_players.get_data_frames()[1]
        print("Team Players Columns:", players.columns)

        print(type(players))
        print(players)

        player_stats = []
        for _, player in players.iterrows():  # Iterate over DataFrame rows properly
            stats = collector.get_player_stats(player['PLAYER_ID'])  # Ensure correct column name
            if stats and stats.get('PTS_AVG') is not None:
                player_stats.append({
                    'id': player['PLAYER_ID'],  # Ensure correct column
                    'name': player['PLAYER_NAME'],  # Adjust if necessary
                    'recent_ppg': stats['PTS_AVG'],
                    'stats': stats
                })

        # Sort by recent scoring average
        return sorted(player_stats, key=lambda x: x['recent_ppg'], reverse=True)
    except Exception as e:
        print(f"Error in get_active_roster: {str(e)}")
        return []

# print(get_active_roster(1610612738))

team_id = 1610612738 

team_players = teamplayerdashboard.TeamPlayerDashboard(team_id=team_id)
players = team_players.get_data_frames()[1]

# Filter out players with no recent games
current_players = players[players['GP'] > 0]

player_stats = []
for _, player in current_players.iterrows():
    try:
        # Get player info to verify active status
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['PLAYER_ID']).get_data_frames()[0]
        
        # Skip if player is not active or on a different team
        if (player_info['ROSTERSTATUS'].iloc[0] != 'Active' or str(player_info['TEAM_ID'].iloc[0]) != str(team_id)):
            continue

        print(player_info)
    except Exception as e:
        print(e)
        continue