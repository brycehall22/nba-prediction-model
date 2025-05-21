import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import ScoreboardV2, teamplayerdashboard, commonplayerinfo
from predictor import NBAPredictor, DataCollector
from betting_integration import BettingDataService, PredictionCalibrator
from player_prop_service import PlayerPropsService
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import os
import re

st.set_page_config(page_title="NBA Game Prediction System", layout="wide")

@st.cache_resource
def get_predictor():
    return NBAPredictor()

@st.cache_resource
def get_betting_service():
    # Initialize betting service
    betting_service = BettingDataService(
        odds_file_path='draftkings_nba_odds.csv',
        update_interval_hours=4  # Update every 4 hours
    )
    return betting_service

@st.cache_resource
def get_calibrator(_betting_service):
    return PredictionCalibrator(_betting_service)

@st.cache_resource
def get_props_service():
    # Initialize player props service
    props_service = PlayerPropsService(
        props_file_path='draftkings_player_points_props.csv',
        update_interval_hours=4  # Update every 4 hours
    )
    return props_service

@st.cache_data(ttl=300)
def get_todays_games():
    try:
        scoreboard = ScoreboardV2().get_data_frames()
        games = scoreboard[0]
        line_score = scoreboard[1]
        if games.empty:
            return None
        merged_games = pd.merge(games, line_score, left_on='GAME_ID', right_on='GAME_ID')
        return merged_games
    except Exception as e:
        st.error(f"Error fetching today's games: {str(e)}")
        return None

def get_team_name(team_id):
    nba_teams = teams.get_teams()
    team = [team for team in nba_teams if team['id'] == team_id][0]
    return team['full_name']

# Calculate expected value for a bet
def calculate_ev(probability, american_odds):
    """Calculate the expected value of a bet"""
    if american_odds > 0:
        return probability * (american_odds / 100) - (1 - probability)
    else:
        return probability * (100 / abs(american_odds)) - (1 - probability)

# Custom CSS for better appearance
st.markdown("""
<style>
    .game-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
    }
    .prediction-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f8f9fa;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 5px;
        margin-top: 5px;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .injury-report {
        background-color: #fff3f3;
        border-left: 3px solid #ff6b6b;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .injured-player {
        color: #dc3545;
        font-weight: bold;
    }
    .questionable-player {
        color: #fd7e14;
        font-style: italic;
    }
    .player-row {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
    }
    .stat-comparison {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .stat-label {
        font-weight: bold;
    }
    .impact-high {
        color: #dc3545;
        font-weight: bold;
    }
    .impact-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .impact-low {
        color: #adb5bd;
        font-weight: normal;
    }
    .market-comparison {
        background-color: #edf7ed;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .value-high {
        color: #28a745;
        font-weight: bold;
    }
    .value-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .value-low {
        color: #dc3545;
        font-weight: normal;
    }
    .vegas-data {
        font-family: monospace;
        font-size: 0.9em;
    }
    .positive-ev {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .neutral-ev {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .negative-ev {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .prop-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        transition: transform 0.2s;
    }
    .prop-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .ev-indicator {
        font-weight: bold;
        border-radius: 4px;
        padding: 2px 6px;
        display: inline-block;
    }
    .ev-positive {
        background-color: #d4edda;
        color: #28a745;
    }
    .ev-neutral {
        background-color: #fff3cd;
        color: #856404;
    }
    .ev-negative {
        background-color: #f8d7da;
        color: #dc3545;
    }
    .prop-details {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .edge-meter {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    .edge-fill {
        height: 100%;
        border-radius: 4px;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
predictor = get_predictor()
collector = DataCollector()
betting_service = get_betting_service()
calibrator = get_calibrator(betting_service)
props_service = get_props_service()

# Check if betting and props data exist and update if needed
if not os.path.exists('draftkings_nba_odds.csv') or st.sidebar.button("Refresh Betting Data"):
    with st.sidebar:
        with st.spinner("Updating betting data..."):
            updated = betting_service.update_betting_data(force=True)
            if updated:
                st.success("Betting data updated successfully")
            else:
                st.error("Failed to update betting data")

if not os.path.exists('draftkings_player_points_props.csv') or st.sidebar.button("Refresh Player Props"):
    with st.sidebar:
        with st.spinner("Updating player props data..."):
            updated = props_service.update_props_data(force=True)
            if updated:
                st.success("Player props data updated successfully")
            else:
                st.error("Failed to update player props data")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Game Predictions", "Player Props", "Today's Games"])

with tab1:
    st.title("NBA Game Prediction System")
    
    all_teams = pd.DataFrame(teams.get_teams())
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Team")
        home_team = st.selectbox("Select Home Team", all_teams['full_name'], key='home')
        home_team_id = all_teams[all_teams['full_name'] == home_team]['id'].iloc[0]
    
    with col2:
        st.subheader("Away Team")
        away_team = st.selectbox("Select Away Team", all_teams['full_name'], key='away')
        away_team_id = all_teams[all_teams['full_name'] == away_team]['id'].iloc[0]
    
    def get_active_roster(team_id):
        try:
            team_players = predictor.get_team_players(team_id)
            if not team_players:
                return []
            return team_players
        except Exception as e:
            st.error(f"Error fetching roster: {str(e)}")
            return []
    
    def create_player_comparison_chart(home_df, away_df, props_data=None):
        fig = go.Figure()
        
        # Add home team bars
        fig.add_trace(go.Bar(
            x=home_df['Player'],
            y=home_df['Projected Points'],
            name=f"Home Team Projected",
            marker_color='#1f77b4',
            text=home_df['Projected Points'].apply(lambda x: f"{x:.1f}"),
            textposition='auto'
        ))
        
        # Add away team bars
        fig.add_trace(go.Bar(
            x=away_df['Player'],
            y=away_df['Projected Points'],
            name=f"Away Team Projected",
            marker_color='#ff7f0e',
            text=away_df['Projected Points'].apply(lambda x: f"{x:.1f}"),
            textposition='auto'
        ))
        
        # If props data is available, add market lines
        if props_data:
            # Extract player names from home and away dataframes
            all_players = list(home_df['Player']) + list(away_df['Player'])
            market_lines = []
            market_players = []
            
            for player in all_players:
                # Try to find the player in props data (exact or partial match)
                prop_player = None
                if player in props_data:
                    prop_player = player
                else:
                    # Try partial match (last name)
                    player_last_name = player.split()[-1]
                    for pp in props_data.keys():
                        if player_last_name in pp:
                            prop_player = pp
                            break
                
                if prop_player:
                    market_lines.append(props_data[prop_player]['line'])
                    market_players.append(player)
            
            # Add market lines as scatter plot
            if market_lines:
                fig.add_trace(go.Scatter(
                    x=market_players,
                    y=market_lines,
                    mode='markers',
                    name='Market Line',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=[f"Line: {line:.1f}" for line in market_lines],
                    hoverinfo='text+name'
                ))
        
        # Update layout
        fig.update_layout(
            title='Player Scoring Projections vs Market Lines',
            xaxis_title='Player',
            yaxis_title='Points',
            barmode='group',
            height=500,
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.1)'
            ),
            xaxis=dict(
                tickangle=-45,
                categoryorder='total descending'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def display_injury_report(injury_info, team_name):
        """Display formatted injury report"""
        if not injury_info or team_name.lower() not in injury_info:
            st.info(f"No reported injuries for {team_name}")
            return
            
        team_injuries = injury_info[team_name.lower()]
        players_out = team_injuries.get('players_out', [])
        impact = team_injuries.get('impact', 0)
        
        if not players_out:
            st.info(f"No reported injuries for {team_name}")
            return
            
        st.markdown(f"<div class='injury-report'>", unsafe_allow_html=True)
        
        # Impact assessment
        impact_class = "impact-high" if impact > 1.5 else "impact-medium" if impact > 0.7 else "impact-low"
        impact_text = "High" if impact > 1.5 else "Medium" if impact > 0.7 else "Low"
        
        st.markdown(f"<h4>{team_name} Injuries</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>Impact Assessment: <span class='{impact_class}'>{impact_text}</span> ({impact:.2f})</p>", unsafe_allow_html=True)
        
        # List injured players
        for player in players_out:
            st.markdown(f"<div class='injured-player'>{player}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    def display_betting_comparison(prediction):
        """Display enhanced betting market comparison data"""
        if not prediction.get('market_data', {}).get('available', False):
            st.warning("Betting market data is not available for this matchup")
            return
            
        market_data = prediction['market_data']
        
        # Create market comparison box with enhanced styling
        st.markdown("""
        <div style='background-color: #f8f9fa; border: 1px solid #dee2e6; 
            border-radius: 10px; padding: 20px; margin-top: 20px;'>
        <h4 style='text-align: center; margin-bottom: 20px; color: #343a40;'>
            <span style='border-bottom: 2px solid #007bff; padding-bottom: 5px;'>
            Betting Market Comparison</span>
        </h4>
        """, unsafe_allow_html=True)
        
        # Create three columns for model vs market vs value
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h5 style='text-align: center;'>Model Prediction</h5>", unsafe_allow_html=True)
            predicted_margin = prediction['home_score'] - prediction['away_score']
            predicted_total = prediction['home_score'] + prediction['away_score']
            
            st.markdown(f"""
            <div style='text-align: center; color: #343a40; padding: 10px; background-color: #e9ecef; border-radius: 5px;'>
                <p><b>Spread:</b> {"Home" if predicted_margin > 0 else "Away"} by {abs(predicted_margin):.1f}</p>
                <p><b>Total:</b> {predicted_total:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h5 style='text-align: center;'>Vegas Line</h5>", unsafe_allow_html=True)
            vegas_spread = market_data.get('home_spread', 'N/A')
            vegas_total = market_data.get('total', 'N/A')
            
            spread_team = "Home" if vegas_spread and vegas_spread < 0 else "Away"
            spread_value = abs(vegas_spread) if vegas_spread is not None else 'N/A'
            
            st.markdown(f"""
            <div style='text-align: center; color: #343a40; padding: 10px; background-color: #e9ecef; border-radius: 5px;'>
                <p><b>Spread:</b> {spread_team} by {spread_value}</p>
                <p><b>Total:</b> {vegas_total}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate edge percentage
            spread_edge = None
            total_edge = None
            
            if market_data.get('home_spread') is not None and predicted_margin != 0:
                vegas_margin = -market_data.get('home_spread', 0)
                spread_edge = abs(predicted_margin - vegas_margin) / max(abs(predicted_margin), abs(vegas_margin)) * 100
                
            if market_data.get('total') is not None and predicted_total > 0:
                vegas_total = market_data.get('total', 0)
                total_edge = abs(predicted_total - vegas_total) / vegas_total * 100
            
            st.markdown("<h5 style='text-align: center;'>Edge Analysis</h5>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='text-align: center; color: #343a40; padding: 10px; background-color: #e9ecef; border-radius: 5px;'>
                <p><b>Spread Edge:</b> {spread_edge:.1f}% {get_edge_emoji(spread_edge)}</p>
                <p><b>Total Edge:</b> {total_edge:.1f}% {get_edge_emoji(total_edge)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate expected value for spread and total bets
        st.markdown("<h5 style='text-align: center; margin-top: 15px;'>Expected Value Analysis</h5>", unsafe_allow_html=True)
        
        spread_col1, spread_col2 = st.columns(2)
        
        with spread_col1:
            # Calculate spread EV
            if market_data.get('home_spread') is not None and predicted_margin != 0:
                vegas_margin = -market_data.get('home_spread', 0)
                
                # Calculate probability of covering
                # Simple sigmoid function
                spread_diff = predicted_margin - vegas_margin
                prob_cover_home = 1 / (1 + np.exp(-0.25 * spread_diff))
                prob_cover_away = 1 - prob_cover_home
                
                # Get odds for the spread
                home_spread_odds = market_data.get('home_spread_odds', -110)
                away_spread_odds = market_data.get('away_spread_odds', -110)
                
                # Calculate EV
                home_spread_ev = calculate_ev(prob_cover_home, home_spread_odds)
                away_spread_ev = calculate_ev(prob_cover_away, away_spread_odds)
                
                # Determine best spread bet
                best_spread_bet = "Home" if home_spread_ev > away_spread_ev else "Away"
                best_spread_ev = max(home_spread_ev, away_spread_ev)
                
                # Color code based on EV value
                spread_ev_class = "positive-ev" if best_spread_ev > 0.05 else "neutral-ev" if best_spread_ev > -0.05 else "negative-ev"
                
                # Display recommendation
                st.markdown(f"""
                <div class='{spread_ev_class}' style='color: #343a40;'>
                    <h6>Spread Recommendation:</h6>
                    <p><b>{best_spread_bet} {abs(market_data.get('home_spread', 0))}</b> ({home_spread_odds if best_spread_bet == "Home" else away_spread_odds})</p>
                    <p>EV: {best_spread_ev*100:.1f}%</p>
                    <p>Model probability: {(prob_cover_home if best_spread_bet == "Home" else prob_cover_away)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with spread_col2:
            # Calculate total EV
            if market_data.get('total') is not None and predicted_total > 0:
                vegas_total = market_data.get('total', 0)
                
                # Calculate probability of over/under
                # Simple sigmoid function
                total_diff = predicted_total - vegas_total
                prob_over = 1 / (1 + np.exp(-0.25 * total_diff))
                prob_under = 1 - prob_over
                
                # Get odds for the total
                over_odds = market_data.get('total_over_odds', -110)
                under_odds = market_data.get('total_under_odds', -110)
                
                # Calculate EV
                over_ev = calculate_ev(prob_over, over_odds)
                under_ev = calculate_ev(prob_under, under_odds)
                
                # Determine best total bet
                best_total_bet = "Over" if over_ev > under_ev else "Under"
                best_total_ev = max(over_ev, under_ev)
                
                # Color code based on EV value
                total_ev_class = "positive-ev" if best_total_ev > 0.05 else "neutral-ev" if best_total_ev > -0.05 else "negative-ev"
                
                # Display recommendation
                st.markdown(f"""
                <div class='{total_ev_class}' style='color: #343a40;'>
                    <h6>Total Recommendation:</h6>
                    <p><b>{best_total_bet} {vegas_total}</b> ({over_odds if best_total_bet == "Over" else under_odds})</p>
                    <p>EV: {best_total_ev*100:.1f}%</p>
                    <p>Model probability: {(prob_over if best_total_bet == "Over" else prob_under)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display moneyline and implied probabilities
        if market_data.get('home_moneyline') is not None and market_data.get('away_moneyline') is not None:
            st.markdown("<h5 style='text-align: center; margin-top: 15px;'>Moneyline Analysis</h5>", unsafe_allow_html=True)
            
            home_ml = market_data.get('home_moneyline')
            away_ml = market_data.get('away_moneyline')
            
            # Calculate model win probabilities based on predicted scores
            # Simple sigmoid function to convert point differential to win probability
            if 'home_score' in prediction and 'away_score' in prediction:
                point_diff = prediction['home_score'] - prediction['away_score']
                model_home_prob = 1 / (1 + np.exp(-0.25 * point_diff))
                model_away_prob = 1 - model_home_prob
            else:
                model_home_prob = 0.5
                model_away_prob = 0.5
            
            # Calculate EV
            home_ml_ev = calculate_ev(model_home_prob, home_ml)
            away_ml_ev = calculate_ev(model_away_prob, away_ml)
            
            # Determine best ML bet
            best_ml_bet = "Home" if home_ml_ev > away_ml_ev else "Away"
            best_ml_ev = max(home_ml_ev, away_ml_ev)
            
            # Color code based on EV value
            ml_ev_class = "positive-ev" if best_ml_ev > 0.05 else "neutral-ev" if best_ml_ev > -0.05 else "negative-ev"
            
            st.markdown(f"""
            <div class='{ml_ev_class}' style='color: #343a40;'>
                <h6>Moneyline Recommendation:</h6>
                <p><b>{best_ml_bet}</b> ({home_ml if best_ml_bet == "Home" else away_ml})</p>
                <p>EV: {best_ml_ev*100:.1f}%</p>
                <p>Model probability: {(model_home_prob if best_ml_bet == "Home" else model_away_prob)*100:.1f}%</p>
                <p>Market implied probability: {(market_data.get('home_implied_probability', 0) if best_ml_bet == "Home" else market_data.get('away_implied_probability', 0))*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display regression information if applied
        if market_data.get('regression_applied', False):
            st.markdown(f"""
            <p style='color: #fd7e14; font-style: italic; margin-top: 15px; text-align: center;'>
                Note: Prediction was adjusted {market_data.get('regression_strength', 0)*100:.1f}% toward market consensus.
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Helper function for emoji indicators
    def get_edge_emoji(edge_percentage):
        if edge_percentage is None:
            return ""
        elif edge_percentage >= 10:
            return "🔥"  # Fire emoji for strong edge
        elif edge_percentage >= 5:
            return "⭐"  # Star emoji for good edge
        else:
            return ""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict Game", use_container_width=True):
            with st.spinner("Fetching stats and making predictions..."):
                try:
                    # Make prediction
                    prediction = predictor.predict_game(home_team_id, away_team_id)
                    
                    # Calibrate prediction with betting data
                    calibrated_prediction = calibrator.calibrate_prediction(prediction, home_team, away_team)
                    
                    # Get player props data
                    props_data = props_service.get_player_props(home_team, away_team)
                    
                    # Compare prediction with market props
                    props_comparison = props_service.compare_prediction_with_market(calibrated_prediction, home_team, away_team)
                    
                    if 'error' in calibrated_prediction and calibrated_prediction['error']:
                        st.error(f"Error making prediction: {calibrated_prediction['error']}")
                    else:
                        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                        
                        # Display injury information if available
                        if 'injuries' in calibrated_prediction:
                            st.subheader("Injury Report")
                            col_home_inj, col_away_inj = st.columns(2)
                            
                            with col_home_inj:
                                display_injury_report(calibrated_prediction['injuries'], 'home')
                                
                            with col_away_inj:
                                display_injury_report(calibrated_prediction['injuries'], 'away')
                        
                        # Display betting market comparison
                        st.subheader("Market Comparison")
                        display_betting_comparison(calibrated_prediction)
                        
                        # Game prediction section
                        st.subheader("Game Prediction")
                        
                        # Score visualizations
                        col_score, col_confidence = st.columns([2, 1])
                        with col_score:
                            # Score bar chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=[home_team],
                                y=[calibrated_prediction['home_score']],
                                name='Home Team',
                                marker_color='#1f77b4',
                                text=[f"{calibrated_prediction['home_score']:.1f}"],
                                textposition='auto'
                            ))
                            fig.add_trace(go.Bar(
                                x=[away_team],
                                y=[calibrated_prediction['away_score']],
                                name='Away Team',
                                marker_color='#ff7f0e',
                                text=[f"{calibrated_prediction['away_score']:.1f}"],
                                textposition='auto'
                            ))
                            fig.update_layout(
                                title='Predicted Final Score',
                                yaxis_title='Points',
                                showlegend=True,
                                barmode='group',
                                height=400,
                                yaxis=dict(
                                    showgrid=True,
                                    zeroline=True,
                                    gridcolor='rgba(0,0,0,0.1)',
                                    zerolinecolor='rgba(0,0,0,0.1)'
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_confidence:
                            # Prediction summary
                            winner = home_team if calibrated_prediction['home_score'] > calibrated_prediction['away_score'] else away_team
                            margin = abs(calibrated_prediction['home_score'] - calibrated_prediction['away_score'])
                            
                            st.markdown(f"""
                            ### Prediction Summary
                            - **Winner**: {winner}
                            - **Margin**: {margin:.1f} points
                            - **Final Score**: {home_team} {calibrated_prediction['home_score']:.1f} - {calibrated_prediction['away_score']:.1f} {away_team}
                            - **Method**: {calibrated_prediction['prediction_method'].capitalize()}
                            """)
                            
                            # Confidence meter with market influence note
                            st.markdown("### Prediction Confidence")
                            confidence = calibrated_prediction['confidence']
                            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                            st.progress(confidence)
                            st.markdown(f"<p style='text-align: center; color: {confidence_color};'>{confidence:.1%}</p>", unsafe_allow_html=True)
                            
                            if 'market_data' in calibrated_prediction and calibrated_prediction['market_data'].get('available', False):
                                market_factor = calibrated_prediction['market_data'].get('market_confidence_factor', 1.0)
                                if market_factor < 0.95:
                                    st.info(f"Confidence adjusted by market comparison")
                            
                            if confidence > 0.7:
                                st.info("High confidence prediction")
                            elif confidence > 0.5:
                                st.warning("Moderate confidence prediction")
                            else:
                                st.error("Low confidence prediction")
                        
                        # Process player predictions
                        home_players = calibrated_prediction.get('home_player_predictions', [])
                        away_players = calibrated_prediction.get('away_player_predictions', [])
                        
                        # Create DataFrames for player predictions with additional columns
                        home_df = pd.DataFrame([
                            {
                                'Player': p['name'],
                                'Projected Points': p['points'],
                                'Projected Minutes': p['minutes'],
                                'Status': p['status']
                            } for p in home_players
                        ])
                        
                        away_df = pd.DataFrame([
                            {
                                'Player': p['name'],
                                'Projected Points': p['points'],
                                'Projected Minutes': p['minutes'],
                                'Status': p['status']
                            } for p in away_players
                        ])
                        
                        # Add market data to player DataFrames if available
                        if props_data:
                            # Function to find player in props data
                            def find_prop_match(player_name):
                                if player_name in props_data:
                                    return props_data[player_name]
                                else:
                                    # Try partial match (last name)
                                    player_last_name = player_name.split()[-1]
                                    for pp in props_data.keys():
                                        if player_last_name in pp:
                                            return props_data[pp]
                                return None
                            
                            # Add market data to home players
                            market_lines = []
                            market_odds = []
                            market_edges = []
                            market_ev = []
                            market_bet = []
                            
                            for player in home_df['Player']:
                                prop = find_prop_match(player)
                                if prop:
                                    # Get player projection from home_df
                                    projection = home_df.loc[home_df['Player'] == player, 'Projected Points'].iloc[0]
                                    
                                    # Calculate edge
                                    edge = projection - prop['line']
                                    edge_pct = (abs(edge) / prop['line']) * 100 if prop['line'] > 0 else 0
                                    
                                    # Calculate probability of covering
                                    if edge > 0:
                                        prob = 0.5 + min(0.35, edge * 0.07)  # Simple probability model
                                        market_odds.append(prop['over_odds'])
                                        market_bet.append('OVER')
                                    else:
                                        prob = 0.5 + min(0.35, abs(edge) * 0.07)
                                        market_odds.append(prop['under_odds'])
                                        market_bet.append('UNDER')
                                    
                                    # Calculate EV
                                    odds = prop['over_odds'] if edge > 0 else prop['under_odds']
                                    ev = calculate_ev(prob, odds)
                                    
                                    market_lines.append(prop['line'])
                                    market_edges.append(edge_pct)
                                    market_ev.append(ev)
                                else:
                                    market_lines.append(None)
                                    market_odds.append(None)
                                    market_edges.append(None)
                                    market_ev.append(None)
                                    market_bet.append(None)
                            
                            home_df['Market Line'] = market_lines
                            home_df['Market Odds'] = market_odds
                            home_df['Edge %'] = market_edges
                            home_df['EV'] = market_ev
                            home_df['Bet'] = market_bet
                            
                            # Same for away players
                            market_lines = []
                            market_odds = []
                            market_edges = []
                            market_ev = []
                            market_bet = []
                            
                            for player in away_df['Player']:
                                prop = find_prop_match(player)
                                if prop:
                                    # Get player projection from away_df
                                    projection = away_df.loc[away_df['Player'] == player, 'Projected Points'].iloc[0]
                                    
                                    # Calculate edge
                                    edge = projection - prop['line']
                                    edge_pct = (abs(edge) / prop['line']) * 100 if prop['line'] > 0 else 0
                                    
                                    # Calculate probability of covering
                                    if edge > 0:
                                        prob = 0.5 + min(0.35, edge * 0.07)  # Simple probability model
                                        market_odds.append(prop['over_odds'])
                                        market_bet.append('OVER')
                                    else:
                                        prob = 0.5 + min(0.35, abs(edge) * 0.07)
                                        market_odds.append(prop['under_odds'])
                                        market_bet.append('UNDER')
                                    
                                    # Calculate EV
                                    odds = prop['over_odds'] if edge > 0 else prop['under_odds']
                                    ev = calculate_ev(prob, odds)
                                    
                                    market_lines.append(prop['line'])
                                    market_edges.append(edge_pct)
                                    market_ev.append(ev)
                                else:
                                    market_lines.append(None)
                                    market_odds.append(None)
                                    market_edges.append(None)
                                    market_ev.append(None)
                                    market_bet.append(None)
                            
                            away_df['Market Line'] = market_lines
                            away_df['Market Odds'] = market_odds
                            away_df['Edge %'] = market_edges
                            away_df['EV'] = market_ev
                            away_df['Bet'] = market_bet
                        
                        # Add player props visualization with tabs
                        player_tabs = st.tabs(["Scoring Projections vs Market", "Player Props Value", "Minutes Distribution"])
                        
                        with player_tabs[0]:
                            # Player points projection chart with market comparison
                            if not home_df.empty and not away_df.empty:
                                player_chart = create_player_comparison_chart(
                                    home_df.head(8), away_df.head(8), props_data
                                )
                                st.plotly_chart(player_chart, use_container_width=True)
                        
                        with player_tabs[1]:
                            # Display value bets if props data is available
                            if props_data and 'players' in props_comparison and props_comparison['players']:
                                st.markdown("### Player Props Value Finder")
                                
                                # Combine home and away dataframes and sort by EV
                                all_players = pd.concat([home_df, away_df])
                                all_players = all_players.dropna(subset=['Market Line', 'EV'])
                                all_players = all_players.sort_values(by='EV', ascending=False)
                                
                                # Display top value bets
                                if not all_players.empty:
                                    top_value = all_players[all_players['EV'] > 0.02].head(5)
                                    if not top_value.empty:
                                        st.markdown("#### Top Value Bets")
                                        for idx, player in top_value.iterrows():
                                            # Determine EV class for styling
                                            ev_class = "ev-positive" if player['EV'] > 0.05 else "ev-neutral" if player['EV'] > 0 else "ev-negative"
                                            
                                            # Create prop card
                                            st.markdown(f"""
                                            <div class="prop-card">
                                                <h5>{player['Player']}</h5>
                                                <div class="prop-details">
                                                    <span>Points: <b>{player['Market Line']}</b> {player['Bet']}</span>
                                                    <span class="ev-indicator {ev_class}">EV: {player['EV']*100:.1f}%</span>
                                                </div>
                                                <p>Projection: {player['Projected Points']:.1f} points</p>
                                                <div class="edge-meter">
                                                    <div class="edge-fill" style="width: {min(100, player['Edge %'])}%; background-color: {'#28a745' if player['EV'] > 0.05 else '#ffc107' if player['EV'] > 0 else '#dc3545'}"></div>
                                                </div>
                                                <p>Edge: {player['Edge %']:.1f}% | Odds: {player['Market Odds']}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.info("No significant value bets found based on model projections")
                                else:
                                    st.warning("No player props with market data available")
                            else:
                                st.warning("Player props data not available for this matchup")
                        
                        with player_tabs[2]:
                            # Minutes distribution chart
                            if not home_df.empty and not away_df.empty:
                                # Create minutes visualization
                                fig = go.Figure()
                                
                                # Add home team bars
                                fig.add_trace(go.Bar(
                                    x=home_df['Player'].head(8),
                                    y=home_df['Projected Minutes'].head(8),
                                    name=f"Home Team Minutes",
                                    marker_color='#1f77b4',
                                    text=home_df['Projected Minutes'].head(8).apply(lambda x: f"{x:.1f}"),
                                    textposition='auto'
                                ))
                                
                                # Add away team bars
                                fig.add_trace(go.Bar(
                                    x=away_df['Player'].head(8),
                                    y=away_df['Projected Minutes'].head(8),
                                    name=f"Away Team Minutes",
                                    marker_color='#ff7f0e',
                                    text=away_df['Projected Minutes'].head(8).apply(lambda x: f"{x:.1f}"),
                                    textposition='auto'
                                ))
                                
                                fig.update_layout(
                                    title='Player Minutes Projection',
                                    xaxis_title='Player',
                                    yaxis_title='Minutes',
                                    barmode='group',
                                    height=500,
                                    yaxis=dict(
                                        showgrid=True,
                                        zeroline=True,
                                        gridcolor='rgba(0,0,0,0.1)',
                                        zerolinecolor='rgba(0,0,0,0.1)'
                                    ),
                                    xaxis=dict(
                                        tickangle=-45,
                                        categoryorder='total descending'
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Player detailed tables
                        st.subheader("Detailed Player Projections")
                        col_home, col_away = st.columns(2)
                        
                        with col_home:
                            st.write(f"### {home_team} Players")
                            if not home_df.empty:
                                # Add status highlighting
                                def highlight_status(val):
                                    if 'injured' in str(val).lower():
                                        return 'background-color: #ffcccc'
                                    elif 'day' in str(val).lower():
                                        return 'background-color: #fff2cc'
                                    else:
                                        return ''
                                
                                # Add EV highlighting
                                def highlight_ev(val):
                                    if val is None or pd.isna(val):
                                        return ''
                                    if val > 0.05:
                                        return 'background-color: #d4edda'
                                    elif val > 0:
                                        return 'background-color: #fff3cd'
                                    else:
                                        return 'background-color: #f8d7da'
                                
                                # Select columns to display
                                display_cols = ['Player', 'Projected Points', 'Status']
                                if 'Market Line' in home_df.columns:
                                    display_cols.extend(['Market Line', 'Bet', 'Edge %', 'EV'])
                                
                                # Apply styling
                                styled_df = home_df[display_cols].style.format({
                                    'Projected Points': '{:.1f}',
                                    'Market Line': '{:.1f}',
                                    'Edge %': '{:.1f}%',
                                    'EV': '{:.1%}'
                                }).applymap(highlight_status, subset=['Status'])
                                
                                if 'EV' in display_cols:
                                    styled_df = styled_df.applymap(highlight_ev, subset=['EV'])
                                
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.warning("No player data available")
                        
                        with col_away:
                            st.write(f"### {away_team} Players")
                            if not away_df.empty:
                                # Select columns to display
                                display_cols = ['Player', 'Projected Points', 'Status']
                                if 'Market Line' in away_df.columns:
                                    display_cols.extend(['Market Line', 'Bet', 'Edge %', 'EV'])
                                
                                # Apply styling
                                styled_df = away_df[display_cols].style.format({
                                    'Projected Points': '{:.1f}',
                                    'Market Line': '{:.1f}',
                                    'Edge %': '{:.1f}%',
                                    'EV': '{:.1%}'
                                }).applymap(highlight_status, subset=['Status'])
                                
                                if 'EV' in display_cols:
                                    styled_df = styled_df.applymap(highlight_ev, subset=['EV'])
                                
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.warning("No player data available")
                        
                        # Additional matchup insights
                        st.subheader("Matchup Insights")
                        
                        # Fetch team stats for insights
                        home_stats = collector.get_team_stats(home_team_id, away_team_id)
                        away_stats = collector.get_team_stats(away_team_id, home_team_id)
                        
                        # Generate insights based on stats
                        insights = []
                        
                        # Add betting insights
                        if 'market_data' in calibrated_prediction and calibrated_prediction['market_data'].get('available', False):
                            market_data = calibrated_prediction['market_data']
                            
                            # Spread value insight
                            if market_data.get('spread_value_rating', 0) >= 7:
                                model_favors = "home" if calibrated_prediction['home_score'] > calibrated_prediction['away_score'] else "away"
                                vegas_favors = "home" if market_data.get('home_spread', 0) < 0 else "away"
                                
                                if model_favors != vegas_favors:
                                    insights.append(f"Model disagrees with Vegas on the favorite, showing high potential value.")
                                else:
                                    insights.append(f"Model shows a stronger edge for the {model_favors} team than the betting market suggests.")
                            
                            # Total value insight
                            if market_data.get('total_value_rating', 0) >= 7:
                                predicted_total = calibrated_prediction['home_score'] + calibrated_prediction['away_score']
                                vegas_total = market_data.get('total', 0)
                                
                                if predicted_total > vegas_total + 5:
                                    insights.append(f"Model predicts a higher-scoring game than the betting market (OVER value).")
                                elif predicted_total < vegas_total - 5:
                                    insights.append(f"Model predicts a lower-scoring game than the betting market (UNDER value).")
                        
                        # Player props insights
                        if props_comparison and props_comparison.get('available', False):
                            if props_comparison.get('summary', {}).get('value_opportunities', 0) > 0:
                                insights.append(f"Found {props_comparison['summary']['value_opportunities']} high-value player prop opportunities in this game.")
                            
                            # Find top value props
                            top_props = []
                            for player_name, prop_data in props_comparison.get('players', {}).items():
                                if prop_data.get('is_value', False):
                                    top_props.append((player_name, prop_data))
                            
                            top_props = sorted(top_props, key=lambda x: x[1].get('edge_strength', 0), reverse=True)[:2]
                            
                            for player_name, prop_data in top_props:
                                position = prop_data.get('market_position', '')
                                line = prop_data.get('market_line', 0)
                                insights.append(f"Strong value on {player_name} {position} {line} points.")
                        
                        # Home court advantage
                        insights.append("Home court advantage typically adds 3-4 points to the home team's score.")
                        
                        # Injury impact insights
                        if 'injuries' in calibrated_prediction:
                            if 'home' in calibrated_prediction['injuries'] and calibrated_prediction['injuries']['home'].get('impact', 0) > 0.7:
                                insights.append(f"{home_team} has significant injuries that may impact performance.")
                            if 'away' in calibrated_prediction['injuries'] and calibrated_prediction['injuries']['away'].get('impact', 0) > 0.7:
                                insights.append(f"{away_team} has significant injuries that may impact performance.")
                        
                        # Rest days advantage
                        if home_stats['REST_DAYS'] > away_stats['REST_DAYS'] + 1:
                            insights.append(f"{home_team} has a significant rest advantage ({home_stats['REST_DAYS']} days vs {away_stats['REST_DAYS']} days).")
                        elif away_stats['REST_DAYS'] > home_stats['REST_DAYS'] + 1:
                            insights.append(f"{away_team} has a significant rest advantage ({away_stats['REST_DAYS']} days vs {home_stats['REST_DAYS']} days).")
                        
                        # Recent form
                        if home_stats['TREND_PPG'] > 2:
                            insights.append(f"{home_team} has been scoring better in recent games (trend: +{home_stats['TREND_PPG']:.1f} PPG).")
                        elif home_stats['TREND_PPG'] < -2:
                            insights.append(f"{home_team} has been scoring worse in recent games (trend: {home_stats['TREND_PPG']:.1f} PPG).")
                        
                        if away_stats['TREND_PPG'] > 2:
                            insights.append(f"{away_team} has been scoring better in recent games (trend: +{away_stats['TREND_PPG']:.1f} PPG).")
                        elif away_stats['TREND_PPG'] < -2:
                            insights.append(f"{away_team} has been scoring worse in recent games (trend: {away_stats['TREND_PPG']:.1f} PPG).")
                        
                        # Pace insights
                        if 'PACE' in home_stats and 'PACE' in away_stats:
                            avg_pace = (home_stats['PACE'] + away_stats['PACE']) / 2
                            if avg_pace > 102:
                                insights.append(f"This game features two fast-paced teams (avg. pace: {avg_pace:.1f}), suggesting a higher-scoring game.")
                            elif avg_pace < 98:
                                insights.append(f"This game features two slower-paced teams (avg. pace: {avg_pace:.1f}), suggesting a lower-scoring game.")
                        
                        # Display insights
                        for insight in insights:
                            st.markdown(f"- {insight}")
                        
                        # Disclaimer
                        st.markdown("---")
                        st.markdown("""
                        <div style='text-align: center; font-style: italic;'>
                            <p>This prediction is based on recent team and player performance data and may differ from betting market consensus. 
                            Predictions are for entertainment purposes only and should not be used as the sole basis for betting decisions.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
with tab2:
    st.title("Player Props Analyzer")
    
    all_teams = pd.DataFrame(teams.get_teams())
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Team")
        home_team = st.selectbox("Select Home Team", all_teams['full_name'], key='props_home')
    
    with col2:
        st.subheader("Away Team")
        away_team = st.selectbox("Select Away Team", all_teams['full_name'], key='props_away')
    
    if st.button("Load Player Props", use_container_width=True):
        with st.spinner("Fetching player props data..."):
            try:
                # Get player props data
                props_data = props_service.get_player_props(home_team, away_team)
                
                if not props_data:
                    st.warning("No player props data available for this matchup")
                else:
                    st.success(f"Found {len(props_data)} player props")
                    
                    # Get team IDs for prediction
                    home_team_id = all_teams[all_teams['full_name'] == home_team]['id'].iloc[0]
                    away_team_id = all_teams[all_teams['full_name'] == away_team]['id'].iloc[0]
                    
                    # Make prediction
                    with st.spinner("Generating player projections..."):
                        prediction = predictor.predict_game(home_team_id, away_team_id)
                        
                        # Calibrate prediction with betting data
                        calibrated_prediction = calibrator.calibrate_prediction(prediction, home_team, away_team)
                        
                        # Compare prediction with market props
                        props_comparison = props_service.compare_prediction_with_market(calibrated_prediction, home_team, away_team)
                    
                    # Process player predictions
                    home_players = calibrated_prediction.get('home_player_predictions', [])
                    away_players = calibrated_prediction.get('away_player_predictions', [])
                    
                    # Create DataFrames for player predictions
                    home_df = pd.DataFrame([
                        {
                            'Player': p['name'],
                            'Projected Points': p['points'],
                            'Projected Minutes': p['minutes'],
                            'Status': p['status']
                        } for p in home_players
                    ])
                    
                    away_df = pd.DataFrame([
                        {
                            'Player': p['name'],
                            'Projected Points': p['points'],
                            'Projected Minutes': p['minutes'],
                            'Status': p['status']
                        } for p in away_players
                    ])
                    
                    # Function to find player in props data
                    def find_prop_match(player_name):
                        if player_name in props_data:
                            return props_data[player_name]
                        else:
                            # Try partial match (last name)
                            player_last_name = player_name.split()[-1]
                            for pp in props_data.keys():
                                if player_last_name in pp:
                                    return props_data[pp]
                        return None
                    
                    # Create combined props list with projections
                    props_list = []
                    
                    # Process all props
                    for player_name, prop in props_data.items():
                        prop_dict = {
                            'Player': player_name,
                            'Line': prop['line'],
                            'Over Odds': prop['over_odds'],
                            'Under Odds': prop['under_odds'],
                        }
                        
                        # Try to find player in projection data
                        found_in_projection = False
                        for df in [home_df, away_df]:
                            for _, row in df.iterrows():
                                player_row_name = row['Player']
                                if player_name == player_row_name or player_name.split()[-1] in player_row_name:
                                    prop_dict['Projected Points'] = row['Projected Points']
                                    prop_dict['Edge'] = row['Projected Points'] - prop['line']
                                    prop_dict['Edge %'] = (abs(prop_dict['Edge']) / prop['line']) * 100 if prop['line'] > 0 else 0
                                    
                                    # Simple probability model
                                    if prop_dict['Edge'] > 0:
                                        prop_dict['Recommended Bet'] = 'OVER'
                                        prob = 0.5 + min(0.35, prop_dict['Edge'] * 0.07)
                                        prop_dict['Win Probability'] = prob
                                        prop_dict['EV'] = calculate_ev(prob, prop['over_odds'])
                                    else:
                                        prop_dict['Recommended Bet'] = 'UNDER'
                                        prob = 0.5 + min(0.35, abs(prop_dict['Edge']) * 0.07)
                                        prop_dict['Win Probability'] = prob
                                        prop_dict['EV'] = calculate_ev(prob, prop['under_odds'])
                                    
                                    found_in_projection = True
                                    break
                            if found_in_projection:
                                break
                        
                        # If not found in projection data, add placeholder
                        if not found_in_projection:
                            prop_dict['Projected Points'] = None
                            prop_dict['Edge'] = None
                            prop_dict['Edge %'] = None
                            prop_dict['Recommended Bet'] = 'N/A'
                            prop_dict['Win Probability'] = None
                            prop_dict['EV'] = None
                        
                        props_list.append(prop_dict)
                    
                    # Create DataFrame
                    props_df = pd.DataFrame(props_list)
                    
                    # Sort by expected value (EV) for best bets
                    if 'EV' in props_df.columns:
                        props_df = props_df.sort_values(by='EV', ascending=False)
                    
                    # Display props data
                    st.subheader("Player Props Analysis")
                    
                    # Create three sections: Best Bets, All Props, and Visualization
                    prop_tabs = st.tabs(["Best Bets", "All Props", "Visualization"])
                    
                    with prop_tabs[0]:
                        # Display top value bets (high EV)
                        st.markdown("### Top Value Bets")
                        
                        top_bets = props_df[(props_df['EV'] > 0.02) & (props_df['Projected Points'].notna())].head(5)
                        
                        if not top_bets.empty:
                            # Create cards for top bets
                            for idx, bet in top_bets.iterrows():
                                # Determine styling based on EV
                                ev_class = "positive-ev" if bet['EV'] > 0.05 else "neutral-ev" if bet['EV'] > 0 else "negative-ev"
                                ev_indicator_class = "ev-positive" if bet['EV'] > 0.05 else "ev-neutral" if bet['EV'] > 0 else "ev-negative"
                                
                                # Create card
                                st.markdown(f"""
                                <div class="prop-card" style='color: #343a40'>
                                    <h4>{bet['Player']}</h4>
                                    <div class="prop-details">
                                        <span><b>{bet['Line']}</b> Points ({bet['Recommended Bet']})</span>
                                        <span class="ev-indicator {ev_indicator_class}">EV: {bet['EV']*100:.1f}%</span>
                                    </div>
                                    <p>Projection: {bet['Projected Points']:.1f} points (Edge: {abs(bet['Edge']):.1f})</p>
                                    <div class="edge-meter">
                                        <div class="edge-fill" style="width: {min(100, bet['Edge %'])}%; background-color: {'#28a745' if bet['EV'] > 0.05 else '#ffc107' if bet['EV'] > 0 else '#dc3545'}"></div>
                                    </div>
                                    <p>Win Probability: {bet['Win Probability']*100:.1f}% | Odds: {bet['Over Odds'] if bet['Recommended Bet'] == 'OVER' else bet['Under Odds']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No high-value bets found based on model projections")
    
                    with prop_tabs[1]:
                        # Display all props with filtering options
                        st.markdown("### All Available Props")
                        
                        # Add filters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            min_ev = st.slider("Min. Expected Value", min_value=-0.2, max_value=0.2, value=-0.05, step=0.01, format="%.2f")
                        with col2:
                            bet_type = st.selectbox("Bet Type", ["All", "OVER", "UNDER"])
                        with col3:
                            min_edge = st.slider("Min. Edge %", min_value=0.0, max_value=20.0, value=2.0, step=0.5, format="%.1f")
                        
                        # Apply filters
                        filtered_df = props_df.copy()
                        
                        if filtered_df['EV'].notna().any():  # Check if EV column exists and has values
                            filtered_df = filtered_df[filtered_df['EV'] >= min_ev]
                        
                        if bet_type != "All":
                            filtered_df = filtered_df[filtered_df['Recommended Bet'] == bet_type]
                        
                        if filtered_df['Edge %'].notna().any():  # Check if Edge % column exists and has values
                            filtered_df = filtered_df[filtered_df['Edge %'] >= min_edge]
                        
                        # Display filtered dataframe
                        if not filtered_df.empty:
                            # Custom formatting and highlighting
                            def highlight_ev(val):
                                if val is None or pd.isna(val):
                                    return ''
                                if val > 0.05:
                                    return 'background-color: #d4edda; color: #343a40'
                                elif val > 0:
                                    return 'background-color: #fff3cd; color: #343a40'
                                else:
                                    return 'background-color: #f8d7da; color: #343a40'
                            
                            # Format the dataframe
                            display_cols = ['Player', 'Line', 'Projected Points', 'Edge', 'Edge %', 'Recommended Bet', 'Win Probability', 'EV']
                            styled_df = filtered_df[display_cols].style.format({
                                'Line': '{:.1f}',
                                'Projected Points': '{:.1f}',
                                'Edge': '{:.1f}',
                                'Edge %': '{:.1f}%',
                                'Win Probability': '{:.1%}',
                                'EV': '{:.1%}'
                            }).applymap(highlight_ev, subset=['EV'])
                            
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.warning("No props match the selected filters")
                    
                    with prop_tabs[2]:
                        # Create visualization comparing projected points to market lines
                        st.markdown("### Projections vs Market Lines")
                        
                        # Filter out props with no projection
                        viz_df = props_df[props_df['Projected Points'].notna()].copy()
                        
                        if not viz_df.empty:
                            # Create dataframe for visualization
                            viz_data = []
                            
                            for _, row in viz_df.iterrows():
                                viz_data.append({
                                    'Player': row['Player'],
                                    'Points': row['Projected Points'],
                                    'Type': 'Projected'
                                })
                                viz_data.append({
                                    'Player': row['Player'],
                                    'Points': row['Line'],
                                    'Type': 'Market Line'
                                })
                            
                            viz_plot_df = pd.DataFrame(viz_data)
                            
                            # Create bar chart
                            fig = px.bar(
                                viz_plot_df,
                                x='Player',
                                y='Points',
                                color='Type',
                                barmode='group',
                                title='Projected Points vs Market Lines',
                                color_discrete_map={'Projected': '#1f77b4', 'Market Line': '#ff7f0e'}
                            )
                            
                            fig.update_layout(
                                xaxis_title='Player',
                                yaxis_title='Points',
                                legend_title='Type',
                                height=500,
                                xaxis=dict(
                                    tickangle=-45,
                                    categoryorder='total descending'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create edge visualization
                            st.markdown("### Projection Edges")
                            
                            # Calculate absolute edge for visualization
                            viz_df['Absolute Edge'] = viz_df['Edge'].abs()
                            viz_df['Edge Direction'] = viz_df.apply(lambda x: 'OVER' if x['Edge'] > 0 else 'UNDER', axis=1)
                            
                            # Sort by absolute edge
                            viz_df = viz_df.sort_values(by='Absolute Edge', ascending=False)
                            
                            # Create bar chart for edges
                            fig = px.bar(
                                viz_df.head(10),
                                x='Player',
                                y='Absolute Edge',
                                color='Edge Direction',
                                title='Top 10 Projection Edges',
                                text='Absolute Edge',
                                color_discrete_map={'OVER': '#28a745', 'UNDER': '#dc3545'}
                            )
                            
                            fig.update_layout(
                                xaxis_title='Player',
                                yaxis_title='Absolute Edge (Points)',
                                height=500,
                                xaxis=dict(
                                    tickangle=-45
                                )
                            )
                            
                            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data available for visualization")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab3:
    st.header("Today's Games")
    
    todays_games = get_todays_games()
    if todays_games is not None and not todays_games.empty:
        unique_games = todays_games.groupby('GAME_ID').first().reset_index()
        num_games = len(unique_games)
        for i in range(0, num_games, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < num_games:
                    game = unique_games.iloc[i + j]
                    with cols[j]:
                        st.markdown('<div class="game-card">', unsafe_allow_html=True)
                        home_score = todays_games[
                            (todays_games['GAME_ID'] == game['GAME_ID']) & 
                            (todays_games['TEAM_ID'] == game['HOME_TEAM_ID'])
                        ]['PTS'].iloc[0]
                        away_score = todays_games[
                            (todays_games['GAME_ID'] == game['GAME_ID']) & 
                            (todays_games['TEAM_ID'] == game['VISITOR_TEAM_ID'])
                        ]['PTS'].iloc[0]
                        home_score = int(home_score) if pd.notna(home_score) else "N/A"
                        away_score = int(away_score) if pd.notna(away_score) else "N/A"
                        st.markdown(f"**{game['GAME_STATUS_TEXT']}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Away:")
                            st.write(get_team_name(game['VISITOR_TEAM_ID']))
                        with col2:
                            st.write("Score:")
                            st.write(f"{away_score}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Home:")
                            st.write(get_team_name(game['HOME_TEAM_ID']))
                        with col2:
                            st.write("Score:")
                            st.write(f"{home_score}")
                        
                        # Get betting lines
                        home_team_name = get_team_name(game['HOME_TEAM_ID'])
                        away_team_name = get_team_name(game['VISITOR_TEAM_ID'])
                        betting_lines = betting_service.get_betting_lines(home_team_name, away_team_name)
                        
                        if betting_lines:
                            st.markdown("<hr style='margin: 8px 0'>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class='vegas-data'>
                                <strong>Spread:</strong> {home_team_name.split()[-1]} {betting_lines.get('home_spread', 'N/A')}<br>
                                <strong>Total:</strong> {betting_lines.get('total', 'N/A')}<br>
                                <strong>ML:</strong> {away_team_name.split()[-1]} {betting_lines.get('away_moneyline', 'N/A')} | {home_team_name.split()[-1]} {betting_lines.get('home_moneyline', 'N/A')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add quick prediction button
                        if st.button(f"Quick Predict", key=f"quick_{game['GAME_ID']}"):
                            with st.spinner("Predicting..."):
                                try:
                                    prediction = predictor.predict_game(
                                        game['HOME_TEAM_ID'], 
                                        game['VISITOR_TEAM_ID']
                                    )
                                    
                                    # Calibrate prediction
                                    calibrated_prediction = calibrator.calibrate_prediction(
                                        prediction,
                                        home_team_name,
                                        away_team_name
                                    )
                                    
                                    # Get player props
                                    props_data = props_service.get_player_props(home_team_name, away_team_name)
                                    props_comparison = props_service.compare_prediction_with_market(calibrated_prediction, home_team_name, away_team_name)
                                    
                                    if 'error' not in calibrated_prediction or not calibrated_prediction['error']:
                                        # Determine if there's betting value
                                        value_indicator = ""
                                        value_details = ""
                                        
                                        if 'market_data' in calibrated_prediction and calibrated_prediction['market_data'].get('available', False):
                                            spread_value = calibrated_prediction['market_data'].get('spread_value_rating', 0)
                                            total_value = calibrated_prediction['market_data'].get('total_value_rating', 0)
                                            
                                            if spread_value >= 7 or total_value >= 7:
                                                value_indicator = " ⭐ HIGH VALUE"
                                                
                                                if spread_value >= 7:
                                                    # Calculate expected value
                                                    predicted_margin = calibrated_prediction['home_score'] - calibrated_prediction['away_score']
                                                    vegas_margin = -calibrated_prediction['market_data'].get('home_spread', 0)
                                                    spread_diff = predicted_margin - vegas_margin
                                                    
                                                    if spread_diff > 0:
                                                        value_details += f"Value on HOME spread ({calibrated_prediction['market_data'].get('home_spread')}).<br>"
                                                    else:
                                                        value_details += f"Value on AWAY spread ({-calibrated_prediction['market_data'].get('home_spread')}).<br>"
                                                
                                                if total_value >= 7:
                                                    predicted_total = calibrated_prediction['home_score'] + calibrated_prediction['away_score']
                                                    vegas_total = calibrated_prediction['market_data'].get('total', 0)
                                                    
                                                    if predicted_total > vegas_total:
                                                        value_details += f"Value on OVER {vegas_total}.<br>"
                                                    else:
                                                        value_details += f"Value on UNDER {vegas_total}.<br>"
                                        
                                        # Check for player prop value
                                        if props_comparison.get('available', False) and props_comparison.get('summary', {}).get('value_opportunities', 0) > 0:
                                            value_indicator += " 📊"
                                            value_details += f"{props_comparison['summary']['value_opportunities']} player prop value opportunities.<br>"
                                        
                                        st.success(f"Prediction: {home_team_name} {calibrated_prediction['home_score']:.1f} - {calibrated_prediction['away_score']:.1f} {away_team_name}{value_indicator}")
                                        
                                        if value_details:
                                            st.markdown(f"<div style='background-color: #d4edda; color: #343a40; padding: 8px; border-radius: 5px; font-size: 0.9em;'>{value_details}</div>", unsafe_allow_html=True)
                                        
                                        st.progress(calibrated_prediction['confidence'])
                                        st.write(f"Confidence: {calibrated_prediction['confidence']:.1%}")
                                    else:
                                        st.error("Prediction failed")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No games scheduled for today.")

# Add sidebar with app information
with st.sidebar:
    st.header("About the App")
    st.markdown("""
    This NBA Game Prediction System uses machine learning to forecast game outcomes based on:
    
    - Team performance metrics
    - Player statistics
    - Home court advantage
    - Rest days and travel
    - Recent form
    - Injury information
    - Betting market data
    
    The predictions combine team-level and player-level forecasts and are calibrated against live betting market data.
    """)
    
    st.header("Betting Analysis Features")
    st.markdown("""
    - **Expected Value (EV) Calculation**: Identifies bets with positive mathematical expectation
    - **Player Props Analysis**: Compares model projections with market lines
    - **Probability-Based Recommendations**: Suggests bets based on win probability and market odds
    - **Edge Finder**: Identifies the largest discrepancies between model and market
    """)
    
    st.header("How It Works")
    st.markdown("""
    1. Select home and away teams
    2. Click "Predict Game" or "Load Player Props"
    3. View predicted scores, player projections, and betting insights
    4. Filter and analyze value opportunities
    
    The system calculates expected value (EV) for each bet by comparing model-projected probabilities with implied market probabilities from the odds.
    """)
    
    st.header("Data Sources")
    st.markdown("""
    - NBA API for live game data
    - Team and player statistics
    - Historical matchup data
    - Injury tracking from multiple sources
    - Live betting odds and player props from DraftKings
    """)
    
    # Add manual reload options
    st.header("Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reload Injury Data"):
            with st.spinner("Updating injury data..."):
                collector.update_injury_data(force=True)
                st.success("Injury data updated!")
    
    with col2:
        if st.button("Reload Betting Data"):
            with st.spinner("Updating betting data..."):
                betting_service.update_betting_data(force=True)
                props_service.update_props_data(force=True)
                st.success("Betting data updated!")