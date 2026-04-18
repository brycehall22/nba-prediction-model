"""
NBA Prediction System — Streamlit Frontend (Rebuilt)

Design direction: Dark, editorial sports-analytics dashboard.
Key improvements:
  - Slate overview: see all games + value signals at a glance
  - Information hierarchy: best bets surface first
  - Single shared EV calculation path (ev_engine)
  - No duplicated logic between tabs
  - Clean component-based layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import teams
from datetime import datetime
import logging

# Our modules
from config import UI, BETTING, PREDICTION
from data_collector import DataCollector
from ev_engine import EVEngine
from betting_data import BettingDataService, PlayerPropsService

logging.basicConfig(level=logging.INFO)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Edge Finder",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ─────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Header ─────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(255,255,255,0.08);
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header .subtitle {
    font-size: 0.82rem;
    opacity: 0.5;
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Metric Cards ───────────────────────────── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
}
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    opacity: 0.45;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
}
.metric-card .detail {
    font-size: 0.78rem;
    opacity: 0.6;
    margin-top: 2px;
}

/* ── EV Badges ──────────────────────────────── */
.ev-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
}
.ev-positive { background: rgba(34,197,94,0.15); color: #22c55e; }
.ev-neutral  { background: rgba(250,204,21,0.12); color: #facc15; }
.ev-negative { background: rgba(239,68,68,0.12);  color: #ef4444; }

/* ── Game Slate Cards ───────────────────────── */
.slate-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.slate-card:hover {
    border-color: rgba(255,255,255,0.18);
}
.slate-teams {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.slate-team {
    font-weight: 700;
    font-size: 1.05rem;
}
.slate-vs {
    font-size: 0.75rem;
    opacity: 0.3;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.slate-line {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    opacity: 0.55;
    padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* ── Props Table ────────────────────────────── */
.props-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    gap: 16px;
}
.props-player {
    flex: 2;
    font-weight: 600;
}
.props-stat {
    flex: 1;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* ── Confidence Bar ─────────────────────────── */
.conf-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* ── Section Headings ───────────────────────── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Disclaimer ─────────────────────────────── */
.disclaimer {
    text-align: center;
    font-size: 0.72rem;
    opacity: 0.3;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)


# ── Initialize Services (cached) ────────────────────────────────────────────

@st.cache_resource
def init_services():
    collector = DataCollector()
    ev = EVEngine()
    odds = BettingDataService()
    props = PlayerPropsService()
    return collector, ev, odds, props

collector, ev_engine, odds_service, props_service = init_services()
all_teams_df = pd.DataFrame(teams.get_teams())


# ── Helper Functions ─────────────────────────────────────────────────────────

def ev_badge(ev_pct: float) -> str:
    """Return an HTML EV badge."""
    if ev_pct > BETTING.EV_THRESHOLD_PCT:
        cls = "ev-positive"
    elif ev_pct > 0:
        cls = "ev-neutral"
    else:
        cls = "ev-negative"
    return f'<span class="ev-badge {cls}">{ev_pct:+.1f}% EV</span>'


def confidence_bar(conf: float, color: str = "#22c55e") -> str:
    pct = max(0, min(100, conf * 100))
    return f"""
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>
    </div>"""


def get_simple_prediction(home_id: int, away_id: int) -> dict:
    """
    Get a statistical prediction for a matchup.
    Uses team stats + adjustments (no ML models required).
    """
    home_stats = collector.get_team_stats(home_id)
    away_stats = collector.get_team_stats(away_id)

    home_base = home_stats.get("pts_avg", 113) if home_stats else 113
    away_base = away_stats.get("pts_avg", 113) if away_stats else 113

    # Opponent defense adjustment
    home_def = home_stats.get("off_rating", 115) if home_stats else 115
    away_def = away_stats.get("off_rating", 115) if away_stats else 115

    home_score = home_base + PREDICTION.BACK_TO_BACK_PENALTY * 0  # placeholder
    away_score = away_base

    # Home court
    home_score += 3.5

    home_std = home_stats.get("pts_std", 12) if home_stats else 12
    away_std = away_stats.get("pts_std", 12) if away_stats else 12

    # Recent form adjustment
    if home_stats and "recent_form" in home_stats:
        trend = home_stats["recent_form"].get("pts_trend", 0)
        home_score += np.clip(trend * 2, -5, 5)
    if away_stats and "recent_form" in away_stats:
        trend = away_stats["recent_form"].get("pts_trend", 0)
        away_score += np.clip(trend * 2, -5, 5)

    win_prob = ev_engine.win_probability(home_score, home_std, away_score, away_std)

    return {
        "home_score": round(home_score, 1),
        "away_score": round(away_score, 1),
        "home_std": home_std,
        "away_std": away_std,
        "total": round(home_score + away_score, 1),
        "margin": round(home_score - away_score, 1),
        "home_win_prob": round(win_prob, 3),
        "confidence": round(min(0.85, 0.5 + abs(home_score - away_score) / 50), 3),
    }


def get_player_projections(team_id: int, opponent_id: int, is_home: bool) -> list:
    """Get player-level scoring projections for a team."""
    roster = collector.get_team_players(team_id)
    projections = []

    for player in roster[:UI.MAX_PLAYERS_DISPLAY]:
        stats = collector.get_player_stats(player["id"])
        if not stats or stats.get("status") == "inactive":
            continue

        pts = stats.get("pts_avg", 0)
        mins = stats.get("minutes_avg", 0)
        std = stats.get("pts_std", ev_engine.estimate_points_std(pts, mins))

        # Home/away adjustment
        if is_home:
            ha = stats.get("situational", {}).get("home_away", {}).get("home_advantage", 0)
            pts += np.clip(ha * 0.5, -2, 2)

        # Recent form
        rf = stats.get("recent_form", {})
        if rf:
            trend = rf.get("trend_l10", 0)
            pts += np.clip(trend, -2, 2)

        projections.append({
            "name": player["full_name"],
            "points": round(max(0, pts), 1),
            "points_std": round(std, 2),
            "minutes": round(mins, 1),
            "role": stats.get("role", "bench"),
        })

    projections.sort(key=lambda x: x["points"], reverse=True)
    return projections


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <h1>🏀 NBA Edge Finder</h1>
    <span class="subtitle">{datetime.now().strftime("%b %d, %Y")}</span>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_slate, tab_predict, tab_props = st.tabs(["📋 Tonight's Slate", "🎯 Game Predictor", "📊 Player Props"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Tonight's Slate Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_slate:
    st.markdown('<div class="section-heading">Games & Value Signals</div>', unsafe_allow_html=True)

    # Read odds file to show today's games
    try:
        odds_df = pd.read_csv(UI.ODDS_FILE)
        if odds_df.empty:
            st.info("No games found in odds data. Upload a fresh odds CSV.")
        else:
            for _, row in odds_df.iterrows():
                away_name = str(row.get("Away Team", ""))
                home_name = str(row.get("Home Team", ""))
                spread = row.get("Home Spread", "")
                total = row.get("Total", "")
                home_ml = row.get("Home Moneyline", "")
                away_ml = row.get("Away Moneyline", "")

                st.markdown(f"""
                <div class="slate-card">
                    <div class="slate-teams">
                        <span class="slate-team">{away_name}</span>
                        <span class="slate-vs">at</span>
                        <span class="slate-team">{home_name}</span>
                    </div>
                    <div class="slate-line">
                        <span>Spread: {spread}</span>
                        <span>Total: {total}</span>
                        <span>ML: {away_ml} / {home_ml}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("No odds file found. Place `draftkings_nba_odds.csv` in the working directory.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Game Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("Home Team", all_teams_df["full_name"], key="pred_home")
    with col_away:
        away_team = st.selectbox("Away Team", all_teams_df["full_name"], key="pred_away")

    home_id = int(all_teams_df[all_teams_df["full_name"] == home_team]["id"].iloc[0])
    away_id = int(all_teams_df[all_teams_df["full_name"] == away_team]["id"].iloc[0])

    if st.button("Run Prediction", use_container_width=True, type="primary"):
        with st.spinner("Analyzing matchup..."):
            pred = get_simple_prediction(home_id, away_id)

            # Get market data
            market = odds_service.get_betting_lines(home_team, away_team)

            # Calibrate with market if available
            if market.get("available"):
                cal = ev_engine.calibrate_with_market(
                    pred["home_score"], pred["away_score"],
                    pred["home_std"], pred["away_std"],
                    pred["confidence"], market,
                )
                display_home = cal["home_score"]
                display_away = cal["away_score"]
                ev_metrics = cal["ev_metrics"]
                regression = cal.get("regression_applied", False)
            else:
                display_home = pred["home_score"]
                display_away = pred["away_score"]
                ev_metrics = {"available": False}
                regression = False

            total = display_home + display_away
            margin = display_home - display_away
            winner = home_team if margin > 0 else away_team

            # ── Score Summary ────────────────────────────────────────
            st.markdown('<div class="section-heading">Predicted Score</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Home</div>
                    <div class="value">{display_home}</div>
                    <div class="detail">{home_team.split()[-1]}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Away</div>
                    <div class="value">{display_away}</div>
                    <div class="detail">{away_team.split()[-1]}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Total</div>
                    <div class="value">{total:.1f}</div>
                    <div class="detail">Combined</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                wp = pred["home_win_prob"]
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Win Prob</div>
                    <div class="value">{wp:.0%}</div>
                    <div class="detail">{winner.split()[-1]}</div>
                </div>""", unsafe_allow_html=True)

            # ── EV Analysis ──────────────────────────────────────────
            if ev_metrics.get("available"):
                st.markdown('<div class="section-heading">Betting Edge Analysis</div>', unsafe_allow_html=True)

                ev_c1, ev_c2, ev_c3 = st.columns(3)

                spread_data = ev_metrics["spread"]
                with ev_c1:
                    best_side = spread_data["best_bet"]
                    line = spread_data["line"]
                    best_ev = spread_data["best_ev"]
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Spread</div>
                        <div class="value">{best_side} {abs(line)}</div>
                        <div class="detail">{ev_badge(best_ev)}</div>
                    </div>""", unsafe_allow_html=True)

                total_data = ev_metrics["total"]
                with ev_c2:
                    best_side = total_data["best_bet"]
                    line = total_data["line"]
                    best_ev = total_data["best_ev"]
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Total</div>
                        <div class="value">{best_side} {line}</div>
                        <div class="detail">{ev_badge(best_ev)}</div>
                    </div>""", unsafe_allow_html=True)

                ml_data = ev_metrics["moneyline"]
                with ev_c3:
                    best_side = ml_data["best_bet"]
                    best_ev = ml_data["best_ev"]
                    odds_val = ml_data["home_odds"] if best_side == "HOME" else ml_data["away_odds"]
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Moneyline</div>
                        <div class="value">{best_side} ({odds_val})</div>
                        <div class="detail">{ev_badge(best_ev)}</div>
                    </div>""", unsafe_allow_html=True)

                # Best overall
                best = ev_metrics.get("best_overall", {})
                if best.get("ev", 0) > BETTING.EV_THRESHOLD_PCT:
                    st.success(f"**Best Bet:** {best['bet']} — {best['ev']:+.1f}% EV")

                if regression:
                    st.caption("Prediction blended with market consensus (dynamic regression).")

            else:
                st.info("No betting lines available for this matchup. Showing raw model prediction.")

            # ── Player Projections (sum-constrained) ─────────────────
            st.markdown('<div class="section-heading">Player Projections</div>', unsafe_allow_html=True)

            p_col1, p_col2 = st.columns(2)

            with p_col1:
                st.markdown(f"**{home_team}**")
                home_players = get_player_projections(home_id, away_id, is_home=True)
                if home_players:
                    # Normalize player projections to sum to team total
                    home_players = ev_engine.normalize_player_projections(
                        home_players, display_home, pts_key="points", std_key="points_std",
                    )
                    raw_sum = sum(p["points"] for p in home_players)
                    for p in home_players[:8]:
                        st.markdown(f"""<div class="props-row">
                            <span class="props-player">{p['name']}</span>
                            <span class="props-stat">{p['points']} pts</span>
                            <span class="props-stat">{p['minutes']} min</span>
                        </div>""", unsafe_allow_html=True)
                    st.caption(f"Player totals normalized to team prediction ({display_home} pts)")
                else:
                    st.caption("Roster data unavailable.")

            with p_col2:
                st.markdown(f"**{away_team}**")
                away_players = get_player_projections(away_id, home_id, is_home=False)
                if away_players:
                    away_players = ev_engine.normalize_player_projections(
                        away_players, display_away, pts_key="points", std_key="points_std",
                    )
                    for p in away_players[:8]:
                        st.markdown(f"""<div class="props-row">
                            <span class="props-player">{p['name']}</span>
                            <span class="props-stat">{p['points']} pts</span>
                            <span class="props-stat">{p['minutes']} min</span>
                        </div>""", unsafe_allow_html=True)
                    st.caption(f"Player totals normalized to team prediction ({display_away} pts)")
                else:
                    st.caption("Roster data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Player Props
# ══════════════════════════════════════════════════════════════════════════════
with tab_props:
    st.markdown('<div class="section-heading">Player Props Value Finder</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        props_home = st.selectbox("Home Team", all_teams_df["full_name"], key="props_home")
    with pc2:
        props_away = st.selectbox("Away Team", all_teams_df["full_name"], key="props_away")

    if st.button("Analyze Props", use_container_width=True, type="primary"):
        with st.spinner("Evaluating player props..."):
            props_home_id = int(all_teams_df[all_teams_df["full_name"] == props_home]["id"].iloc[0])
            props_away_id = int(all_teams_df[all_teams_df["full_name"] == props_away]["id"].iloc[0])

            # Get props from CSV
            props_data = props_service.get_player_props(props_home, props_away)

            if not props_data:
                st.warning("No player props found for this matchup.")
            else:
                # Get player projections and normalize to team totals
                pred_for_norm = get_simple_prediction(props_home_id, props_away_id)
                home_projs = get_player_projections(props_home_id, props_away_id, is_home=True)
                away_projs = get_player_projections(props_away_id, props_home_id, is_home=False)

                # Apply sum constraint so player projections are internally consistent
                home_projs = ev_engine.normalize_player_projections(
                    home_projs, pred_for_norm["home_score"], pts_key="points", std_key="points_std",
                )
                away_projs = ev_engine.normalize_player_projections(
                    away_projs, pred_for_norm["away_score"], pts_key="points", std_key="points_std",
                )

                all_projs = {p["name"]: p for p in home_projs + away_projs}

                # Match props to projections and evaluate
                evaluations = []
                for player_name, prop in props_data.items():
                    # Find matching projection (exact or last-name match)
                    proj = all_projs.get(player_name)
                    if not proj:
                        last = player_name.split()[-1].lower()
                        for pn, pv in all_projs.items():
                            if pn.split()[-1].lower() == last:
                                proj = pv
                                break

                    if proj:
                        ev_result = ev_engine.evaluate_player_prop(
                            proj["points"], proj["points_std"],
                            prop["line"], prop["over_odds"], prop["under_odds"],
                        )
                        ev_result["player"] = player_name
                        ev_result["proj_pts"] = proj["points"]
                        ev_result["proj_min"] = proj["minutes"]
                        evaluations.append(ev_result)

                if not evaluations:
                    st.info("Could not match any props to player projections.")
                else:
                    # Sort by best EV
                    evaluations.sort(key=lambda x: x["best_ev"], reverse=True)

                    # Show value bets first
                    value_bets = [e for e in evaluations if e["best_ev"] > BETTING.EV_THRESHOLD_PCT]

                    if value_bets:
                        st.markdown(f'<div class="section-heading">🔥 Value Bets ({len(value_bets)} found)</div>',
                                    unsafe_allow_html=True)
                        for e in value_bets:
                            badge = ev_badge(e["best_ev"])
                            edge = e["edge_pts"]
                            st.markdown(f"""<div class="props-row" style="border-color:rgba(34,197,94,0.2)">
                                <span class="props-player">{e['player']}</span>
                                <span class="props-stat">{e['best_bet']} {e['line']}</span>
                                <span class="props-stat">Proj: {e['proj_pts']}</span>
                                <span class="props-stat">Edge: {edge} pts</span>
                                <span class="props-stat">{badge}</span>
                            </div>""", unsafe_allow_html=True)

                        # ── Correlated Bet Warnings ──────────────────────
                        # Build game-level bets from market data for correlation check
                        game_market = odds_service.get_betting_lines(props_home, props_away)
                        game_bets_for_corr = []
                        if game_market.get("available"):
                            # Check which game bets would also be recommended
                            pred_for_corr = get_simple_prediction(props_home_id, props_away_id)
                            cal_for_corr = ev_engine.calibrate_with_market(
                                pred_for_corr["home_score"], pred_for_corr["away_score"],
                                pred_for_corr["home_std"], pred_for_corr["away_std"],
                                pred_for_corr["confidence"], game_market,
                            )
                            evm = cal_for_corr.get("ev_metrics", {})
                            if evm.get("available"):
                                if evm["spread"]["best_ev"] > BETTING.EV_THRESHOLD_PCT:
                                    game_bets_for_corr.append({
                                        "type": "SPREAD",
                                        "bet": evm["spread"]["best_bet"],
                                        "ev": evm["spread"]["best_ev"],
                                        "kelly_fraction": ev_engine.kelly_fraction(
                                            evm["spread"]["home_cover_prob"]
                                            if evm["spread"]["best_bet"] == "HOME"
                                            else 1 - evm["spread"]["home_cover_prob"],
                                            game_market.get("home_spread_odds", -110)
                                            if evm["spread"]["best_bet"] == "HOME"
                                            else game_market.get("away_spread_odds", -110),
                                        ),
                                    })
                                if evm["total"]["best_ev"] > BETTING.EV_THRESHOLD_PCT:
                                    game_bets_for_corr.append({
                                        "type": "TOTAL",
                                        "bet": evm["total"]["best_bet"],
                                        "ev": evm["total"]["best_ev"],
                                        "kelly_fraction": ev_engine.kelly_fraction(
                                            evm["total"]["over_prob"]
                                            if evm["total"]["best_bet"] == "OVER"
                                            else evm["total"]["under_prob"],
                                            game_market.get("total_over_odds", -110)
                                            if evm["total"]["best_bet"] == "OVER"
                                            else game_market.get("total_under_odds", -110),
                                        ),
                                    })

                        # Build prop bets list for correlation check
                        prop_bets_for_corr = []
                        home_proj_names = {p["name"] for p in home_projs}
                        for e in value_bets:
                            team_side = "home" if e["player"] in home_proj_names or any(
                                e["player"].split()[-1].lower() == pn.split()[-1].lower()
                                for pn in home_proj_names
                            ) else "away"
                            prop_bets_for_corr.append({
                                "player": e["player"],
                                "team": team_side,
                                "best_bet": e["best_bet"],
                                "ev": e["best_ev"],
                                "kelly_fraction": e["kelly_fraction"],
                            })

                        corr_warnings = ev_engine.detect_correlated_bets(
                            game_bets_for_corr, prop_bets_for_corr,
                        )

                        if corr_warnings:
                            st.markdown(
                                '<div class="section-heading">⚠️ Correlation Warnings</div>',
                                unsafe_allow_html=True,
                            )
                            for w in corr_warnings:
                                bets_str = " + ".join(w["group"])
                                st.markdown(f"""<div class="props-row" style="border-color:rgba(250,204,21,0.3);background:rgba(250,204,21,0.04)">
                                    <div style="flex:1">
                                        <div style="font-weight:600;margin-bottom:4px">
                                            {bets_str}
                                        </div>
                                        <div style="font-size:0.8rem;opacity:0.7">
                                            {w['reason']}
                                        </div>
                                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;margin-top:4px;opacity:0.8">
                                            Correlation: {w['correlation']:.0%} · 
                                            Kelly reduced by {w['reduction_pct']:.0f}% 
                                            ({w['original_kelly_total']:.2%} → {w['adjusted_kelly_total']:.2%})
                                        </div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                    else:
                        st.info("No high-value props found for this matchup based on model projections.")

                    # Full table
                    st.markdown('<div class="section-heading">All Props Analysis</div>', unsafe_allow_html=True)

                    table_data = []
                    for e in evaluations:
                        table_data.append({
                            "Player": e["player"],
                            "Line": e["line"],
                            "Projection": e["proj_pts"],
                            "Edge (pts)": e["edge_pts"],
                            "Best Bet": e["best_bet"],
                            "Over Prob": f"{e['over']['prob']:.0%}",
                            "Under Prob": f"{e['under']['prob']:.0%}",
                            "EV %": e["best_ev"],
                            "Confidence": e["confidence"],
                            "Rating": e["value_rating"],
                        })

                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df.style.background_gradient(subset=["EV %"], cmap="RdYlGn", vmin=-10, vmax=15)
                            .format({"EV %": "{:+.1f}%", "Confidence": "{:.0%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Visualization
                    st.markdown('<div class="section-heading">Projections vs Lines</div>', unsafe_allow_html=True)

                    fig = go.Figure()
                    names = [e["player"].split()[-1] for e in evaluations[:15]]
                    projs = [e["proj_pts"] for e in evaluations[:15]]
                    lines = [e["line"] for e in evaluations[:15]]

                    fig.add_trace(go.Bar(
                        x=names, y=projs, name="Projection",
                        marker_color="rgba(99,102,241,0.7)",
                        text=[f"{v:.1f}" for v in projs], textposition="auto",
                    ))
                    fig.add_trace(go.Scatter(
                        x=names, y=lines, name="Market Line",
                        mode="markers+lines",
                        marker=dict(color="#facc15", size=10, symbol="diamond"),
                        line=dict(color="rgba(250,204,21,0.3)", dash="dot"),
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=420,
                        margin=dict(t=30, b=60),
                        legend=dict(orientation="h", y=1.08),
                        xaxis=dict(tickangle=-45),
                        yaxis_title="Points",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    Projections are for informational and entertainment purposes only.
    Not financial advice. Past performance does not guarantee future results.
    Please gamble responsibly.
</div>
""", unsafe_allow_html=True)
