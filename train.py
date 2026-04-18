#!/usr/bin/env python3
"""
NBA Prediction Model Training Pipeline.

Trains ensemble models (XGBoost, LightGBM, Random Forest) for:
- Team score prediction
- Player points prediction
- Player minutes prediction

Features temporal isolation (no data leakage), canonical feature schemas,
and comprehensive evaluation against recent games.

Usage:
    python train.py                     # Full training pipeline
    python train.py --evaluate-only     # Evaluate existing models
    python train.py --force             # Force retrain even if models exist
    python train.py --quick             # Quick train with fewer samples
    python train.py --verbose           # Detailed logging

Typical runtime: 15-45 minutes (depends on API rate limiting).
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import TRAINING, LEAGUE
from data_collector import DataCollector
from model_trainer import ModelTrainer

# ── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logger = logging.getLogger("train")
    logger.info(f"Log file: {log_file}")
    return logger


# ── Dependency Check ─────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    """Verify all required packages are installed."""
    required = [
        "pandas", "numpy", "sklearn", "xgboost", "lightgbm",
        "scipy", "nba_api", "joblib",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    # Check models directory
    Path(TRAINING.MODELS_DIR).mkdir(exist_ok=True)
    return True


# ── Training ─────────────────────────────────────────────────────────────────

def train_models(trainer: ModelTrainer, logger: logging.Logger) -> bool:
    """Run the full model training pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Seasons: {TRAINING.SEASONS}")
    logger.info(f"Season weights: {TRAINING.SEASON_WEIGHTS}")
    logger.info(f"Models directory: {TRAINING.MODELS_DIR}")

    start = time.time()

    success = trainer.train_ensemble_models()

    elapsed = time.time() - start
    minutes = elapsed / 60

    if success:
        logger.info(f"Training SUCCEEDED in {minutes:.1f} minutes")
        trainer.save_models()
        logger.info(f"Models saved to {TRAINING.MODELS_DIR}/")

        # Log model details
        logger.info(f"Team models: {list(trainer.team_models.keys())}")
        logger.info(f"Team ensemble weights: {trainer.team_ensemble_weights}")
        logger.info(f"Player points models: {list(trainer.player_models.get('points', {}).keys())}")
        logger.info(f"Player minutes models: {list(trainer.player_models.get('minutes', {}).keys())}")
        logger.info(f"Player ensemble weights: {trainer.player_ensemble_weights}")

        return True
    else:
        logger.error(f"Training FAILED after {minutes:.1f} minutes")
        return False


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_models(
    trainer: ModelTrainer,
    collector: DataCollector,
    logger: logging.Logger,
    n_games: int = 30,
) -> dict:
    """
    Evaluate trained models against recent completed games.

    For each game:
    1. Use the trained model to predict team scores
    2. Compare against actual results
    3. Compute aggregate accuracy metrics
    """
    logger.info("=" * 60)
    logger.info(f"EVALUATING MODELS ON {n_games} RECENT GAMES")
    logger.info("=" * 60)

    if not trainer.team_models:
        logger.error("No team models loaded — cannot evaluate")
        return {"error": "No models loaded"}

    try:
        from nba_api.stats.endpoints import LeagueGameLog
        from nba_api.stats.static import teams as nba_teams_static

        collector._rate_limit()
        endpoint = LeagueGameLog(
            season=LEAGUE.SEASON,
            season_type_all_star="Regular Season",
            headers=collector._get_headers(),
        )
        dfs = collector._safe_get_data_frames(endpoint)
        if not dfs or dfs[0].empty:
            return {"error": "Could not fetch game log"}

        game_log = dfs[0]
        grouped = game_log.groupby("GAME_ID")

        team_errors = []
        total_errors = []
        margin_errors = []
        winner_correct = []
        games_evaluated = 0

        for game_id, game_data in grouped:
            if games_evaluated >= n_games:
                break
            if len(game_data) != 2:
                continue

            try:
                home_row = game_data[game_data["MATCHUP"].str.contains("vs.", na=False)]
                away_row = game_data[game_data["MATCHUP"].str.contains("@", na=False)]
                if home_row.empty or away_row.empty:
                    continue

                home_id = int(home_row.iloc[0]["TEAM_ID"])
                away_id = int(away_row.iloc[0]["TEAM_ID"])
                actual_home = float(home_row.iloc[0]["PTS"])
                actual_away = float(away_row.iloc[0]["PTS"])

                # Get model predictions
                home_pred = trainer.predict_team_score(home_id, away_id, is_home=True)
                away_pred = trainer.predict_team_score(away_id, home_id, is_home=False)

                if not home_pred or not away_pred:
                    continue

                pred_home = home_pred["prediction"]
                pred_away = away_pred["prediction"]

                # Errors
                h_err = abs(pred_home - actual_home)
                a_err = abs(pred_away - actual_away)
                team_errors.extend([h_err, a_err])
                total_errors.append(h_err + a_err)
                margin_errors.append(abs((actual_home - actual_away) - (pred_home - pred_away)))

                actual_winner = "home" if actual_home > actual_away else "away"
                pred_winner = "home" if pred_home > pred_away else "away"
                winner_correct.append(actual_winner == pred_winner)

                games_evaluated += 1

                # Log individual games
                w = "✓" if actual_winner == pred_winner else "✗"
                home_name = collector.get_team_name(home_id).split()[-1]
                away_name = collector.get_team_name(away_id).split()[-1]
                logger.info(
                    f"  {w} {away_name} {actual_away:.0f} @ {home_name} {actual_home:.0f} "
                    f"| Pred: {pred_away:.1f} @ {pred_home:.1f} "
                    f"| Err: {h_err + a_err:.1f}"
                )

            except Exception as e:
                logger.warning(f"Eval error for game {game_id}: {e}")
                continue

        if not team_errors:
            return {"error": "No games evaluated successfully"}

        metrics = {
            "games_evaluated": games_evaluated,
            "team_score_mae": round(np.mean(team_errors), 2),
            "team_score_median_ae": round(np.median(team_errors), 2),
            "total_score_mae": round(np.mean(total_errors), 2),
            "margin_mae": round(np.mean(margin_errors), 2),
            "winner_accuracy_pct": round(np.mean(winner_correct) * 100, 1),
            "worst_total_error": round(max(total_errors), 1),
            "best_total_error": round(min(total_errors), 1),
        }

        # Performance grade
        mae = metrics["team_score_mae"]
        win_acc = metrics["winner_accuracy_pct"]
        if mae <= 8 and win_acc >= 68:
            grade = "A"
        elif mae <= 10 and win_acc >= 63:
            grade = "B"
        elif mae <= 12 and win_acc >= 58:
            grade = "C"
        else:
            grade = "D"
        metrics["grade"] = grade

        logger.info("")
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Games Evaluated:    {metrics['games_evaluated']}")
        logger.info(f"Team Score MAE:     {metrics['team_score_mae']:.2f} pts")
        logger.info(f"Team Score Median:  {metrics['team_score_median_ae']:.2f} pts")
        logger.info(f"Total Score MAE:    {metrics['total_score_mae']:.2f} pts")
        logger.info(f"Margin MAE:         {metrics['margin_mae']:.2f} pts")
        logger.info(f"Winner Accuracy:    {metrics['winner_accuracy_pct']:.1f}%")
        logger.info(f"Best Game Error:    {metrics['best_total_error']:.1f} pts")
        logger.info(f"Worst Game Error:   {metrics['worst_total_error']:.1f} pts")
        logger.info(f"Grade:              {grade}")
        logger.info("=" * 60)

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}


# ── Test Prediction ──────────────────────────────────────────────────────────

def test_prediction(trainer: ModelTrainer, collector: DataCollector, logger: logging.Logger):
    """Quick smoke test: predict a single matchup."""
    from nba_api.stats.static import teams as nba_teams_static

    all_teams = nba_teams_static.get_teams()
    if len(all_teams) < 2:
        logger.error("Could not load team data")
        return

    # Pick two well-known teams
    test_pairs = [
        ("Los Angeles Lakers", "Golden State Warriors"),
        ("Boston Celtics", "Milwaukee Bucks"),
    ]

    for home_name, away_name in test_pairs:
        home_info = next((t for t in all_teams if t["full_name"] == home_name), None)
        away_info = next((t for t in all_teams if t["full_name"] == away_name), None)
        if not home_info or not away_info:
            continue

        home_id = home_info["id"]
        away_id = away_info["id"]

        logger.info(f"\nTest: {away_name} @ {home_name}")

        home_pred = trainer.predict_team_score(home_id, away_id, is_home=True)
        away_pred = trainer.predict_team_score(away_id, home_id, is_home=False)

        if home_pred and away_pred:
            h = home_pred["prediction"]
            a = away_pred["prediction"]
            logger.info(f"  Score: {away_name.split()[-1]} {a:.1f} @ {home_name.split()[-1]} {h:.1f}")
            logger.info(f"  Total: {h + a:.1f}")
            logger.info(f"  Margin: {h - a:+.1f}")
            logger.info(f"  Home std: {home_pred['std']:.2f}, Away std: {away_pred['std']:.2f}")
            logger.info(f"  Individual predictions: {home_pred['individual_predictions']}")
        else:
            logger.warning(f"  Prediction failed for this matchup")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Model Training")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if models exist")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only evaluate existing models")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run smoke test predictions")
    parser.add_argument("--eval-games", type=int, default=30,
                        help="Number of games for evaluation (default: 30)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--clean-cache", action="store_true",
                        help="Clean data cache before training")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    logger.info("NBA Prediction Model Training Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'evaluate-only' if args.evaluate_only else 'test-only' if args.test_only else 'full training'}")

    if not check_dependencies():
        return 1

    # Initialize
    collector = DataCollector()
    trainer = ModelTrainer(collector)

    if args.clean_cache:
        logger.info("Cleaning data cache...")
        collector.cleanup_cache()

    # ── Evaluate Only ────────────────────────────────────────────────
    if args.evaluate_only:
        loaded = trainer.load_models()
        if not loaded:
            logger.error("No models found to evaluate")
            return 1
        evaluate_models(trainer, collector, logger, args.eval_games)
        return 0

    # ── Test Only ────────────────────────────────────────────────────
    if args.test_only:
        loaded = trainer.load_models()
        if not loaded:
            logger.error("No models found to test")
            return 1
        test_prediction(trainer, collector, logger)
        return 0

    # ── Full Pipeline ────────────────────────────────────────────────

    # Check if models already exist
    models_exist = os.path.exists(f"{TRAINING.MODELS_DIR}/team_xgb.joblib")
    if models_exist and not args.force:
        logger.info("Models already exist. Use --force to retrain.")
        logger.info("Loading existing models for evaluation...")
        trainer.load_models()
    else:
        # Train
        success = train_models(trainer, logger)
        if not success:
            logger.error("Training failed")
            return 1

    # Evaluate
    logger.info("")
    metrics = evaluate_models(trainer, collector, logger, args.eval_games)
    if "error" in metrics:
        logger.warning(f"Evaluation had issues: {metrics['error']}")

    # Smoke test
    logger.info("")
    test_prediction(trainer, collector, logger)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models directory: {TRAINING.MODELS_DIR}/")
    logger.info(f"Model files:")
    if os.path.exists(TRAINING.MODELS_DIR):
        for f in sorted(os.listdir(TRAINING.MODELS_DIR)):
            size = os.path.getsize(f"{TRAINING.MODELS_DIR}/{f}")
            logger.info(f"  {f:40s} {size / 1024:.0f} KB")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
