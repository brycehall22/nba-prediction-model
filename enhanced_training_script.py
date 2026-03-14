#!/usr/bin/env python3
"""
Enhanced NBA Prediction Model Training Script

This script provides a complete training pipeline for the enhanced NBA prediction system
with advanced feature engineering, ensemble models, and comprehensive evaluation.

Usage:
    python enhanced_training_script.py [options]

Options:
    --full-retrain: Complete model retraining from scratch
    --evaluate-only: Only evaluate existing models
    --clean-cache: Clean data cache before training
    --verbose: Enable verbose logging
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Import our enhanced modules
from predictor import EnhancedNBAPredictor
from data_collector import EnhancedDataCollector
from model_trainer import EnhancedModelTrainer

def setup_logging(verbose: bool = False):
    """Setup comprehensive logging"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def validate_environment():
    """Validate that all required dependencies are available"""
    logger = logging.getLogger(__name__)
    
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import lightgbm
        import scipy
        import nba_api
        import sqlite3
        
        logger.info("All required dependencies found")
        
        # Check data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Check models directory
        models_dir = Path("enhanced_models")
        models_dir.mkdir(exist_ok=True)
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return False

def clean_cache(collector: EnhancedDataCollector):
    """Clean old cached data"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Cleaning cached data...")
        collector.cleanup_cache(days_old=3)  # Clean data older than 3 days
        logger.info("Cache cleaning completed")
        
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")

def train_models(predictor: EnhancedNBAPredictor, force_retrain: bool = False):
    """Train the enhanced models"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training...")
        start_time = time.time()
        
        # Train models
        success = predictor.train_models(force_retrain=force_retrain)
        
        if success:
            training_time = time.time() - start_time
            logger.info(f"Model training completed successfully in {training_time:.1f} seconds")
            
            # Get model information
            model_info = predictor.get_model_info()
            logger.info(f"Models loaded: {model_info['models_loaded']}")
            logger.info(f"Team models: {model_info['model_types']['team_models']}")
            logger.info(f"Player models: {model_info['model_types']['player_models']}")
            
            return True
        else:
            logger.error("Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def evaluate_models(predictor: EnhancedNBAPredictor, n_games: int = 30):
    """Evaluate model performance"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Evaluating models on {n_games} recent games...")
        
        evaluation_results = predictor.evaluate_predictions(n_games=n_games)
        
        if 'error' in evaluation_results:
            logger.error(f"Evaluation failed: {evaluation_results['error']}")
            return False
        
        metrics = evaluation_results['metrics']
        
        logger.info("=== MODEL PERFORMANCE METRICS ===")
        logger.info(f"Games Evaluated: {metrics['games_evaluated']}")
        logger.info(f"Team Score MAE: {metrics['team_score_mae']:.2f} points")
        logger.info(f"Total Score MAE: {metrics['total_score_mae']:.2f} points")
        logger.info(f"Margin MAE: {metrics['margin_mae']:.2f} points")
        logger.info(f"Winner Accuracy: {metrics['winner_accuracy']:.1f}%")
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.3f}")
        logger.info("==================================")
        
        # Performance benchmarks
        benchmarks = {
            'excellent_mae': 8.0,
            'good_mae': 10.0,
            'acceptable_mae': 12.0,
            'excellent_winner_acc': 70.0,
            'good_winner_acc': 65.0,
            'acceptable_winner_acc': 60.0
        }
        
        # Assess performance
        mae = metrics['team_score_mae']
        winner_acc = metrics['winner_accuracy']
        
        if mae <= benchmarks['excellent_mae'] and winner_acc >= benchmarks['excellent_winner_acc']:
            performance_level = "EXCELLENT"
        elif mae <= benchmarks['good_mae'] and winner_acc >= benchmarks['good_winner_acc']:
            performance_level = "GOOD"
        elif mae <= benchmarks['acceptable_mae'] and winner_acc >= benchmarks['acceptable_winner_acc']:
            performance_level = "ACCEPTABLE"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        logger.info(f"Overall Performance Level: {performance_level}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return False

def test_prediction_pipeline(predictor: EnhancedNBAPredictor):
    """Test the prediction pipeline with sample predictions"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing prediction pipeline...")
        
        # Get some team IDs for testing
        from nba_api.stats.static import teams
        nba_teams = teams.get_teams()
        
        if len(nba_teams) < 2:
            logger.error("Could not get team data for testing")
            return False
        
        # Test with a few team matchups
        test_matchups = [
            (nba_teams[0]['id'], nba_teams[1]['id']),  # First two teams
            (nba_teams[2]['id'], nba_teams[3]['id']),  # Next two teams
        ]
        
        for i, (home_id, away_id) in enumerate(test_matchups):
            logger.info(f"Testing prediction {i+1}: {teams.get_teams()[home_id-1]['full_name']} vs {teams.get_teams()[away_id-1]['full_name']}")
            
            # Make prediction
            start_time = time.time()
            prediction = predictor.predict_game(home_id, away_id, detailed=True)
            prediction_time = time.time() - start_time
            
            if 'error' in prediction:
                logger.error(f"Prediction failed: {prediction['error']}")
                continue
            
            # Log prediction summary
            preds = prediction['predictions']
            uncertainty = prediction['uncertainty']
            
            logger.info(f"  Predicted Score: {preds['home_score']} - {preds['away_score']}")
            logger.info(f"  Total: {preds['total_score']}")
            logger.info(f"  Home Win Prob: {preds['home_win_probability']:.1%}")
            logger.info(f"  Confidence: {uncertainty['overall_confidence']:.3f}")
            logger.info(f"  Prediction Time: {prediction_time:.2f}s")
            
            # Test player predictions
            player_preds = prediction.get('player_predictions', {})
            if player_preds:
                home_players = player_preds.get('home_players', [])
                if home_players:
                    top_scorer = home_players[0]
                    logger.info(f"  Top Scorer: {top_scorer['name']} - {top_scorer['points']} pts")
        
        logger.info("Prediction pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing prediction pipeline: {e}")
        return False

def generate_training_report(predictor: EnhancedNBAPredictor):
    """Generate a comprehensive training report"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Generating training report...")
        
        model_info = predictor.get_model_info()
        
        report = f"""
=== NBA PREDICTION MODEL TRAINING REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL CONFIGURATION:
- Enhanced Models: {model_info['models_loaded']}
- Team Models: {', '.join(model_info['model_types']['team_models'])}
- Player Models: {model_info['model_types']['player_models']}

PERFORMANCE METRICS:
- Team Score MAE: {model_info['performance'].get('team_mae', 'N/A')}
- Last Evaluation: {model_info['performance'].get('last_evaluation', 'N/A')}

DATA SOURCES:
- Enhanced Features: {model_info['data_sources']['enhanced_features']}
- Advanced Stats: {model_info['data_sources']['advanced_stats']}
- Situational Analysis: {model_info['data_sources']['situational_analysis']}
- Caching Enabled: {model_info['data_sources']['caching_enabled']}

ENSEMBLE WEIGHTS:
Team Models: {model_info['model_types']['ensemble_weights'].get('team', {})}

============================================
"""
        
        # Save report to file
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Training report saved to: {report_file}")
        print(report)  # Also print to console
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating training report: {e}")
        return False

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced NBA Prediction Model Training')
    parser.add_argument('--full-retrain', action='store_true', 
                       help='Force complete model retraining')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing models')
    parser.add_argument('--clean-cache', action='store_true',
                       help='Clean data cache before training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test prediction pipeline')
    parser.add_argument('--evaluation-games', type=int, default=30,
                       help='Number of games to use for evaluation (default: 30)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("=== Enhanced NBA Prediction Model Training ===")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please install missing dependencies.")
        return 1
    
    try:
        # Initialize components
        logger.info("Initializing prediction system...")
        collector = EnhancedDataCollector()
        predictor = EnhancedNBAPredictor(use_models=True)
        
        # Clean cache if requested
        if args.clean_cache:
            clean_cache(collector)
        
        # Handle different modes
        if args.test_only:
            logger.info("Running in test-only mode...")
            success = test_prediction_pipeline(predictor)
            if not success:
                return 1
        
        elif args.evaluate_only:
            logger.info("Running in evaluation-only mode...")
            if not predictor.models_loaded:
                logger.error("No models loaded for evaluation")
                return 1
            
            success = evaluate_models(predictor, args.evaluation_games)
            if not success:
                return 1
        
        else:
            # Full training pipeline
            logger.info("Running full training pipeline...")
            
            # Train models
            training_success = train_models(predictor, args.full_retrain)
            if not training_success:
                logger.error("Model training failed")
                return 1
            
            # Evaluate models
            evaluation_success = evaluate_models(predictor, args.evaluation_games)
            if not evaluation_success:
                logger.warning("Model evaluation failed, but training was successful")
            
            # Test prediction pipeline
            test_success = test_prediction_pipeline(predictor)
            if not test_success:
                logger.warning("Pipeline test failed, but models were trained")
        
        # Generate training report
        generate_training_report(predictor)
        
        logger.info("=== Training pipeline completed successfully ===")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in training pipeline: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)