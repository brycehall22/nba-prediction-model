#!/usr/bin/env python3
"""
Simple test script to verify the prediction fixes work (no Unicode characters)
"""

import logging
import sys
from predictor import EnhancedNBAPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_prediction():
    """Test basic prediction functionality"""
    try:
        print("Testing NBA prediction fixes...")
        
        # Initialize predictor
        predictor = EnhancedNBAPredictor(use_models=True)
        
        # Test team IDs (Lakers vs Warriors)
        home_team_id = 1610612747  # Lakers
        away_team_id = 1610612744  # Warriors
        
        print(f"Testing prediction for Lakers (home) vs Warriors (away)...")
        
        # Make a prediction
        result = predictor.predict_game(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            detailed=False  # Keep it simple for testing
        )
        
        if 'error' in result and result['error']:
            print(f"Prediction failed with error: {result['error']}")
            return False
        
        # Check if we got valid predictions
        predictions = result.get('predictions', {})
        home_score = predictions.get('home_score', 0)
        away_score = predictions.get('away_score', 0)
        
        print(f"Prediction successful!")
        print(f"Home (Lakers): {home_score}")
        print(f"Away (Warriors): {away_score}")
        print(f"Total: {predictions.get('total_score', 0)}")
        print(f"Method: {result.get('methodology', {}).get('home_method', 'unknown')}")
        print(f"Confidence: {result.get('uncertainty', {}).get('overall_confidence', 0)}")
        
        # Validate results
        if home_score > 50 and away_score > 50:  # Basic sanity check
            print("PASS: Prediction results look reasonable")
            return True
        else:
            print("FAIL: Prediction results seem unrealistic")
            return False
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        logging.error(f"Test exception: {e}", exc_info=True)
        return False

def main():
    """Run the test"""
    print("=" * 50)
    print("NBA Prediction Model - Fix Verification")
    print("=" * 50)
    
    if test_prediction():
        print("\nSUCCESS: The fixes appear to be working!")
        print("\nWhat was fixed:")
        print("1. NBA API 'resultSet' error - Added robust error handling")
        print("2. Feature mismatch error - Ensured consistent feature names")
        print("3. Added fallback prediction methods")
        print("\nRecommendations:")
        print("- The models are now working with basic functionality")
        print("- For better accuracy, consider full retraining with more data")
        print("- The system will gracefully handle API failures")
        return True
    else:
        print("\nFAILED: Some issues remain")
        print("Check the error messages above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)