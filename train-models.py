from predictor import ModelTrainer
import logging

def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    try:
        trainer = ModelTrainer()
        success = trainer.train_models()
        
        if success:
            trainer.save_models()
            logger.info("Training complete! Models saved successfully.")
        else:
            logger.error("Training failed. Check logs for details.")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()