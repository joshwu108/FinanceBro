"""
Complete ML Pipeline for Stock Prediction
This script demonstrates the full workflow from data collection to model training
"""
import asyncio
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.data_collector import DataCollector
from app.services.feature_engineering import FeatureEngineer
from app.services.ml_models import MLModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockPredictionPipeline:
    """Complete pipeline for stock prediction"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_manager = MLModelManager()
        
        # Popular stocks for training
        self.training_symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
            "META", "NVDA", "NFLX", "ADBE", "CRM"
        ]
    
    async def run_complete_pipeline(self, symbols: list = None, period: str = "1y"):
        """"
        Args:
            symbols: List of stock symbols to train on
            period: Time period for data collection
        """
        if symbols is None:
            symbols = self.training_symbols
        
        logger.info("Starting complete ML pipeline...")
        logger.info(f"Training on {len(symbols)} symbols: {symbols}")
        
        # Step 1: Data Collection
        logger.info("Step 1: Collecting stock data...")
        stock_data = await self._collect_data(symbols, period)
        
        if not stock_data:
            logger.error("No data collected. Exiting pipeline.")
            return
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Engineering features...")
        processed_data = self._engineer_features(stock_data)
        
        # Step 3: Model Training
        logger.info("Step 3: Training ML models...")
        model_results = await self._train_models(processed_data)
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Evaluating models...")
        self._evaluate_models(model_results)
        
        logger.info("Pipeline completed successfully!")
        return model_results
    
    async def _collect_data(self, symbols: list, period: str) -> dict:
        """Collect stock data for all symbols"""
        try:
            # Collect data from Yahoo Finance
            stock_data = await self.data_collector.get_multiple_stocks(
                symbols, source="yahoo", period=period
            )
            
            # Validate and save data
            valid_data = {}
            for symbol, data in stock_data.items():
                if self.data_collector.validate_data(data):
                    self.data_collector.save_data(data, symbol, format="csv")
                    valid_data[symbol] = data
                    logger.info(f"{symbol}: {len(data)} records")
                else:
                    logger.warning(f"⚠️ {symbol}: Invalid data, skipping")
            
            return valid_data
            
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            return {}
    
    def _engineer_features(self, stock_data: dict) -> dict:
        """Engineer features for all stocks"""
        processed_data = {}
        
        for symbol, data in stock_data.items():
            logger.info(f"Processing features for {symbol}...")
            
            try:
                # Calculate technical indicators
                features_df = self.feature_engineer.calculate_technical_indicators(data)
                
                # Create target variables
                features_df = self.feature_engineer.create_target_variables(features_df)
                
                # Save processed data
                os.makedirs("data/processed", exist_ok=True)
                features_df.to_csv(f"data/processed/{symbol}_features.csv", index=False)
                
                processed_data[symbol] = features_df
                logger.info(f"{symbol}: {len(features_df.columns)} features calculated")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        return processed_data
    
    async def _train_models(self, processed_data: dict) -> dict:
        """Train ML models for each stock"""
        model_results = {}
        
        for symbol, data in processed_data.items():
            logger.info(f"Training models for {symbol}...")
            
            try:
                # Prepare ML data
                X_train, X_test, y_train, y_test = self.feature_engineer.prepare_ml_data(
                    data, target_column='target_direction_1d', test_size=0.2
                )
                
                if len(X_train) < 100:  # Need sufficient data
                    logger.warning(f"{symbol}: Insufficient data ({len(X_train)} samples), skipping")
                    continue
                
                # Train different models
                models_info = {}
                
                # 1. Random Forest
                try:
                    rf_info = self.ml_manager.train_random_forest(
                        X_train, y_train, X_test, y_test,
                        model_name=f"{symbol}_rf", task="classification"
                    )
                    models_info['random_forest'] = rf_info
                    logger.info(f"{symbol} Random Forest: {rf_info['metrics']['accuracy']:.4f}")
                except Exception as e:
                    logger.error(f"{symbol} Random Forest failed: {str(e)}")
                
                # 2. XGBoost
                try:
                    xgb_info = self.ml_manager.train_xgboost(
                        X_train, y_train, X_test, y_test,
                        model_name=f"{symbol}_xgb", task="classification"
                    )
                    models_info['xgboost'] = xgb_info
                    logger.info(f"{symbol} XGBoost: {xgb_info['metrics']['accuracy']:.4f}")
                except Exception as e:
                    logger.error(f"{symbol} XGBoost failed: {str(e)}")
                
                # 3. LSTM (if we have enough data)
                if len(X_train) >= 500:  # LSTM needs more data
                    try:
                        lstm_info = self.ml_manager.train_lstm(
                            X_train, y_train, X_test, y_test,
                            model_name=f"{symbol}_lstm", task="classification"
                        )
                        models_info['lstm'] = lstm_info
                        logger.info(f"{symbol} LSTM: {lstm_info['metrics']['accuracy']:.4f}")
                    except Exception as e:
                        logger.error(f"{symbol} LSTM failed: {str(e)}")
                
                model_results[symbol] = models_info
                
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {str(e)}")
                continue
        
        return model_results
    
    def _evaluate_models(self, model_results: dict):
        """Evaluate and compare model performance"""
        logger.info("📊 Model Performance Summary:")
        logger.info("=" * 60)
        
        # Collect all results
        all_results = []
        
        for symbol, models in model_results.items():
            logger.info(f"\n{symbol}:")
            
            for model_name, info in models.items():
                metrics = info['metrics']
                accuracy = metrics.get('accuracy', metrics.get('r2', 'N/A'))
                
                logger.info(f"  {model_name.upper()}:")
                logger.info(f"    Accuracy/R²: {accuracy:.4f}")
                logger.info(f"    Precision: {metrics.get('precision', 'N/A'):.4f}")
                logger.info(f"    Recall: {metrics.get('recall', 'N/A'):.4f}")
                logger.info(f"    F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
                
                all_results.append({
                    'symbol': symbol,
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision': metrics.get('precision', 'N/A'),
                    'recall': metrics.get('recall', 'N/A'),
                    'f1_score': metrics.get('f1_score', 'N/A')
                })
        
        # Find best models
        if all_results:
            best_accuracy = max(all_results, key=lambda x: x['accuracy'] if x['accuracy'] != 'N/A' else 0)
            logger.info(f"\n🏆 Best Model: {best_accuracy['symbol']} {best_accuracy['model']} "
                       f"(Accuracy: {best_accuracy['accuracy']:.4f})")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("data/model_results.csv", index=False)
        logger.info(f"\n💾 Results saved to data/model_results.csv")
    
    def get_model_summary(self) -> dict:
        """Get summary of all trained models"""
        return self.ml_manager.get_model_summary()


async def main():
    """Main function to run the pipeline"""
    pipeline = StockPredictionPipeline()
    
    # You can customize these parameters
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]  # Start with fewer symbols for testing
    period = "6mo"  # Use 6 months for faster testing
    
    # Run the complete pipeline
    results = await pipeline.run_complete_pipeline(symbols, period)
    
    if results:
        # Get final summary
        summary = pipeline.get_model_summary()
        logger.info(f"\nFinal Summary: {summary['total_models']} models trained")
        
        # Show best performing models
        for model_name, info in summary['models'].items():
            metrics = info['metrics']
            accuracy = metrics.get('accuracy', metrics.get('r2', 'N/A'))
            logger.info(f"  {model_name}: {accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main()) 