"""
Quick Test Script for ML Pipeline
Test the complete pipeline with a single stock
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.data_collector import DataCollector
from app.services.feature_engineering import FeatureEngineer
from app.services.ml_models import MLModelManager


async def test_single_stock():
    
    # Initialize services
    data_collector = DataCollector()
    feature_engineer = FeatureEngineer()
    ml_manager = MLModelManager()
    
    # Test with Apple stock
    symbol = "GOOGL"
    period = "3y"  # 6 months for more data
    
    print(f"Collecting data for {symbol}...")
    
    # Step 1: Collect data
    stock_data = await data_collector.get_multiple_stocks([symbol], source="yahoo", period=period)
    
    if not stock_data:
        print("Failed to collect data")
        return
    
    data = stock_data[symbol]
    print(f"Collected {len(data)} records for {symbol}")
    
    # Step 2: Engineer features
    print(f"🔧 Engineering features for {symbol}...")
    features_df = feature_engineer.calculate_technical_indicators(data)
    features_df = feature_engineer.create_target_variables(features_df)
    print(features_df.head())
    print(f"Created {len(features_df.columns)} features")
    
    # Step 3: Prepare ML data
    print(f"Preparing ML data...")
    X_train, X_test, y_train, y_test = feature_engineer.prepare_ml_data(
        features_df, target_column='target_direction_1d', test_size=0.2, sequence_length=3
    )
    
    if X_train.shape[0] == 0:
        print("\nNot enough data. Trying with sequence_length=1...")
        X_train, X_test, y_train, y_test = feature_engineer.prepare_ml_data(
            features_df, target_column='target_direction_1d', test_size=0.2, sequence_length=1
        )
        
        if X_train.shape[0] == 0:
            print("Still no data. The issue is with NaN handling.")
            return
        else:
            print(f"Success with sequence_length=1: {X_train.shape[0]} training samples")
    else:
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[2]}")
    
    # Step 4: Train a simple model (Random Forest)
    print(f"🎯 Training Random Forest model...")
    
    # Flatten data for Random Forest
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    try:
        rf_info = ml_manager.train_lstm(
            X_train, y_train, X_test, y_test,
            model_name=f"{symbol}_test_rf", task="classification"
        )
        
        accuracy = rf_info['metrics']['accuracy']
        print(f"LSTM trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {rf_info['metrics']['precision']:.4f}")
        print(f"Recall: {rf_info['metrics']['recall']:.4f}")
        print(f"F1-Score: {rf_info['metrics']['f1_score']:.4f}")

        rf_info = ml_manager.train_xgboost(
            X_train_flat, y_train, X_test_flat, y_test,
            model_name=f"{symbol}_test_rf", task="classification"
        )
        accuracy = rf_info['metrics']['accuracy']
        print(f"XGBoost trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {rf_info['metrics']['precision']:.4f}")
        print(f"Recall: {rf_info['metrics']['recall']:.4f}")
        print(f"F1-Score: {rf_info['metrics']['f1_score']:.4f}")

        rf_info = ml_manager.train_random_forest(
            X_train_flat, y_train, X_test_flat, y_test,
            model_name=f"{symbol}_test_rf", task="classification"
        )
        accuracy = rf_info['metrics']['accuracy']
        print(f"Random Forest trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {rf_info['metrics']['precision']:.4f}")
        print(f"Recall: {rf_info['metrics']['recall']:.4f}")
        print(f"F1-Score: {rf_info['metrics']['f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_single_stock()) 