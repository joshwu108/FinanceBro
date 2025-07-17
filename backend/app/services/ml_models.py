"""
Machine Learning Models Service
Trains and manages different ML models for stock prediction
"""
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = self.dropout(out[:, -1, :])
        
        # Pass through fully connected layer
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out


class MLModelManager:
    """Manages multiple ML models for stock prediction"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lstm': {
                'hidden_size': 50,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "random_forest",
        task: str = "classification"
    ) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_name: Name to save the model
            task: 'classification' or 'regression'
        
        Returns:
            Dictionary with model info and metrics
        """
        logger.info(f"Training Random Forest model for {task}...")
        
        # Reshape data if needed (for time series)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Choose model type
        if task == "classification":
            model = RandomForestClassifier(**self.model_configs['random_forest'])
        else:
            model = RandomForestRegressor(**self.model_configs['random_forest'])
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if task == "classification" else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        
        # Save model and scaler
        self._save_model(model, scaler, model_name)
        
        # Store model info
        model_info = {
            'model_type': 'random_forest',
            'task': task,
            'features': X_train_flat.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': metrics,
            'feature_importance': dict(zip(range(X_train_flat.shape[1]), model.feature_importances_))
        }
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.model_metrics[model_name] = metrics
        
        logger.info(f"Random Forest training completed. Accuracy: {metrics.get('accuracy', metrics.get('r2', 'N/A'))}")
        
        return model_info
    
    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "xgboost",
        task: str = "classification"
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_name: Name to save the model
            task: 'classification' or 'regression'
        
        Returns:
            Dictionary with model info and metrics
        """
        logger.info(f"Training XGBoost model for {task}...")
        
        # Reshape data if needed
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Choose model type
        if task == "classification":
            model = xgb.XGBClassifier(**self.model_configs['xgboost'])
        else:
            model = xgb.XGBRegressor(**self.model_configs['xgboost'])
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if task == "classification" else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        
        # Save model and scaler
        self._save_model(model, scaler, model_name)
        
        # Store model info
        model_info = {
            'model_type': 'xgboost',
            'task': task,
            'features': X_train_flat.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': metrics,
            'feature_importance': dict(zip(range(X_train_flat.shape[1]), model.feature_importances_))
        }
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.model_metrics[model_name] = metrics
        
        logger.info(f"XGBoost training completed. Accuracy: {metrics.get('accuracy', metrics.get('r2', 'N/A'))}")
        
        return model_info
    
    def train_lstm(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "lstm",
        task: str = "classification"
    ) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X_train, y_train: Training data (X should be 3D for LSTM)
            X_test, y_test: Test data
            model_name: Name to save the model
            task: 'classification' or 'regression'
        
        Returns:
            Dictionary with model info and metrics
        """
        logger.info(f"Training LSTM model for {task}...")
        
        # Ensure data is 3D for LSTM
        if len(X_train.shape) != 3:
            raise ValueError("LSTM requires 3D input data (samples, timesteps, features)")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_configs['lstm']['batch_size'], 
            shuffle=True
        )
        
        # Initialize model
        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.model_configs['lstm']['hidden_size'],
            num_layers=self.model_configs['lstm']['num_layers'],
            dropout=self.model_configs['lstm']['dropout']
        )
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.model_configs['lstm']['learning_rate'])
        
        # Training loop
        model.train()
        for epoch in range(self.model_configs['lstm']['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.model_configs['lstm']['epochs']}], Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            y_pred = (test_outputs.squeeze() > 0.5).float().numpy()
            y_pred_proba = test_outputs.squeeze().numpy()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task)
        
        # Save model
        self._save_model(model, None, model_name)
        
        # Store model info
        model_info = {
            'model_type': 'lstm',
            'task': task,
            'features': X_train.shape[2],
            'sequence_length': X_train.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': metrics
        }
        
        self.models[model_name] = model
        self.model_metrics[model_name] = metrics
        
        logger.info(f"LSTM training completed. Accuracy: {metrics.get('accuracy', metrics.get('r2', 'N/A'))}")
        
        return model_info
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None,
        task: str = "classification"
    ) -> Dict[str, float]:
        """Calculate model performance metrics"""
        
        if task == "classification":
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                # For binary classification, use positive class probability
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                metrics['prediction_probabilities'] = y_pred_proba.tolist()
        else:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
        
        return metrics
    
    def _save_model(self, model: Any, scaler: Optional[StandardScaler], model_name: str):
        """Save model and scaler to disk"""
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if hasattr(model, 'state_dict'):  # PyTorch model
            torch.save(model.state_dict(), model_path)
        else:  # Scikit-learn model
            joblib.dump(model, model_path)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str) -> Tuple[Any, Optional[StandardScaler]]:
        """Load model and scaler from disk"""
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
        
        # Load model
        if model_name.startswith('lstm'):
            # Load LSTM model
            input_size = 50  # You might want to store this in model info
            model = LSTMModel(input_size=input_size)
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            # Load scikit-learn model
            model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def predict(
        self, 
        model_name: str, 
        X: np.ndarray,
        return_probability: bool = False
    ) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model to use
            X: Input features
            return_probability: Whether to return probabilities (for classification)
        
        Returns:
            Predictions
        """
        if model_name not in self.models:
            # Try to load model
            model, scaler = self.load_model(model_name)
            self.models[model_name] = model
            if scaler:
                self.scalers[model_name] = scaler
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Preprocess data
        if scaler is not None:
            if len(X.shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
                X_scaled = scaler.transform(X_flat)
                X_scaled = X_scaled.reshape(X.shape)
            else:
                X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        if hasattr(model, 'predict_proba') and return_probability:
            return model.predict_proba(X_scaled)
        else:
            return model.predict(X_scaled)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            'total_models': len(self.models),
            'models': {}
        }
        
        for model_name, metrics in self.model_metrics.items():
            summary['models'][model_name] = {
                'metrics': metrics,
                'model_type': self.models[model_name].__class__.__name__ if model_name in self.models else 'Unknown'
            }
        
        return summary


# Global instance
ml_manager = MLModelManager()


# Example usage and testing
def test_ml_models():
    """Test the ML model training"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    sequence_length = 30
    
    # Create synthetic data
    X_train = np.random.randn(n_samples, sequence_length, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_test = np.random.randn(200, sequence_length, n_features)
    y_test = np.random.randint(0, 2, 200)
    
    # Flatten for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print("Testing ML model training...")
    
    # Test Random Forest
    rf_info = ml_manager.train_random_forest(
        X_train_flat, y_train, X_test_flat, y_test, 
        model_name="test_rf", task="classification"
    )
    print(f"Random Forest - Accuracy: {rf_info['metrics']['accuracy']:.4f}")
    
    # Test XGBoost
    xgb_info = ml_manager.train_xgboost(
        X_train_flat, y_train, X_test_flat, y_test,
        model_name="test_xgb", task="classification"
    )
    print(f"XGBoost - Accuracy: {xgb_info['metrics']['accuracy']:.4f}")
    
    # Test LSTM
    lstm_info = ml_manager.train_lstm(
        X_train, y_train, X_test, y_test,
        model_name="test_lstm", task="classification"
    )
    print(f"LSTM - Accuracy: {lstm_info['metrics']['accuracy']:.4f}")
    
    # Get summary
    summary = ml_manager.get_model_summary()
    print(f"\nModel Summary: {summary['total_models']} models trained")
    
    return summary


if __name__ == "__main__":
    test_ml_models() 