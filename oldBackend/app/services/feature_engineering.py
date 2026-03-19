"""
Feature Engineering Service
Calculates technical indicators and prepares data for ML models
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from raw stock data"""
    
    def __init__(self):
        self.feature_columns = []
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data

        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators added
        """
        if data.empty:
            return data
        df = data.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        '''Technical Indicators'''
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_features(df)
        df = self._add_time_features(df)

        logger.info(f"Added {len(self.feature_columns)} technical indicators")
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        
        # Moving Averages
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # Parabolic SAR
        df['psar'] = ta.trend.psar_down(df['high'], df['low'], df['close'])
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        self.feature_columns.extend([
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_diff',
            'psar', 'adx'
        ])
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.rsi(df['close'])
        
        # Stochastic Oscillator
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ROC (Rate of Change)
        df['roc'] = ta.momentum.roc(df['close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        self.feature_columns.extend([
            'rsi', 'stoch', 'stoch_signal', 'williams_r', 'roc', 'cci', 'mfi'
        ])
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        # Bollinger Band position (with safety check)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, np.nan)
        
        # Average True Range
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channel
        df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
        df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
        
        self.feature_columns.extend([
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'atr', 'kc_upper', 'kc_lower'
        ])
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # Volume SMA (using rolling mean instead)
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Rate of Change (using pandas)
        df['volume_roc'] = df['volume'].pct_change()
        
        # Accumulation/Distribution Line
        df['adl'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        self.feature_columns.extend([
            'volume_sma_5', 'volume_sma_20', 'obv', 'volume_roc', 'adl'
        ])
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_2d'] = df['close'].pct_change(periods=2)
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Price position within day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Rolling statistics
        df['close_std_5'] = df['close'].rolling(window=5).std()
        df['close_std_20'] = df['close'].rolling(window=20).std()
        
        # Price ratios (with safety checks)
        df['close_sma_ratio'] = df['close'] / df['sma_20'].replace(0, np.nan)
        df['close_ema_ratio'] = df['close'] / df['ema_26'].replace(0, np.nan)
        
        self.feature_columns.extend([
            'price_change', 'price_change_2d', 'price_change_5d',
            'hl_spread', 'oc_spread', 'price_position',
            'close_std_5', 'close_std_20',
            'close_sma_ratio', 'close_ema_ratio'
        ])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.warning("No date column found, skipping time features")
            return df
        
        # Convert to datetime if needed
        df['date'] = pd.to_datetime(df['date'])
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Month
        df['month'] = df['date'].dt.month
        
        # Quarter
        df['quarter'] = df['date'].dt.quarter
        
        # Day of year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Is weekend
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Days since start
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        self.feature_columns.extend([
            'day_of_week', 'month', 'quarter', 'day_of_year',
            'is_weekend', 'days_since_start'
        ])
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, target_days: List[int] = [1, 3, 5]) -> pd.DataFrame:
        """
        Create target variables for ML models
        
        Args:
            df: DataFrame with price data
            target_days: List of days ahead to predict
        
        Returns:
            DataFrame with target variables added
        """
        df = df.copy()
        
        for days in target_days:
            # Future price change
            df[f'target_price_change_{days}d'] = df['close'].shift(-days).pct_change(days)
            
            # Future price direction (1 for up, 0 for down)
            df[f'target_direction_{days}d'] = (df[f'target_price_change_{days}d'] > 0).astype(int)
            
            # Future volatility
            df[f'target_volatility_{days}d'] = df['close'].rolling(window=days).std().shift(-days)
        
        return df
    
    def prepare_ml_data(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'target_direction_1d',
        test_size: float = 0.2,
        sequence_length: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for ML models
        
        Args:
            df: DataFrame with features and targets
            target_column: Name of target column
            test_size: Fraction of data for testing
            sequence_length: Number of days to use as sequence
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        df_clean = df.dropna(subset=[target_column])
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Replace infinity values with NaN, then fill
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        df_clean = df_clean.fillna(0)
        
        print(f"Data shape after cleaning: {df_clean.shape}")
        print(f"Sample of cleaned data:")
        print(df_clean.head(3))
        
        # Select feature columns (exclude date, symbol, and target columns)
        exclude_cols = ['date', 'symbol'] + [col for col in df_clean.columns if col.startswith('target_')]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = df_clean[feature_cols].values
        y = df_clean[target_column].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split into train/test
        split_idx = int(len(X_sequences) * (1 - test_size))
        
        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y_sequences[:split_idx]
        y_test = y_sequences[split_idx:]
        
        if X_train.shape[0] == 0:
            logger.warning("No training samples created. Not enough data after preprocessing.")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        logger.info(f"Prepared ML data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        logger.info(f"Features: {X_train.shape[2]}, Target: {target_column}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_guide(self) -> Dict[str, str]:
        """Return a guide explaining what each feature represents"""
        return {
            # Trend Indicators
            'sma_5': 'Simple Moving Average (5 days) - Short-term trend',
            'sma_20': 'Simple Moving Average (20 days) - Medium-term trend',
            'sma_200': 'Simple Moving Average (200 days) - Long-term trend',
            'macd': 'MACD Line - Trend momentum',
            'adx': 'Average Directional Index - Trend strength',
            
            # Momentum Indicators
            'rsi': 'Relative Strength Index - Overbought/oversold conditions',
            'stoch': 'Stochastic Oscillator - Momentum',
            'williams_r': "Williams %R - Momentum oscillator",
            'roc': 'Rate of Change - Price momentum',
            
            # Volatility Indicators
            'bb_width': 'Bollinger Band Width - Volatility measure',
            'atr': 'Average True Range - Volatility',
            'bb_position': 'Price position within Bollinger Bands',
            
            # Volume Indicators
            'obv': 'On Balance Volume - Volume-price relationship',
            'volume_roc': 'Volume Rate of Change - Volume momentum',
            
            # Price Features
            'price_change': 'Daily price change percentage',
            'hl_spread': 'High-Low spread as percentage of close',
            'close_sma_ratio': 'Current price relative to 20-day SMA',
        }

feature_engineer = FeatureEngineer()