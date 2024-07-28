import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import logging
from Config import get_config
from utility import (
    calculate_rsi, calculate_adx, calculate_cci, calculate_wt, calculate_fractal,
    apply_downsampling, get_feature_columns, calculate_kernel_regression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self._validate_input(df)
        self.df = df.copy()
        self.config = get_config()
        self.feature_list = get_feature_columns()
        self.custom_features = []
        
    def _validate_input(self, df: pd.DataFrame) -> None:
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

    def create_features(self) -> pd.DataFrame:
        self._calculate_indicators()
        self._add_price_action_features()
        self._add_kernel_regression_features()
        self._add_custom_features()
        self._handle_missing_values()
        self.df = apply_downsampling(self.df)
        return self.df

    def _calculate_indicators(self) -> None:
        try:
            self.df['RSI'] = calculate_rsi(self.df['close'])
            self.df['ADX'] = calculate_adx(self.df['high'], self.df['low'], self.df['close'])
            self.df['CCI'] = calculate_cci(self.df['high'], self.df['low'], self.df['close'])
            self.df['WT1'], self.df['WT2'] = calculate_wt((self.df['high'] + self.df['low'] + self.df['close']) / 3)
            if self.config.lorentzian.use_remote_fractals:
                self.df['Bullish_Fractal'], self.df['Bearish_Fractal'] = calculate_fractal(self.df['high'], self.df['low'])
            logger.info("Indicators calculated successfully.")
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    def _add_price_action_features(self) -> None:
        self.df['Price_Change'] = self.df['close'].pct_change()
        self.df['High_Low_Range'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df['Distance_From_MA'] = (self.df['close'] - self.df['close'].rolling(20).mean()) / self.df['close']
        logger.info("Price action features added.")

    def _add_kernel_regression_features(self) -> None:
        if self.config.use_kernel_regression:
            self.df['Kernel_Estimate'], self.df['Upper_Band'], self.df['Lower_Band'] = calculate_kernel_regression(self.df['close'])
            self.df['Distance_From_Kernel'] = (self.df['close'] - self.df['Kernel_Estimate']) / self.df['Kernel_Estimate']
            logger.info("Kernel regression features added.")

    def _add_custom_features(self) -> None:
        for feature_name, feature_func in self.custom_features:
            self.df[feature_name] = feature_func(self.df)
        logger.info(f"Added {len(self.custom_features)} custom features.")

    def _handle_missing_values(self) -> None:
        before_count = len(self.df)
        self.df.dropna(inplace=True)
        after_count = len(self.df)
        logger.info(f"Removed {before_count - after_count} rows with missing values.")

    def prepare_features_for_model(self) -> Tuple[pd.DataFrame, pd.Series]:
        X = self.df[self.feature_list + [f[0] for f in self.custom_features]]
        y = (self.df['close'].shift(-1) > self.df['close']).astype(int)
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        return X_scaled, y[:-1]

    def add_custom_feature(self, feature_name: str, feature_function: callable) -> None:
        self.custom_features.append((feature_name, feature_function))
        logger.info(f"Custom feature '{feature_name}' added.")

    def detect_mean_reversion(self) -> pd.Series:
        if 'Kernel_Estimate' not in self.df.columns:
            raise ValueError("Kernel regression features must be calculated first.")
        
        conditions = [
            (self.df['close'] > self.df['Upper_Band']),
            (self.df['close'] < self.df['Lower_Band']),
            (self.df['close'] > self.df['Kernel_Estimate']),
            (self.df['close'] < self.df['Kernel_Estimate'])
        ]
        choices = ['strong_downward', 'strong_upward', 'downward', 'upward']
        return pd.Series(np.select(conditions, choices, default='none'), index=self.df.index)

    def detect_first_pullback(self) -> Tuple[pd.Series, pd.Series]:
        bullish_pullback = (
            (self.df['close'] < self.df['close'].shift(1)) & 
            (self.df['close'].rolling(5).mean() > self.df['close'].rolling(5).mean().shift(1))
        )
        bearish_pullback = (
            (self.df['close'] > self.df['close'].shift(1)) & 
            (self.df['close'].rolling(5).mean() < self.df['close'].rolling(5).mean().shift(1))
        )
        return bullish_pullback, bearish_pullback

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    engineer = FeatureEngineer(df)
    df_with_features = engineer.create_features()
    X, y = engineer.prepare_features_for_model()
    
    # Add mean reversion and first pullback detection
    df_with_features['Mean_Reversion'] = engineer.detect_mean_reversion()
    df_with_features['Bullish_Pullback'], df_with_features['Bearish_Pullback'] = engineer.detect_first_pullback()
    
    return df_with_features, X, y