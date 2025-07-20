"""
Configuration Management System for Lorentzian Trading Strategy
Provides centralized configuration with validation and environment-specific settings.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, validator


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    source_file: str = "/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv"
    processed_dir: str = "/home/QuantNova/GrandModel/lorentzian_strategy/data/processed"
    cache_dir: str = "/home/QuantNova/GrandModel/lorentzian_strategy/data/cache"
    validation_dir: str = "/home/QuantNova/GrandModel/lorentzian_strategy/data/validation"
    chunk_size: int = 10000
    cache_enabled: bool = True
    validate_ohlc: bool = True
    remove_duplicates: bool = True
    fill_missing_method: str = "forward"  # forward, backward, interpolate, drop


@dataclass
class LorentzianConfig:
    """Lorentzian distance calculation configuration"""
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    loop_back: int = 1
    
    # Distance calculation parameters
    distance_metric: str = "lorentzian"  # lorentzian, euclidean, manhattan
    use_dynamic_exits: bool = True
    kernel_settings: Dict[str, float] = field(default_factory=lambda: {
        "regression": True,
        "start_long_trades": True,
        "start_short_trades": True,
        "use_ema_filter": False,
        "ema_period": 200
    })


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    rsi_length: int = 14
    wt_channel_length: int = 9
    wt_average_length: int = 12
    
    # Rolling window settings
    short_window: int = 21
    medium_window: int = 50
    long_window: int = 200
    
    # Volatility settings
    atr_length: int = 14
    volatility_lookback: int = 20
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = "zscore"  # zscore, minmax, robust


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000.0
    commission: float = 0.0001  # 1 basis point
    slippage: float = 0.0001    # 1 basis point
    position_size: float = 0.02  # 2% of capital per trade
    
    # Risk management
    max_position_size: float = 0.1  # Maximum 10% of capital
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.04   # 4% take profit
    max_drawdown_limit: float = 0.15  # 15% maximum drawdown


@dataclass
class OptimizationConfig:
    """Optimization settings"""
    use_numba: bool = True
    use_gpu: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # JIT compilation settings
    cache_jit: bool = True
    nopython: bool = True
    fastmath: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_dir: str = "/home/QuantNova/GrandModel/lorentzian_strategy/logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class LorentzianStrategyConfig:
    """Main configuration class for the Lorentzian Trading Strategy"""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.config_path = config_path
        
        # Initialize default configurations
        self.data = DataConfig()
        self.lorentzian = LorentzianConfig()
        self.features = FeatureConfig()
        self.backtest = BacktestConfig()
        self.optimization = OptimizationConfig()
        self.logging = LoggingConfig()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML or JSON file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'data' in config_data:
                self._update_dataclass(self.data, config_data['data'])
            if 'lorentzian' in config_data:
                self._update_dataclass(self.lorentzian, config_data['lorentzian'])
            if 'features' in config_data:
                self._update_dataclass(self.features, config_data['features'])
            if 'backtest' in config_data:
                self._update_dataclass(self.backtest, config_data['backtest'])
            if 'optimization' in config_data:
                self._update_dataclass(self.optimization, config_data['optimization'])
            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])
                
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def _update_dataclass(self, instance, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data"""
        for key, value in data.items():
            if hasattr(instance, key):
                if isinstance(getattr(instance, key), dict) and isinstance(value, dict):
                    # Merge dictionaries
                    getattr(instance, key).update(value)
                else:
                    setattr(instance, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == "production":
            # Production optimizations
            self.optimization.use_numba = True
            self.optimization.parallel_processing = True
            self.logging.level = "WARNING"
            self.data.cache_enabled = True
            
        elif self.environment == "testing":
            # Testing configurations
            self.logging.level = "DEBUG"
            self.optimization.use_numba = False  # For easier debugging
            self.data.chunk_size = 1000  # Smaller chunks for testing
            
        elif self.environment == "development":
            # Development configurations
            self.logging.level = "DEBUG"
            self.optimization.use_numba = False
            self.data.cache_enabled = False  # Always fresh data
    
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate data configuration
        if not os.path.exists(os.path.dirname(self.data.source_file)):
            errors.append(f"Source file directory does not exist: {os.path.dirname(self.data.source_file)}")
        
        # Validate lorentzian parameters
        if self.lorentzian.neighbors_count <= 0:
            errors.append("neighbors_count must be positive")
        if self.lorentzian.max_bars_back <= 0:
            errors.append("max_bars_back must be positive")
        
        # Validate feature parameters
        if self.features.rsi_length <= 0:
            errors.append("rsi_length must be positive")
        
        # Validate backtest parameters
        if self.backtest.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        if not 0 <= self.backtest.commission <= 1:
            errors.append("commission must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    def save_to_file(self, output_path: str):
        """Save current configuration to file"""
        config_dict = {
            'environment': self.environment,
            'data': self.data.__dict__,
            'lorentzian': self.lorentzian.__dict__,
            'features': self.features.__dict__,
            'backtest': self.backtest.__dict__,
            'optimization': self.optimization.__dict__,
            'logging': self.logging.__dict__
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            if output_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data-related paths"""
        return {
            'source_file': self.data.source_file,
            'processed_dir': self.data.processed_dir,
            'cache_dir': self.data.cache_dir,
            'validation_dir': self.data.validation_dir,
            'log_dir': self.logging.log_dir
        }
    
    def ensure_directories(self):
        """Create all necessary directories"""
        paths = self.get_data_paths()
        for path in paths.values():
            if path.endswith(('.csv', '.pkl', '.json')):
                # It's a file, create the directory
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                # It's a directory
                os.makedirs(path, exist_ok=True)
    
    def __str__(self):
        """String representation of configuration"""
        return f"""
Lorentzian Strategy Configuration ({self.environment}):
=================================================
Data:
  - Source: {self.data.source_file}
  - Cache enabled: {self.data.cache_enabled}
  - Chunk size: {self.data.chunk_size:,}

Lorentzian:
  - Neighbors: {self.lorentzian.neighbors_count}
  - Max bars back: {self.lorentzian.max_bars_back:,}
  - Features: {self.lorentzian.feature_count}

Backtest:
  - Period: {self.backtest.start_date} to {self.backtest.end_date}
  - Capital: ${self.backtest.initial_capital:,.2f}
  - Commission: {self.backtest.commission:.4f}

Optimization:
  - Numba: {self.optimization.use_numba}
  - GPU: {self.optimization.use_gpu}
  - Parallel: {self.optimization.parallel_processing}
        """.strip()


# Global configuration instance
config = None

def get_config(config_path: Optional[str] = None, environment: str = "development") -> LorentzianStrategyConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config = LorentzianStrategyConfig(config_path, environment)
    return config

def reset_config():
    """Reset global configuration instance"""
    global config
    config = None