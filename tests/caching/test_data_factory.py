"""
Centralized test data factories and generators for maximum efficiency.
Agent 4 Mission: Test Data Management & Caching System
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import random
import pickle
import gzip
import pytest
from unittest.mock import Mock

class MarketRegime(Enum):
    """Market regime types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"

class AssetType(Enum):
    """Asset types for testing."""
    FUTURES = "futures"
    FOREX = "forex"
    STOCKS = "stocks"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"

@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    asset_type: AssetType = AssetType.FUTURES
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    end_date: datetime = field(default_factory=datetime.now)
    frequency: str = "5min"  # 1min, 5min, 30min, 1h, 1d
    num_assets: int = 10
    volatility_level: float = 0.02
    trend_strength: float = 0.0
    correlation_level: float = 0.3
    noise_level: float = 0.1
    include_gaps: bool = True
    include_outliers: bool = True
    seed: Optional[int] = None

class MarketDataGenerator:
    """Generates realistic market data for testing."""
    
    def __init__(self, config: TestDataConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
    
    def generate_ohlcv_data(self) -> pd.DataFrame:
        """Generate OHLCV data for single asset."""
        # Create time index
        freq_map = {
            "1min": "1min",
            "5min": "5min", 
            "30min": "30min",
            "1h": "1h",
            "1d": "1d"
        }
        
        time_index = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=freq_map[self.config.frequency]
        )
        
        n_periods = len(time_index)
        
        # Generate price movements based on regime
        returns = self._generate_returns(n_periods)
        
        # Create price series
        initial_price = 100.0
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(time_index, prices)):
            # Generate open, high, low based on volatility
            daily_range = close * self.config.volatility_level * np.random.uniform(0.5, 2.0)
            
            open_price = close + np.random.normal(0, daily_range * 0.1)
            high_price = max(open_price, close) + np.random.uniform(0, daily_range * 0.5)
            low_price = min(open_price, close) - np.random.uniform(0, daily_range * 0.5)
            
            # Ensure logical price relationships
            high_price = max(high_price, open_price, close)
            low_price = min(low_price, open_price, close)
            
            # Generate volume
            base_volume = 1000000
            volume = max(100, int(base_volume * np.random.lognormal(0, 0.5)))
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add gaps if requested
        if self.config.include_gaps:
            df = self._add_price_gaps(df)
        
        # Add outliers if requested
        if self.config.include_outliers:
            df = self._add_outliers(df)
        
        return df
    
    def _generate_returns(self, n_periods: int) -> np.ndarray:
        """Generate returns based on market regime."""
        base_returns = np.random.normal(0, self.config.volatility_level, n_periods)
        
        if self.config.market_regime == MarketRegime.TRENDING:
            # Add trend component
            trend = np.linspace(0, self.config.trend_strength, n_periods)
            base_returns += trend / n_periods
        
        elif self.config.market_regime == MarketRegime.VOLATILE:
            # Increase volatility with regime changes
            volatility_multiplier = 1 + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 100)
            base_returns *= volatility_multiplier
        
        elif self.config.market_regime == MarketRegime.BULLISH:
            # Positive bias with occasional corrections
            base_returns += 0.0001  # Small positive drift
            correction_mask = np.random.random(n_periods) < 0.1
            base_returns[correction_mask] *= -2
        
        elif self.config.market_regime == MarketRegime.BEARISH:
            # Negative bias with occasional rallies
            base_returns -= 0.0001  # Small negative drift
            rally_mask = np.random.random(n_periods) < 0.1
            base_returns[rally_mask] *= -2
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_level * self.config.volatility_level, n_periods)
        base_returns += noise
        
        return base_returns
    
    def _add_price_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic price gaps to data."""
        if len(df) < 10:
            return df
        
        # Add random gaps
        n_gaps = max(1, int(len(df) * 0.02))  # 2% of data points
        gap_indices = np.random.choice(range(1, len(df)), n_gaps, replace=False)
        
        for idx in gap_indices:
            gap_size = np.random.normal(0, 0.05)  # 5% average gap
            df.loc[idx, 'open'] = df.loc[idx-1, 'close'] * (1 + gap_size)
            df.loc[idx, 'high'] = max(df.loc[idx, 'high'], df.loc[idx, 'open'])
            df.loc[idx, 'low'] = min(df.loc[idx, 'low'], df.loc[idx, 'open'])
        
        return df
    
    def _add_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic outliers to data."""
        if len(df) < 10:
            return df
        
        # Add random outliers
        n_outliers = max(1, int(len(df) * 0.005))  # 0.5% of data points
        outlier_indices = np.random.choice(range(len(df)), n_outliers, replace=False)
        
        for idx in outlier_indices:
            outlier_multiplier = np.random.choice([0.9, 1.1]) * (1 + np.random.exponential(0.1))
            df.loc[idx, 'high'] *= outlier_multiplier
            df.loc[idx, 'low'] *= (2 - outlier_multiplier)
        
        return df
    
    def generate_multi_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Generate correlated multi-asset data."""
        assets = {}
        
        # Generate base correlation matrix
        if self.config.num_assets > 1:
            correlation_matrix = self._generate_correlation_matrix()
        else:
            correlation_matrix = np.array([[1.0]])
        
        # Generate correlated returns
        base_returns = self._generate_correlated_returns(correlation_matrix)
        
        # Generate data for each asset
        for i in range(self.config.num_assets):
            asset_name = f"ASSET_{i:02d}"
            
            # Create time index
            freq_map = {
                "1min": "1min",
                "5min": "5min",
                "30min": "30min", 
                "1h": "1h",
                "1d": "1d"
            }
            
            time_index = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date,
                freq=freq_map[self.config.frequency]
            )
            
            # Use correlated returns for this asset
            asset_returns = base_returns[:, i]
            
            # Generate prices
            initial_price = 100.0 + np.random.uniform(-20, 20)
            prices = initial_price * np.exp(np.cumsum(asset_returns))
            
            # Generate OHLCV data
            data = []
            for j, (timestamp, close) in enumerate(zip(time_index, prices)):
                daily_range = close * self.config.volatility_level * np.random.uniform(0.5, 2.0)
                
                open_price = close + np.random.normal(0, daily_range * 0.1)
                high_price = max(open_price, close) + np.random.uniform(0, daily_range * 0.5)
                low_price = min(open_price, close) - np.random.uniform(0, daily_range * 0.5)
                
                high_price = max(high_price, open_price, close)
                low_price = min(low_price, open_price, close)
                
                base_volume = 1000000
                volume = max(100, int(base_volume * np.random.lognormal(0, 0.5)))
                
                data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2), 
                    'low': round(low_price, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            
            if self.config.include_gaps:
                df = self._add_price_gaps(df)
            
            if self.config.include_outliers:
                df = self._add_outliers(df)
            
            assets[asset_name] = df
        
        return assets
    
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate realistic correlation matrix."""
        n = self.config.num_assets
        
        # Start with identity matrix
        correlation_matrix = np.eye(n)
        
        # Add random correlations
        for i in range(n):
            for j in range(i+1, n):
                # Generate correlation with some clustering
                if np.random.random() < 0.3:  # 30% chance of correlation
                    correlation = np.random.normal(
                        self.config.correlation_level,
                        self.config.correlation_level * 0.3
                    )
                    correlation = np.clip(correlation, -0.9, 0.9)
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Minimum eigenvalue
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Rescale diagonal to 1
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)
        
        return correlation_matrix
    
    def _generate_correlated_returns(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Generate correlated returns using Cholesky decomposition."""
        n_assets = correlation_matrix.shape[0]
        
        # Create time index to get number of periods
        freq_map = {
            "1min": "1min",
            "5min": "5min",
            "30min": "30min",
            "1h": "1h", 
            "1d": "1d"
        }
        
        time_index = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=freq_map[self.config.frequency]
        )
        
        n_periods = len(time_index)
        
        # Generate independent returns
        independent_returns = np.random.normal(0, self.config.volatility_level, (n_periods, n_assets))
        
        # Apply correlation using Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        correlated_returns = independent_returns @ L.T
        
        # Apply regime-specific modifications
        for i in range(n_assets):
            regime_returns = self._apply_regime_effects(correlated_returns[:, i])
            correlated_returns[:, i] = regime_returns
        
        return correlated_returns
    
    def _apply_regime_effects(self, returns: np.ndarray) -> np.ndarray:
        """Apply regime-specific effects to returns."""
        if self.config.market_regime == MarketRegime.TRENDING:
            trend = np.linspace(0, self.config.trend_strength, len(returns))
            returns += trend / len(returns)
        
        elif self.config.market_regime == MarketRegime.VOLATILE:
            volatility_multiplier = 1 + 0.5 * np.sin(np.arange(len(returns)) * 2 * np.pi / 100)
            returns *= volatility_multiplier
        
        elif self.config.market_regime == MarketRegime.BULLISH:
            returns += 0.0001
            correction_mask = np.random.random(len(returns)) < 0.1
            returns[correction_mask] *= -2
        
        elif self.config.market_regime == MarketRegime.BEARISH:
            returns -= 0.0001
            rally_mask = np.random.random(len(returns)) < 0.1
            returns[rally_mask] *= -2
        
        return returns

class TestDataFactory:
    """Factory for creating test data with versioning and lifecycle management."""
    
    def __init__(self, cache_dir: str = ".pytest_cache/test_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.version_file = self.cache_dir / "data_versions.json"
        self.metadata_file = self.cache_dir / "data_metadata.json"
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load data metadata from disk."""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    self.versions = json.load(f)
            else:
                self.versions = {}
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.versions = {}
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def create_market_data(self, name: str, config: TestDataConfig, 
                          version: str = "latest") -> Dict[str, pd.DataFrame]:
        """Create or load cached market data."""
        data_key = f"{name}_{version}"
        data_file = self.cache_dir / f"{data_key}.pkl.gz"
        
        # Check if cached version exists and is valid
        if data_file.exists() and data_key in self.metadata:
            try:
                with gzip.open(data_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cached data matches config
                cached_config = TestDataConfig(**self.metadata[data_key]['config'])
                if self._configs_match(config, cached_config):
                    return cached_data
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        # Generate new data
        generator = MarketDataGenerator(config)
        
        if config.num_assets == 1:
            data = {"main": generator.generate_ohlcv_data()}
        else:
            data = generator.generate_multi_asset_data()
        
        # Cache the data
        try:
            with gzip.open(data_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.versions[name] = version
            self.metadata[data_key] = {
                'config': config.__dict__,
                'created_at': datetime.now().isoformat(),
                'file_size': data_file.stat().st_size,
                'num_assets': len(data),
                'data_points': sum(len(df) for df in data.values())
            }
            
            self._save_metadata()
        except Exception as e:
            print(f"Error caching data: {e}")
        
        return data
    
    def _configs_match(self, config1: TestDataConfig, config2: TestDataConfig) -> bool:
        """Check if two configurations match."""
        # Compare all important fields
        fields_to_compare = [
            'asset_type', 'market_regime', 'frequency', 'num_assets',
            'volatility_level', 'trend_strength', 'correlation_level',
            'noise_level', 'include_gaps', 'include_outliers', 'seed'
        ]
        
        for field in fields_to_compare:
            if getattr(config1, field) != getattr(config2, field):
                return False
        
        # Compare date ranges (within tolerance)
        date_tolerance = timedelta(hours=1)
        if abs(config1.start_date - config2.start_date) > date_tolerance:
            return False
        if abs(config1.end_date - config2.end_date) > date_tolerance:
            return False
        
        return True
    
    def list_cached_data(self) -> Dict[str, Any]:
        """List all cached data sets."""
        return {
            'versions': self.versions,
            'metadata': self.metadata,
            'cache_size': sum(
                (self.cache_dir / f"{key}.pkl.gz").stat().st_size
                for key in self.metadata.keys()
                if (self.cache_dir / f"{key}.pkl.gz").exists()
            )
        }
    
    def cleanup_old_data(self, max_age_days: int = 7) -> None:
        """Clean up old cached data."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        keys_to_remove = []
        for key, meta in self.metadata.items():
            try:
                created_at = datetime.fromisoformat(meta['created_at'])
                if created_at < cutoff_date:
                    keys_to_remove.append(key)
            except Exception:
                keys_to_remove.append(key)  # Remove corrupted entries
        
        for key in keys_to_remove:
            data_file = self.cache_dir / f"{key}.pkl.gz"
            if data_file.exists():
                data_file.unlink()
            
            del self.metadata[key]
            
            # Update versions
            for name, version in list(self.versions.items()):
                if f"{name}_{version}" == key:
                    del self.versions[name]
        
        self._save_metadata()
    
    def get_data_statistics(self, name: str, version: str = "latest") -> Dict[str, Any]:
        """Get statistics about cached data."""
        data_key = f"{name}_{version}"
        
        if data_key not in self.metadata:
            return {}
        
        meta = self.metadata[data_key]
        data_file = self.cache_dir / f"{data_key}.pkl.gz"
        
        stats = {
            'config': meta['config'],
            'created_at': meta['created_at'],
            'file_size_mb': meta['file_size'] / (1024 * 1024),
            'num_assets': meta['num_assets'],
            'data_points': meta['data_points'],
            'exists': data_file.exists()
        }
        
        return stats

# Global factory instance
test_data_factory = TestDataFactory()

# Pytest fixtures
@pytest.fixture(scope="session")
def market_data_generator():
    """Provide market data generator."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=1,
        seed=42
    )
    return MarketDataGenerator(config)

@pytest.fixture(scope="session")
def test_data_factory_fixture():
    """Provide test data factory."""
    return test_data_factory

@pytest.fixture
def sample_market_data():
    """Provide sample market data."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=1,
        seed=42
    )
    generator = MarketDataGenerator(config)
    return generator.generate_ohlcv_data()

@pytest.fixture
def multi_asset_data():
    """Provide multi-asset market data."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=5,
        correlation_level=0.5,
        seed=42
    )
    generator = MarketDataGenerator(config)
    return generator.generate_multi_asset_data()

# Tests for the data factory
class TestMarketDataGenerator:
    """Tests for market data generator."""
    
    def test_single_asset_generation(self, market_data_generator):
        """Test single asset data generation."""
        data = market_data_generator.generate_ohlcv_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Check price relationships
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()
    
    def test_multi_asset_generation(self):
        """Test multi-asset data generation."""
        config = TestDataConfig(num_assets=3, seed=42)
        generator = MarketDataGenerator(config)
        
        data = generator.generate_multi_asset_data()
        
        assert isinstance(data, dict)
        assert len(data) == 3
        
        for asset_name, asset_data in data.items():
            assert isinstance(asset_data, pd.DataFrame)
            assert len(asset_data) > 0
            assert all(col in asset_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def test_market_regime_effects(self):
        """Test different market regime effects."""
        regimes = [MarketRegime.BULLISH, MarketRegime.BEARISH, MarketRegime.VOLATILE]
        
        for regime in regimes:
            config = TestDataConfig(market_regime=regime, seed=42)
            generator = MarketDataGenerator(config)
            data = generator.generate_ohlcv_data()
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
    
    def test_correlation_matrix_generation(self):
        """Test correlation matrix generation."""
        config = TestDataConfig(num_assets=5, correlation_level=0.5, seed=42)
        generator = MarketDataGenerator(config)
        
        correlation_matrix = generator._generate_correlation_matrix()
        
        assert correlation_matrix.shape == (5, 5)
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Symmetric
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal is 1
        
        # Check positive definite
        eigenvals = np.linalg.eigvals(correlation_matrix)
        assert (eigenvals > 0).all()

class TestTestDataFactory:
    """Tests for test data factory."""
    
    def test_data_creation_and_caching(self):
        """Test data creation and caching."""
        factory = TestDataFactory()
        
        config = TestDataConfig(
            asset_type=AssetType.FUTURES,
            market_regime=MarketRegime.SIDEWAYS,
            frequency="5min",
            num_assets=2,
            seed=42
        )
        
        # First call should create data
        data1 = factory.create_market_data("test_data", config, "v1")
        assert isinstance(data1, dict)
        assert len(data1) == 2
        
        # Second call should load from cache
        data2 = factory.create_market_data("test_data", config, "v1")
        assert data1.keys() == data2.keys()
        
        # Verify data is identical
        for key in data1.keys():
            pd.testing.assert_frame_equal(data1[key], data2[key])
    
    def test_version_management(self):
        """Test version management."""
        factory = TestDataFactory()
        
        config = TestDataConfig(seed=42)
        
        # Create different versions
        data_v1 = factory.create_market_data("versioned_data", config, "v1")
        data_v2 = factory.create_market_data("versioned_data", config, "v2")
        
        # Should be able to access both versions
        cached_v1 = factory.create_market_data("versioned_data", config, "v1")
        cached_v2 = factory.create_market_data("versioned_data", config, "v2")
        
        assert data_v1.keys() == cached_v1.keys()
        assert data_v2.keys() == cached_v2.keys()
    
    def test_metadata_tracking(self):
        """Test metadata tracking."""
        factory = TestDataFactory()
        
        config = TestDataConfig(num_assets=3, seed=42)
        factory.create_market_data("metadata_test", config, "v1")
        
        metadata = factory.list_cached_data()
        
        assert 'versions' in metadata
        assert 'metadata' in metadata
        assert 'cache_size' in metadata
        assert 'metadata_test' in metadata['versions']
    
    def test_data_statistics(self):
        """Test data statistics."""
        factory = TestDataFactory()
        
        config = TestDataConfig(num_assets=2, seed=42)
        factory.create_market_data("stats_test", config, "v1")
        
        stats = factory.get_data_statistics("stats_test", "v1")
        
        assert 'config' in stats
        assert 'created_at' in stats
        assert 'file_size_mb' in stats
        assert 'num_assets' in stats
        assert 'data_points' in stats
        assert stats['num_assets'] == 2
    
    def test_cleanup_functionality(self):
        """Test data cleanup functionality."""
        factory = TestDataFactory()
        
        config = TestDataConfig(seed=42)
        factory.create_market_data("cleanup_test", config, "v1")
        
        # Verify data exists
        stats_before = factory.get_data_statistics("cleanup_test", "v1")
        assert stats_before['exists'] == True
        
        # Cleanup with very short age (should remove everything)
        factory.cleanup_old_data(max_age_days=0)
        
        # Verify data is removed
        stats_after = factory.get_data_statistics("cleanup_test", "v1")
        assert stats_after == {} or stats_after.get('exists', False) == False