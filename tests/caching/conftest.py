"""
Comprehensive pytest configuration for advanced test caching system.
Agent 4 Mission: Test Data Management & Caching System
"""
import pytest
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List

# Import caching components
from .test_cache_manager import TestCacheManager, IncrementalTestRunner, cache_manager
from .test_data_factory import TestDataFactory, MarketDataGenerator, TestDataConfig, AssetType, MarketRegime
from .database_fixtures import DatabaseFixtureManager, DatabaseConfig, MockExternalServiceManager
from .memory_optimization import MemoryMappedTestData, MemoryMapConfig, CompressionManager

# Global instances for session-wide caching
_session_cache_manager = None
_session_data_factory = None
_session_db_manager = None
_session_mock_manager = None
_session_memory_manager = None

def pytest_configure(config):
    """Configure pytest with caching capabilities."""
    global _session_cache_manager, _session_data_factory, _session_db_manager
    global _session_mock_manager, _session_memory_manager
    
    # Initialize session-wide managers
    _session_cache_manager = TestCacheManager()
    _session_data_factory = TestDataFactory()
    _session_db_manager = DatabaseFixtureManager(DatabaseConfig())
    _session_mock_manager = MockExternalServiceManager()
    _session_memory_manager = MemoryMappedTestData(MemoryMapConfig())
    
    # Configure pytest markers
    config.addinivalue_line(
        "markers", 
        "cached: Tests that benefit from result caching"
    )
    config.addinivalue_line(
        "markers", 
        "incremental: Tests for incremental execution"
    )
    config.addinivalue_line(
        "markers", 
        "data_factory: Tests using test data factories"
    )
    config.addinivalue_line(
        "markers", 
        "database_fixture: Tests requiring database fixtures"
    )
    config.addinivalue_line(
        "markers", 
        "memory_mapped: Tests using memory-mapped data"
    )
    config.addinivalue_line(
        "markers", 
        "mock_services: Tests using mock external services"
    )

def pytest_unconfigure(config):
    """Clean up session-wide resources."""
    global _session_cache_manager, _session_data_factory, _session_db_manager
    global _session_mock_manager, _session_memory_manager
    
    # Cleanup resources
    if _session_db_manager:
        _session_db_manager.cleanup_schemas()
        _session_db_manager.cleanup_containers()
    
    if _session_mock_manager:
        _session_mock_manager.cleanup_mocks()
    
    if _session_memory_manager:
        _session_memory_manager.close()
    
    if _session_data_factory:
        _session_data_factory.cleanup_old_data(max_age_days=1)

def pytest_collection_modifyitems(config, items):
    """Modify test collection to support incremental execution."""
    if not config.getoption("--lf") and not config.getoption("--ff"):
        return
    
    # Use incremental test runner to determine which tests to run
    incremental_runner = IncrementalTestRunner(_session_cache_manager)
    tests_to_run = set(incremental_runner.get_tests_to_run())
    
    # Filter items based on incremental execution
    filtered_items = []
    for item in items:
        test_file = str(item.fspath)
        if test_file in tests_to_run or not tests_to_run:
            filtered_items.append(item)
    
    items[:] = filtered_items

def pytest_runtest_setup(item):
    """Setup hook called before each test."""
    # Check if test should be skipped based on cache
    if hasattr(item, 'fspath'):
        test_file = str(item.fspath)
        test_id = f"{item.fspath.basename}::{item.name}"
        
        # Skip if cached and passing (only if not forced)
        if not item.config.getoption("--lf") and not item.config.getoption("--ff"):
            if not _session_cache_manager.should_run_test(test_id, test_file):
                pytest.skip(f"Test {test_id} skipped due to cache")

def pytest_runtest_teardown(item, nextitem):
    """Teardown hook called after each test."""
    # Store test result in cache
    if hasattr(item, 'fspath'):
        test_file = str(item.fspath)
        test_id = f"{item.fspath.basename}::{item.name}"
        
        # Get test outcome and duration
        outcome = "passed"  # Default
        duration = 0.0
        
        # Try to get actual outcome from test result
        if hasattr(item, 'user_properties'):
            for prop in item.user_properties:
                if prop[0] == 'outcome':
                    outcome = prop[1]
                elif prop[0] == 'duration':
                    duration = prop[1]
        
        # Store in cache
        _session_cache_manager.store_result(test_id, outcome, duration, test_file)

@pytest.fixture(scope="session")
def session_cache_manager():
    """Provide session-wide cache manager."""
    return _session_cache_manager

@pytest.fixture(scope="session")
def session_data_factory():
    """Provide session-wide data factory."""
    return _session_data_factory

@pytest.fixture(scope="session")
def session_db_manager():
    """Provide session-wide database manager."""
    return _session_db_manager

@pytest.fixture(scope="session")
def session_mock_manager():
    """Provide session-wide mock manager."""
    return _session_mock_manager

@pytest.fixture(scope="session")
def session_memory_manager():
    """Provide session-wide memory manager."""
    return _session_memory_manager

@pytest.fixture(scope="function")
def test_cache_manager():
    """Provide test-specific cache manager."""
    return _session_cache_manager

@pytest.fixture(scope="function")
def test_data_factory():
    """Provide test-specific data factory."""
    return _session_data_factory

@pytest.fixture(scope="function")
def incremental_runner():
    """Provide incremental test runner."""
    return IncrementalTestRunner(_session_cache_manager)

@pytest.fixture(scope="function")
def market_data_generator():
    """Provide market data generator with default config."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=1,
        seed=42
    )
    return MarketDataGenerator(config)

@pytest.fixture(scope="function")
def multi_asset_generator():
    """Provide multi-asset market data generator."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=5,
        correlation_level=0.5,
        seed=42
    )
    return MarketDataGenerator(config)

@pytest.fixture(scope="function")
def volatile_market_generator():
    """Provide volatile market data generator."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.VOLATILE,
        frequency="1min",
        num_assets=3,
        volatility_level=0.05,
        seed=123
    )
    return MarketDataGenerator(config)

@pytest.fixture(scope="function")
def cached_market_data():
    """Provide cached market data."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=1,
        seed=42
    )
    return _session_data_factory.create_market_data("test_market_data", config)

@pytest.fixture(scope="function")
def cached_multi_asset_data():
    """Provide cached multi-asset data."""
    config = TestDataConfig(
        asset_type=AssetType.FUTURES,
        market_regime=MarketRegime.SIDEWAYS,
        frequency="5min",
        num_assets=5,
        correlation_level=0.5,
        seed=42
    )
    return _session_data_factory.create_market_data("test_multi_asset_data", config)

@pytest.fixture(scope="function")
def isolated_db_schema():
    """Provide isolated database schema."""
    schema_name = f"test_schema_{int(time.time() * 1000)}"
    
    try:
        _session_db_manager.create_schema(schema_name)
        _session_db_manager.setup_test_tables(schema_name)
        yield schema_name
    finally:
        _session_db_manager.drop_schema(schema_name)

@pytest.fixture(scope="function")
def db_with_market_data(isolated_db_schema):
    """Provide database with market data."""
    _session_db_manager.populate_test_data(isolated_db_schema, "market")
    return isolated_db_schema

@pytest.fixture(scope="function")
def db_with_risk_data(isolated_db_schema):
    """Provide database with risk data."""
    _session_db_manager.populate_test_data(isolated_db_schema, "risk")
    return isolated_db_schema

@pytest.fixture(scope="function")
def db_with_positions(isolated_db_schema):
    """Provide database with positions data."""
    _session_db_manager.populate_test_data(isolated_db_schema, "positions")
    return isolated_db_schema

@pytest.fixture(scope="function")
def db_with_orders(isolated_db_schema):
    """Provide database with orders data."""
    _session_db_manager.populate_test_data(isolated_db_schema, "orders")
    return isolated_db_schema

@pytest.fixture(scope="function")
def db_connection(isolated_db_schema):
    """Provide database connection."""
    with _session_db_manager.get_sync_connection(isolated_db_schema) as conn:
        yield conn

@pytest.fixture(scope="function")
async def async_db_connection(isolated_db_schema):
    """Provide async database connection."""
    async with _session_db_manager.get_async_connection(isolated_db_schema) as conn:
        yield conn

@pytest.fixture(scope="function")
def mock_broker():
    """Provide mock broker."""
    broker = _session_mock_manager.setup_broker_mock(f"test_broker_{int(time.time())}")
    yield broker
    # Mock manager handles cleanup in session teardown

@pytest.fixture(scope="function")
def mock_data_provider():
    """Provide mock data provider."""
    provider = _session_mock_manager.setup_market_data_mock(f"test_provider_{int(time.time())}")
    yield provider
    # Mock manager handles cleanup in session teardown

@pytest.fixture(scope="function")
def mock_risk_service():
    """Provide mock risk service."""
    service = _session_mock_manager.setup_risk_service_mock(f"test_risk_service_{int(time.time())}")
    yield service
    # Mock manager handles cleanup in session teardown

@pytest.fixture(scope="function")
def memory_mapped_test_data():
    """Provide memory-mapped test data."""
    import pandas as pd
    import numpy as np
    
    # Create test data
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Create memory-mapped file
    filename = f"test_data_{int(time.time())}"
    _session_memory_manager.create_memory_mapped_file(test_df, filename)
    
    yield filename
    
    # Cleanup is handled by session teardown

@pytest.fixture(scope="function")
def memory_mapped_array():
    """Provide memory-mapped array data."""
    import numpy as np
    
    # Create test array
    test_array = np.random.randn(50, 10)
    
    # Create memory-mapped file
    filename = f"test_array_{int(time.time())}"
    _session_memory_manager.create_memory_mapped_file(test_array, filename)
    
    yield filename
    
    # Cleanup is handled by session teardown

@pytest.fixture(scope="function")
def compression_manager():
    """Provide compression manager."""
    return CompressionManager()

@pytest.fixture(scope="function")
def performance_timer():
    """Provide performance timer context manager."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
        
        def get_duration(self):
            return self.duration
    
    return PerformanceTimer()

@pytest.fixture(scope="function")
def async_event_loop():
    """Provide async event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# Pytest hooks for better caching integration
def pytest_runtest_protocol(item, nextitem):
    """Custom test protocol for caching integration."""
    # Mark test start time
    item.user_properties.append(('start_time', time.time()))
    
    # Run default protocol
    return None  # Use default protocol

def pytest_runtest_makereport(item, call):
    """Generate test report with caching info."""
    if call.when == "call":
        # Calculate duration
        start_time = None
        for prop in item.user_properties:
            if prop[0] == 'start_time':
                start_time = prop[1]
                break
        
        if start_time:
            duration = time.time() - start_time
            item.user_properties.append(('duration', duration))
        
        # Add outcome
        outcome = "passed" if call.excinfo is None else "failed"
        item.user_properties.append(('outcome', outcome))

# Custom pytest collection for caching
class CachingCollector:
    """Custom collector that respects caching rules."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
    
    def should_collect_test(self, test_path, test_name):
        """Determine if test should be collected based on cache."""
        test_id = f"{test_path}::{test_name}"
        return self.cache_manager.should_run_test(test_id, test_path)

# Register custom markers
pytest_plugins = []