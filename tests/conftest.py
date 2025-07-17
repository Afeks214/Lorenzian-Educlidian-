"""
Global pytest configuration and fixtures for GrandModel test suite.
"""
import pytest
import sys
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any, Generator

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import core components
try:
    from src.core.event_bus import EventBus
    from src.core.events import EventType
    from src.core.kernel import AlgoSpaceKernel
    HAS_CORE = True
except ImportError:
    HAS_CORE = False

# Test environment detection
def pytest_configure(config):
    """Configure pytest environment."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "strategic: Strategic MARL tests")
    config.addinivalue_line("markers", "tactical: Tactical MARL tests")
    config.addinivalue_line("markers", "risk: Risk management tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_docker: Tests requiring Docker")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_data: Tests requiring external data")


def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Skip GPU tests in CI environment
    if "requires_gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    
    # Skip Docker tests if Docker not available
    if "requires_docker" in [mark.name for mark in item.iter_markers()]:
        try:
            import docker
            client = docker.from_env()
            client.ping()
        except Exception:
            pytest.skip("Docker not available")


# ============================================================================
# Core Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "data_handler": {
            "type": "backtest",
            "file_path": "/tmp/test_data.csv"
        },
        "matrix_assemblers": {
            "30m": {
                "window_size": 48,
                "features": ["mlmi_value", "mlmi_signal", "nwrqk_value"]
            },
            "5m": {
                "window_size": 60,
                "features": ["fvg_bullish_active", "fvg_bearish_active"]
            }
        },
        "strategic_marl": {
            "enabled": True,
            "n_agents": 3,
            "learning_rate": 0.001
        }
    }


# ============================================================================
# Event System Fixtures
# ============================================================================

@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for testing."""
    if not HAS_CORE:
        pytest.skip("Core components not available")
    return EventBus()


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus for testing."""
    mock_bus = Mock()
    mock_bus.publish = Mock()
    mock_bus.subscribe = Mock()
    mock_bus.unsubscribe = Mock()
    mock_bus.start = Mock()
    mock_bus.stop = Mock()
    return mock_bus


# ============================================================================
# Kernel and Component Fixtures
# ============================================================================

@pytest.fixture
def mock_kernel(mock_config, mock_event_bus):
    """Create a mock AlgoSpaceKernel for testing."""
    kernel = Mock(spec=AlgoSpaceKernel)
    kernel.config = mock_config
    kernel.event_bus = mock_event_bus
    kernel.components = {}
    kernel.running = False
    kernel.get_component = Mock(return_value=None)
    kernel.get_event_bus = Mock(return_value=mock_event_bus)
    return kernel


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate 1000 data points (about 4 days of 5-minute bars)
    n_bars = 1000
    start_time = datetime.now() - timedelta(days=4)
    
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_bars)]
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    prices = []
    
    for i in range(n_bars):
        if i == 0:
            open_price = base_price
        else:
            open_price = prices[-1]["close"]
        
        # Random walk with slight upward bias
        price_change = np.random.normal(0.01, 0.5)
        close_price = open_price * (1 + price_change / 100)
        
        # High and low around open/close
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.2)) / 100)
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.2)) / 100)
        
        # Volume
        volume = int(np.random.lognormal(8, 1))
        
        prices.append({
            "timestamp": timestamps[i],
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": volume
        })
    
    return pd.DataFrame(prices)


# ============================================================================
# MARL Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_marl_env():
    """Create a mock MARL environment for testing."""
    from unittest.mock import Mock
    
    env = Mock()
    env.reset = Mock(return_value={
        "strategic_agent": [0.0] * 48,  # 48 features for 30m
        "tactical_agent": [0.0] * 60,   # 60 features for 5m
        "risk_agent": [0.0] * 20        # 20 risk features
    })
    env.step = Mock(return_value=(
        {"agent": [0.0] * 48},  # observations
        {"agent": 0.0},         # rewards
        {"agent": False},       # dones
        {"agent": {}}          # infos
    ))
    env.observation_space = Mock()
    env.action_space = Mock()
    env.possible_agents = ["strategic_agent", "tactical_agent", "risk_agent"]
    env.agents = env.possible_agents
    return env


@pytest.fixture
def mock_strategic_agent():
    """Create a mock strategic agent for testing."""
    agent = Mock()
    agent.act = Mock(return_value={"position": 0.0, "confidence": 0.5})
    agent.learn = Mock()
    agent.update = Mock()
    agent.save = Mock()
    agent.load = Mock()
    return agent


@pytest.fixture
def mock_tactical_agent():
    """Create a mock tactical agent for testing."""
    agent = Mock()
    agent.act = Mock(return_value={"entry_signal": 0.0, "exit_signal": 0.0})
    agent.learn = Mock()
    agent.update = Mock()
    agent.save = Mock()
    agent.load = Mock()
    return agent


@pytest.fixture
def mock_risk_agent():
    """Create a mock risk agent for testing."""
    agent = Mock()
    agent.act = Mock(return_value={"position_size": 0.01, "stop_loss": 0.02})
    agent.learn = Mock()
    agent.update = Mock()
    agent.save = Mock()
    agent.load = Mock()
    return agent


# ============================================================================
# Matrix Assembler Fixtures
# ============================================================================

@pytest.fixture
def mock_matrix_assembler_30m():
    """Create a mock 30-minute matrix assembler."""
    assembler = Mock()
    assembler.window_size = 48
    assembler.features = ["mlmi_value", "mlmi_signal", "nwrqk_value"]
    assembler.assemble = Mock(return_value=[[0.0] * 48])
    assembler.on_indicators_ready = Mock()
    return assembler


@pytest.fixture
def mock_matrix_assembler_5m():
    """Create a mock 5-minute matrix assembler."""
    assembler = Mock()
    assembler.window_size = 60
    assembler.features = ["fvg_bullish_active", "fvg_bearish_active"]
    assembler.assemble = Mock(return_value=[[0.0] * 60])
    assembler.on_indicators_ready = Mock()
    return assembler


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "max_inference_time_ms": 50,
        "max_memory_usage_mb": 512,
        "min_throughput_ops_per_sec": 1000,
        "test_duration_seconds": 30
    }


@pytest.fixture
def memory_profiler():
    """Memory profiling utilities."""
    import psutil
    import os
    
    class MemoryProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
        
        def start(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_current_usage(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_peak_usage(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_memory_delta(self):
            if self.start_memory is None:
                return 0
            return self.get_current_usage() - self.start_memory
    
    return MemoryProfiler()


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_event(event_type: str, data: Dict[str, Any] = None):
    """Create a sample event for testing."""
    from src.core.events import Event
    
    return Event(
        type=event_type,
        data=data or {},
        timestamp=pytest.approx(time.time(), rel=1e-3)
    )


def assert_event_published(mock_event_bus, expected_event_type: str, expected_data: Dict[str, Any] = None):
    """Assert that an event was published to the event bus."""
    mock_event_bus.publish.assert_called()
    
    # Check the most recent call
    call_args = mock_event_bus.publish.call_args
    event = call_args[0][0]
    
    assert event.type == expected_event_type
    if expected_data:
        assert event.data == expected_data


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        def elapsed_ms(self):
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return (self.end_time - self.start_time) * 1000
    
    return Timer()


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_indicators():
    """Sample indicator data for testing."""
    return {
        "mlmi_value": 0.75,
        "mlmi_signal": 1.0,
        "nwrqk_value": 0.25,
        "nwrqk_slope": 0.1,
        "fvg_bullish_active": 2,
        "fvg_bearish_active": 0,
        "fvg_nearest_level": 100.5,
        "lvn_distance_points": 5.2,
        "lvn_nearest_strength": 0.8
    }


@pytest.fixture 
def sample_30m_matrix():
    """Sample 30-minute matrix data."""
    import numpy as np
    return np.random.rand(48, 8)  # 48 bars, 8 features


@pytest.fixture
def sample_5m_matrix():
    """Sample 5-minute matrix data."""
    import numpy as np
    return np.random.rand(60, 7)  # 60 bars, 7 features


# Note: event_loop fixture already defined above with session scope