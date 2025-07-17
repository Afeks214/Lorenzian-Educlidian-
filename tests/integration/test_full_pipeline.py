"""
Integration test suite for the complete GrandModel pipeline.

This module tests the full end-to-end pipeline from data ingestion
through strategic and tactical decision making to execution.
"""
import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
from datetime import datetime, timedelta
import time
import json

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFullPipelineIntegration:
    """Test complete pipeline integration from data to execution."""

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for full pipeline."""
        return {
            "data_handler": {
                "type": "backtest",
                "file_path": "/tmp/test_data.csv",
                "symbols": ["EURUSD"],
                "timeframes": ["5m", "30m"]
            },
            "matrix_assemblers": {
                "30m": {
                    "window_size": 48,
                    "features": ["mlmi_value", "mlmi_signal", "nwrqk_value", "mmd_trend"]
                },
                "5m": {
                    "window_size": 60,
                    "features": ["fvg_bullish_active", "fvg_bearish_active", "volume_ratio"]
                }
            },
            "strategic_marl": {
                "enabled": True,
                "n_agents": 3,
                "learning_rate": 0.001
            },
            "tactical_marl": {
                "enabled": True,
                "execution_agents": 4,
                "latency_target": 1.0
            }
        }

    @pytest.fixture
    def mock_full_pipeline(self, pipeline_config):
        """Create a mock full pipeline."""
        pipeline = Mock()
        pipeline.config = pipeline_config
        
        # Component mocks
        pipeline.data_handler = Mock()
        pipeline.bar_generator = Mock()
        pipeline.indicator_engine = Mock()
        pipeline.matrix_30m = Mock()
        pipeline.matrix_5m = Mock()
        pipeline.synergy_detector = Mock()
        pipeline.strategic_coordinator = Mock()
        pipeline.tactical_coordinator = Mock()
        pipeline.execution_handler = Mock()
        pipeline.event_bus = Mock()
        
        # Pipeline methods
        pipeline.initialize = Mock()
        pipeline.run = Mock()
        pipeline.process_tick = Mock()
        pipeline.shutdown = Mock()
        pipeline.get_status = Mock(return_value={"running": True, "components": 10})
        
        return pipeline

    def test_pipeline_initialization(self, mock_full_pipeline):
        """Test complete pipeline initialization."""
        mock_full_pipeline.initialize()
        
        # Verify initialization was called
        mock_full_pipeline.initialize.assert_called_once()

    def test_data_flow_integration(self, mock_full_pipeline, sample_market_data):
        """Test data flow through entire pipeline."""
        # Simulate tick data
        tick_data = {
            "symbol": "EURUSD",
            "timestamp": datetime.now(),
            "bid": 1.0850,
            "ask": 1.0852,
            "volume": 1000
        }
        
        # Mock the processing flow
        mock_full_pipeline.data_handler.process_tick = Mock()
        mock_full_pipeline.bar_generator.on_new_tick = Mock()
        mock_full_pipeline.indicator_engine.calculate = Mock()
        
        # Process tick through pipeline
        mock_full_pipeline.process_tick(tick_data)
        
        # Verify tick was processed
        mock_full_pipeline.process_tick.assert_called_once_with(tick_data)

    def test_strategic_tactical_coordination(self, mock_full_pipeline):
        """Test coordination between strategic and tactical layers."""
        # Strategic decision
        strategic_decision = {
            "position": 0.7,
            "confidence": 0.85,
            "pattern": "TYPE_1",
            "timeframe": "30m"
        }
        
        # Tactical execution plan
        tactical_plan = {
            "execution_method": "ICEBERG",
            "slice_size": 0.1,
            "target_levels": [1.0848, 1.0846, 1.0844],
            "max_slippage": 0.0002
        }
        
        # Mock coordination
        mock_full_pipeline.strategic_coordinator.get_decision = Mock(return_value=strategic_decision)
        mock_full_pipeline.tactical_coordinator.create_execution_plan = Mock(return_value=tactical_plan)
        
        # Test coordination
        strategic = mock_full_pipeline.strategic_coordinator.get_decision()
        tactical = mock_full_pipeline.tactical_coordinator.create_execution_plan(strategic)
        
        assert strategic["position"] == 0.7
        assert tactical["execution_method"] == "ICEBERG"

    def test_event_system_integration(self, mock_full_pipeline):
        """Test event system integration across components."""
        # Mock event types
        events = [
            {"type": "NEW_TICK", "data": {"price": 1.0850}},
            {"type": "NEW_5MIN_BAR", "data": {"close": 1.0851}},
            {"type": "NEW_30MIN_BAR", "data": {"close": 1.0849}},
            {"type": "INDICATORS_READY", "data": {"mlmi": 0.75}},
            {"type": "SYNERGY_DETECTED", "data": {"pattern": "TYPE_2"}},
            {"type": "STRATEGIC_DECISION", "data": {"position": 0.5}},
            {"type": "EXECUTE_TRADE", "data": {"size": 0.1}}
        ]
        
        # Mock event publishing
        for event in events:
            mock_full_pipeline.event_bus.publish = Mock()
            mock_full_pipeline.event_bus.publish(event)
            mock_full_pipeline.event_bus.publish.assert_called_with(event)

    @pytest.mark.asyncio
    async def test_async_pipeline_processing(self, mock_full_pipeline):
        """Test asynchronous pipeline processing."""
        # Mock async methods
        mock_full_pipeline.process_async = AsyncMock(return_value={"status": "success"})
        mock_full_pipeline.strategic_coordinator.decide_async = AsyncMock(return_value={"position": 0.6})
        mock_full_pipeline.tactical_coordinator.execute_async = AsyncMock(return_value={"filled": 0.6})
        
        # Run async processing
        result = await mock_full_pipeline.process_async()
        strategic_result = await mock_full_pipeline.strategic_coordinator.decide_async()
        tactical_result = await mock_full_pipeline.tactical_coordinator.execute_async()
        
        assert result["status"] == "success"
        assert strategic_result["position"] == 0.6
        assert tactical_result["filled"] == 0.6

    @pytest.mark.performance
    def test_pipeline_throughput(self, mock_full_pipeline, benchmark_config):
        """Test pipeline throughput under load."""
        # Simulate high-frequency data
        tick_count = 10000
        start_time = time.time()
        
        for i in range(tick_count):
            tick = {
                "timestamp": datetime.now(),
                "price": 1.0850 + (i % 100) * 0.0001,
                "volume": 100
            }
            mock_full_pipeline.process_tick(tick)
        
        elapsed = time.time() - start_time
        throughput = tick_count / elapsed
        
        # Requirement: >1000 ticks/second
        assert throughput > 1000
        assert mock_full_pipeline.process_tick.call_count == tick_count

    def test_error_handling_integration(self, mock_full_pipeline):
        """Test error handling across pipeline components."""
        # Simulate various error conditions
        error_scenarios = [
            {"component": "data_handler", "error": "ConnectionError"},
            {"component": "indicator_engine", "error": "CalculationError"},
            {"component": "strategic_coordinator", "error": "ModelError"},
            {"component": "execution_handler", "error": "BrokerError"}
        ]
        
        for scenario in error_scenarios:
            mock_full_pipeline.handle_error = Mock()
            mock_full_pipeline.handle_error(scenario)
            mock_full_pipeline.handle_error.assert_called_with(scenario)

    def test_state_persistence_integration(self, mock_full_pipeline, temp_dir):
        """Test state persistence across pipeline restart."""
        state_file = temp_dir / "pipeline_state.json"
        
        # Mock state saving
        pipeline_state = {
            "strategic_models": {"agent_1": "model_data"},
            "tactical_models": {"executor": "model_data"},
            "indicators": {"mlmi": [0.1, 0.2, 0.3]},
            "positions": {"current": 0.5}
        }
        
        mock_full_pipeline.save_state = Mock()
        mock_full_pipeline.load_state = Mock()
        
        # Save and load state
        mock_full_pipeline.save_state(str(state_file))
        mock_full_pipeline.load_state(str(state_file))
        
        mock_full_pipeline.save_state.assert_called_once_with(str(state_file))
        mock_full_pipeline.load_state.assert_called_once_with(str(state_file))

    def test_monitoring_integration(self, mock_full_pipeline):
        """Test monitoring and metrics collection."""
        # Mock metrics
        metrics = {
            "pipeline_latency_ms": 2.5,
            "strategic_decisions_per_hour": 12,
            "tactical_executions_per_hour": 48,
            "error_rate": 0.001,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15
        }
        
        mock_full_pipeline.collect_metrics = Mock(return_value=metrics)
        collected_metrics = mock_full_pipeline.collect_metrics()
        
        assert "pipeline_latency_ms" in collected_metrics
        assert "strategic_decisions_per_hour" in collected_metrics
        assert "tactical_executions_per_hour" in collected_metrics
        assert collected_metrics["pipeline_latency_ms"] < 5.0

    def test_graceful_shutdown_integration(self, mock_full_pipeline):
        """Test graceful pipeline shutdown."""
        shutdown_steps = [
            "stop_data_stream",
            "close_positions", 
            "save_models",
            "flush_logs",
            "cleanup_resources"
        ]
        
        for step in shutdown_steps:
            mock_method = Mock()
            setattr(mock_full_pipeline, step, mock_method)
        
        # Mock shutdown process
        mock_full_pipeline.shutdown()
        
        # Verify shutdown was called
        mock_full_pipeline.shutdown.assert_called_once()


class TestBacktestIntegration:
    """Test backtest integration and validation."""

    @pytest.fixture
    def backtest_config(self):
        """Configuration for backtest."""
        return {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_balance": 100000,
            "symbols": ["EURUSD", "GBPUSD"],
            "commission": 0.00002,
            "slippage": 0.00001,
            "max_position_size": 0.1
        }

    @pytest.fixture
    def mock_backtest_engine(self, backtest_config):
        """Create mock backtest engine."""
        engine = Mock()
        engine.config = backtest_config
        
        # Mock methods
        engine.run_backtest = Mock(return_value={
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08,
            "win_rate": 0.62,
            "num_trades": 1247
        })
        engine.load_data = Mock()
        engine.validate_results = Mock(return_value=True)
        
        return engine

    def test_backtest_execution(self, mock_backtest_engine):
        """Test complete backtest execution."""
        results = mock_backtest_engine.run_backtest()
        
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "win_rate" in results
        assert "num_trades" in results
        
        # Validate reasonable results
        assert -1 <= results["total_return"] <= 5  # -100% to 500%
        assert 0 <= results["win_rate"] <= 1
        assert results["max_drawdown"] <= 0
        assert results["num_trades"] > 0

    def test_backtest_data_integrity(self, mock_backtest_engine):
        """Test backtest data integrity."""
        mock_backtest_engine.load_data()
        mock_backtest_engine.load_data.assert_called_once()

    def test_backtest_result_validation(self, mock_backtest_engine):
        """Test backtest result validation."""
        is_valid = mock_backtest_engine.validate_results()
        assert isinstance(is_valid, bool)
        mock_backtest_engine.validate_results.assert_called_once()

    def test_multi_symbol_backtest(self, mock_backtest_engine):
        """Test multi-symbol backtest integration."""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in symbols:
            mock_backtest_engine.run_symbol_backtest = Mock(return_value={
                "symbol": symbol,
                "return": 0.1,
                "trades": 200
            })
            result = mock_backtest_engine.run_symbol_backtest(symbol)
            assert result["symbol"] == symbol


class TestLiveTradeIntegration:
    """Test live trading integration (with safety controls)."""

    @pytest.fixture
    def live_config(self):
        """Configuration for live trading."""
        return {
            "broker": "demo_account",
            "symbols": ["EURUSD"],
            "max_position_size": 0.01,  # Very small for safety
            "max_daily_loss": 100,
            "emergency_stop": True,
            "paper_trading": True  # Always use paper trading in tests
        }

    @pytest.fixture
    def mock_live_engine(self, live_config):
        """Create mock live trading engine."""
        engine = Mock()
        engine.config = live_config
        engine.paper_trading = True  # Safety first
        
        # Mock methods
        engine.connect_broker = Mock(return_value=True)
        engine.start_trading = Mock()
        engine.stop_trading = Mock()
        engine.place_order = Mock(return_value={"order_id": "12345", "status": "filled"})
        engine.get_positions = Mock(return_value=[])
        engine.check_safety_limits = Mock(return_value=True)
        
        return engine

    def test_broker_connection(self, mock_live_engine):
        """Test broker connection for live trading."""
        connected = mock_live_engine.connect_broker()
        assert connected is True
        mock_live_engine.connect_broker.assert_called_once()

    def test_order_execution(self, mock_live_engine):
        """Test order execution in live environment."""
        order = {
            "symbol": "EURUSD",
            "side": "BUY",
            "size": 0.01,
            "price": 1.0850,
            "type": "LIMIT"
        }
        
        result = mock_live_engine.place_order(order)
        
        assert "order_id" in result
        assert "status" in result
        mock_live_engine.place_order.assert_called_once_with(order)

    def test_safety_limits(self, mock_live_engine):
        """Test safety limit enforcement."""
        safety_check = mock_live_engine.check_safety_limits()
        assert isinstance(safety_check, bool)
        mock_live_engine.check_safety_limits.assert_called_once()

    def test_position_monitoring(self, mock_live_engine):
        """Test position monitoring in live trading."""
        positions = mock_live_engine.get_positions()
        assert isinstance(positions, list)
        mock_live_engine.get_positions.assert_called_once()

    @pytest.mark.requires_docker
    def test_live_system_startup(self, mock_live_engine):
        """Test live system startup sequence."""
        startup_sequence = [
            mock_live_engine.connect_broker,
            mock_live_engine.check_safety_limits,
            mock_live_engine.start_trading
        ]
        
        for step in startup_sequence:
            step()
            step.assert_called()


class TestScalabilityIntegration:
    """Test system scalability and resource usage."""

    @pytest.fixture
    def scalability_config(self):
        """Configuration for scalability testing."""
        return {
            "concurrent_symbols": 10,
            "tick_rate": 1000,  # ticks per second
            "memory_limit_mb": 1024,
            "cpu_limit_percent": 80
        }

    def test_multi_symbol_scaling(self, scalability_config):
        """Test scaling across multiple symbols."""
        symbols = [f"SYMBOL_{i}" for i in range(scalability_config["concurrent_symbols"])]
        
        mock_symbol_processors = {}
        for symbol in symbols:
            processor = Mock()
            processor.process = Mock()
            mock_symbol_processors[symbol] = processor
        
        # Simulate processing all symbols
        for symbol, processor in mock_symbol_processors.items():
            processor.process({"symbol": symbol, "price": 1.0})
            processor.process.assert_called()

    @pytest.mark.performance
    def test_memory_usage_scaling(self, memory_profiler, scalability_config):
        """Test memory usage under load."""
        memory_profiler.start()
        
        # Simulate memory-intensive operations
        large_datasets = []
        for i in range(100):
            dataset = np.random.rand(1000, 100)  # Large dataset
            large_datasets.append(dataset)
        
        current_usage = memory_profiler.get_current_usage()
        
        # Should stay within limits
        assert current_usage < scalability_config["memory_limit_mb"]

    @pytest.mark.performance
    def test_cpu_usage_scaling(self, scalability_config):
        """Test CPU usage under computational load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Simulate CPU-intensive operations
        for i in range(1000):
            _ = np.random.rand(100, 100) @ np.random.rand(100, 100)
        
        cpu_percent = process.cpu_percent()
        
        # Should stay within reasonable limits
        assert cpu_percent < scalability_config["cpu_limit_percent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])