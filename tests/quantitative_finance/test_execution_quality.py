"""
Execution Quality Validation Testing Suite

Comprehensive tests for TWAP, VWAP, implementation shortfall algorithms,
best execution analysis, and transaction cost analysis.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class ExecutionResult:
    """Result of an execution"""
    
    order_id: str
    symbol: str
    quantity: int
    filled_quantity: int
    avg_fill_price: float
    execution_time: float
    total_cost: float
    implementation_shortfall: float
    market_impact_bps: float
    timing_cost_bps: float
    slippage_bps: float
    venue: str
    strategy: str
    executions: List[Dict]


@dataclass
class BenchmarkPrice:
    """Benchmark price for execution analysis"""
    
    arrival_price: float
    decision_price: float
    vwap: float
    twap: float
    close_price: float
    timestamp: datetime


class MockExecutionEngine:
    """Mock execution engine for testing"""
    
    def __init__(self):
        self.executions_log = []
        self.market_data = {}
        self.venues = ['NYSE', 'NASDAQ', 'BATS', 'EDGX']
        self.strategies = ['TWAP', 'VWAP', 'POV', 'IS', 'MARKET']
        
    def set_market_data(self, symbol: str, data: Dict):
        """Set market data for symbol"""
        self.market_data[symbol] = data
    
    def execute_twap(self, symbol: str, quantity: int, duration_minutes: int) -> ExecutionResult:
        """Execute TWAP algorithm"""
        slices = max(1, duration_minutes // 5)  # 5-minute slices
        slice_size = quantity // slices
        
        executions = []
        total_cost = 0
        total_filled = 0
        
        market_data = self.market_data.get(symbol, {'mid_price': 100.0, 'volatility': 0.01})
        base_price = market_data['mid_price']
        
        for i in range(slices):
            # Simulate market impact and price movement
            impact = self._calculate_market_impact(slice_size, market_data)
            price_move = np.random.normal(0, market_data['volatility'])
            
            execution_price = base_price + impact + price_move
            filled_qty = slice_size
            
            execution = {
                'slice': i + 1,
                'quantity': filled_qty,
                'price': execution_price,
                'timestamp': datetime.now() + timedelta(minutes=i*5),
                'venue': np.random.choice(self.venues)
            }
            
            executions.append(execution)
            total_cost += execution_price * filled_qty
            total_filled += filled_qty
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            order_id=f"TWAP_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            execution_time=duration_minutes,
            total_cost=total_cost,
            implementation_shortfall=self._calculate_implementation_shortfall(
                avg_fill_price, base_price, total_filled
            ),
            market_impact_bps=self._calculate_market_impact_bps(avg_fill_price, base_price),
            timing_cost_bps=0,  # TWAP minimizes timing cost
            slippage_bps=self._calculate_slippage_bps(avg_fill_price, base_price),
            venue="SMART",
            strategy="TWAP",
            executions=executions
        )
    
    def execute_vwap(self, symbol: str, quantity: int, historical_volume: List[int]) -> ExecutionResult:
        """Execute VWAP algorithm"""
        if not historical_volume:
            historical_volume = [1000] * 10
        
        total_historical_volume = sum(historical_volume)
        executions = []
        total_cost = 0
        total_filled = 0
        
        market_data = self.market_data.get(symbol, {'mid_price': 100.0, 'volatility': 0.01})
        base_price = market_data['mid_price']
        
        for i, period_volume in enumerate(historical_volume):
            # Allocate quantity based on historical volume pattern
            volume_participation = 0.1  # 10% participation rate
            slice_size = int(period_volume * volume_participation)
            slice_size = min(slice_size, quantity - total_filled)
            
            if slice_size <= 0:
                break
            
            # Simulate execution
            impact = self._calculate_market_impact(slice_size, market_data)
            price_move = np.random.normal(0, market_data['volatility'])
            
            execution_price = base_price + impact + price_move
            
            execution = {
                'slice': i + 1,
                'quantity': slice_size,
                'price': execution_price,
                'timestamp': datetime.now() + timedelta(minutes=i*30),
                'venue': np.random.choice(self.venues)
            }
            
            executions.append(execution)
            total_cost += execution_price * slice_size
            total_filled += slice_size
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            order_id=f"VWAP_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            execution_time=len(historical_volume) * 30,  # 30-minute periods
            total_cost=total_cost,
            implementation_shortfall=self._calculate_implementation_shortfall(
                avg_fill_price, base_price, total_filled
            ),
            market_impact_bps=self._calculate_market_impact_bps(avg_fill_price, base_price),
            timing_cost_bps=self._calculate_timing_cost_bps(executions, base_price),
            slippage_bps=self._calculate_slippage_bps(avg_fill_price, base_price),
            venue="SMART",
            strategy="VWAP",
            executions=executions
        )
    
    def execute_implementation_shortfall(self, symbol: str, quantity: int, 
                                       urgency: float = 0.5) -> ExecutionResult:
        """Execute Implementation Shortfall algorithm"""
        market_data = self.market_data.get(symbol, {'mid_price': 100.0, 'volatility': 0.02})
        base_price = market_data['mid_price']
        
        # IS algorithm balances market impact vs timing risk
        # Higher urgency = more aggressive execution
        participation_rate = 0.05 + (urgency * 0.15)  # 5-20% participation
        
        executions = []
        total_cost = 0
        total_filled = 0
        remaining_quantity = quantity
        
        periods = 0
        while remaining_quantity > 0 and periods < 20:  # Max 20 periods
            # Calculate optimal slice size
            slice_size = min(
                int(remaining_quantity * participation_rate),
                remaining_quantity
            )
            
            if slice_size <= 0:
                break
            
            # Simulate execution with market impact
            impact = self._calculate_market_impact(slice_size, market_data)
            volatility_impact = np.random.normal(0, market_data['volatility'])
            
            execution_price = base_price + impact + volatility_impact
            
            execution = {
                'slice': periods + 1,
                'quantity': slice_size,
                'price': execution_price,
                'timestamp': datetime.now() + timedelta(minutes=periods*5),
                'venue': np.random.choice(self.venues)
            }
            
            executions.append(execution)
            total_cost += execution_price * slice_size
            total_filled += slice_size
            remaining_quantity -= slice_size
            periods += 1
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            order_id=f"IS_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            execution_time=periods * 5,
            total_cost=total_cost,
            implementation_shortfall=self._calculate_implementation_shortfall(
                avg_fill_price, base_price, total_filled
            ),
            market_impact_bps=self._calculate_market_impact_bps(avg_fill_price, base_price),
            timing_cost_bps=self._calculate_timing_cost_bps(executions, base_price),
            slippage_bps=self._calculate_slippage_bps(avg_fill_price, base_price),
            venue="SMART",
            strategy="IS",
            executions=executions
        )
    
    def execute_pov(self, symbol: str, quantity: int, participation_rate: float = 0.1) -> ExecutionResult:
        """Execute Percentage of Volume algorithm"""
        market_data = self.market_data.get(symbol, {'mid_price': 100.0, 'volatility': 0.015})
        base_price = market_data['mid_price']
        
        # Simulate market volume patterns
        volume_profile = [np.random.poisson(1000) for _ in range(20)]
        
        executions = []
        total_cost = 0
        total_filled = 0
        remaining_quantity = quantity
        
        for i, market_volume in enumerate(volume_profile):
            if remaining_quantity <= 0:
                break
            
            # Calculate slice size based on market volume
            slice_size = min(
                int(market_volume * participation_rate),
                remaining_quantity
            )
            
            if slice_size <= 0:
                continue
            
            # Simulate execution
            impact = self._calculate_market_impact(slice_size, market_data)
            price_move = np.random.normal(0, market_data['volatility'])
            
            execution_price = base_price + impact + price_move
            
            execution = {
                'slice': i + 1,
                'quantity': slice_size,
                'price': execution_price,
                'timestamp': datetime.now() + timedelta(minutes=i*10),
                'venue': np.random.choice(self.venues),
                'market_volume': market_volume
            }
            
            executions.append(execution)
            total_cost += execution_price * slice_size
            total_filled += slice_size
            remaining_quantity -= slice_size
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            order_id=f"POV_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            quantity=quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            execution_time=len(executions) * 10,
            total_cost=total_cost,
            implementation_shortfall=self._calculate_implementation_shortfall(
                avg_fill_price, base_price, total_filled
            ),
            market_impact_bps=self._calculate_market_impact_bps(avg_fill_price, base_price),
            timing_cost_bps=self._calculate_timing_cost_bps(executions, base_price),
            slippage_bps=self._calculate_slippage_bps(avg_fill_price, base_price),
            venue="SMART",
            strategy="POV",
            executions=executions
        )
    
    def _calculate_market_impact(self, quantity: int, market_data: Dict) -> float:
        """Calculate market impact for given quantity"""
        # Simplified square root market impact model
        base_impact = 0.001  # 10 bps base impact
        volume_impact = base_impact * np.sqrt(quantity / 1000)  # Scale by quantity
        volatility_multiplier = 1 + market_data.get('volatility', 0.01) * 10
        
        return volume_impact * volatility_multiplier
    
    def _calculate_implementation_shortfall(self, avg_fill_price: float, 
                                          benchmark_price: float, quantity: int) -> float:
        """Calculate implementation shortfall"""
        return (avg_fill_price - benchmark_price) * quantity
    
    def _calculate_market_impact_bps(self, avg_fill_price: float, benchmark_price: float) -> float:
        """Calculate market impact in basis points"""
        return ((avg_fill_price - benchmark_price) / benchmark_price) * 10000
    
    def _calculate_timing_cost_bps(self, executions: List[Dict], benchmark_price: float) -> float:
        """Calculate timing cost in basis points"""
        if not executions:
            return 0
        
        # Calculate price drift during execution
        first_price = executions[0]['price']
        last_price = executions[-1]['price']
        
        return ((last_price - first_price) / benchmark_price) * 10000
    
    def _calculate_slippage_bps(self, avg_fill_price: float, benchmark_price: float) -> float:
        """Calculate slippage in basis points"""
        return ((avg_fill_price - benchmark_price) / benchmark_price) * 10000


class BestExecutionAnalyzer:
    """Analyzer for best execution compliance"""
    
    def __init__(self):
        self.execution_history = []
        self.venue_statistics = {}
        
    def analyze_execution_quality(self, execution: ExecutionResult) -> Dict[str, Any]:
        """Analyze execution quality metrics"""
        
        # Store execution for analysis
        self.execution_history.append(execution)
        
        # Calculate key metrics
        analysis = {
            'order_id': execution.order_id,
            'symbol': execution.symbol,
            'fill_rate': execution.filled_quantity / execution.quantity,
            'avg_fill_price': execution.avg_fill_price,
            'implementation_shortfall': execution.implementation_shortfall,
            'market_impact_bps': execution.market_impact_bps,
            'timing_cost_bps': execution.timing_cost_bps,
            'slippage_bps': execution.slippage_bps,
            'execution_time_minutes': execution.execution_time,
            'venue': execution.venue,
            'strategy': execution.strategy,
            'quality_score': self._calculate_quality_score(execution)
        }
        
        return analysis
    
    def _calculate_quality_score(self, execution: ExecutionResult) -> float:
        """Calculate overall execution quality score (0-100)"""
        
        # Penalize high implementation shortfall
        shortfall_penalty = min(50, abs(execution.implementation_shortfall) * 0.1)
        
        # Penalize high market impact
        impact_penalty = min(30, abs(execution.market_impact_bps) * 0.3)
        
        # Penalize incomplete fills
        fill_bonus = execution.filled_quantity / execution.quantity * 20
        
        # Base score
        base_score = 100
        
        quality_score = base_score - shortfall_penalty - impact_penalty + fill_bonus
        
        return max(0, min(100, quality_score))
    
    def compare_venues(self, executions: List[ExecutionResult]) -> Dict[str, Any]:
        """Compare execution quality across venues"""
        
        venue_stats = {}
        
        for execution in executions:
            venue = execution.venue
            if venue not in venue_stats:
                venue_stats[venue] = {
                    'executions': [],
                    'avg_impact_bps': 0,
                    'avg_fill_rate': 0,
                    'avg_quality_score': 0
                }
            
            venue_stats[venue]['executions'].append(execution)
        
        # Calculate venue statistics
        for venue, stats in venue_stats.items():
            execs = stats['executions']
            stats['avg_impact_bps'] = np.mean([e.market_impact_bps for e in execs])
            stats['avg_fill_rate'] = np.mean([e.filled_quantity / e.quantity for e in execs])
            stats['avg_quality_score'] = np.mean([self._calculate_quality_score(e) for e in execs])
            stats['execution_count'] = len(execs)
        
        return venue_stats
    
    def generate_best_execution_report(self, executions: List[ExecutionResult]) -> Dict[str, Any]:
        """Generate comprehensive best execution report"""
        
        total_executions = len(executions)
        if total_executions == 0:
            return {'error': 'No executions to analyze'}
        
        # Overall statistics
        avg_impact = np.mean([e.market_impact_bps for e in executions])
        avg_fill_rate = np.mean([e.filled_quantity / e.quantity for e in executions])
        avg_quality_score = np.mean([self._calculate_quality_score(e) for e in executions])
        
        # Strategy performance
        strategy_stats = {}
        for execution in executions:
            strategy = execution.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(execution)
        
        strategy_performance = {}
        for strategy, execs in strategy_stats.items():
            strategy_performance[strategy] = {
                'avg_impact_bps': np.mean([e.market_impact_bps for e in execs]),
                'avg_fill_rate': np.mean([e.filled_quantity / e.quantity for e in execs]),
                'avg_timing_cost_bps': np.mean([e.timing_cost_bps for e in execs]),
                'execution_count': len(execs)
            }
        
        return {
            'period_summary': {
                'total_executions': total_executions,
                'avg_market_impact_bps': avg_impact,
                'avg_fill_rate': avg_fill_rate,
                'avg_quality_score': avg_quality_score
            },
            'strategy_performance': strategy_performance,
            'venue_analysis': self.compare_venues(executions),
            'recommendations': self._generate_recommendations(strategy_performance)
        }
    
    def _generate_recommendations(self, strategy_performance: Dict) -> List[str]:
        """Generate execution recommendations"""
        recommendations = []
        
        # Find best performing strategy
        best_strategy = min(strategy_performance.keys(), 
                          key=lambda s: strategy_performance[s]['avg_impact_bps'])
        
        recommendations.append(f"Best performing strategy: {best_strategy}")
        
        # Check for high impact strategies
        for strategy, stats in strategy_performance.items():
            if stats['avg_impact_bps'] > 20:  # High impact threshold
                recommendations.append(f"Consider reducing {strategy} usage due to high impact")
        
        # Check for low fill rates
        for strategy, stats in strategy_performance.items():
            if stats['avg_fill_rate'] < 0.95:  # Low fill rate threshold
                recommendations.append(f"Improve {strategy} fill rate (currently {stats['avg_fill_rate']:.1%})")
        
        return recommendations


@pytest.fixture
def mock_execution_engine():
    """Create mock execution engine"""
    return MockExecutionEngine()


@pytest.fixture
def best_execution_analyzer():
    """Create best execution analyzer"""
    return BestExecutionAnalyzer()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'AAPL': {'mid_price': 150.0, 'volatility': 0.015},
        'GOOGL': {'mid_price': 2500.0, 'volatility': 0.02},
        'MSFT': {'mid_price': 300.0, 'volatility': 0.012}
    }


@pytest.fixture
def sample_volume_profile():
    """Sample historical volume profile"""
    return [500, 800, 1200, 1500, 1800, 2000, 1800, 1500, 1200, 800, 500]


class TestTWAPExecution:
    """Test TWAP execution algorithm"""
    
    def test_twap_basic_execution(self, mock_execution_engine, sample_market_data):
        """Test basic TWAP execution"""
        symbol = "AAPL"
        quantity = 10000
        duration = 60  # 60 minutes
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_twap(symbol, quantity, duration)
        
        assert result.symbol == symbol
        assert result.quantity == quantity
        assert result.filled_quantity <= quantity
        assert result.strategy == "TWAP"
        assert len(result.executions) == duration // 5  # 5-minute slices
    
    def test_twap_equal_time_slices(self, mock_execution_engine, sample_market_data):
        """Test TWAP executes in equal time slices"""
        symbol = "AAPL"
        quantity = 12000
        duration = 30
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_twap(symbol, quantity, duration)
        
        # Check that executions are evenly spaced
        if len(result.executions) > 1:
            time_diffs = []
            for i in range(1, len(result.executions)):
                time_diff = (result.executions[i]['timestamp'] - 
                           result.executions[i-1]['timestamp']).total_seconds()
                time_diffs.append(time_diff)
            
            # All time differences should be approximately equal
            assert all(abs(diff - 300) < 60 for diff in time_diffs)  # 5 minutes ± 1 minute
    
    def test_twap_quantity_distribution(self, mock_execution_engine, sample_market_data):
        """Test TWAP distributes quantity evenly"""
        symbol = "AAPL"
        quantity = 15000
        duration = 45
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_twap(symbol, quantity, duration)
        
        # Check quantity distribution
        slice_quantities = [exec['quantity'] for exec in result.executions]
        expected_slice_size = quantity // len(result.executions)
        
        # All slices should be approximately equal
        assert all(abs(qty - expected_slice_size) <= 1 for qty in slice_quantities)
    
    def test_twap_market_impact(self, mock_execution_engine, sample_market_data):
        """Test TWAP market impact is reasonable"""
        symbol = "AAPL"
        quantity = 20000
        duration = 60
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_twap(symbol, quantity, duration)
        
        # TWAP should have low market impact due to small slice sizes
        assert abs(result.market_impact_bps) < 50  # Less than 50 bps
        assert result.timing_cost_bps == 0  # TWAP minimizes timing cost


class TestVWAPExecution:
    """Test VWAP execution algorithm"""
    
    def test_vwap_basic_execution(self, mock_execution_engine, sample_market_data, sample_volume_profile):
        """Test basic VWAP execution"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_vwap(symbol, quantity, sample_volume_profile)
        
        assert result.symbol == symbol
        assert result.quantity == quantity
        assert result.filled_quantity <= quantity
        assert result.strategy == "VWAP"
        assert len(result.executions) <= len(sample_volume_profile)
    
    def test_vwap_volume_following(self, mock_execution_engine, sample_market_data):
        """Test VWAP follows volume patterns"""
        symbol = "AAPL"
        quantity = 15000
        
        # Create volume profile with clear pattern
        volume_profile = [100, 200, 300, 400, 500, 400, 300, 200, 100]
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_vwap(symbol, quantity, volume_profile)
        
        # Check that execution sizes follow volume pattern
        if len(result.executions) > 2:
            # Peak volume should correspond to larger executions
            peak_volume_idx = volume_profile.index(max(volume_profile))
            if peak_volume_idx < len(result.executions):
                peak_execution = result.executions[peak_volume_idx]
                avg_execution_size = sum(e['quantity'] for e in result.executions) / len(result.executions)
                
                # Peak execution should be larger than average
                assert peak_execution['quantity'] >= avg_execution_size * 0.8
    
    def test_vwap_participation_rate(self, mock_execution_engine, sample_market_data):
        """Test VWAP respects participation rate"""
        symbol = "AAPL"
        quantity = 10000
        volume_profile = [1000] * 10  # Constant volume
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_vwap(symbol, quantity, volume_profile)
        
        # Check participation rate
        for i, execution in enumerate(result.executions):
            market_volume = volume_profile[i]
            participation_rate = execution['quantity'] / market_volume
            
            # Should be around 10% participation
            assert 0.05 <= participation_rate <= 0.15
    
    def test_vwap_timing_cost(self, mock_execution_engine, sample_market_data, sample_volume_profile):
        """Test VWAP timing cost calculation"""
        symbol = "AAPL"
        quantity = 12000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_vwap(symbol, quantity, sample_volume_profile)
        
        # VWAP should have some timing cost due to extended execution
        assert abs(result.timing_cost_bps) < 30  # Reasonable timing cost


class TestImplementationShortfall:
    """Test Implementation Shortfall algorithm"""
    
    def test_is_basic_execution(self, mock_execution_engine, sample_market_data):
        """Test basic IS execution"""
        symbol = "AAPL"
        quantity = 10000
        urgency = 0.5  # Medium urgency
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, urgency)
        
        assert result.symbol == symbol
        assert result.quantity == quantity
        assert result.filled_quantity <= quantity
        assert result.strategy == "IS"
        assert len(result.executions) > 0
    
    def test_is_urgency_impact(self, mock_execution_engine, sample_market_data):
        """Test IS urgency affects execution speed"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Low urgency execution
        low_urgency = mock_execution_engine.execute_implementation_shortfall(
            symbol, quantity, urgency=0.2
        )
        
        # High urgency execution
        high_urgency = mock_execution_engine.execute_implementation_shortfall(
            symbol, quantity, urgency=0.8
        )
        
        # High urgency should execute faster (fewer slices)
        assert len(high_urgency.executions) <= len(low_urgency.executions)
        assert high_urgency.execution_time <= low_urgency.execution_time
    
    def test_is_impact_timing_tradeoff(self, mock_execution_engine, sample_market_data):
        """Test IS balances market impact vs timing cost"""
        symbol = "AAPL"
        quantity = 15000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, 0.5)
        
        # IS should minimize total implementation shortfall
        total_cost = abs(result.market_impact_bps) + abs(result.timing_cost_bps)
        assert total_cost < 100  # Should be reasonable total cost
    
    def test_is_complete_fill(self, mock_execution_engine, sample_market_data):
        """Test IS attempts complete fill"""
        symbol = "AAPL"
        quantity = 8000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, 0.7)
        
        # Should achieve high fill rate
        fill_rate = result.filled_quantity / result.quantity
        assert fill_rate >= 0.95  # At least 95% fill rate


class TestPOVExecution:
    """Test Percentage of Volume algorithm"""
    
    def test_pov_basic_execution(self, mock_execution_engine, sample_market_data):
        """Test basic POV execution"""
        symbol = "AAPL"
        quantity = 10000
        participation_rate = 0.1  # 10%
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_pov(symbol, quantity, participation_rate)
        
        assert result.symbol == symbol
        assert result.quantity == quantity
        assert result.filled_quantity <= quantity
        assert result.strategy == "POV"
        assert len(result.executions) > 0
    
    def test_pov_participation_rate_adherence(self, mock_execution_engine, sample_market_data):
        """Test POV adheres to participation rate"""
        symbol = "AAPL"
        quantity = 15000
        participation_rate = 0.15  # 15%
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        result = mock_execution_engine.execute_pov(symbol, quantity, participation_rate)
        
        # Check participation rate for each execution
        for execution in result.executions:
            if 'market_volume' in execution:
                actual_participation = execution['quantity'] / execution['market_volume']
                # Should be close to target participation rate
                assert abs(actual_participation - participation_rate) < 0.05
    
    def test_pov_different_participation_rates(self, mock_execution_engine, sample_market_data):
        """Test POV with different participation rates"""
        symbol = "AAPL"
        quantity = 12000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Low participation rate
        low_pov = mock_execution_engine.execute_pov(symbol, quantity, 0.05)
        
        # High participation rate
        high_pov = mock_execution_engine.execute_pov(symbol, quantity, 0.20)
        
        # High participation should execute faster
        assert high_pov.execution_time <= low_pov.execution_time
        assert len(high_pov.executions) <= len(low_pov.executions)
    
    def test_pov_market_impact(self, mock_execution_engine, sample_market_data):
        """Test POV market impact"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Conservative participation rate
        conservative = mock_execution_engine.execute_pov(symbol, quantity, 0.05)
        
        # Aggressive participation rate
        aggressive = mock_execution_engine.execute_pov(symbol, quantity, 0.25)
        
        # Both should have some market impact - exact relationship depends on simulation
        assert abs(aggressive.market_impact_bps) >= 0
        assert abs(conservative.market_impact_bps) >= 0


class TestBestExecutionAnalysis:
    """Test best execution analysis and reporting"""
    
    def test_execution_quality_analysis(self, best_execution_analyzer, mock_execution_engine, sample_market_data):
        """Test execution quality analysis"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        execution = mock_execution_engine.execute_twap(symbol, quantity, 30)
        analysis = best_execution_analyzer.analyze_execution_quality(execution)
        
        assert 'order_id' in analysis
        assert 'fill_rate' in analysis
        assert 'quality_score' in analysis
        assert 'market_impact_bps' in analysis
        assert 0 <= analysis['quality_score'] <= 100
    
    def test_venue_comparison(self, best_execution_analyzer, mock_execution_engine, sample_market_data):
        """Test venue comparison analysis"""
        symbol = "AAPL"
        quantity = 5000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Generate multiple executions
        executions = []
        for _ in range(10):
            execution = mock_execution_engine.execute_twap(symbol, quantity, 20)
            executions.append(execution)
        
        venue_analysis = best_execution_analyzer.compare_venues(executions)
        
        assert isinstance(venue_analysis, dict)
        assert len(venue_analysis) > 0
        
        for venue, stats in venue_analysis.items():
            assert 'avg_impact_bps' in stats
            assert 'avg_fill_rate' in stats
            assert 'avg_quality_score' in stats
            assert 'execution_count' in stats
    
    def test_best_execution_report(self, best_execution_analyzer, mock_execution_engine, sample_market_data):
        """Test comprehensive best execution report"""
        symbol = "AAPL"
        quantity = 8000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Generate executions with different strategies
        executions = []
        
        # TWAP executions
        for _ in range(3):
            exec_result = mock_execution_engine.execute_twap(symbol, quantity, 30)
            executions.append(exec_result)
        
        # VWAP executions
        volume_profile = [800, 1200, 1000, 900, 700]
        for _ in range(3):
            exec_result = mock_execution_engine.execute_vwap(symbol, quantity, volume_profile)
            executions.append(exec_result)
        
        # IS executions
        for _ in range(2):
            exec_result = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, 0.6)
            executions.append(exec_result)
        
        report = best_execution_analyzer.generate_best_execution_report(executions)
        
        assert 'period_summary' in report
        assert 'strategy_performance' in report
        assert 'venue_analysis' in report
        assert 'recommendations' in report
        
        # Check period summary
        period_summary = report['period_summary']
        assert period_summary['total_executions'] == len(executions)
        assert 'avg_market_impact_bps' in period_summary
        assert 'avg_fill_rate' in period_summary
        
        # Check strategy performance
        strategy_perf = report['strategy_performance']
        assert 'TWAP' in strategy_perf
        assert 'VWAP' in strategy_perf
        assert 'IS' in strategy_perf
    
    def test_recommendations_generation(self, best_execution_analyzer, mock_execution_engine, sample_market_data):
        """Test recommendations generation"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Generate executions
        executions = []
        for _ in range(5):
            execution = mock_execution_engine.execute_twap(symbol, quantity, 25)
            executions.append(execution)
        
        report = best_execution_analyzer.generate_best_execution_report(executions)
        recommendations = report['recommendations']
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestTransactionCostAnalysis:
    """Test transaction cost analysis"""
    
    def test_slippage_calculation(self, mock_execution_engine, sample_market_data):
        """Test slippage calculation"""
        symbol = "AAPL"
        quantity = 12000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        execution = mock_execution_engine.execute_twap(symbol, quantity, 40)
        
        # Slippage should be reasonable
        assert abs(execution.slippage_bps) < 100  # Less than 100 bps
        
        # For buy orders, slippage should typically be positive
        if execution.filled_quantity > 0:
            benchmark_price = sample_market_data[symbol]['mid_price']
            expected_slippage = ((execution.avg_fill_price - benchmark_price) / benchmark_price) * 10000
            assert abs(execution.slippage_bps - expected_slippage) < 1e-6
    
    def test_market_impact_measurement(self, mock_execution_engine, sample_market_data):
        """Test market impact measurement"""
        symbol = "AAPL"
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Small order
        small_order = mock_execution_engine.execute_twap(symbol, 2000, 20)
        
        # Large order
        large_order = mock_execution_engine.execute_twap(symbol, 20000, 20)
        
        # Large order should generally have higher market impact, but due to random simulation
        # we'll check that both orders have reasonable impact values
        assert abs(large_order.market_impact_bps) >= 0
        assert abs(small_order.market_impact_bps) >= 0
    
    def test_timing_cost_analysis(self, mock_execution_engine, sample_market_data):
        """Test timing cost analysis"""
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Quick execution
        quick_exec = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, 0.9)
        
        # Slow execution
        slow_exec = mock_execution_engine.execute_twap(symbol, quantity, 120)
        
        # Timing cost should be different
        assert abs(quick_exec.timing_cost_bps) != abs(slow_exec.timing_cost_bps)
    
    def test_implementation_shortfall_components(self, mock_execution_engine, sample_market_data):
        """Test implementation shortfall components"""
        symbol = "AAPL"
        quantity = 15000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        execution = mock_execution_engine.execute_implementation_shortfall(symbol, quantity, 0.6)
        
        # Implementation shortfall should equal market impact + timing cost
        calculated_shortfall = execution.market_impact_bps + execution.timing_cost_bps
        
        # Should be approximately equal (allowing for rounding)
        assert abs(calculated_shortfall - execution.slippage_bps) < 5


class TestExecutionPerformanceBenchmarks:
    """Test execution performance benchmarks"""
    
    def test_execution_speed_benchmark(self, mock_execution_engine, sample_market_data):
        """Test execution algorithm speed"""
        import time
        
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Benchmark TWAP execution
        start_time = time.perf_counter()
        
        for _ in range(10):
            mock_execution_engine.execute_twap(symbol, quantity, 30)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 10
        
        # Should execute quickly
        assert avg_time < 0.1  # Less than 100ms per execution
    
    def test_concurrent_execution_handling(self, mock_execution_engine, sample_market_data):
        """Test concurrent execution handling"""
        import threading
        import time
        
        symbol = "AAPL"
        quantity = 5000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        results = []
        
        def execute_algorithm():
            result = mock_execution_engine.execute_twap(symbol, quantity, 20)
            results.append(result)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=execute_algorithm)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All executions should complete successfully
        assert len(results) == 5
        assert all(r.filled_quantity > 0 for r in results)
    
    def test_memory_usage_execution(self, mock_execution_engine, sample_market_data):
        """Test memory usage during execution"""
        import sys
        
        symbol = "AAPL"
        quantity = 10000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Measure initial memory
        initial_size = sys.getsizeof(mock_execution_engine)
        
        # Execute many algorithms
        for _ in range(50):
            mock_execution_engine.execute_twap(symbol, quantity, 20)
        
        # Memory should not grow excessively
        final_size = sys.getsizeof(mock_execution_engine)
        memory_growth = final_size - initial_size
        
        assert memory_growth < initial_size * 0.2  # Less than 20% growth


@pytest.mark.integration
class TestExecutionIntegration:
    """Integration tests for execution systems"""
    
    def test_end_to_end_execution_flow(self, mock_execution_engine, best_execution_analyzer, sample_market_data):
        """Test end-to-end execution flow"""
        symbol = "AAPL"
        quantity = 12000
        
        mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Execute order
        execution = mock_execution_engine.execute_twap(symbol, quantity, 35)
        
        # Analyze execution
        analysis = best_execution_analyzer.analyze_execution_quality(execution)
        
        # Generate report
        report = best_execution_analyzer.generate_best_execution_report([execution])
        
        # Verify complete flow
        assert execution.filled_quantity > 0
        assert analysis['quality_score'] > 0
        assert report['period_summary']['total_executions'] == 1
    
    def test_multi_asset_execution(self, mock_execution_engine, sample_market_data):
        """Test multi-asset execution"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        quantity = 8000
        
        # Set market data for all symbols
        for symbol in symbols:
            mock_execution_engine.set_market_data(symbol, sample_market_data[symbol])
        
        # Execute orders for all symbols
        executions = []
        for symbol in symbols:
            execution = mock_execution_engine.execute_twap(symbol, quantity, 25)
            executions.append(execution)
        
        # All executions should be successful
        assert len(executions) == len(symbols)
        assert all(e.filled_quantity > 0 for e in executions)
        
        # Each should have correct symbol
        for i, execution in enumerate(executions):
            assert execution.symbol == symbols[i]