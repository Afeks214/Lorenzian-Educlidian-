"""
Realistic Execution Integration for Backtesting Framework
======================================================

AGENT 4 - REALISTIC EXECUTION ENGINE INTEGRATION

This module integrates the realistic execution engine with the backtesting framework
to eliminate backtest-live divergence by providing:

- Realistic slippage and commission modeling
- Dynamic market conditions simulation
- Order book depth and partial fill scenarios
- Market impact and execution timing
- Comprehensive execution cost modeling

Integration Features:
- Replaces simplified execution with realistic market simulation
- Maintains compatibility with existing backtesting framework
- Provides detailed execution analytics and reporting
- Supports various instrument types (NQ futures focus)

Author: AGENT 4 - Realistic Execution Engine Integration
Date: 2025-07-17
Mission: Eliminate backtest-live divergence through realistic execution
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Import realistic execution engine components
from execution.realistic_execution_engine import (
    RealisticExecutionEngine,
    ExecutionOrder,
    OrderSide,
    OrderType,
    OrderStatus,
    MarketConditions,
    ExecutionResult,
    NQFuturesSpecs
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestExecutionConfig:
    """Configuration for realistic execution in backtesting"""
    # Market simulation parameters
    enable_realistic_slippage: bool = True
    enable_market_impact: bool = True
    enable_execution_latency: bool = True
    enable_partial_fills: bool = True
    enable_order_book_simulation: bool = True
    
    # Cost modeling parameters
    use_dynamic_commission: bool = True
    include_exchange_fees: bool = True
    include_regulatory_fees: bool = True
    model_funding_costs: bool = True
    
    # Market condition parameters
    volatility_regime_factor: float = 1.0
    liquidity_adjustment_factor: float = 1.0
    stress_condition_probability: float = 0.1
    
    # Execution timing parameters
    min_execution_delay_ms: float = 50.0
    max_execution_delay_ms: float = 500.0
    network_latency_factor: float = 1.0


class RealisticBacktestExecutionHandler:
    """
    Realistic execution handler that integrates with backtesting framework
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 config: BacktestExecutionConfig = None):
        """
        Initialize realistic execution handler
        
        Args:
            initial_capital: Starting capital for backtesting
            config: Configuration for realistic execution
        """
        self.initial_capital = initial_capital
        self.config = config or BacktestExecutionConfig()
        
        # Initialize realistic execution engine
        self.execution_engine = RealisticExecutionEngine(
            account_value=initial_capital,
            config=self._create_engine_config()
        )
        
        # Track execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
            'total_slippage_cost': 0.0,
            'total_commission_paid': 0.0,
            'total_latency_ms': 0.0,
            'execution_quality_score': 0.0
        }
        
        # Market condition history
        self.market_conditions_history = []
        
        # Execution results cache
        self.execution_results = []
        
        logger.info("RealisticBacktestExecutionHandler initialized")
    
    def _create_engine_config(self) -> Dict[str, Any]:
        """Create configuration for realistic execution engine"""
        return {
            'enable_realistic_slippage': self.config.enable_realistic_slippage,
            'enable_market_impact': self.config.enable_market_impact,
            'enable_execution_latency': self.config.enable_execution_latency,
            'volatility_factor': self.config.volatility_regime_factor,
            'liquidity_factor': self.config.liquidity_adjustment_factor
        }
    
    async def execute_backtest_trade(self, 
                                   trade_data: Dict[str, Any],
                                   market_data: pd.Series,
                                   portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade using realistic execution conditions
        
        Args:
            trade_data: Trade information (signal, size, etc.)
            market_data: Current market data row
            portfolio_state: Current portfolio state
            
        Returns:
            Execution result with realistic fills and costs
        """
        try:
            # Create market conditions from backtest data
            market_conditions = self._create_market_conditions_from_backtest_data(
                market_data, trade_data['timestamp']
            )
            
            # Create execution order
            order = self._create_execution_order(trade_data, market_conditions)
            
            # Execute order with realistic conditions
            execution_result = await self.execution_engine.execute_order(
                order, market_conditions
            )
            
            # Update execution statistics
            self._update_execution_stats(execution_result)
            
            # Store execution result
            self.execution_results.append(execution_result)
            
            # Convert to backtest format
            backtest_result = self._convert_execution_to_backtest_format(
                execution_result, trade_data, portfolio_state
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Realistic execution failed: {e}")
            return self._create_failed_execution_result(trade_data, str(e))
    
    def _create_market_conditions_from_backtest_data(self, 
                                                   market_data: pd.Series,
                                                   timestamp: datetime) -> MarketConditions:
        """
        Create realistic market conditions from backtest data
        """
        # Get basic price data
        if 'Close' in market_data.index:
            current_price = market_data['Close']
        else:
            current_price = market_data.iloc[0]  # Use first available price
        
        # Simulate bid-ask spread based on market conditions
        volatility = self._estimate_volatility(market_data)
        volume_factor = self._estimate_volume_factor(market_data)
        
        # Dynamic spread calculation
        if self.config.enable_order_book_simulation:
            spread_points = self._calculate_dynamic_spread(
                current_price, volatility, volume_factor
            )
        else:
            spread_points = 0.5  # Default spread for NQ futures
        
        # Create bid-ask prices
        bid_price = current_price - spread_points / 2
        ask_price = current_price + spread_points / 2
        
        # Simulate order book depth
        bid_size = max(10, int(np.random.normal(50, 20)))
        ask_size = max(10, int(np.random.normal(50, 20)))
        
        # Time of day factor
        time_of_day_factor = self._calculate_time_of_day_factor(timestamp)
        
        # Volume rate relative to average
        volume_rate = volume_factor if volume_factor > 0 else 1.0
        
        # Volatility regime
        volatility_regime = min(1.0, volatility * self.config.volatility_regime_factor)
        
        # Stress indicator
        stress_indicator = self._calculate_stress_indicator(market_data)
        
        market_conditions = MarketConditions(
            timestamp=timestamp,
            current_price=current_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            volume_rate=volume_rate,
            volatility_regime=volatility_regime,
            time_of_day_factor=time_of_day_factor,
            stress_indicator=stress_indicator
        )
        
        # Store market conditions for analysis
        self.market_conditions_history.append(market_conditions)
        
        return market_conditions
    
    def _estimate_volatility(self, market_data: pd.Series) -> float:
        """Estimate current volatility from market data"""
        try:
            # Use high-low range as volatility proxy
            if 'High' in market_data.index and 'Low' in market_data.index:
                price_range = market_data['High'] - market_data['Low']
                current_price = market_data['Close'] if 'Close' in market_data.index else market_data['High']
                return price_range / current_price if current_price > 0 else 0.01
            else:
                return 0.01  # Default volatility
        except:
            return 0.01
    
    def _estimate_volume_factor(self, market_data: pd.Series) -> float:
        """Estimate volume factor from market data"""
        try:
            if 'Volume' in market_data.index:
                volume = market_data['Volume']
                # Normalize volume (simplified)
                return min(3.0, max(0.1, volume / 1000000))  # Assume 1M average volume
            else:
                return 1.0  # Default volume factor
        except:
            return 1.0
    
    def _calculate_dynamic_spread(self, price: float, volatility: float, volume_factor: float) -> float:
        """Calculate dynamic bid-ask spread"""
        # Base spread for NQ futures
        base_spread = 0.25  # 1 tick
        
        # Volatility adjustment
        volatility_adjustment = volatility * 2.0  # Higher volatility = wider spread
        
        # Volume adjustment (higher volume = tighter spread)
        volume_adjustment = max(0.5, 1.0 / volume_factor)
        
        # Time-based adjustment
        time_adjustment = 1.0  # Could be enhanced with time-of-day logic
        
        spread = base_spread * (1 + volatility_adjustment) * volume_adjustment * time_adjustment
        
        return max(0.25, min(spread, 2.0))  # Constrain between 1 tick and 2 points
    
    def _calculate_time_of_day_factor(self, timestamp: datetime) -> float:
        """Calculate time of day liquidity factor"""
        hour = timestamp.hour
        
        # Market hours (9:30 AM - 4:00 PM EST)
        if 9 <= hour <= 16:
            return 1.0  # Peak liquidity
        elif 6 <= hour <= 9 or 16 <= hour <= 18:
            return 0.7  # Pre/post market
        else:
            return 0.3  # Overnight
    
    def _calculate_stress_indicator(self, market_data: pd.Series) -> float:
        """Calculate market stress indicator"""
        try:
            # Use volatility and volume as stress indicators
            volatility = self._estimate_volatility(market_data)
            volume_factor = self._estimate_volume_factor(market_data)
            
            # High volatility and extreme volume indicate stress
            stress_from_volatility = min(1.0, volatility * 10)  # Scale volatility
            stress_from_volume = abs(volume_factor - 1.0)  # Deviation from normal volume
            
            # Combine factors
            stress_indicator = min(1.0, (stress_from_volatility + stress_from_volume) / 2)
            
            # Add random stress events
            if np.random.random() < self.config.stress_condition_probability:
                stress_indicator = max(stress_indicator, 0.8)
            
            return stress_indicator
        except:
            return 0.0
    
    def _create_execution_order(self, trade_data: Dict[str, Any], 
                              market_conditions: MarketConditions) -> ExecutionOrder:
        """Create execution order from trade data"""
        # Determine order side
        signal = trade_data.get('signal', 0)
        if signal > 0:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Calculate quantity (convert from size to contracts)
        size = abs(trade_data.get('size', 0))
        price = trade_data.get('price', market_conditions.current_price)
        
        # Convert size to number of contracts
        if size < 1:
            # Size is likely a percentage, convert to contracts
            account_value = self.execution_engine.account_value
            trade_value = size * account_value
            quantity = max(1, int(trade_value / (price * 20)))  # NQ point value = $20
        else:
            quantity = int(size)
        
        # Create order
        order = ExecutionOrder(
            order_id=f"BT_{int(datetime.now().timestamp() * 1000)}",
            symbol=trade_data.get('symbol', 'NQ'),
            side=side,
            order_type=OrderType.MARKET,  # Default to market orders in backtesting
            quantity=quantity,
            price=price,
            timestamp_created=trade_data['timestamp'],
            account_value=self.execution_engine.account_value,
            risk_percent=0.02  # Default 2% risk
        )
        
        return order
    
    def _update_execution_stats(self, execution_result: ExecutionResult):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if execution_result.execution_success:
            self.execution_stats['filled_orders'] += 1
            
            # Update costs
            order = execution_result.order
            self.execution_stats['total_slippage_cost'] += order.slippage_points * order.quantity * 20  # NQ point value
            self.execution_stats['total_commission_paid'] += order.commission_paid
            self.execution_stats['total_latency_ms'] += order.latency_ms
            
            # Check for partial fills
            if order.fill_quantity < order.quantity:
                self.execution_stats['partial_fills'] += 1
        else:
            self.execution_stats['rejected_orders'] += 1
    
    def _convert_execution_to_backtest_format(self, 
                                            execution_result: ExecutionResult,
                                            trade_data: Dict[str, Any],
                                            portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert realistic execution result to backtest format"""
        order = execution_result.order
        
        # Calculate actual trade value with realistic fill price
        if execution_result.execution_success:
            actual_price = order.fill_price
            actual_quantity = order.fill_quantity
            trade_value = actual_quantity * actual_price * 20  # NQ point value
            
            # Calculate shares for portfolio tracking
            shares = actual_quantity
            
            # Include all costs
            total_costs = order.commission_paid + order.total_fees + \
                         (order.slippage_points * order.quantity * 20)
            
            # Net trade value after costs
            net_trade_value = trade_value - total_costs
            
        else:
            # Trade failed
            actual_price = 0
            actual_quantity = 0
            trade_value = 0
            shares = 0
            total_costs = 0
            net_trade_value = 0
        
        # Create backtest-compatible result
        backtest_result = {
            'success': execution_result.execution_success,
            'timestamp': trade_data['timestamp'],
            'symbol': trade_data.get('symbol', 'NQ'),
            'signal': trade_data.get('signal', 0),
            'requested_price': trade_data.get('price', 0),
            'fill_price': actual_price,
            'requested_size': trade_data.get('size', 0),
            'fill_quantity': actual_quantity,
            'shares': shares,
            'trade_value': trade_value,
            'net_trade_value': net_trade_value,
            'total_costs': total_costs,
            'slippage_points': order.slippage_points if execution_result.execution_success else 0,
            'commission': order.commission_paid if execution_result.execution_success else 0,
            'latency_ms': order.latency_ms if execution_result.execution_success else 0,
            'market_impact': execution_result.market_impact,
            'execution_quality': self._calculate_execution_quality(execution_result),
            'type': trade_data.get('type', 'market')
        }
        
        return backtest_result
    
    def _calculate_execution_quality(self, execution_result: ExecutionResult) -> float:
        """Calculate execution quality score (0-100)"""
        if not execution_result.execution_success:
            return 0.0
        
        order = execution_result.order
        
        # Base quality factors
        fill_rate = order.fill_quantity / order.quantity if order.quantity > 0 else 0
        
        # Slippage quality (lower slippage = higher quality)
        slippage_quality = max(0, 1 - (order.slippage_points / 2.0))  # Normalize by 2 points
        
        # Latency quality (lower latency = higher quality)
        latency_quality = max(0, 1 - (order.latency_ms / 1000.0))  # Normalize by 1 second
        
        # Market impact quality
        impact_quality = max(0, 1 - (execution_result.market_impact / 1.0))  # Normalize by 1 point
        
        # Combined quality score
        quality_score = (fill_rate * 0.4 + slippage_quality * 0.3 + 
                        latency_quality * 0.2 + impact_quality * 0.1) * 100
        
        return min(100, max(0, quality_score))
    
    def _create_failed_execution_result(self, trade_data: Dict[str, Any], 
                                      error_msg: str) -> Dict[str, Any]:
        """Create result for failed execution"""
        return {
            'success': False,
            'timestamp': trade_data['timestamp'],
            'symbol': trade_data.get('symbol', 'NQ'),
            'signal': trade_data.get('signal', 0),
            'requested_price': trade_data.get('price', 0),
            'fill_price': 0,
            'requested_size': trade_data.get('size', 0),
            'fill_quantity': 0,
            'shares': 0,
            'trade_value': 0,
            'net_trade_value': 0,
            'total_costs': 0,
            'slippage_points': 0,
            'commission': 0,
            'latency_ms': 0,
            'market_impact': 0,
            'execution_quality': 0,
            'type': trade_data.get('type', 'market'),
            'error': error_msg
        }
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        total_orders = self.execution_stats['total_orders']
        
        if total_orders == 0:
            return {'error': 'No executions recorded'}
        
        # Calculate rates and averages
        fill_rate = self.execution_stats['filled_orders'] / total_orders
        rejection_rate = self.execution_stats['rejected_orders'] / total_orders
        partial_fill_rate = self.execution_stats['partial_fills'] / total_orders
        
        avg_slippage_cost = self.execution_stats['total_slippage_cost'] / max(1, self.execution_stats['filled_orders'])
        avg_commission = self.execution_stats['total_commission_paid'] / max(1, self.execution_stats['filled_orders'])
        avg_latency = self.execution_stats['total_latency_ms'] / max(1, self.execution_stats['filled_orders'])
        
        # Calculate execution quality metrics
        quality_scores = [self._calculate_execution_quality(result) for result in self.execution_results]
        avg_execution_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Market conditions analysis
        if self.market_conditions_history:
            avg_volatility = np.mean([mc.volatility_regime for mc in self.market_conditions_history])
            avg_spread = np.mean([mc.ask_price - mc.bid_price for mc in self.market_conditions_history])
            avg_stress = np.mean([mc.stress_indicator for mc in self.market_conditions_history])
        else:
            avg_volatility = avg_spread = avg_stress = 0
        
        return {
            'execution_performance': {
                'total_orders': total_orders,
                'fill_rate': fill_rate,
                'rejection_rate': rejection_rate,
                'partial_fill_rate': partial_fill_rate,
                'avg_execution_quality': avg_execution_quality
            },
            'cost_analysis': {
                'total_slippage_cost': self.execution_stats['total_slippage_cost'],
                'total_commission_paid': self.execution_stats['total_commission_paid'],
                'avg_slippage_cost_per_trade': avg_slippage_cost,
                'avg_commission_per_trade': avg_commission,
                'total_execution_costs': self.execution_stats['total_slippage_cost'] + self.execution_stats['total_commission_paid']
            },
            'timing_analysis': {
                'total_latency_ms': self.execution_stats['total_latency_ms'],
                'avg_latency_ms': avg_latency,
                'min_latency_ms': self.config.min_execution_delay_ms,
                'max_latency_ms': self.config.max_execution_delay_ms
            },
            'market_conditions': {
                'avg_volatility_regime': avg_volatility,
                'avg_bid_ask_spread': avg_spread,
                'avg_stress_indicator': avg_stress,
                'total_market_snapshots': len(self.market_conditions_history)
            },
            'configuration': {
                'realistic_slippage_enabled': self.config.enable_realistic_slippage,
                'market_impact_enabled': self.config.enable_market_impact,
                'execution_latency_enabled': self.config.enable_execution_latency,
                'partial_fills_enabled': self.config.enable_partial_fills,
                'order_book_simulation_enabled': self.config.enable_order_book_simulation
            }
        }
    
    def generate_execution_report(self) -> str:
        """Generate comprehensive execution report"""
        analytics = self.get_execution_analytics()
        
        if 'error' in analytics:
            return analytics['error']
        
        report = []
        report.append("=" * 80)
        report.append("REALISTIC EXECUTION ANALYTICS REPORT")
        report.append("=" * 80)
        
        # Execution Performance
        ep = analytics['execution_performance']
        report.append(f"📊 EXECUTION PERFORMANCE")
        report.append(f"   Total Orders: {ep['total_orders']}")
        report.append(f"   Fill Rate: {ep['fill_rate']:.2%}")
        report.append(f"   Rejection Rate: {ep['rejection_rate']:.2%}")
        report.append(f"   Partial Fill Rate: {ep['partial_fill_rate']:.2%}")
        report.append(f"   Avg Execution Quality: {ep['avg_execution_quality']:.1f}/100")
        report.append("")
        
        # Cost Analysis
        ca = analytics['cost_analysis']
        report.append(f"💰 COST ANALYSIS")
        report.append(f"   Total Slippage Cost: ${ca['total_slippage_cost']:.2f}")
        report.append(f"   Total Commission Paid: ${ca['total_commission_paid']:.2f}")
        report.append(f"   Total Execution Costs: ${ca['total_execution_costs']:.2f}")
        report.append(f"   Avg Slippage Cost/Trade: ${ca['avg_slippage_cost_per_trade']:.2f}")
        report.append(f"   Avg Commission/Trade: ${ca['avg_commission_per_trade']:.2f}")
        report.append("")
        
        # Timing Analysis
        ta = analytics['timing_analysis']
        report.append(f"⏱️ TIMING ANALYSIS")
        report.append(f"   Total Latency: {ta['total_latency_ms']:.0f}ms")
        report.append(f"   Average Latency: {ta['avg_latency_ms']:.1f}ms")
        report.append(f"   Latency Range: {ta['min_latency_ms']:.0f}ms - {ta['max_latency_ms']:.0f}ms")
        report.append("")
        
        # Market Conditions
        mc = analytics['market_conditions']
        report.append(f"🏛️ MARKET CONDITIONS")
        report.append(f"   Average Volatility Regime: {mc['avg_volatility_regime']:.3f}")
        report.append(f"   Average Bid-Ask Spread: {mc['avg_bid_ask_spread']:.3f} points")
        report.append(f"   Average Stress Indicator: {mc['avg_stress_indicator']:.3f}")
        report.append(f"   Market Snapshots: {mc['total_market_snapshots']}")
        report.append("")
        
        # Configuration
        config = analytics['configuration']
        report.append(f"⚙️ CONFIGURATION")
        report.append(f"   Realistic Slippage: {'✅' if config['realistic_slippage_enabled'] else '❌'}")
        report.append(f"   Market Impact: {'✅' if config['market_impact_enabled'] else '❌'}")
        report.append(f"   Execution Latency: {'✅' if config['execution_latency_enabled'] else '❌'}")
        report.append(f"   Partial Fills: {'✅' if config['partial_fills_enabled'] else '❌'}")
        report.append(f"   Order Book Simulation: {'✅' if config['order_book_simulation_enabled'] else '❌'}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def reset_statistics(self):
        """Reset execution statistics"""
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
            'total_slippage_cost': 0.0,
            'total_commission_paid': 0.0,
            'total_latency_ms': 0.0,
            'execution_quality_score': 0.0
        }
        
        self.market_conditions_history.clear()
        self.execution_results.clear()
        
        logger.info("Execution statistics reset")


class RealisticBacktestFramework:
    """
    Enhanced backtesting framework with realistic execution integration
    """
    
    def __init__(self, 
                 original_framework,
                 execution_config: BacktestExecutionConfig = None):
        """
        Initialize realistic backtest framework
        
        Args:
            original_framework: Original ProfessionalBacktestFramework instance
            execution_config: Configuration for realistic execution
        """
        self.original_framework = original_framework
        self.execution_config = execution_config or BacktestExecutionConfig()
        
        # Initialize realistic execution handler
        self.realistic_handler = RealisticBacktestExecutionHandler(
            initial_capital=original_framework.initial_capital,
            config=self.execution_config
        )
        
        # Override execution method
        self.original_framework._execute_trade = self._realistic_execute_trade
        
        logger.info("RealisticBacktestFramework initialized")
    
    def _realistic_execute_trade(self, trade_data: Dict[str, Any], 
                               size: float, price: float):
        """
        Realistic trade execution that replaces the original method
        """
        # Prepare trade data for realistic execution
        realistic_trade_data = {
            'timestamp': trade_data['timestamp'],
            'symbol': trade_data.get('symbol', 'NQ'),
            'signal': trade_data.get('signal', 0),
            'size': size,
            'price': price,
            'type': trade_data.get('type', 'market')
        }
        
        # Get current market data
        timestamp = trade_data['timestamp']
        
        # Create mock market data series (in real implementation, this would come from data)
        market_data = pd.Series({
            'Close': price,
            'High': price * 1.002,
            'Low': price * 0.998,
            'Volume': 1000000
        })
        
        # Execute trade realistically (synchronous wrapper for async execution)
        execution_result = asyncio.run(
            self.realistic_handler.execute_backtest_trade(
                realistic_trade_data,
                market_data,
                self.original_framework.portfolio_state
            )
        )
        
        if execution_result['success']:
            # Update portfolio using realistic execution results
            self._update_portfolio_with_realistic_execution(execution_result)
        else:
            logger.warning(f"Trade execution failed: {execution_result.get('error', 'Unknown error')}")
    
    def _update_portfolio_with_realistic_execution(self, execution_result: Dict[str, Any]):
        """Update portfolio state with realistic execution results"""
        try:
            # Get portfolio state
            portfolio_state = self.original_framework.portfolio_state
            
            # Update trade record with realistic execution details
            trade_record = {
                'timestamp': execution_result['timestamp'],
                'symbol': execution_result['symbol'],
                'signal': execution_result['signal'],
                'price': execution_result['fill_price'],  # Use actual fill price
                'shares': execution_result['shares'],
                'value': execution_result['net_trade_value'],  # Use net value after costs
                'type': execution_result['type'],
                'slippage_points': execution_result['slippage_points'],
                'commission': execution_result['commission'],
                'latency_ms': execution_result['latency_ms'],
                'execution_quality': execution_result['execution_quality']
            }
            
            # Add to trades with realistic execution data
            self.original_framework.trades.append(trade_record)
            
            # Update portfolio positions with realistic fills
            symbol = execution_result['symbol']
            if symbol not in portfolio_state['positions']:
                portfolio_state['positions'][symbol] = {
                    'shares': 0,
                    'value': 0,
                    'avg_price': 0
                }
            
            position = portfolio_state['positions'][symbol]
            fill_price = execution_result['fill_price']
            shares = execution_result['shares']
            trade_value = execution_result['trade_value']
            costs = execution_result['total_costs']
            
            # Update position with realistic execution
            if execution_result['signal'] > 0:  # Buy
                total_shares = position['shares'] + shares
                total_value = position['value'] + trade_value
                position['shares'] = total_shares
                position['value'] = total_value
                position['avg_price'] = total_value / total_shares if total_shares > 0 else fill_price
                
                # Update cash (subtract trade value and costs)
                portfolio_state['cash'] -= (trade_value + costs)
            else:  # Sell
                position['shares'] = max(0, position['shares'] - shares)
                position['value'] = max(0, position['value'] - trade_value)
                
                # Update cash (add trade value, subtract costs)
                portfolio_state['cash'] += (trade_value - costs)
            
            # Update total exposure
            portfolio_state['total_exposure'] = sum(
                pos['value'] for pos in portfolio_state['positions'].values()
            )
            
        except Exception as e:
            logger.error(f"Portfolio update error: {e}")
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get realistic execution analytics"""
        return self.realistic_handler.get_execution_analytics()
    
    def generate_execution_report(self) -> str:
        """Generate realistic execution report"""
        return self.realistic_handler.generate_execution_report()
    
    def save_execution_analytics(self, file_path: str = None) -> str:
        """Save execution analytics to file"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'/home/QuantNova/GrandModel/results/execution/realistic_execution_analytics_{timestamp}.json'
        
        import os
        import json
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        analytics = self.get_execution_analytics()
        
        with open(file_path, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        
        logger.info(f"Execution analytics saved: {file_path}")
        return file_path


def create_realistic_backtest_framework(original_framework,
                                      execution_config: BacktestExecutionConfig = None):
    """
    Create a realistic backtest framework that integrates with existing framework
    
    Args:
        original_framework: Original ProfessionalBacktestFramework instance
        execution_config: Configuration for realistic execution
        
    Returns:
        Enhanced framework with realistic execution
    """
    return RealisticBacktestFramework(original_framework, execution_config)