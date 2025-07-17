"""
Enhanced Realistic Backtesting Framework
======================================

AGENT 4 - REALISTIC EXECUTION ENGINE INTEGRATION

This module provides a drop-in replacement for the existing backtesting framework
that integrates realistic execution conditions to eliminate backtest-live divergence.

Key Features:
- Seamless integration with existing backtesting workflows
- Realistic execution with dynamic slippage and market conditions
- Comprehensive cost modeling including commissions and fees
- Market impact and execution timing simulation
- Detailed execution analytics and reporting

Usage:
Replace ProfessionalBacktestFramework with EnhancedRealisticBacktestFramework
for realistic execution without changing existing code structure.

Author: AGENT 4 - Realistic Execution Engine Integration
Date: 2025-07-17
Mission: Eliminate backtest-live divergence through realistic execution
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
import os
import json
import logging

# Import original framework components
from backtesting.framework import ProfessionalBacktestFramework
from backtesting.realistic_execution_integration import (
    RealisticBacktestExecutionHandler,
    BacktestExecutionConfig,
    RealisticBacktestFramework
)

logger = logging.getLogger(__name__)


class EnhancedRealisticBacktestFramework(ProfessionalBacktestFramework):
    """
    Enhanced backtesting framework with realistic execution integration
    
    This class extends the original ProfessionalBacktestFramework to include
    realistic execution conditions while maintaining full compatibility with
    existing code.
    """
    
    def __init__(self, 
                 strategy_name: str = "Strategy",
                 benchmark_symbol: str = "SPY",
                 initial_capital: float = 100000,
                 risk_free_rate: float = 0.02,
                 risk_params: Dict[str, Any] = None,
                 execution_config: BacktestExecutionConfig = None):
        """
        Initialize enhanced realistic backtesting framework
        
        Args:
            strategy_name: Name of the strategy
            benchmark_symbol: Benchmark symbol for comparison
            initial_capital: Initial capital for backtesting
            risk_free_rate: Risk-free rate for calculations
            risk_params: Risk management parameters
            execution_config: Realistic execution configuration
        """
        # Initialize parent framework
        super().__init__(
            strategy_name=strategy_name,
            benchmark_symbol=benchmark_symbol,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            risk_params=risk_params
        )
        
        # Initialize realistic execution components
        self.execution_config = execution_config or BacktestExecutionConfig()
        self.realistic_handler = RealisticBacktestExecutionHandler(
            initial_capital=initial_capital,
            config=self.execution_config
        )
        
        # Track execution divergence metrics
        self.divergence_metrics = {
            'simple_vs_realistic_pnl': [],
            'simple_vs_realistic_fills': [],
            'cost_impact_analysis': [],
            'execution_quality_tracking': []
        }
        
        # Enhanced analytics
        self.execution_analytics_enabled = True
        self.market_microstructure_data = []
        
        print("✅ Enhanced Realistic Backtesting Framework initialized")
        print(f"   🔄 Realistic Execution: {'Enabled' if self.execution_config.enable_realistic_slippage else 'Disabled'}")
        print(f"   💰 Dynamic Costs: {'Enabled' if self.execution_config.use_dynamic_commission else 'Disabled'}")
        print(f"   📊 Market Impact: {'Enabled' if self.execution_config.enable_market_impact else 'Disabled'}")
        print(f"   ⏱️ Execution Timing: {'Enabled' if self.execution_config.enable_execution_latency else 'Disabled'}")
    
    def _execute_trade(self, trade_data: Dict[str, Any], 
                      size: float, price: float):
        """
        Enhanced trade execution with realistic conditions
        
        This method overrides the parent class method to provide realistic
        execution while maintaining compatibility with existing code.
        """
        try:
            # Prepare enhanced trade data
            enhanced_trade_data = {
                'timestamp': trade_data['timestamp'],
                'symbol': trade_data.get('symbol', 'STRATEGY'),
                'signal': trade_data.get('signal', 0),
                'size': size,
                'price': price,
                'type': trade_data.get('type', 'market'),
                'strategy_name': self.strategy_name
            }
            
            # Get current market data from the price series
            timestamp = trade_data['timestamp']
            
            # Create enhanced market data series
            market_data = self._create_enhanced_market_data(timestamp, price)
            
            # Execute trade with realistic conditions
            execution_result = asyncio.run(
                self.realistic_handler.execute_backtest_trade(
                    enhanced_trade_data,
                    market_data,
                    self.portfolio_state
                )
            )
            
            # Process execution result
            if execution_result['success']:
                self._process_realistic_execution_result(execution_result, trade_data)
            else:
                self._handle_execution_failure(execution_result, trade_data)
                
            # Track divergence metrics
            self._track_execution_divergence(trade_data, execution_result, size, price)
            
        except Exception as e:
            logger.error(f"Enhanced trade execution failed: {e}")
            # Fallback to simple execution
            self._fallback_to_simple_execution(trade_data, size, price)
    
    def _create_enhanced_market_data(self, timestamp: pd.Timestamp, price: float) -> pd.Series:
        """Create enhanced market data with realistic market conditions"""
        # Base market data
        market_data = pd.Series({
            'Close': price,
            'High': price * (1 + np.random.uniform(0.0001, 0.005)),
            'Low': price * (1 - np.random.uniform(0.0001, 0.005)),
            'Volume': np.random.randint(500000, 2000000)
        })
        
        # Add market microstructure data
        hour = timestamp.hour
        
        # Volume patterns based on time of day
        if 9 <= hour <= 11:  # Market open
            volume_multiplier = 1.5
        elif 14 <= hour <= 16:  # Market close
            volume_multiplier = 1.3
        elif 12 <= hour <= 14:  # Lunch time
            volume_multiplier = 0.7
        else:  # Off hours
            volume_multiplier = 0.3
        
        market_data['Volume'] *= volume_multiplier
        
        # Add volatility indicators
        market_data['Volatility'] = np.random.uniform(0.01, 0.05)
        market_data['Spread'] = np.random.uniform(0.01, 0.05)
        
        return market_data
    
    def _process_realistic_execution_result(self, execution_result: Dict[str, Any], 
                                          original_trade_data: Dict[str, Any]):
        """Process realistic execution result and update portfolio"""
        # Create enhanced trade record
        trade_record = {
            'timestamp': execution_result['timestamp'],
            'symbol': execution_result['symbol'],
            'signal': execution_result['signal'],
            'requested_price': execution_result['requested_price'],
            'fill_price': execution_result['fill_price'],
            'requested_size': execution_result['requested_size'],
            'fill_quantity': execution_result['fill_quantity'],
            'shares': execution_result['shares'],
            'value': execution_result['net_trade_value'],
            'type': execution_result['type'],
            
            # Enhanced execution metrics
            'slippage_points': execution_result['slippage_points'],
            'slippage_cost': execution_result['slippage_points'] * execution_result['fill_quantity'] * 20,
            'commission': execution_result['commission'],
            'total_costs': execution_result['total_costs'],
            'latency_ms': execution_result['latency_ms'],
            'market_impact': execution_result['market_impact'],
            'execution_quality': execution_result['execution_quality'],
            
            # Execution realism indicators
            'realistic_execution': True,
            'execution_success_rate': 1.0 if execution_result['success'] else 0.0
        }
        
        # Add to trades with enhanced data
        self.trades.append(trade_record)
        
        # Update portfolio with realistic execution impacts
        self._update_portfolio_with_realistic_impacts(execution_result)
        
        # Store market microstructure data
        self._store_market_microstructure_data(execution_result)
    
    def _update_portfolio_with_realistic_impacts(self, execution_result: Dict[str, Any]):
        """Update portfolio accounting for realistic execution impacts"""
        symbol = execution_result['symbol']
        
        # Initialize position if doesn't exist
        if symbol not in self.portfolio_state['positions']:
            self.portfolio_state['positions'][symbol] = {
                'shares': 0,
                'value': 0,
                'avg_price': 0,
                'unrealized_pnl': 0,
                'total_costs': 0
            }
        
        position = self.portfolio_state['positions'][symbol]
        
        # Extract execution details
        fill_price = execution_result['fill_price']
        shares = execution_result['shares']
        trade_value = execution_result['trade_value']
        total_costs = execution_result['total_costs']
        signal = execution_result['signal']
        
        # Update position with realistic execution
        if signal > 0:  # Buy
            # Update shares and average price
            total_shares = position['shares'] + shares
            total_value = position['value'] + trade_value
            position['shares'] = total_shares
            position['value'] = total_value
            position['avg_price'] = total_value / total_shares if total_shares > 0 else fill_price
            position['total_costs'] += total_costs
            
            # Update cash (subtract trade value and all costs)
            self.portfolio_state['cash'] -= (trade_value + total_costs)
            
        else:  # Sell
            # Update shares and value
            position['shares'] = max(0, position['shares'] - shares)
            position['value'] = max(0, position['value'] - trade_value)
            position['total_costs'] += total_costs
            
            # Update cash (add trade value, subtract costs)
            self.portfolio_state['cash'] += (trade_value - total_costs)
        
        # Update total exposure
        self.portfolio_state['total_exposure'] = sum(
            pos['value'] for pos in self.portfolio_state['positions'].values()
        )
        
        # Update total value accounting for costs
        total_position_value = sum(
            pos['value'] for pos in self.portfolio_state['positions'].values()
        )
        self.portfolio_state['total_value'] = self.portfolio_state['cash'] + total_position_value
    
    def _handle_execution_failure(self, execution_result: Dict[str, Any], 
                                 original_trade_data: Dict[str, Any]):
        """Handle execution failure with proper logging and fallback"""
        logger.warning(f"Realistic execution failed: {execution_result.get('error', 'Unknown error')}")
        
        # Record failed execution
        failed_trade_record = {
            'timestamp': execution_result['timestamp'],
            'symbol': execution_result['symbol'],
            'signal': execution_result['signal'],
            'requested_price': execution_result['requested_price'],
            'fill_price': 0,
            'requested_size': execution_result['requested_size'],
            'fill_quantity': 0,
            'shares': 0,
            'value': 0,
            'type': execution_result['type'],
            'execution_failed': True,
            'failure_reason': execution_result.get('error', 'Unknown error'),
            'realistic_execution': True
        }
        
        self.trades.append(failed_trade_record)
    
    def _fallback_to_simple_execution(self, trade_data: Dict[str, Any], 
                                    size: float, price: float):
        """Fallback to simple execution when realistic execution fails"""
        logger.info("Falling back to simple execution")
        
        # Call parent class execution method
        super()._execute_trade(trade_data, size, price)
        
        # Mark last trade as simple execution
        if self.trades:
            self.trades[-1]['realistic_execution'] = False
            self.trades[-1]['execution_fallback'] = True
    
    def _track_execution_divergence(self, original_trade_data: Dict[str, Any],
                                  execution_result: Dict[str, Any],
                                  requested_size: float, requested_price: float):
        """Track divergence between simple and realistic execution"""
        if not execution_result['success']:
            return
        
        # Calculate what simple execution would have been
        simple_fill_price = requested_price
        simple_trade_value = requested_size * self.portfolio_state['total_value']
        simple_shares = simple_trade_value / requested_price
        
        # Calculate realistic execution values
        realistic_fill_price = execution_result['fill_price']
        realistic_trade_value = execution_result['trade_value']
        realistic_shares = execution_result['shares']
        total_costs = execution_result['total_costs']
        
        # Calculate divergence
        price_divergence = (realistic_fill_price - simple_fill_price) / simple_fill_price
        value_divergence = (realistic_trade_value - simple_trade_value) / simple_trade_value if simple_trade_value != 0 else 0
        cost_impact = total_costs / realistic_trade_value if realistic_trade_value != 0 else 0
        
        # Store divergence metrics
        divergence_record = {
            'timestamp': execution_result['timestamp'],
            'price_divergence': price_divergence,
            'value_divergence': value_divergence,
            'cost_impact': cost_impact,
            'slippage_points': execution_result['slippage_points'],
            'execution_quality': execution_result['execution_quality'],
            'latency_ms': execution_result['latency_ms']
        }
        
        self.divergence_metrics['simple_vs_realistic_pnl'].append(divergence_record)
    
    def _store_market_microstructure_data(self, execution_result: Dict[str, Any]):
        """Store market microstructure data for analysis"""
        microstructure_record = {
            'timestamp': execution_result['timestamp'],
            'symbol': execution_result['symbol'],
            'fill_price': execution_result['fill_price'],
            'market_impact': execution_result['market_impact'],
            'execution_quality': execution_result['execution_quality'],
            'latency_ms': execution_result['latency_ms']
        }
        
        self.market_microstructure_data.append(microstructure_record)
    
    def run_comprehensive_backtest(self, 
                                 strategy_function: Callable,
                                 data: pd.DataFrame,
                                 benchmark_data: pd.Series = None,
                                 generate_charts: bool = True,
                                 stress_test: bool = True,
                                 execution_analytics: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic execution
        
        This method extends the parent class method to include realistic
        execution analytics and reporting.
        """
        print("🚀 Starting Enhanced Realistic Backtesting")
        print("=" * 60)
        
        # Run the original comprehensive backtest
        results = super().run_comprehensive_backtest(
            strategy_function=strategy_function,
            data=data,
            benchmark_data=benchmark_data,
            generate_charts=generate_charts,
            stress_test=stress_test
        )
        
        # Add realistic execution analytics
        if execution_analytics:
            print("📊 Step 8: Realistic Execution Analytics...")
            execution_analytics_results = self._generate_execution_analytics()
            results['execution_analytics'] = execution_analytics_results
            
            # Generate execution divergence analysis
            divergence_analysis = self._analyze_execution_divergence()
            results['execution_divergence'] = divergence_analysis
            
            # Generate market microstructure analysis
            microstructure_analysis = self._analyze_market_microstructure()
            results['market_microstructure'] = microstructure_analysis
        
        # Enhanced recommendations
        enhanced_recommendations = self._generate_enhanced_recommendations(results)
        results['enhanced_recommendations'] = enhanced_recommendations
        
        # Save enhanced results
        self._save_enhanced_results(results)
        
        print("✅ Enhanced Realistic Backtesting Complete!")
        self._print_enhanced_executive_summary(results)
        
        return results
    
    def _generate_execution_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive execution analytics"""
        return self.realistic_handler.get_execution_analytics()
    
    def _analyze_execution_divergence(self) -> Dict[str, Any]:
        """Analyze divergence between simple and realistic execution"""
        if not self.divergence_metrics['simple_vs_realistic_pnl']:
            return {'error': 'No divergence data available'}
        
        pnl_divergences = self.divergence_metrics['simple_vs_realistic_pnl']
        
        # Calculate aggregate metrics
        avg_price_divergence = np.mean([d['price_divergence'] for d in pnl_divergences])
        avg_value_divergence = np.mean([d['value_divergence'] for d in pnl_divergences])
        avg_cost_impact = np.mean([d['cost_impact'] for d in pnl_divergences])
        avg_slippage = np.mean([d['slippage_points'] for d in pnl_divergences])
        avg_execution_quality = np.mean([d['execution_quality'] for d in pnl_divergences])
        
        # Calculate total cost impact
        total_cost_impact = sum(d['cost_impact'] for d in pnl_divergences)
        
        # Risk metrics
        price_divergence_std = np.std([d['price_divergence'] for d in pnl_divergences])
        max_slippage = max([d['slippage_points'] for d in pnl_divergences])
        
        return {
            'summary': {
                'total_executions': len(pnl_divergences),
                'avg_price_divergence': avg_price_divergence,
                'avg_value_divergence': avg_value_divergence,
                'avg_cost_impact': avg_cost_impact,
                'total_cost_impact': total_cost_impact,
                'avg_slippage_points': avg_slippage,
                'avg_execution_quality': avg_execution_quality
            },
            'risk_metrics': {
                'price_divergence_volatility': price_divergence_std,
                'max_slippage_points': max_slippage,
                'execution_quality_range': [
                    min([d['execution_quality'] for d in pnl_divergences]),
                    max([d['execution_quality'] for d in pnl_divergences])
                ]
            },
            'impact_analysis': {
                'performance_impact': avg_value_divergence,
                'cost_drag': avg_cost_impact,
                'execution_efficiency': avg_execution_quality
            }
        }
    
    def _analyze_market_microstructure(self) -> Dict[str, Any]:
        """Analyze market microstructure data"""
        if not self.market_microstructure_data:
            return {'error': 'No market microstructure data available'}
        
        # Calculate microstructure metrics
        avg_market_impact = np.mean([d['market_impact'] for d in self.market_microstructure_data])
        avg_execution_quality = np.mean([d['execution_quality'] for d in self.market_microstructure_data])
        avg_latency = np.mean([d['latency_ms'] for d in self.market_microstructure_data])
        
        # Market impact distribution
        market_impacts = [d['market_impact'] for d in self.market_microstructure_data]
        impact_percentiles = {
            'p50': np.percentile(market_impacts, 50),
            'p75': np.percentile(market_impacts, 75),
            'p90': np.percentile(market_impacts, 90),
            'p95': np.percentile(market_impacts, 95)
        }
        
        return {
            'summary': {
                'total_executions': len(self.market_microstructure_data),
                'avg_market_impact': avg_market_impact,
                'avg_execution_quality': avg_execution_quality,
                'avg_latency_ms': avg_latency
            },
            'market_impact_distribution': impact_percentiles,
            'execution_quality_metrics': {
                'min_quality': min([d['execution_quality'] for d in self.market_microstructure_data]),
                'max_quality': max([d['execution_quality'] for d in self.market_microstructure_data]),
                'quality_std': np.std([d['execution_quality'] for d in self.market_microstructure_data])
            }
        }
    
    def _generate_enhanced_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on realistic execution"""
        recommendations = []
        
        # Get execution analytics
        exec_analytics = results.get('execution_analytics', {})
        divergence_analysis = results.get('execution_divergence', {})
        
        # Execution performance recommendations
        if exec_analytics and 'execution_performance' in exec_analytics:
            perf = exec_analytics['execution_performance']
            
            if perf.get('fill_rate', 0) < 0.95:
                recommendations.append("LOW FILL RATE: Review order sizing and market conditions")
            
            if perf.get('avg_execution_quality', 0) < 70:
                recommendations.append("POOR EXECUTION QUALITY: Consider improving execution algorithms")
        
        # Cost analysis recommendations
        if exec_analytics and 'cost_analysis' in exec_analytics:
            costs = exec_analytics['cost_analysis']
            
            if costs.get('avg_slippage_cost_per_trade', 0) > 50:
                recommendations.append("HIGH SLIPPAGE COSTS: Reduce order sizes or improve timing")
            
            if costs.get('total_execution_costs', 0) > 1000:
                recommendations.append("HIGH EXECUTION COSTS: Monitor transaction cost impact on returns")
        
        # Divergence recommendations
        if divergence_analysis and 'summary' in divergence_analysis:
            div_summary = divergence_analysis['summary']
            
            if div_summary.get('avg_cost_impact', 0) > 0.02:
                recommendations.append("SIGNIFICANT COST IMPACT: Backtest divergence may affect live performance")
            
            if div_summary.get('avg_price_divergence', 0) > 0.01:
                recommendations.append("HIGH PRICE DIVERGENCE: Consider more conservative position sizing")
        
        # Overall execution recommendation
        if exec_analytics and divergence_analysis:
            exec_quality = exec_analytics.get('execution_performance', {}).get('avg_execution_quality', 0)
            cost_impact = divergence_analysis.get('summary', {}).get('avg_cost_impact', 0)
            
            if exec_quality > 80 and cost_impact < 0.01:
                recommendations.append("EXCELLENT EXECUTION REALISM: Strategy ready for live deployment")
            elif exec_quality > 60 and cost_impact < 0.02:
                recommendations.append("GOOD EXECUTION REALISM: Consider paper trading before live deployment")
            else:
                recommendations.append("EXECUTION CONCERNS: Further optimization required before live deployment")
        
        return recommendations
    
    def _save_enhanced_results(self, results: Dict[str, Any]):
        """Save enhanced backtest results with execution analytics"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'/home/QuantNova/GrandModel/results/backtests/enhanced_realistic_backtest_{self.strategy_name}_{timestamp}.json'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            # Save enhanced results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save execution analytics separately
            if 'execution_analytics' in results:
                exec_file = f'/home/QuantNova/GrandModel/results/execution/execution_analytics_{self.strategy_name}_{timestamp}.json'
                os.makedirs(os.path.dirname(exec_file), exist_ok=True)
                
                with open(exec_file, 'w') as f:
                    json.dump(results['execution_analytics'], f, indent=2, default=str)
            
            print(f"   💾 Enhanced results saved: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save enhanced results: {e}")
    
    def _print_enhanced_executive_summary(self, results: Dict[str, Any]):
        """Print enhanced executive summary with execution metrics"""
        print("\n" + "=" * 80)
        print("ENHANCED REALISTIC BACKTESTING SUMMARY")
        print("=" * 80)
        
        # Original metrics
        try:
            perf = results.get('performance_analysis', {}).get('performance_summary', {})
            print(f"📈 Total Return: {perf.get('total_return', 0):.2%}")
            print(f"📊 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"📉 Max Drawdown: {results.get('performance_analysis', {}).get('drawdown_analysis', {}).get('max_drawdown', 0):.2%}")
            
            # Execution metrics
            exec_analytics = results.get('execution_analytics', {})
            if exec_analytics and 'execution_performance' in exec_analytics:
                exec_perf = exec_analytics['execution_performance']
                print(f"🎯 Fill Rate: {exec_perf.get('fill_rate', 0):.1%}")
                print(f"⭐ Avg Execution Quality: {exec_perf.get('avg_execution_quality', 0):.1f}/100")
                
                cost_analysis = exec_analytics.get('cost_analysis', {})
                print(f"💰 Total Execution Costs: ${cost_analysis.get('total_execution_costs', 0):.2f}")
                print(f"📉 Avg Slippage Cost: ${cost_analysis.get('avg_slippage_cost_per_trade', 0):.2f}")
            
            # Divergence metrics
            divergence = results.get('execution_divergence', {})
            if divergence and 'summary' in divergence:
                div_summary = divergence['summary']
                print(f"🔄 Execution Divergence: {div_summary.get('avg_value_divergence', 0):.2%}")
                print(f"💸 Cost Impact: {div_summary.get('avg_cost_impact', 0):.2%}")
            
            # Enhanced recommendations
            enhanced_recs = results.get('enhanced_recommendations', [])
            if enhanced_recs:
                print(f"\n🎯 ENHANCED RECOMMENDATION: {enhanced_recs[-1]}")
                
        except Exception as e:
            print(f"❌ Error printing enhanced summary: {e}")
        
        print("=" * 80)
    
    def generate_execution_report(self) -> str:
        """Generate detailed execution report"""
        return self.realistic_handler.generate_execution_report()
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        return self.realistic_handler.get_execution_analytics()
    
    def get_divergence_analysis(self) -> Dict[str, Any]:
        """Get execution divergence analysis"""
        return self._analyze_execution_divergence()
    
    def get_market_microstructure_analysis(self) -> Dict[str, Any]:
        """Get market microstructure analysis"""
        return self._analyze_market_microstructure()


# Convenience function for easy migration
def create_enhanced_realistic_backtest_framework(strategy_name: str = "Strategy",
                                               benchmark_symbol: str = "SPY",
                                               initial_capital: float = 100000,
                                               risk_free_rate: float = 0.02,
                                               risk_params: Dict[str, Any] = None,
                                               execution_config: BacktestExecutionConfig = None):
    """
    Create an enhanced realistic backtesting framework
    
    This function provides a drop-in replacement for the original framework
    with realistic execution capabilities.
    """
    return EnhancedRealisticBacktestFramework(
        strategy_name=strategy_name,
        benchmark_symbol=benchmark_symbol,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        risk_params=risk_params,
        execution_config=execution_config
    )