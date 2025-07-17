"""
VectorBT Realistic Execution Integration
======================================

AGENT 3 - TRADING EXECUTION REALISM SPECIALIST

This module integrates the realistic execution engine with VectorBT backtesting
to provide realistic execution conditions while preserving exact signal logic.

Key Features:
- Preserves existing strategy signal generation
- Applies realistic NQ futures execution conditions
- Proper slippage, commissions, and latency modeling
- Risk-based position sizing
- Comprehensive cost analysis

Author: AGENT 3 - Trading Execution Realism Specialist
Date: 2025-07-16
Mission: Transform perfect execution into realistic trading conditions
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
import asyncio
import logging
from pathlib import Path
import json

from .realistic_execution_engine import (
    RealisticExecutionEngine, NQFuturesSpecs, OrderSide, OrderType,
    ExecutionOrder, MarketConditions, ExecutionResult
)

logger = logging.getLogger(__name__)


class RealisticVectorBTBacktest:
    """
    VectorBT backtesting with realistic execution conditions
    
    This class acts as a wrapper around VectorBT to apply realistic
    execution conditions while preserving the original strategy logic.
    """
    
    def __init__(self, 
                 account_value: float = 100000.0,
                 risk_per_trade: float = 0.02,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize realistic VectorBT backtest
        
        Args:
            account_value: Starting account value
            risk_per_trade: Risk percentage per trade (default 2%)
            config: Additional configuration
        """
        self.account_value = account_value
        self.risk_per_trade = risk_per_trade
        self.config = config or {}
        
        # Initialize realistic execution engine
        self.execution_engine = RealisticExecutionEngine(
            account_value=account_value,
            config=config
        )
        
        # Backtest tracking
        self.backtest_data = None
        self.original_signals = None
        self.realistic_trades = []
        self.performance_metrics = {}
        
        # Results storage
        self.results_dir = Path('results/realistic_backtest')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RealisticVectorBTBacktest initialized with ${account_value:,.2f}")
        
    def prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare market data with additional columns needed for realistic execution
        """
        # Ensure required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Add execution-specific columns
        df = df.copy()
        
        # Calculate volume ratio (current vs average)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(50, min_periods=1).mean()
        
        # Calculate volatility regime
        returns = df['Close'].pct_change()
        df['volatility'] = returns.rolling(20, min_periods=1).std()
        df['volatility_regime'] = df['volatility'] / df['volatility'].rolling(100, min_periods=1).mean()
        
        # Time of day factor
        if df.index.name == 'Timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['time_of_day_factor'] = df['hour'].apply(self._calculate_time_factor)
        else:
            df['time_of_day_factor'] = 1.0  # Default to market hours
            
        # Stress indicator (simplified as high volatility periods)
        df['stress_indicator'] = (df['volatility_regime'] > 1.5).astype(float) * 0.5
        
        # Fill any NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
        
    def _calculate_time_factor(self, hour: int) -> float:
        """Calculate time of day liquidity factor"""
        if 9 <= hour <= 16:  # Market hours
            return 1.0
        elif 6 <= hour <= 9 or 16 <= hour <= 18:  # Pre/post market
            return 0.7
        else:  # Overnight
            return 0.3
            
    def create_market_conditions_series(self, df: pd.DataFrame) -> List[MarketConditions]:
        """
        Create market conditions for each bar in the data
        """
        market_conditions_list = []
        
        for idx, row in df.iterrows():
            # Simple bid-ask spread simulation
            spread_points = np.random.uniform(0.25, 1.0)
            bid_price = row['Close'] - spread_points / 2
            ask_price = row['Close'] + spread_points / 2
            
            conditions = MarketConditions(
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                current_price=row['Close'],
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=np.random.randint(10, 100),
                ask_size=np.random.randint(10, 100),
                volume_rate=row.get('volume_ratio', 1.0),
                volatility_regime=row.get('volatility_regime', 0.5),
                time_of_day_factor=row.get('time_of_day_factor', 1.0),
                stress_indicator=row.get('stress_indicator', 0.0)
            )
            
            market_conditions_list.append(conditions)
            
        return market_conditions_list
        
    async def apply_realistic_execution(self,
                                       df: pd.DataFrame,
                                       entries: pd.Series,
                                       exits: pd.Series,
                                       stop_loss_pct: float = 0.02) -> Dict[str, Any]:
        """
        Apply realistic execution to VectorBT signals
        
        Args:
            df: Market data
            entries: Entry signals from strategy
            exits: Exit signals from strategy
            stop_loss_pct: Stop loss percentage for position sizing
            
        Returns:
            Realistic execution results
        """
        logger.info("Applying realistic execution to strategy signals...")
        
        # Prepare data
        df_prepared = self.prepare_market_data(df)
        market_conditions_list = self.create_market_conditions_series(df_prepared)
        
        # Track realistic execution
        realistic_entries = pd.Series(False, index=df.index)
        realistic_exits = pd.Series(False, index=df.index)
        realistic_prices = pd.Series(np.nan, index=df.index)
        execution_details = []
        
        current_position = None
        
        for i, (timestamp, row) in enumerate(df_prepared.iterrows()):
            market_conditions = market_conditions_list[i]
            
            # Process entry signals
            if entries.iloc[i] and current_position is None:
                # Calculate position size with risk management
                entry_price = row['Close']
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                
                # Create entry order
                entry_order = self.execution_engine.create_order(
                    side=OrderSide.BUY,
                    quantity=1,  # Will be adjusted by position sizer
                    order_type=OrderType.MARKET,
                    price=entry_price,
                    stop_price=stop_loss_price,
                    risk_percent=self.risk_per_trade
                )
                
                # Execute with realistic conditions
                try:
                    execution_result = await self.execution_engine.execute_order(
                        entry_order, market_conditions
                    )
                    
                    if execution_result.execution_success:
                        current_position = entry_order
                        realistic_entries.iloc[i] = True
                        realistic_prices.iloc[i] = entry_order.fill_price
                        
                        execution_details.append({
                            'timestamp': timestamp,
                            'type': 'entry',
                            'order': entry_order,
                            'execution_result': execution_result
                        })
                        
                        logger.debug(f"Entry executed at {timestamp}: {entry_order.quantity} contracts @ {entry_order.fill_price:.2f}")
                        
                except Exception as e:
                    logger.error(f"Entry execution failed at {timestamp}: {str(e)}")
                    
            # Process exit signals
            elif exits.iloc[i] and current_position is not None:
                # Create exit order
                exit_order = self.execution_engine.create_order(
                    side=OrderSide.SELL,
                    quantity=current_position.quantity,
                    order_type=OrderType.MARKET
                )
                
                # Execute with realistic conditions
                try:
                    execution_result = await self.execution_engine.execute_order(
                        exit_order, market_conditions
                    )
                    
                    if execution_result.execution_success:
                        realistic_exits.iloc[i] = True
                        realistic_prices.iloc[i] = exit_order.fill_price
                        
                        execution_details.append({
                            'timestamp': timestamp,
                            'type': 'exit',
                            'order': exit_order,
                            'execution_result': execution_result
                        })
                        
                        # Calculate trade PnL
                        trade_pnl = self.execution_engine.pnl_calculator.calculate_trade_pnl(
                            current_position, exit_order
                        )
                        
                        self.realistic_trades.append({
                            'entry': current_position,
                            'exit': exit_order,
                            'pnl': trade_pnl,
                            'entry_timestamp': current_position.timestamp_created,
                            'exit_timestamp': timestamp
                        })
                        
                        logger.debug(f"Exit executed at {timestamp}: PnL = ${trade_pnl['net_pnl']:.2f}")
                        
                        current_position = None
                        
                except Exception as e:
                    logger.error(f"Exit execution failed at {timestamp}: {str(e)}")
        
        # Compile results
        results = {
            'realistic_entries': realistic_entries,
            'realistic_exits': realistic_exits,
            'realistic_prices': realistic_prices,
            'execution_details': execution_details,
            'market_conditions': market_conditions_list,
            'trades': self.realistic_trades,
            'execution_metrics': self.execution_engine.get_performance_metrics()
        }
        
        logger.info(f"Realistic execution applied: {realistic_entries.sum()} entries, {realistic_exits.sum()} exits")
        
        return results
        
    def run_realistic_backtest(self,
                              df: pd.DataFrame,
                              signal_generator: Callable,
                              signal_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run complete realistic backtest
        
        Args:
            df: Market data
            signal_generator: Function that generates entry/exit signals
            signal_params: Parameters for signal generator
            
        Returns:
            Complete backtest results
        """
        logger.info("Starting realistic backtest...")
        
        # Generate original signals
        if signal_params:
            entries, exits = signal_generator(df, **signal_params)
        else:
            entries, exits = signal_generator(df)
            
        self.original_signals = {'entries': entries, 'exits': exits}
        
        # Run VectorBT backtest with original signals (for comparison)
        original_portfolio = vbt.Portfolio.from_signals(
            df['Close'],
            entries=entries,
            exits=exits,
            init_cash=self.account_value,
            size=0.95,
            size_type='percent',
            fees=0.001,
            slippage=0.0005,
            freq='5min'
        )
        
        # Apply realistic execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            realistic_results = loop.run_until_complete(
                self.apply_realistic_execution(df, entries, exits)
            )
        finally:
            loop.close()
        
        # Run VectorBT backtest with realistic results
        realistic_portfolio = None
        if realistic_results['realistic_entries'].sum() > 0:
            # Create realistic price series for VectorBT
            realistic_close = df['Close'].copy()
            fill_mask = ~realistic_results['realistic_prices'].isna()
            realistic_close[fill_mask] = realistic_results['realistic_prices'][fill_mask]
            
            realistic_portfolio = vbt.Portfolio.from_signals(
                realistic_close,
                entries=realistic_results['realistic_entries'],
                exits=realistic_results['realistic_exits'],
                init_cash=self.account_value,
                size=0.95,
                size_type='percent',
                fees=0.0,  # Fees already accounted for in realistic execution
                slippage=0.0,  # Slippage already accounted for
                freq='5min'
            )
        
        # Compile comprehensive results
        results = {
            'original_portfolio': original_portfolio,
            'realistic_portfolio': realistic_portfolio,
            'realistic_execution': realistic_results,
            'comparison_metrics': self._compare_portfolios(original_portfolio, realistic_portfolio),
            'execution_engine_metrics': self.execution_engine.get_performance_metrics(),
            'signal_comparison': self._compare_signals(
                self.original_signals, 
                realistic_results['realistic_entries'], 
                realistic_results['realistic_exits']
            )
        }
        
        # Store results
        self.performance_metrics = results
        
        logger.info("Realistic backtest completed")
        return results
        
    def _compare_portfolios(self, 
                           original: vbt.Portfolio, 
                           realistic: Optional[vbt.Portfolio]) -> Dict[str, Any]:
        """Compare original vs realistic portfolio performance"""
        if realistic is None:
            return {'comparison_available': False}
            
        original_stats = original.stats()
        realistic_stats = realistic.stats()
        
        comparison = {
            'comparison_available': True,
            'total_return': {
                'original': float(original_stats.get('Total Return [%]', 0)),
                'realistic': float(realistic_stats.get('Total Return [%]', 0)),
                'difference': float(realistic_stats.get('Total Return [%]', 0)) - float(original_stats.get('Total Return [%]', 0))
            },
            'sharpe_ratio': {
                'original': float(original_stats.get('Sharpe Ratio', 0)),
                'realistic': float(realistic_stats.get('Sharpe Ratio', 0)),
                'difference': float(realistic_stats.get('Sharpe Ratio', 0)) - float(original_stats.get('Sharpe Ratio', 0))
            },
            'max_drawdown': {
                'original': float(original_stats.get('Max Drawdown [%]', 0)),
                'realistic': float(realistic_stats.get('Max Drawdown [%]', 0)),
                'difference': float(realistic_stats.get('Max Drawdown [%]', 0)) - float(original_stats.get('Max Drawdown [%]', 0))
            },
            'total_trades': {
                'original': int(original_stats.get('# Trades', 0)),
                'realistic': int(realistic_stats.get('# Trades', 0)),
                'difference': int(realistic_stats.get('# Trades', 0)) - int(original_stats.get('# Trades', 0))
            }
        }
        
        # Calculate impact of realistic execution
        comparison['execution_impact'] = {
            'return_reduction_pct': comparison['total_return']['difference'],
            'sharpe_degradation': comparison['sharpe_ratio']['difference'],
            'trade_execution_rate': comparison['total_trades']['realistic'] / max(1, comparison['total_trades']['original'])
        }
        
        return comparison
        
    def _compare_signals(self, 
                        original: Dict[str, pd.Series],
                        realistic_entries: pd.Series,
                        realistic_exits: pd.Series) -> Dict[str, Any]:
        """Compare original vs realistic signals"""
        return {
            'original_entries': int(original['entries'].sum()),
            'realistic_entries': int(realistic_entries.sum()),
            'entry_execution_rate': float(realistic_entries.sum() / max(1, original['entries'].sum())),
            'original_exits': int(original['exits'].sum()),
            'realistic_exits': int(realistic_exits.sum()),
            'exit_execution_rate': float(realistic_exits.sum() / max(1, original['exits'].sum())),
            'signal_preservation': {
                'entries_matched': int((original['entries'] & realistic_entries).sum()),
                'entries_missed': int((original['entries'] & ~realistic_entries).sum()),
                'exits_matched': int((original['exits'] & realistic_exits).sum()),
                'exits_missed': int((original['exits'] & ~realistic_exits).sum())
            }
        }
        
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive realistic execution report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f'realistic_backtest_report_{timestamp}.json'
        
        # Prepare serializable report
        report = {
            'metadata': {
                'strategy': 'Realistic Execution Backtest',
                'timestamp': timestamp,
                'account_value': self.account_value,
                'risk_per_trade': self.risk_per_trade,
                'nq_futures_specs': {
                    'point_value': self.execution_engine.specs.point_value,
                    'tick_size': self.execution_engine.specs.tick_size,
                    'commission_per_rt': self.execution_engine.specs.commission_per_rt
                }
            },
            'portfolio_comparison': results['comparison_metrics'],
            'execution_metrics': results['execution_engine_metrics'],
            'signal_analysis': results['signal_comparison'],
            'realistic_trades': [
                {
                    'entry_timestamp': trade['entry_timestamp'].isoformat(),
                    'exit_timestamp': trade['exit_timestamp'].isoformat(),
                    'contracts': trade['entry'].quantity,
                    'entry_price': trade['entry'].fill_price,
                    'exit_price': trade['exit'].fill_price,
                    'gross_pnl': trade['pnl']['gross_pnl'],
                    'net_pnl': trade['pnl']['net_pnl'],
                    'total_costs': trade['pnl']['total_costs'],
                    'slippage_cost': trade['pnl']['total_slippage_cost'],
                    'commission': trade['pnl']['total_commission']
                }
                for trade in self.realistic_trades
            ]
        }
        
        # Add performance assessment
        report['performance_assessment'] = self._assess_realistic_performance(results)
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        self._print_realistic_summary(report)
        
        logger.info(f"Comprehensive realistic backtest report saved: {report_file}")
        return str(report_file)
        
    def _assess_realistic_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of realistic execution"""
        comparison = results['comparison_metrics']
        execution_metrics = results['execution_engine_metrics']
        
        assessment = {
            'execution_quality': 'Unknown',
            'cost_impact': 'Unknown',
            'signal_preservation': 'Unknown',
            'recommendations': []
        }
        
        if comparison['comparison_available']:
            # Assess execution quality
            return_diff = comparison['total_return']['difference']
            if return_diff > -2:
                assessment['execution_quality'] = 'Excellent'
            elif return_diff > -5:
                assessment['execution_quality'] = 'Good'
            elif return_diff > -10:
                assessment['execution_quality'] = 'Fair'
            else:
                assessment['execution_quality'] = 'Poor'
                
            # Assess cost impact
            avg_slippage = execution_metrics['execution_metrics']['avg_slippage_points']
            if avg_slippage < 0.5:
                assessment['cost_impact'] = 'Low'
            elif avg_slippage < 1.0:
                assessment['cost_impact'] = 'Moderate'
            else:
                assessment['cost_impact'] = 'High'
                
            # Assess signal preservation
            entry_rate = results['signal_comparison']['entry_execution_rate']
            if entry_rate > 0.9:
                assessment['signal_preservation'] = 'Excellent'
            elif entry_rate > 0.8:
                assessment['signal_preservation'] = 'Good'
            elif entry_rate > 0.7:
                assessment['signal_preservation'] = 'Fair'
            else:
                assessment['signal_preservation'] = 'Poor'
                
            # Generate recommendations
            if avg_slippage > 1.0:
                assessment['recommendations'].append('Consider using limit orders to reduce slippage')
            if entry_rate < 0.8:
                assessment['recommendations'].append('Review signal timing to improve execution rate')
            if return_diff < -5:
                assessment['recommendations'].append('Consider optimizing position sizing or risk management')
                
        return assessment
        
    def _print_realistic_summary(self, report: Dict[str, Any]):
        """Print formatted summary of realistic backtest results"""
        print("\n" + "="*80)
        print("🎯 REALISTIC EXECUTION BACKTEST RESULTS")
        print("="*80)
        
        # Metadata
        metadata = report['metadata']
        print(f"\n📊 BACKTEST METADATA:")
        print(f"   Strategy: {metadata['strategy']}")
        print(f"   Account Value: ${metadata['account_value']:,.2f}")
        print(f"   Risk per Trade: {metadata['risk_per_trade']:.1%}")
        print(f"   NQ Point Value: ${metadata['nq_futures_specs']['point_value']}")
        print(f"   Commission: ${metadata['nq_futures_specs']['commission_per_rt']:.2f} per RT")
        
        # Portfolio comparison
        if report['portfolio_comparison']['comparison_available']:
            comparison = report['portfolio_comparison']
            print(f"\n📈 PERFORMANCE COMPARISON (Original vs Realistic):")
            print(f"   Total Return: {comparison['total_return']['original']:.2f}% → {comparison['total_return']['realistic']:.2f}% "
                  f"({comparison['total_return']['difference']:+.2f}%)")
            print(f"   Sharpe Ratio: {comparison['sharpe_ratio']['original']:.2f} → {comparison['sharpe_ratio']['realistic']:.2f} "
                  f"({comparison['sharpe_ratio']['difference']:+.2f})")
            print(f"   Max Drawdown: {comparison['max_drawdown']['original']:.2f}% → {comparison['max_drawdown']['realistic']:.2f}% "
                  f"({comparison['max_drawdown']['difference']:+.2f}%)")
            print(f"   Total Trades: {comparison['total_trades']['original']} → {comparison['total_trades']['realistic']} "
                  f"({comparison['total_trades']['difference']:+d})")
        
        # Execution metrics
        exec_metrics = report['execution_metrics']['execution_metrics']
        print(f"\n🎯 EXECUTION QUALITY:")
        print(f"   Fill Rate: {exec_metrics['fill_rate']:.1%}")
        print(f"   Avg Slippage: {exec_metrics['avg_slippage_points']:.2f} points (${exec_metrics['avg_slippage_cost']:.2f})")
        print(f"   Avg Latency: {exec_metrics['avg_latency_ms']:.1f}ms")
        print(f"   Commission per Trade: ${exec_metrics['avg_commission_per_trade']:.2f}")
        
        # Signal analysis
        signal_analysis = report['signal_analysis']
        print(f"\n📊 SIGNAL EXECUTION ANALYSIS:")
        print(f"   Entry Signals: {signal_analysis['original_entries']} → {signal_analysis['realistic_entries']} "
              f"({signal_analysis['entry_execution_rate']:.1%} execution rate)")
        print(f"   Exit Signals: {signal_analysis['original_exits']} → {signal_analysis['realistic_exits']} "
              f"({signal_analysis['exit_execution_rate']:.1%} execution rate)")
        
        # Trade analysis
        if report['realistic_trades']:
            total_net_pnl = sum(trade['net_pnl'] for trade in report['realistic_trades'])
            winning_trades = sum(1 for trade in report['realistic_trades'] if trade['net_pnl'] > 0)
            total_trades = len(report['realistic_trades'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            print(f"\n💰 TRADE ANALYSIS:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Total Net PnL: ${total_net_pnl:,.2f}")
            print(f"   Avg PnL per Trade: ${total_net_pnl/total_trades:.2f}" if total_trades > 0 else "   No trades")
        
        # Assessment
        if 'performance_assessment' in report:
            assessment = report['performance_assessment']
            print(f"\n🔍 PERFORMANCE ASSESSMENT:")
            print(f"   Execution Quality: {assessment['execution_quality']}")
            print(f"   Cost Impact: {assessment['cost_impact']}")
            print(f"   Signal Preservation: {assessment['signal_preservation']}")
            
            if assessment['recommendations']:
                print(f"   Recommendations:")
                for rec in assessment['recommendations']:
                    print(f"     • {rec}")
        
        print("\n" + "="*80)


# Integration function for existing strategies
def integrate_realistic_execution_with_strategy(strategy_function: Callable,
                                              df: pd.DataFrame,
                                              account_value: float = 100000.0,
                                              risk_per_trade: float = 0.02,
                                              **strategy_params) -> Dict[str, Any]:
    """
    Integrate realistic execution with any existing strategy
    
    Args:
        strategy_function: Function that returns (entries, exits) signals
        df: Market data
        account_value: Starting account value
        risk_per_trade: Risk percentage per trade
        **strategy_params: Parameters to pass to strategy function
        
    Returns:
        Complete realistic backtest results
    """
    # Initialize realistic backtest
    realistic_backtest = RealisticVectorBTBacktest(
        account_value=account_value,
        risk_per_trade=risk_per_trade
    )
    
    # Run realistic backtest
    results = realistic_backtest.run_realistic_backtest(
        df=df,
        signal_generator=strategy_function,
        signal_params=strategy_params
    )
    
    # Generate report
    report_file = realistic_backtest.generate_comprehensive_report(results)
    
    return {
        'results': results,
        'report_file': report_file,
        'realistic_backtest_engine': realistic_backtest
    }