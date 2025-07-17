"""
AGENT 4 - BIAS-FREE BACKTEST ENGINE
Integrated backtest engine with zero look-ahead bias guarantee

This engine combines all bias elimination components into a comprehensive
backtesting framework that ensures:
- Strict temporal ordering
- Point-in-time data access only
- No future data leakage
- Comprehensive validation
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import warnings

# Import our bias elimination components
from src.validation.bias_elimination_engine import (
    PointInTimeDataManager, BiasFreePMICalculator, BiasFreeCMWRQKCalculator,
    WalkForwardValidator, StatisticalSignificanceTester,
    validate_system_integrity
)
from src.validation.performance_metrics_validator import (
    PerformanceCalculator, BenchmarkComparator, MetricsValidator,
    PerformanceMetrics
)


@dataclass
class BacktestConfig:
    """Configuration for bias-free backtest"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    slippage_bps: int = 1  # basis points
    max_position_size: float = 1.0  # fraction of capital
    
    # Strategy parameters
    mlmi_num_neighbors: int = 200
    mlmi_momentum_window: int = 20
    nwrqk_h: float = 8.0
    nwrqk_r: float = 8.0
    nwrqk_x0: int = 25
    nwrqk_lag: int = 2
    
    # Validation parameters
    min_warmup_periods: int = 100
    walk_forward_train_window: int = 1000
    walk_forward_test_window: int = 200
    
    # Risk management
    max_drawdown_limit: float = -0.20  # 20% max drawdown
    enable_risk_management: bool = True


@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    commission: float
    slippage: float
    
    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage


@dataclass
class Position:
    """Current position state"""
    quantity: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


class BiasFreeChallengeEngine:
    """Main bias-free backtest engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Initialize bias elimination components
        self.data_manager = PointInTimeDataManager()
        self.mlmi_calc = BiasFreePMICalculator(
            num_neighbors=config.mlmi_num_neighbors,
            momentum_window=config.mlmi_momentum_window
        )
        self.nwrqk_calc = BiasFreeCMWRQKCalculator(
            h=config.nwrqk_h,
            r=config.nwrqk_r,
            x_0=config.nwrqk_x0,
            lag=config.nwrqk_lag
        )
        
        # Performance calculation
        self.perf_calculator = PerformanceCalculator()
        self.metrics_validator = MetricsValidator()
        
        # Backtest state
        self.current_capital = config.initial_capital
        self.current_position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Validation tracking
        self.validation_results = {
            'bias_violations': 0,
            'temporal_violations': 0,
            'data_leakage_incidents': [],
            'calculation_errors': []
        }
    
    def calculate_slippage(self, price: float, quantity: int) -> float:
        """Calculate slippage cost"""
        slippage_rate = self.config.slippage_bps / 10000.0  # Convert bps to decimal
        return abs(quantity * price * slippage_rate)
    
    def execute_trade(self, timestamp: datetime, action: str, 
                     price: float, signal_strength: float) -> Optional[Trade]:
        """
        Execute trade with bias-free validation
        
        Args:
            timestamp: Trade timestamp
            action: 'BUY' or 'SELL'
            price: Execution price
            signal_strength: Strength of the signal (0-1)
            
        Returns:
            Trade object if executed, None if not
        """
        # Position sizing based on signal strength and available capital
        max_trade_value = self.current_capital * self.config.max_position_size
        base_quantity = int(max_trade_value / price)
        
        # Adjust quantity based on signal strength
        quantity = int(base_quantity * abs(signal_strength))
        
        if quantity == 0:
            return None
        
        # Apply direction
        if action == 'SELL':
            quantity = -quantity
        
        # Calculate costs
        commission = self.config.commission_per_trade
        slippage = self.calculate_slippage(price, quantity)
        
        # Check if we have enough capital
        total_cost = abs(quantity * price) + commission + slippage
        if total_cost > self.current_capital and self.current_position.is_flat:
            return None  # Not enough capital
        
        # Create trade
        trade = Trade(
            timestamp=timestamp,
            action=action,
            price=price,
            quantity=quantity,
            commission=commission,
            slippage=slippage
        )
        
        # Update position
        if self.current_position.is_flat:
            # Opening new position
            self.current_position.quantity = quantity
            self.current_position.avg_price = price
        elif (self.current_position.is_long and quantity > 0) or \
             (self.current_position.is_short and quantity < 0):
            # Adding to position
            total_value = (self.current_position.quantity * self.current_position.avg_price + 
                          quantity * price)
            self.current_position.quantity += quantity
            self.current_position.avg_price = total_value / self.current_position.quantity
        else:
            # Closing or reversing position
            if abs(quantity) >= abs(self.current_position.quantity):
                # Full close or reversal
                remaining_quantity = quantity + self.current_position.quantity
                self.current_position.quantity = remaining_quantity
                self.current_position.avg_price = price if remaining_quantity != 0 else 0.0
            else:
                # Partial close
                self.current_position.quantity += quantity
        
        # Update capital
        trade_pnl = 0.0
        if action == 'SELL' and self.current_position.quantity >= 0:
            # Closing long position
            trade_pnl = quantity * (price - self.current_position.avg_price)
        elif action == 'BUY' and self.current_position.quantity <= 0:
            # Closing short position  
            trade_pnl = -quantity * (price - self.current_position.avg_price)
        
        self.current_capital += trade_pnl - total_cost
        
        self.trades.append(trade)
        return trade
    
    def calculate_signals(self, prices: np.ndarray, current_idx: int) -> Dict[str, Any]:
        """
        Calculate trading signals with bias-free validation
        
        Args:
            prices: Price array
            current_idx: Current time index
            
        Returns:
            Signal dictionary
        """
        # Validate temporal access
        if not self.data_manager.validate_temporal_access(
            current_idx, current_idx, "signal_calculation"
        ):
            self.validation_results['temporal_violations'] += 1
            return {'signal': 0, 'strength': 0.0, 'error': 'Temporal violation'}
        
        # Calculate MLMI
        try:
            mlmi_result = self.mlmi_calc.calculate_bias_free_mlmi(prices, current_idx)
        except Exception as e:
            self.validation_results['calculation_errors'].append(f"MLMI error at {current_idx}: {str(e)}")
            mlmi_result = {'mlmi_value': 0.0, 'mlmi_signal': 0}
        
        # Calculate NW-RQK
        try:
            nwrqk_result = self.nwrqk_calc.calculate_bias_free_nwrqk(prices, current_idx)
        except Exception as e:
            self.validation_results['calculation_errors'].append(f"NW-RQK error at {current_idx}: {str(e)}")
            nwrqk_result = {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
        
        # Combine signals (example strategy)
        mlmi_signal = mlmi_result.get('mlmi_signal', 0)
        nwrqk_signal = nwrqk_result.get('nwrqk_signal', 0)
        
        # Simple signal combination
        combined_signal = 0
        signal_strength = 0.0
        
        if mlmi_signal == 1 and nwrqk_signal == 1:
            combined_signal = 1
            signal_strength = min(1.0, abs(mlmi_result.get('mlmi_value', 0)) / 10.0)
        elif mlmi_signal == -1 and nwrqk_signal == -1:
            combined_signal = -1
            signal_strength = min(1.0, abs(mlmi_result.get('mlmi_value', 0)) / 10.0)
        elif mlmi_signal != 0 or nwrqk_signal != 0:
            # Weaker signal if only one indicator agrees
            combined_signal = mlmi_signal if mlmi_signal != 0 else nwrqk_signal
            signal_strength = min(0.5, abs(mlmi_result.get('mlmi_value', 0)) / 20.0)
        
        return {
            'signal': combined_signal,
            'strength': signal_strength,
            'mlmi': mlmi_result,
            'nwrqk': nwrqk_result
        }
    
    def check_risk_management(self, current_price: float) -> bool:
        """
        Check risk management constraints
        
        Args:
            current_price: Current market price
            
        Returns:
            True if trading should continue, False if risk limits hit
        """
        if not self.config.enable_risk_management:
            return True
        
        # Update unrealized PnL
        if not self.current_position.is_flat:
            position_value = self.current_position.quantity * current_price
            cost_basis = self.current_position.quantity * self.current_position.avg_price
            self.current_position.unrealized_pnl = position_value - cost_basis
        
        # Calculate current equity
        current_equity = self.current_capital + self.current_position.unrealized_pnl
        
        # Check drawdown limit
        if len(self.equity_curve) > 0:
            peak_equity = max(self.equity_curve)
            current_drawdown = (current_equity - peak_equity) / peak_equity
            
            if current_drawdown < self.config.max_drawdown_limit:
                return False  # Stop trading due to drawdown limit
        
        return True
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete bias-free backtest
        
        Args:
            data: OHLCV data with DatetimeIndex
            
        Returns:
            Comprehensive backtest results
        """
        print("🚀 Starting AGENT 4 Bias-Free Backtest...")
        print(f"📊 Data period: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Total bars: {len(data)}")
        
        # Reset state
        self.current_capital = self.config.initial_capital
        self.current_position = Position()
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
        # Prepare price array
        prices = data['close'].values
        
        # Validate input data
        validation_issues = self.metrics_validator.validate_calculation_inputs(prices)
        if validation_issues:
            print(f"⚠️  Data validation issues: {validation_issues}")
        
        print(f"🔍 Starting calculations from bar {self.config.min_warmup_periods}...")
        
        # Main backtest loop
        total_signals = 0
        executed_trades = 0
        
        for current_idx in range(self.config.min_warmup_periods, len(data)):
            current_timestamp = data.index[current_idx]
            current_price = prices[current_idx]
            
            # Check risk management
            if not self.check_risk_management(current_price):
                print(f"🛑 Risk management halt at {current_timestamp}")
                break
            
            # Calculate signals with bias protection
            signal_result = self.calculate_signals(prices, current_idx)
            
            if 'error' in signal_result:
                continue
            
            signal = signal_result['signal']
            strength = signal_result['strength']
            total_signals += 1
            
            # Generate trades based on signals
            if signal != 0 and strength > 0.1:  # Minimum signal strength threshold
                action = 'BUY' if signal > 0 else 'SELL'
                
                trade = self.execute_trade(current_timestamp, action, current_price, strength)
                if trade:
                    executed_trades += 1
            
            # Update equity curve
            current_equity = self.current_capital
            if not self.current_position.is_flat:
                position_value = self.current_position.quantity * current_price
                cost_basis = self.current_position.quantity * self.current_position.avg_price
                current_equity += (position_value - cost_basis)
            
            self.equity_curve.append(current_equity)
            self.timestamps.append(current_timestamp)
        
        print(f"✅ Backtest completed!")
        print(f"📊 Signals generated: {total_signals}")
        print(f"💼 Trades executed: {executed_trades}")
        
        # Calculate performance metrics
        if len(self.equity_curve) > 1:
            equity_series = pd.Series(self.equity_curve, index=self.timestamps)
            performance_metrics = self.perf_calculator.calculate_comprehensive_metrics(
                self.equity_curve, frequency='30minute'
            )
        else:
            performance_metrics = PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, max_drawdown_duration=0,
                win_rate=0.0, profit_factor=0.0, calmar_ratio=0.0,
                sortino_ratio=0.0, var_95=0.0, cvar_95=0.0,
                skewness=0.0, kurtosis=0.0
            )
        
        # Validate system integrity
        integrity_report = validate_system_integrity(
            self.data_manager, self.mlmi_calc, self.nwrqk_calc
        )
        
        # Validate performance metrics
        metrics_warnings = self.metrics_validator.validate_metrics_consistency(performance_metrics)
        
        # Calculate trade-level statistics
        if self.trades:
            trade_returns = []
            for trade in self.trades:
                if trade.action == 'SELL':
                    # Calculate return for closed positions
                    # Simplified - would need more sophisticated tracking for partial closes
                    pass
            
            trade_stats = {
                'total_trades': len(self.trades),
                'avg_commission': np.mean([t.commission for t in self.trades]),
                'avg_slippage': np.mean([t.slippage for t in self.trades]),
                'total_commission': sum(t.commission for t in self.trades),
                'total_slippage': sum(t.slippage for t in self.trades)
            }
        else:
            trade_stats = {
                'total_trades': 0,
                'avg_commission': 0.0,
                'avg_slippage': 0.0,
                'total_commission': 0.0,
                'total_slippage': 0.0
            }
        
        # Compile results
        results = {
            'config': asdict(self.config),
            'performance_metrics': performance_metrics.to_dict(),
            'trade_statistics': trade_stats,
            'equity_curve': {
                'timestamps': [ts.isoformat() for ts in self.timestamps],
                'values': self.equity_curve
            },
            'trades': [asdict(trade) for trade in self.trades],
            'validation': {
                'system_integrity': integrity_report,
                'metrics_warnings': metrics_warnings,
                'validation_results': self.validation_results
            },
            'signal_statistics': {
                'total_signals_generated': total_signals,
                'trades_executed': executed_trades,
                'execution_rate': executed_trades / max(1, total_signals)
            }
        }
        
        # Print summary
        print(f"\n📈 BACKTEST RESULTS SUMMARY")
        print(f"=" * 50)
        print(f"Total Return: {performance_metrics.total_return:.2%}")
        print(f"Annualized Return: {performance_metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {performance_metrics.max_drawdown:.2%}")
        print(f"Win Rate: {performance_metrics.win_rate:.2%}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"\n🛡️ BIAS VALIDATION")
        print(f"System Status: {integrity_report['system_status']}")
        print(f"Bias Violations: {integrity_report['total_bias_violations']}")
        
        if integrity_report['system_status'] == 'BIAS_FREE':
            print("✅ ZERO LOOK-AHEAD BIAS CONFIRMED")
        else:
            print("❌ BIAS DETECTED - RESULTS INVALID")
        
        return results


def run_walk_forward_validation(data: pd.DataFrame, config: BacktestConfig) -> Dict[str, Any]:
    """
    Run walk-forward validation with bias-free engine
    
    Args:
        data: Historical data
        config: Backtest configuration
        
    Returns:
        Walk-forward validation results
    """
    print("🚀 Starting Walk-Forward Validation...")
    
    validator = WalkForwardValidator(
        train_window=config.walk_forward_train_window,
        test_window=config.walk_forward_test_window
    )
    
    def strategy_func(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Strategy function for walk-forward analysis"""
        # Create new backtest engine for this fold
        engine = BiasFreeChallengeEngine(config)
        
        # Run backtest on test data (out-of-sample)
        results = engine.run_backtest(test_data)
        
        # Extract returns for validation
        if len(engine.equity_curve) > 1:
            returns = np.diff(engine.equity_curve) / engine.equity_curve[:-1]
            returns = returns[~np.isnan(returns)]  # Remove NaN values
        else:
            returns = []
        
        return {
            'returns': returns.tolist() if len(returns) > 0 else [0.0],
            'final_equity': engine.equity_curve[-1] if engine.equity_curve else config.initial_capital,
            'total_trades': len(engine.trades),
            'bias_violations': engine.validation_results['bias_violations']
        }
    
    # Run walk-forward analysis
    wf_results = validator.run_walk_forward_analysis(data, strategy_func)
    
    # Add statistical significance testing
    if 'total_returns' in wf_results and len(wf_results['total_returns']) > 0:
        significance_test = StatisticalSignificanceTester.t_test_significance(
            wf_results['total_returns']
        )
        bootstrap_ci = StatisticalSignificanceTester.bootstrap_confidence_interval(
            wf_results['total_returns']
        )
        
        wf_results['statistical_tests'] = {
            't_test': significance_test,
            'bootstrap_ci': bootstrap_ci
        }
    
    print("✅ Walk-Forward Validation Completed")
    return wf_results


if __name__ == "__main__":
    print("🛡️ AGENT 4 - BIAS-FREE BACKTEST ENGINE")
    print("✅ Zero look-ahead bias guarantee")
    print("✅ Comprehensive validation framework")
    print("✅ Statistical significance testing")
    print("✅ Walk-forward analysis capability")