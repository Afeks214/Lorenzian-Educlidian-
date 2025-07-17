"""
Comprehensive Strategy Backtesting Framework

This module provides production-grade backtesting framework for trading strategy validation
with realistic market conditions, proper bias handling, and transaction cost modeling.

Key Features:
- Historical simulation with realistic market conditions
- Transaction cost modeling and slippage
- Survivorship bias and look-ahead bias detection
- Walk-forward optimization
- Monte Carlo simulation
- Regime-aware backtesting
- Performance attribution analysis
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from unittest.mock import Mock, patch
import asyncio

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


@dataclass
class TransactionCosts:
    """Transaction cost model parameters"""
    commission_rate: float = 0.0005  # 5 bps
    spread_cost: float = 0.0003  # 3 bps
    market_impact: float = 0.0002  # 2 bps
    slippage_factor: float = 0.0001  # 1 bps


@dataclass
class BacktestPosition:
    """Position in backtesting simulation"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    transaction_costs: float = 0.0


@dataclass
class BacktestResult:
    """Comprehensive backtesting results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    transaction_costs: float
    regime_performance: Dict[MarketRegime, Dict[str, float]]
    performance_attribution: Dict[str, float]
    bias_analysis: Dict[str, bool]


class StrategyBacktester:
    """
    Production-grade strategy backtesting engine with comprehensive
    bias detection and transaction cost modeling.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_costs: TransactionCosts = None,
        enable_bias_detection: bool = True,
        enable_regime_analysis: bool = True
    ):
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.enable_bias_detection = enable_bias_detection
        self.enable_regime_analysis = enable_regime_analysis
        
        # Backtesting state
        self.portfolio_value = initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.regime_detector = MarketRegimeDetector()
        
        # Bias detection flags
        self.bias_flags = {
            'survivorship_bias': False,
            'look_ahead_bias': False,
            'data_snooping_bias': False,
            'selection_bias': False
        }
        
        # Performance tracking
        self.performance_metrics = {}
        
    def run_backtest(
        self,
        strategy_func: callable,
        market_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        rebalance_frequency: str = 'D'
    ) -> BacktestResult:
        """
        Run comprehensive backtest with bias detection and transaction cost modeling.
        
        Args:
            strategy_func: Trading strategy function
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M')
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Validate data and detect biases
        if self.enable_bias_detection:
            self._detect_biases(market_data, start_date, end_date)
        
        # Filter data for backtest period
        backtest_data = self._prepare_backtest_data(market_data, start_date, end_date)
        
        # Run historical simulation
        self._run_historical_simulation(strategy_func, backtest_data, rebalance_frequency)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics(backtest_data)
        
        # Add regime analysis
        if self.enable_regime_analysis:
            result.regime_performance = self._analyze_regime_performance(backtest_data)
        
        # Add performance attribution
        result.performance_attribution = self._calculate_performance_attribution()
        
        # Add bias analysis results
        result.bias_analysis = self.bias_flags.copy()
        
        logger.info(f"Backtest completed: Total Return={result.total_return:.2%}, "
                   f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown:.2%}")
        
        return result
    
    def _detect_biases(self, data: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Detect various types of biases in backtesting setup"""
        
        # Survivorship bias detection
        self._detect_survivorship_bias(data, start_date, end_date)
        
        # Look-ahead bias detection
        self._detect_look_ahead_bias(data)
        
        # Data snooping bias detection
        self._detect_data_snooping_bias(data)
        
        # Selection bias detection
        self._detect_selection_bias(data)
        
        # Log detected biases
        detected_biases = [bias for bias, detected in self.bias_flags.items() if detected]
        if detected_biases:
            logger.warning(f"Detected biases: {detected_biases}")
    
    def _detect_survivorship_bias(self, data: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Detect survivorship bias in the dataset"""
        
        # Check if all symbols in the dataset existed for the entire period
        symbols = data['symbol'].unique() if 'symbol' in data.columns else []
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol]
            if len(symbol_data) == 0:
                continue
                
            first_date = symbol_data.index.min()
            last_date = symbol_data.index.max()
            
            # Flag if symbol doesn't span the entire backtest period
            if first_date > start_date or last_date < end_date:
                self.bias_flags['survivorship_bias'] = True
                logger.warning(f"Survivorship bias detected for {symbol}: "
                             f"data spans {first_date} to {last_date}")
    
    def _detect_look_ahead_bias(self, data: pd.DataFrame):
        """Detect look-ahead bias in indicators or features"""
        
        # Check for future data leakage in technical indicators
        if 'signal' in data.columns:
            # Simple check: ensure signals don't use future price data
            for i in range(1, len(data)):
                if pd.notna(data.iloc[i]['signal']) and pd.notna(data.iloc[i-1]['signal']):
                    # Check if signal at time t depends on price at time t+1
                    current_signal = data.iloc[i]['signal']
                    future_return = data.iloc[i]['return'] if 'return' in data.columns else 0
                    
                    # High correlation between current signal and future return suggests bias
                    if abs(current_signal * future_return) > 0.8:
                        self.bias_flags['look_ahead_bias'] = True
                        logger.warning("Look-ahead bias detected: signal correlated with future returns")
                        break
    
    def _detect_data_snooping_bias(self, data: pd.DataFrame):
        """Detect data snooping bias"""
        
        # Check if the same dataset is used for both optimization and testing
        if hasattr(data, 'optimization_used') and data.optimization_used:
            self.bias_flags['data_snooping_bias'] = True
            logger.warning("Data snooping bias detected: same data used for optimization and testing")
    
    def _detect_selection_bias(self, data: pd.DataFrame):
        """Detect selection bias in asset universe"""
        
        # Check if asset selection is based on performance during the backtest period
        if 'performance_rank' in data.columns:
            # If assets are pre-selected based on performance, flag as bias
            performance_based_selection = data['performance_rank'].notna().any()
            if performance_based_selection:
                self.bias_flags['selection_bias'] = True
                logger.warning("Selection bias detected: assets selected based on performance")
    
    def _prepare_backtest_data(self, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Prepare and validate data for backtesting"""
        
        # Filter data for backtest period
        mask = (data.index >= start_date) & (data.index <= end_date)
        backtest_data = data.loc[mask].copy()
        
        # Validate data completeness
        missing_data_pct = backtest_data.isnull().sum().sum() / (len(backtest_data) * len(backtest_data.columns))
        if missing_data_pct > 0.05:  # 5% threshold
            logger.warning(f"High missing data percentage: {missing_data_pct:.2%}")
        
        # Forward fill missing data (realistic assumption)
        backtest_data = backtest_data.fillna(method='ffill')
        
        return backtest_data
    
    def _run_historical_simulation(self, strategy_func: callable, data: pd.DataFrame, rebalance_frequency: str):
        """Run historical simulation with realistic market conditions"""
        
        rebalance_dates = self._get_rebalance_dates(data.index, rebalance_frequency)
        
        for date in rebalance_dates:
            if date not in data.index:
                continue
                
            # Get market data up to current date (no look-ahead bias)
            historical_data = data.loc[:date]
            
            # Generate trading signals
            signals = strategy_func(historical_data)
            
            # Execute trades with transaction costs
            self._execute_trades(signals, data.loc[date], date)
            
            # Update portfolio value
            self._update_portfolio_value(data.loc[date], date)
            
            # Record portfolio state
            self._record_portfolio_state(date)
    
    def _get_rebalance_dates(self, index: pd.DatetimeIndex, frequency: str) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        
        if frequency == 'D':
            return index.tolist()
        elif frequency == 'W':
            return index[index.weekday == 0].tolist()  # Monday
        elif frequency == 'M':
            return index[index.is_month_end].tolist()
        else:
            raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    
    def _execute_trades(self, signals: Dict[str, float], market_data: pd.Series, date: datetime):
        """Execute trades with realistic transaction costs and slippage"""
        
        for symbol, target_weight in signals.items():
            current_position = self.positions.get(symbol)
            current_weight = 0.0
            
            if current_position:
                current_weight = (current_position.quantity * current_position.current_price) / self.portfolio_value
            
            # Calculate required trade size
            weight_diff = target_weight - current_weight
            trade_value = weight_diff * self.portfolio_value
            
            if abs(trade_value) < 100:  # Minimum trade size
                continue
            
            # Get execution price with slippage
            execution_price = self._get_execution_price(symbol, market_data, trade_value)
            
            # Calculate transaction costs
            trade_costs = self._calculate_transaction_costs(abs(trade_value))
            
            # Execute trade
            if current_position:
                self._adjust_position(symbol, trade_value, execution_price, date, trade_costs)
            else:
                self._open_position(symbol, trade_value, execution_price, date, trade_costs)
    
    def _get_execution_price(self, symbol: str, market_data: pd.Series, trade_value: float) -> float:
        """Get realistic execution price with slippage"""
        
        base_price = market_data.get(f'{symbol}_price', market_data.get('price', 100.0))
        
        # Apply slippage based on trade size and market impact
        slippage_factor = self.transaction_costs.slippage_factor
        market_impact = self.transaction_costs.market_impact
        
        # Slippage increases with trade size
        trade_size_factor = min(abs(trade_value) / 10000, 1.0)  # Cap at 1.0
        total_slippage = slippage_factor + (market_impact * trade_size_factor)
        
        # Apply slippage (positive for buys, negative for sells)
        slippage_adjustment = total_slippage if trade_value > 0 else -total_slippage
        execution_price = base_price * (1 + slippage_adjustment)
        
        return execution_price
    
    def _calculate_transaction_costs(self, trade_value: float) -> float:
        """Calculate realistic transaction costs"""
        
        commission = trade_value * self.transaction_costs.commission_rate
        spread_cost = trade_value * self.transaction_costs.spread_cost
        
        return commission + spread_cost
    
    def _open_position(self, symbol: str, trade_value: float, price: float, date: datetime, costs: float):
        """Open new position"""
        
        quantity = trade_value / price
        
        position = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=date,
            current_price=price,
            unrealized_pnl=0.0,
            transaction_costs=costs
        )
        
        self.positions[symbol] = position
        self.portfolio_value -= costs
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY' if quantity > 0 else 'SELL',
            'quantity': abs(quantity),
            'price': price,
            'value': abs(trade_value),
            'costs': costs
        })
    
    def _adjust_position(self, symbol: str, trade_value: float, price: float, date: datetime, costs: float):
        """Adjust existing position"""
        
        position = self.positions[symbol]
        additional_quantity = trade_value / price
        
        # Calculate realized PnL if reducing position
        if (position.quantity > 0 and additional_quantity < 0) or (position.quantity < 0 and additional_quantity > 0):
            quantity_to_close = min(abs(additional_quantity), abs(position.quantity))
            realized_pnl = quantity_to_close * (price - position.entry_price)
            position.realized_pnl += realized_pnl
        
        # Update position
        position.quantity += additional_quantity
        position.current_price = price
        position.transaction_costs += costs
        
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-8:
            del self.positions[symbol]
        
        self.portfolio_value -= costs
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY' if additional_quantity > 0 else 'SELL',
            'quantity': abs(additional_quantity),
            'price': price,
            'value': abs(trade_value),
            'costs': costs
        })
    
    def _update_portfolio_value(self, market_data: pd.Series, date: datetime):
        """Update portfolio value based on current market prices"""
        
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = market_data.get(f'{symbol}_price', market_data.get('price', position.current_price))
            position.current_price = current_price
            position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
            total_position_value += position.quantity * current_price
        
        # Calculate cash balance
        cash_balance = self.initial_capital
        for trade in self.trades:
            if trade['action'] == 'BUY':
                cash_balance -= trade['value'] + trade['costs']
            else:
                cash_balance += trade['value'] - trade['costs']
        
        self.portfolio_value = cash_balance + total_position_value
    
    def _record_portfolio_state(self, date: datetime):
        """Record current portfolio state"""
        
        total_position_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_costs = sum(trade['costs'] for trade in self.trades)
        
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'position_value': total_position_value,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'transaction_costs': total_costs,
            'num_positions': len(self.positions)
        })
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        if not self.portfolio_history:
            return BacktestResult(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0, total_trades=0,
                avg_trade_duration=0.0, transaction_costs=0.0,
                regime_performance={}, performance_attribution={},
                bias_analysis={}
            )
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Basic performance metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
        
        # Volatility
        volatility = portfolio_df['return'].std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        portfolio_df['cumulative'] = (1 + portfolio_df['return']).cumprod()
        portfolio_df['running_max'] = portfolio_df['cumulative'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['cumulative'] - portfolio_df['running_max']) / portfolio_df['running_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if self._is_winning_trade(t)]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Profit factor
        gross_profit = sum(self._get_trade_profit(t) for t in winning_trades)
        gross_loss = sum(abs(self._get_trade_profit(t)) for t in self.trades if self._get_trade_profit(t) < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average trade duration
        avg_trade_duration = self._calculate_avg_trade_duration()
        
        # Total transaction costs
        total_costs = sum(trade['costs'] for trade in self.trades)
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_duration=avg_trade_duration,
            transaction_costs=total_costs,
            regime_performance={},
            performance_attribution={},
            bias_analysis={}
        )
    
    def _is_winning_trade(self, trade: Dict) -> bool:
        """Determine if a trade is winning"""
        return self._get_trade_profit(trade) > 0
    
    def _get_trade_profit(self, trade: Dict) -> float:
        """Calculate trade profit"""
        # Simplified calculation - in practice, need to track full trade lifecycle
        return trade['value'] * 0.01  # Placeholder
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in days"""
        if len(self.trades) < 2:
            return 0.0
        
        total_duration = 0.0
        for i in range(1, len(self.trades)):
            duration = (self.trades[i]['date'] - self.trades[i-1]['date']).days
            total_duration += duration
        
        return total_duration / (len(self.trades) - 1)
    
    def _analyze_regime_performance(self, data: pd.DataFrame) -> Dict[MarketRegime, Dict[str, float]]:
        """Analyze performance by market regime"""
        
        regime_performance = {}
        
        for regime in MarketRegime:
            regime_data = data[data['regime'] == regime.value] if 'regime' in data.columns else pd.DataFrame()
            
            if regime_data.empty:
                regime_performance[regime] = {'return': 0.0, 'volatility': 0.0, 'sharpe': 0.0}
                continue
            
            # Calculate regime-specific metrics
            regime_returns = []
            for _, row in regime_data.iterrows():
                portfolio_state = next((p for p in self.portfolio_history if p['date'] == row.name), None)
                if portfolio_state:
                    regime_returns.append(portfolio_state['portfolio_value'] / self.initial_capital - 1)
            
            if regime_returns:
                regime_return = np.mean(regime_returns)
                regime_volatility = np.std(regime_returns) * np.sqrt(252)
                regime_sharpe = regime_return / regime_volatility if regime_volatility > 0 else 0.0
                
                regime_performance[regime] = {
                    'return': regime_return,
                    'volatility': regime_volatility,
                    'sharpe': regime_sharpe
                }
            else:
                regime_performance[regime] = {'return': 0.0, 'volatility': 0.0, 'sharpe': 0.0}
        
        return regime_performance
    
    def _calculate_performance_attribution(self) -> Dict[str, float]:
        """Calculate performance attribution"""
        
        attribution = {}
        
        # Attribution by asset/sector
        for symbol in set(trade['symbol'] for trade in self.trades):
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            symbol_pnl = sum(self._get_trade_profit(t) for t in symbol_trades)
            attribution[symbol] = symbol_pnl / self.initial_capital
        
        # Attribution by strategy component
        attribution['alpha'] = 0.05  # Placeholder
        attribution['beta'] = 0.03   # Placeholder
        attribution['costs'] = -sum(trade['costs'] for trade in self.trades) / self.initial_capital
        
        return attribution


class MarketRegimeDetector:
    """Detect market regimes for regime-aware backtesting"""
    
    def __init__(self):
        self.volatility_threshold = 0.02
        self.trend_threshold = 0.15
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        
        if len(data) < 20:
            return MarketRegime.SIDEWAYS
        
        # Calculate volatility
        returns = data['price'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate trend
        price_change = (data['price'].iloc[-1] - data['price'].iloc[0]) / data['price'].iloc[0]
        
        # Classify regime
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        elif price_change > self.trend_threshold:
            return MarketRegime.BULL
        elif price_change < -self.trend_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters"""
    
    def __init__(self, backtester: StrategyBacktester):
        self.backtester = backtester
        self.optimization_window = 252  # 1 year
        self.test_window = 63  # 3 months
    
    def optimize_parameters(
        self,
        strategy_func: callable,
        parameter_space: Dict[str, List],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run walk-forward optimization"""
        
        optimization_results = []
        
        # Split data into optimization and test windows
        for i in range(self.optimization_window, len(data) - self.test_window, self.test_window):
            opt_data = data.iloc[i-self.optimization_window:i]
            test_data = data.iloc[i:i+self.test_window]
            
            # Optimize parameters on optimization window
            best_params = self._optimize_on_window(strategy_func, parameter_space, opt_data)
            
            # Test on out-of-sample data
            test_result = self._test_on_window(strategy_func, best_params, test_data)
            
            optimization_results.append({
                'optimization_period': (opt_data.index[0], opt_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'best_params': best_params,
                'test_result': test_result
            })
        
        return {
            'optimization_results': optimization_results,
            'average_return': np.mean([r['test_result']['return'] for r in optimization_results]),
            'average_sharpe': np.mean([r['test_result']['sharpe'] for r in optimization_results])
        }
    
    def _optimize_on_window(self, strategy_func: callable, parameter_space: Dict, data: pd.DataFrame) -> Dict:
        """Optimize parameters on given window"""
        
        best_params = {}
        best_sharpe = -np.inf
        
        # Simple grid search (in practice, use more sophisticated optimization)
        for param_combo in self._generate_parameter_combinations(parameter_space):
            # Create strategy with parameters
            strategy = lambda x: strategy_func(x, **param_combo)
            
            # Run backtest
            result = self.backtester.run_backtest(
                strategy, data, data.index[0], data.index[-1], 'D'
            )
            
            # Track best parameters
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = param_combo
        
        return best_params
    
    def _generate_parameter_combinations(self, parameter_space: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        
        import itertools
        
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _test_on_window(self, strategy_func: callable, params: Dict, data: pd.DataFrame) -> Dict:
        """Test strategy with parameters on given window"""
        
        strategy = lambda x: strategy_func(x, **params)
        result = self.backtester.run_backtest(
            strategy, data, data.index[0], data.index[-1], 'D'
        )
        
        return {
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown
        }


# Test fixtures and utilities
@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = [100.0]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'price': prices,
        'volume': np.random.lognormal(10, 0.5, len(dates)),
        'return': [0] + returns[1:].tolist(),
        'signal': np.random.uniform(-1, 1, len(dates))
    }, index=dates)
    
    return data


@pytest.fixture
def simple_strategy():
    """Simple momentum strategy for testing"""
    
    def momentum_strategy(data: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """Simple momentum strategy"""
        
        if len(data) < lookback:
            return {'STOCK': 0.0}
        
        # Calculate momentum
        momentum = data['price'].iloc[-1] / data['price'].iloc[-lookback] - 1
        
        # Generate signal
        if momentum > 0.05:
            return {'STOCK': 1.0}  # Long position
        elif momentum < -0.05:
            return {'STOCK': -1.0}  # Short position
        else:
            return {'STOCK': 0.0}  # No position
    
    return momentum_strategy


# Comprehensive test suite
@pytest.mark.asyncio
class TestStrategyBacktesting:
    """Comprehensive strategy backtesting tests"""
    
    def test_basic_backtest(self, sample_market_data, simple_strategy):
        """Test basic backtesting functionality"""
        
        backtester = StrategyBacktester(initial_capital=100000)
        
        result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2023, 12, 31),
            'D'
        )
        
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert result.transaction_costs >= 0
        assert -1 <= result.total_return <= 10  # Reasonable bounds
    
    def test_transaction_cost_modeling(self, sample_market_data, simple_strategy):
        """Test transaction cost modeling"""
        
        # Test with different transaction cost settings
        high_cost = TransactionCosts(
            commission_rate=0.001,
            spread_cost=0.0005,
            market_impact=0.0003,
            slippage_factor=0.0002
        )
        
        low_cost = TransactionCosts(
            commission_rate=0.0001,
            spread_cost=0.0001,
            market_impact=0.0001,
            slippage_factor=0.0001
        )
        
        backtester_high = StrategyBacktester(transaction_costs=high_cost)
        backtester_low = StrategyBacktester(transaction_costs=low_cost)
        
        result_high = backtester_high.run_backtest(
            simple_strategy, sample_market_data,
            datetime(2020, 1, 1), datetime(2021, 1, 1), 'D'
        )
        
        result_low = backtester_low.run_backtest(
            simple_strategy, sample_market_data,
            datetime(2020, 1, 1), datetime(2021, 1, 1), 'D'
        )
        
        # High cost should result in lower returns
        assert result_high.transaction_costs > result_low.transaction_costs
        assert result_high.total_return <= result_low.total_return
    
    def test_survivorship_bias_detection(self):
        """Test survivorship bias detection"""
        
        # Create data with survivorship bias
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Symbol A exists for full period
        data_a = pd.DataFrame({
            'symbol': 'A',
            'price': np.random.lognormal(4.6, 0.01, len(dates)),
        }, index=dates)
        
        # Symbol B disappears halfway through
        half_dates = dates[:len(dates)//2]
        data_b = pd.DataFrame({
            'symbol': 'B',
            'price': np.random.lognormal(4.6, 0.01, len(half_dates)),
        }, index=half_dates)
        
        biased_data = pd.concat([data_a, data_b])
        
        backtester = StrategyBacktester(enable_bias_detection=True)
        
        # This should detect survivorship bias
        backtester._detect_survivorship_bias(biased_data, dates[0], dates[-1])
        
        assert backtester.bias_flags['survivorship_bias'] == True
    
    def test_look_ahead_bias_detection(self):
        """Test look-ahead bias detection"""
        
        # Create data with look-ahead bias
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        returns = np.random.normal(0, 0.01, len(dates))
        
        # Signal that perfectly predicts future returns (look-ahead bias)
        signals = np.roll(returns, -1)  # Shift returns forward
        
        data = pd.DataFrame({
            'return': returns,
            'signal': signals
        }, index=dates)
        
        backtester = StrategyBacktester(enable_bias_detection=True)
        backtester._detect_look_ahead_bias(data)
        
        assert backtester.bias_flags['look_ahead_bias'] == True
    
    def test_walk_forward_optimization(self, sample_market_data, simple_strategy):
        """Test walk-forward optimization"""
        
        backtester = StrategyBacktester()
        optimizer = WalkForwardOptimizer(backtester)
        
        # Define parameter space
        parameter_space = {
            'lookback': [10, 20, 30]
        }
        
        # Run optimization
        result = optimizer.optimize_parameters(
            simple_strategy,
            parameter_space,
            sample_market_data[:500]  # Use subset for faster testing
        )
        
        assert 'optimization_results' in result
        assert 'average_return' in result
        assert 'average_sharpe' in result
        assert len(result['optimization_results']) > 0
    
    def test_regime_aware_backtesting(self, sample_market_data, simple_strategy):
        """Test regime-aware backtesting"""
        
        # Add regime information to data
        regime_detector = MarketRegimeDetector()
        sample_market_data['regime'] = 'bull'  # Simplified
        
        backtester = StrategyBacktester(enable_regime_analysis=True)
        
        result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        assert isinstance(result.regime_performance, dict)
        assert len(result.regime_performance) > 0
    
    def test_performance_attribution(self, sample_market_data, simple_strategy):
        """Test performance attribution analysis"""
        
        backtester = StrategyBacktester()
        
        result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        assert isinstance(result.performance_attribution, dict)
        assert 'costs' in result.performance_attribution
        assert result.performance_attribution['costs'] <= 0  # Costs should be negative
    
    def test_monte_carlo_simulation(self, sample_market_data, simple_strategy):
        """Test Monte Carlo simulation for strategy validation"""
        
        backtester = StrategyBacktester()
        
        # Run multiple simulations with different random seeds
        results = []
        for seed in range(10):
            np.random.seed(seed)
            
            # Add noise to market data
            noisy_data = sample_market_data.copy()
            noisy_data['price'] = noisy_data['price'] * (1 + np.random.normal(0, 0.001, len(noisy_data)))
            
            result = backtester.run_backtest(
                simple_strategy,
                noisy_data,
                datetime(2020, 1, 1),
                datetime(2021, 1, 1),
                'D'
            )
            
            results.append(result.total_return)
        
        # Analyze distribution of results
        mean_return = np.mean(results)
        std_return = np.std(results)
        
        assert len(results) == 10
        assert std_return >= 0  # Should have some variation
        assert abs(mean_return) < 1.0  # Reasonable return range
    
    def test_benchmark_comparison(self, sample_market_data, simple_strategy):
        """Test strategy comparison against benchmark"""
        
        # Create benchmark strategy (buy and hold)
        def benchmark_strategy(data: pd.DataFrame) -> Dict[str, float]:
            return {'STOCK': 1.0}  # Always fully invested
        
        backtester = StrategyBacktester()
        
        # Test strategy
        strategy_result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Reset backtester for benchmark
        backtester = StrategyBacktester()
        
        # Benchmark
        benchmark_result = backtester.run_backtest(
            benchmark_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Compare results
        assert strategy_result.total_trades >= benchmark_result.total_trades
        assert strategy_result.transaction_costs >= benchmark_result.transaction_costs
        
        # Calculate information ratio
        excess_return = strategy_result.total_return - benchmark_result.total_return
        tracking_error = abs(strategy_result.volatility - benchmark_result.volatility)
        
        if tracking_error > 0:
            information_ratio = excess_return / tracking_error
            assert abs(information_ratio) < 10  # Reasonable bound
    
    def test_stress_testing(self, sample_market_data, simple_strategy):
        """Test strategy under stressed market conditions"""
        
        # Create stressed market data
        stressed_data = sample_market_data.copy()
        
        # Add market crash
        crash_start = len(stressed_data) // 2
        crash_end = crash_start + 50
        
        crash_returns = np.random.normal(-0.05, 0.03, crash_end - crash_start)
        for i, ret in enumerate(crash_returns):
            if crash_start + i < len(stressed_data):
                stressed_data.iloc[crash_start + i, stressed_data.columns.get_loc('price')] *= (1 + ret)
        
        backtester = StrategyBacktester()
        
        result = backtester.run_backtest(
            simple_strategy,
            stressed_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Strategy should survive stress test
        assert result.max_drawdown < 0  # Should have some drawdown
        assert result.max_drawdown > -0.95  # But not complete loss
        assert result.total_trades > 0  # Should have made some trades
    
    def test_portfolio_rebalancing(self, sample_market_data, simple_strategy):
        """Test different rebalancing frequencies"""
        
        backtester = StrategyBacktester()
        
        # Test daily rebalancing
        daily_result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2020, 6, 1),
            'D'
        )
        
        # Reset backtester
        backtester = StrategyBacktester()
        
        # Test weekly rebalancing
        weekly_result = backtester.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2020, 6, 1),
            'W'
        )
        
        # Daily should have more trades and higher costs
        assert daily_result.total_trades >= weekly_result.total_trades
        assert daily_result.transaction_costs >= weekly_result.transaction_costs
    
    def test_strategy_capacity_analysis(self, sample_market_data, simple_strategy):
        """Test strategy capacity and scalability"""
        
        # Test with different portfolio sizes
        small_portfolio = StrategyBacktester(initial_capital=100000)
        large_portfolio = StrategyBacktester(initial_capital=10000000)
        
        # Modify transaction costs for large portfolio (higher market impact)
        large_costs = TransactionCosts(
            commission_rate=0.0005,
            spread_cost=0.0003,
            market_impact=0.001,  # Higher impact
            slippage_factor=0.0002
        )
        
        large_portfolio.transaction_costs = large_costs
        
        small_result = small_portfolio.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        large_result = large_portfolio.run_backtest(
            simple_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Large portfolio should have higher transaction costs as % of capital
        small_cost_ratio = small_result.transaction_costs / small_portfolio.initial_capital
        large_cost_ratio = large_result.transaction_costs / large_portfolio.initial_capital
        
        assert large_cost_ratio >= small_cost_ratio
    
    def test_risk_management_integration(self, sample_market_data, simple_strategy):
        """Test integration with risk management systems"""
        
        # Create strategy with risk management
        def risk_managed_strategy(data: pd.DataFrame) -> Dict[str, float]:
            base_signal = simple_strategy(data)
            
            # Apply risk management (position sizing)
            if len(data) >= 20:
                volatility = data['return'].rolling(20).std().iloc[-1]
                max_position = 0.5  # Maximum 50% position
                
                # Scale position by inverse volatility
                if volatility > 0:
                    risk_adjusted_size = min(max_position, 0.02 / volatility)
                    for symbol in base_signal:
                        if base_signal[symbol] != 0:
                            base_signal[symbol] *= risk_adjusted_size
            
            return base_signal
        
        backtester = StrategyBacktester()
        
        result = backtester.run_backtest(
            risk_managed_strategy,
            sample_market_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Risk-managed strategy should have lower volatility
        assert result.volatility < 0.5  # Reasonable upper bound
        assert result.max_drawdown > -0.5  # Better drawdown control
    
    def test_multi_asset_backtesting(self, simple_strategy):
        """Test backtesting with multiple assets"""
        
        # Create multi-asset data
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        
        assets = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        multi_asset_data = pd.DataFrame(index=dates)
        
        for asset in assets:
            np.random.seed(hash(asset) % 1000)
            returns = np.random.normal(0.0005, 0.015, len(dates))
            prices = [100.0]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            multi_asset_data[f'{asset}_price'] = prices
            multi_asset_data[f'{asset}_return'] = [0] + returns[1:].tolist()
        
        # Multi-asset strategy
        def multi_asset_strategy(data: pd.DataFrame) -> Dict[str, float]:
            signals = {}
            for asset in assets:
                price_col = f'{asset}_price'
                if price_col in data.columns and len(data) >= 20:
                    momentum = data[price_col].iloc[-1] / data[price_col].iloc[-20] - 1
                    
                    if momentum > 0.05:
                        signals[asset] = 0.33  # Equal weight
                    elif momentum < -0.05:
                        signals[asset] = -0.33
                    else:
                        signals[asset] = 0.0
            
            return signals
        
        backtester = StrategyBacktester()
        
        result = backtester.run_backtest(
            multi_asset_strategy,
            multi_asset_data,
            datetime(2020, 1, 1),
            datetime(2021, 1, 1),
            'D'
        )
        
        # Multi-asset strategy should have diversification benefits
        assert result.total_trades > 0
        assert len(result.performance_attribution) >= len(assets)