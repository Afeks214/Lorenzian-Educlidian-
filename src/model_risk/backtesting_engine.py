"""
Backtesting Engine for Model Risk Management

This module provides comprehensive backtesting capabilities for model validation
and performance assessment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..core.event_bus import EventBus

logger = structlog.get_logger()


class BacktestStatus(Enum):
    """Status of backtest execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.2
    rebalance_frequency: str = "daily"
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02
    lookback_window: int = 252
    out_of_sample_ratio: float = 0.2


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    tracking_error: float
    alpha: float
    beta: float
    r_squared: float
    trades_count: int
    avg_trade_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
            "alpha": self.alpha,
            "beta": self.beta,
            "r_squared": self.r_squared,
            "trades_count": self.trades_count,
            "avg_trade_duration": self.avg_trade_duration
        }


@dataclass
class BacktestResult:
    """Result of backtest execution"""
    backtest_id: str
    model_id: str
    model_version: str
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Optional[BacktestMetrics]
    returns: Optional[pd.Series]
    positions: Optional[pd.DataFrame]
    trades: Optional[pd.DataFrame]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def is_successful(self) -> bool:
        """Check if backtest was successful"""
        return self.status == BacktestStatus.COMPLETED and self.metrics is not None


class BacktestingEngine:
    """Main backtesting engine for model validation"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.backtest_history: List[BacktestResult] = []
        self.running_backtests: Dict[str, BacktestResult] = {}
        
        logger.info("Backtesting Engine initialized")
    
    def run_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        config: BacktestConfig,
        model_id: str,
        model_version: str
    ) -> BacktestResult:
        """Run comprehensive backtest"""
        backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = BacktestResult(
            backtest_id=backtest_id,
            model_id=model_id,
            model_version=model_version,
            config=config,
            status=BacktestStatus.RUNNING,
            start_time=datetime.now(),
            end_time=None,
            metrics=None,
            returns=None,
            positions=None,
            trades=None
        )
        
        self.running_backtests[backtest_id] = result
        
        try:
            # Filter data by date range
            mask = (data.index >= config.start_date) & (data.index <= config.end_date)
            backtest_data = data.loc[mask].copy()
            
            if backtest_data.empty:
                raise ValueError("No data available for backtest period")
            
            # Split data for out-of-sample testing
            split_idx = int(len(backtest_data) * (1 - config.out_of_sample_ratio))
            train_data = backtest_data.iloc[:split_idx]
            test_data = backtest_data.iloc[split_idx:]
            
            # Generate signals
            signals = self._generate_signals(model, train_data, test_data)
            
            # Execute trades
            returns, positions, trades = self._execute_backtest(
                signals, test_data, config
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(returns, positions, trades, config)
            
            # Update result
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now()
            result.metrics = metrics
            result.returns = returns
            result.positions = positions
            result.trades = trades
            
            logger.info(
                "Backtest completed successfully",
                backtest_id=backtest_id,
                model_id=model_id,
                execution_time=result.execution_time,
                total_return=metrics.total_return,
                sharpe_ratio=metrics.sharpe_ratio
            )
            
        except Exception as e:
            result.status = BacktestStatus.FAILED
            result.end_time = datetime.now()
            result.errors.append(str(e))
            
            logger.error(
                "Backtest failed",
                backtest_id=backtest_id,
                model_id=model_id,
                error=str(e)
            )
        
        finally:
            # Move to history
            self.backtest_history.append(result)
            if backtest_id in self.running_backtests:
                del self.running_backtests[backtest_id]
        
        return result
    
    def _generate_signals(self, model: Any, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using the model"""
        signals = pd.Series(index=test_data.index, dtype=float)
        
        # Mock signal generation - in practice, this would use the actual model
        for i in range(len(test_data)):
            # Simple momentum strategy as example
            if i >= 20:  # Need lookback period
                recent_returns = test_data.iloc[i-20:i]['close'].pct_change().mean()
                signals.iloc[i] = np.sign(recent_returns) * min(abs(recent_returns) * 10, 1.0)
            else:
                signals.iloc[i] = 0.0
        
        return signals
    
    def _execute_backtest(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        config: BacktestConfig
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Execute backtest with position management"""
        
        # Initialize tracking variables
        portfolio_value = config.initial_capital
        positions = pd.DataFrame(index=data.index, columns=['position', 'value'])
        trades = []
        returns = pd.Series(index=data.index, dtype=float)
        
        current_position = 0.0
        last_price = None
        
        for i, (date, row) in enumerate(data.iterrows()):
            price = row['close']
            signal = signals.loc[date]
            
            # Calculate target position
            target_position = np.clip(signal, -config.max_position_size, config.max_position_size)
            
            # Calculate position change
            position_change = target_position - current_position
            
            # Execute trade if significant position change
            if abs(position_change) > 0.01:  # 1% threshold
                trade_value = position_change * portfolio_value
                transaction_cost = abs(trade_value) * config.transaction_cost
                slippage_cost = abs(trade_value) * config.slippage
                
                # Record trade
                trades.append({
                    'date': date,
                    'symbol': 'PORTFOLIO',
                    'side': 'BUY' if position_change > 0 else 'SELL',
                    'quantity': abs(position_change),
                    'price': price,
                    'value': trade_value,
                    'cost': transaction_cost + slippage_cost
                })
                
                # Update portfolio value
                portfolio_value -= (transaction_cost + slippage_cost)
                current_position = target_position
            
            # Calculate portfolio value
            if last_price is not None:
                price_return = (price - last_price) / last_price
                portfolio_return = current_position * price_return
                portfolio_value *= (1 + portfolio_return)
                returns.loc[date] = portfolio_return
            else:
                returns.loc[date] = 0.0
            
            # Record position
            positions.loc[date, 'position'] = current_position
            positions.loc[date, 'value'] = portfolio_value
            
            last_price = price
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return returns, positions, trades_df
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        config: BacktestConfig
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        
        # Basic return metrics
        total_return = (positions['value'].iloc[-1] / config.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = (annual_return - config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Trade analysis
        if not trades.empty:
            win_trades = trades[trades['value'] > 0]
            lose_trades = trades[trades['value'] < 0]
            
            win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
            profit_factor = win_trades['value'].sum() / abs(lose_trades['value'].sum()) if len(lose_trades) > 0 else np.inf
            
            # Average trade duration (mock calculation)
            avg_trade_duration = 5.0  # days
            trades_count = len(trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
            trades_count = 0
        
        # Additional risk metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Benchmark comparison (mock values)
        information_ratio = 0.1
        tracking_error = 0.05
        alpha = 0.02
        beta = 0.95
        r_squared = 0.8
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            trades_count=trades_count,
            avg_trade_duration=avg_trade_duration
        )
    
    def get_backtest_history(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[BacktestResult]:
        """Get backtest history"""
        history = self.backtest_history.copy()
        
        if model_id:
            history = [h for h in history if h.model_id == model_id]
        
        # Sort by start time (newest first)
        history.sort(key=lambda x: x.start_time, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_running_backtests(self) -> Dict[str, BacktestResult]:
        """Get currently running backtests"""
        return self.running_backtests.copy()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get backtesting engine status"""
        return {
            "total_backtests": len(self.backtest_history),
            "running_backtests": len(self.running_backtests),
            "successful_backtests": len([b for b in self.backtest_history if b.is_successful()]),
            "failed_backtests": len([b for b in self.backtest_history if b.status == BacktestStatus.FAILED]),
            "average_execution_time": np.mean([b.execution_time for b in self.backtest_history if b.execution_time > 0]),
            "last_backtest": self.backtest_history[-1].start_time.isoformat() if self.backtest_history else None
        }