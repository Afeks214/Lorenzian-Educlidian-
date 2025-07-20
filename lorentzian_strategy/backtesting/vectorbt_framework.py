"""
VectorBT Backtesting Framework for MARL Trading System
====================================================

Production-ready VectorBT framework with:
- Custom indicator pipeline
- Signal generation framework
- Portfolio simulation with transaction costs
- Performance analytics and reporting

Author: Claude Code
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import numba as nb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Data parameters
    data_path: str = "/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Portfolio parameters
    initial_cash: float = 100000.0
    commission: float = 0.0001  # 0.01% commission
    slippage: float = 0.0002   # 0.02% slippage
    
    # Strategy parameters
    fast_period: int = 8
    slow_period: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_position_size: float = 0.25  # 25% of portfolio per position
    
    # Performance targets
    target_sharpe: float = 2.0
    target_max_dd: float = 0.15  # 15% max drawdown
    target_win_rate: float = 0.60  # 60% win rate
    target_profit_factor: float = 1.5

class LorentzianIndicators:
    """Custom Lorentzian Classification indicators for VectorBT"""
    
    @staticmethod
    @nb.jit(nopython=True)
    def lorentzian_distance(x1: float, x2: float, sigma: float = 1.0) -> float:
        """Calculate Lorentzian distance between two points"""
        return np.log(1 + np.abs(x1 - x2) / sigma)
    
    @staticmethod
    @nb.jit(nopython=True)
    def feature_engineering(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                          volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Generate features for Lorentzian classification"""
        n = len(close)
        features = np.zeros((n, 8))  # 8 features
        
        for i in range(period, n):
            # RSI
            gains = 0.0
            losses = 0.0
            for j in range(1, period + 1):
                diff = close[i - j + 1] - close[i - j]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            # Williams %R
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            if highest_high != lowest_low:
                williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                williams_r = -50
            
            # Price momentum
            price_mom = (close[i] - close[i - period]) / close[i - period] * 100
            
            # Volume momentum  
            vol_mom = (volume[i] - volume[i - period]) / volume[i - period] * 100 if volume[i - period] > 0 else 0
            
            # Volatility (rolling std)
            volatility = np.std(close[i - period + 1:i + 1])
            
            # True range
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1]) if i > 0 else tr1
            tr3 = abs(low[i] - close[i - 1]) if i > 0 else tr1
            true_range = max(tr1, tr2, tr3)
            
            # CCI approximation
            typical_price = (high[i] + low[i] + close[i]) / 3
            sma_tp = np.mean(close[i - period + 1:i + 1])  # Approximation
            mean_deviation = np.mean(np.abs(close[i - period + 1:i + 1] - sma_tp))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation) if mean_deviation > 0 else 0
            
            # Waddah Attar Explosion
            wae = volatility * 100  # Simplified version
            
            features[i] = [rsi, williams_r, price_mom, vol_mom, volatility, true_range, cci, wae]
        
        return features
    
    @staticmethod
    @nb.jit(nopython=True)
    def lorentzian_classification(features: np.ndarray, close: np.ndarray, 
                                lookback: int = 100, neighbors: int = 8) -> np.ndarray:
        """Lorentzian k-NN classification for signal generation"""
        n = len(features)
        signals = np.zeros(n)
        
        for i in range(lookback, n - 1):  # Predict next bar
            # Get current features
            current_features = features[i]
            
            # Calculate distances to historical points
            distances = np.zeros(min(lookback, i))
            outcomes = np.zeros(min(lookback, i))
            
            for j in range(min(lookback, i)):
                hist_idx = i - j - 1
                if hist_idx >= 0:
                    # Calculate Lorentzian distance
                    dist = 0.0
                    for k in range(len(current_features)):
                        if not np.isnan(current_features[k]) and not np.isnan(features[hist_idx][k]):
                            dist += LorentzianIndicators.lorentzian_distance(
                                current_features[k], features[hist_idx][k]
                            )
                    distances[j] = dist
                    
                    # Outcome: 1 if price went up, -1 if down
                    if hist_idx + 1 < len(close):
                        outcomes[j] = 1.0 if close[hist_idx + 1] > close[hist_idx] else -1.0
            
            # Get k nearest neighbors
            if len(distances) >= neighbors:
                # Sort by distance and get nearest neighbors
                sorted_indices = np.argsort(distances)[:neighbors]
                neighbor_outcomes = outcomes[sorted_indices]
                
                # Weighted voting (closer neighbors have more weight)
                weights = 1.0 / (distances[sorted_indices] + 1e-10)
                weighted_sum = np.sum(weights * neighbor_outcomes)
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    signals[i + 1] = weighted_sum / total_weight
        
        return signals

class VectorBTFramework:
    """Production VectorBT backtesting framework"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data = None
        self.features = None
        self.signals = None
        self.portfolio = None
        self.results = {}
        
        # Initialize VectorBT settings
        vbt.settings.set_theme('dark')
        vbt.settings.parallel['caching'] = True
        vbt.settings.parallel['chunk_len'] = 1000
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare market data"""
        logger.info(f"Loading data from {self.config.data_path}")
        
        # Load data
        df = pd.read_csv(self.config.data_path)
        
        # Parse timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('Timestamp', inplace=True)
        
        # Ensure proper column names
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Handle date filtering
        if self.config.start_date:
            start_date = pd.to_datetime(self.config.start_date)
            df = df[df.index >= start_date]
        
        if self.config.end_date:
            end_date = pd.to_datetime(self.config.end_date)
            df = df[df.index <= end_date]
        
        # Data quality checks
        df = df.dropna()
        df = df[df['Volume'] > 0]  # Remove zero volume bars
        
        # Calculate additional technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Simple moving averages
        df[f'SMA_{self.config.fast_period}'] = df['Close'].rolling(self.config.fast_period).mean()
        df[f'SMA_{self.config.slow_period}'] = df['Close'].rolling(self.config.slow_period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        self.data = df
        logger.info(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def generate_lorentzian_features(self) -> np.ndarray:
        """Generate Lorentzian classification features"""
        logger.info("Generating Lorentzian features...")
        
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Extract price and volume data
        close = self.data['Close'].values
        high = self.data['High'].values
        low = self.data['Low'].values
        volume = self.data['Volume'].values
        
        # Generate features using Lorentzian indicators
        self.features = LorentzianIndicators.feature_engineering(
            close, high, low, volume, period=20
        )
        
        logger.info(f"Generated {self.features.shape[1]} features for {self.features.shape[0]} data points")
        return self.features
    
    def generate_signals(self) -> pd.Series:
        """Generate trading signals using Lorentzian classification"""
        logger.info("Generating trading signals...")
        
        if self.features is None:
            self.generate_lorentzian_features()
        
        # Generate raw signals using Lorentzian k-NN
        raw_signals = LorentzianIndicators.lorentzian_classification(
            self.features, self.data['Close'].values, lookback=100, neighbors=8
        )
        
        # Convert to pandas series
        signal_series = pd.Series(raw_signals, index=self.data.index)
        
        # Apply signal filtering and regime detection
        signal_series = self._filter_signals(signal_series)
        
        # Convert to entry/exit signals
        entries = (signal_series > 0.3) & (signal_series.shift(1) <= 0.3)
        exits = (signal_series < -0.3) | (signal_series < 0.1)
        
        # Create final signal series: 1 for long, -1 for short, 0 for hold
        final_signals = pd.Series(0, index=self.data.index)
        final_signals[entries] = 1
        final_signals[exits] = 0
        
        # Forward fill positions
        final_signals = final_signals.replace(0, np.nan).ffill().fillna(0)
        
        self.signals = final_signals
        logger.info(f"Generated {(final_signals != 0).sum()} position signals")
        
        return final_signals
    
    def _filter_signals(self, signals: pd.Series) -> pd.Series:
        """Apply signal filtering based on market regime and volatility"""
        
        # Volatility filter
        volatility = self.data['Close'].rolling(20).std() / self.data['Close'].rolling(20).mean()
        vol_threshold = volatility.quantile(0.8)  # Top 20% volatility
        
        # Trend filter using moving averages
        fast_ma = self.data[f'SMA_{self.config.fast_period}']
        slow_ma = self.data[f'SMA_{self.config.slow_period}']
        trend = (fast_ma > slow_ma).astype(int) - (fast_ma < slow_ma).astype(int)
        
        # RSI filter for overbought/oversold
        rsi_filter = (self.data['RSI'] > 30) & (self.data['RSI'] < 70)
        
        # Apply filters
        filtered_signals = signals.copy()
        
        # Reduce signal strength in high volatility
        filtered_signals[volatility > vol_threshold] *= 0.5
        
        # Align signals with trend
        filtered_signals = filtered_signals * trend * 0.7 + filtered_signals * 0.3
        
        # Apply RSI filter
        filtered_signals[~rsi_filter] *= 0.8
        
        return filtered_signals
    
    def run_backtest(self) -> vbt.Portfolio:
        """Run comprehensive backtest with VectorBT"""
        logger.info("Running VectorBT backtest...")
        
        if self.signals is None:
            self.generate_signals()
        
        # Prepare data for VectorBT
        close = self.data['Close']
        
        # Create entries and exits
        entries = self.signals == 1
        exits = self.signals == 0
        
        # Calculate position sizes with risk management
        position_sizes = self._calculate_position_sizes()
        
        # Run portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=position_sizes,
            init_cash=self.config.initial_cash,
            fees=self.config.commission,
            slippage=self.config.slippage,
            freq='30T'  # 30-minute frequency
        )
        
        self.portfolio = portfolio
        logger.info("Backtest completed successfully")
        
        return portfolio
    
    def _calculate_position_sizes(self) -> pd.Series:
        """Calculate position sizes with risk management"""
        
        # Base position size as percentage of portfolio
        base_size = self.config.max_position_size
        
        # Adjust size based on volatility
        volatility = self.data['Close'].rolling(20).std() / self.data['Close'].rolling(20).mean()
        vol_adjustment = 1 / (1 + volatility * 10)  # Reduce size in high volatility
        
        # ATR-based position sizing
        atr = self._calculate_atr(period=14)
        atr_adjustment = 1 / (1 + atr / self.data['Close'] * 100)
        
        # Combine adjustments
        position_sizes = base_size * vol_adjustment * atr_adjustment
        position_sizes = position_sizes.clip(0.01, self.config.max_position_size)
        
        return position_sizes
    
    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = self.data['High']
        low = self.data['Low'] 
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def optimize_parameters(self, param_ranges: Dict[str, List]) -> Dict:
        """Multi-parameter optimization using VectorBT"""
        logger.info("Starting parameter optimization...")
        
        best_params = {}
        best_sharpe = -np.inf
        optimization_results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for i, params in enumerate(param_combinations):
            try:
                # Update config with new parameters
                for key, value in params.items():
                    setattr(self.config, key, value)
                
                # Re-generate signals and run backtest
                self.generate_signals()
                portfolio = self.run_backtest()
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(portfolio)
                
                # Store results
                result = {
                    'params': params.copy(),
                    'metrics': metrics
                }
                optimization_results.append(result)
                
                # Check if this is the best result
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = params.copy()
                
                if i % 10 == 0:
                    logger.info(f"Optimization progress: {i}/{len(param_combinations)}")
                    
            except Exception as e:
                logger.warning(f"Optimization failed for params {params}: {e}")
                continue
        
        logger.info(f"Optimization completed. Best Sharpe ratio: {best_sharpe:.3f}")
        
        # Reset to best parameters
        for key, value in best_params.items():
            setattr(self.config, key, value)
        
        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'all_results': optimization_results
        }
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations for optimization"""
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def walk_forward_analysis(self, window_months: int = 12, step_months: int = 3) -> Dict:
        """Perform walk-forward analysis"""
        logger.info("Starting walk-forward analysis...")
        
        if self.data is None:
            self.load_data()
        
        # Calculate window parameters
        window_days = window_months * 30
        step_days = step_months * 30
        
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        
        results = []
        current_start = start_date
        
        while current_start + timedelta(days=window_days) <= end_date:
            window_end = current_start + timedelta(days=window_days)
            test_start = window_end
            test_end = min(test_start + timedelta(days=step_days), end_date)
            
            try:
                # Filter data for current window
                train_data = self.data[current_start:window_end]
                test_data = self.data[test_start:test_end]
                
                if len(train_data) < 100 or len(test_data) < 10:
                    current_start += timedelta(days=step_days)
                    continue
                
                # Train on window data
                self.data = train_data
                self.generate_signals()
                
                # Test on out-of-sample data  
                self.data = test_data
                portfolio = self.run_backtest()
                
                # Calculate metrics for this period
                metrics = self.calculate_performance_metrics(portfolio)
                
                result = {
                    'train_start': current_start,
                    'train_end': window_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'metrics': metrics
                }
                results.append(result)
                
                logger.info(f"Walk-forward period: {test_start.date()} to {test_end.date()}, "
                          f"Sharpe: {metrics['sharpe_ratio']:.3f}")
                
            except Exception as e:
                logger.warning(f"Walk-forward failed for period {current_start}: {e}")
            
            current_start += timedelta(days=step_days)
        
        # Restore full dataset
        self.load_data()
        
        return {
            'periods': results,
            'avg_sharpe': np.mean([r['metrics']['sharpe_ratio'] for r in results]),
            'consistency': np.std([r['metrics']['sharpe_ratio'] for r in results])
        }
    
    def calculate_performance_metrics(self, portfolio: vbt.Portfolio = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if portfolio is None:
            portfolio = self.portfolio
        
        if portfolio is None:
            raise ValueError("No portfolio available for metrics calculation")
        
        # Basic metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        calmar_ratio = portfolio.calmar_ratio()
        
        # Trade-level metrics
        trades = portfolio.trades.records_readable
        
        if len(trades) > 0:
            win_rate = (trades['pnl'] > 0).mean()
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if (trades['pnl'] < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            
            # Trade duration
            avg_trade_duration = trades['duration'].mean()
            
            # Maximum consecutive losses
            consecutive_losses = self._calculate_consecutive_losses(trades['pnl'])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = pd.Timedelta(0)
            consecutive_losses = 0
        
        # Portfolio value metrics
        portfolio_value = portfolio.value()
        
        # Calculate CAGR
        years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
        cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Sortino ratio (using downside deviation)
        returns = portfolio_value.pct_change().dropna()
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252 * 48)  # Annualized for 30min data
        sortino_ratio = (returns.mean() * 252 * 48) / downside_deviation if downside_deviation > 0 else np.inf
        
        # VaR and CVaR (5% confidence level)
        var_5 = returns.quantile(0.05)
        cvar_5 = returns[returns <= var_5].mean()
        
        # Market correlation (if benchmark available)
        market_correlation = 0  # Placeholder - would need market data
        
        # Information ratio
        information_ratio = 0  # Placeholder - would need benchmark returns
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(portfolio_value)
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252 * 48),  # Annualized
            'ulcer_index': ulcer_index,
            'var_5': var_5,
            'cvar_5': cvar_5,
            
            # Trade metrics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'avg_trade_duration': avg_trade_duration,
            'max_consecutive_losses': consecutive_losses,
            
            # Market metrics
            'market_correlation': market_correlation,
            
            # Portfolio metrics
            'final_value': portfolio_value.iloc[-1],
            'exposure_time': portfolio.positions.count()[0] / len(portfolio_value),
            
            # Target achievement
            'target_sharpe_achieved': sharpe_ratio >= self.config.target_sharpe,
            'target_drawdown_achieved': max_drawdown <= self.config.target_max_dd,
            'target_win_rate_achieved': win_rate >= self.config.target_win_rate,
            'target_profit_factor_achieved': profit_factor >= self.config.target_profit_factor
        }
        
        return metrics
    
    def _calculate_consecutive_losses(self, pnl_series: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_series:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_ulcer_index(self, portfolio_value: pd.Series) -> float:
        """Calculate Ulcer Index (a measure of downside risk)"""
        running_max = portfolio_value.expanding().max()
        drawdown_pct = (portfolio_value - running_max) / running_max * 100
        ulcer_index = np.sqrt((drawdown_pct ** 2).mean())
        return ulcer_index
    
    def generate_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")
        
        if self.portfolio is None:
            raise ValueError("No portfolio available. Run backtest first.")
        
        # Calculate all metrics
        metrics = self.calculate_performance_metrics()
        
        # Generate report
        report = {
            'summary': {
                'strategy': 'Lorentzian Classification MARL',
                'period': f"{self.data.index[0].date()} to {self.data.index[-1].date()}",
                'total_bars': len(self.data),
                'backtest_date': datetime.now().isoformat()
            },
            'performance': metrics,
            'configuration': {
                'initial_cash': self.config.initial_cash,
                'commission': self.config.commission,
                'slippage': self.config.slippage,
                'max_position_size': self.config.max_position_size,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct
            },
            'target_achievement': {
                'sharpe_ratio': {
                    'target': self.config.target_sharpe,
                    'achieved': metrics['sharpe_ratio'],
                    'passed': metrics['target_sharpe_achieved']
                },
                'max_drawdown': {
                    'target': self.config.target_max_dd,
                    'achieved': metrics['max_drawdown'],
                    'passed': metrics['target_drawdown_achieved']
                },
                'win_rate': {
                    'target': self.config.target_win_rate,
                    'achieved': metrics['win_rate'],
                    'passed': metrics['target_win_rate_achieved']
                },
                'profit_factor': {
                    'target': self.config.target_profit_factor,
                    'achieved': metrics['profit_factor'],
                    'passed': metrics['target_profit_factor_achieved']
                }
            }
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot comprehensive backtest results"""
        if self.portfolio is None:
            raise ValueError("No portfolio available. Run backtest first.")
        
        # Create VectorBT plots
        fig = self.portfolio.plot(
            subplots=[
                'orders', 'trade_pnl', 'cum_returns', 'drawdowns'
            ],
            figsize=figsize
        )
        
        return fig

def create_vectorbt_framework(config: BacktestConfig = None) -> VectorBTFramework:
    """Factory function to create VectorBT framework"""
    if config is None:
        config = BacktestConfig()
    
    return VectorBTFramework(config)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = BacktestConfig(
        data_path="/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv",
        initial_cash=100000,
        commission=0.0001,
        target_sharpe=2.0
    )
    
    # Create framework
    framework = create_vectorbt_framework(config)
    
    # Load data and run backtest
    framework.load_data()
    framework.generate_signals()
    portfolio = framework.run_backtest()
    
    # Generate report
    report = framework.generate_report()
    
    print("VectorBT Framework initialized successfully!")
    print(f"Sharpe Ratio: {report['performance']['sharpe_ratio']:.3f}")
    print(f"Total Return: {report['performance']['total_return']:.2%}")
    print(f"Max Drawdown: {report['performance']['max_drawdown']:.2%}")