"""
Comprehensive Performance Metrics System
=======================================

Advanced performance analytics for MARL trading systems with:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis and recovery metrics
- Trade-level analytics and pattern recognition
- Market regime performance analysis
- Stress testing and scenario analysis

Author: Claude Code
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import scipy.stats as stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTargets:
    """Target performance metrics for validation"""
    target_sharpe: float = 2.0
    target_max_drawdown: float = 0.15  # 15%
    target_win_rate: float = 0.60      # 60%
    target_profit_factor: float = 1.5
    target_calmar: float = 1.0
    target_sortino: float = 2.5
    target_cagr: float = 0.20          # 20%
    max_var_5: float = -0.03           # -3% daily VaR
    min_sharpe_consistency: float = 0.7 # Minimum rolling Sharpe consistency

class PerformanceAnalyzer:
    """Comprehensive performance analysis system"""
    
    def __init__(self, portfolio_values: pd.Series, trades: pd.DataFrame = None,
                 benchmark: pd.Series = None, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Args:
            portfolio_values: Time series of portfolio values
            trades: DataFrame with trade records (optional)
            benchmark: Benchmark return series (optional)
            risk_free_rate: Annual risk-free rate
        """
        self.portfolio_values = portfolio_values
        self.trades = trades
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        
        # Calculate returns
        self.returns = portfolio_values.pct_change().dropna()
        self.log_returns = np.log(portfolio_values / portfolio_values.shift(1)).dropna()
        
        # Annualization factor (assuming 30-minute data)
        self.annual_factor = 252 * 48  # 252 trading days * 48 30-min periods
        
        logger.info(f"Initialized performance analyzer with {len(portfolio_values)} observations")
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        # Return metrics
        total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        
        # CAGR calculation
        years = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days / 365.25
        cagr = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = self.returns.std() * np.sqrt(self.annual_factor)
        
        # Sharpe ratio
        excess_returns = self.returns.mean() * self.annual_factor - self.risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        # Downside metrics
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.annual_factor)
        
        # Sortino ratio
        excess_returns = self.returns.mean() * self.annual_factor - self.risk_free_rate
        sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else np.inf
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_1 = self.returns.quantile(0.01)
        var_5 = self.returns.quantile(0.05)
        var_10 = self.returns.quantile(0.10)
        
        cvar_1 = self.returns[self.returns <= var_1].mean()
        cvar_5 = self.returns[self.returns <= var_5].mean()
        cvar_10 = self.returns[self.returns <= var_10].mean()
        
        # Maximum consecutive losses
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index()
        
        # Pain Index (average drawdown)
        running_max = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - running_max) / running_max
        pain_index = abs(drawdowns.mean())
        
        # Recovery factor
        recovery_factor = abs(self.calculate_basic_metrics()['total_return'] / 
                            self.calculate_basic_metrics()['max_drawdown'])
        
        # Tail ratio
        tail_ratio = abs(self.returns.quantile(0.95) / self.returns.quantile(0.05))
        
        return {
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'var_1': var_1,
            'var_5': var_5,
            'var_10': var_10,
            'cvar_1': cvar_1,
            'cvar_5': cvar_5,
            'cvar_10': cvar_10,
            'max_consecutive_losses': max_consecutive_losses,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'recovery_factor': recovery_factor,
            'tail_ratio': tail_ratio
        }
    
    def calculate_trade_metrics(self) -> Dict[str, Union[float, int]]:
        """Calculate trade-level performance metrics"""
        
        if self.trades is None or len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_trade_duration': pd.Timedelta(0),
                'avg_time_between_trades': pd.Timedelta(0)
            }
        
        # Ensure trades have PnL column
        if 'pnl' not in self.trades.columns:
            logger.warning("No 'pnl' column found in trades data")
            return {}
        
        pnl = self.trades['pnl']
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0
        
        # Trade duration analysis
        if 'duration' in self.trades.columns:
            avg_trade_duration = self.trades['duration'].mean()
        else:
            avg_trade_duration = pd.Timedelta(0)
        
        # Time between trades
        if 'exit_time' in self.trades.columns and len(self.trades) > 1:
            exit_times = pd.to_datetime(self.trades['exit_time']).sort_values()
            time_between_trades = exit_times.diff()[1:]
            avg_time_between_trades = time_between_trades.mean()
        else:
            avg_time_between_trades = pd.Timedelta(0)
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive_wins()
        consecutive_losses = self._calculate_max_consecutive_losses_trades()
        
        # Win/loss streaks
        win_loss_ratio = len(winning_trades) / len(losing_trades) if len(losing_trades) > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration': avg_trade_duration,
            'avg_time_between_trades': avg_time_between_trades,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy
        }
    
    def calculate_benchmark_metrics(self) -> Dict[str, float]:
        """Calculate benchmark-relative metrics"""
        
        if self.benchmark is None:
            return {
                'beta': np.nan,
                'alpha': np.nan,
                'correlation': np.nan,
                'tracking_error': np.nan,
                'information_ratio': np.nan,
                'treynor_ratio': np.nan
            }
        
        # Align benchmark with portfolio returns
        aligned_benchmark = self.benchmark.reindex(self.returns.index, method='nearest')
        benchmark_returns = aligned_benchmark.pct_change().dropna()
        
        # Ensure same length
        min_length = min(len(self.returns), len(benchmark_returns))
        portfolio_rets = self.returns.iloc[-min_length:]
        benchmark_rets = benchmark_returns.iloc[-min_length:]
        
        # Beta calculation
        covariance = np.cov(portfolio_rets, benchmark_rets)[0][1]
        benchmark_variance = np.var(benchmark_rets)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Alpha calculation (CAPM)
        portfolio_return = portfolio_rets.mean() * self.annual_factor
        benchmark_return = benchmark_rets.mean() * self.annual_factor
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation
        correlation = portfolio_rets.corr(benchmark_rets)
        
        # Tracking error
        excess_returns = portfolio_rets - benchmark_rets
        tracking_error = excess_returns.std() * np.sqrt(self.annual_factor)
        
        # Information ratio
        information_ratio = excess_returns.mean() * self.annual_factor / tracking_error if tracking_error != 0 else 0
        
        # Treynor ratio
        treynor_ratio = (portfolio_return - self.risk_free_rate) / beta if beta != 0 else 0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio
        }
    
    def rolling_metrics_analysis(self, window_days: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        # Convert window to number of periods
        periods_per_day = self.annual_factor / 252
        window_periods = int(window_days * periods_per_day)
        
        rolling_metrics = pd.DataFrame(index=self.returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = self.returns.rolling(window_periods).mean() * self.annual_factor
        rolling_metrics['rolling_volatility'] = self.returns.rolling(window_periods).std() * np.sqrt(self.annual_factor)
        
        # Rolling Sharpe ratio
        excess_rolling = rolling_metrics['rolling_return'] - self.risk_free_rate
        rolling_metrics['rolling_sharpe'] = excess_rolling / rolling_metrics['rolling_volatility']
        
        # Rolling maximum drawdown
        rolling_max = self.portfolio_values.rolling(window_periods).max()
        rolling_dd = (self.portfolio_values - rolling_max) / rolling_max
        rolling_metrics['rolling_max_dd'] = rolling_dd.rolling(window_periods).min()
        
        # Rolling win rate (if trades available)
        if self.trades is not None and len(self.trades) > 0:
            # This is approximate - would need more detailed trade timing
            rolling_metrics['rolling_win_rate'] = np.nan  # Placeholder
        
        return rolling_metrics.dropna()
    
    def regime_analysis(self, volatility_threshold: float = 0.02) -> Dict[str, Dict]:
        """Analyze performance across different market regimes"""
        
        # Calculate volatility regime
        rolling_vol = self.returns.rolling(20).std()
        high_vol_periods = rolling_vol > volatility_threshold
        low_vol_periods = rolling_vol <= volatility_threshold
        
        # Calculate trend regime
        price_ma_short = self.portfolio_values.rolling(10).mean()
        price_ma_long = self.portfolio_values.rolling(30).mean()
        uptrend_periods = price_ma_short > price_ma_long
        downtrend_periods = price_ma_short <= price_ma_long
        
        regimes = {
            'high_volatility': {
                'periods': high_vol_periods.sum(),
                'returns': self.returns[high_vol_periods].mean() * self.annual_factor,
                'volatility': self.returns[high_vol_periods].std() * np.sqrt(self.annual_factor),
                'sharpe': (self.returns[high_vol_periods].mean() * self.annual_factor - self.risk_free_rate) / 
                         (self.returns[high_vol_periods].std() * np.sqrt(self.annual_factor))
            },
            'low_volatility': {
                'periods': low_vol_periods.sum(),
                'returns': self.returns[low_vol_periods].mean() * self.annual_factor,
                'volatility': self.returns[low_vol_periods].std() * np.sqrt(self.annual_factor),
                'sharpe': (self.returns[low_vol_periods].mean() * self.annual_factor - self.risk_free_rate) / 
                         (self.returns[low_vol_periods].std() * np.sqrt(self.annual_factor))
            },
            'uptrend': {
                'periods': uptrend_periods.sum(),
                'returns': self.returns[uptrend_periods].mean() * self.annual_factor,
                'volatility': self.returns[uptrend_periods].std() * np.sqrt(self.annual_factor),
                'sharpe': (self.returns[uptrend_periods].mean() * self.annual_factor - self.risk_free_rate) / 
                         (self.returns[uptrend_periods].std() * np.sqrt(self.annual_factor))
            },
            'downtrend': {
                'periods': downtrend_periods.sum(),
                'returns': self.returns[downtrend_periods].mean() * self.annual_factor,
                'volatility': self.returns[downtrend_periods].std() * np.sqrt(self.annual_factor),
                'sharpe': (self.returns[downtrend_periods].mean() * self.annual_factor - self.risk_free_rate) / 
                         (self.returns[downtrend_periods].std() * np.sqrt(self.annual_factor))
            }
        }
        
        return regimes
    
    def stress_testing(self) -> Dict[str, Dict]:
        """Perform stress testing scenarios"""
        
        scenarios = {}
        
        # Market crash scenario (-20% in one day)
        crash_returns = self.returns.copy()
        crash_returns.iloc[-1] = -0.20
        crash_portfolio = (1 + crash_returns).cumprod() * self.portfolio_values.iloc[0]
        
        scenarios['market_crash'] = {
            'final_value': crash_portfolio.iloc[-1],
            'total_return': (crash_portfolio.iloc[-1] / crash_portfolio.iloc[0]) - 1,
            'max_drawdown': ((crash_portfolio - crash_portfolio.expanding().max()) / 
                           crash_portfolio.expanding().max()).min()
        }
        
        # High volatility scenario (double current volatility)
        vol_multiplier = 2.0
        high_vol_returns = self.returns * vol_multiplier
        high_vol_portfolio = (1 + high_vol_returns).cumprod() * self.portfolio_values.iloc[0]
        
        scenarios['high_volatility'] = {
            'final_value': high_vol_portfolio.iloc[-1],
            'total_return': (high_vol_portfolio.iloc[-1] / high_vol_portfolio.iloc[0]) - 1,
            'volatility': high_vol_returns.std() * np.sqrt(self.annual_factor),
            'max_drawdown': ((high_vol_portfolio - high_vol_portfolio.expanding().max()) / 
                           high_vol_portfolio.expanding().max()).min()
        }
        
        # Extended drawdown scenario
        worst_month = self.returns.groupby(self.returns.index.to_period('M')).sum().min()
        extended_dd_returns = self.returns.copy()
        # Simulate 3 consecutive months of worst performance
        extended_dd_returns.iloc[-90:] = worst_month / 30  # Distribute over 90 periods
        extended_dd_portfolio = (1 + extended_dd_returns).cumprod() * self.portfolio_values.iloc[0]
        
        scenarios['extended_drawdown'] = {
            'final_value': extended_dd_portfolio.iloc[-1],
            'total_return': (extended_dd_portfolio.iloc[-1] / extended_dd_portfolio.iloc[0]) - 1,
            'max_drawdown': ((extended_dd_portfolio - extended_dd_portfolio.expanding().max()) / 
                           extended_dd_portfolio.expanding().max()).min()
        }
        
        return scenarios
    
    def performance_attribution(self) -> Dict[str, float]:
        """Analyze performance attribution by time periods"""
        
        attribution = {}
        
        # Monthly attribution
        monthly_returns = self.returns.groupby(self.returns.index.to_period('M')).sum()
        attribution['best_month'] = monthly_returns.max()
        attribution['worst_month'] = monthly_returns.min()
        attribution['avg_monthly_return'] = monthly_returns.mean()
        attribution['monthly_volatility'] = monthly_returns.std()
        
        # Quarterly attribution
        quarterly_returns = self.returns.groupby(self.returns.index.to_period('Q')).sum()
        attribution['best_quarter'] = quarterly_returns.max()
        attribution['worst_quarter'] = quarterly_returns.min()
        attribution['avg_quarterly_return'] = quarterly_returns.mean()
        
        # Yearly attribution
        yearly_returns = self.returns.groupby(self.returns.index.to_period('Y')).sum()
        attribution['best_year'] = yearly_returns.max()
        attribution['worst_year'] = yearly_returns.min()
        attribution['avg_yearly_return'] = yearly_returns.mean()
        
        # Day of week attribution
        dow_returns = self.returns.groupby(self.returns.index.dayofweek).mean()
        attribution['dow_performance'] = dow_returns.to_dict()
        
        # Hour of day attribution (for intraday data)
        hour_returns = self.returns.groupby(self.returns.index.hour).mean()
        attribution['hour_performance'] = hour_returns.to_dict()
        
        return attribution
    
    def monte_carlo_analysis(self, num_simulations: int = 1000, periods: int = 252) -> Dict:
        """Monte Carlo simulation for future performance"""
        
        # Calculate historical statistics
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        # Run simulations
        simulated_paths = np.zeros((num_simulations, periods))
        final_values = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, periods)
            
            # Calculate cumulative performance
            cumulative_returns = (1 + random_returns).cumprod()
            simulated_paths[i] = cumulative_returns
            final_values[i] = cumulative_returns[-1]
        
        # Calculate confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        confidence_intervals = np.percentile(final_values, percentiles)
        
        # Calculate probability of positive returns
        prob_positive = (final_values > 1.0).mean()
        
        return {
            'simulated_paths': simulated_paths,
            'final_values': final_values,
            'confidence_intervals': dict(zip(percentiles, confidence_intervals)),
            'probability_positive': prob_positive,
            'expected_return': np.mean(final_values) - 1,
            'downside_risk': (final_values < 0.9).mean()  # Probability of >10% loss
        }
    
    def target_achievement_analysis(self, targets: PerformanceTargets) -> Dict[str, Dict]:
        """Analyze achievement of performance targets"""
        
        # Calculate current metrics
        basic_metrics = self.calculate_basic_metrics()
        risk_metrics = self.calculate_risk_metrics()
        trade_metrics = self.calculate_trade_metrics()
        
        achievement = {}
        
        # Sharpe ratio target
        achievement['sharpe_ratio'] = {
            'target': targets.target_sharpe,
            'achieved': basic_metrics['sharpe_ratio'],
            'passed': basic_metrics['sharpe_ratio'] >= targets.target_sharpe,
            'score': min(basic_metrics['sharpe_ratio'] / targets.target_sharpe, 2.0)  # Cap at 2x
        }
        
        # Maximum drawdown target
        achievement['max_drawdown'] = {
            'target': targets.target_max_drawdown,
            'achieved': abs(basic_metrics['max_drawdown']),
            'passed': abs(basic_metrics['max_drawdown']) <= targets.target_max_drawdown,
            'score': max(0, 2 - abs(basic_metrics['max_drawdown']) / targets.target_max_drawdown)
        }
        
        # Win rate target
        achievement['win_rate'] = {
            'target': targets.target_win_rate,
            'achieved': trade_metrics.get('win_rate', 0),
            'passed': trade_metrics.get('win_rate', 0) >= targets.target_win_rate,
            'score': trade_metrics.get('win_rate', 0) / targets.target_win_rate
        }
        
        # Profit factor target
        achievement['profit_factor'] = {
            'target': targets.target_profit_factor,
            'achieved': trade_metrics.get('profit_factor', 0),
            'passed': trade_metrics.get('profit_factor', 0) >= targets.target_profit_factor,
            'score': min(trade_metrics.get('profit_factor', 0) / targets.target_profit_factor, 2.0)
        }
        
        # CAGR target
        achievement['cagr'] = {
            'target': targets.target_cagr,
            'achieved': basic_metrics['cagr'],
            'passed': basic_metrics['cagr'] >= targets.target_cagr,
            'score': basic_metrics['cagr'] / targets.target_cagr
        }
        
        # Sortino ratio target
        achievement['sortino_ratio'] = {
            'target': targets.target_sortino,
            'achieved': risk_metrics['sortino_ratio'],
            'passed': risk_metrics['sortino_ratio'] >= targets.target_sortino,
            'score': min(risk_metrics['sortino_ratio'] / targets.target_sortino, 2.0)
        }
        
        # Overall score (average of all scores)
        scores = [metric['score'] for metric in achievement.values() if not np.isnan(metric['score'])]
        overall_score = np.mean(scores) if scores else 0
        
        achievement['overall'] = {
            'score': overall_score,
            'grade': self._calculate_grade(overall_score),
            'targets_met': sum(1 for metric in achievement.values() if metric.get('passed', False)),
            'total_targets': len(achievement)
        }
        
        return achievement
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on score"""
        if score >= 1.5:
            return 'A+'
        elif score >= 1.2:
            return 'A'
        elif score >= 1.0:
            return 'A-'
        elif score >= 0.9:
            return 'B+'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'B-'
        elif score >= 0.6:
            return 'C+'
        elif score >= 0.5:
            return 'C'
        else:
            return 'F'
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive negative returns"""
        consecutive = 0
        max_consecutive = 0
        
        for ret in self.returns:
            if ret < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for pnl in self.trades['pnl']:
            if pnl > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses_trades(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for pnl in self.trades['pnl']:
            if pnl < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index (measure of downside risk)"""
        running_max = self.portfolio_values.expanding().max()
        drawdown_pct = (self.portfolio_values - running_max) / running_max * 100
        ulcer_index = np.sqrt((drawdown_pct ** 2).mean())
        return ulcer_index
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        logger.info("Generating comprehensive performance report...")
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics()
        risk_metrics = self.calculate_risk_metrics()
        trade_metrics = self.calculate_trade_metrics()
        benchmark_metrics = self.calculate_benchmark_metrics()
        regime_analysis = self.regime_analysis()
        stress_testing = self.stress_testing()
        attribution = self.performance_attribution()
        
        # Target achievement
        targets = PerformanceTargets()
        achievement = self.target_achievement_analysis(targets)
        
        # Compile comprehensive report
        report = {
            'summary': {
                'analysis_date': datetime.now().isoformat(),
                'period_start': self.portfolio_values.index[0].isoformat(),
                'period_end': self.portfolio_values.index[-1].isoformat(),
                'total_observations': len(self.portfolio_values),
                'strategy_name': 'Lorentzian Classification MARL'
            },
            'performance': {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'trade_metrics': trade_metrics,
                'benchmark_metrics': benchmark_metrics
            },
            'analysis': {
                'regime_analysis': regime_analysis,
                'stress_testing': stress_testing,
                'performance_attribution': attribution
            },
            'target_achievement': achievement,
            'recommendations': self._generate_recommendations(achievement)
        }
        
        return report
    
    def _generate_recommendations(self, achievement: Dict) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Sharpe ratio recommendations
        if not achievement['sharpe_ratio']['passed']:
            if achievement['sharpe_ratio']['achieved'] < 1.0:
                recommendations.append("Critical: Sharpe ratio below 1.0. Consider reducing position sizes and improving signal quality.")
            else:
                recommendations.append("Sharpe ratio below target. Focus on reducing volatility through better risk management.")
        
        # Drawdown recommendations
        if not achievement['max_drawdown']['passed']:
            recommendations.append("Maximum drawdown exceeds target. Implement stricter stop-losses and position sizing.")
        
        # Win rate recommendations
        if not achievement['win_rate']['passed']:
            recommendations.append("Win rate below target. Improve entry signal precision and reduce false signals.")
        
        # Profit factor recommendations
        if not achievement['profit_factor']['passed']:
            if achievement['profit_factor']['achieved'] < 1.0:
                recommendations.append("Critical: Profit factor below 1.0. Strategy is unprofitable.")
            else:
                recommendations.append("Profit factor below target. Focus on letting winners run and cutting losses quickly.")
        
        # Overall performance
        overall_score = achievement['overall']['score']
        if overall_score < 0.8:
            recommendations.append("Overall performance needs significant improvement. Consider strategy redesign.")
        elif overall_score < 1.0:
            recommendations.append("Performance is acceptable but has room for improvement.")
        else:
            recommendations.append("Excellent performance! Consider scaling up strategy allocation.")
        
        return recommendations
    
    def create_performance_dashboard(self, save_path: str = None) -> go.Figure:
        """Create interactive performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Portfolio Value', 'Rolling Sharpe Ratio',
                'Drawdown', 'Monthly Returns',
                'Return Distribution', 'Risk-Return Scatter'
            ],
            specs=[[{}, {}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # Portfolio value chart
        fig.add_trace(
            go.Scatter(x=self.portfolio_values.index, y=self.portfolio_values.values,
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Rolling Sharpe ratio
        rolling_metrics = self.rolling_metrics_analysis()
        if not rolling_metrics.empty:
            fig.add_trace(
                go.Scatter(x=rolling_metrics.index, y=rolling_metrics['rolling_sharpe'],
                          name='Rolling Sharpe', line=dict(color='green')),
                row=1, col=2
            )
        
        # Drawdown chart
        running_max = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns.values,
                      name='Drawdown %', fill='tonexty', line=dict(color='red')),
            row=2, col=1
        )
        
        # Monthly returns heatmap
        monthly_returns = self.returns.groupby([self.returns.index.year, self.returns.index.month]).sum()
        fig.add_trace(
            go.Scatter(x=list(range(len(monthly_returns))), y=monthly_returns.values,
                      mode='markers', name='Monthly Returns',
                      marker=dict(color=monthly_returns.values, colorscale='RdYlGn')),
            row=2, col=2
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(x=self.returns.values, name='Return Distribution',
                        nbinsx=50, marker=dict(color='lightblue')),
            row=3, col=1
        )
        
        # Risk-return scatter (placeholder)
        fig.add_trace(
            go.Scatter(x=[self.returns.std() * np.sqrt(self.annual_factor)],
                      y=[self.returns.mean() * self.annual_factor],
                      mode='markers', name='Strategy',
                      marker=dict(size=10, color='red')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Performance Dashboard',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

def create_performance_analyzer(portfolio_values: pd.Series, trades: pd.DataFrame = None,
                              benchmark: pd.Series = None) -> PerformanceAnalyzer:
    """Factory function to create performance analyzer"""
    return PerformanceAnalyzer(portfolio_values, trades, benchmark)

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='30T')
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.01, len(dates))  # Sample returns
    portfolio_values = (1 + pd.Series(returns, index=dates)).cumprod() * 100000
    
    # Create analyzer
    analyzer = create_performance_analyzer(portfolio_values)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    print("Performance Analyzer initialized successfully!")
    print(f"Sharpe Ratio: {report['performance']['basic_metrics']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {report['performance']['basic_metrics']['max_drawdown']:.2%}")
    print(f"Overall Grade: {report['target_achievement']['overall']['grade']}")