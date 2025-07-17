"""
Professional Performance Analytics Module
========================================

Institutional-grade performance metrics and analysis including:
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown periods and recovery analysis
- Trade analysis (duration, win streaks, etc.)
- Monthly/yearly performance breakdowns
- Risk-adjusted return metrics
- Benchmark comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """Comprehensive performance analytics for institutional backtesting"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def analyze_returns(self, returns: pd.Series, prices: pd.Series = None, 
                       benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Comprehensive return analysis
        
        Args:
            returns: Strategy returns (daily)
            prices: Price series for additional calculations
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        # Ensure returns are properly formatted
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        # Calculate core metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        annualized_volatility = self._calculate_annualized_volatility(returns)
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # Drawdown analysis
        drawdown_metrics = self._analyze_drawdowns(returns)
        
        # Trade statistics
        trade_stats = self._analyze_trade_statistics(returns)
        
        # Time-based analysis
        time_analysis = self._analyze_time_periods(returns)
        
        # Benchmark comparison
        benchmark_comparison = {}
        if benchmark_returns is not None:
            benchmark_comparison = self._compare_to_benchmark(returns, benchmark_returns)
        
        # Additional risk metrics
        risk_metrics = self._calculate_additional_risk_metrics(returns)
        
        results = {
            'performance_summary': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            },
            'drawdown_analysis': drawdown_metrics,
            'trade_statistics': trade_stats,
            'time_analysis': time_analysis,
            'risk_metrics': risk_metrics,
            'benchmark_comparison': benchmark_comparison,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.results = results
        return results
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total cumulative return"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        days = len(returns)
        years = days / 252  # 252 trading days per year
        total_return = self._calculate_total_return(returns)
        return (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return (excess_returns.mean() * 252) / downside_deviation
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annualized_return = self._calculate_annualized_return(returns)
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0
        return annualized_return / abs(max_dd)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Any]:
        """Comprehensive drawdown analysis"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        # Maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        dd_periods = []
        current_period = None
        
        for date, is_dd in in_drawdown.items():
            if is_dd and current_period is None:
                current_period = {'start': date, 'min_dd': 0, 'min_date': date}
            elif not is_dd and current_period is not None:
                current_period['end'] = date
                current_period['duration'] = (date - current_period['start']).days
                dd_periods.append(current_period)
                current_period = None
            elif is_dd and current_period is not None:
                if drawdown[date] < current_period['min_dd']:
                    current_period['min_dd'] = drawdown[date]
                    current_period['min_date'] = date
        
        # Close any open drawdown period
        if current_period is not None:
            current_period['end'] = returns.index[-1]
            current_period['duration'] = (current_period['end'] - current_period['start']).days
            dd_periods.append(current_period)
        
        # Recovery analysis
        avg_recovery_time = np.mean([p['duration'] for p in dd_periods]) if dd_periods else 0
        max_recovery_time = max([p['duration'] for p in dd_periods]) if dd_periods else 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date.isoformat() if max_dd_date else None,
            'avg_drawdown_duration_days': avg_recovery_time,
            'max_drawdown_duration_days': max_recovery_time,
            'total_drawdown_periods': len(dd_periods),
            'drawdown_periods': [
                {
                    'start': p['start'].isoformat(),
                    'end': p['end'].isoformat(),
                    'duration_days': p['duration'],
                    'min_drawdown': p['min_dd'],
                    'min_date': p['min_date'].isoformat()
                } for p in dd_periods[:10]  # Limit to top 10 for brevity
            ]
        }
    
    def _analyze_trade_statistics(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze trade-level statistics"""
        # Separate winning and losing periods
        winning_periods = returns[returns > 0]
        losing_periods = returns[returns < 0]
        flat_periods = returns[returns == 0]
        
        # Win rate
        total_periods = len(returns)
        win_rate = len(winning_periods) / total_periods if total_periods > 0 else 0
        
        # Average win/loss
        avg_win = winning_periods.mean() if len(winning_periods) > 0 else 0
        avg_loss = losing_periods.mean() if len(losing_periods) > 0 else 0
        
        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_periods(returns > 0)
        consecutive_losses = self._calculate_consecutive_periods(returns < 0)
        
        # Profit factor
        total_wins = winning_periods.sum()
        total_losses = abs(losing_periods.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_periods': total_periods,
            'winning_periods': len(winning_periods),
            'losing_periods': len(losing_periods),
            'flat_periods': len(flat_periods),
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'max_consecutive_wins': consecutive_wins['max'],
            'max_consecutive_losses': consecutive_losses['max'],
            'avg_consecutive_wins': consecutive_wins['avg'],
            'avg_consecutive_losses': consecutive_losses['avg']
        }
    
    def _calculate_consecutive_periods(self, condition_series: pd.Series) -> Dict[str, float]:
        """Calculate consecutive period statistics"""
        consecutive_counts = []
        current_count = 0
        
        for value in condition_series:
            if value:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        # Don't forget the last streak
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        if not consecutive_counts:
            return {'max': 0, 'avg': 0, 'count': 0}
        
        return {
            'max': max(consecutive_counts),
            'avg': np.mean(consecutive_counts),
            'count': len(consecutive_counts)
        }
    
    def _analyze_time_periods(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance by time periods"""
        # Monthly analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Yearly analysis
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Best/worst periods
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        best_month_date = monthly_returns.idxmax()
        worst_month_date = monthly_returns.idxmin()
        
        # Monthly statistics
        positive_months = len(monthly_returns[monthly_returns > 0])
        total_months = len(monthly_returns)
        monthly_win_rate = positive_months / total_months if total_months > 0 else 0
        
        return {
            'monthly_analysis': {
                'total_months': total_months,
                'positive_months': positive_months,
                'monthly_win_rate': monthly_win_rate,
                'best_month': best_month,
                'worst_month': worst_month,
                'best_month_date': best_month_date.isoformat() if best_month_date else None,
                'worst_month_date': worst_month_date.isoformat() if worst_month_date else None,
                'avg_monthly_return': monthly_returns.mean(),
                'monthly_volatility': monthly_returns.std()
            },
            'yearly_analysis': {
                'yearly_returns': {
                    year.year: ret for year, ret in yearly_returns.items()
                },
                'best_year': yearly_returns.max() if len(yearly_returns) > 0 else 0,
                'worst_year': yearly_returns.min() if len(yearly_returns) > 0 else 0,
                'avg_yearly_return': yearly_returns.mean() if len(yearly_returns) > 0 else 0,
                'yearly_volatility': yearly_returns.std() if len(yearly_returns) > 0 else 0
            }
        }
    
    def _compare_to_benchmark(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Compare strategy to benchmark"""
        # Align the series
        aligned_data = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No overlapping data between strategy and benchmark'}
        
        strategy_ret = aligned_data['strategy']
        benchmark_ret = aligned_data['benchmark']
        
        # Calculate relative metrics
        excess_returns = strategy_ret - benchmark_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation
        covariance = np.cov(strategy_ret, benchmark_ret)[0, 1]
        benchmark_variance = np.var(benchmark_ret)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (CAPM)
        strategy_annual = strategy_ret.mean() * 252
        benchmark_annual = benchmark_ret.mean() * 252
        alpha = strategy_annual - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
        
        # Cumulative performance comparison
        strategy_cumulative = (1 + strategy_ret).cumprod()
        benchmark_cumulative = (1 + benchmark_ret).cumprod()
        
        strategy_total = strategy_cumulative.iloc[-1] - 1
        benchmark_total = benchmark_cumulative.iloc[-1] - 1
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'correlation': strategy_ret.corr(benchmark_ret),
            'excess_return': strategy_total - benchmark_total,
            'strategy_total_return': strategy_total,
            'benchmark_total_return': benchmark_total,
            'outperformance_days': len(excess_returns[excess_returns > 0]),
            'total_comparison_days': len(excess_returns)
        }
    
    def _calculate_additional_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate additional risk metrics"""
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio
        positive_tail = np.percentile(returns, 95)
        negative_tail = abs(np.percentile(returns, 5))
        tail_ratio = positive_tail / negative_tail if negative_tail > 0 else float('inf')
        
        # Maximum daily loss
        max_daily_loss = returns.min()
        max_daily_gain = returns.max()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'max_daily_loss': max_daily_loss,
            'max_daily_gain': max_daily_gain,
            'downside_deviation': returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive text performance report"""
        if not self.results:
            return "No analysis results available. Run analyze_returns() first."
        
        report = []
        report.append("=" * 80)
        report.append("INSTITUTIONAL PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {self.results['analysis_timestamp']}")
        report.append("")
        
        # Performance Summary
        perf = self.results['performance_summary']
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Return:           {perf['total_return']:.2%}")
        report.append(f"Annualized Return:      {perf['annualized_return']:.2%}")
        report.append(f"Annualized Volatility:  {perf['annualized_volatility']:.2%}")
        report.append(f"Sharpe Ratio:           {perf['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio:          {perf['sortino_ratio']:.3f}")
        report.append(f"Calmar Ratio:           {perf['calmar_ratio']:.3f}")
        report.append("")
        
        # Drawdown Analysis
        dd = self.results['drawdown_analysis']
        report.append("DRAWDOWN ANALYSIS")
        report.append("-" * 40)
        report.append(f"Maximum Drawdown:       {dd['max_drawdown']:.2%}")
        report.append(f"Max Drawdown Date:      {dd['max_drawdown_date']}")
        report.append(f"Avg Drawdown Duration:  {dd['avg_drawdown_duration_days']:.1f} days")
        report.append(f"Max Drawdown Duration:  {dd['max_drawdown_duration_days']:.1f} days")
        report.append(f"Total Drawdown Periods: {dd['total_drawdown_periods']}")
        report.append("")
        
        # Trade Statistics
        trade = self.results['trade_statistics']
        report.append("TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Win Rate:               {trade['win_rate']:.1%}")
        report.append(f"Profit Factor:          {trade['profit_factor']:.2f}")
        report.append(f"Average Win:            {trade['average_win']:.2%}")
        report.append(f"Average Loss:           {trade['average_loss']:.2%}")
        report.append(f"Win/Loss Ratio:         {trade['win_loss_ratio']:.2f}")
        report.append(f"Max Consecutive Wins:   {trade['max_consecutive_wins']}")
        report.append(f"Max Consecutive Losses: {trade['max_consecutive_losses']}")
        report.append("")
        
        # Risk Metrics
        risk = self.results['risk_metrics']
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"VaR (95%):              {risk['var_95']:.2%}")
        report.append(f"VaR (99%):              {risk['var_99']:.2%}")
        report.append(f"CVaR (95%):             {risk['cvar_95']:.2%}")
        report.append(f"CVaR (99%):             {risk['cvar_99']:.2%}")
        report.append(f"Skewness:               {risk['skewness']:.3f}")
        report.append(f"Kurtosis:               {risk['kurtosis']:.3f}")
        report.append(f"Max Daily Loss:         {risk['max_daily_loss']:.2%}")
        report.append(f"Max Daily Gain:         {risk['max_daily_gain']:.2%}")
        report.append("")
        
        # Benchmark Comparison (if available)
        if self.results['benchmark_comparison']:
            bench = self.results['benchmark_comparison']
            report.append("BENCHMARK COMPARISON")
            report.append("-" * 40)
            report.append(f"Alpha:                  {bench['alpha']:.2%}")
            report.append(f"Beta:                   {bench['beta']:.3f}")
            report.append(f"Information Ratio:      {bench['information_ratio']:.3f}")
            report.append(f"Tracking Error:         {bench['tracking_error']:.2%}")
            report.append(f"Correlation:            {bench['correlation']:.3f}")
            report.append(f"Excess Return:          {bench['excess_return']:.2%}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)