"""
Professional Reporting System
============================

Institutional-grade reporting with comprehensive trade attribution,
performance charts, visualizations, and benchmark comparisons.

Features:
- Detailed trade logs with attribution
- Performance charts and visualizations  
- Risk metrics dashboard
- Comparison to benchmarks
- Executive summary reports
- Tear sheet generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProfessionalReporter:
    """Institutional-grade reporting system"""
    
    def __init__(self, strategy_name: str = "Strategy", benchmark_name: str = "Benchmark"):
        """
        Initialize professional reporter
        
        Args:
            strategy_name: Name of the strategy being analyzed
            benchmark_name: Name of the benchmark for comparison
        """
        self.strategy_name = strategy_name
        self.benchmark_name = benchmark_name
        self.reports = {}
        self.charts = {}
        
        # Report configuration
        self.chart_size = (12, 8)
        self.dpi = 300
        self.color_scheme = {
            'strategy': '#2E86AB',
            'benchmark': '#A23B72', 
            'positive': '#F18F01',
            'negative': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        print("✅ Professional Reporting System initialized")
        print(f"   📊 Strategy: {strategy_name}")
        print(f"   📊 Benchmark: {benchmark_name}")
    
    def generate_comprehensive_report(self, 
                                    performance_results: Dict[str, Any],
                                    risk_results: Dict[str, Any],
                                    trade_data: pd.DataFrame = None,
                                    price_data: pd.DataFrame = None,
                                    returns_data: pd.Series = None,
                                    benchmark_data: pd.Series = None) -> Dict[str, Any]:
        """
        Generate comprehensive institutional report
        
        Args:
            performance_results: Results from PerformanceAnalyzer
            risk_results: Results from RiskManager
            trade_data: DataFrame with trade details
            price_data: Price series data
            returns_data: Strategy returns
            benchmark_data: Benchmark returns
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'strategy_name': self.strategy_name,
                'benchmark_name': self.benchmark_name,
                'report_type': 'comprehensive_institutional_analysis'
            },
            'executive_summary': self._generate_executive_summary(performance_results, risk_results),
            'performance_analysis': performance_results,
            'risk_analysis': risk_results,
            'trade_attribution': {},
            'charts_generated': [],
            'recommendations': []
        }
        
        # Generate trade attribution if trade data available
        if trade_data is not None and not trade_data.empty:
            report['trade_attribution'] = self._generate_trade_attribution(trade_data)
        
        # Generate charts if data available
        if returns_data is not None:
            charts = self._generate_performance_charts(
                returns_data, benchmark_data, price_data
            )
            report['charts_generated'] = list(charts.keys())
            self.charts.update(charts)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            performance_results, risk_results
        )
        
        self.reports['comprehensive'] = report
        return report
    
    def _generate_executive_summary(self, performance_results: Dict[str, Any], 
                                  risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        perf = performance_results.get('performance_summary', {})
        risk = risk_results.get('risk_limits', {})
        
        # Key metrics
        total_return = perf.get('total_return', 0)
        sharpe_ratio = perf.get('sharpe_ratio', 0)
        max_dd = performance_results.get('drawdown_analysis', {}).get('max_drawdown', 0)
        win_rate = performance_results.get('trade_statistics', {}).get('win_rate', 0)
        
        # Performance rating
        performance_score = self._calculate_performance_score(perf)
        risk_score = self._calculate_risk_score(performance_results, risk_results)
        
        # Key insights
        insights = []
        if total_return > 0.1:
            insights.append("Strong absolute returns generated")
        if sharpe_ratio > 1.0:
            insights.append("Excellent risk-adjusted returns")
        if max_dd < -0.1:
            insights.append("Significant drawdown periods observed")
        if win_rate > 0.6:
            insights.append("High win rate achieved")
        
        return {
            'performance_score': performance_score,
            'risk_score': risk_score,
            'overall_rating': (performance_score + risk_score) / 2,
            'key_metrics': {
                'total_return': total_return,
                'annualized_return': perf.get('annualized_return', 0),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'win_rate': win_rate
            },
            'key_insights': insights,
            'recommendation': self._get_overall_recommendation(performance_score, risk_score)
        }
    
    def _calculate_performance_score(self, performance_summary: Dict[str, Any]) -> float:
        """Calculate performance score (0-100)"""
        try:
            # Weight different metrics
            total_return = performance_summary.get('total_return', 0)
            sharpe_ratio = performance_summary.get('sharpe_ratio', 0)
            
            # Score components
            return_score = min(100, max(0, (total_return + 0.5) * 100))  # -50% to 50% -> 0 to 100
            sharpe_score = min(100, max(0, (sharpe_ratio + 1) * 25))     # -1 to 3 -> 0 to 100
            
            # Weighted average
            final_score = (return_score * 0.6) + (sharpe_score * 0.4)
            return round(final_score, 1)
        except:
            return 50.0  # Neutral score on error
    
    def _calculate_risk_score(self, performance_results: Dict[str, Any], 
                            risk_results: Dict[str, Any]) -> float:
        """Calculate risk score (0-100, higher is better)"""
        try:
            # Get risk metrics
            max_dd = abs(performance_results.get('drawdown_analysis', {}).get('max_drawdown', 0))
            volatility = performance_results.get('performance_summary', {}).get('annualized_volatility', 0)
            var_95 = abs(performance_results.get('risk_metrics', {}).get('var_95', 0))
            
            # Score components (lower risk = higher score)
            dd_score = max(0, 100 - (max_dd * 500))        # 20% DD = 0 score
            vol_score = max(0, 100 - (volatility * 250))   # 40% vol = 0 score
            var_score = max(0, 100 - (var_95 * 1000))      # 10% daily VaR = 0 score
            
            # Weighted average
            final_score = (dd_score * 0.4) + (vol_score * 0.3) + (var_score * 0.3)
            return round(final_score, 1)
        except:
            return 50.0  # Neutral score on error
    
    def _get_overall_recommendation(self, perf_score: float, risk_score: float) -> str:
        """Get overall recommendation based on scores"""
        overall_score = (perf_score + risk_score) / 2
        
        if overall_score >= 80:
            return "STRONG BUY - Excellent risk-adjusted performance"
        elif overall_score >= 65:
            return "BUY - Good performance with acceptable risk"
        elif overall_score >= 50:
            return "HOLD - Mixed performance, monitor closely"
        elif overall_score >= 35:
            return "WEAK HOLD - Below average performance"
        else:
            return "AVOID - Poor risk-adjusted returns"
    
    def _generate_trade_attribution(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed trade attribution analysis"""
        if trade_data.empty:
            return {'error': 'No trade data available'}
        
        try:
            # Ensure required columns exist
            required_cols = ['entry_time', 'exit_time', 'pnl', 'symbol']
            missing_cols = [col for col in required_cols if col not in trade_data.columns]
            if missing_cols:
                return {'error': f'Missing columns: {missing_cols}'}
            
            # Basic trade statistics
            total_trades = len(trade_data)
            winning_trades = len(trade_data[trade_data['pnl'] > 0])
            losing_trades = len(trade_data[trade_data['pnl'] < 0])
            
            # PnL analysis
            total_pnl = trade_data['pnl'].sum()
            avg_win = trade_data[trade_data['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trade_data[trade_data['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Trade duration analysis
            if 'entry_time' in trade_data.columns and 'exit_time' in trade_data.columns:
                trade_data['duration'] = pd.to_datetime(trade_data['exit_time']) - pd.to_datetime(trade_data['entry_time'])
                avg_duration = trade_data['duration'].mean()
                max_duration = trade_data['duration'].max()
                min_duration = trade_data['duration'].min()
            else:
                avg_duration = max_duration = min_duration = None
            
            # Symbol-level attribution
            symbol_attribution = {}
            if 'symbol' in trade_data.columns:
                for symbol in trade_data['symbol'].unique():
                    symbol_trades = trade_data[trade_data['symbol'] == symbol]
                    symbol_attribution[symbol] = {
                        'total_trades': len(symbol_trades),
                        'total_pnl': symbol_trades['pnl'].sum(),
                        'win_rate': len(symbol_trades[symbol_trades['pnl'] > 0]) / len(symbol_trades),
                        'avg_pnl_per_trade': symbol_trades['pnl'].mean()
                    }
            
            # Time-based attribution
            time_attribution = {}
            if 'entry_time' in trade_data.columns:
                trade_data['entry_hour'] = pd.to_datetime(trade_data['entry_time']).dt.hour
                trade_data['entry_day'] = pd.to_datetime(trade_data['entry_time']).dt.day_name()
                
                # Hour analysis
                hour_perf = trade_data.groupby('entry_hour')['pnl'].agg(['sum', 'mean', 'count'])
                time_attribution['hourly'] = {
                    int(hour): {
                        'total_pnl': float(row['sum']),
                        'avg_pnl': float(row['mean']),
                        'trade_count': int(row['count'])
                    } for hour, row in hour_perf.iterrows()
                }
                
                # Day analysis
                day_perf = trade_data.groupby('entry_day')['pnl'].agg(['sum', 'mean', 'count'])
                time_attribution['daily'] = {
                    day: {
                        'total_pnl': float(row['sum']),
                        'avg_pnl': float(row['mean']),
                        'trade_count': int(row['count'])
                    } for day, row in day_perf.iterrows()
                }
            
            # Best and worst trades
            best_trades = trade_data.nlargest(5, 'pnl')[['entry_time', 'symbol', 'pnl']].to_dict('records')
            worst_trades = trade_data.nsmallest(5, 'pnl')[['entry_time', 'symbol', 'pnl']].to_dict('records')
            
            return {
                'trade_summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': float(total_pnl),
                    'average_win': float(avg_win),
                    'average_loss': float(avg_loss),
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                },
                'duration_analysis': {
                    'average_duration': str(avg_duration) if avg_duration else None,
                    'max_duration': str(max_duration) if max_duration else None,
                    'min_duration': str(min_duration) if min_duration else None
                },
                'symbol_attribution': symbol_attribution,
                'time_attribution': time_attribution,
                'best_trades': best_trades,
                'worst_trades': worst_trades
            }
        except Exception as e:
            return {'error': f'Trade attribution failed: {str(e)}'}
    
    def _generate_performance_charts(self, returns_data: pd.Series, 
                                   benchmark_data: pd.Series = None,
                                   price_data: pd.Series = None) -> Dict[str, str]:
        """Generate performance visualization charts"""
        charts = {}
        
        try:
            # Ensure returns have datetime index
            if not isinstance(returns_data.index, pd.DatetimeIndex):
                returns_data.index = pd.to_datetime(returns_data.index)
            
            # 1. Cumulative Returns Chart
            fig, ax = plt.subplots(figsize=self.chart_size)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_data).cumprod()
            ax.plot(cumulative_returns.index, cumulative_returns.values, 
                   label=self.strategy_name, color=self.color_scheme['strategy'], linewidth=2)
            
            if benchmark_data is not None:
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    benchmark_data.index = pd.to_datetime(benchmark_data.index)
                
                # Align with strategy data
                aligned_benchmark = benchmark_data.reindex(returns_data.index, method='ffill')
                cumulative_benchmark = (1 + aligned_benchmark).cumprod()
                ax.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                       label=self.benchmark_name, color=self.color_scheme['benchmark'], linewidth=2)
            
            ax.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Return', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_path = f'/home/QuantNova/GrandModel/results/charts/cumulative_returns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            charts['cumulative_returns'] = chart_path
            
            # 2. Drawdown Chart
            fig, ax = plt.subplots(figsize=self.chart_size)
            
            # Calculate drawdowns
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            
            ax.fill_between(drawdown.index, drawdown.values, 0, 
                          color=self.color_scheme['negative'], alpha=0.7, label='Drawdown')
            ax.plot(drawdown.index, drawdown.values, color=self.color_scheme['negative'], linewidth=1)
            
            ax.set_title('Drawdown Analysis', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_path = f'/home/QuantNova/GrandModel/results/charts/drawdown_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            charts['drawdown'] = chart_path
            
            # 3. Rolling Sharpe Ratio
            fig, ax = plt.subplots(figsize=self.chart_size)
            
            # Calculate rolling Sharpe ratio (252-day window)
            rolling_window = min(252, len(returns_data) // 4)  # Quarterly or annual
            if rolling_window > 30:
                rolling_mean = returns_data.rolling(rolling_window).mean() * 252
                rolling_std = returns_data.rolling(rolling_window).std() * np.sqrt(252)
                rolling_sharpe = rolling_mean / rolling_std
                
                ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
                       color=self.color_scheme['strategy'], linewidth=2)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
                ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.5, label='Sharpe = 0.0')
                
                ax.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day window)', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Sharpe Ratio', fontsize=12)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                chart_path = f'/home/QuantNova/GrandModel/results/charts/rolling_sharpe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                charts['rolling_sharpe'] = chart_path
            
            # 4. Monthly Returns Heatmap
            if len(returns_data) > 30:  # At least a month of data
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create monthly returns matrix
                monthly_returns = returns_data.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns.index = monthly_returns.index.to_period('M')
                
                # Create pivot table for heatmap
                if len(monthly_returns) > 1:
                    monthly_df = pd.DataFrame({
                        'Year': monthly_returns.index.year,
                        'Month': monthly_returns.index.month,
                        'Return': monthly_returns.values
                    })
                    
                    pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
                    
                    # Create heatmap
                    sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                               center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
                    
                    ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Month', fontsize=12)
                    ax.set_ylabel('Year', fontsize=12)
                    
                    # Set month labels
                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    ax.set_xticklabels(month_labels)
                    
                    plt.tight_layout()
                    
                    chart_path = f'/home/QuantNova/GrandModel/results/charts/monthly_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()
                    charts['monthly_heatmap'] = chart_path
            
            # 5. Return Distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            ax1.hist(returns_data.values, bins=50, alpha=0.7, color=self.color_scheme['strategy'], edgecolor='black')
            ax1.axvline(returns_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns_data.mean():.2%}')
            ax1.axvline(returns_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {returns_data.median():.2%}')
            ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Daily Return', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(returns_data.values, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = f'/home/QuantNova/GrandModel/results/charts/return_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            charts['return_distribution'] = chart_path
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return charts
    
    def _generate_recommendations(self, performance_results: Dict[str, Any], 
                                risk_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Performance analysis
            perf = performance_results.get('performance_summary', {})
            sharpe = perf.get('sharpe_ratio', 0)
            total_return = perf.get('total_return', 0)
            
            # Risk analysis
            max_dd = performance_results.get('drawdown_analysis', {}).get('max_drawdown', 0)
            win_rate = performance_results.get('trade_statistics', {}).get('win_rate', 0)
            
            # Performance recommendations
            if sharpe < 0.5:
                recommendations.append("LOW SHARPE RATIO: Consider improving risk-adjusted returns through better entry/exit timing")
            elif sharpe > 2.0:
                recommendations.append("EXCELLENT SHARPE RATIO: Strategy demonstrates strong risk-adjusted performance")
            
            if total_return < 0:
                recommendations.append("NEGATIVE RETURNS: Review strategy logic and consider reducing position sizes")
            elif total_return > 0.3:
                recommendations.append("STRONG RETURNS: Consider increasing allocation if risk tolerance allows")
            
            # Risk recommendations
            if abs(max_dd) > 0.2:
                recommendations.append("HIGH DRAWDOWN: Implement tighter stop-losses and position sizing controls")
            
            if win_rate < 0.4:
                recommendations.append("LOW WIN RATE: Analyze losing trades for pattern improvements")
            elif win_rate > 0.7:
                recommendations.append("HIGH WIN RATE: Strategy shows good trade selection capability")
            
            # Risk management recommendations
            risk_breaches = risk_results.get('risk_breaches', {}).get('total_breaches', 0)
            if risk_breaches > 10:
                recommendations.append("FREQUENT RISK BREACHES: Review and tighten risk management parameters")
            
            # General recommendations
            if not recommendations:
                recommendations.append("BALANCED PERFORMANCE: Strategy operating within acceptable parameters")
            
            recommendations.append("ONGOING MONITORING: Continue to monitor performance and adjust parameters as needed")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def export_report_to_json(self, report_name: str = 'comprehensive') -> str:
        """Export report to JSON file"""
        if report_name not in self.reports:
            raise ValueError(f"Report '{report_name}' not found")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'/home/QuantNova/GrandModel/results/reports/{self.strategy_name}_{report_name}_report_{timestamp}.json'
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.reports[report_name], f, indent=2, default=str)
        
        return filename
    
    def generate_text_summary(self, report_name: str = 'comprehensive') -> str:
        """Generate text summary of the report"""
        if report_name not in self.reports:
            return "Report not found"
        
        report = self.reports[report_name]
        
        summary = []
        summary.append("=" * 80)
        summary.append(f"INSTITUTIONAL STRATEGY ANALYSIS: {self.strategy_name}")
        summary.append("=" * 80)
        summary.append(f"Report Generated: {report['report_metadata']['generated_at']}")
        summary.append("")
        
        # Executive Summary
        exec_summary = report.get('executive_summary', {})
        summary.append("EXECUTIVE SUMMARY")
        summary.append("-" * 40)
        summary.append(f"Overall Rating: {exec_summary.get('overall_rating', 0):.1f}/100")
        summary.append(f"Performance Score: {exec_summary.get('performance_score', 0):.1f}/100")
        summary.append(f"Risk Score: {exec_summary.get('risk_score', 0):.1f}/100")
        summary.append(f"Recommendation: {exec_summary.get('recommendation', 'N/A')}")
        summary.append("")
        
        # Key Metrics
        key_metrics = exec_summary.get('key_metrics', {})
        summary.append("KEY PERFORMANCE METRICS")
        summary.append("-" * 40)
        summary.append(f"Total Return: {key_metrics.get('total_return', 0):.2%}")
        summary.append(f"Annualized Return: {key_metrics.get('annualized_return', 0):.2%}")
        summary.append(f"Sharpe Ratio: {key_metrics.get('sharpe_ratio', 0):.3f}")
        summary.append(f"Maximum Drawdown: {key_metrics.get('max_drawdown', 0):.2%}")
        summary.append(f"Win Rate: {key_metrics.get('win_rate', 0):.1%}")
        summary.append("")
        
        # Key Insights
        insights = exec_summary.get('key_insights', [])
        if insights:
            summary.append("KEY INSIGHTS")
            summary.append("-" * 40)
            for insight in insights:
                summary.append(f"• {insight}")
            summary.append("")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            summary.append("RECOMMENDATIONS")
            summary.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                summary.append(f"{i}. {rec}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)