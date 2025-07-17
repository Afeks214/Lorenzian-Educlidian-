"""
Professional Backtesting Framework
==================================

Institutional-grade backtesting framework that integrates all components:
- Performance Analytics
- Risk Management
- Professional Reporting
- Data Quality Assurance
- Strategy Integration

This framework provides a complete solution for institutional backtesting
with comprehensive analytics, risk controls, and professional reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
import os
import json
warnings.filterwarnings('ignore')

from .performance_analytics import PerformanceAnalyzer
from .risk_management import RiskManager, create_default_stress_scenarios
from .reporting import ProfessionalReporter
from .data_quality import DataQualityAssurance

class ProfessionalBacktestFramework:
    """
    Institutional-grade backtesting framework
    
    Integrates performance analytics, risk management, reporting,
    and data quality assurance for comprehensive strategy evaluation.
    """
    
    def __init__(self, 
                 strategy_name: str = "Strategy",
                 benchmark_symbol: str = "SPY",
                 initial_capital: float = 100000,
                 risk_free_rate: float = 0.02,
                 risk_params: Dict[str, Any] = None):
        """
        Initialize professional backtesting framework
        
        Args:
            strategy_name: Name of the strategy
            benchmark_symbol: Benchmark symbol for comparison
            initial_capital: Initial capital for backtesting
            risk_free_rate: Risk-free rate for calculations
            risk_params: Risk management parameters
        """
        self.strategy_name = strategy_name
        self.benchmark_symbol = benchmark_symbol
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer(risk_free_rate)
        
        # Risk management parameters
        if risk_params is None:
            risk_params = {
                'max_position_size': 0.1,
                'max_daily_loss': 0.02,
                'max_total_exposure': 1.0,
                'correlation_threshold': 0.7,
                'var_confidence': 0.95
            }
        self.risk_manager = RiskManager(**risk_params)
        
        self.reporter = ProfessionalReporter(strategy_name, benchmark_symbol)
        self.data_quality = DataQualityAssurance()
        
        # Framework state
        self.backtest_results = {}
        self.portfolio_state = {
            'total_value': initial_capital,
            'positions': {},
            'cash': initial_capital,
            'daily_pnl': 0,
            'returns_history': [],
            'total_exposure': 0
        }
        
        # Trade tracking
        self.trades = []
        self.signals = []
        
        print("✅ Professional Backtesting Framework initialized")
        print(f"   📊 Strategy: {strategy_name}")
        print(f"   💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"   📈 Benchmark: {benchmark_symbol}")
        print(f"   🛡️ Risk Controls: Enabled")
    
    def run_comprehensive_backtest(self, 
                                 strategy_function: Callable,
                                 data: pd.DataFrame,
                                 benchmark_data: pd.Series = None,
                                 generate_charts: bool = True,
                                 stress_test: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive institutional backtest
        
        Args:
            strategy_function: Strategy function that generates signals
            data: Price/market data for backtesting
            benchmark_data: Benchmark data for comparison
            generate_charts: Whether to generate visualization charts
            stress_test: Whether to perform stress testing
            
        Returns:
            Comprehensive backtest results
        """
        print("🚀 Starting Comprehensive Institutional Backtest")
        print("=" * 60)
        
        # Step 1: Data Quality Assessment
        print("📊 Step 1: Data Quality Assessment...")
        quality_report = self._assess_data_quality(data)
        
        if quality_report['quality_score'] < 70:
            print(f"⚠️ WARNING: Low data quality score: {quality_report['quality_score']:.1f}")
            print("Consider data preprocessing before proceeding")
        
        # Step 2: Strategy Signal Generation
        print("🎯 Step 2: Strategy Signal Generation...")
        signals = self._generate_strategy_signals(strategy_function, data)
        
        # Step 3: Risk-Managed Execution
        print("🛡️ Step 3: Risk-Managed Trade Execution...")
        execution_results = self._execute_risk_managed_trades(signals, data)
        
        # Step 4: Performance Analysis
        print("📈 Step 4: Performance Analysis...")
        performance_results = self._analyze_performance(benchmark_data)
        
        # Step 5: Risk Analysis
        print("🔍 Step 5: Risk Analysis...")
        risk_results = self._analyze_risk()
        
        # Step 6: Stress Testing
        stress_results = {}
        if stress_test:
            print("⚡ Step 6: Stress Testing...")
            stress_results = self._perform_stress_tests()
        
        # Step 7: Professional Reporting
        print("📋 Step 7: Generating Professional Reports...")
        reports = self._generate_comprehensive_reports(
            performance_results, risk_results, quality_report,
            data, benchmark_data, generate_charts
        )
        
        # Compile final results
        final_results = {
            'framework_metadata': {
                'strategy_name': self.strategy_name,
                'backtest_date': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'total_trades': len(self.trades),
                'total_signals': len(self.signals)
            },
            'data_quality': quality_report,
            'execution_summary': execution_results,
            'performance_analysis': performance_results,
            'risk_analysis': risk_results,
            'stress_test_results': stress_results,
            'professional_reports': reports,
            'recommendations': self._generate_final_recommendations(
                performance_results, risk_results, quality_report
            )
        }
        
        # Save results
        self.backtest_results = final_results
        self._save_results(final_results)
        
        print("✅ Comprehensive Backtesting Complete!")
        self._print_executive_summary(final_results)
        
        return final_results
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality before backtesting"""
        try:
            # Define column types for analysis
            price_columns = ['Open', 'High', 'Low', 'Close']
            volume_columns = ['Volume'] if 'Volume' in data.columns else []
            
            quality_report = self.data_quality.comprehensive_quality_check(
                data, price_columns, volume_columns
            )
            
            print(f"   📊 Data Quality Score: {quality_report['quality_score']:.1f}/100")
            print(f"   📅 Data Period: {data.index[0]} to {data.index[-1]}")
            print(f"   📈 Total Records: {len(data)}")
            
            return quality_report
        except Exception as e:
            print(f"❌ Data quality assessment failed: {e}")
            return {'quality_score': 50, 'error': str(e)}
    
    def _generate_strategy_signals(self, strategy_function: Callable, 
                                 data: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy signals"""
        try:
            # Call strategy function to generate signals
            signals = strategy_function(data)
            
            # Ensure signals are in DataFrame format
            if not isinstance(signals, pd.DataFrame):
                # Convert to DataFrame if needed
                if isinstance(signals, pd.Series):
                    signals = pd.DataFrame({'signal': signals})
                else:
                    raise ValueError("Strategy function must return DataFrame or Series")
            
            # Store signals
            self.signals = signals
            
            signal_count = len(signals[signals.get('signal', 0) != 0])
            print(f"   🎯 Generated {signal_count} trading signals")
            
            return signals
        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return pd.DataFrame()
    
    def _execute_risk_managed_trades(self, signals: pd.DataFrame, 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trades with risk management"""
        try:
            executed_trades = 0
            blocked_trades = 0
            risk_adjustments = 0
            
            # Iterate through signals
            for timestamp, signal_row in signals.iterrows():
                if signal_row.get('signal', 0) == 0:
                    continue
                
                # Get current price data
                if timestamp not in data.index:
                    continue
                    
                price_data = data.loc[timestamp]
                current_price = price_data['Close']
                
                # Create trade data
                signal_strength = abs(signal_row.get('signal', 0))
                base_position_size = self._calculate_position_size(signal_strength)
                
                trade_data = {
                    'symbol': 'STRATEGY',
                    'timestamp': timestamp,
                    'signal': signal_row.get('signal', 0),
                    'price': current_price,
                    'size': base_position_size,
                    'type': 'market'
                }
                
                # Risk validation
                validation_result = self.risk_manager.validate_trade(
                    trade_data, self.portfolio_state
                )
                
                if validation_result['approved']:
                    # Execute trade (possibly with adjustments)
                    final_size = validation_result['adjustments'].get('size', base_position_size)
                    if final_size != base_position_size:
                        risk_adjustments += 1
                    
                    # Execute the trade
                    self._execute_trade(trade_data, final_size, current_price)
                    executed_trades += 1
                else:
                    blocked_trades += 1
                    print(f"   🛡️ Trade blocked by risk management at {timestamp}")
                
                # Update portfolio state
                self._update_portfolio_state(timestamp, data)
            
            execution_summary = {
                'total_signals': len(signals[signals.get('signal', 0) != 0]),
                'executed_trades': executed_trades,
                'blocked_trades': blocked_trades,
                'risk_adjustments': risk_adjustments,
                'execution_rate': executed_trades / max(1, len(signals[signals.get('signal', 0) != 0]))
            }
            
            print(f"   ✅ Executed {executed_trades} trades")
            print(f"   🛡️ Blocked {blocked_trades} trades")
            print(f"   ⚙️ Applied {risk_adjustments} risk adjustments")
            
            return execution_summary
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            return {'error': str(e)}
    
    def _calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength"""
        # Base position size as fraction of portfolio
        base_size = 0.05  # 5% of portfolio
        
        # Adjust based on signal strength
        adjusted_size = base_size * min(signal_strength, 2.0)  # Cap at 2x base
        
        return adjusted_size
    
    def _execute_trade(self, trade_data: Dict[str, Any], 
                      size: float, price: float):
        """Execute a trade and update records"""
        try:
            trade_value = size * self.portfolio_state['total_value']
            shares = trade_value / price
            
            # Record trade
            trade_record = {
                'timestamp': trade_data['timestamp'],
                'symbol': trade_data['symbol'],
                'signal': trade_data['signal'],
                'price': price,
                'shares': shares,
                'value': trade_value,
                'type': trade_data['type']
            }
            
            self.trades.append(trade_record)
            
            # Update portfolio positions
            symbol = trade_data['symbol']
            if symbol not in self.portfolio_state['positions']:
                self.portfolio_state['positions'][symbol] = {
                    'shares': 0,
                    'value': 0,
                    'avg_price': 0
                }
            
            position = self.portfolio_state['positions'][symbol]
            
            # Update position
            if trade_data['signal'] > 0:  # Buy
                total_shares = position['shares'] + shares
                total_value = position['value'] + trade_value
                position['shares'] = total_shares
                position['value'] = total_value
                position['avg_price'] = total_value / total_shares if total_shares > 0 else price
                
                # Update cash
                self.portfolio_state['cash'] -= trade_value
            else:  # Sell
                position['shares'] = max(0, position['shares'] - shares)
                position['value'] = max(0, position['value'] - trade_value)
                
                # Update cash
                self.portfolio_state['cash'] += trade_value
            
            # Update total exposure
            self.portfolio_state['total_exposure'] = sum(
                pos['value'] for pos in self.portfolio_state['positions'].values()
            )
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
    def _update_portfolio_state(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Update portfolio state with current market values"""
        try:
            if timestamp not in data.index:
                return
            
            current_price = data.loc[timestamp, 'Close']
            
            # Update position values
            total_position_value = 0
            for symbol, position in self.portfolio_state['positions'].items():
                if position['shares'] > 0:
                    current_value = position['shares'] * current_price
                    position['current_value'] = current_value
                    total_position_value += current_value
            
            # Update portfolio totals
            previous_value = self.portfolio_state['total_value']
            self.portfolio_state['total_value'] = self.portfolio_state['cash'] + total_position_value
            self.portfolio_state['total_exposure'] = total_position_value
            
            # Calculate daily return
            daily_return = (self.portfolio_state['total_value'] / previous_value - 1) if previous_value > 0 else 0
            self.portfolio_state['returns_history'].append(daily_return)
            self.portfolio_state['daily_pnl'] = self.portfolio_state['total_value'] - previous_value
            
        except Exception as e:
            print(f"❌ Portfolio update error: {e}")
    
    def _analyze_performance(self, benchmark_data: pd.Series = None) -> Dict[str, Any]:
        """Analyze strategy performance"""
        try:
            # Create returns series
            returns_series = pd.Series(
                self.portfolio_state['returns_history'],
                index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=len(self.portfolio_state['returns_history'])), 
                                  periods=len(self.portfolio_state['returns_history']), freq='D')
            )
            
            # Calculate portfolio value series
            portfolio_values = [self.initial_capital]
            for ret in self.portfolio_state['returns_history']:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
            
            portfolio_series = pd.Series(portfolio_values[1:], index=returns_series.index)
            
            # Run performance analysis
            performance_results = self.performance_analyzer.analyze_returns(
                returns_series, portfolio_series, benchmark_data
            )
            
            perf_summary = performance_results['performance_summary']
            print(f"   📈 Total Return: {perf_summary['total_return']:.2%}")
            print(f"   📊 Sharpe Ratio: {perf_summary['sharpe_ratio']:.3f}")
            print(f"   📉 Max Drawdown: {performance_results['drawdown_analysis']['max_drawdown']:.2%}")
            
            return performance_results
        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics(self.portfolio_state)
            risk_report = self.risk_manager.generate_risk_report()
            
            risk_results = {
                'current_metrics': risk_metrics,
                'risk_report': risk_report,
                'risk_breaches': len(self.risk_manager.risk_breaches),
                'risk_utilization': risk_metrics.get('risk_utilization', {})
            }
            
            print(f"   🛡️ Risk Breaches: {risk_results['risk_breaches']}")
            print(f"   📊 Portfolio Volatility: {risk_metrics.get('portfolio_volatility', 0):.2%}")
            
            return risk_results
        except Exception as e:
            print(f"❌ Risk analysis failed: {e}")
            return {'error': str(e)}
    
    def _perform_stress_tests(self) -> Dict[str, Any]:
        """Perform stress testing"""
        try:
            # Create stress scenarios
            scenarios = create_default_stress_scenarios()
            
            # Run stress tests
            stress_results = self.risk_manager.perform_stress_test(
                self.portfolio_state, scenarios
            )
            
            summary = stress_results.get('summary', {})
            print(f"   ⚡ Worst Case Loss: {summary.get('worst_case_loss', 0):.2%}")
            print(f"   📊 Scenarios Tested: {summary.get('total_scenarios', 0)}")
            
            return stress_results
        except Exception as e:
            print(f"❌ Stress testing failed: {e}")
            return {'error': str(e)}
    
    def _generate_comprehensive_reports(self, performance_results: Dict[str, Any],
                                      risk_results: Dict[str, Any],
                                      quality_report: Dict[str, Any],
                                      data: pd.DataFrame,
                                      benchmark_data: pd.Series = None,
                                      generate_charts: bool = True) -> Dict[str, Any]:
        """Generate comprehensive professional reports"""
        try:
            # Prepare trade data
            trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
            
            # Prepare returns data
            returns_series = pd.Series(
                self.portfolio_state['returns_history'],
                index=pd.date_range(start=data.index[0], periods=len(self.portfolio_state['returns_history']), freq='D')
            ) if self.portfolio_state['returns_history'] else pd.Series()
            
            # Generate comprehensive report
            comprehensive_report = self.reporter.generate_comprehensive_report(
                performance_results=performance_results,
                risk_results=risk_results,
                trade_data=trades_df,
                price_data=data['Close'] if 'Close' in data.columns else None,
                returns_data=returns_series,
                benchmark_data=benchmark_data
            )
            
            # Export reports
            report_files = {}
            
            # JSON report
            json_file = self.reporter.export_report_to_json('comprehensive')
            report_files['json_report'] = json_file
            
            # Text summary
            text_summary = self.reporter.generate_text_summary('comprehensive')
            summary_file = f'/home/QuantNova/GrandModel/results/reports/{self.strategy_name}_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            
            with open(summary_file, 'w') as f:
                f.write(text_summary)
            report_files['text_summary'] = summary_file
            
            # Data quality report
            quality_text = self.data_quality.generate_quality_report_text(quality_report)
            quality_file = f'/home/QuantNova/GrandModel/results/reports/{self.strategy_name}_data_quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            with open(quality_file, 'w') as f:
                f.write(quality_text)
            report_files['data_quality_report'] = quality_file
            
            print(f"   📋 Reports generated: {len(report_files)} files")
            
            return {
                'comprehensive_report': comprehensive_report,
                'report_files': report_files,
                'charts_generated': comprehensive_report.get('charts_generated', [])
            }
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_final_recommendations(self, performance_results: Dict[str, Any],
                                      risk_results: Dict[str, Any],
                                      quality_report: Dict[str, Any]) -> List[str]:
        """Generate final strategic recommendations"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            perf_summary = performance_results.get('performance_summary', {})
            sharpe_ratio = perf_summary.get('sharpe_ratio', 0)
            total_return = perf_summary.get('total_return', 0)
            
            if sharpe_ratio > 1.5:
                recommendations.append("EXCELLENT RISK-ADJUSTED PERFORMANCE: Consider increasing allocation")
            elif sharpe_ratio < 0.5:
                recommendations.append("POOR RISK-ADJUSTED PERFORMANCE: Strategy requires significant improvement")
            
            if total_return > 0.2:
                recommendations.append("STRONG ABSOLUTE RETURNS: Strategy shows good alpha generation")
            elif total_return < 0:
                recommendations.append("NEGATIVE RETURNS: Fundamental strategy review required")
            
            # Risk-based recommendations
            risk_breaches = risk_results.get('risk_breaches', 0)
            if risk_breaches > 10:
                recommendations.append("FREQUENT RISK BREACHES: Tighten risk management parameters")
            elif risk_breaches == 0:
                recommendations.append("EXCELLENT RISK DISCIPLINE: No risk limit breaches detected")
            
            # Data quality recommendations
            quality_score = quality_report.get('quality_score', 100)
            if quality_score < 80:
                recommendations.append("DATA QUALITY CONCERNS: Improve data sources before live deployment")
            
            # Execution recommendations
            if len(self.trades) < 10:
                recommendations.append("LOW TRADE FREQUENCY: Consider increasing signal sensitivity")
            elif len(self.trades) > 1000:
                recommendations.append("HIGH TRADE FREQUENCY: Monitor transaction costs and market impact")
            
            # Overall recommendation
            if sharpe_ratio > 1.0 and total_return > 0.1 and quality_score > 85:
                recommendations.append("STRATEGY APPROVED FOR INSTITUTIONAL DEPLOYMENT")
            elif sharpe_ratio > 0.7 and total_return > 0.05:
                recommendations.append("STRATEGY SHOWS PROMISE - CONSIDER PAPER TRADING FIRST")
            else:
                recommendations.append("STRATEGY REQUIRES FURTHER DEVELOPMENT BEFORE DEPLOYMENT")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'/home/QuantNova/GrandModel/results/backtests/{self.strategy_name}_comprehensive_backtest_{timestamp}.json'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"   💾 Results saved: {results_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _print_executive_summary(self, results: Dict[str, Any]):
        """Print executive summary of backtest results"""
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY")
        print("=" * 80)
        
        try:
            # Performance summary
            perf = results.get('performance_analysis', {}).get('performance_summary', {})
            print(f"📈 Total Return: {perf.get('total_return', 0):.2%}")
            print(f"📊 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"📉 Max Drawdown: {results.get('performance_analysis', {}).get('drawdown_analysis', {}).get('max_drawdown', 0):.2%}")
            
            # Risk summary
            risk = results.get('risk_analysis', {})
            print(f"🛡️ Risk Breaches: {risk.get('risk_breaches', 0)}")
            
            # Execution summary
            exec_summary = results.get('execution_summary', {})
            print(f"⚡ Execution Rate: {exec_summary.get('execution_rate', 0):.1%}")
            print(f"📊 Total Trades: {exec_summary.get('executed_trades', 0)}")
            
            # Data quality
            quality_score = results.get('data_quality', {}).get('quality_score', 0)
            print(f"📋 Data Quality: {quality_score:.1f}/100")
            
            # Overall recommendation
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n🎯 KEY RECOMMENDATION: {recommendations[-1]}")
            
        except Exception as e:
            print(f"❌ Error printing summary: {e}")
        
        print("=" * 80)

def create_simple_strategy_function():
    """Create a simple example strategy function"""
    def simple_momentum_strategy(data: pd.DataFrame) -> pd.DataFrame:
        """Simple momentum strategy for demonstration"""
        if 'Close' not in data.columns:
            return pd.DataFrame()
        
        # Calculate simple momentum signals
        close = data['Close']
        sma_short = close.rolling(window=10).mean()
        sma_long = close.rolling(window=30).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Buy when short MA crosses above long MA
        signals.loc[sma_short > sma_long, 'signal'] = 1
        # Sell when short MA crosses below long MA
        signals.loc[sma_short < sma_long, 'signal'] = -1
        
        return signals
    
    return simple_momentum_strategy