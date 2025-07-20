"""
Comprehensive VectorBT Backtesting Pipeline
==========================================

Production-ready backtesting pipeline that integrates:
- VectorBT framework with Lorentzian indicators
- Performance metrics and analysis
- Production monitoring and alerting
- Walk-forward optimization
- Risk management and validation

Author: Claude Code
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# Add project paths
project_root = Path("/home/QuantNova/GrandModel")
sys.path.append(str(project_root))

# Import our custom modules
from lorentzian_strategy.backtesting.vectorbt_framework import (
    VectorBTFramework, BacktestConfig, create_vectorbt_framework
)
from lorentzian_strategy.backtesting.performance_metrics import (
    PerformanceAnalyzer, PerformanceTargets, create_performance_analyzer
)
from lorentzian_strategy.utils.monitoring import (
    ProductionMonitoringSystem, AlertConfig, NotificationConfig,
    create_monitoring_system
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/QuantNova/GrandModel/logs/backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestingPipeline:
    """Comprehensive backtesting pipeline for MARL trading system"""
    
    def __init__(self, 
                 data_path: str = "/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv",
                 results_dir: str = "/home/QuantNova/GrandModel/results/vectorbt_backtesting",
                 config: BacktestConfig = None):
        
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or BacktestConfig(data_path=data_path)
        
        # Components
        self.framework = None
        self.performance_analyzer = None
        self.monitoring_system = None
        
        # Results storage
        self.results = {}
        self.optimization_results = {}
        self.validation_results = {}
        
        logger.info(f"Backtesting pipeline initialized for data: {data_path}")
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize VectorBT framework
            self.framework = create_vectorbt_framework(self.config)
            
            # Initialize monitoring system
            alert_config = AlertConfig(
                min_sharpe_ratio=self.config.target_sharpe,
                max_drawdown_pct=self.config.target_max_dd * 100,
                min_win_rate=self.config.target_win_rate
            )
            
            notification_config = NotificationConfig(
                alert_recipients=["trader@example.com"]  # Configure as needed
            )
            
            self.monitoring_system = create_monitoring_system(alert_config, notification_config)
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate market data"""
        logger.info("Loading and validating market data...")
        
        try:
            # Load data using framework
            data = self.framework.load_data()
            
            # Data quality checks
            if self.monitoring_system:
                self.monitoring_system.check_data_quality(data)
            
            # Basic validation
            if len(data) < 1000:
                raise ValueError(f"Insufficient data points: {len(data)} (minimum 1000 required)")
            
            # Check for required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check data integrity
            if data['Close'].isnull().sum() > 0:
                logger.warning(f"Found {data['Close'].isnull().sum()} null values in Close prices")
                data = data.dropna()
            
            # Validate price data
            if (data['High'] < data['Low']).any():
                raise ValueError("High prices should be >= Low prices")
            
            if (data['Close'] > data['High']).any() or (data['Close'] < data['Low']).any():
                raise ValueError("Close prices should be between High and Low")
            
            # Check for extreme values
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                if q99 / q01 > 10:  # Prices vary by more than 10x
                    logger.warning(f"Large price variation detected in {col}: {q99/q01:.2f}x")
            
            logger.info(f"Data validation completed. Dataset shape: {data.shape}")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def run_basic_backtest(self) -> Dict[str, Any]:
        """Run basic backtest with default parameters"""
        logger.info("Running basic backtest...")
        
        try:
            # Load and validate data
            data = self.load_and_validate_data()
            
            # Generate features and signals
            self.framework.generate_lorentzian_features()
            signals = self.framework.generate_signals()
            
            logger.info(f"Generated {(signals != 0).sum()} trading signals")
            
            # Run backtest
            portfolio = self.framework.run_backtest()
            
            # Calculate performance metrics
            metrics = self.framework.calculate_performance_metrics(portfolio)
            
            # Create performance analyzer
            self.performance_analyzer = create_performance_analyzer(
                portfolio_values=portfolio.value(),
                trades=portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else None
            )
            
            # Generate comprehensive analysis
            comprehensive_report = self.performance_analyzer.generate_comprehensive_report()
            
            # Store results
            self.results['basic_backtest'] = {
                'portfolio': portfolio,
                'metrics': metrics,
                'comprehensive_report': comprehensive_report,
                'signals': signals,
                'data': data
            }
            
            # Update monitoring system
            if self.monitoring_system:
                self.monitoring_system.update_performance_metrics(
                    portfolio_value=portfolio.value().iloc[-1],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    drawdown=metrics['max_drawdown'],
                    win_rate=metrics['win_rate']
                )
            
            logger.info("Basic backtest completed successfully")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            
            return self.results['basic_backtest']
            
        except Exception as e:
            logger.error(f"Basic backtest failed: {e}")
            raise
    
    def run_parameter_optimization(self, 
                                 param_ranges: Dict[str, List] = None,
                                 max_iterations: int = 100) -> Dict[str, Any]:
        """Run parameter optimization"""
        logger.info("Starting parameter optimization...")
        
        try:
            # Default parameter ranges if not provided
            if param_ranges is None:
                param_ranges = {
                    'fast_period': [5, 8, 12, 15],
                    'slow_period': [15, 21, 30, 40],
                    'rsi_period': [10, 14, 20, 25],
                    'rsi_overbought': [65, 70, 75, 80],
                    'rsi_oversold': [20, 25, 30, 35],
                    'stop_loss_pct': [0.01, 0.02, 0.03, 0.04],
                    'take_profit_pct': [0.02, 0.04, 0.06, 0.08],
                    'max_position_size': [0.15, 0.25, 0.35, 0.50]
                }
            
            # Limit combinations if too many
            total_combinations = 1
            for values in param_ranges.values():
                total_combinations *= len(values)
            
            if total_combinations > max_iterations:
                logger.warning(f"Too many parameter combinations ({total_combinations}). "
                             f"Reducing to {max_iterations} iterations.")
                # Sample random combinations
                param_ranges = self._sample_parameter_combinations(param_ranges, max_iterations)
            
            # Run optimization
            optimization_results = self.framework.optimize_parameters(param_ranges)
            
            # Store results
            self.optimization_results = optimization_results
            
            logger.info("Parameter optimization completed")
            logger.info(f"Best Sharpe ratio: {optimization_results['best_sharpe']:.3f}")
            logger.info(f"Best parameters: {optimization_results['best_params']}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise
    
    def run_walk_forward_analysis(self, 
                                window_months: int = 12,
                                step_months: int = 3) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        logger.info(f"Starting walk-forward analysis (window: {window_months}m, step: {step_months}m)...")
        
        try:
            # Run walk-forward analysis
            wf_results = self.framework.walk_forward_analysis(window_months, step_months)
            
            # Analyze results
            period_results = wf_results['periods']
            
            if not period_results:
                raise ValueError("No valid walk-forward periods generated")
            
            # Calculate stability metrics
            sharpe_ratios = [p['metrics']['sharpe_ratio'] for p in period_results]
            returns = [p['metrics']['total_return'] for p in period_results]
            drawdowns = [p['metrics']['max_drawdown'] for p in period_results]
            
            stability_analysis = {
                'sharpe_consistency': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'positive_periods': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios)
                },
                'return_consistency': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'positive_periods': sum(1 for r in returns if r > 0) / len(returns)
                },
                'drawdown_consistency': {
                    'mean': np.mean(drawdowns),
                    'std': np.std(drawdowns),
                    'worst': np.min(drawdowns),
                    'best': np.max(drawdowns)
                }
            }
            
            # Store results
            self.validation_results['walk_forward'] = {
                'results': wf_results,
                'stability_analysis': stability_analysis,
                'period_count': len(period_results)
            }
            
            logger.info("Walk-forward analysis completed")
            logger.info(f"Analyzed {len(period_results)} periods")
            logger.info(f"Average Sharpe ratio: {stability_analysis['sharpe_consistency']['mean']:.3f}")
            logger.info(f"Sharpe consistency (std): {stability_analysis['sharpe_consistency']['std']:.3f}")
            logger.info(f"Positive periods: {stability_analysis['sharpe_consistency']['positive_periods']:.1%}")
            
            return self.validation_results['walk_forward']
            
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            raise
    
    def run_stress_testing(self) -> Dict[str, Any]:
        """Run stress testing scenarios"""
        logger.info("Running stress testing scenarios...")
        
        try:
            if self.performance_analyzer is None:
                raise ValueError("Performance analyzer not available. Run basic backtest first.")
            
            # Run stress tests
            stress_results = self.performance_analyzer.stress_testing()
            
            # Monte Carlo analysis
            monte_carlo_results = self.performance_analyzer.monte_carlo_analysis(
                num_simulations=1000, periods=252
            )
            
            # Regime analysis
            regime_results = self.performance_analyzer.regime_analysis()
            
            # Compile stress testing results
            stress_analysis = {
                'scenario_analysis': stress_results,
                'monte_carlo': monte_carlo_results,
                'regime_analysis': regime_results,
                'risk_assessment': self._assess_risk_scenarios(stress_results, monte_carlo_results)
            }
            
            # Store results
            self.validation_results['stress_testing'] = stress_analysis
            
            logger.info("Stress testing completed")
            logger.info(f"Monte Carlo positive probability: {monte_carlo_results['probability_positive']:.1%}")
            logger.info(f"Expected return: {monte_carlo_results['expected_return']:.2%}")
            logger.info(f"Downside risk (>10% loss): {monte_carlo_results['downside_risk']:.1%}")
            
            return stress_analysis
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            raise
    
    def validate_performance_targets(self) -> Dict[str, Any]:
        """Validate performance against targets"""
        logger.info("Validating performance against targets...")
        
        try:
            if self.performance_analyzer is None:
                raise ValueError("Performance analyzer not available. Run basic backtest first.")
            
            # Define targets
            targets = PerformanceTargets(
                target_sharpe=self.config.target_sharpe,
                target_max_drawdown=self.config.target_max_dd,
                target_win_rate=self.config.target_win_rate,
                target_profit_factor=self.config.target_profit_factor
            )
            
            # Run target achievement analysis
            achievement_results = self.performance_analyzer.target_achievement_analysis(targets)
            
            # Calculate overall score
            passed_targets = sum(1 for metric in achievement_results.values() 
                               if metric.get('passed', False))
            total_targets = len([k for k in achievement_results.keys() if k != 'overall'])
            
            validation_summary = {
                'targets_passed': passed_targets,
                'total_targets': total_targets,
                'pass_rate': passed_targets / total_targets if total_targets > 0 else 0,
                'overall_grade': achievement_results['overall']['grade'],
                'overall_score': achievement_results['overall']['score'],
                'achievement_details': achievement_results
            }
            
            # Store results
            self.validation_results['target_validation'] = validation_summary
            
            logger.info("Performance target validation completed")
            logger.info(f"Targets passed: {passed_targets}/{total_targets} ({passed_targets/total_targets:.1%})")
            logger.info(f"Overall grade: {achievement_results['overall']['grade']}")
            logger.info(f"Overall score: {achievement_results['overall']['score']:.2f}")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"Performance target validation failed: {e}")
            raise
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtesting report"""
        logger.info("Generating comprehensive backtesting report...")
        
        try:
            # Ensure we have basic results
            if 'basic_backtest' not in self.results:
                logger.warning("Basic backtest not run. Running now...")
                self.run_basic_backtest()
            
            # Compile all results
            comprehensive_report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_path': self.data_path,
                    'strategy': 'Lorentzian Classification MARL',
                    'framework_version': '1.0.0'
                },
                'configuration': {
                    'backtest_config': {
                        'initial_cash': self.config.initial_cash,
                        'commission': self.config.commission,
                        'slippage': self.config.slippage,
                        'max_position_size': self.config.max_position_size,
                        'stop_loss_pct': self.config.stop_loss_pct,
                        'take_profit_pct': self.config.take_profit_pct
                    },
                    'targets': {
                        'target_sharpe': self.config.target_sharpe,
                        'target_max_dd': self.config.target_max_dd,
                        'target_win_rate': self.config.target_win_rate,
                        'target_profit_factor': self.config.target_profit_factor
                    }
                },
                'basic_backtest': self.results.get('basic_backtest', {}),
                'optimization': self.optimization_results,
                'validation': self.validation_results,
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            report_path = self.results_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            logger.info(f"Comprehensive report saved to: {report_path}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise
    
    def save_results(self, include_portfolio: bool = False) -> str:
        """Save all results to files"""
        logger.info("Saving backtesting results...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save basic metrics
            if 'basic_backtest' in self.results:
                metrics_path = self.results_dir / f"metrics_{timestamp}.json"
                with open(metrics_path, 'w') as f:
                    metrics = self.results['basic_backtest']['metrics'].copy()
                    json.dump(metrics, f, indent=2, default=str)
                
                # Save signals
                signals_path = self.results_dir / f"signals_{timestamp}.csv"
                self.results['basic_backtest']['signals'].to_csv(signals_path)
                
                # Save portfolio data (optional)
                if include_portfolio and 'portfolio' in self.results['basic_backtest']:
                    portfolio_path = self.results_dir / f"portfolio_{timestamp}.pkl"
                    with open(portfolio_path, 'wb') as f:
                        pickle.dump(self.results['basic_backtest']['portfolio'], f)
            
            # Save optimization results
            if self.optimization_results:
                opt_path = self.results_dir / f"optimization_{timestamp}.json"
                with open(opt_path, 'w') as f:
                    json.dump(self.optimization_results, f, indent=2, default=str)
            
            # Save validation results
            if self.validation_results:
                val_path = self.results_dir / f"validation_{timestamp}.json"
                with open(val_path, 'w') as f:
                    json.dump(self.validation_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to directory: {self.results_dir}")
            
            return str(self.results_dir)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def run_complete_pipeline(self, 
                            run_optimization: bool = True,
                            run_walk_forward: bool = True,
                            run_stress_tests: bool = True) -> Dict[str, Any]:
        """Run the complete backtesting pipeline"""
        logger.info("Starting complete backtesting pipeline...")
        
        start_time = datetime.now()
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Start monitoring
            if self.monitoring_system:
                self.monitoring_system.start_monitoring()
            
            # 1. Run basic backtest
            logger.info("Step 1: Running basic backtest...")
            basic_results = self.run_basic_backtest()
            
            # 2. Parameter optimization (optional)
            if run_optimization:
                logger.info("Step 2: Running parameter optimization...")
                self.run_parameter_optimization()
            
            # 3. Walk-forward analysis (optional)
            if run_walk_forward:
                logger.info("Step 3: Running walk-forward analysis...")
                self.run_walk_forward_analysis()
            
            # 4. Stress testing (optional)
            if run_stress_tests:
                logger.info("Step 4: Running stress testing...")
                self.run_stress_testing()
            
            # 5. Validate performance targets
            logger.info("Step 5: Validating performance targets...")
            self.validate_performance_targets()
            
            # 6. Generate comprehensive report
            logger.info("Step 6: Generating comprehensive report...")
            comprehensive_report = self.generate_comprehensive_report()
            
            # 7. Save all results
            logger.info("Step 7: Saving results...")
            self.save_results(include_portfolio=True)
            
            # Calculate total time
            total_time = datetime.now() - start_time
            
            logger.info(f"Complete backtesting pipeline finished in {total_time}")
            logger.info("="*50)
            logger.info("PIPELINE SUMMARY:")
            logger.info("="*50)
            
            # Print key results
            if 'basic_backtest' in self.results:
                metrics = self.results['basic_backtest']['metrics']
                logger.info(f"Total Return: {metrics['total_return']:.2%}")
                logger.info(f"CAGR: {metrics['cagr']:.2%}")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
                logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            
            if 'target_validation' in self.validation_results:
                validation = self.validation_results['target_validation']
                logger.info(f"Targets Passed: {validation['targets_passed']}/{validation['total_targets']}")
                logger.info(f"Overall Grade: {validation['overall_grade']}")
            
            logger.info("="*50)
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {e}")
            raise
            
        finally:
            # Stop monitoring
            if self.monitoring_system:
                self.monitoring_system.stop_monitoring()
    
    def _sample_parameter_combinations(self, param_ranges: Dict[str, List], 
                                     max_combinations: int) -> Dict[str, List]:
        """Sample parameter combinations to limit optimization iterations"""
        import itertools
        import random
        
        # Generate all combinations
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        all_combinations = list(itertools.product(*values))
        
        # Sample randomly
        sampled = random.sample(all_combinations, min(max_combinations, len(all_combinations)))
        
        # Convert back to parameter ranges format
        sampled_ranges = {key: [] for key in keys}
        for combo in sampled:
            for i, key in enumerate(keys):
                if combo[i] not in sampled_ranges[key]:
                    sampled_ranges[key].append(combo[i])
        
        return sampled_ranges
    
    def _assess_risk_scenarios(self, stress_results: Dict, monte_carlo_results: Dict) -> Dict[str, Any]:
        """Assess risk scenarios"""
        risk_assessment = {
            'high_risk_scenarios': [],
            'risk_score': 0,
            'recommendations': []
        }
        
        # Check stress test results
        for scenario, results in stress_results.items():
            if results['max_drawdown'] < -0.3:  # >30% drawdown
                risk_assessment['high_risk_scenarios'].append({
                    'scenario': scenario,
                    'max_drawdown': results['max_drawdown'],
                    'severity': 'HIGH'
                })
        
        # Check Monte Carlo downside risk
        if monte_carlo_results['downside_risk'] > 0.2:  # >20% chance of >10% loss
            risk_assessment['high_risk_scenarios'].append({
                'scenario': 'monte_carlo_downside',
                'downside_risk': monte_carlo_results['downside_risk'],
                'severity': 'MEDIUM'
            })
        
        # Calculate overall risk score
        high_risk_count = len([s for s in risk_assessment['high_risk_scenarios'] 
                              if s['severity'] == 'HIGH'])
        medium_risk_count = len([s for s in risk_assessment['high_risk_scenarios'] 
                               if s['severity'] == 'MEDIUM'])
        
        risk_assessment['risk_score'] = high_risk_count * 3 + medium_risk_count * 1
        
        return risk_assessment
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if 'basic_backtest' in self.results:
            metrics = self.results['basic_backtest']['metrics']
            
            # Sharpe ratio recommendations
            if metrics['sharpe_ratio'] < 1.0:
                recommendations.append("CRITICAL: Sharpe ratio below 1.0. Strategy needs significant improvement.")
            elif metrics['sharpe_ratio'] < self.config.target_sharpe:
                recommendations.append("Sharpe ratio below target. Consider optimizing signal quality and risk management.")
            
            # Drawdown recommendations
            if abs(metrics['max_drawdown']) > self.config.target_max_dd:
                recommendations.append("Maximum drawdown exceeds target. Implement stricter position sizing and stop-losses.")
            
            # Win rate recommendations
            if metrics['win_rate'] < self.config.target_win_rate:
                recommendations.append("Win rate below target. Focus on signal precision and entry timing.")
            
            # Profit factor recommendations
            if metrics['profit_factor'] < self.config.target_profit_factor:
                recommendations.append("Profit factor below target. Optimize exit timing and let winners run longer.")
        
        # Walk-forward recommendations
        if 'walk_forward' in self.validation_results:
            stability = self.validation_results['walk_forward']['stability_analysis']
            if stability['sharpe_consistency']['std'] > 1.0:
                recommendations.append("High Sharpe ratio volatility across periods. Strategy may not be robust.")
            
            if stability['sharpe_consistency']['positive_periods'] < 0.7:
                recommendations.append("Less than 70% positive periods. Strategy lacks consistency.")
        
        # Stress testing recommendations
        if 'stress_testing' in self.validation_results:
            stress_results = self.validation_results['stress_testing']
            if stress_results['risk_assessment']['risk_score'] > 5:
                recommendations.append("High risk score in stress testing. Implement additional risk controls.")
        
        if not recommendations:
            recommendations.append("Strategy performance meets all targets. Consider scaling up allocation.")
        
        return recommendations

def create_backtesting_pipeline(data_path: str = None, 
                              results_dir: str = None,
                              config: BacktestConfig = None) -> BacktestingPipeline:
    """Factory function to create backtesting pipeline"""
    return BacktestingPipeline(data_path, results_dir, config)

# Example usage and demonstration
if __name__ == "__main__":
    # Create configuration
    config = BacktestConfig(
        data_path="/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv",
        initial_cash=100000,
        commission=0.0001,
        target_sharpe=2.0,
        target_max_dd=0.15,
        target_win_rate=0.60
    )
    
    # Create pipeline
    pipeline = create_backtesting_pipeline(config=config)
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            run_optimization=True,
            run_walk_forward=True,
            run_stress_tests=True
        )
        
        print("\n" + "="*60)
        print("VECTORBT BACKTESTING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {pipeline.results_dir}")
        print("\nKey Performance Metrics:")
        
        if 'basic_backtest' in pipeline.results:
            metrics = pipeline.results['basic_backtest']['metrics']
            print(f"  • Total Return: {metrics['total_return']:.2%}")
            print(f"  • CAGR: {metrics['cagr']:.2%}")
            print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  • Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  • Win Rate: {metrics['win_rate']:.2%}")
            print(f"  • Profit Factor: {metrics['profit_factor']:.2f}")
        
        if 'target_validation' in pipeline.validation_results:
            validation = pipeline.validation_results['target_validation']
            print(f"\nTarget Achievement:")
            print(f"  • Targets Passed: {validation['targets_passed']}/{validation['total_targets']}")
            print(f"  • Overall Grade: {validation['overall_grade']}")
            print(f"  • Overall Score: {validation['overall_score']:.2f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nERROR: Pipeline execution failed: {e}")
        sys.exit(1)