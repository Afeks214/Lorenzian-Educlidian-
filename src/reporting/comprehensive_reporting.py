"""
Comprehensive Reporting Framework for GrandModel
===============================================

Institutional-grade reporting system with executive summaries, detailed performance analysis,
risk assessment reports, strategy comparison analysis, and statistical validation reports.

Features:
- Executive summary reports with key insights
- Detailed performance analysis with attribution
- Risk assessment and monitoring reports
- Strategy comparison and benchmarking
- Statistical validation and significance testing
- Automated report generation and scheduling
- Multi-format output (PDF, HTML, JSON, CSV)
- Professional presentation templates

Author: Agent 6 - Visualization and Reporting System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import json
import io
import base64
from pathlib import Path
from dataclasses import dataclass, asdict
from jinja2 import Template
import pdfkit
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    template_style: str = "professional"
    include_charts: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True
    output_formats: List[str] = None
    color_scheme: Dict[str, str] = None
    font_family: str = "Arial"
    font_size: int = 11
    company_name: str = "GrandModel Trading System"
    company_logo: Optional[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'pdf', 'json']
        if self.color_scheme is None:
            self.color_scheme = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'positive': '#28A745',
                'negative': '#DC3545',
                'warning': '#FFC107',
                'info': '#17A2B8',
                'neutral': '#6C757D'
            }


@dataclass
class ReportData:
    """Structured report data container"""
    strategy_name: str
    report_period: Tuple[datetime, datetime]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    trade_statistics: Dict[str, Any]
    market_data: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, str]
    metadata: Dict[str, Any]


class ComprehensiveReporter:
    """
    Comprehensive reporting system for trading strategies
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize comprehensive reporter
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.reports = {}
        self.templates = {}
        self.schedulers = {}
        
        # Create results directory
        self.results_dir = Path("/home/QuantNova/GrandModel/results/reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load templates
        self._load_templates()
        
        # Initialize email client
        self.email_client = None
        
        logger.info("Comprehensive Reporting System initialized")
    
    def generate_executive_summary(self, 
                                 strategy_data: Dict[str, Any],
                                 benchmark_data: Dict[str, Any] = None,
                                 period_days: int = 30) -> Dict[str, Any]:
        """
        Generate executive summary report
        
        Args:
            strategy_data: Strategy performance data
            benchmark_data: Benchmark comparison data
            period_days: Report period in days
            
        Returns:
            Executive summary report
        """
        try:
            # Extract key metrics
            returns = strategy_data.get('returns', pd.Series())
            if returns.empty:
                raise ValueError("No returns data provided")
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(returns)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Generate insights
            insights = self._generate_key_insights(performance_metrics, risk_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(performance_metrics, risk_metrics)
            
            # Calculate benchmark comparison
            benchmark_comparison = {}
            if benchmark_data:
                benchmark_comparison = self._compare_to_benchmark(
                    returns, benchmark_data.get('returns', pd.Series())
                )
            
            # Create executive summary
            executive_summary = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'strategy_name': strategy_data.get('name', 'Unknown Strategy'),
                    'period_start': (datetime.now() - timedelta(days=period_days)).isoformat(),
                    'period_end': datetime.now().isoformat(),
                    'report_type': 'executive_summary'
                },
                'key_metrics': {
                    'total_return': performance_metrics.get('total_return', 0),
                    'annualized_return': performance_metrics.get('annualized_return', 0),
                    'volatility': performance_metrics.get('volatility', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': performance_metrics.get('sortino_ratio', 0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0),
                    'win_rate': performance_metrics.get('win_rate', 0),
                    'profit_factor': performance_metrics.get('profit_factor', 0)
                },
                'risk_assessment': {
                    'var_95': risk_metrics.get('var_95', 0),
                    'expected_shortfall': risk_metrics.get('expected_shortfall', 0),
                    'tail_ratio': risk_metrics.get('tail_ratio', 0),
                    'risk_score': risk_metrics.get('risk_score', 0)
                },
                'benchmark_comparison': benchmark_comparison,
                'key_insights': insights,
                'recommendations': recommendations,
                'performance_score': self._calculate_performance_score(performance_metrics, risk_metrics),
                'risk_rating': self._calculate_risk_rating(risk_metrics),
                'overall_rating': self._calculate_overall_rating(performance_metrics, risk_metrics)
            }
            
            return executive_summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {'error': str(e)}
    
    def generate_detailed_performance_report(self, 
                                           strategy_data: Dict[str, Any],
                                           trade_data: pd.DataFrame = None,
                                           benchmark_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate detailed performance analysis report
        
        Args:
            strategy_data: Strategy performance data
            trade_data: Individual trade data
            benchmark_data: Benchmark comparison data
            
        Returns:
            Detailed performance report
        """
        try:
            returns = strategy_data.get('returns', pd.Series())
            if returns.empty:
                raise ValueError("No returns data provided")
            
            # Performance analysis
            performance_analysis = self._detailed_performance_analysis(returns)
            
            # Trade analysis
            trade_analysis = {}
            if trade_data is not None and not trade_data.empty:
                trade_analysis = self._detailed_trade_analysis(trade_data)
            
            # Time-based analysis
            time_analysis = self._time_based_analysis(returns)
            
            # Risk analysis
            risk_analysis = self._detailed_risk_analysis(returns)
            
            # Statistical analysis
            statistical_analysis = self._statistical_analysis(returns)
            
            # Attribution analysis
            attribution_analysis = self._attribution_analysis(returns, trade_data)
            
            # Create detailed report
            detailed_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'strategy_name': strategy_data.get('name', 'Unknown Strategy'),
                    'report_type': 'detailed_performance_analysis',
                    'data_period': {
                        'start': returns.index[0].isoformat() if not returns.empty else None,
                        'end': returns.index[-1].isoformat() if not returns.empty else None,
                        'observations': len(returns)
                    }
                },
                'performance_analysis': performance_analysis,
                'trade_analysis': trade_analysis,
                'time_analysis': time_analysis,
                'risk_analysis': risk_analysis,
                'statistical_analysis': statistical_analysis,
                'attribution_analysis': attribution_analysis,
                'benchmark_comparison': self._compare_to_benchmark(
                    returns, benchmark_data.get('returns', pd.Series()) if benchmark_data else pd.Series()
                ),
                'conclusions': self._generate_conclusions(performance_analysis, risk_analysis),
                'recommendations': self._generate_detailed_recommendations(
                    performance_analysis, risk_analysis, trade_analysis
                )
            }
            
            return detailed_report
            
        except Exception as e:
            logger.error(f"Error generating detailed performance report: {e}")
            return {'error': str(e)}
    
    def generate_risk_assessment_report(self, 
                                      strategy_data: Dict[str, Any],
                                      market_data: Dict[str, Any] = None,
                                      stress_scenarios: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment report
        
        Args:
            strategy_data: Strategy performance data
            market_data: Market environment data
            stress_scenarios: Stress test scenarios
            
        Returns:
            Risk assessment report
        """
        try:
            returns = strategy_data.get('returns', pd.Series())
            if returns.empty:
                raise ValueError("No returns data provided")
            
            # Risk metrics calculation
            risk_metrics = self._comprehensive_risk_metrics(returns)
            
            # VaR analysis
            var_analysis = self._var_analysis(returns)
            
            # Stress testing
            stress_test_results = self._stress_testing(returns, stress_scenarios)
            
            # Scenario analysis
            scenario_analysis = self._scenario_analysis(returns, market_data)
            
            # Tail risk analysis
            tail_risk_analysis = self._tail_risk_analysis(returns)
            
            # Risk attribution
            risk_attribution = self._risk_attribution_analysis(returns, strategy_data)
            
            # Risk limits monitoring
            risk_limits = self._risk_limits_monitoring(returns)
            
            # Create risk assessment report
            risk_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'strategy_name': strategy_data.get('name', 'Unknown Strategy'),
                    'report_type': 'risk_assessment',
                    'risk_model_version': '1.0',
                    'confidence_level': 0.95
                },
                'risk_summary': {
                    'overall_risk_score': risk_metrics.get('overall_risk_score', 0),
                    'risk_rating': self._calculate_risk_rating(risk_metrics),
                    'key_risk_factors': self._identify_key_risk_factors(risk_metrics),
                    'risk_trend': self._calculate_risk_trend(returns)
                },
                'risk_metrics': risk_metrics,
                'var_analysis': var_analysis,
                'stress_test_results': stress_test_results,
                'scenario_analysis': scenario_analysis,
                'tail_risk_analysis': tail_risk_analysis,
                'risk_attribution': risk_attribution,
                'risk_limits_monitoring': risk_limits,
                'risk_recommendations': self._generate_risk_recommendations(risk_metrics),
                'monitoring_alerts': self._generate_monitoring_alerts(risk_metrics)
            }
            
            return risk_report
            
        except Exception as e:
            logger.error(f"Error generating risk assessment report: {e}")
            return {'error': str(e)}
    
    def generate_strategy_comparison_report(self, 
                                          strategies: Dict[str, Dict[str, Any]],
                                          benchmark_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate strategy comparison analysis report
        
        Args:
            strategies: Dictionary of strategy data
            benchmark_data: Benchmark comparison data
            
        Returns:
            Strategy comparison report
        """
        try:
            # Comparative analysis
            comparative_analysis = self._comparative_analysis(strategies)
            
            # Performance ranking
            performance_ranking = self._performance_ranking(strategies)
            
            # Risk-adjusted comparison
            risk_adjusted_comparison = self._risk_adjusted_comparison(strategies)
            
            # Statistical significance tests
            significance_tests = self._strategy_significance_tests(strategies)
            
            # Correlation analysis
            correlation_analysis = self._strategy_correlation_analysis(strategies)
            
            # Diversification benefits
            diversification_analysis = self._diversification_analysis(strategies)
            
            # Create comparison report
            comparison_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'strategy_comparison',
                    'strategies_analyzed': list(strategies.keys()),
                    'comparison_period': self._get_comparison_period(strategies)
                },
                'executive_summary': {
                    'best_performer': performance_ranking[0] if performance_ranking else None,
                    'best_risk_adjusted': risk_adjusted_comparison.get('best_sharpe', None),
                    'most_stable': risk_adjusted_comparison.get('lowest_volatility', None),
                    'key_insights': self._generate_comparison_insights(comparative_analysis)
                },
                'comparative_analysis': comparative_analysis,
                'performance_ranking': performance_ranking,
                'risk_adjusted_comparison': risk_adjusted_comparison,
                'significance_tests': significance_tests,
                'correlation_analysis': correlation_analysis,
                'diversification_analysis': diversification_analysis,
                'recommendations': self._generate_strategy_recommendations(comparative_analysis),
                'portfolio_optimization': self._portfolio_optimization_suggestions(strategies)
            }
            
            return comparison_report
            
        except Exception as e:
            logger.error(f"Error generating strategy comparison report: {e}")
            return {'error': str(e)}
    
    def generate_statistical_validation_report(self, 
                                             strategy_data: Dict[str, Any],
                                             out_of_sample_data: Dict[str, Any] = None,
                                             monte_carlo_runs: int = 1000) -> Dict[str, Any]:
        """
        Generate statistical validation report
        
        Args:
            strategy_data: Strategy performance data
            out_of_sample_data: Out-of-sample test data
            monte_carlo_runs: Number of Monte Carlo simulations
            
        Returns:
            Statistical validation report
        """
        try:
            returns = strategy_data.get('returns', pd.Series())
            if returns.empty:
                raise ValueError("No returns data provided")
            
            # Statistical tests
            statistical_tests = self._comprehensive_statistical_tests(returns)
            
            # Bootstrap analysis
            bootstrap_analysis = self._bootstrap_analysis(returns)
            
            # Monte Carlo simulation
            monte_carlo_results = self._monte_carlo_simulation(returns, monte_carlo_runs)
            
            # Regime analysis
            regime_analysis = self._regime_analysis(returns)
            
            # Robustness tests
            robustness_tests = self._robustness_tests(returns)
            
            # Out-of-sample validation
            oos_validation = {}
            if out_of_sample_data:
                oos_validation = self._out_of_sample_validation(returns, out_of_sample_data)
            
            # Create validation report
            validation_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'strategy_name': strategy_data.get('name', 'Unknown Strategy'),
                    'report_type': 'statistical_validation',
                    'validation_methods': [
                        'statistical_tests', 'bootstrap_analysis', 'monte_carlo_simulation',
                        'regime_analysis', 'robustness_tests'
                    ]
                },
                'validation_summary': {
                    'overall_significance': statistical_tests.get('overall_significance', False),
                    'confidence_level': statistical_tests.get('confidence_level', 0.95),
                    'p_value': statistical_tests.get('combined_p_value', 1.0),
                    'validation_score': self._calculate_validation_score(statistical_tests, bootstrap_analysis)
                },
                'statistical_tests': statistical_tests,
                'bootstrap_analysis': bootstrap_analysis,
                'monte_carlo_results': monte_carlo_results,
                'regime_analysis': regime_analysis,
                'robustness_tests': robustness_tests,
                'out_of_sample_validation': oos_validation,
                'conclusions': self._generate_validation_conclusions(statistical_tests, bootstrap_analysis),
                'recommendations': self._generate_validation_recommendations(statistical_tests)
            }
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error generating statistical validation report: {e}")
            return {'error': str(e)}
    
    def export_report_to_pdf(self, report: Dict[str, Any], filename: str) -> str:
        """
        Export report to PDF format
        
        Args:
            report: Report data
            filename: Output filename
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Create PDF document
            pdf_path = self.results_dir / f"{filename}.pdf"
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            
            # Build PDF content
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph(f"<b>{report.get('report_metadata', {}).get('strategy_name', 'Strategy Report')}</b>", 
                            styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Executive Summary
            if 'executive_summary' in report or 'key_metrics' in report:
                story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
                
                # Key metrics table
                metrics_data = []
                if 'key_metrics' in report:
                    for key, value in report['key_metrics'].items():
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:.2%}" if 'return' in key.lower() or 'ratio' in key.lower() else f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                        metrics_data.append([key.replace('_', ' ').title(), formatted_value])
                
                if metrics_data:
                    metrics_table = Table(metrics_data)
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(metrics_table)
                    story.append(Spacer(1, 12))
            
            # Recommendations
            if 'recommendations' in report:
                story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
                for i, rec in enumerate(report['recommendations'], 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report exported to {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error exporting PDF report: {e}")
            return ""
    
    def export_report_to_html(self, report: Dict[str, Any], filename: str) -> str:
        """
        Export report to HTML format
        
        Args:
            report: Report data
            filename: Output filename
            
        Returns:
            Path to generated HTML file
        """
        try:
            html_path = self.results_dir / f"{filename}.html"
            
            # Use template
            template = self.templates.get('html_report', self._get_default_html_template())
            
            # Render template
            html_content = template.render(report=report, timestamp=datetime.now())
            
            # Write to file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report exported to {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"Error exporting HTML report: {e}")
            return ""
    
    def export_report_to_json(self, report: Dict[str, Any], filename: str) -> str:
        """
        Export report to JSON format
        
        Args:
            report: Report data
            filename: Output filename
            
        Returns:
            Path to generated JSON file
        """
        try:
            json_path = self.results_dir / f"{filename}.json"
            
            # Convert to JSON-serializable format
            json_report = self._convert_to_json_serializable(report)
            
            # Write to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report exported to {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
            return ""
    
    def schedule_report_generation(self, 
                                 report_type: str,
                                 schedule_time: str,
                                 strategy_data_source: callable,
                                 email_recipients: List[str] = None):
        """
        Schedule automated report generation
        
        Args:
            report_type: Type of report to generate
            schedule_time: Schedule time (e.g., "09:00", "daily", "weekly")
            strategy_data_source: Function to get strategy data
            email_recipients: List of email addresses to send reports to
        """
        try:
            def generate_and_send_report():
                try:
                    # Get strategy data
                    strategy_data = strategy_data_source()
                    
                    # Generate report
                    if report_type == 'executive_summary':
                        report = self.generate_executive_summary(strategy_data)
                    elif report_type == 'performance_analysis':
                        report = self.generate_detailed_performance_report(strategy_data)
                    elif report_type == 'risk_assessment':
                        report = self.generate_risk_assessment_report(strategy_data)
                    else:
                        logger.error(f"Unknown report type: {report_type}")
                        return
                    
                    # Export report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{report_type}_{timestamp}"
                    
                    html_path = self.export_report_to_html(report, filename)
                    pdf_path = self.export_report_to_pdf(report, filename)
                    
                    # Send email if recipients provided
                    if email_recipients and html_path:
                        self._send_email_report(html_path, pdf_path, email_recipients, report_type)
                    
                    logger.info(f"Scheduled report generated: {report_type}")
                    
                except Exception as e:
                    logger.error(f"Error in scheduled report generation: {e}")
            
            # Schedule the report
            if schedule_time == "daily":
                schedule.every().day.at("09:00").do(generate_and_send_report)
            elif schedule_time == "weekly":
                schedule.every().monday.at("09:00").do(generate_and_send_report)
            elif ":" in schedule_time:
                schedule.every().day.at(schedule_time).do(generate_and_send_report)
            else:
                logger.error(f"Invalid schedule time: {schedule_time}")
                return
            
            # Store scheduler reference
            self.schedulers[f"{report_type}_{schedule_time}"] = generate_and_send_report
            
            logger.info(f"Report scheduled: {report_type} at {schedule_time}")
            
        except Exception as e:
            logger.error(f"Error scheduling report: {e}")
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = metrics['annualized_return'] / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown metrics
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            metrics['max_drawdown'] = drawdown.min()
            
            # Trade-based metrics (simplified)
            winning_days = (returns > 0).sum()
            total_days = len(returns)
            metrics['win_rate'] = winning_days / total_days if total_days > 0 else 0
            
            avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if (total_days - winning_days) > 0 else 0
            metrics['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            risk_metrics = {}
            
            # VaR and Expected Shortfall
            risk_metrics['var_95'] = returns.quantile(0.05)
            risk_metrics['var_99'] = returns.quantile(0.01)
            risk_metrics['expected_shortfall'] = returns[returns <= risk_metrics['var_95']].mean()
            
            # Tail risk
            risk_metrics['tail_ratio'] = abs(returns.quantile(0.95) / returns.quantile(0.05))
            
            # Skewness and Kurtosis
            risk_metrics['skewness'] = returns.skew()
            risk_metrics['kurtosis'] = returns.kurtosis()
            
            # Maximum consecutive losses
            consecutive_losses = self._calculate_consecutive_losses(returns)
            risk_metrics['max_consecutive_losses'] = consecutive_losses
            
            # Risk score (0-100, lower is better)
            volatility_score = min(100, (returns.std() * np.sqrt(252)) * 100)
            drawdown_score = min(100, abs(self._calculate_max_drawdown(returns)) * 100)
            risk_metrics['risk_score'] = (volatility_score + drawdown_score) / 2
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _generate_key_insights(self, performance_metrics: Dict[str, float], risk_metrics: Dict[str, float]) -> List[str]:
        """Generate key insights from metrics"""
        insights = []
        
        try:
            # Performance insights
            sharpe = performance_metrics.get('sharpe_ratio', 0)
            if sharpe > 2.0:
                insights.append("Exceptional risk-adjusted returns with Sharpe ratio > 2.0")
            elif sharpe > 1.0:
                insights.append("Strong risk-adjusted returns with Sharpe ratio > 1.0")
            elif sharpe < 0.5:
                insights.append("Below-average risk-adjusted returns")
            
            # Risk insights
            max_dd = abs(performance_metrics.get('max_drawdown', 0))
            if max_dd > 0.2:
                insights.append("High drawdown risk observed (>20%)")
            elif max_dd < 0.05:
                insights.append("Low drawdown risk maintained (<5%)")
            
            # Volatility insights
            volatility = performance_metrics.get('volatility', 0)
            if volatility > 0.3:
                insights.append("High volatility strategy requiring careful risk management")
            elif volatility < 0.1:
                insights.append("Low volatility strategy suitable for conservative investors")
            
            # Win rate insights
            win_rate = performance_metrics.get('win_rate', 0)
            if win_rate > 0.6:
                insights.append("High win rate indicating good trade selection")
            elif win_rate < 0.4:
                insights.append("Low win rate may indicate need for strategy refinement")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Error generating insights"]
    
    def _generate_recommendations(self, performance_metrics: Dict[str, float], risk_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Risk management recommendations
            max_dd = abs(performance_metrics.get('max_drawdown', 0))
            if max_dd > 0.15:
                recommendations.append("Consider implementing stricter stop-loss mechanisms")
            
            # Position sizing recommendations
            volatility = performance_metrics.get('volatility', 0)
            if volatility > 0.25:
                recommendations.append("Reduce position sizes to manage high volatility")
            
            # Strategy improvement recommendations
            sharpe = performance_metrics.get('sharpe_ratio', 0)
            if sharpe < 1.0:
                recommendations.append("Focus on improving risk-adjusted returns")
            
            # Diversification recommendations
            if risk_metrics.get('tail_ratio', 0) > 5:
                recommendations.append("Consider diversification to reduce tail risk")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _load_templates(self):
        """Load report templates"""
        try:
            # HTML template
            self.templates['html_report'] = self._get_default_html_template()
            
            # Email template
            self.templates['email_report'] = self._get_default_email_template()
            
            logger.info("Report templates loaded")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _get_default_html_template(self) -> Template:
        """Get default HTML report template"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.report_metadata.strategy_name }} - Strategy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .header { background-color: #2E86AB; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2E86AB; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .table th { background-color: #f2f2f2; }
                .insight { background-color: #e9ecef; padding: 10px; margin: 10px 0; border-left: 4px solid #17a2b8; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.report_metadata.strategy_name }}</h1>
                <p>Strategy Performance Report</p>
                <p>Generated: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            {% if report.key_metrics %}
            <div class="section">
                <h2>Key Performance Metrics</h2>
                {% for key, value in report.key_metrics.items() %}
                    <div class="metric">
                        <strong>{{ key.replace('_', ' ').title() }}:</strong>
                        {% if 'return' in key.lower() or 'ratio' in key.lower() %}
                            <span class="{{ 'positive' if value > 0 else 'negative' }}">{{ "%.2f%%"|format(value * 100) }}</span>
                        {% else %}
                            <span>{{ "%.4f"|format(value) }}</span>
                        {% endif %}
                    </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if report.key_insights %}
            <div class="section">
                <h2>Key Insights</h2>
                {% for insight in report.key_insights %}
                    <div class="insight">{{ insight }}</div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if report.recommendations %}
            <div class="section">
                <h2>Recommendations</h2>
                {% for recommendation in report.recommendations %}
                    <div class="recommendation">{{ recommendation }}</div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div style="margin-top: 50px; text-align: center; color: #666;">
                <p>Generated by GrandModel Comprehensive Reporting System</p>
            </div>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _get_default_email_template(self) -> Template:
        """Get default email template"""
        template_str = """
        Subject: {{ report.report_metadata.strategy_name }} - Strategy Report
        
        Dear Recipient,
        
        Please find attached the latest strategy performance report for {{ report.report_metadata.strategy_name }}.
        
        Key Highlights:
        {% if report.key_metrics %}
        - Total Return: {{ "%.2f%%"|format(report.key_metrics.get('total_return', 0) * 100) }}
        - Sharpe Ratio: {{ "%.3f"|format(report.key_metrics.get('sharpe_ratio', 0)) }}
        - Maximum Drawdown: {{ "%.2f%%"|format(report.key_metrics.get('max_drawdown', 0) * 100) }}
        {% endif %}
        
        {% if report.key_insights %}
        Key Insights:
        {% for insight in report.key_insights %}
        - {{ insight }}
        {% endfor %}
        {% endif %}
        
        For detailed analysis, please refer to the attached reports.
        
        Best regards,
        GrandModel Reporting System
        """
        return Template(template_str)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    # Additional helper methods would continue here...
    # (Implementation of detailed analysis methods, statistical tests, etc.)
    
    def _calculate_performance_score(self, performance_metrics: Dict[str, float], risk_metrics: Dict[str, float]) -> float:
        """Calculate overall performance score"""
        try:
            sharpe = performance_metrics.get('sharpe_ratio', 0)
            max_dd = abs(performance_metrics.get('max_drawdown', 0))
            
            # Score components
            sharpe_score = min(100, max(0, (sharpe + 1) * 25))
            dd_score = max(0, 100 - (max_dd * 500))
            
            # Weighted average
            return (sharpe_score * 0.6) + (dd_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0
    
    def _calculate_risk_rating(self, risk_metrics: Dict[str, float]) -> str:
        """Calculate risk rating"""
        try:
            risk_score = risk_metrics.get('risk_score', 50)
            
            if risk_score < 20:
                return "Low Risk"
            elif risk_score < 40:
                return "Moderate Risk"
            elif risk_score < 60:
                return "High Risk"
            else:
                return "Very High Risk"
                
        except Exception as e:
            logger.error(f"Error calculating risk rating: {e}")
            return "Unknown"
    
    def _calculate_overall_rating(self, performance_metrics: Dict[str, float], risk_metrics: Dict[str, float]) -> str:
        """Calculate overall strategy rating"""
        try:
            perf_score = self._calculate_performance_score(performance_metrics, risk_metrics)
            
            if perf_score >= 80:
                return "Excellent"
            elif perf_score >= 65:
                return "Good"
            elif perf_score >= 50:
                return "Fair"
            elif perf_score >= 35:
                return "Poor"
            else:
                return "Very Poor"
                
        except Exception as e:
            logger.error(f"Error calculating overall rating: {e}")
            return "Unknown"


# Global instance
comprehensive_reporter = ComprehensiveReporter()