"""
Professional Backtesting Framework
=================================

Institutional-grade backtesting framework with comprehensive performance analytics,
risk management, and professional reporting capabilities.

Components:
- Performance Analytics: Sharpe, Sortino, Calmar ratios, drawdown analysis
- Risk Management: Position limits, loss controls, correlation analysis
- Professional Reporting: Trade attribution, visualizations, benchmarking
- Data Quality: Missing data detection, outlier identification, consistency checks
"""

from .performance_analytics import PerformanceAnalyzer
from .risk_management import RiskManager
from .reporting import ProfessionalReporter
from .data_quality import DataQualityAssurance
from .framework import ProfessionalBacktestFramework

__all__ = [
    'PerformanceAnalyzer',
    'RiskManager', 
    'ProfessionalReporter',
    'DataQualityAssurance',
    'ProfessionalBacktestFramework'
]