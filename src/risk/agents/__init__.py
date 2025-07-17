"""
Risk Management Agents Module

This module contains all risk management agents implementing the Risk Management MARL System:
- BaseRiskAgent: Abstract base class for all risk agents
- RiskMonitorAgent: Real-time risk monitoring and emergency response with <10ms precision
- EmergencyActionSystem: Ultra-fast emergency action execution
- RealTimeRiskAssessor: Comprehensive real-time risk assessment
- MarketStressDetector: Flash crash and crisis detection
- PerformanceOptimizer: <10ms calculation optimization
- IntegrationFramework: Seamless system integration

All agents coordinate through the centralized risk critic for optimal portfolio protection.
"""

# Core imports that should always be available
from .base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics

# Try to import other modules with fallbacks
try:
    from .risk_monitor_agent import RiskMonitorAgent
    RISK_MONITOR_AVAILABLE = True
except ImportError:
    RISK_MONITOR_AVAILABLE = False

# Add other optional imports with error handling
try:
    from .emergency_action_system import EmergencyActionExecutor, ActionPriority, ExecutionStatus
    EMERGENCY_SYSTEM_AVAILABLE = True
except ImportError:
    EMERGENCY_SYSTEM_AVAILABLE = False

try:
    from .real_time_risk_assessor import RealTimeRiskAssessor, RiskBreach, BreachSeverity, RiskMetricType
    RISK_ASSESSOR_AVAILABLE = True
except ImportError:
    RISK_ASSESSOR_AVAILABLE = False

try:
    from .market_stress_detector import MarketStressDetector, MarketRegime, StressSignal
    STRESS_DETECTOR_AVAILABLE = True
except ImportError:
    STRESS_DETECTOR_AVAILABLE = False

try:
    from .performance_optimizer import PerformanceOptimizer, PerformanceCache
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from .integration_framework import RiskMonitorIntegration, VaRCalculatorBridge, CorrelationTrackerBridge
    INTEGRATION_FRAMEWORK_AVAILABLE = True
except ImportError:
    INTEGRATION_FRAMEWORK_AVAILABLE = False

# Position Sizing Agent (π₁) - Agent 2 Implementation
try:
    from .position_sizing_agent_v2 import PositionSizingAgentV2, create_position_sizing_agent_v2
    from .position_sizing_reward_system import PositionSizingRewardSystem, create_position_sizing_reward_system
    POSITION_SIZING_AVAILABLE = True
except ImportError:
    POSITION_SIZING_AVAILABLE = False

# Portfolio Optimizer Agent (π₄) - Agent 5 Implementation
try:
    from .portfolio_optimizer_agent import PortfolioOptimizerAgent, StrategyPerformance, PortfolioState
    from .portfolio_correlation_manager import PortfolioCorrelationManager, CorrelationRiskMetrics, DiversificationMetrics
    from .multi_objective_optimizer import MultiObjectiveOptimizer, OptimizationObjectives, OptimizationConstraints
    from .risk_parity_engine import RiskParityEngine, RiskParityMethod, RiskBudget, RiskParityResult
    from .performance_attribution import PerformanceAttributionEngine, StrategyPerformanceMetrics, AttributionMethod
    from .dynamic_rebalancing_engine import DynamicRebalancingEngine, RebalanceConfig, RebalanceTrigger, RebalanceUrgency
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPTIMIZER_AVAILABLE = False

# Legacy imports for backward compatibility
try:
    from .position_sizing_agent import PositionSizingAgent
    from .stop_target_agent import StopTargetAgent
    LEGACY_AGENTS_AVAILABLE = True
except ImportError:
    LEGACY_AGENTS_AVAILABLE = False

# Core exports that should always be available
__all__ = [
    'BaseRiskAgent',
    'RiskState', 
    'RiskAction',
    'RiskMetrics'
]

# Add conditional exports based on what's available
if RISK_MONITOR_AVAILABLE:
    __all__.append('RiskMonitorAgent')

if EMERGENCY_SYSTEM_AVAILABLE:
    __all__.extend(['EmergencyActionExecutor', 'ActionPriority', 'ExecutionStatus'])

if RISK_ASSESSOR_AVAILABLE:
    __all__.extend(['RealTimeRiskAssessor', 'RiskBreach', 'BreachSeverity', 'RiskMetricType'])

if STRESS_DETECTOR_AVAILABLE:
    __all__.extend(['MarketStressDetector', 'MarketRegime', 'StressSignal'])

if PERFORMANCE_OPTIMIZER_AVAILABLE:
    __all__.extend(['PerformanceOptimizer', 'PerformanceCache'])

if INTEGRATION_FRAMEWORK_AVAILABLE:
    __all__.extend(['RiskMonitorIntegration', 'VaRCalculatorBridge', 'CorrelationTrackerBridge'])

# Add Position Sizing Agent (π₁) if available
if POSITION_SIZING_AVAILABLE:
    __all__.extend([
        'PositionSizingAgentV2',
        'create_position_sizing_agent_v2',
        'PositionSizingRewardSystem',
        'create_position_sizing_reward_system'
    ])

# Add Portfolio Optimizer Agent (π₄) if available
if PORTFOLIO_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'PortfolioOptimizerAgent',
        'StrategyPerformance',
        'PortfolioState',
        'PortfolioCorrelationManager',
        'CorrelationRiskMetrics',
        'DiversificationMetrics',
        'MultiObjectiveOptimizer',
        'OptimizationObjectives',
        'OptimizationConstraints',
        'RiskParityEngine',
        'RiskParityMethod',
        'RiskBudget',
        'RiskParityResult',
        'PerformanceAttributionEngine',
        'StrategyPerformanceMetrics',
        'AttributionMethod',
        'DynamicRebalancingEngine',
        'RebalanceConfig',
        'RebalanceTrigger',
        'RebalanceUrgency'
    ])

# Add legacy agents if available
if LEGACY_AGENTS_AVAILABLE:
    __all__.extend([
        'PositionSizingAgent',
        'StopTargetAgent'
    ])