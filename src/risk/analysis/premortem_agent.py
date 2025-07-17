"""
Pre-Mortem Analysis Agent - The Devil's Advocate

The ultimate "what could go wrong" system that uses high-speed Monte Carlo simulation
to evaluate the failure probability of every significant trading decision before execution.

MISSION CRITICAL OBJECTIVE: Build the Pre-Mortem Analysis Agent that acts as the 
ultimate risk gatekeeper, preventing bad decisions before they can hurt the portfolio.

ULTIMATE GOAL: Create a Monte Carlo simulation engine that can generate 10,000 
potential future scenarios in <100ms and provide clear GO/CAUTION/NO-GO recommendations 
with quantified failure probabilities.

Features:
- High-speed Monte Carlo simulation (<100ms for 10,000 paths)
- Advanced market models (GBM, jump diffusion, stochastic volatility)
- 3-tier recommendation system (GO/CAUTION/NO-GO)
- Automatic decision interception and routing
- Human review triggers for high-risk decisions
- Real-time performance monitoring
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import structlog
from threading import Lock
import json

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.simulation.monte_carlo_engine import (
    MonteCarloEngine, SimulationParameters, SimulationResults
)
from src.risk.simulation.advanced_market_models import (
    GeometricBrownianMotion, JumpDiffusionModel, HestonStochasticVolatility,
    RegimeSwitchingModel, GBMParameters, JumpDiffusionParameters, 
    HestonParameters, MarketRegime, CorrelationGenerator
)
from src.risk.analysis.failure_probability_calculator import (
    FailureProbabilityCalculator, FailureMetrics, FailureThresholds, RiskRecommendation
)
from src.risk.integration.decision_interceptor import (
    DecisionInterceptor, DecisionContext, InterceptionResult, InterceptionConfig
)
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class PreMortemConfig:
    """Configuration for pre-mortem analysis agent"""
    # Monte Carlo parameters
    default_num_paths: int = 10000               # Default simulation paths
    max_num_paths: int = 50000                   # Maximum paths for detailed analysis
    simulation_horizon_hours: float = 24.0      # Default simulation horizon
    time_steps: int = 1440                       # 1-minute granularity
    
    # Performance targets
    max_analysis_time_ms: float = 100.0         # Target analysis time
    max_queue_time_ms: float = 50.0             # Max time in queue
    
    # Market model parameters
    default_volatility: float = 0.20            # Default annual volatility
    default_drift: float = 0.10                 # Default annual drift
    jump_intensity: float = 5.0                 # Jumps per year
    jump_mean: float = 0.0                      # Mean jump size
    jump_std: float = 0.03                      # Jump volatility
    
    # Risk thresholds
    failure_thresholds: FailureThresholds = field(default_factory=FailureThresholds)
    interception_config: InterceptionConfig = field(default_factory=InterceptionConfig)
    
    # System behavior
    enable_parallel_analysis: bool = True       # Enable parallel processing
    enable_gpu_acceleration: bool = True        # Enable GPU if available
    enable_adaptive_paths: bool = True          # Adapt num_paths based on complexity
    enable_regime_detection: bool = True        # Enable regime-aware modeling
    
    # Emergency settings
    crisis_mode_threshold: float = 0.30         # Crisis mode VaR threshold
    emergency_override_enabled: bool = True     # Allow emergency overrides


@dataclass
class PreMortemAnalysisResult:
    """Complete pre-mortem analysis result"""
    decision_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core recommendation
    recommendation: RiskRecommendation = RiskRecommendation.GO
    failure_probability: float = 0.0
    confidence: float = 0.0
    
    # Detailed metrics
    failure_metrics: Optional[FailureMetrics] = None
    simulation_results: Optional[SimulationResults] = None
    
    # Performance data
    total_analysis_time_ms: float = 0.0
    simulation_time_ms: float = 0.0
    calculation_time_ms: float = 0.0
    
    # Decision context
    original_context: Optional[DecisionContext] = None
    
    # Risk insights
    primary_risk_factors: List[str] = field(default_factory=list)
    risk_mitigation_suggestions: List[str] = field(default_factory=list)
    
    # Human review
    requires_human_review: bool = False
    human_review_priority: str = "NORMAL"
    escalation_reasons: List[str] = field(default_factory=list)


class PreMortemAgent(BaseRiskAgent):
    """
    Pre-Mortem Analysis Agent - The Devil's Advocate
    
    The ultimate "what could go wrong" system that prevents bad trading decisions
    through comprehensive Monte Carlo simulation and risk analysis.
    
    Key Capabilities:
    - <100ms Monte Carlo simulation (10,000 paths)
    - Advanced market modeling (jumps, stochastic volatility, regime switching)
    - Automatic decision interception from all MARL agents
    - 3-tier recommendation system with failure probabilities
    - Human review triggers for high-risk scenarios
    - Real-time performance monitoring and optimization
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Pre-Mortem Analysis Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for system integration
        """
        # Initialize base risk agent
        super().__init__(config, event_bus)
        
        # Pre-mortem specific configuration
        self.premortem_config = PreMortemConfig(**config.get('premortem_config', {}))
        
        # Core components
        self.monte_carlo_engine = MonteCarloEngine(
            enable_gpu=self.premortem_config.enable_gpu_acceleration,
            max_threads=config.get('max_threads'),
            memory_limit_gb=config.get('memory_limit_gb', 8.0)
        )
        
        self.failure_calculator = FailureProbabilityCalculator(
            thresholds=self.premortem_config.failure_thresholds,
            confidence_level=config.get('confidence_level', 0.95)
        )
        
        # Decision interception system
        self.decision_interceptor = DecisionInterceptor(
            config=self.premortem_config.interception_config,
            event_bus=event_bus,
            premortem_analyzer=self._analyze_decision_internal
        )
        
        # State management
        self.analysis_cache: Dict[str, PreMortemAnalysisResult] = {}
        self.market_state = self._initialize_market_state()
        self.portfolio_state = self._initialize_portfolio_state()
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'go_recommendations': 0,
            'caution_recommendations': 0,
            'no_go_recommendations': 0,
            'human_reviews_triggered': 0,
            'avg_analysis_time_ms': 0.0,
            'performance_target_met_rate': 0.0
        }
        
        # Thread safety
        self.analysis_lock = Lock()
        
        logger.info("Pre-Mortem Analysis Agent initialized",
                   name=self.name,
                   simulation_paths=self.premortem_config.default_num_paths,
                   performance_target_ms=self.premortem_config.max_analysis_time_ms)
    
    def _initialize_market_state(self) -> Dict[str, Any]:
        """Initialize market state for simulation"""
        return {
            'current_prices': np.array([100.0]),  # Default single asset
            'volatilities': np.array([self.premortem_config.default_volatility]),
            'correlations': np.array([[1.0]]),
            'drift_rates': np.array([self.premortem_config.default_drift]),
            'market_regime': MarketRegime.BULL,
            'last_update': datetime.now()
        }
    
    def _initialize_portfolio_state(self) -> Dict[str, Any]:
        """Initialize portfolio state"""
        return {
            'current_value': 1.0,
            'positions': {},
            'weights': np.array([1.0]),
            'last_update': datetime.now()
        }
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[int, float]:
        """
        Calculate risk action based on current risk state
        
        For pre-mortem agent, this primarily monitors for emergency conditions
        and updates market state based on risk signals.
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (action, confidence)
        """
        try:
            # Update market state based on risk signals
            self._update_market_state_from_risk(risk_state)
            
            # Check for emergency conditions
            if self._detect_emergency_conditions(risk_state):
                logger.warning("Emergency conditions detected",
                              var_estimate=risk_state.var_estimate_5pct,
                              market_stress=risk_state.market_stress_level)
                return RiskAction.CLOSE_ALL.value, 0.9
            
            # Check for high stress conditions
            if risk_state.market_stress_level > 0.7 or risk_state.var_estimate_5pct > 0.1:
                return RiskAction.REDUCE_POSITION.value, 0.7
            
            # Normal monitoring mode
            return RiskAction.NO_ACTION.value, 0.8
            
        except Exception as e:
            logger.error("Error in pre-mortem risk calculation", error=str(e))
            return RiskAction.NO_ACTION.value, 0.1
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """
        Validate risk constraints for pre-mortem agent
        
        Args:
            risk_state: Current risk state
            
        Returns:
            True if constraints are satisfied
        """
        # Check VaR limits
        if risk_state.var_estimate_5pct > self.premortem_config.crisis_mode_threshold:
            return False
        
        # Check drawdown limits
        if risk_state.current_drawdown_pct > 0.20:  # 20% drawdown limit
            return False
        
        # Check margin usage
        if risk_state.margin_usage_pct > 0.90:  # 90% margin limit
            return False
        
        return True
    
    def analyze_trading_decision(self, decision_context: DecisionContext) -> PreMortemAnalysisResult:
        """
        Analyze a trading decision using comprehensive pre-mortem analysis
        
        Args:
            decision_context: Trading decision to analyze
            
        Returns:
            Complete analysis result with recommendation
        """
        start_time = time.perf_counter()
        
        try:
            with self.analysis_lock:
                # Check cache first
                cached_result = self.analysis_cache.get(decision_context.decision_id)
                if cached_result and self._is_cache_valid(cached_result):
                    logger.debug("Using cached analysis result", 
                               decision_id=decision_context.decision_id)
                    return cached_result
                
                # Run full analysis
                result = self._perform_complete_analysis(decision_context)
                
                # Cache result
                self.analysis_cache[decision_context.decision_id] = result
                
                # Update statistics
                self._update_analysis_stats(result)
                
                # Log analysis completion
                logger.info("Pre-mortem analysis completed",
                           decision_id=decision_context.decision_id,
                           recommendation=result.recommendation.value,
                           failure_probability=f"{result.failure_probability:.3f}",
                           analysis_time_ms=f"{result.total_analysis_time_ms:.2f}")
                
                return result
                
        except Exception as e:
            logger.error("Pre-mortem analysis failed", 
                        decision_id=decision_context.decision_id,
                        error=str(e))
            
            # Return conservative result
            return PreMortemAnalysisResult(
                decision_id=decision_context.decision_id,
                recommendation=RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW,
                failure_probability=1.0,
                confidence=0.0,
                requires_human_review=True,
                escalation_reasons=[f"Analysis error: {str(e)}"],
                original_context=decision_context
            )
    
    def _perform_complete_analysis(self, context: DecisionContext) -> PreMortemAnalysisResult:
        """Perform complete pre-mortem analysis"""
        analysis_start = time.perf_counter()
        
        # 1. Setup simulation parameters
        sim_params = self._build_simulation_parameters(context)
        
        # 2. Run Monte Carlo simulation
        sim_start = time.perf_counter()
        simulation_results = self.monte_carlo_engine.run_simulation(
            sim_params, self.portfolio_state['weights']
        )
        simulation_time = (time.perf_counter() - sim_start) * 1000
        
        # 3. Calculate failure metrics
        calc_start = time.perf_counter()
        failure_metrics = self.failure_calculator.calculate_failure_metrics(
            simulation_results, self.portfolio_state['current_value']
        )
        calculation_time = (time.perf_counter() - calc_start) * 1000
        
        # 4. Generate risk insights
        risk_factors, mitigation_suggestions = self._generate_risk_insights(
            context, simulation_results, failure_metrics
        )
        
        # 5. Determine human review requirements
        requires_review, review_priority, escalation_reasons = self._assess_human_review_need(
            context, failure_metrics
        )
        
        total_time = (time.perf_counter() - analysis_start) * 1000
        
        # Build result
        result = PreMortemAnalysisResult(
            decision_id=context.decision_id,
            recommendation=failure_metrics.recommendation,
            failure_probability=failure_metrics.failure_probability,
            confidence=failure_metrics.recommendation_confidence,
            failure_metrics=failure_metrics,
            simulation_results=simulation_results,
            total_analysis_time_ms=total_time,
            simulation_time_ms=simulation_time,
            calculation_time_ms=calculation_time,
            original_context=context,
            primary_risk_factors=risk_factors,
            risk_mitigation_suggestions=mitigation_suggestions,
            requires_human_review=requires_review,
            human_review_priority=review_priority,
            escalation_reasons=escalation_reasons
        )
        
        return result
    
    def _build_simulation_parameters(self, context: DecisionContext) -> SimulationParameters:
        """Build Monte Carlo simulation parameters for decision analysis"""
        # Determine number of paths based on decision importance
        num_paths = self._determine_simulation_paths(context)
        
        # Build parameters
        return SimulationParameters(
            num_paths=num_paths,
            time_horizon_hours=self.premortem_config.simulation_horizon_hours,
            time_steps=self.premortem_config.time_steps,
            initial_prices=self.market_state['current_prices'].copy(),
            drift_rates=self.market_state['drift_rates'].copy(),
            volatilities=self.market_state['volatilities'].copy(),
            correlation_matrix=self.market_state['correlations'].copy(),
            jump_intensity=self.premortem_config.jump_intensity,
            jump_mean=self.premortem_config.jump_mean,
            jump_std=self.premortem_config.jump_std,
            enable_regime_switching=self.premortem_config.enable_regime_detection,
            enable_stochastic_vol=True,
            random_seed=None  # Use random seed for each analysis
        )
    
    def _determine_simulation_paths(self, context: DecisionContext) -> int:
        """Determine optimal number of simulation paths based on decision complexity"""
        if not self.premortem_config.enable_adaptive_paths:
            return self.premortem_config.default_num_paths
        
        # Base paths
        num_paths = self.premortem_config.default_num_paths
        
        # Increase for high-impact decisions
        if context.portfolio_impact_percent > 50:
            num_paths = min(num_paths * 2, self.premortem_config.max_num_paths)
        
        # Increase for high-priority decisions
        if context.priority.value >= 4:  # HIGH or CRITICAL priority
            num_paths = min(num_paths * 1.5, self.premortem_config.max_num_paths)
        
        # Increase for large position changes
        if abs(context.position_change_percent) > 100:
            num_paths = min(num_paths * 1.5, self.premortem_config.max_num_paths)
        
        return int(num_paths)
    
    def _generate_risk_insights(self, 
                               context: DecisionContext,
                               sim_results: SimulationResults,
                               failure_metrics: FailureMetrics) -> Tuple[List[str], List[str]]:
        """Generate actionable risk insights and mitigation suggestions"""
        risk_factors = failure_metrics.primary_risk_factors.copy()
        mitigation_suggestions = []
        
        # Analyze simulation results for additional insights
        if failure_metrics.var_95_percent > 0.05:
            risk_factors.append("High Value-at-Risk")
            mitigation_suggestions.append("Consider reducing position size by 25-50%")
        
        if failure_metrics.max_drawdown_probability > 0.3:
            risk_factors.append("High drawdown risk")
            mitigation_suggestions.append("Implement tighter stop-loss levels")
        
        if failure_metrics.return_skewness < -1.0:
            risk_factors.append("Negative return skewness")
            mitigation_suggestions.append("Consider hedging against tail risk")
        
        # Context-specific insights
        if context.decision_type.value == "position_sizing" and context.position_change_percent > 100:
            risk_factors.append("Large position size increase")
            mitigation_suggestions.append("Stage the position increase over multiple periods")
        
        return risk_factors, mitigation_suggestions
    
    def _assess_human_review_need(self, 
                                 context: DecisionContext,
                                 failure_metrics: FailureMetrics) -> Tuple[bool, str, List[str]]:
        """Assess if human review is required"""
        requires_review = failure_metrics.requires_human_review
        escalation_reasons = failure_metrics.human_review_reasons.copy()
        
        # Determine priority
        if failure_metrics.failure_probability > 0.25:
            review_priority = "CRITICAL"
            escalation_reasons.append("Very high failure probability")
        elif failure_metrics.failure_probability > 0.15:
            review_priority = "HIGH"
        elif failure_metrics.var_95_percent > 0.06:
            review_priority = "HIGH"
            escalation_reasons.append("High VaR estimate")
        else:
            review_priority = "NORMAL"
        
        # Large position changes always need review
        if abs(context.position_change_percent) > 200:
            requires_review = True
            escalation_reasons.append("Very large position change")
        
        return requires_review, review_priority, escalation_reasons
    
    def _analyze_decision_internal(self, context: DecisionContext) -> InterceptionResult:
        """
        Internal method for decision interceptor integration
        
        Args:
            context: Decision context from interceptor
            
        Returns:
            Interception result for immediate use
        """
        try:
            # Run full pre-mortem analysis
            analysis_result = self.analyze_trading_decision(context)
            
            # Convert to interception result
            return InterceptionResult(
                decision_id=context.decision_id,
                status="approved" if analysis_result.recommendation == RiskRecommendation.GO else "rejected",
                failure_probability=analysis_result.failure_probability,
                recommendation=analysis_result.recommendation.value,
                confidence=analysis_result.confidence,
                var_95=analysis_result.failure_metrics.var_95_percent if analysis_result.failure_metrics else 0.0,
                expected_shortfall=analysis_result.failure_metrics.expected_shortfall_95 if analysis_result.failure_metrics else 0.0,
                max_drawdown_prob=analysis_result.failure_metrics.max_drawdown_probability if analysis_result.failure_metrics else 0.0,
                analysis_time_ms=analysis_result.total_analysis_time_ms,
                requires_human_review=analysis_result.requires_human_review,
                human_review_reason="; ".join(analysis_result.escalation_reasons)
            )
            
        except Exception as e:
            logger.error("Error in internal decision analysis", error=str(e))
            return InterceptionResult(
                decision_id=context.decision_id,
                status="error",
                error_message=str(e),
                recommendation="NO_GO"
            )
    
    def _update_market_state_from_risk(self, risk_state: RiskState) -> None:
        """Update market state based on risk signals"""
        # Update volatility based on market stress
        stress_multiplier = 1.0 + risk_state.market_stress_level
        self.market_state['volatilities'] *= stress_multiplier
        
        # Update correlations based on regime
        if risk_state.volatility_regime > 0.8:  # High volatility regime
            # Increase correlations (crisis correlation)
            self.market_state['correlations'] = np.minimum(
                self.market_state['correlations'] * 1.5, 0.9
            )
        
        self.market_state['last_update'] = datetime.now()
    
    def _detect_emergency_conditions(self, risk_state: RiskState) -> bool:
        """Detect emergency market conditions"""
        # High VaR
        if risk_state.var_estimate_5pct > self.premortem_config.crisis_mode_threshold:
            return True
        
        # Extreme market stress
        if risk_state.market_stress_level > 0.9:
            return True
        
        # Large drawdown
        if risk_state.current_drawdown_pct > 0.25:
            return True
        
        return False
    
    def _is_cache_valid(self, cached_result: PreMortemAnalysisResult) -> bool:
        """Check if cached analysis result is still valid"""
        # Cache timeout (5 minutes)
        if datetime.now() - cached_result.timestamp > timedelta(minutes=5):
            return False
        
        # Market state changed significantly
        if datetime.now() - self.market_state['last_update'] < timedelta(minutes=1):
            return False
        
        return True
    
    def _update_analysis_stats(self, result: PreMortemAnalysisResult) -> None:
        """Update analysis statistics"""
        self.analysis_stats['total_analyses'] += 1
        
        # Recommendation tracking
        if result.recommendation == RiskRecommendation.GO:
            self.analysis_stats['go_recommendations'] += 1
        elif result.recommendation == RiskRecommendation.GO_WITH_CAUTION:
            self.analysis_stats['caution_recommendations'] += 1
        else:
            self.analysis_stats['no_go_recommendations'] += 1
        
        if result.requires_human_review:
            self.analysis_stats['human_reviews_triggered'] += 1
        
        # Performance tracking
        self.analysis_stats['avg_analysis_time_ms'] = (
            (self.analysis_stats['avg_analysis_time_ms'] * (self.analysis_stats['total_analyses'] - 1) +
             result.total_analysis_time_ms) / self.analysis_stats['total_analyses']
        )
        
        target_met = result.total_analysis_time_ms <= self.premortem_config.max_analysis_time_ms
        self.analysis_stats['performance_target_met_rate'] = (
            (self.analysis_stats['performance_target_met_rate'] * (self.analysis_stats['total_analyses'] - 1) +
             (1.0 if target_met else 0.0)) / self.analysis_stats['total_analyses']
        )
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        stats = self.analysis_stats.copy()
        
        # Add component stats
        stats.update({
            'monte_carlo_stats': self.monte_carlo_engine.get_performance_stats(),
            'failure_calc_stats': self.failure_calculator.get_performance_stats(),
            'interception_stats': self.decision_interceptor.get_interception_stats(),
            'cache_size': len(self.analysis_cache),
            'market_state_age_minutes': (datetime.now() - self.market_state['last_update']).total_seconds() / 60
        })
        
        return stats
    
    def enable_crisis_mode(self) -> None:
        """Enable crisis mode for emergency situations"""
        self.decision_interceptor.enable_crisis_mode()
        logger.warning("Pre-mortem agent CRISIS MODE enabled")
    
    def disable_crisis_mode(self) -> None:
        """Disable crisis mode"""
        self.decision_interceptor.disable_crisis_mode()
        logger.info("Pre-mortem agent crisis mode disabled")
    
    def clear_analysis_cache(self) -> None:
        """Clear analysis result cache"""
        with self.analysis_lock:
            self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def export_analysis_report(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Export detailed analysis report for a decision"""
        result = self.analysis_cache.get(decision_id)
        if not result:
            return None
        
        return {
            'decision_id': decision_id,
            'timestamp': result.timestamp.isoformat(),
            'recommendation': result.recommendation.value,
            'failure_probability': result.failure_probability,
            'confidence': result.confidence,
            'analysis_time_ms': result.total_analysis_time_ms,
            'risk_factors': result.primary_risk_factors,
            'mitigation_suggestions': result.risk_mitigation_suggestions,
            'requires_human_review': result.requires_human_review,
            'escalation_reasons': result.escalation_reasons,
            'failure_metrics': {
                'var_95': result.failure_metrics.var_95_percent if result.failure_metrics else None,
                'expected_shortfall': result.failure_metrics.expected_shortfall_95 if result.failure_metrics else None,
                'max_drawdown_prob': result.failure_metrics.max_drawdown_probability if result.failure_metrics else None
            } if result.failure_metrics else None
        }
    
    def shutdown(self) -> None:
        """Shutdown pre-mortem agent"""
        logger.info("Shutting down Pre-Mortem Analysis Agent")
        
        # Shutdown decision interceptor
        self.decision_interceptor.shutdown()
        
        # Clear cache
        self.clear_analysis_cache()
        
        logger.info("Pre-mortem agent shutdown complete")